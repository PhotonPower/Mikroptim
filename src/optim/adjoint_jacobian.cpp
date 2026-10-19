// Adjungierte Jacobi-Matrix, Host-Teil und CPU-Referenz (docs/numerik.md, Abschnitt
// „Rückwärtsdurchlauf (M5-T2)“; docs/architektur.md, Abschnitt „Modus ADJOINT: Jacobi-Matrix“).

#include "mikroptim/optim/adjoint_jacobian.hpp"

#include "mikroptim/glass/dispersion.hpp"
#include "mikroptim/merit/ray_residual.hpp"
#include "mikroptim/optim/fd_formula.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "mikroptim/optim/variant_generator.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/adjoint_record.hpp"
#include "mikroptim/trace/surface_extrema.hpp"

#include "parallel_variants.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace mko {

namespace {

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument("Adjungierte Jacobi: " + message);
}

void check_extra(const AdjointJacobianOptions& options) {
    if (options.extra.count < 0) {
        invalid("negative Zahl von Zusatzresiduen");
    }
    if (options.extra.count > 0 && !options.extra.evaluate) {
        invalid("Zusatzresiduen ohne Auswertungsfunktion");
    }
    check_trace_extra_residuals(options.trace_extra, "Adjungierte Jacobi");
}

// Zahl der Läufe aus x0 (Prüfung wie make_fd_jacobian_batch).
int run_count(const VariableSet& set, std::span<const double> x0) {
    const int p = set.size();
    if (p == 0) {
        invalid("keine Variablen");
    }
    if (x0.empty() || x0.size() % static_cast<std::size_t>(p) != 0) {
        invalid("x0 hat " + std::to_string(x0.size()) + " Werte, erwartet ein Vielfaches von P = " +
                std::to_string(p) + " (mindestens ein Lauf)");
    }
    const std::size_t runs = x0.size() / static_cast<std::size_t>(p);
    if (runs > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        invalid("zu viele Läufe");
    }
    return static_cast<int>(runs);
}

// Setzt Spalte `p` für den Parameter `slot` (−1 = frei); jeder Parameter höchstens eine Variable.
void assign(std::vector<int>& slots, std::size_t slot, int p, const std::string& name) {
    if (slots[slot] >= 0) {
        invalid("Parameter von " + name + " gehört schon zu einer anderen Variable");
    }
    slots[slot] = p;
}

} // namespace

// ---- AdjointParameters ---------------------------------------------------------------------

AdjointParameters::AdjointParameters(const VariableSet& set, const SystemBatchHost& batch)
    : variables_(set.size()), surfaces_(batch.surfaces()), wavelengths_(batch.wavelengths()),
      ghost_plates_(batch.topology().ghost_plates), ghost_terms_(batch.topology().ghost_terms) {
    if (!(batch.topology() == set.topology())) {
        invalid("Batch mit anderer Topologie als die Variablenmenge");
    }
    const auto S = static_cast<std::size_t>(surfaces_);
    const auto L = static_cast<std::size_t>(wavelengths_);
    const auto GK =
        static_cast<std::size_t>(ghost_plates_) * static_cast<std::size_t>(ghost_terms_);
    curvature_.assign(S, -1);
    thickness_.assign(S, -1);
    glass_.assign(S * 3, -1);
    ghost_.assign(GK, -1);
    bool has_glass = false;
    for (int k = 0; k < variables_; ++k) {
        const Variable& var = set[k];
        const auto s = static_cast<std::size_t>(var.batch_surface);
        switch (var.type) {
        case VariableType::Curvature:
            assign(curvature_, s, k, var.name);
            break;
        case VariableType::Thickness:
        case VariableType::WorkingDistance:
            // Spalt mit Geisterplatten: Batch-Element = x − offset, Ableitung 1
            assign(thickness_, s, k, var.name);
            break;
        case VariableType::GhostCoefficient:
            assign(ghost_,
                   static_cast<std::size_t>(var.ghost_plate) *
                           static_cast<std::size_t>(ghost_terms_) +
                       static_cast<std::size_t>(var.ghost_order - 1),
                   k, var.name);
            break;
        case VariableType::Glass:
            assign(glass_, s * 3 + static_cast<std::size_t>(var.glass_component), k, var.name);
            has_glass = true;
            break;
        }
    }
    slope_.assign(L * 3, 0.0);
    for (std::size_t l = 0; l < L; ++l) {
        double w = 0.0;
        if (has_glass) {
            const double um = nm_to_um(batch.wavelength_nm(static_cast<int>(l)));
            if (!(buchdahl_denominator(um) > 0.0)) {
                invalid("Buchdahl-Koordinate bei " +
                        std::to_string(batch.wavelength_nm(static_cast<int>(l))) +
                        " nm nicht definiert");
            }
            w = buchdahl_omega(um);
        }
        slope_[l * 3 + 0] = 1.0;
        slope_[l * 3 + 1] = w;
        slope_[l * 3 + 2] = w * w;
    }
}

AdjointParameterMap AdjointParameters::view() const {
    AdjointParameterMap m;
    m.variables = variables_;
    m.surfaces = surfaces_;
    m.wavelengths = wavelengths_;
    m.ghost_plates = ghost_plates_;
    m.ghost_terms = ghost_terms_;
    m.curvature = curvature_.data();
    m.thickness = thickness_.data();
    m.glass = glass_.data();
    m.glass_slope = slope_.data();
    m.ghost = ghost_.empty() ? nullptr : ghost_.data();
    return m;
}

// ---- Ergebnis --------------------------------------------------------------------------------

std::size_t AdjointJacobian::hard_failed_count() const {
    std::size_t n = 0;
    for (const std::uint8_t s : status) {
        if (is_hard_failure(static_cast<TraceStatus>(s))) {
            ++n;
        }
    }
    return n;
}

// ---- Batch und Zusatzzeilen ------------------------------------------------------------------

SystemBatchHost make_adjoint_batch(const VariableSet& set, const SystemBatchHost& base,
                                   std::span<const double> x0) {
    const int runs = run_count(set, x0);
    if (base.variants() != 1 && base.variants() != runs) {
        invalid("die Vorlage hat " + std::to_string(base.variants()) +
                " Varianten, erwartet 1 oder " + std::to_string(runs) + " (eine pro Lauf)");
    }
    const auto P = static_cast<std::size_t>(set.size());
    SystemBatchHost out = base; // gemeinsame Felder aus der Vorlage
    out.resize(runs);
    for (int i = 0; i < runs; ++i) {
        out.copy_variant_from(base, base.variants() == 1 ? 0 : i, i);
        set.apply(out, i, x0.subspan(static_cast<std::size_t>(i) * P, P));
    }
    return out;
}

AdjointJacobianOptions adjoint_options_of(const FdJacobianOptions& options) {
    AdjointJacobianOptions ao;
    ao.quadrature = options.quadrature;
    ao.extra = options.extra;
    ao.trace_extra = options.trace_extra;
    ao.extra_threads = options.extra_threads;
    return ao;
}

FdJacobianOptions fd_options_of(const AdjointJacobianOptions& options) {
    FdJacobianOptions fo;
    fo.quadrature = options.quadrature;
    fo.scheme = FdScheme::Central;
    fo.extra = options.extra;
    fo.trace_extra = options.trace_extra;
    fo.extra_threads = options.extra_threads;
    return fo;
}

ExtraRowsJacobian extra_rows_from_fd(const FdJacobianBatch& fd, std::span<const double> extra,
                                     int extra_count) {
    if (fd.runs < 1 || fd.patterns.size() != static_cast<std::size_t>(fd.runs)) {
        invalid("FD-Batch ohne Läufe oder ohne Muster");
    }
    const FdScheme scheme = fd.patterns.front().scheme;
    const int P = static_cast<int>(fd.patterns.front().nominal.size());
    const int n = fd.variants_per_run;
    if (extra_count < 0 || extra.size() != static_cast<std::size_t>(fd.batch.variants()) *
                                               static_cast<std::size_t>(extra_count)) {
        invalid("Zusatzresiduen passen nicht zum FD-Batch");
    }
    ExtraRowsJacobian out;
    out.runs = fd.runs;
    out.variables = P;
    out.rows = extra_count;
    const auto E = static_cast<std::size_t>(extra_count);
    out.residuals.resize(static_cast<std::size_t>(fd.runs) * E);
    out.jacobian.resize(static_cast<std::size_t>(fd.runs) * static_cast<std::size_t>(P) * E);
    const auto value = [&](int v, std::size_t e) {
        return extra[static_cast<std::size_t>(v) * E + e];
    };
    for (int i = 0; i < fd.runs; ++i) {
        const FdPattern& pat = fd.patterns[static_cast<std::size_t>(i)];
        const int v0 = i * n;
        for (std::size_t e = 0; e < E; ++e) {
            out.residuals[static_cast<std::size_t>(i) * E + e] = value(v0, e);
        }
        for (int k = 0; k < P; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            const int va = v0 + fd_plus_variant(scheme, k);
            const int vb = scheme == FdScheme::Central ? v0 + fd_minus_variant(k) : va;
            const double da = pat.step_plus[kk];
            const double db = scheme == FdScheme::Central ? pat.step_minus[kk] : 0.0;
            double* col = out.jacobian.data() +
                          (static_cast<std::size_t>(i) * static_cast<std::size_t>(P) + kk) * E;
            for (std::size_t e = 0; e < E; ++e) {
                col[e] = fd_derivative(pat.stencil[kk], value(v0, e), value(va, e), value(vb, e),
                                       da, db);
            }
        }
    }
    return out;
}

ExtraRowsJacobian extra_rows_untraced(const VariableSet& set, const SystemBatchHost& base,
                                      std::span<const double> x0,
                                      const AdjointJacobianOptions& options) {
    check_extra(options);
    if (options.trace_extra.count != 0) {
        invalid("extra_rows_untraced: trace-abhängige Zusatzresiduen brauchen den FD-Batch "
                "aller Läufe (extra_rows_cpu)");
    }
    const int runs = run_count(set, x0);
    if (base.variants() != 1 && base.variants() != runs) {
        invalid("die Vorlage hat " + std::to_string(base.variants()) +
                " Varianten, erwartet 1 oder " + std::to_string(runs) + " (eine pro Lauf)");
    }
    const int P = set.size();
    const int E = options.extra_count();
    ExtraRowsJacobian out;
    out.runs = runs;
    out.variables = P;
    out.rows = E;
    if (E == 0) {
        return out;
    }
    const auto e = static_cast<std::size_t>(E);
    const auto p = static_cast<std::size_t>(P);
    out.residuals.resize(static_cast<std::size_t>(runs) * e);
    out.jacobian.resize(static_cast<std::size_t>(runs) * p * e);
    // Vorlage pro Lauf: bei einer Variante für alle Läufe `base` selbst, sonst eine Kopie mit
    // einer Variante (gemeinsame Felder einmal kopiert), in die die Variante des Laufs kommt
    SystemBatchHost proto;
    if (base.variants() != 1) {
        proto = base;
        proto.resize(1);
    }
    detail::parallel_for_variants(0, runs, options.extra_threads, [&](int i) {
        const auto ii = static_cast<std::size_t>(i);
        const std::span<const double> xi = x0.subspan(ii * p, p);
        // Eine Variante als Arbeitsfläche: Kopie der Vorlage, dann apply(y) für x₀ und jeden
        // FD-Punkt wie write_fd_variants (apply schreibt alle von x abhängigen Felder, die
        // Variante ist also bitgleich zur entsprechenden Variante des FD-Batches)
        SystemBatchHost one = base.variants() == 1 ? base : proto;
        if (base.variants() != 1) {
            one.copy_variant_from(base, i, 0);
        }
        const FdPattern pat = fd_pattern(set, xi, FdScheme::Central);
        std::vector<double> y(xi.begin(), xi.end());
        std::vector<double> r0(e);
        std::vector<double> ra(e);
        std::vector<double> rb(e);
        set.apply(one, 0, y);
        options.extra.evaluate(one, 0, r0);
        std::copy(r0.begin(), r0.end(),
                  out.residuals.begin() + static_cast<std::ptrdiff_t>(ii * e));
        for (std::size_t k = 0; k < p; ++k) {
            y[k] = pat.point_plus[k];
            set.apply(one, 0, y);
            options.extra.evaluate(one, 0, ra);
            y[k] = pat.point_minus[k];
            set.apply(one, 0, y);
            options.extra.evaluate(one, 0, rb);
            y[k] = xi[k];
            double* col = out.jacobian.data() + (ii * p + k) * e;
            for (std::size_t j = 0; j < e; ++j) {
                col[j] = fd_derivative(pat.stencil[k], r0[j], ra[j], rb[j], pat.step_plus[k],
                                       pat.step_minus[k]);
            }
        }
    });
    return out;
}

ExtraRowsJacobian extra_rows_cpu(const VariableSet& set, const SystemBatchHost& base,
                                 std::span<const double> x0,
                                 const AdjointJacobianOptions& options) {
    check_extra(options);
    const int runs = run_count(set, x0);
    const int E = options.extra_count();
    if (E == 0) {
        ExtraRowsJacobian out;
        out.runs = runs;
        out.variables = set.size();
        return out;
    }
    if (options.trace_extra.count == 0) {
        return extra_rows_untraced(set, base, x0, options);
    }
    const FdJacobianBatch fd = make_fd_jacobian_batch(set, base, x0, FdScheme::Central);
    const ResidualLayout layout(base.topology(), options.quadrature);
    const SurfaceExtremaResult extrema = options.trace_extra.count > 0
                                             ? surface_extrema_cpu(fd.batch, layout)
                                             : SurfaceExtremaResult{};
    const std::vector<double> extra = fd_extra_residuals(fd_options_of(options), fd.batch, extrema);
    return extra_rows_from_fd(fd, extra, E);
}

void insert_extra_rows(const ExtraRowsJacobian& extra, AdjointJacobian& out) {
    const JacobianShape& sh = out.shape;
    if (extra.runs != sh.runs || extra.variables != sh.variables || extra.rows != sh.extra_rows ||
        out.jacobian.size() != sh.jacobian_size() || out.residuals.size() != sh.residual_size()) {
        invalid("Zusatzzeilen passen nicht zur Jacobi-Matrix");
    }
    const auto E = static_cast<std::size_t>(extra.rows);
    if (E == 0) {
        return;
    }
    const auto rv = static_cast<std::size_t>(sh.ray_rows);
    const auto R = static_cast<std::size_t>(sh.rows());
    for (int i = 0; i < sh.runs; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        for (std::size_t e = 0; e < E; ++e) {
            out.residuals[ii * R + rv + e] = extra.residuals[ii * E + e];
        }
        for (int p = 0; p < sh.variables; ++p) {
            const std::size_t src =
                (ii * static_cast<std::size_t>(sh.variables) + static_cast<std::size_t>(p)) * E;
            double* col = out.jacobian.data() + out.column_offset(i, p) + rv;
            for (std::size_t e = 0; e < E; ++e) {
                col[e] = extra.jacobian[src + e];
            }
        }
    }
}

// ---- CPU-Referenz ----------------------------------------------------------------------------

AdjointJacobian adjoint_jacobian_cpu(const VariableSet& set, const SystemBatchHost& base,
                                     std::span<const double> x0,
                                     const AdjointJacobianOptions& options) {
    check_extra(options);
    const ResidualLayout layout(base.topology(), options.quadrature);
    const SystemBatchHost batch = make_adjoint_batch(set, base, x0);
    const AdjointParameters params(set, batch);
    const AdjointParameterMap map = params.view();

    const int runs = batch.variants();
    const int P = set.size();
    const std::int64_t rays = layout.rays_per_variant();
    const std::int64_t rv = layout.trace_rows();
    AdjointJacobian out;
    out.shape = JacobianShape{runs, P, rv, options.extra_count()};
    out.rays_per_variant = rays;
    const std::int64_t R = out.shape.rows();
    out.jacobian.assign(out.shape.jacobian_size(), 0.0);
    out.residuals.assign(out.shape.residual_size(), 0.0);
    out.status.assign(static_cast<std::size_t>(runs) * static_cast<std::size_t>(rays), 0);

    // r₀ der Trace-Zeilen und Status pro Variante, danach pro Paar (Variante, Feld) für jede
    // Wellenlänge Vorwärts- und Rückwärtsdurchlauf und Kolbenprojektion. Jede Einheit schreibt
    // nur ihre eigenen Zeilen; das Ergebnis hängt nicht von der Zahl der Threads ab.
    detail::parallel_for_variants(0, runs, options.threads, [&](int v) {
        const ResidualResult res = residuals_cpu(batch, v, layout);
        double* const r0 =
            out.residuals.data() + static_cast<std::size_t>(v) * static_cast<std::size_t>(R);
        for (std::int64_t r = 0; r < rv; ++r) {
            r0[r] = res.residual(0, r);
        }
        std::copy(res.status.begin(), res.status.end(),
                  out.status.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(v) *
                                                                   static_cast<std::size_t>(rays)));
    });
    const SystemBatchView<double> view = batch.view();
    const bool soft_mode = layout.soft_failure();
    const bool point_source = view.source == SourceKind::Point;
    const int F = layout.fields();
    detail::parallel_for_variants(0, runs * F, options.threads, [&](int pair) {
        const int v = pair / F;
        const int f = pair % F;
        const auto rec = std::make_unique<AdjointRecord<double>>();
        std::vector<double> weight;
        std::vector<double> root_weight;
        std::vector<unsigned char> piston;
        const SurfaceTableView<double> table = view.surface_table(v);
        double* const col0 = out.jacobian.data() + out.column_offset(v, 0);
        const PupilQuadrature<double>& q = layout.quadrature(f);
        const Vec3<double> center = batch_image_point(view, table, v, f);
        const int n = layout.samples(f);
        for (int l = 0; l < layout.wavelengths(); ++l) {
            const std::int64_t offset = layout.offset(f, l);
            const double scale = residual_scale(view, f, l);
            const double pscale =
                soft_mode ? penalty_scale(view, f, l, layout.settings().fail_weight) : 0.0;
            weight.assign(static_cast<std::size_t>(n), 0.0);
            root_weight.assign(static_cast<std::size_t>(n), 0.0);
            piston.assign(static_cast<std::size_t>(n), 0);
            for (int k = 0; k < n; ++k) {
                const auto kk = static_cast<std::size_t>(k);
                const PupilSample<double> s = sample(q, k);
                SoftFailure<double> soft{0.0, TraceStatus::Ok};
                NoSurfaceObserver* const no_observer = nullptr;
                double opl = 0.0;
                const TraceStatus st =
                    batch_ray_opl(view, table, l, v, f, l, s, center, &opl, no_observer,
                                  soft_mode ? &soft : nullptr, rec.get());
                const TraceStatus expected = static_cast<TraceStatus>(
                    out.status[static_cast<std::size_t>(v) * static_cast<std::size_t>(rays) +
                               static_cast<std::size_t>(offset + k)]);
                if (st != expected) {
                    throw std::logic_error("Adjungierte Jacobi: Trace mit Aufzeichnung "
                                           "liefert einen anderen Status");
                }
                weight[kk] = s.weight;
                root_weight[kk] = residual_weight(s);
                piston[kk] = in_piston(st) ? 1 : 0;
                const JacobianRowWriter<double> opl_row{map, l, col0, R, offset + k, 1.0};
                const JacobianRowWriter<double> penalty_row{
                    map, l, col0, R, rays + offset + k, pscale * root_weight[kk]};
                ray_gradient_rows(table, l, *rec, st, soft.first, soft_mode, center, point_source,
                                  opl_row, penalty_row);
            }
            // Kolbenabzug als lineare Projektion (Summen aufsteigend in k)
            double w_ok = 0.0;
            for (int k = 0; k < n; ++k) {
                if (piston[static_cast<std::size_t>(k)] != 0) {
                    w_ok += weight[static_cast<std::size_t>(k)];
                }
            }
            for (int p = 0; p < P; ++p) {
                double* col = col0 + static_cast<std::int64_t>(p) * R + offset;
                if (!(w_ok > 0.0)) {
                    for (int k = 0; k < n; ++k) {
                        col[k] = 0.0;
                    }
                    continue;
                }
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                    if (piston[static_cast<std::size_t>(k)] != 0) {
                        sum += weight[static_cast<std::size_t>(k)] * col[k];
                    }
                }
                const double mean = sum / w_ok;
                for (int k = 0; k < n; ++k) {
                    const auto kk = static_cast<std::size_t>(k);
                    col[k] = piston[kk] != 0 ? scale * root_weight[kk] * (col[k] - mean) : 0.0;
                }
            }
        }
    });
    if (options.extra_count() > 0) {
        insert_extra_rows(extra_rows_cpu(set, base, x0, options), out);
    }
    return out;
}

} // namespace mko
