// Jacobi-Matrix per finiten Differenzen, Host-Teil (docs/optimierung.md, Abschnitt
// „Finite-Differenzen-Jacobi“).

#include "mikroptim/optim/fd_jacobian.hpp"

#include "mikroptim/optim/fd_formula.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "mikroptim/trace/surface_ops.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace mko {

namespace {

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument("FD-Jacobi: " + message);
}

void check_extra(const ExtraResiduals& extra) {
    if (extra.count < 0) {
        invalid("negative Zahl von Zusatzresiduen");
    }
    if (extra.count > 0 && !extra.evaluate) {
        invalid("Zusatzresiduen ohne Auswertungsfunktion");
    }
}

} // namespace

std::size_t JacobianShape::jacobian_size() const {
    return column_count() * static_cast<std::size_t>(rows());
}

std::size_t JacobianShape::residual_size() const {
    return static_cast<std::size_t>(runs) * static_cast<std::size_t>(rows());
}

std::size_t JacobianShape::column_count() const {
    return static_cast<std::size_t>(runs) * static_cast<std::size_t>(variables);
}

FdJacobianBatch make_fd_jacobian_batch(const VariableSet& set, const SystemBatchHost& base,
                                       std::span<const double> x0, FdScheme scheme) {
    const int p = set.size();
    if (p == 0) {
        invalid("keine Variablen");
    }
    if (x0.empty() || x0.size() % static_cast<std::size_t>(p) != 0) {
        invalid("x0 hat " + std::to_string(x0.size()) + " Werte, erwartet ein Vielfaches von P = " +
                std::to_string(p) + " (mindestens ein Lauf)");
    }
    const std::size_t runs = x0.size() / static_cast<std::size_t>(p);
    const int n = fd_variant_count(scheme, p);
    if (runs > static_cast<std::size_t>(std::numeric_limits<int>::max() / n)) {
        invalid("zu viele Läufe für einen Batch");
    }
    if (base.variants() != 1 && static_cast<std::size_t>(base.variants()) != runs) {
        invalid("die Vorlage hat " + std::to_string(base.variants()) +
                " Varianten, erwartet 1 oder " + std::to_string(runs) + " (eine pro Lauf)");
    }
    FdJacobianBatch fd;
    fd.runs = static_cast<int>(runs);
    fd.variants_per_run = n;
    fd.batch = base; // gemeinsame Felder aus der Vorlage
    fd.batch.resize(fd.runs * n);
    fd.patterns.reserve(runs);
    for (int i = 0; i < fd.runs; ++i) {
        const auto xi = x0.subspan(static_cast<std::size_t>(i) * static_cast<std::size_t>(p),
                                   static_cast<std::size_t>(p));
        const int tmpl = base.variants() == 1 ? 0 : i;
        fd.patterns.push_back(write_fd_variants(set, base, tmpl, xi, scheme, fd.batch, i * n));
    }
    return fd;
}

void fd_jacobian_from_residuals(const FdJacobianBatch& fd, const ResidualResult& rays,
                                std::span<const double> extra, int extra_count, FdJacobian& out) {
    if (fd.runs < 1 || fd.patterns.size() != static_cast<std::size_t>(fd.runs)) {
        invalid("FD-Batch ohne Läufe oder ohne Muster");
    }
    const FdScheme scheme = fd.patterns.front().scheme;
    const int p = static_cast<int>(fd.patterns.front().nominal.size());
    const int n = fd.variants_per_run;
    if (n != fd_variant_count(scheme, p) || fd.batch.variants() != fd.runs * n) {
        invalid("FD-Batch passt nicht zu seinen Mustern");
    }
    const int V = fd.batch.variants();
    const std::int64_t nrays = rays.rays_per_variant; // Strahlen (Status)
    const std::int64_t rv = rays.trace_rows();        // Strahl- und Strafresiduen
    if (rays.variants != V || rays.penalty_rows < 0 || rays.stride < rv ||
        rays.residuals.size() !=
            static_cast<std::size_t>(V) * static_cast<std::size_t>(rays.stride)) {
        invalid("Residuen passen nicht zum FD-Batch (" + std::to_string(rays.variants) + " statt " +
                std::to_string(V) + " Varianten)");
    }
    if (rays.status.size() != static_cast<std::size_t>(V) * static_cast<std::size_t>(nrays)) {
        invalid("Status pro Strahl fehlt");
    }
    if (extra_count < 0 ||
        extra.size() != static_cast<std::size_t>(V) * static_cast<std::size_t>(extra_count)) {
        invalid("Zusatzresiduen passen nicht zum FD-Batch");
    }

    out.shape = JacobianShape{fd.runs, p, rv, extra_count};
    out.scheme = scheme;
    out.patterns = fd.patterns;
    const std::int64_t R = out.shape.rows();
    out.jacobian.resize(out.shape.jacobian_size());
    out.residuals.resize(out.shape.residual_size());
    out.status_changes.assign(out.shape.column_count(), 0);

    const auto e = static_cast<std::size_t>(extra_count);
    // Zeile r der Variante v: Strahlresiduum oder Zusatzresiduum.
    const auto value = [&](int v, std::int64_t r) {
        if (r < rv) {
            return rays
                .residuals[static_cast<std::size_t>(v) * static_cast<std::size_t>(rays.stride) +
                           static_cast<std::size_t>(r)];
        }
        return extra[static_cast<std::size_t>(v) * e + static_cast<std::size_t>(r - rv)];
    };
    const auto valid = [&](int v, std::int64_t r) {
        return rays.status[static_cast<std::size_t>(v) * static_cast<std::size_t>(nrays) +
                           static_cast<std::size_t>(r)] ==
               static_cast<std::uint8_t>(TraceStatus::Ok);
    };

    for (int i = 0; i < fd.runs; ++i) {
        const FdPattern& pat = fd.patterns[static_cast<std::size_t>(i)];
        const int v0 = i * n;
        double* r0 =
            out.residuals.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(R);
        for (std::int64_t r = 0; r < R; ++r) {
            r0[r] = value(v0, r);
        }
        for (int k = 0; k < p; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            const FdStencil stencil = pat.stencil[kk];
            const int va = v0 + fd_plus_variant(scheme, k);
            const int vb = scheme == FdScheme::Central ? v0 + fd_minus_variant(k) : va;
            const double da = pat.step_plus[kk];
            const double db = scheme == FdScheme::Central ? pat.step_minus[kk] : 0.0;
            double* col = out.jacobian.data() + out.column_offset(i, k);
            for (std::int64_t r = 0; r < R; ++r) {
                col[r] = fd_derivative(stencil, r0[r], value(va, r), value(vb, r), da, db);
            }
            std::int64_t changes = 0;
            for (std::int64_t r = 0; r < nrays; ++r) {
                const bool ok0 = valid(v0, r);
                if (valid(va, r) != ok0 || valid(vb, r) != ok0) {
                    ++changes;
                }
            }
            out.status_changes[static_cast<std::size_t>(i) * static_cast<std::size_t>(p) + kk] =
                changes;
        }
    }
}

std::vector<double> fd_extra_residuals(const FdJacobianOptions& options,
                                       const SystemBatchHost& batch,
                                       const SurfaceExtremaResult& extrema) {
    check_extra(options.extra);
    check_trace_extra_residuals(options.trace_extra, "FD-Jacobi");
    const int V = batch.variants();
    std::vector<double> extra(static_cast<std::size_t>(V) *
                              static_cast<std::size_t>(options.extra.count));
    evaluate_extra_residuals(options.extra, batch, extra, 0, -1, options.extra_threads);
    if (options.trace_extra.count == 0) {
        return extra;
    }
    std::vector<double> traced(static_cast<std::size_t>(V) *
                               static_cast<std::size_t>(options.trace_extra.count));
    evaluate_trace_extra_residuals(options.trace_extra, batch, extrema, 0, traced, 0, -1,
                                   options.extra_threads);
    return join_extra_rows(extra, options.extra.count, traced, options.trace_extra.count, V);
}

FdJacobian fd_jacobian_cpu(const VariableSet& set, const SystemBatchHost& base,
                           std::span<const double> x0, const FdJacobianOptions& options) {
    check_extra(options.extra);
    check_trace_extra_residuals(options.trace_extra, "FD-Jacobi");
    const ResidualLayout layout(base.topology(), options.quadrature);
    const FdJacobianBatch fd = make_fd_jacobian_batch(set, base, x0, options.scheme);
    const ResidualResult rays = residuals_cpu(fd.batch, layout);
    // Extrema jeder FD-Variante aus ihrem eigenen Trace (nur für trace-abhängige Residuen)
    const SurfaceExtremaResult extrema = options.trace_extra.count > 0
                                             ? surface_extrema_cpu(fd.batch, layout)
                                             : SurfaceExtremaResult{};
    const std::vector<double> extra = fd_extra_residuals(options, fd.batch, extrema);
    FdJacobian out;
    fd_jacobian_from_residuals(fd, rays, extra, options.extra_count(), out);
    return out;
}

} // namespace mko
