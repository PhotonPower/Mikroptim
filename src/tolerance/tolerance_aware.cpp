// Toleranzbewusste Optimierung (M11-T6): erwartete Merit über feste Störungen eines
// Toleranzsatzes als zusätzliche Varianten im Batch des gebündelten LM
// (docs/optimierung-erweiterungen.md, Abschnitt 6).

#include "mikroptim/tolerance/tolerance_aware.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace mko {

namespace {

std::size_t to_size(std::int64_t i) {
    return static_cast<std::size_t>(i);
}

[[noreturn]] void invalid(const std::string& what) {
    throw std::invalid_argument("toleranzbewusste Optimierung: " + what);
}

// Topologie der inneren Batches: nicht koaxial, volle Pupille (make_non_coaxial).
BatchTopology inner_topology(const SystemBatchHost& nominal) {
    if (nominal.variants() < 1) {
        invalid("Ausgangsbatch ohne Varianten");
    }
    const std::array<int, 1> first{0};
    return make_non_coaxial(lm_template_batch(nominal, first)).topology();
}

// Systembeschreibung ohne Schranken der Variablen (für die inneren Läufe).
SystemDescription without_bounds(const SystemDescription& system) {
    SystemDescription out = system;
    if (out.variables) {
        for (VariableSpec& v : out.variables->items) {
            v.min.reset();
            v.max.reset();
        }
    }
    return out;
}

} // namespace

const char* to_string(ToleranceAwareScheme scheme) {
    switch (scheme) {
    case ToleranceAwareScheme::PlusMinus:
        return "plus_minus";
    case ToleranceAwareScheme::Samples:
        return "samples";
    }
    return "?";
}

std::optional<ToleranceAwareScheme> tolerance_aware_scheme_from_string(std::string_view name) {
    if (name == "plus_minus") {
        return ToleranceAwareScheme::PlusMinus;
    }
    if (name == "samples") {
        return ToleranceAwareScheme::Samples;
    }
    return std::nullopt;
}

ToleranceInnerFactory cpu_tolerance_inner() {
    return [](const VariableSet& set, const LmOptions& options, const SystemBatchHost& base) {
        return std::unique_ptr<LmBackend>(std::make_unique<CpuLmBackend>(set, base, options));
    };
}

ToleranceAwareModel::ToleranceAwareModel(const SystemDescription& system,
                                         const SystemBatchHost& nominal, const VariableSet& set,
                                         const ToleranceAwareSettings& settings)
    : settings_(settings) {
    if (!(settings.alpha > 0.0 && settings.alpha <= 1.0)) {
        invalid("Gewicht alpha muss in (0, 1] liegen");
    }
    if (!system.tolerances || system.tolerances->items.empty()) {
        invalid("kein Toleranzsatz (Abschnitt tolerances fehlt oder ist leer)");
    }
    const BatchTopology topology = inner_topology(nominal);
    model_ = ToleranceModel(system, topology);
    if (model_.draws() == 0) {
        invalid("der Toleranzsatz hat keine Toleranzen");
    }
    inner_set_ = VariableSet(without_bounds(system), topology);
    if (inner_set_.size() != set.size() || inner_set_.empty()) {
        invalid("die Variablen passen nicht zur Systembeschreibung");
    }
    for (int p = 0; p < set.size(); ++p) {
        const Variable& a = set[p];
        const Variable& b = inner_set_[p];
        if (a.type != b.type || a.file_surface != b.file_surface ||
            a.ghost_plate != b.ghost_plate || a.ghost_order != b.ghost_order ||
            a.glass_component != b.glass_component) {
            invalid("die Variablen passen nicht zur Systembeschreibung (Variable " + a.name + ")");
        }
    }
    // Störungen von Variablen: relative Radien skalieren die Krümmung (D_s), Brechzahl und Abbe an
    // Glasvariablen sind nicht darstellbar (die Variable schreibt die Brechzahlen neu)
    radius_tolerances_.assign(to_size(set.size()), {});
    const std::span<const ResolvedTolerance> tolerances = model_.tolerances();
    for (std::size_t k = 0; k < tolerances.size(); ++k) {
        const ResolvedTolerance& t = tolerances[k];
        for (int p = 0; p < set.size(); ++p) {
            const Variable& v = inner_set_[p];
            if (t.type == ToleranceType::Radius && !t.radius_fringes &&
                v.type == VariableType::Curvature && v.file_surface == t.surface) {
                radius_tolerances_[to_size(p)].push_back(static_cast<int>(k));
            }
            if ((t.type == ToleranceType::Index || t.type == ToleranceType::Abbe) &&
                v.type == VariableType::Glass && v.file_surface == t.surface) {
                invalid("Toleranz " + std::string(to_string(t.type)) + " an Fläche " +
                        std::to_string(t.surface) +
                        " betrifft ein Glas mit Glasvariablen (nicht darstellbar)");
            }
        }
    }
    const int draws = model_.draws();
    const auto D = to_size(draws);
    if (settings.scheme == ToleranceAwareScheme::PlusMinus) {
        samples_ = 2 * draws;
        xi_.assign(to_size(samples_) * D, 0.0);
        for (std::size_t d = 0; d < D; ++d) {
            xi_[(2 * d) * D + d] = 1.0;
            xi_[(2 * d + 1) * D + d] = -1.0;
        }
    } else {
        if (settings.samples < 1) {
            invalid("Stichproben: S >= 1 erwartet");
        }
        samples_ = settings.samples;
        xi_.assign(to_size(samples_) * D, 0.0);
        for (int s = 0; s < samples_; ++s) {
            model_.deviates(settings.seed, static_cast<std::uint64_t>(s),
                            std::span<double>(xi_).subspan(to_size(s) * D, D));
        }
    }
    alpha_ = settings.alpha;
    nominal_weight_ = 1.0 - alpha_;
    sample_weight_ = alpha_ / static_cast<double>(samples_);
}

std::span<const double> ToleranceAwareModel::deviates(int s) const {
    if (s < 1 || s > samples_) {
        throw std::out_of_range("ToleranceAwareModel::deviates: Störung " + std::to_string(s) +
                                " existiert nicht");
    }
    const auto D = to_size(model_.draws());
    return std::span<const double>(xi_).subspan(to_size(s - 1) * D, D);
}

SystemBatchHost ToleranceAwareModel::inner_base(const SystemBatchHost& outer) const {
    const SystemBatchHost nominal = make_non_coaxial(outer);
    if (nominal.topology() != model_.topology()) {
        invalid("die Vorlage passt nicht zum Ausgangsbatch");
    }
    const int B = nominal.variants();
    const int S1 = samples_ + 1;
    SystemBatchHost out = nominal;
    out.resize(B * S1);
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S1; ++s) {
            const int v = b * S1 + s;
            out.copy_variant_from(nominal, b, v);
            if (s > 0) {
                model_.apply(out, v, deviates(s));
            }
        }
    }
    return out;
}

void ToleranceAwareModel::perturbed_point(const SystemBatchHost& inner, int b, int s,
                                          std::span<const double> x, SystemBatchHost& scratch,
                                          std::span<double> xs, std::span<double> scale) const {
    const auto P = to_size(inner_set_.size());
    if (x.size() != P || xs.size() != P || scale.size() != P) {
        invalid("perturbed_point: Länge passt nicht zu P");
    }
    if (s == 0) {
        std::copy(x.begin(), x.end(), xs.begin());
        std::fill(scale.begin(), scale.end(), 1.0);
        return;
    }
    const std::span<const double> xi = deviates(s);
    // Nominelles System mit x, darauf die Störung s; x'_s aus dem gestörten System
    scratch.copy_variant_from(inner, b * (samples_ + 1), 0);
    inner_set_.apply(scratch, 0, x);
    model_.apply(scratch, 0, xi);
    inner_set_.extract(scratch, 0, xs);
    // D_s: c' = c/(1 + ε) je relativer Radius-Toleranz an der Krümmung (ToleranceModel::apply)
    const std::span<const ResolvedTolerance> tolerances = model_.tolerances();
    for (std::size_t p = 0; p < P; ++p) {
        double d = 1.0;
        for (const int k : radius_tolerances_[p]) {
            const ResolvedTolerance& t = tolerances[to_size(k)];
            const double eps = t.limit * xi[to_size(model_.draw_offset(k))] + 0.0;
            d /= 1.0 + eps;
        }
        scale[p] = d;
    }
}

ToleranceAwareBackend::ToleranceAwareBackend(const ToleranceAwareModel& model,
                                             const ToleranceInnerFactory& inner,
                                             const SystemBatchHost& outer_base,
                                             const LmOptions& options)
    : model_(model), outer_templates_(outer_base.variants()),
      inner_base_(model.inner_base(outer_base)) {
    if (!inner) {
        invalid("kein inneres Backend");
    }
    const std::array<int, 1> first{0};
    scratch_ = make_non_coaxial(lm_template_batch(outer_base, first));
    inner_ = inner(model_.inner_set(), options, inner_base_);
    if (!inner_) {
        invalid("das innere Backend fehlt");
    }
}

void ToleranceAwareBackend::inner_points(std::span<const int> runs, std::span<const double> x) {
    const auto P = to_size(model_.outer_variables());
    const auto S1 = to_size(model_.samples() + 1);
    const std::size_t n = runs.size();
    if (x.size() != n * P) {
        invalid("x passt nicht zu den Läufen");
    }
    inner_runs_.resize(n * S1);
    inner_x_.resize(n * S1 * P);
    scale_.resize(n * S1 * P);
    for (std::size_t j = 0; j < n; ++j) {
        const int b = outer_templates_ == 1 ? 0 : runs[j];
        if (b < 0 || b >= outer_templates_) {
            throw std::out_of_range("toleranzbewusste Optimierung: Lauf " +
                                    std::to_string(runs[j]) + " hat keine Vorlage");
        }
        for (std::size_t s = 0; s < S1; ++s) {
            const std::size_t i = j * S1 + s;
            inner_runs_[i] = b * static_cast<int>(S1) + static_cast<int>(s);
            model_.perturbed_point(inner_base_, b, static_cast<int>(s), x.subspan(j * P, P),
                                   scratch_, std::span<double>(inner_x_).subspan(i * P, P),
                                   std::span<double>(scale_).subspan(i * P, P));
        }
    }
}

void ToleranceAwareBackend::normal_equations(std::span<const int> runs, std::span<const double> x,
                                             std::span<double> a, std::span<double> g) {
    const auto P = to_size(model_.outer_variables());
    const auto S1 = to_size(model_.samples() + 1);
    const std::size_t n = runs.size();
    inner_points(runs, x);
    std::vector<double> ia(n * S1 * P * P);
    std::vector<double> ig(n * S1 * P);
    inner_->normal_equations(inner_runs_, inner_x_, ia, ig);
    // A = Σ_s w_s D_s A_s D_s, g = Σ_s w_s D_s g_s in fester Reihenfolge s = 0 … S
    for (std::size_t j = 0; j < n; ++j) {
        const std::span<double> aj = a.subspan(j * P * P, P * P);
        const std::span<double> gj = g.subspan(j * P, P);
        std::fill(aj.begin(), aj.end(), 0.0);
        std::fill(gj.begin(), gj.end(), 0.0);
        for (std::size_t s = 0; s < S1; ++s) {
            const std::size_t i = j * S1 + s;
            const double w = model_.weight(static_cast<int>(s));
            const double* d = scale_.data() + i * P;
            const double* as = ia.data() + i * P * P;
            const double* gs = ig.data() + i * P;
            for (std::size_t k = 0; k < P; ++k) {
                const double wk = w * d[k];
                for (std::size_t l = 0; l < P; ++l) {
                    aj[k * P + l] += wk * d[l] * as[k * P + l];
                }
                gj[k] += wk * gs[k];
            }
        }
    }
}

ToleranceAwareMerits ToleranceAwareBackend::merits(std::span<const int> runs,
                                                   std::span<const double> x) {
    inner_points(runs, x);
    const std::size_t m = inner_runs_.size();
    ToleranceAwareMerits out;
    out.merit.assign(m, 0.0);
    out.failed.assign(m, 0);
    out.hard_failed.assign(m, 0);
    inner_->merit(inner_runs_, inner_x_, out.merit, out.failed, out.hard_failed);
    return out;
}

void ToleranceAwareBackend::merit(std::span<const int> runs, std::span<const double> x,
                                  std::span<double> merit, std::span<std::int64_t> failed,
                                  std::span<std::int64_t> hard_failed) {
    const auto S1 = to_size(model_.samples() + 1);
    const ToleranceAwareMerits m = merits(runs, x);
    for (std::size_t j = 0; j < runs.size(); ++j) {
        double f = 0.0;
        std::int64_t nf = 0;
        std::int64_t nh = 0;
        for (std::size_t s = 0; s < S1; ++s) {
            const std::size_t i = j * S1 + s;
            f += model_.weight(static_cast<int>(s)) * m.merit[i];
            nf += m.failed[i];
            nh += m.hard_failed[i];
        }
        merit[j] = f;
        failed[j] = nf;
        hard_failed[j] = nh;
    }
}

LmResult optimize_lm_tolerance_aware(const ToleranceAwareModel& model,
                                     const ToleranceInnerFactory& inner, const VariableSet& set,
                                     const SystemBatchHost& base, std::span<const double> x0,
                                     const LmOptions& options) {
    if (set.size() != model.outer_variables()) {
        invalid("die Variablen passen nicht zum Modell");
    }
    const LmBackendFactory factory = [&model, &inner](const LmOptions& o,
                                                      const SystemBatchHost& b) {
        return std::unique_ptr<LmBackend>(
            std::make_unique<ToleranceAwareBackend>(model, inner, b, o));
    };
    return optimize_lm_rounds(factory, set, base, x0, options);
}

} // namespace mko
