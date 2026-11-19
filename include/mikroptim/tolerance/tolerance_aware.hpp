#pragma once

// Toleranzbewusste Optimierung (M11-T6, Host): erwartete Merit über S feste Störungen eines
// Toleranzsatzes,
//   F_tol(x) = (1 − α) F(x) + α/S · Σ_s F(x ⊕ δ_s),
// mit den gestörten Systemen als gewöhnliche Varianten im Batch (ADR-0004). Ein Backend des
// gebündelten LM (LmBackend) legt pro Lauf 1 + S Läufe eines inneren Backends (CPU oder GPU) an
// und setzt A = JᵀJ, g = Jᵀr und F in fester Reihenfolge aus ihnen zusammen. Störungen über
// ToleranceModel::apply (M8-T4, ADR-0011), deterministisch: ±δ je Zufallsgröße oder eine feste
// Stichprobe mit Startwert. Verfahren und Formeln: docs/optimierung-erweiterungen.md, Abschnitt 6.

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "mikroptim/tolerance/tolerance_model.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace mko {

// Wahl der festen Störungen.
enum class ToleranceAwareScheme {
    PlusMinus, // S = 2 D: pro Zufallsgröße ξ = +1 und ξ = −1, alle übrigen 0 (Standard)
    Samples,   // S Stichproben ToleranceModel::deviates(seed, 0 … S − 1) (wie die Tolerierung)
};

const char* to_string(ToleranceAwareScheme scheme);
std::optional<ToleranceAwareScheme> tolerance_aware_scheme_from_string(std::string_view name);

inline constexpr double kDefaultToleranceAwareWeight = 0.5;

struct ToleranceAwareSettings {
    double alpha = kDefaultToleranceAwareWeight; // Gewicht des gestörten Anteils, 0 < α ≤ 1
    ToleranceAwareScheme scheme = ToleranceAwareScheme::PlusMinus;
    int samples = 0;        // nur Samples: S ≥ 1
    std::uint64_t seed = 0; // nur Samples
};

// Inneres Backend für eine Menge von Variablen und eine Vorlage (CPU: CpuLmBackend, GPU:
// cuda::GpuLmBackend über mikroptim/cuda/tolerance.hpp). set, base und options leben länger als
// das Backend.
using ToleranceInnerFactory = std::function<std::unique_ptr<LmBackend>(
    const VariableSet& set, const LmOptions& options, const SystemBatchHost& base)>;

// CPU-Backend (CpuLmBackend) als inneres Backend.
ToleranceInnerFactory cpu_tolerance_inner();

// Feste Störungen eines Toleranzsatzes für die Optimierung mit den Variablen `set` am
// Ausgangsbatch `nominal` (koaxial oder nicht; die inneren Batches sind nicht koaxial mit voller
// Pupille, make_non_coaxial).
//
// Störungen von Parametern, die Variablen sind (Radius an einer variablen Krümmung, Dicke bzw.
// axiale Elementverschiebung δz an einer variablen Dicke oder am Arbeitsabstand), wirken über den
// Variablenvektor: Das gestörte System an x ist apply_s(x) = Störung s auf das nominelle System
// mit x, sein Variablenvektor x'_s = extract(apply_s(x)); die Vorlage der Störung s trägt die
// übrigen (festen) Störungen. Mit D_s = diag(∂x'_s/∂x) (1/(1 + ε) für einen relativen Radius,
// sonst 1) gilt J_s(x) = J(x'_s) D_s.
//
// Wirft FormatError bei ungültigem Toleranzsatz, std::invalid_argument ohne Toleranzen, bei
// ungültigen Einstellungen, wenn Datei, Batch und Variablen nicht zusammenpassen, oder bei einer
// Brechzahl- bzw. Abbe-Toleranz an einem Glas mit Glasvariablen (nicht darstellbar).
class ToleranceAwareModel {
public:
    ToleranceAwareModel(const SystemDescription& system, const SystemBatchHost& nominal,
                        const VariableSet& set, const ToleranceAwareSettings& settings);

    int samples() const { return samples_; } // S
    double alpha() const { return alpha_; }
    // Gewicht des Teilsystems s: w_0 = 1 − α (nominal), w_s = α/S (s = 1 … S).
    double weight(int s) const { return s == 0 ? nominal_weight_ : sample_weight_; }
    const ToleranceAwareSettings& settings() const { return settings_; }
    const ToleranceModel& model() const { return model_; }
    // Variablen der inneren Batches (nicht koaxiale Topologie, ohne Schranken: die gestörten
    // Punkte x'_s dürfen außerhalb der Schranken liegen).
    const VariableSet& inner_set() const { return inner_set_; }
    int outer_variables() const { return inner_set_.size(); }
    // ξ der Störung s = 1 … S (Länge model().draws()).
    std::span<const double> deviates(int s) const;

    // Vorlagen der inneren Läufe zu einer äußeren Vorlage `outer` (B Varianten, Topologie wie der
    // Ausgangsbatch): Variante b(1 + S) + s = nominelle (s = 0) bzw. gestörte Vorlage s von
    // Variante b, nicht koaxial.
    SystemBatchHost inner_base(const SystemBatchHost& outer) const;

    // x'_s und D_s (je Länge P) für Variante b der Vorlage `inner` (aus inner_base) am Punkt x;
    // s = 0: x und 1. `scratch` ist ein Batch mit der Topologie von `inner` (wird überschrieben).
    void perturbed_point(const SystemBatchHost& inner, int b, int s, std::span<const double> x,
                         SystemBatchHost& scratch, std::span<double> xs,
                         std::span<double> scale) const;

private:
    ToleranceAwareSettings settings_;
    ToleranceModel model_;
    VariableSet inner_set_;
    int samples_ = 0;
    double alpha_ = 0.0;
    double nominal_weight_ = 0.0;
    double sample_weight_ = 0.0;
    std::vector<double> xi_; // [S][draws]
    // pro Variable: Indizes der relativen Radius-Toleranzen an ihrer Krümmung (sonst leer)
    std::vector<std::vector<int>> radius_tolerances_;
};

// Merit der Teilsysteme von Läufen: [Lauf][1 + S], nominal zuerst.
struct ToleranceAwareMerits {
    std::vector<double> merit;
    std::vector<std::int64_t> failed;
    std::vector<std::int64_t> hard_failed;
};

// LM-Backend der erwarteten Merit. Zu jedem Lauf r (Vorlage b = r bzw. 0 bei einer Vorlage)
// gehören die inneren Läufe b(1 + S) + s an x'_s; alle Läufe eines Aufrufs in einem Aufruf des
// inneren Backends (ein Batch, ein Block pro Variante, Feld und Wellenlänge). Zusammensetzung
// pro Lauf in fester Reihenfolge s = 0 … S:
//   A = Σ_s w_s D_s A_s D_s,  g = Σ_s w_s D_s g_s,  F = Σ_s w_s F_s,
// versagte Strahlen als Summe über alle Teilsysteme. Bitgleich für jede Zahl und Reihenfolge der
// Läufe, wenn das innere Backend es ist.
class ToleranceAwareBackend final : public LmBackend {
public:
    ToleranceAwareBackend(const ToleranceAwareModel& model, const ToleranceInnerFactory& inner,
                          const SystemBatchHost& outer_base, const LmOptions& options);

    void normal_equations(std::span<const int> runs, std::span<const double> x, std::span<double> a,
                          std::span<double> g) override;
    void merit(std::span<const int> runs, std::span<const double> x, std::span<double> merit,
               std::span<std::int64_t> failed, std::span<std::int64_t> hard_failed) override;

    // Merit jedes Teilsystems (ungewichtet).
    ToleranceAwareMerits merits(std::span<const int> runs, std::span<const double> x);

private:
    void inner_points(std::span<const int> runs, std::span<const double> x);

    const ToleranceAwareModel& model_;
    int outer_templates_ = 0;
    SystemBatchHost inner_base_;
    SystemBatchHost scratch_;
    std::unique_ptr<LmBackend> inner_;
    std::vector<int> inner_runs_;
    std::vector<double> inner_x_;
    std::vector<double> scale_;
};

// Gebündelter LM mit der erwarteten Merit (optimize_lm_rounds mit ToleranceAwareBackend; das
// äußere Strafverfahren des weichen Versagens wie bei optimize_lm_cpu). `set` sind die Variablen
// des Ausgangsbatches `base` (Schranken, typische Größen).
LmResult optimize_lm_tolerance_aware(const ToleranceAwareModel& model,
                                     const ToleranceInnerFactory& inner, const VariableSet& set,
                                     const SystemBatchHost& base, std::span<const double> x0,
                                     const LmOptions& options = {});

} // namespace mko
