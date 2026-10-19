#pragma once

// Geglättete L1-Strafe der Geisterkoeffizienten als Zusatzresiduen (M4-T6, Host). Ein Residuum
// pro Koeffizient a_{g,k} einer bestraften Geisterplatte,
//
//     r = √(μ w_k) · a / √(ε + √(a² + ε²)),   r² = μ w_k (√(a² + ε²) − ε) ≈ μ w_k |a|,
//
// mit w_k = w₀ κ^k. Gradient von r² wie bei der Form √(μ w_k) (a² + ε²)^{1/4} aus dem Plan,
// aber r(0) = 0 und dr/da(0) = √(μ w_k / (2ε)) ≠ 0, sodass das Gauß-Newton-Modell die
// Krümmung μ w_k/ε der Strafe bei a = 0 exakt enthält. Definitionen, Begründung und
// Abschätzung der Restkoeffizienten: docs/optimierung.md, Abschnitte 2 und 3.
// Einstellungen in der Systemdatei: Nebenbedingung `ghost_penalty` (docs/dateiformat.md).

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/optim/extra_residuals.hpp"

#include <span>
#include <vector>

namespace mko {

struct SystemDescription;

// Ein Strafterm: Koeffizient a_k der Geisterplatte `plate` (k = order, 1 … K).
struct GhostPenaltyTerm {
    int plate = 0;
    int order = 1;
    double weight = 0.0;    // μ · w_k ≥ 0 (Einheit Wellen²/mm, wie die Merit pro mm Profil)
    double smoothing = 0.0; // ε > 0 [mm]
    int item = -1;          // Eintrag in constraints.items (−1: von Hand gebaut)

    friend bool operator==(const GhostPenaltyTerm&, const GhostPenaltyTerm&) = default;
};

struct GhostPenalty {
    // Platten in aufsteigender Reihenfolge, je Platte die Ordnungen 1 … K aufsteigend.
    std::vector<GhostPenaltyTerm> terms;

    int size() const { return static_cast<int>(terms.size()); }
    bool empty() const { return terms.empty(); }
};

// Ordnungsgewicht w_k = w₀ · κ^k (k ≥ 1), als Produkt in fester Reihenfolge
// (w₀ · κ · κ · …, plattformunabhängig; std::pow ist nicht überall korrekt gerundet).
double ghost_order_weight(int order, double order_weight, double order_factor);

// Residuum √(weight) · a / √(ε + √(a² + ε²)) und sein Quadrat weight · (√(a² + ε²) − ε)
// (Strafbeitrag zur Merit). weight ≥ 0, ε > 0. Die Wurzel √(a² + ε²) wird mit std::hypot
// gebildet (kein Überlauf für große |a|).
double ghost_penalty_residual(double a, double weight, double smoothing);
double ghost_penalty_square(double a, double weight, double smoothing);
// Ableitung dr/da = √(weight) · (ε + s/2 + ε²/(2s)) / (ε + s)^{3/2} mit s = √(a² + ε²)
// (für Tests und Abschätzungen; die Jacobi-Matrix entsteht wie alle Zusatzresiduen per FD).
double ghost_penalty_derivative(double a, double weight, double smoothing);

// Strafe aus der Systembeschreibung: alle Einträge `ghost_penalty` von `constraints`
// (geprüft mit check_constraints), für jede gewählte Platte alle Ordnungen 1 … K der Platte
// mit weight = μ · w₀ κ^k und ε des Eintrags. Ohne solche Einträge leer.
// Wirft FormatError bei ungültigem Abschnitt.
GhostPenalty make_ghost_penalty(const SystemDescription& system);

// Prüft die Terme gegen eine Batch-Topologie: Platte in [0, G), Ordnung in [1, K], Gewicht
// endlich und ≥ 0, ε endlich und > 0. Wirft std::invalid_argument.
void check_ghost_penalty(const GhostPenalty& penalty, const BatchTopology& topology);

// Residuen der Variante v (out.size() == penalty.size()), in der Reihenfolge der Terme, aus
// ghost_coeff[v][g][k − 1]. Wirft std::invalid_argument (Länge), std::out_of_range (v, Indizes).
void ghost_penalty_residuals(const GhostPenalty& penalty, const SystemBatchHost& batch, int v,
                             std::span<double> out);

// Strafbeitrag Σ r² der Terme einer Platte (alle Platten bei plate < 0) in Variante v,
// aufsteigend in der Reihenfolge der Terme.
double ghost_penalty_merit(const GhostPenalty& penalty, const SystemBatchHost& batch, int v,
                           int plate = -1);

// Zusatzresiduen für den Residuenvektor (count = size(), threadsicher), kombinierbar mit
// make_paraxial_residuals über concat_extra_residuals (extra_residuals.hpp).
ExtraResiduals make_ghost_penalty_residuals(GhostPenalty penalty);

// Gradient ∂F/∂a_k der Wellenfrontmerit (Wellen²) bei a = 0, den ein Restfehler mit RMS
// `rms_waves` (in Wellen bei λ_ref) erzeugt, dessen Pupillenverlauf genau dem Modus
// P_k(2u − 1) der Geisterplatte entspricht (Strahlbündel füllt den Normierungsradius,
// gleichmäßig gewichtete Pupille): 2 σ (n_G − n_m) / (λ_ref √(2k + 1)) [Wellen²/mm].
// Ein Koeffizient wird aktiv, sobald dieser Gradient μ · w_k übersteigt; mit
// μ · w_k = ghost_activation_gradient(σ_Ziel, …) entspricht die Schwelle einem
// Restfehler σ_Ziel pro Modus, z. B. σ_Ziel = 1/14 Welle (Beugungsbegrenzung, Maréchal;
// docs/optimierung.md, Abschnitt 3).
double ghost_activation_gradient(double rms_waves, double delta_n, double reference_nm, int order);

} // namespace mko
