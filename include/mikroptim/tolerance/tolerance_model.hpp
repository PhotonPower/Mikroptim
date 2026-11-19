#pragma once

// Monte-Carlo-Tolerierung (M8-T4, Host): Stichproben eines Toleranzsatzes (Abschnitt
// `tolerances` der Systemdatei, mikroptim/model/tolerances.hpp) und ihre Anwendung auf die
// Varianten eines nicht koaxialen SystemBatch. Zufallszahlen portabel und bitgleich auf allen
// Plattformen (wie der Variantengenerator, M3-T1); jede Stichprobe hängt nur vom Startwert und
// ihrem Index ab. Verfahren und Formeln: docs/optimierung.md, Abschnitt „Monte-Carlo-Tolerierung“.

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/model/tolerances.hpp"

#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace mko {

// Zufallsgröße ξ ∈ [−1, 1] aus dem Generator `engine`, nur mit exakten bzw. korrekt gerundeten
// Operationen (unit_uniform, Additionen, eine Division):
//   Uniform: ξ = 2u − 1 (eine Ziehung).
//   Normal:  z = Σ_{i=1}^{12} u_i − 6 (Irwin-Hall, Mittel 0, Varianz 1, Näherung der
//            Standardnormalverteilung), neu gezogen, solange |z| > c; ξ = z/c. Damit ist der
//            Wert δ·ξ normalverteilt mit σ = δ/c und auf ±δ abgeschnitten.
// Verlangt 0,5 ≤ c ≤ 6 bei Normal (sonst std::invalid_argument).
double tolerance_deviate(std::mt19937_64& engine, ToleranceDistribution distribution,
                         double cutoff);

// Lage einer Fläche relativ zu ihrer nominellen (koaxialen) Lage: Verschiebung des Scheitels
// (Δx, Δy, δz) [mm] und Verkippung (α, β) [rad] wie in ADR-0011 (R = R_y(β) R_x(α) um den
// Scheitel). δz ist kein Parameter der Fläche, sondern ändert die Dicken (vor der Fläche +δz,
// danach −δz).
struct SurfacePlacement {
    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;
    double alpha = 0.0;
    double beta = 0.0;

    friend bool operator==(const SurfacePlacement&, const SurfacePlacement&) = default;
};

// Starrkörperbewegung eines Elements (ADR-0011, Punkt 2): Drehung R = R_y(β) R_x(α) um den
// Punkt P = (0, 0, pivot_z) auf der Achse, danach Verschiebung (dx, dy, 0):
// x → P + R (x − P) + (dx, dy, 0).
struct ElementMotion {
    double dx = 0.0;
    double dy = 0.0;
    double alpha = 0.0;
    double beta = 0.0;
    double pivot_z = 0.0;
};

// Lage der Fläche mit nominellem Scheitel (0, 0, z_s) nach der Bewegung `motion`, ausgehend von
// ihrer bisherigen Lage `p` (Scheitel V = (p.dx, p.dy, z_s + p.dz), Drehung R_p): Scheitel
// P + R(V − P) + Δ, Drehung R · R_p, zerlegt in R_y(β) R_x(α) R_z(γ); die Drehung R_z(γ) um die
// eigene Achse ist bei Kugel- und Planflächen mit kreisförmigem Rand wirkungslos und entfällt
// (α = atan2(−n_y, √(n_x² + n_z²)), β = atan2(n_x, n_z) mit der Flächennormalen n = R R_p e_z).
// Ohne Drehung (α = β = 0) wird nur verschoben: dx, dy addiert, alles übrige bitgleich.
SurfacePlacement move_surface(const SurfacePlacement& p, double z_s, const ElementMotion& motion);

// Feste Liste von Toleranzen für Batches einer nicht koaxialen Topologie (make_non_coaxial bzw.
// topology_of eines dezentrierten Systems). Zufallsgrößen pro Stichprobe in der Reihenfolge
// von resolve_tolerances, je Toleranz draws() Ziehungen (Dezentrierung x, y; Verkippung α, β).
class ToleranceModel {
public:
    ToleranceModel() = default;
    // Wirft FormatError bei ungültigem Abschnitt `tolerances`, std::invalid_argument bei einer
    // koaxialen Topologie oder wenn Datei und Topologie nicht zusammenpassen.
    ToleranceModel(const SystemDescription& system, const BatchTopology& topology);

    std::span<const ResolvedTolerance> tolerances() const { return tolerances_; }
    // Zufallsgrößen pro Stichprobe (Summe von draws() über alle Toleranzen).
    int draws() const { return draws_; }
    // Erste Ziehung der Toleranz k im Vektor ξ einer Stichprobe.
    int draw_offset(int k) const;
    const BatchTopology& topology() const { return topology_; }

    // ξ der Stichprobe `index` (Länge draws()): eigener std::mt19937_64 mit Startwert
    // variant_seed(seed, index) (mikroptim/optim/variant_generator.hpp), Ziehungen in der
    // Reihenfolge der Toleranzen. Hängt nur von seed und index ab (nicht von Tranchen,
    // Batch-Größe, Zahl der GPUs). Wirft std::invalid_argument bei falscher Länge.
    void deviates(std::uint64_t seed, std::uint64_t index, std::span<double> xi) const;
    std::vector<double> deviates(std::uint64_t seed, std::uint64_t index) const;

    // Wendet die Störung ξ (Werte limit_k · ξ) auf Variante v an, die vorher die nominelle
    // Variante sein muss (docs/optimierung.md, „Monte-Carlo-Tolerierung“):
    //  Radius c' = c/(1 + ε) bzw. in Passen c' = c + Δc; Dicke t' = t + δ (bei einem Spalt mit
    //  Geisterplatten das letzte Teilstück); Brechzahl und Abbe pro Glas
    //  n'(λ) = n_ref + Δn + (n(λ) − n_ref)/(1 + e) (n_ref = paraxial_index bei λ_ref, ohne
    //  Abbe-Toleranz n(λ) + Δn), paraxial_index' = n_ref + Δn; Dezentrierung und Verkippung der
    //  Flächen zur Transformation der Variante (bei 0 direkt gesetzt, sonst über die Drehungen
    //  zusammengesetzt), danach die Elementbewegungen in Toleranzreihenfolge (move_surface) mit
    //  den Dickenänderungen ±δz. Wirft std::invalid_argument bei falscher Länge oder anderer
    //  Topologie, std::out_of_range bei ungültigem v.
    void apply(SystemBatchHost& batch, int v, std::span<const double> xi) const;

    // Schreibt `count` Stichproben first_index … first_index + count − 1 nach
    // out[first … first + count): Kopie der Variante `base_variant` von `nominal`, dann
    // apply(deviates(seed, first_index + j)). `out` hat dieselbe Topologie und genug Varianten.
    void write_samples(const SystemBatchHost& nominal, int base_variant, std::uint64_t seed,
                       std::uint64_t first_index, SystemBatchHost& out, int first, int count) const;

private:
    void check_batch(const SystemBatchHost& batch) const;

    std::vector<ResolvedTolerance> tolerances_;
    std::vector<int> offset_; // erste Ziehung pro Toleranz
    int draws_ = 0;
    int file_surfaces_ = 0;
    std::vector<int> table_of_file_; // Dateifläche → Tabellenfläche
    std::vector<double> vertex_z_;   // nominelle Scheitellage pro Dateifläche
    BatchTopology topology_;
};

} // namespace mko
