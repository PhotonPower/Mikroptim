#pragma once

// Variantengenerator (Host, docs/optimierung.md, Abschnitt „Variantengenerator“): aus einem
// Basisvektor x₀ und einer Vorlagenvariante Batches mit (a) zufällig gestörten Varianten
// (deterministisch und plattformunabhängig) oder (b) dem Muster der finiten Differenzen.
// Alle Varianten sind Kopien der Vorlage, nur die Variablen unterscheiden sich.

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/optim/fd_formula.hpp"
#include "mikroptim/optim/variable_set.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace mko {

// ---- Zufallszahlen ----------------------------------------------------------------------
// Plattformunabhängig: std::mt19937_64 ist im Standard bitgenau festgelegt, die
// Verteilungen der Standardbibliothek nicht. Daher eigene Umwandlungen ohne transzendente
// Funktionen (deren Rundung je nach Bibliothek um ein ulp abweichen kann).

// SplitMix64-Mischfunktion (Steele, Lea, Flood 2014): Startwert des Generators der Variante
// j ist splitmix64(seed + j·0x9E3779B97F4A7C15) (modulo 2⁶⁴).
std::uint64_t splitmix64(std::uint64_t x);
// Startwert des Generators für Variante j (Zählung innerhalb eines Aufrufs).
std::uint64_t variant_seed(std::uint64_t seed, std::uint64_t j);
// Gleichverteilt in [0, 1): (bits >> 11) · 2⁻⁵³ (exakt, 53 Bit).
double unit_uniform(std::uint64_t bits);

enum class Distribution {
    Uniform,   // ξ = 2u − 1, gleichverteilt in [−1, 1)
    IrwinHall, // ξ = Σ_{i=1}^{12} u_i − 6: Mittel 0, Varianz 1, begrenzt auf (−6, 6),
               // nahe an der Standardnormalverteilung; nur Additionen, daher bitgleich
};

// Störungsgröße s = absolute + relative·|x₀| pro Variablentyp.
struct PerturbationScale {
    double relative = 0.0;
    double absolute = 0.0;
};

struct PerturbationOptions {
    PerturbationScale curvature{0.01, 0.0};
    PerturbationScale thickness{0.01, 0.0};
    PerturbationScale working_distance{0.01, 0.0};
    Distribution distribution = Distribution::Uniform;
    std::uint64_t seed = 0;
    bool first_is_base = false; // Variante 0 ist x₀ selbst (ungestört)

    const PerturbationScale& scale(VariableType type) const;
};

// Vektor der Variante j: y_k = clamp(x₀_k + s_k·ξ_{j,k}) mit s_k = fma(relative, |x₀_k|,
// absolute), y = fma(s, ξ, x₀) und Projektion auf die Schranken. ξ_{j,k} kommt aus einem eigenen
// std::mt19937_64 pro Variante (Startwert variant_seed(seed, j)), in der Reihenfolge k = 0 …
// P−1 (IrwinHall: 12 Ziehungen pro k). Variante j hängt also nicht von der Zahl der übrigen
// Varianten ab. Bei first_is_base ist Variante 0 = x₀ (und Variante j ≥ 1 wie oben mit j).
// Wirft std::invalid_argument bei falscher Länge, nicht endlichem x₀, x₀ außerhalb der
// Schranken oder negativen/nicht endlichen Störungsgrößen.
std::vector<double> perturbed_vector(const VariableSet& set, std::span<const double> x0,
                                     const PerturbationOptions& options, std::uint64_t j);

// Schreibt `count` gestörte Varianten nach out[first … first + count): Kopie der Variante
// `base_variant` von `base`, dann apply(perturbed_vector(j)) für j = 0 … count − 1. `out` muss
// dieselbe Topologie haben (auch dasselbe Objekt wie `base` ist erlaubt) und genug Varianten.
void write_perturbed_variants(const VariableSet& set, const SystemBatchHost& base, int base_variant,
                              std::span<const double> x0, const PerturbationOptions& options,
                              SystemBatchHost& out, int first, int count);

// Neuer Batch mit `variants` gestörten Varianten (gemeinsame Felder aus `base`).
SystemBatchHost make_perturbed_batch(const VariableSet& set, const SystemBatchHost& base,
                                     int base_variant, std::span<const double> x0, int variants,
                                     const PerturbationOptions& options);

// ---- Finite Differenzen ------------------------------------------------------------------

// ε_m^{1/3} = 2^{−52/3} (korrekt gerundet, keine Laufzeitberechnung mit cbrt).
inline constexpr double kFdStepFactor = 6.0554544523933395e-06;

// Nominelle Schrittweite h = ε_m^{1/3} · max(|x|, x_typ) (docs/optimierung.md, Abschnitt 4).
double fd_step(double x, double typical);

enum class FdScheme {
    Central, // 2P + 1 Varianten: 0 = x₀, 2k + 1 = x₀ + d⁺_k e_k, 2k + 2 = x₀ + d⁻_k e_k
    Forward, // P + 1 Varianten: 0 = x₀, k + 1 = x₀ + d⁺_k e_k
};

int fd_variant_count(FdScheme scheme, int variables);
// Variante mit dem Schritt d⁺_k bzw. (nur zentral) d⁻_k, relativ zum ersten Index des Musters.
int fd_plus_variant(FdScheme scheme, int k);
int fd_minus_variant(int k); // nur Central

// Tatsächlich verwendete Punkte und Schritte: Variante fd_plus_variant(k) hat x_k = x⁺_k
// (point_plus, exakt der geschriebene Wert), step_plus[k] = d⁺_k = fl(x⁺_k − x₀_k); bei
// Central entsprechend x⁻_k, d⁻_k (sonst leer). d ist exakt für |x₀| ≥ x_typ und für x₀ = 0
// (Sterbenz-Lemma), sonst auf ½ ulp genau. stencil[k] sagt, wie die Ableitung aus den Punkten
// geschätzt wird (fd_derivative in fd_formula.hpp, M3-T2): Bei Central und Truncated gilt
// d⁺ ≥ 0 ≥ d⁻, bei OneSided liegen beide Punkte auf derselben Seite (d⁺ ≈ σh, d⁻ ≈ 2σh;
// „plus“ und „minus“ sind dann nur die Namen der ersten und zweiten Variante).
struct FdPattern {
    FdScheme scheme = FdScheme::Central;
    std::vector<double> nominal;     // h_k
    std::vector<double> point_plus;  // x⁺_k
    std::vector<double> point_minus; // x⁻_k (nur Central)
    std::vector<double> step_plus;   // d⁺_k
    std::vector<double> step_minus;  // d⁻_k (nur Central)
    std::vector<FdStencil> stencil;  // Schablone pro Variable (Forward: immer Forward)

    friend bool operator==(const FdPattern&, const FdPattern&) = default;
};

// Schritte (x₀ muss zulässig sein; docs/optimierung.md, Abschnitt „Finite-Differenzen-Jacobi“):
//  Central: (1) x₀ − h ≥ lower und x₀ + h ≤ upper: x⁺ = x₀ + h, x⁻ = x₀ − h (FdStencil::Central).
//           (2) sonst σ = +1, wenn upper − x₀ ≥ x₀ − lower, sonst −1; liegt x₀ + 2σh in den
//           Schranken: x⁺ = x₀ + σh, x⁻ = x₀ + 2σh (FdStencil::OneSided, zweite Ordnung).
//           (3) sonst (Bereich kürzer als etwa 3h): x⁺ = min(x₀ + h, upper),
//           x⁻ = max(x₀ − h, lower) (FdStencil::Truncated, erste Ordnung); liegt x₀ auf einer
//           Schranke, ist der Schritt dorthin 0. d⁺ − d⁻ > 0 gilt bei (1) und (3) immer, weil
//           lower < upper; bei (2) sind d⁺ und d⁻ ungleich null und verschieden.
//  Forward: x⁺ = x₀ + h, falls ≤ upper; sonst x₀ − h, falls ≥ lower (Rückwärtsschritt, d⁺ < 0);
//           sonst die Schranke mit dem größeren Abstand.
FdPattern fd_pattern(const VariableSet& set, std::span<const double> x0, FdScheme scheme);

// Schreibt das Muster nach out[first … first + fd_variant_count): jede Variante eine Kopie der
// Vorlage mit x₀ bzw. x₀ + d e_k. Rückgabe: die verwendeten Schritte.
FdPattern write_fd_variants(const VariableSet& set, const SystemBatchHost& base, int base_variant,
                            std::span<const double> x0, FdScheme scheme, SystemBatchHost& out,
                            int first);

struct FdBatch {
    SystemBatchHost batch;
    FdPattern pattern;
};
FdBatch make_fd_batch(const VariableSet& set, const SystemBatchHost& base, int base_variant,
                      std::span<const double> x0, FdScheme scheme = FdScheme::Central);

} // namespace mko
