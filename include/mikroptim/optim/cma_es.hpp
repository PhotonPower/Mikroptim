#pragma once

// CMA-ES (Covariance Matrix Adaptation Evolution Strategy, Hansen 2016, „The CMA Evolution
// Strategy: A Tutorial“) für die memetische Suche (M11-T7, Host). Eigene Umsetzung ohne externe
// Bibliothek (AGENTS.md §9). Deterministisch und plattformunabhängig bitgleich: Zufallszahlen aus
// std::mt19937_64 mit Irwin-Hall-Normalnäherung (wie der Variantengenerator), Eigenzerlegung
// mit dem zyklischen Jacobi-Verfahren, exp und log durch eigene Reihen nur aus korrekt gerundeten
// Grundoperationen (+, −, ·, /, √, frexp/ldexp), alle Summen in fester Reihenfolge.
// Formeln: docs/optimierung.md, Abschnitt 15.

#include <cstdint>
#include <span>
#include <vector>

namespace mko {

// e^x nur aus Grundoperationen (Reduktion x = k ln 2 + r, |r| ≤ ln 2 / 2, Taylor-Reihe bis r²⁰,
// dann ldexp): auf jeder Plattform bitgleich, relativer Fehler ≤ 4·10⁻¹⁶. +∞ für x > 709,78,
// 0 für x < −745, NaN für NaN.
double portable_exp(double x);
// ln x nur aus Grundoperationen (x = m·2^e mit m ∈ [√½, √2), ln m = 2 artanh((m−1)/(m+1)) als
// Reihe bis s³¹): bitgleich auf jeder Plattform, relativer Fehler ≤ 4·10⁻¹⁶ (absolut nahe 1).
// −∞ für 0, NaN für x < 0 und NaN, +∞ für +∞.
double portable_log(double x);

// Zyklisches Jacobi-Verfahren für eine symmetrische n × n-Matrix `a` (spaltenweise, wird
// überschrieben): Eigenwerte nach `values` (n, ohne Sortierung), Eigenvektoren spaltenweise nach
// `vectors` (n × n, orthonormal). Rotationen in der festen Reihenfolge p < q zeilenweise; Abbruch,
// wenn alle Nebendiagonalelemente 0 sind oder nach 64 Durchgängen. Wirft std::invalid_argument bei
// falschen Längen.
void symmetric_eigen(int n, std::span<double> a, std::span<double> values,
                     std::span<double> vectors);

// Strategieparameter (Hansen 2016, Tabelle 1, nur positive Gewichte).
struct CmaParameters {
    int dimension = 0; // n
    int lambda = 0;    // Populationsgröße λ
    int mu = 0;        // Eltern μ = ⌊λ/2⌋
    // w_i = (ln((λ+1)/2) − ln i) / Σ_j (…), i = 1 … μ (Summe 1, absteigend)
    std::vector<double> weights;
    double mu_eff = 0.0;  // 1 / Σ w_i²
    double c_sigma = 0.0; // (μ_eff + 2)/(n + μ_eff + 5)
    double d_sigma = 0.0; // 1 + 2 max(0, √((μ_eff − 1)/(n + 1)) − 1) + c_σ
    double c_c = 0.0;     // (4 + μ_eff/n)/(n + 4 + 2 μ_eff/n)
    double c_1 = 0.0;     // 2/((n + 1,3)² + μ_eff)
    double c_mu = 0.0;    // min(1 − c_1, 2 (μ_eff − 2 + 1/μ_eff)/((n + 2)² + μ_eff))
    double chi_n = 0.0;   // E‖N(0, I)‖ ≈ √n (1 − 1/(4n) + 1/(21 n²))
    // Obergrenze der Mahalanobis-Länge eingespeister Punkte (Hansen 2011, „Injecting external
    // solutions into CMA-ES“): √n + 2n/(n + 2).
    double clip_norm = 0.0;
};

// Standardgröße der Population: λ = 4 + ⌊3 ln n⌋ (n ≥ 1).
int cma_default_lambda(int dimension);

// Parameter für n ≥ 1 und λ ≥ 2 (λ = 0: cma_default_lambda). Wirft std::invalid_argument.
CmaParameters cma_parameters(int dimension, int lambda = 0);

// Zustand der Strategie: Mittelwert m, Schrittweite σ, Kovarianz C = B diag(D²) Bᵀ, Pfade p_σ,
// p_c, Generationenzähler g.
class CmaEs {
public:
    // Start m = mean, σ = sigma > 0, C = I. Wirft std::invalid_argument bei leerem oder nicht
    // endlichem Mittelwert, σ ≤ 0 oder ungültigem λ.
    CmaEs(std::span<const double> mean, double sigma, int lambda = 0);

    const CmaParameters& parameters() const { return params_; }
    int dimension() const { return params_.dimension; }
    int lambda() const { return params_.lambda; }
    int generation() const { return generation_; }
    double sigma() const { return sigma_; }
    std::span<const double> mean() const { return mean_; }
    std::span<const double> covariance() const { return c_; } // n × n spaltenweise
    std::span<const double> path_sigma() const { return p_sigma_; }
    std::span<const double> path_c() const { return p_c_; }

    // Probe i der aktuellen Generation: x = m + σ B (D ∘ z), z_k = Σ_{j=1}^{12} u_j − 6
    // (Irwin-Hall) aus einem eigenen std::mt19937_64 mit Startwert variant_seed(variant_seed(seed,
    // g), i), Ziehungen in der Reihenfolge k = 0 … n − 1. Hängt nur von seed, g, i und dem Zustand
    // ab.
    void sample(std::uint64_t seed, std::uint64_t i, std::span<double> out) const;

    // Mahalanobis-Länge ‖C^{−1/2} v‖ eines Vektors v (Länge n).
    double mahalanobis_norm(std::span<const double> v) const;

    // Eine Generation fortschreiben: `points` enthält λ Punkte hintereinander (beliebige Punkte,
    // z. B. lokal verfeinerte), `ranking` die Indizes der Punkte vom besten an (mindestens μ,
    // jeder Index höchstens einmal). Schritte y_i = (x_i − m)/σ werden auf die Mahalanobis-Länge
    // clip_norm begrenzt; Rekombination y_w = Σ w_i y_{i:λ}, m ← m + σ y_w; Pfade, Rang-1- und
    // Rang-μ-Anpassung von C und Schrittweite σ ← σ exp(min(1, (c_σ/d_σ)(‖p_σ‖/χ_n − 1))) nach
    // docs/optimierung.md, Abschnitt 15; danach Eigenzerlegung von C. `sigma_points` (leer oder
    // λ Punkte in derselben Reihenfolge, z. B. die gezogenen Starts vor der Verfeinerung) ersetzt
    // `points` nur im Pfad p_σ der Schrittweite. Wirft std::invalid_argument bei falschen Längen,
    // nicht endlichen Punkten oder ungültigem Ranking.
    void update(std::span<const double> points, std::span<const int> ranking,
                std::span<const double> sigma_points = {});

private:
    void decompose();

    CmaParameters params_;
    int generation_ = 0;
    double sigma_ = 0.0;
    double sigma_decay_ = 1.0; // (1 − c_σ)^{2g} für h_σ
    std::vector<double> mean_;
    std::vector<double> c_; // C
    std::vector<double> b_; // Eigenvektoren (spaltenweise)
    std::vector<double> d_; // √Eigenwerte
    std::vector<double> p_sigma_;
    std::vector<double> p_c_;
};

} // namespace mko
