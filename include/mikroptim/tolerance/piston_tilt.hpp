#pragma once

// Abzug von Kolben und Kippung über die Pupillenquadratur eines Tupels (Host). Eine Definition für
// die Bewertung der Tolerierung (tuple_quality, M8-T4), die Empfindlichkeiten (M10-T3) und den
// Kippungsabzug der toleranzbewussten Merit (M11-T6b): gewichtete kleinste Quadrate
// v_k ≈ c₀ + t_x x_k + t_y y_k über die Stützstellen k der Menge K (x = ρ sin φ, y = ρ cos φ,
// Gewichte W_k der Quadratur), 3 × 3-Normalgleichungen mit Summen in aufsteigender Reihenfolge k,
// Cholesky; ist ein Pivot ≤ 10⁻¹²·Σ_K W (Achsfall, weniger als drei Strahlen), nur der Kolben.
// docs/numerik.md, Abschnitt 16; docs/optimierung.md, Abschnitt 14.

#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/pupil/quadrature.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace mko {

// Normalgleichungen der Anpassung für eine Quadratur und eine Menge K von Stützstellen.
class PistonTiltFit {
public:
    PistonTiltFit() = default;
    // use[k] ≠ 0: Stützstelle k gehört zu K (sample_count(q) Einträge, sonst
    // std::invalid_argument).
    PistonTiltFit(const PupilQuadrature<double>& q, std::span<const std::uint8_t> use);

    int samples() const { return static_cast<int>(w_.size()); }
    double x(int k) const { return x_[static_cast<std::size_t>(k)]; }
    double y(int k) const { return y_[static_cast<std::size_t>(k)]; }
    double weight(int k) const { return w_[static_cast<std::size_t>(k)]; }
    bool used(int k) const { return use_[static_cast<std::size_t>(k)] != 0; }
    double total_weight() const { return w_all_; } // Σ W über alle Stützstellen
    double fit_weight() const { return a00_; }     // Σ_K W
    bool valid() const { return a00_ > 0.0; }      // mindestens eine Stützstelle in K
    bool tilt() const { return tilt_; }            // Kippung angepasst (sonst nur Kolben)

    // (c₀, t_x, t_y) zu den rechten Seiten b₀ = Σ_K W v, b₁ = Σ_K W x v, b₂ = Σ_K W y v (nur mit
    // valid(); ohne Kippung t_x = t_y = 0 und c₀ = b₀/Σ_K W).
    std::array<double, 3> solve(double b0, double b1, double b2) const;

private:
    std::vector<double> x_;
    std::vector<double> y_;
    std::vector<double> w_;
    std::vector<std::uint8_t> use_;
    double w_all_ = 0.0;
    double a00_ = 0.0;
    bool tilt_ = false;
    double l00_ = 0.0, l10_ = 0.0, l20_ = 0.0, l11_ = 0.0, l21_ = 0.0, l22_ = 0.0;
};

// Kippungsabzug der Strahlresiduen einer Variante (M11-T6b). Die Strahlresiduen sind
// r_k = s_{f,l} √W_k OPD_k (docs/numerik.md, Abschnitt 7) mit OPD nach Kolbenabzug über die
// Strahlen in_piston. Pro Tupel (f, l) mit K = {k : in_piston(Status_k)} und PistonTiltFit:
//   b = Σ_K √W_k (1, x_k, y_k) r_k,  (c₀, t_x, t_y) = solve(b),
//   r_k ← r_k − √W_k (c₀ + t_x x_k + t_y y_k)  (k ∈ K; sonst unverändert, dort ist r_k = 0).
// Das ist die orthogonale Projektion P auf das Komplement von span{√W (1, x, y)} über K, also
// P r = s √W e mit dem Rest e der Anpassung der OPD (tuple_quality mit derselben Menge K), und
// unabhängig von s. Linear in r: Auf die Spalten einer Jacobi-Matrix angewandt, ergibt sie die
// Ableitung der projizierten Residuen (Maske K fest). Strafresiduen und weitere Zeilen hinter den
// Strahlresiduen bleiben unverändert. Ohne Kippung (Achsfall, wenige Strahlen) ändert sich nichts
// (die Residuen sind bereits kolbenfrei).
class TiltProjection {
public:
    // `status`: rays_per_variant Einträge (TraceStatus). Wirft std::invalid_argument bei
    // falscher Länge.
    TiltProjection(const ResidualLayout& layout, std::span<const std::uint8_t> status);

    // r ← P r für die ersten rays_per_variant Einträge von `rows` (mindestens so viele, sonst
    // std::invalid_argument).
    void apply(std::span<double> rows) const;

private:
    struct Tuple {
        std::int64_t offset = 0;
        PistonTiltFit fit;
        std::vector<double> root; // √W_k
    };
    std::int64_t rays_ = 0;
    std::vector<Tuple> tuples_; // nur Tupel mit Kippung
};

} // namespace mko
