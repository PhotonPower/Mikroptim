#pragma once

// Aberrationsbericht pro Fläche (M10-T1, M10-T1b; docs/entwurfssteuerung.md, Abschnitt 3.2):
// paraxiale Rand- und Hauptstrahlen, Seidel-Beiträge S_I … S_V, chromatische Beiträge C_I, C_II
// und Sphärochromasie 3. Ordnung pro Dateifläche; aus realen Randstrahlen die sphärische
// Aberration aller Ordnungen pro Fläche (Rand und Zone) und die Verletzung der Sinusbedingung.
// Formeln und Vorzeichen: docs/numerik.md, Abschnitt 14. Host-Code, Bibliothek Mikroptim::paraxial.

#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/model/system.hpp"

#include <array>
#include <optional>
#include <vector>

namespace mko {

// Beiträge einer Dateifläche. Strahldaten bei λ_ref; u, ū vor (`_before`) und nach (`_after`)
// der Brechung, y, ȳ an der Fläche. Seidel-Summen in mm (Welford), Wellenfront siehe
// SeidelWaves.
struct SurfaceAberration {
    int surface = 0; // Dateifläche
    double curvature = 0.0;
    double n_before = 1.0; // bei λ_ref
    double n_after = 1.0;
    double y = 0.0, u_before = 0.0, u_after = 0.0;          // Randstrahl
    double ybar = 0.0, ubar_before = 0.0, ubar_after = 0.0; // Hauptstrahl
    double A = 0.0;                                         // n i = n (u + y c)
    double Abar = 0.0;                                      // n ī = n (ū + ȳ c)
    std::array<double, 5> S{};                              // S_I … S_V
    std::array<double, 2> C{};                              // C_I, C_II
    double spherochromatism = 0.0;  // S_I(λ_kurz) − S_I(λ_lang), je mit eigenen Strahlen
    bool distortion_defined = true; // false, wenn A = 0 (S_V der Fläche nicht definiert)
    // Aus realen Randstrahlen bei λ_ref (M10-T1b; NaN ohne reale Strahlen): sphärische
    // Aberration aller Ordnungen, normiert wie S_I (gleich S_I bei kleiner Apertur), am Rand
    // (ρ = 1) und in der Zone (ρ = kZoneRho, dort ≈ S_I ρ⁴); Anteil höherer Ordnung am Rand.
    double full_order_spherical = 0.0;
    double higher_order_spherical = 0.0;
    double full_order_spherical_zone = 0.0;
};

// Pupillenzone der Zonenwerte (ρ² = 1/2).
inline constexpr double kZoneRho = 0.70710678118654752;

// Seidel-Summen als Wellenfrontkoeffizienten in Wellen bei λ_ref (Bezug: volles Feld des
// Hauptstrahls, Pupillenrand): W040 = S_I/8, W131 = S_II/2, W222 = S_III/2, W220 = S_IV/4,
// W311 = S_V/2, ΔW020 = C_I/2, ΔW111 = C_II, Sphärochromasie ΔW040 = (S_I(λ_k) − S_I(λ_l))/8.
struct SeidelWaves {
    double W040 = 0.0, W131 = 0.0, W222 = 0.0, W220 = 0.0, W311 = 0.0;
    double dW020 = 0.0, dW111 = 0.0, dW040 = 0.0;
    // Vollordnung (M10-T1b), gleiche Normierung /8: Rand, Anteil höherer Ordnung, Zone
    double W040_full = 0.0, W040_higher = 0.0, W040_zone = 0.0;
};

struct AberrationOptions {
    // Feldwinkel des Hauptstrahls [Grad]; Standard: größter Feldwinkel der Quelle.
    std::optional<double> field_angle_deg;
    // Wellenlängen [nm]; Standard: spectrum.reference_nm, kürzeste und längste Wellenlänge.
    std::optional<double> reference_nm;
    std::optional<double> short_nm;
    std::optional<double> long_nm;
    // Reale Randstrahlen für Vollordnung, Zone und OSC (M10-T1b).
    bool real_rays = true;
};

struct AberrationReport {
    double reference_nm = 0.0;
    double short_nm = 0.0;
    double long_nm = 0.0;
    double field_angle_deg = 0.0;
    double stop_semi_diameter = 0.0;
    double lagrange = 0.0; // H = n (ū y − u ȳ), an Fläche 0 vor der Brechung
    std::vector<SurfaceAberration> surfaces;
    std::array<double, 5> S{};
    std::array<double, 2> C{};
    double spherochromatism = 0.0;
    bool distortion_defined = true;
    // M10-T1b: Summen der Vollordnung und Verletzung der Sinusbedingung (OSC, Conrady) am Rand
    // und in der Zone; real_rays_valid = false, wenn ein realer Randstrahl versagt (Werte NaN).
    double full_order_spherical = 0.0;
    double higher_order_spherical = 0.0;
    double full_order_spherical_zone = 0.0;
    double osc = 0.0;
    double osc_zone = 0.0;
    bool real_rays_valid = false;
    // Geisterplatten werden nicht berücksichtigt (Bericht über die Dateiflächen); Zahl der
    // Platten mit Koeffizienten ungleich 0, deren Wirkung damit fehlt.
    int ghost_plates_ignored = 0;

    // Summen als Wellenfrontkoeffizienten in Wellen bei λ_ref.
    SeidelWaves waves() const;
    // Beiträge einer Fläche als Wellenfrontkoeffizienten in Wellen bei λ_ref.
    SeidelWaves waves(const SurfaceAberration& s) const;
};

// Bericht nur für Systeme mit kollimierter Quelle (Blende = Fläche 0, Objekt im Unendlichen).
// Wirft std::invalid_argument bei Punktquelle oder ungültigen Optionen und GlassError, wenn
// eine Brechzahl nicht bestimmbar ist.
AberrationReport compute_aberrations(const SystemDescription& system, const GlassCatalog& catalog,
                                     const AberrationOptions& options = {});

} // namespace mko
