#pragma once

// Testsysteme und zufällig gestörte Varianten für die Residuen- und Merit-Tests (CPU und GPU,
// M2-T2, M2-T3), Umrechnung zwischen Strahlresiduen und OPD (M4-T1).

#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/merit/ray_residual.hpp"
#include "mikroptim/pupil/quadrature.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace mko::test {

inline const std::string kResidualSystems = MKO_SOURCE_DIR "/data/systems/";

inline LoadedSystem load_residual_system(const std::string& file) {
    return load_system(kResidualSystems + file);
}

// Batch mit `variants` Varianten des Systems; Variante 0 bleibt ungestört, alle anderen werden
// mit festem Seed gestört: Krümmungen der Sphären (±30 %), Dicken (±30 %, Arbeitsabstand
// ±10 %), Pupillenradius (Faktor 0,5 … 2), Brechzahlen der Gläser (±0,05, gleich für alle
// Wellenlängen), Feldwinkel und Feldziele der Feldpunkte (±50 % bzw. ±10 %). Etwa jede
// siebte Variante ist "extrem" (Pupille ×2 … ×5, Krümmungen ×1 … ×4), damit Strahlen versagen
// (Fläche verfehlt, Totalreflexion, rückwärts). Achsfelder bleiben auf der Achse.
inline SystemBatchHost perturbed_batch(const LoadedSystem& system, int variants,
                                       std::uint64_t seed) {
    SystemBatchHost b = make_system_batch(system, variants);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const int S = b.surfaces();
    for (int v = 1; v < variants; ++v) {
        const bool extreme = unit(rng) < 0.15;
        for (int s = 1; s < S; ++s) {
            if (b.type(s) == SurfaceKind::Sphere) {
                b.curvature(v, s) *= 1.0 + 0.3 * u(rng);
                if (extreme) {
                    b.curvature(v, s) *= 1.0 + 1.5 * (1.0 + u(rng));
                }
            }
        }
        for (int s = 0; s < S - 1; ++s) {
            b.thickness(v, s) *= 1.0 + 0.3 * u(rng);
        }
        b.thickness(v, S - 1) *= 1.0 + 0.1 * u(rng);
        b.pupil_radius(v) *= std::exp(0.693 * u(rng));
        if (extreme) {
            b.pupil_radius(v) *= 3.5 + 1.5 * u(rng);
        }
        for (int s = 0; s < S; ++s) {
            if (b.index(v, s, 0) > 1.0) {
                const double dn = 0.05 * u(rng);
                for (int l = 0; l < b.wavelengths(); ++l) {
                    b.index(v, s, l) += dn;
                }
            }
        }
        for (int f = 0; f < b.fields(); ++f) {
            if (b.field_angle(v, f) != 0.0) {
                b.field_angle(v, f) *= 1.0 + 0.5 * u(rng);
                b.field_target(v, f) *= 1.0 + 0.1 * u(rng);
            }
        }
    }
    return b;
}

// Stapel aus S Flächen (Standard kMaxSurfaces): Blende, (S − 2)/2 schwach gekrümmte Linsen
// (Gläser n = 1,5 und 1,7 abwechselnd), letzte Fläche (Luft auf beiden Seiten) mit
// Arbeitsabstand; zwei Felder (Achse, Feldpunkt). S gerade, 4 <= S <= kMaxSurfaces.
inline SystemBatchHost many_surface_batch(int variants, std::uint64_t seed,
                                          int surfaces = kMaxSurfaces) {
    BatchTopology t;
    t.surfaces = surfaces;
    t.wavelengths = 2;
    t.fields = 2;
    t.type.assign(static_cast<std::size_t>(t.surfaces), SurfaceKind::Sphere);
    t.type[0] = SurfaceKind::Stop;
    t.pupil_mode = {PupilMode::Axis, PupilMode::Half};
    SystemBatchHost b(t, variants);
    b.wavelength_nm(0) = 486.13;
    b.wavelength_nm(1) = 656.27;
    b.wavelength_weight(0) = b.wavelength_weight(1) = 1.0;
    b.field_weight(0) = b.field_weight(1) = 1.0;
    b.reference_wavelength_nm = 587.56;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const int S = t.surfaces;
    for (int v = 0; v < variants; ++v) {
        const double p = v == 0 ? 0.0 : 1.0; // Variante 0 ungestört
        for (int s = 1; s < S; ++s) {
            const bool glass = s % 2 == 1 && s < S - 1; // Medium nach Fläche s
            b.curvature(v, s) = (glass ? 0.0008 : -0.0006) * (1.0 + 0.2 * p * u(rng));
            b.thickness(v, s) = (glass ? 0.8 : 0.4) * (1.0 + 0.2 * p * u(rng));
            const double n = glass ? (s % 4 == 1 ? 1.5 : 1.7) : 1.0;
            for (int l = 0; l < 2; ++l) {
                b.index(v, s, l) = glass ? n + 0.01 * (1 - l) : 1.0;
            }
        }
        for (int l = 0; l < 2; ++l) {
            b.index(v, 0, l) = 1.0;
        }
        b.thickness(v, 0) = 2.0;
        b.thickness(v, S - 1) = 40.0 * (1.0 + 0.1 * p * u(rng));
        b.pupil_radius(v) = 3.0 * (1.0 + 0.3 * p * u(rng));
        b.field_angle(v, 1) = 0.02 * (1.0 + 0.3 * p * u(rng));
        b.field_target(v, 1) = 0.8 * (1.0 + 0.1 * p * u(rng));
    }
    b.validate_values();
    return b;
}

// Umrechnung Strahlresiduum ↔ OPD (M4-T1): r_k = u_k · OPD_k mit u_k = s_{f,l} · √(W_i V_j)
// [1/mm] (residual_scale und residual_weight), Ablage des Layouts (Länge rays_per_variant;
// gleich für alle Varianten, weil die Gewichte gemeinsam sind). Eine Schranke δ [mm] für die
// OPD eines Strahls ist damit die Schranke u_k · δ für sein Residuum (die Rundung des Produkts,
// ≤ 2ε |r|, liegt weit unter den relativen Anteilen der Toleranzen).
inline std::vector<double> residual_units(const SystemBatchHost& batch,
                                          const ResidualLayout& layout) {
    const SystemBatchView<double> view = batch.view();
    std::vector<double> u(static_cast<std::size_t>(layout.rays_per_variant()));
    for (int f = 0; f < layout.fields(); ++f) {
        const PupilQuadrature<double>& q = layout.quadrature(f);
        for (int l = 0; l < layout.wavelengths(); ++l) {
            const double s = residual_scale(view, f, l);
            const std::int64_t o = layout.offset(f, l);
            for (int k = 0; k < layout.samples(f); ++k) {
                u[static_cast<std::size_t>(o + k)] = s * residual_weight(sample(q, k));
            }
        }
    }
    return u;
}

// Dasselbe für die Strafresiduen des weichen Versagens (M4-T2): p_k = v_k · π_k mit
// v_k = penalty_scale(f, l, w_fail) · √(W_i V_j) und der dimensionslosen Strafgröße π_k.
inline std::vector<double> penalty_units(const SystemBatchHost& batch,
                                         const ResidualLayout& layout) {
    const SystemBatchView<double> view = batch.view();
    std::vector<double> u(static_cast<std::size_t>(layout.rays_per_variant()));
    for (int f = 0; f < layout.fields(); ++f) {
        const PupilQuadrature<double>& q = layout.quadrature(f);
        for (int l = 0; l < layout.wavelengths(); ++l) {
            const double s = penalty_scale(view, f, l, layout.settings().fail_weight);
            const std::int64_t o = layout.offset(f, l);
            for (int k = 0; k < layout.samples(f); ++k) {
                u[static_cast<std::size_t>(o + k)] = s * residual_weight(sample(q, k));
            }
        }
    }
    return u;
}

// Schranke pro Strahlresiduum aus einer Toleranz der Form max(abs_mm, rel · |OPD|)
// (z. B. kCpuGpuFp64, kCpuGpuFp64Relative): max(u · abs_mm, rel · |r|), weil r = u · OPD.
inline double residual_bound(double unit, double r, double abs_mm, double rel) {
    return std::max(unit * abs_mm, rel * std::abs(r));
}

// √(Σ_(f,l) s_(f,l)²) [1/mm]: Faktor, mit dem eine Schranke δ [mm], die für die OPD aller
// Strahlen gilt, die Norm ‖δr‖₂ einer Variante begrenzt (Σ_k W_k <= 1 pro Tupel).
inline double residual_scale_norm(const SystemBatchHost& batch, const ResidualLayout& layout) {
    const SystemBatchView<double> view = batch.view();
    double sum = 0.0;
    for (int f = 0; f < layout.fields(); ++f) {
        for (int l = 0; l < layout.wavelengths(); ++l) {
            const double s = residual_scale(view, f, l);
            sum += s * s;
        }
    }
    return std::sqrt(sum);
}

// System `file` mit anderen Feld- und Spektralgewichten (M4-T1); leere Liste: wie in der Datei.
inline LoadedSystem weighted_system(const std::string& file, std::vector<double> field_weights,
                                    std::vector<double> spectrum_weights) {
    SystemDescription d = load_residual_system(file).system;
    if (!field_weights.empty()) {
        d.source.field_weights = std::move(field_weights);
    }
    if (!spectrum_weights.empty()) {
        d.spectrum.weights = std::move(spectrum_weights);
    }
    return prepare_system(std::move(d), std::filesystem::path(kResidualSystems));
}

// Testfälle der Abnahme M4-T1: gleiche und ungleiche Gewichte, ausgeschaltete Felder und
// Wellenlängen (Gewicht 0), Geisterplatten, Punktquelle neben der Achse (ohne versagte Strahlen).
struct WeightedCase {
    std::string name;
    LoadedSystem system;
};

inline std::vector<WeightedCase> weighted_cases() {
    SystemDescription point = load_residual_system("einzellinse_bk7.json").system;
    point.source.type = SourceType::Point;
    point.source.field_angles_deg.clear();
    point.source.field_weights.clear();
    point.source.numerical_aperture = 0.05;
    point.source.point = {0.0, 2.0, -300.0};
    point.surfaces.back().thickness = 120.0;
    point.spectrum.weights = {0.2, 1.0, 0.6};
    std::vector<WeightedCase> out;
    out.push_back(
        {"Mehrlinser, gleiche Gewichte", load_residual_system("testsystem_mehrlinser.json")});
    out.push_back(
        {"Mehrlinser, ungleiche Gewichte",
         weighted_system("testsystem_mehrlinser.json", {1.0, 0.7, 0.3}, {0.5, 1.0, 0.8, 0.3})});
    out.push_back(
        {"Mehrlinser, Feld 1, Wellenlängen 0 und 3 aus",
         weighted_system("testsystem_mehrlinser.json", {2.0, 0.0, 1.0}, {0.0, 1.0, 1.0, 0.0})});
    out.push_back({"Einzellinse, ungleiche Gewichte",
                   weighted_system("einzellinse_bk7.json", {3.0, 2.0, 1.0}, {1.0, 4.0, 1.0})});
    out.push_back(
        {"Einzellinse mit Geisterplatten", load_residual_system("einzellinse_geisterplatte.json")});
    out.push_back({"Punktquelle", prepare_system(point, std::filesystem::path(kResidualSystems))});
    return out;
}

// Erwartete Merit nach M4-T1 aus den RMS-Werten der Tupel in Wellen bei λ_ref (Ablage
// [f · L + l] wie WavefrontResult::maps), unabhängig von residual_scale direkt aus den Gewichten
// der Beschreibung gerechnet: Σ_f Σ_λ ŵ_f ŵ_λ RMS_fλ² mit ŵ = w / Σ w (Punktquelle: ein Feld mit
// Gewicht 1). Tupel mit Gewicht 0 tragen nicht bei (auch nicht mit RMS NaN).
inline double weighted_rms_merit(const SystemDescription& d, const std::vector<double>& rms_waves) {
    const std::vector<double> wf =
        d.source.type == SourceType::Point ? std::vector<double>{1.0} : d.source.field_weights;
    const std::vector<double>& wl = d.spectrum.weights;
    double sum_f = 0.0;
    double sum_l = 0.0;
    for (const double w : wf) {
        sum_f += w;
    }
    for (const double w : wl) {
        sum_l += w;
    }
    double merit = 0.0;
    for (std::size_t f = 0; f < wf.size(); ++f) {
        for (std::size_t l = 0; l < wl.size(); ++l) {
            const double w = (wf[f] / sum_f) * (wl[l] / sum_l);
            const double rms = rms_waves[f * wl.size() + l];
            merit += w > 0.0 ? w * rms * rms : 0.0;
        }
    }
    return merit;
}

} // namespace mko::test
