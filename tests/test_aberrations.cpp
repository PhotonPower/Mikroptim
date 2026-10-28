// Aberrationsbericht pro Fläche (M10-T1): Seidel-Beiträge, chromatische Beiträge und
// Sphärochromasie gegen unabhängige Orakel (getracte Wellenfront, Paraxialoptik).

#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/merit/wavefront.hpp"
#include "mikroptim/paraxial/aberrations.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "mikroptim/trace/trace.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using mko::test::kFullOrderSmallAperture;
using mko::test::kOscThirdOrder;
using mko::test::kSeidelChromaticFirstOrder;
using mko::test::kSeidelIdentity;
using mko::test::kSeidelSmallAperture;

namespace {

std::filesystem::path system_path(const std::string& name) {
    return std::filesystem::path(MKO_SOURCE_DIR) / "data" / "systems" / name;
}

// System mit verkleinerter Blende und einem Feldpunkt (0, θ) für die Wellenfront-Orakel.
mko::LoadedSystem scaled_system(const std::string& name, double stop_scale, double field_deg) {
    mko::LoadedSystem l = mko::load_system(system_path(name));
    l.system.surfaces[0].semi_diameter *= stop_scale;
    l.system.source.field_angles_deg = {0.0, field_deg};
    l.system.source.field_weights = {1.0, 1.0};
    return mko::prepare_system(l.system, system_path(name).parent_path());
}

// Kleinste Quadrate über wenige Basisfunktionen (Normalgleichungen, Gauß mit Pivotsuche).
std::vector<double> least_squares(const std::vector<std::vector<double>>& X,
                                  const std::vector<double>& y) {
    const std::size_t p = X.front().size();
    std::vector<double> A(p * p, 0.0);
    std::vector<double> b(p, 0.0);
    for (std::size_t i = 0; i < X.size(); ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            b[j] += X[i][j] * y[i];
            for (std::size_t k = 0; k < p; ++k) {
                A[j * p + k] += X[i][j] * X[i][k];
            }
        }
    }
    for (std::size_t c = 0; c < p; ++c) {
        std::size_t piv = c;
        for (std::size_t r = c + 1; r < p; ++r) {
            if (std::abs(A[r * p + c]) > std::abs(A[piv * p + c])) {
                piv = r;
            }
        }
        for (std::size_t k = 0; k < p; ++k) {
            std::swap(A[c * p + k], A[piv * p + k]);
        }
        std::swap(b[c], b[piv]);
        for (std::size_t r = 0; r < p; ++r) {
            if (r == c) {
                continue;
            }
            const double f = A[r * p + c] / A[c * p + c];
            for (std::size_t k = 0; k < p; ++k) {
                A[r * p + k] -= f * A[c * p + k];
            }
            b[r] -= f * b[c];
        }
    }
    std::vector<double> x(p);
    for (std::size_t c = 0; c < p; ++c) {
        x[c] = b[c] / A[c * p + c];
    }
    return x;
}

// Koeffizienten von ρ⁴, ρ³cos φ und ρ²cos²φ der OPD am Feldpunkt 1 bei λ_ref (in mm), aus
// einer Anpassung mit Termen bis zur 6. Ordnung. Die OPD von Mikroptim ist −W (Welford).
struct FittedWave {
    double W040, W131, W222;
    double W220_total; // ρ²-Koeffizient am Feldrand minus auf der Achse: (S_III + S_IV)/4
};

FittedWave fit_wavefront(const mko::LoadedSystem& l) {
    mko::WavefrontOptions o;
    o.rings = 12;
    o.spokes = 12;
    const mko::WavefrontResult wf = mko::evaluate_wavefront(l, o);
    int lref = 0;
    for (int k = 0; k < wf.wavelengths; ++k) {
        if (wf.at(1, k).wavelength_nm == l.system.spectrum.reference_nm) {
            lref = k;
        }
    }
    const mko::WavefrontMap& m = wf.at(1, lref);
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    for (int k = 0; k < m.samples(); ++k) {
        const double r = m.pupil[k].rho;
        const double c = std::cos(m.pupil[k].phi);
        X.push_back({1.0, r * r, r * r * r * r, r * c, r * r * r * c, r * r * c * c, std::pow(r, 6),
                     std::pow(r, 5) * c, std::pow(r, 4) * c * c, std::pow(r, 3) * c * c * c});
        y.push_back(-m.opd[k]);
    }
    const std::vector<double> x = least_squares(X, y);
    // Achse: nur ρ-Abhängigkeit (Defokus der Bildebene steckt in beiden ρ²-Koeffizienten)
    const mko::WavefrontMap& a = wf.at(0, lref);
    std::vector<std::vector<double>> X0;
    std::vector<double> y0;
    for (int k = 0; k < a.samples(); ++k) {
        const double r = a.pupil[k].rho;
        X0.push_back({1.0, r * r, r * r * r * r, std::pow(r, 6)});
        y0.push_back(-a.opd[k]);
    }
    const std::vector<double> x0 = least_squares(X0, y0);
    return {x[2], x[4], x[5], x[1] - x0[1]};
}

void expect_relative(double value, double reference, double tol, const std::string& what) {
    EXPECT_LE(std::abs(value - reference), tol * std::abs(reference))
        << what << ": " << value << " gegen " << reference;
}

} // namespace

// Seidel-Summen gegen die getracte Wellenfront bei kleiner Apertur und kleinem Feld.
TEST(Aberrations, SeidelSumsMatchTracedWavefront) {
    for (const auto& [name, scale, field] :
         {std::tuple{std::string("einzellinse_bk7.json"), 0.2, 0.5},
          std::tuple{std::string("testsystem_mehrlinser.json"), 0.15, 0.3}}) {
        const mko::LoadedSystem l = scaled_system(name, scale, field);
        const mko::AberrationReport rep = mko::compute_aberrations(l.system, l.catalog);
        const FittedWave fit = fit_wavefront(l);
        expect_relative(rep.S[0] / 8.0, fit.W040, kSeidelSmallAperture, name + " W040");
        expect_relative(rep.S[1] / 2.0, fit.W131, kSeidelSmallAperture, name + " W131");
        expect_relative(rep.S[2] / 2.0, fit.W222, kSeidelSmallAperture, name + " W222");
        // Bildfeldwölbung (M10-T2, Feldklassen): (S_III + S_IV)/4
        expect_relative((rep.S[2] + rep.S[3]) / 4.0, fit.W220_total, kSeidelSmallAperture,
                        name + " W220 gesamt");
        const mko::SeidelWaves w = rep.waves();
        EXPECT_DOUBLE_EQ(w.W040, rep.S[0] / 8.0 / (rep.reference_nm * 1e-6));
    }
}

// S_IV = H² P (Petzvalsumme der Paraxialoptik), Lagrange-Invariante an jeder Fläche, Summen.
TEST(Aberrations, ExactIdentities) {
    for (const char* name : {"einzellinse_bk7.json", "testsystem_mehrlinser.json",
                             "zweilinser_petzval.json", "zielspezifikation_start.json"}) {
        const mko::LoadedSystem l = mko::load_system(system_path(name));
        mko::AberrationOptions opt;
        opt.field_angle_deg = 1.0;
        const mko::AberrationReport rep = mko::compute_aberrations(l.system, l.catalog, opt);
        const mko::ParaxialData p = mko::compute_paraxial(l.system, l.catalog);
        const double H = rep.lagrange;
        expect_relative(rep.S[3], H * H * p.petzval_sum, kSeidelIdentity,
                        std::string(name) + " Petzval");
        std::array<double, 5> sum{};
        for (const mko::SurfaceAberration& s : rep.surfaces) {
            const double h = s.n_after * (s.ubar_after * s.y - s.u_after * s.ybar);
            expect_relative(h, H, kSeidelIdentity, std::string(name) + " Lagrange");
            for (int k = 0; k < 5; ++k) {
                sum[static_cast<std::size_t>(k)] += s.S[static_cast<std::size_t>(k)];
            }
        }
        for (int k = 0; k < 4; ++k) {
            EXPECT_EQ(sum[static_cast<std::size_t>(k)], rep.S[static_cast<std::size_t>(k)]);
        }
        EXPECT_EQ(rep.surfaces.front().S[0], 0.0) << "Blende ohne Beitrag";
    }
}

// C_I gegen den paraxialen Farblängsfehler: C_I ≈ −n'u'² (s'_F(λ_kurz) − s'_F(λ_lang)).
TEST(Aberrations, AxialColorMatchesParaxialFocalShift) {
    for (const char* name : {"einzellinse_bk7.json", "testsystem_mehrlinser.json"}) {
        const mko::LoadedSystem l = mko::load_system(system_path(name));
        const mko::AberrationReport rep = mko::compute_aberrations(l.system, l.catalog);
        const double dl =
            mko::compute_paraxial(l.system, l.catalog, rep.short_nm).back_focal_distance -
            mko::compute_paraxial(l.system, l.catalog, rep.long_nm).back_focal_distance;
        const mko::SurfaceAberration& last = rep.surfaces.back();
        expect_relative(rep.C[0], -last.n_after * last.u_after * last.u_after * dl,
                        kSeidelChromaticFirstOrder, std::string(name) + " C_I");
    }
}

// Sphärochromasie = S_I(λ_kurz) − S_I(λ_lang), je mit eigenen paraxialen Strahlen.
TEST(Aberrations, SpherochromatismIsDifferenceOfSphericalSums) {
    const mko::LoadedSystem l = mko::load_system(system_path("testsystem_mehrlinser.json"));
    const mko::AberrationReport rep = mko::compute_aberrations(l.system, l.catalog);
    mko::AberrationOptions s;
    s.reference_nm = rep.short_nm;
    mko::AberrationOptions g;
    g.reference_nm = rep.long_nm;
    const double diff = mko::compute_aberrations(l.system, l.catalog, s).S[0] -
                        mko::compute_aberrations(l.system, l.catalog, g).S[0];
    expect_relative(rep.spherochromatism, diff, 1e-12, "Sphärochromasie");
}

TEST(Aberrations, OptionsAndErrors) {
    mko::LoadedSystem l = mko::load_system(system_path("einzellinse_bk7.json"));
    const mko::AberrationReport rep = mko::compute_aberrations(l.system, l.catalog);
    EXPECT_EQ(rep.field_angle_deg, 2.0); // größter Feldwinkel der Datei
    EXPECT_EQ(rep.surfaces.size(), l.system.surfaces.size());
    mko::AberrationOptions zero;
    zero.field_angle_deg = 0.0;
    const mko::AberrationReport axial = mko::compute_aberrations(l.system, l.catalog, zero);
    EXPECT_EQ(axial.S[1], 0.0);
    EXPECT_EQ(axial.S[3], 0.0);
    EXPECT_EQ(axial.S[0], rep.S[0]); // Randstrahl hängt nicht vom Feld ab
    mko::AberrationOptions bad;
    bad.field_angle_deg = 95.0;
    EXPECT_THROW((void)mko::compute_aberrations(l.system, l.catalog, bad), std::invalid_argument);
    l.system.source.type = mko::SourceType::Point;
    EXPECT_THROW((void)mko::compute_aberrations(l.system, l.catalog), std::invalid_argument);
}

// --- M10-T1b: Vollordnung, Zone, OSC ---------------------------------------------------------

// Bei kleiner Apertur geht die Vollordnung pro Fläche in S_I über, in der Zone in S_I ρ⁴.
TEST(Aberrations, FullOrderTendsToSeidelAtSmallAperture) {
    for (const char* name :
         {"einzellinse_bk7.json", "testsystem_mehrlinser.json", "zweilinser_petzval.json"}) {
        const mko::LoadedSystem l = scaled_system(name, 0.02, 1.0);
        const mko::AberrationReport rep = mko::compute_aberrations(l.system, l.catalog);
        ASSERT_TRUE(rep.real_rays_valid) << name;
        const double r4 = std::pow(mko::kZoneRho, 4);
        for (const mko::SurfaceAberration& s : rep.surfaces) {
            if (s.S[0] == 0.0) {
                continue;
            }
            expect_relative(s.full_order_spherical, s.S[0], kFullOrderSmallAperture,
                            std::string(name) + " Fläche " + std::to_string(s.surface));
            expect_relative(s.full_order_spherical_zone, s.S[0] * r4, kFullOrderSmallAperture,
                            std::string(name) + " Zone, Fläche " + std::to_string(s.surface));
            EXPECT_EQ(s.higher_order_spherical, s.full_order_spherical - s.S[0]);
        }
    }
}

// Summe der Flächenbeiträge = −2 n′u′ sin U′ δL′ des realen Randstrahls im Endraum (exakt bis
// auf Rundung, auch bei voller Apertur mit großem Anteil höherer Ordnung).
TEST(Aberrations, FullOrderSumEqualsFinalRealRayAberration) {
    for (const char* name : {"testsystem_mehrlinser.json", "zielspezifikation_start.json"}) {
        mko::LoadedSystem l = mko::load_system(system_path(name));
        const mko::AberrationReport rep = mko::compute_aberrations(l.system, l.catalog);
        ASSERT_TRUE(rep.real_rays_valid) << name;
        // eigener Trace bei λ_ref ohne Geisterplatten
        mko::SystemDescription sys = l.system;
        sys.ghost_plates.clear();
        sys.variables = {}; // Variablen verweisen u. U. auf die Geisterplatten
        sys.constraints = {};
        sys.spectrum.wavelengths_nm = {rep.reference_nm};
        sys.spectrum.weights = {1.0};
        const mko::LoadedSystem p = mko::prepare_system(sys, system_path(name).parent_path());
        const auto view = p.table.view();
        mko::Ray<double> ray = mko::collimated_ray<double>(rep.stop_semi_diameter, 0.0, 1.0, 0.0);
        auto rec = std::make_unique<mko::TraceRecord<double>>();
        int failed = -1;
        ASSERT_EQ(mko::trace_ray(view, 0, ray, rec.get(), &failed), mko::TraceStatus::Ok);
        const std::size_t S = rep.surfaces.size();
        const mko::SurfaceAberration& last = rep.surfaces.back();
        const mko::Vec3<double>& pt = rec->p[S - 1];
        const mko::Vec3<double>& d = rec->d[S - 1];
        const double L = pt.z - pt.y * d.z / d.y;
        const double l_par = -last.y / last.u_after;
        const double final_value = -2.0 * last.n_after * last.u_after * d.y * (L - l_par);
        expect_relative(rep.full_order_spherical, final_value, 1e-11, std::string(name) + " Σ");
        EXPECT_NE(rep.higher_order_spherical, 0.0);
    }
}

// OSC → −S_II / (2H) bei kleiner Apertur (Koma 3. Ordnung); ohne reale Strahlen NaN.
TEST(Aberrations, OscMatchesThirdOrderComa) {
    for (const char* name :
         {"einzellinse_bk7.json", "testsystem_mehrlinser.json", "zweilinser_petzval.json"}) {
        const mko::LoadedSystem l = scaled_system(name, 0.02, 1.0);
        const mko::AberrationReport rep = mko::compute_aberrations(l.system, l.catalog);
        expect_relative(rep.osc, -rep.S[1] / (2.0 * rep.lagrange), kOscThirdOrder,
                        std::string(name) + " OSC");
        // Zone: dritte Ordnung skaliert mit ρ²
        expect_relative(rep.osc_zone, 0.5 * rep.osc, 2.0 * kOscThirdOrder,
                        std::string(name) + " OSC Zone");
    }
    mko::LoadedSystem l = mko::load_system(system_path("einzellinse_bk7.json"));
    mko::AberrationOptions o;
    o.real_rays = false;
    const mko::AberrationReport rep = mko::compute_aberrations(l.system, l.catalog, o);
    EXPECT_FALSE(rep.real_rays_valid);
    EXPECT_TRUE(std::isnan(rep.osc));
    EXPECT_TRUE(std::isnan(rep.surfaces[1].full_order_spherical));
}

// M10-T2: chromatischer Fokusverlauf über die Spektrallinien gegen die Paraxialoptik.
TEST(Aberrations, FocalShiftsMatchParaxialFoci) {
    const mko::LoadedSystem l = mko::load_system(system_path("testsystem_mehrlinser.json"));
    const mko::AberrationReport rep = mko::compute_aberrations(l.system, l.catalog);
    ASSERT_EQ(rep.focal_shifts.size(), std::size(mko::kSpectralLines));
    const double ref =
        mko::compute_paraxial(l.system, l.catalog, rep.reference_nm).back_focal_distance;
    for (const mko::FocalShift& f : rep.focal_shifts) {
        const double bfd = mko::compute_paraxial(l.system, l.catalog, f.nm).back_focal_distance;
        EXPECT_EQ(f.shift, bfd - ref) << f.line;
        if (f.nm == rep.reference_nm) {
            EXPECT_EQ(f.shift, 0.0);
        }
    }
    mko::AberrationOptions o;
    o.focal_shift = false;
    EXPECT_TRUE(mko::compute_aberrations(l.system, l.catalog, o).focal_shifts.empty());
}
