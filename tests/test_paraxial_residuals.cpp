// Paraxiale Nebenbedingungen (M3-T3, Petzval M4-T4): ein Test pro Nebenbedingung gegen
// geschlossene Formeln (Linsenmacherformel, Schnittweite, Summen der Dicken, Petzvalsumme),
// einseitige Bedingungen, Geisterplatten, Paraxialtrace aller Varianten parallel = seriell
// bitgleich, Fehlerfälle.

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/optim/extra_residuals.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "mikroptim/trace/ghost_plate.hpp"
#include "residual_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using mko::BatchTopology;
using mko::ConstraintMode;
using mko::ConstraintType;
using mko::ParaxialConstraint;
using mko::ParaxialConstraints;
using mko::ParaxialData;
using mko::SourceKind;
using mko::SurfaceKind;
using mko::SystemBatchHost;
using mko::test::kLengthFp64;
using mko::test::kParaxialRelative;

namespace {

const std::string kSystems = MKO_SOURCE_DIR "/data/systems/";

bool same_bits(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

// Dicke Linse in Luft (wie test_paraxial.cpp).
struct ThickLens {
    double n, c1, c2, t;

    // Linsenmacherformel 1/f = (n-1) [c1 - c2 + (n-1) t c1 c2 / n]
    double power() const { return (n - 1.0) * (c1 - c2 + (n - 1.0) * t * c1 * c2 / n); }
    double focal_length() const { return 1.0 / power(); }
    // Hintere Schnittweite f (1 - (n-1) t c1 / n)
    double bfl() const { return focal_length() * (1.0 - (n - 1.0) * t * c1 / n); }
};

// Hintere Krümmung, mit der die Linse die Brennweite f hat (Linsenmacherformel nach c2 gelöst).
double c2_for_focal_length(double n, double c1, double t, double f) {
    return (1.0 / (f * (n - 1.0)) - c1) / ((n - 1.0) * t * c1 / n - 1.0);
}

// Von Hand gebauter Batch: Blende (Abstand 5 mm zur Linse), Linse, Bildebene; Brechzahlen bei
// λ_ref in paraxial_index (index wird für die Paraxialrechnung nicht gebraucht).
constexpr double kStopDistance = 5.0;

SystemBatchHost lens_batch(int variants) {
    BatchTopology t;
    t.surfaces = 3;
    t.wavelengths = 1;
    t.fields = 1;
    t.type = {SurfaceKind::Stop, SurfaceKind::Sphere, SurfaceKind::Sphere};
    SystemBatchHost b(t, variants);
    b.wavelength_nm(0) = 587.56;
    b.wavelength_weight(0) = 1.0;
    b.field_weight(0) = 1.0;
    b.reference_wavelength_nm = 587.56;
    return b;
}

void set_lens(SystemBatchHost& b, int v, const ThickLens& L, double image_distance,
              double pupil_radius = 2.0) {
    b.curvature(v, 1) = L.c1;
    b.curvature(v, 2) = L.c2;
    b.thickness(v, 0) = kStopDistance;
    b.thickness(v, 1) = L.t;
    b.thickness(v, 2) = image_distance;
    const double n[3] = {1.0, L.n, 1.0};
    for (int s = 0; s < 3; ++s) {
        b.index(v, s, 0) = n[s];
        b.paraxial_index(v, s) = n[s];
    }
    b.pupil_radius(v) = pupil_radius;
}

ParaxialConstraint constraint(ConstraintType type, double target, double weight = 1.0,
                              ConstraintMode mode = ConstraintMode::Equal) {
    ParaxialConstraint c;
    c.type = type;
    c.target = target;
    c.weight = weight;
    c.mode = mode;
    return c;
}

double residual(const ParaxialConstraint& c, const SystemBatchHost& b, int v) {
    ParaxialConstraints cs;
    cs.items.push_back(c);
    double r = 0.0;
    mko::paraxial_residuals(cs, b, v, std::span<double>(&r, 1));
    return r;
}

// Systembeschreibung mit Zusatzabschnitten aus einer Datei.
mko::SystemDescription with_constraints(const std::string& file, const mko::Target& target,
                                        std::vector<mko::ConstraintSpec> items) {
    mko::SystemDescription sys = mko::read_system_file(kSystems + file);
    sys.target = target;
    sys.constraints = mko::Constraints{std::move(items)};
    mko::validate_system(sys);
    return sys;
}

} // namespace

// ---- Brennweite ------------------------------------------------------------------------------

// r = √w f_s (1 − f_s φ) mit φ aus der Linsenmacherformel; gleich √w f_s (f − f_s)/f.
TEST(ParaxialResiduals, FocalLengthMatchesLensmakerFormula) {
    const ThickLens lenses[] = {{1.5, 0.02, -0.015, 4.0},
                                {1.7, 0.05, 0.01, 6.0},
                                {1.6, -0.03, 0.02, 3.0}, // Zerstreuungslinse
                                {1.8, 0.2, 0.15, 10.0}}; // dicker Meniskus
    SystemBatchHost b = lens_batch(4);
    for (int v = 0; v < 4; ++v) {
        set_lens(b, v, lenses[v], 50.0);
    }
    constexpr double kTarget = 50.0;
    constexpr double kWeight = 4.0;
    const ParaxialConstraint c = constraint(ConstraintType::FocalLength, kTarget, kWeight);
    for (int v = 0; v < 4; ++v) {
        const double f = lenses[v].focal_length();
        const double expected = 2.0 * kTarget * (1.0 - kTarget * lenses[v].power());
        const double tol = kParaxialRelative * 2.0 * kTarget * (1.0 + std::abs(kTarget / f));
        const double r = residual(c, b, v);
        EXPECT_NEAR(r, expected, tol) << "Variante " << v;
        EXPECT_NEAR(r, 2.0 * kTarget * (f - kTarget) / f, tol) << "Variante " << v;
        // Vorzeichen wie φ_s − φ (bei Sammellinsen wie f − f_s)
        EXPECT_EQ(r > 0.0, lenses[v].power() < 1.0 / kTarget) << "Variante " << v;
    }
}

// Nahe dem Soll ist r = √w (f − f_s) bis auf Terme zweiter Ordnung; im Soll verschwindet r.
TEST(ParaxialResiduals, FocalLengthNearTargetIsLengthDifference) {
    constexpr double kTarget = 3.29;
    constexpr double kN = 1.5;
    constexpr double kC1 = 0.4;
    constexpr double kT = 1.5;
    SystemBatchHost b = lens_batch(3);
    const double eps[] = {0.0, 1e-4, -1e-3};
    for (int v = 0; v < 3; ++v) {
        const double f = kTarget * (1.0 + eps[v]);
        set_lens(b, v, ThickLens{kN, kC1, c2_for_focal_length(kN, kC1, kT, f), kT}, 2.0);
    }
    const ParaxialConstraint c = constraint(ConstraintType::FocalLength, kTarget, 9.0);
    EXPECT_NEAR(residual(c, b, 0), 0.0, 1e-12);
    for (int v = 1; v < 3; ++v) {
        const double df = kTarget * eps[v];
        const double r = residual(c, b, v);
        // r / (√w Δf) = f_s / f = 1/(1 + ε) = 1 − ε + O(ε²)
        EXPECT_NEAR(r / (3.0 * df), 1.0 - eps[v], 2.0 * eps[v] * eps[v] + 1e-9) << v;
    }
}

// Formulierung über die Brechkraft: stetig und endlich durch φ = 0 (f = ±∞), linear in φ.
TEST(ParaxialResiduals, FocalLengthContinuousThroughZeroPower) {
    constexpr double kTarget = 20.0;
    SystemBatchHost b = lens_batch(5);
    // c1 = 0, c2 = δ: φ = −(n − 1) δ, also r = f_s (1 + (n − 1) f_s δ), steigend in δ.
    const double c2[] = {-1e-3, -1e-9, 0.0, 1e-9, 1e-3};
    for (int v = 0; v < 5; ++v) {
        set_lens(b, v, ThickLens{1.5, 0.0, c2[v], 2.0}, 10.0);
    }
    const ParaxialConstraint c = constraint(ConstraintType::FocalLength, kTarget);
    const ParaxialData plate = mko::batch_paraxial(b, 2);
    EXPECT_TRUE(plate.afocal);
    EXPECT_TRUE(std::isinf(plate.focal_length));
    EXPECT_EQ(residual(c, b, 2), kTarget); // φ = 0: r = √w f_s, endlich
    double previous = -std::numeric_limits<double>::infinity();
    for (int v = 0; v < 5; ++v) {
        const double r = residual(c, b, v);
        const double phi = -0.5 * c2[v];
        EXPECT_NEAR(r, kTarget * (1.0 - kTarget * phi), kLengthFp64) << v;
        EXPECT_GT(r, previous) << v; // streng monoton in δ, auch beim Vorzeichenwechsel von φ
        previous = r;
    }
    EXPECT_NEAR(residual(c, b, 1), kTarget, 1e-6); // stetig an φ = 0
    EXPECT_NEAR(residual(c, b, 3), kTarget, 1e-6);
}

// ---- Arbeitsabstand ------------------------------------------------------------------------

// Gleichung √w (w − w_s); Untergrenze √w min(w − w_s, 0); Obergrenze √w max(w − w_s, 0);
// stetig an der Grenze, inaktiv exakt 0.
TEST(ParaxialResiduals, WorkingDistanceEqualAndOneSided) {
    constexpr double kTarget = 0.35;
    const double w[] = {0.2, std::nextafter(kTarget, 0.0), kTarget, std::nextafter(kTarget, 1.0),
                        1.0};
    SystemBatchHost b = lens_batch(5);
    for (int v = 0; v < 5; ++v) {
        set_lens(b, v, ThickLens{1.5, 0.1, -0.1, 2.0}, w[v]);
    }
    const ParaxialConstraint eq = constraint(ConstraintType::WorkingDistance, kTarget, 25.0);
    const ParaxialConstraint lo =
        constraint(ConstraintType::WorkingDistance, kTarget, 25.0, ConstraintMode::Min);
    const ParaxialConstraint hi =
        constraint(ConstraintType::WorkingDistance, kTarget, 25.0, ConstraintMode::Max);
    for (int v = 0; v < 5; ++v) {
        const double q = w[v] - kTarget;
        EXPECT_EQ(residual(eq, b, v), 5.0 * q) << v;
        EXPECT_EQ(residual(lo, b, v), q < 0.0 ? 5.0 * q : 0.0) << v;
        EXPECT_EQ(residual(hi, b, v), q > 0.0 ? 5.0 * q : 0.0) << v;
    }
    // aktiv bzw. inaktiv
    EXPECT_LT(residual(lo, b, 0), 0.0);
    EXPECT_EQ(residual(lo, b, 4), 0.0);
    EXPECT_GT(residual(hi, b, 4), 0.0);
    EXPECT_EQ(residual(hi, b, 0), 0.0);
    // stetig an der Grenze: ein ulp neben dem Soll ist |r| <= √w · ulp
    for (int v = 1; v <= 3; ++v) {
        EXPECT_LE(std::abs(residual(lo, b, v)), 5.0 * 1e-16) << v;
        EXPECT_LE(std::abs(residual(hi, b, v)), 5.0 * 1e-16) << v;
    }
    // Gewicht 0 schaltet die Bedingung ab
    EXPECT_EQ(residual(constraint(ConstraintType::WorkingDistance, kTarget, 0.0), b, 0), 0.0);
    // Der Arbeitsabstand braucht keinen Paraxialtrace (Brechzahlen bei λ_ref dürfen fehlen)
    b.paraxial_index(0, 1) = 0.0;
    EXPECT_EQ(residual(eq, b, 0), 5.0 * (0.2 - kTarget));
}

// ---- Abgleichlänge ---------------------------------------------------------------------------

// L = offset + Σ_{j ≥ Referenzfläche} t_j (bis zur Bildebene); Referenzfläche und Versatz.
TEST(ParaxialResiduals, ParfocalLengthWithReferenceSurfaceAndOffset) {
    SystemBatchHost b = lens_batch(2);
    set_lens(b, 0, ThickLens{1.5, 0.1, -0.1, 2.0}, 0.5);
    set_lens(b, 1, ThickLens{1.5, 0.1, -0.1, 3.5}, 40.25);
    ParaxialConstraint from_stop = constraint(ConstraintType::ParfocalLength, 45.0, 4.0);
    ParaxialConstraint from_lens = from_stop;
    from_lens.reference_surface = 1;
    from_lens.offset = 2.5;
    ParaxialConstraint upper = from_lens;
    upper.mode = ConstraintMode::Max;
    // Variante 0: 5 + 2 + 0,5 = 7,5 bzw. 2,5 + 2 + 0,5 = 5
    EXPECT_EQ(residual(from_stop, b, 0), 2.0 * (7.5 - 45.0));
    EXPECT_EQ(residual(from_lens, b, 0), 2.0 * (5.0 - 45.0));
    EXPECT_EQ(residual(upper, b, 0), 0.0);
    // Variante 1: 5 + 3,5 + 40,25 = 48,75 bzw. 2,5 + 3,5 + 40,25 = 46,25
    EXPECT_EQ(residual(from_stop, b, 1), 2.0 * (48.75 - 45.0));
    EXPECT_EQ(residual(from_lens, b, 1), 2.0 * (46.25 - 45.0));
    EXPECT_EQ(residual(upper, b, 1), 2.0 * (46.25 - 45.0));
    // Referenzfläche = letzte Fläche: nur Versatz + Arbeitsabstand
    ParaxialConstraint last = from_lens;
    last.reference_surface = 2;
    EXPECT_EQ(residual(last, b, 1), 2.0 * (2.5 + 40.25 - 45.0));
}

// Mit Geisterplatten: Die Referenzfläche (Dateifläche) wird auf die Tabellenfläche abgebildet;
// ein geteilter Luftspalt zählt mit seiner ganzen Dicke.
TEST(ParaxialResiduals, ParfocalLengthWithGhostPlates) {
    mko::Target target;
    target.parfocal_length = 110.0;
    mko::ConstraintSpec spec;
    spec.type = ConstraintType::ParfocalLength;
    mko::ConstraintSpec spec1 = spec;
    spec1.reference_surface = 1;
    spec1.offset = 1.0;
    // je eine Bedingung pro Datei (jeder Typ höchstens einmal)
    const auto sys0 = with_constraints("einzellinse_geisterplatte.json", target, {spec});
    const auto sys1 = with_constraints("einzellinse_geisterplatte.json", target, {spec1});
    const ParaxialConstraints c0 = mko::make_paraxial_constraints(sys0);
    const ParaxialConstraints c1 = mko::make_paraxial_constraints(sys1);
    ASSERT_EQ(c0.size(), 1);
    ASSERT_EQ(c1.size(), 1);
    EXPECT_EQ(c0.items[0].reference_surface, 0);
    EXPECT_EQ(c1.items[0].reference_surface, 3); // Dateifläche 1 = Tabellenfläche 3
    EXPECT_EQ(c1.items[0].offset, 1.0);
    const SystemBatchHost b = mko::make_system_batch(mko::prepare_system(sys0, MKO_SOURCE_DIR));
    ASSERT_EQ(b.surfaces(), 5);
    double r = 0.0;
    mko::make_paraxial_residuals(c0).evaluate(b, 0, std::span<double>(&r, 1));
    // Tabellendicken 1 + 2,5 + 1,5 (Spalt 5 mm mit zwei Platten) + 6 + 98,23
    EXPECT_NEAR(r, 5.0 + 6.0 + 98.23 - 110.0, kLengthFp64);
    mko::make_paraxial_residuals(c1).evaluate(b, 0, std::span<double>(&r, 1));
    EXPECT_NEAR(r, 1.0 + 6.0 + 98.23 - 110.0, kLengthFp64);
}

// ---- Bildebene im paraxialen Fokus -----------------------------------------------------------

// r = √w a (A + t C/n'): Höhe des Randstrahls (Einfallshöhe a) in der Bildebene, = −√w a C δ
// mit dem Defokus δ = s'_F − t; 0 in der Schnittweite der Linsenmacherformel.
TEST(ParaxialResiduals, ParaxialFocusMatchesDefocus) {
    const ThickLens lens{1.6, 0.04, -0.02, 5.0};
    const double bfl = lens.bfl();
    const double t[] = {bfl, bfl - 0.3, bfl + 1.2};
    SystemBatchHost b = lens_batch(3);
    for (int v = 0; v < 3; ++v) {
        set_lens(b, v, lens, t[v], 2.0 + v);
    }
    const ParaxialConstraint c = constraint(ConstraintType::ParaxialFocus, 0.0, 16.0);
    EXPECT_NEAR(residual(c, b, 0), 0.0, 4.0 * 2.0 * kLengthFp64);
    for (int v = 1; v < 3; ++v) {
        const ParaxialData p = mko::batch_paraxial(b, v);
        const double a = 2.0 + v;
        const double expected = -4.0 * a * p.abcd[2] * (bfl - t[v]); // C = −φ
        EXPECT_NEAR(residual(c, b, v), expected, kParaxialRelative * std::abs(expected)) << v;
        EXPECT_NEAR(p.image_defocus, bfl - t[v], kLengthFp64) << v;
        // Bildebene zu nah (δ > 0) bei Sammellinse: Strahl noch über der Achse, r > 0
        EXPECT_EQ(residual(c, b, v) > 0.0, t[v] < bfl) << v;
    }
}

// ---- Geisterplatten --------------------------------------------------------------------------

// Einzelne Geisterplatte zwischen Planflächen in Luft: Brechkraft des Systems = φ_G =
// −2 (n_G − 1) t₂ mit t₂ = Σ a_k (−1)^{k+1} k (k+1) / r_g², Petzval φ_G / n_G.
TEST(ParaxialResiduals, GhostPlatePowerClosedForm) {
    BatchTopology t;
    t.surfaces = 3;
    t.wavelengths = 1;
    t.fields = 1;
    t.ghost_plates = 1;
    t.ghost_terms = 3;
    t.type = {SurfaceKind::Stop, SurfaceKind::GhostPlate, SurfaceKind::Plane};
    SystemBatchHost b(t, 1);
    b.reference_wavelength_nm = 587.56;
    b.ghost_norm_radius(0) = 5.0;
    b.ghost_coeff(0, 0, 0) = 1e-3;
    b.ghost_coeff(0, 0, 1) = -2e-4;
    b.ghost_coeff(0, 0, 2) = 5e-5;
    b.paraxial_ghost_index(0, 0) = 1.5;
    for (int s = 0; s < 3; ++s) {
        b.paraxial_index(0, s) = 1.0;
        b.thickness(0, s) = 2.0 + s;
    }
    b.pupil_radius(0) = 2.0;
    const double t2 = (1e-3 * 2.0 - (-2e-4) * 6.0 + 5e-5 * 12.0) / 25.0;
    const double phi_g = -2.0 * 0.5 * t2;
    const ParaxialData p = mko::batch_paraxial(b, 0);
    EXPECT_NEAR(p.power, phi_g, kParaxialRelative * std::abs(phi_g));
    EXPECT_NEAR(p.petzval_sum, phi_g / 1.5, kParaxialRelative * std::abs(phi_g));
    constexpr double kTarget = -200.0;
    const double r = residual(constraint(ConstraintType::FocalLength, kTarget), b, 0);
    EXPECT_NEAR(r, kTarget * (1.0 - kTarget * phi_g), kParaxialRelative * std::abs(kTarget));
    // Brechzahl des Glases bei λ_ref geht ein (nicht ghost_index)
    b.ghost_index(0, 0, 0) = 1.9;
    EXPECT_EQ(mko::batch_paraxial(b, 0).power, p.power);
    b.paraxial_ghost_index(0, 0) = 1.9;
    EXPECT_NEAR(mko::batch_paraxial(b, 0).power, -2.0 * 0.9 * t2,
                kParaxialRelative * std::abs(phi_g));
}

// Batch-Varianten aus Dateien: gleich compute_paraxial(system, catalog) bei λ_ref, auch mit
// Geisterplatten; ohne Geisterkoeffizienten gleich dem System ohne Platten.
TEST(ParaxialResiduals, BatchParaxialMatchesSystem) {
    for (const char* file :
         {"einzellinse_bk7.json", "testsystem_mehrlinser.json", "einzellinse_geisterplatte.json"}) {
        const mko::LoadedSystem ls = mko::load_system(kSystems + file);
        const SystemBatchHost b = mko::make_system_batch(ls, 2);
        const ParaxialData expected = mko::compute_paraxial(ls.system, ls.catalog);
        const ParaxialData actual = mko::batch_paraxial(b, 1);
        EXPECT_TRUE(same_bits(actual.power, expected.power)) << file;
        EXPECT_TRUE(same_bits(actual.back_focal_distance, expected.back_focal_distance)) << file;
        EXPECT_TRUE(same_bits(actual.image_defocus, expected.image_defocus)) << file;
        EXPECT_TRUE(same_bits(actual.petzval_sum, expected.petzval_sum)) << file;
        EXPECT_TRUE(same_bits(actual.exit_pupil_position, expected.exit_pupil_position)) << file;
    }

    const mko::LoadedSystem ls = mko::load_system(kSystems + "einzellinse_geisterplatte.json");
    SystemBatchHost b = mko::make_system_batch(ls, 2);
    for (int g = 0; g < b.ghost_plates(); ++g) {
        for (int k = 0; k < b.ghost_terms(); ++k) {
            b.ghost_coeff(1, g, k) = 0.0;
        }
    }
    mko::SystemDescription plain = ls.system;
    plain.ghost_plates.clear();
    const ParaxialData lens = mko::compute_paraxial(plain, ls.catalog);
    const ParaxialData with = mko::batch_paraxial(b, 0);
    const ParaxialData without = mko::batch_paraxial(b, 1);
    EXPECT_NEAR(without.power, lens.power, kParaxialRelative * std::abs(lens.power));
    EXPECT_NEAR(without.petzval_sum, lens.petzval_sum,
                kParaxialRelative * std::abs(lens.petzval_sum));
    // Die Geisterplatten tragen zur Brechkraft bei (Koeffizienten der Datei: φ_G ≠ 0)
    EXPECT_GT(std::abs(with.power - lens.power), 1e-6);
    const ParaxialConstraint c = constraint(ConstraintType::FocalLength, 100.0);
    EXPECT_NE(residual(c, b, 0), residual(c, b, 1));
}

// ---- Batch: parallel = seriell -------------------------------------------------------------

// Paraxialtrace und Residuen aller Varianten: bitgleich für jede Threadzahl und gleich der
// seriellen Auswertung pro Variante.
TEST(ParaxialResiduals, ParallelEqualsSerialBitwise) {
    const mko::LoadedSystem ls = mko::load_system(kSystems + "testsystem_mehrlinser.json");
    constexpr int kVariants = 157;
    SystemBatchHost b = mko::test::perturbed_batch(ls, kVariants, 20261018);
    for (int v = 1; v < kVariants; ++v) { // auch die Brechzahlen bei λ_ref stören
        for (int s = 0; s < b.surfaces(); ++s) {
            if (b.paraxial_index(v, s) > 1.0) {
                b.paraxial_index(v, s) += 1e-3 * (v % 7 - 3);
            }
        }
    }
    const double S0 = b.thickness(0, b.surfaces() - 1);
    ParaxialConstraints cs;
    cs.items = {constraint(ConstraintType::FocalLength, 3.29, 100.0),
                constraint(ConstraintType::ParfocalLength, 45.0, 2.0, ConstraintMode::Max),
                constraint(ConstraintType::WorkingDistance, S0, 10.0, ConstraintMode::Min),
                constraint(ConstraintType::ParaxialFocus, 0.0, 1.0)};
    cs.items[1].reference_surface = 1;
    cs.items[1].offset = 3.0;
    const mko::ExtraResiduals extra = mko::make_paraxial_residuals(cs);
    ASSERT_EQ(extra.count, 4);

    std::vector<double> serial(static_cast<std::size_t>(kVariants * extra.count));
    for (int v = 0; v < kVariants; ++v) {
        mko::paraxial_residuals(
            cs, b, v, std::span<double>(serial).subspan(static_cast<std::size_t>(v) * 4, 4));
    }
    int active_min = 0;
    for (int v = 0; v < kVariants; ++v) {
        active_min += serial[static_cast<std::size_t>(v) * 4 + 2] != 0.0 ? 1 : 0;
    }
    EXPECT_GT(active_min, 0);         // einseitige Bedingung in einigen Varianten aktiv …
    EXPECT_LT(active_min, kVariants); // … und in anderen nicht

    for (const int threads : {1, 2, 3, 8, 64, 0}) {
        std::vector<double> parallel(serial.size(), -1.0);
        mko::evaluate_extra_residuals(extra, b, parallel, 0, -1, threads);
        EXPECT_TRUE(same_bits(parallel, serial)) << threads << " Threads";
    }
    // Teilbereich
    std::vector<double> part(static_cast<std::size_t>(10 * 4));
    mko::evaluate_extra_residuals(extra, b, part, 100, 10, 3);
    EXPECT_TRUE(same_bits(part, std::vector<double>(serial.begin() + 400, serial.begin() + 440)));

    const std::vector<ParaxialData> p1 = mko::batch_paraxial_all(b, 1);
    const std::vector<ParaxialData> p5 = mko::batch_paraxial_all(b, 5);
    ASSERT_EQ(p1.size(), static_cast<std::size_t>(kVariants));
    for (int v = 0; v < kVariants; ++v) {
        const auto& x = p1[static_cast<std::size_t>(v)];
        const auto& y = p5[static_cast<std::size_t>(v)];
        ASSERT_TRUE(same_bits(x.power, y.power) && same_bits(x.petzval_sum, y.petzval_sum) &&
                    same_bits(x.back_focal_distance, y.back_focal_distance) &&
                    same_bits(x.abcd[1], y.abcd[1]))
            << "Variante " << v;
        ASSERT_TRUE(same_bits(x.power, mko::batch_paraxial(b, v).power)) << "Variante " << v;
    }
}

// Ausnahmen in Threads: weitergeworfen wird die der kleinsten Variante, unabhängig von der
// Threadzahl.
TEST(ParaxialResiduals, ErrorsPropagateDeterministically) {
    const mko::LoadedSystem ls = mko::load_system(kSystems + "einzellinse_bk7.json");
    SystemBatchHost b = mko::make_system_batch(ls, 24);
    b.paraxial_index(5, 1) = 0.0;                                  // "muss > 0 sein"
    b.curvature(17, 1) = std::numeric_limits<double>::quiet_NaN(); // "nicht endlich"
    ParaxialConstraints cs;
    cs.items = {constraint(ConstraintType::FocalLength, 100.0)};
    const mko::ExtraResiduals extra = mko::make_paraxial_residuals(cs);
    std::vector<double> out(24);
    for (const int threads : {1, 4, 24}) {
        try {
            mko::evaluate_extra_residuals(extra, b, out, 0, -1, threads);
            ADD_FAILURE() << "keine Ausnahme";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("> 0"), std::string::npos) << e.what();
        }
        EXPECT_THROW((void)mko::batch_paraxial_all(b, threads), std::invalid_argument);
    }
    // Ausgabe- und Bereichsprüfung
    std::vector<double> wrong(23);
    EXPECT_THROW(mko::evaluate_extra_residuals(extra, b, wrong), std::invalid_argument);
    EXPECT_THROW(mko::evaluate_extra_residuals(extra, b, out, 1, 24), std::out_of_range);
    EXPECT_THROW(mko::evaluate_extra_residuals(extra, b, out, -1, 1), std::out_of_range);
    EXPECT_THROW((void)mko::batch_paraxial(b, 24), std::out_of_range);
    double r2[2];
    EXPECT_THROW(mko::paraxial_residuals(cs, b, 0, std::span<double>(r2, 2)),
                 std::invalid_argument);
    // Keine Zusatzresiduen: nichts zu tun
    std::vector<double> none;
    EXPECT_NO_THROW(mko::evaluate_extra_residuals(mko::ExtraResiduals{}, b, none));
}

// ---- Aus der Systemdatei -----------------------------------------------------------------

// Beispieldatei: Sollwerte aus target, Standardwerte, Residuen gegen die Formeln.
TEST(ParaxialResiduals, FromExampleFile) {
    const mko::LoadedSystem ls =
        mko::load_system(kSystems + "einzellinse_bk7_nebenbedingungen.json");
    const ParaxialConstraints cs = mko::make_paraxial_constraints(ls.system);
    ASSERT_EQ(cs.size(), 3);
    EXPECT_EQ(cs.items[0].type, ConstraintType::FocalLength);
    EXPECT_EQ(cs.items[0].weight, 100.0);
    EXPECT_EQ(cs.items[0].target, 100.0);
    EXPECT_EQ(cs.items[0].item, 0);
    EXPECT_EQ(cs.items[1].type, ConstraintType::ParfocalLength);
    EXPECT_EQ(cs.items[1].weight, 1.0);
    EXPECT_EQ(cs.items[1].target, 115.0);
    EXPECT_EQ(cs.items[1].reference_surface, 0);
    EXPECT_EQ(cs.items[1].offset, 5.0);
    EXPECT_EQ(cs.items[2].type, ConstraintType::WorkingDistance);
    EXPECT_EQ(cs.items[2].mode, ConstraintMode::Min);
    EXPECT_EQ(cs.items[2].target, 95.0);
    EXPECT_TRUE(cs.needs_trace());

    SystemBatchHost b = mko::make_system_batch(ls, 2);
    b.image_distance(1) = 90.0; // Untergrenze 95 verletzt
    const mko::ExtraResiduals extra = mko::make_paraxial_residuals(cs);
    std::vector<double> r(6);
    mko::evaluate_extra_residuals(extra, b, r);
    const double phi = mko::compute_paraxial(ls.system, ls.catalog).power;
    EXPECT_EQ(r[0], 10.0 * (100.0 * (1.0 - 100.0 * phi)));
    EXPECT_NEAR(r[1], 5.0 + 5.0 + 6.0 + 99.05 - 115.0, kLengthFp64);
    EXPECT_EQ(r[2], 0.0);                           // 99,05 >= 95
    EXPECT_NEAR(r[3], r[0], 1e-9 * std::abs(r[0])); // Arbeitsabstand ändert φ nicht
    EXPECT_NEAR(r[4], 5.0 + 5.0 + 6.0 + 90.0 - 115.0, kLengthFp64);
    EXPECT_EQ(r[5], std::sqrt(10.0) * (90.0 - 95.0));

    // Ohne Abschnitt: keine Bedingungen
    mko::SystemDescription plain = ls.system;
    plain.constraints.reset();
    EXPECT_EQ(mko::make_paraxial_constraints(plain).size(), 0);
    ParaxialConstraints wd_only;
    wd_only.items = {cs.items[2]};
    EXPECT_FALSE(wd_only.needs_trace());
}

// ---- Prüfung gegen die Topologie ------------------------------------------------------------

TEST(ParaxialResiduals, CheckAgainstTopology) {
    const mko::LoadedSystem ls = mko::load_system(kSystems + "einzellinse_geisterplatte.json");
    const SystemBatchHost b = mko::make_system_batch(ls, 1);
    const auto rejected = [&](const ParaxialConstraint& c) {
        ParaxialConstraints cs;
        cs.items = {c};
        EXPECT_THROW(mko::check_paraxial_constraints(cs, b.topology()), std::invalid_argument);
        double r = 0.0;
        EXPECT_THROW(mko::make_paraxial_residuals(cs).evaluate(b, 0, std::span<double>(&r, 1)),
                     std::invalid_argument);
    };
    ParaxialConstraint c = constraint(ConstraintType::ParfocalLength, 45.0);
    c.reference_surface = 5;
    rejected(c); // außerhalb
    c.reference_surface = 1;
    rejected(c); // Geisterplatte
    c.reference_surface = 0;
    c.offset = std::numeric_limits<double>::infinity();
    rejected(c);
    rejected(constraint(ConstraintType::FocalLength, 100.0, 1.0, ConstraintMode::Min));
    rejected(constraint(ConstraintType::FocalLength, 0.0));
    rejected(constraint(ConstraintType::ParaxialFocus, 0.0, 1.0, ConstraintMode::Max));
    rejected(constraint(ConstraintType::WorkingDistance, 1.0, -1.0));
    rejected(constraint(ConstraintType::WorkingDistance, std::nan("")));

    ParaxialConstraints ok;
    ok.items = {constraint(ConstraintType::ParfocalLength, 45.0),
                constraint(ConstraintType::ParaxialFocus, 0.0)};
    EXPECT_NO_THROW(mko::check_paraxial_constraints(ok, b.topology()));

    // Punktquelle: kein Brennpunkt F' als Ziel
    mko::SystemDescription sys = mko::read_system_file(kSystems + "einzellinse_bk7.json");
    sys.source = mko::Source{};
    sys.source.type = mko::SourceType::Point;
    sys.source.point = {0.0, 0.0, -50.0};
    sys.source.numerical_aperture = 0.1;
    const SystemBatchHost point = mko::make_system_batch(mko::prepare_system(sys, MKO_SOURCE_DIR));
    ParaxialConstraints focus;
    focus.items = {constraint(ConstraintType::ParaxialFocus, 0.0)};
    EXPECT_THROW(mko::check_paraxial_constraints(focus, point.topology()), std::invalid_argument);
}

// ---- Petzval (M4-T4) -------------------------------------------------------------------------

namespace {

// Zwei Linsen hintereinander: Blende, Linse 1 (n1; c1, c2), Luft, Linse 2 (n2; c3, c4), Bildraum
// mit Brechzahl n_img (1 = Luft).
struct TwoLenses {
    double n1, c1, c2, n2, c3, c4, n_img;

    // P = Σ c (n' − n)/(n n') über die Flächen
    double petzval_sum() const {
        return c1 * (n1 - 1.0) / n1 + c2 * (1.0 - n1) / n1 + c3 * (n2 - 1.0) / n2 +
               c4 * (n_img - n2) / (n2 * n_img);
    }
    // Für Linsen in Luft: Σ φ_i/n_i mit der Dünnlinsenbrechkraft φ_i = (n_i − 1)(c_a − c_b),
    // unabhängig von den Dicken.
    double thin_lens_sum() const {
        return (n1 - 1.0) * (c1 - c2) / n1 + (n2 - 1.0) * (c3 - c4) / n2;
    }
    double curvature() const { return -n_img * petzval_sum(); } // κ_P = −n'_L P
};

SystemBatchHost two_lens_batch(int variants) {
    BatchTopology t;
    t.surfaces = 5;
    t.wavelengths = 1;
    t.fields = 1;
    t.type = {SurfaceKind::Stop, SurfaceKind::Sphere, SurfaceKind::Sphere, SurfaceKind::Sphere,
              SurfaceKind::Sphere};
    SystemBatchHost b(t, variants);
    b.wavelength_nm(0) = 587.56;
    b.wavelength_weight(0) = 1.0;
    b.field_weight(0) = 1.0;
    b.reference_wavelength_nm = 587.56;
    return b;
}

void set_two_lenses(SystemBatchHost& b, int v, const TwoLenses& L, double thickness_scale = 1.0) {
    const double c[5] = {0.0, L.c1, L.c2, L.c3, L.c4};
    const double t[5] = {2.0, 4.0, 3.0, 2.0, 20.0};
    const double n[5] = {1.0, L.n1, 1.0, L.n2, L.n_img};
    for (int s = 0; s < 5; ++s) {
        b.curvature(v, s) = c[s];
        b.thickness(v, s) = thickness_scale * t[s];
        b.index(v, s, 0) = n[s];
        b.paraxial_index(v, s) = n[s];
    }
    b.pupil_radius(v) = 2.0;
}

ParaxialConstraint petzval(double kappa_max, double weight = 1.0) {
    return constraint(ConstraintType::Petzval, kappa_max, weight, ConstraintMode::Max);
}

} // namespace

// r = √w · max(0, |κ_P| − κ_max) mit κ_P = −n'_L P und der Petzvalsumme aus der geschlossenen
// Formel; für Linsen in Luft P = Σ φ_i/n_i (dünne Linsen), unabhängig von den Dicken.
TEST(ParaxialResiduals, PetzvalMatchesClosedForm) {
    const TwoLenses systems[] = {
        {1.8, 0.06, -0.01, 1.45, -0.05, 0.02, 1.0},   // Sammel- + Zerstreuungslinse
        {1.5, 0.02, -0.015, 1.7, 0.05, 0.01, 1.0},    // zwei Sammellinsen
        {1.6, -0.03, 0.02, 1.5, -0.04, 0.01, 1.0},    // zwei Zerstreuungslinsen (κ_P > 0)
        {1.7, 0.05, -0.02, 1.52, 0.03, -0.01, 1.33}}; // Bildraum Wasser: κ_P = −1,33 P
    SystemBatchHost b = two_lens_batch(8);
    for (int v = 0; v < 8; ++v) {
        set_two_lenses(b, v, systems[v % 4], v < 4 ? 1.0 : 2.5); // Dicken ändern nichts
    }
    constexpr double kWeight = 9.0;
    for (int v = 0; v < 8; ++v) {
        const TwoLenses& L = systems[v % 4];
        const double kappa = L.curvature();
        const ParaxialData p = mko::batch_paraxial(b, v);
        EXPECT_NEAR(p.petzval_sum, L.petzval_sum(), kParaxialRelative * std::abs(L.petzval_sum()))
            << v;
        EXPECT_NEAR(p.petzval_curvature, kappa, kParaxialRelative * std::abs(kappa)) << v;
        if (L.n_img == 1.0) {
            EXPECT_NEAR(L.petzval_sum(), L.thin_lens_sum(),
                        kParaxialRelative * std::abs(L.thin_lens_sum()))
                << v;
        }
        // aktiv: κ_max halb so groß wie |κ_P|
        const double kmax = 0.5 * std::abs(kappa);
        EXPECT_NEAR(residual(petzval(kmax, kWeight), b, v), 3.0 * (std::abs(kappa) - kmax),
                    3.0 * kParaxialRelative * std::abs(kappa))
            << v;
        // inaktiv: exakt 0
        EXPECT_EQ(residual(petzval(2.0 * std::abs(kappa), kWeight), b, v), 0.0) << v;
        // κ_max = 0: r = √w |κ_P|
        EXPECT_NEAR(residual(petzval(0.0, kWeight), b, v), 3.0 * std::abs(kappa),
                    3.0 * kParaxialRelative * std::abs(kappa))
            << v;
    }
    EXPECT_LT(mko::batch_paraxial(b, 0).petzval_curvature, 0.0); // Sammelsystem
    EXPECT_GT(mko::batch_paraxial(b, 2).petzval_curvature, 0.0); // Zerstreuungssystem
}

// Vorzeichen und Stetigkeit: |κ_P| geht ein (gespiegeltes System mit −κ_P liefert dasselbe
// Residuum), r ist stetig an der Grenze (0 genau dort, linear darüber) und mit w = 0 aus.
TEST(ParaxialResiduals, PetzvalOneSidedSignAndContinuity) {
    const TwoLenses pos{1.8, 0.06, -0.01, 1.45, -0.05, 0.02, 1.0};
    const TwoLenses neg{pos.n1, -pos.c1, -pos.c2, pos.n2, -pos.c3, -pos.c4, 1.0};
    SystemBatchHost b = two_lens_batch(2);
    set_two_lenses(b, 0, pos);
    set_two_lenses(b, 1, neg);
    const double k0 = mko::batch_paraxial(b, 0).petzval_curvature;
    const double k1 = mko::batch_paraxial(b, 1).petzval_curvature;
    ASSERT_NE(k0, 0.0);
    EXPECT_TRUE(same_bits(k1, -k0)); // alle Krümmungen negiert: P exakt negiert
    const double a = std::abs(k0);
    for (const double kmax : {0.0, 0.25 * a, 0.999 * a, a, 1.001 * a, 4.0 * a}) {
        const double r0 = residual(petzval(kmax, 2.0), b, 0);
        EXPECT_TRUE(same_bits(r0, residual(petzval(kmax, 2.0), b, 1))) << kmax;
        EXPECT_EQ(r0, std::sqrt(2.0) * std::max(0.0, a - kmax)) << kmax;
        EXPECT_GE(r0, 0.0);
        EXPECT_EQ(residual(petzval(kmax, 0.0), b, 0), 0.0);
    }
    // Stetig an der Grenze: r → 0, wenn κ_max von unten gegen |κ_P| geht
    for (const double eps : {1e-3, 1e-6, 1e-9}) {
        EXPECT_NEAR(residual(petzval(a * (1.0 - eps)), b, 0), a * eps, 1e-17) << eps;
    }
    EXPECT_EQ(residual(petzval(a), b, 0), 0.0);
}

// Geisterplatte im Medium n_m (Luft bzw. Wasser zwischen Blende und Bildebene): Beitrag
// φ_G/(n_m n_G) zur Petzvalsumme mit φ_G = −2 (n_G − n_m) t₂, κ_P = −n_m P. In der Systemdatei
// mit Linse und Geisterplatten addieren sich Linse und Σ φ_G/n_G.
TEST(ParaxialResiduals, PetzvalWithGhostPlate) {
    BatchTopology t;
    t.surfaces = 3;
    t.wavelengths = 1;
    t.fields = 1;
    t.ghost_plates = 1;
    t.ghost_terms = 3;
    t.type = {SurfaceKind::Stop, SurfaceKind::GhostPlate, SurfaceKind::Plane};
    SystemBatchHost b(t, 2);
    b.reference_wavelength_nm = 587.56;
    b.ghost_norm_radius(0) = 5.0;
    const double coeff[3] = {1e-3, -2e-4, 5e-5};
    const double t2 = (coeff[0] * 2.0 - coeff[1] * 6.0 + coeff[2] * 12.0) / 25.0;
    constexpr double kGlass = 1.5;
    const double medium[2] = {1.0, 1.33};
    for (int v = 0; v < 2; ++v) {
        for (int k = 0; k < 3; ++k) {
            b.ghost_coeff(v, 0, k) = coeff[k];
        }
        b.paraxial_ghost_index(v, 0) = kGlass;
        for (int s = 0; s < 3; ++s) {
            b.paraxial_index(v, s) = medium[v];
            b.thickness(v, s) = 2.0 + s;
        }
        b.pupil_radius(v) = 2.0;
    }
    for (int v = 0; v < 2; ++v) {
        const double nm = medium[v];
        const double phi_g = -2.0 * (kGlass - nm) * t2;
        const double P = phi_g / (nm * kGlass);
        const double kappa = -nm * P;
        const ParaxialData p = mko::batch_paraxial(b, v);
        EXPECT_NEAR(p.petzval_sum, P, kParaxialRelative * std::abs(P)) << v;
        EXPECT_NEAR(p.petzval_curvature, kappa, kParaxialRelative * std::abs(kappa)) << v;
        const double kmax = 0.1 * std::abs(kappa);
        EXPECT_NEAR(residual(petzval(kmax, 4.0), b, v), 2.0 * (std::abs(kappa) - kmax),
                    2.0 * kParaxialRelative * std::abs(kappa))
            << v;
    }

    // Systemdatei: Variante 0 mit den Koeffizienten der Datei, Variante 1 ohne (Linse allein)
    const mko::LoadedSystem ls = mko::load_system(kSystems + "einzellinse_geisterplatte.json");
    SystemBatchHost g = mko::make_system_batch(ls, 2);
    double ghost_sum = 0.0; // Σ φ_G/(n_m n_G) der Variante 0
    for (int p = 0; p < g.ghost_plates(); ++p) {
        std::vector<double> a(static_cast<std::size_t>(g.ghost_terms()));
        for (int k = 0; k < g.ghost_terms(); ++k) {
            a[static_cast<std::size_t>(k)] = g.ghost_coeff(0, p, k);
            g.ghost_coeff(1, p, k) = 0.0;
        }
        ghost_sum += mko::ghost_plate_petzval(a.data(), g.ghost_terms(), g.ghost_norm_radius(p),
                                              g.paraxial_ghost_index(0, p),
                                              g.paraxial_index(0, g.ghost_surface(p)));
    }
    ASSERT_NE(ghost_sum, 0.0);
    const ParaxialData with = mko::batch_paraxial(g, 0);
    const ParaxialData lens = mko::batch_paraxial(g, 1);
    EXPECT_NEAR(with.petzval_sum, lens.petzval_sum + ghost_sum,
                kParaxialRelative * std::abs(with.petzval_sum));
    const ParaxialConstraint c = petzval(0.0);
    EXPECT_EQ(residual(c, g, 0), std::abs(with.petzval_curvature));
    EXPECT_EQ(residual(c, g, 1), std::abs(lens.petzval_curvature));
    EXPECT_NE(residual(c, g, 0), residual(c, g, 1));
}

// Batch: Petzval-Residuen aller Varianten parallel = seriell bitgleich, gleich der Formel aus
// batch_paraxial; die Bedingung ist in einigen Varianten aktiv, in anderen nicht.
TEST(ParaxialResiduals, PetzvalParallelEqualsSerialBitwise) {
    const mko::LoadedSystem ls = mko::load_system(kSystems + "testsystem_mehrlinser.json");
    constexpr int kVariants = 131;
    const SystemBatchHost b = mko::test::perturbed_batch(ls, kVariants, 20261019);
    const double k0 = std::abs(mko::batch_paraxial(b, 0).petzval_curvature);
    ParaxialConstraints cs;
    cs.items = {petzval(k0, 250.0), constraint(ConstraintType::FocalLength, 20.6, 1e-2)};
    const mko::ExtraResiduals extra = mko::make_paraxial_residuals(cs);
    ASSERT_EQ(extra.count, 2);
    std::vector<double> serial(static_cast<std::size_t>(kVariants * 2));
    int active = 0;
    for (int v = 0; v < kVariants; ++v) {
        const auto i = static_cast<std::size_t>(v) * 2;
        mko::paraxial_residuals(cs, b, v, std::span<double>(serial).subspan(i, 2));
        const double kappa = std::abs(mko::batch_paraxial(b, v).petzval_curvature);
        EXPECT_EQ(serial[i], std::sqrt(250.0) * std::max(0.0, kappa - k0)) << v;
        active += serial[i] > 0.0 ? 1 : 0;
    }
    EXPECT_GT(active, 0);
    EXPECT_LT(active, kVariants);
    for (const int threads : {1, 2, 3, 8, 0}) {
        std::vector<double> parallel(serial.size(), -1.0);
        mko::evaluate_extra_residuals(extra, b, parallel, 0, -1, threads);
        EXPECT_TRUE(same_bits(parallel, serial)) << threads << " Threads";
    }
}

// Aus der Systemdatei: Grenze aus max_curvature, Modus Max, Gewicht; Residuum gegen
// compute_paraxial(system, catalog) bei λ_ref; Prüfung gegen die Topologie.
TEST(ParaxialResiduals, PetzvalFromFileAndChecks) {
    const mko::LoadedSystem ls = mko::load_system(kSystems + "zweilinser_petzval.json");
    const ParaxialConstraints cs = mko::make_paraxial_constraints(ls.system);
    ASSERT_EQ(cs.size(), 2);
    const ParaxialConstraint& c = cs.items[1];
    EXPECT_EQ(c.type, ConstraintType::Petzval);
    EXPECT_EQ(c.mode, ConstraintMode::Max);
    EXPECT_EQ(c.target, 0.01);
    EXPECT_EQ(c.weight, 1676769510.9367275); // 1.000 mm² · 1/(F·L·λ_ref²)
    EXPECT_EQ(c.item, 1);
    ParaxialConstraints only;
    only.items = {c};
    EXPECT_TRUE(only.needs_trace());

    const SystemBatchHost b = mko::make_system_batch(ls, 1);
    std::vector<double> r(2);
    mko::evaluate_extra_residuals(mko::make_paraxial_residuals(cs), b, r);
    const ParaxialData p = mko::compute_paraxial(ls.system, ls.catalog);
    EXPECT_NEAR(p.petzval_curvature, -0.02676, 1e-5); // Startwert laut Dateibeschreibung
    EXPECT_EQ(r[1], std::sqrt(c.weight) * (std::abs(p.petzval_curvature) - 0.01));

    const auto rejected = [&](const ParaxialConstraint& bad) {
        ParaxialConstraints one;
        one.items = {bad};
        EXPECT_THROW(mko::check_paraxial_constraints(one, b.topology()), std::invalid_argument);
    };
    rejected(constraint(ConstraintType::Petzval, 0.01, 1.0, ConstraintMode::Equal));
    rejected(constraint(ConstraintType::Petzval, 0.01, 1.0, ConstraintMode::Min));
    rejected(petzval(-0.01));
    rejected(petzval(std::nan("")));
    rejected(petzval(std::numeric_limits<double>::infinity()));
    rejected(petzval(0.01, -1.0));
    ParaxialConstraints ok;
    ok.items = {petzval(0.0)};
    EXPECT_NO_THROW(mko::check_paraxial_constraints(ok, b.topology()));
}
