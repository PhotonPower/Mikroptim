// Paraxialoptik (M1-T6): Orakel sind geschlossene Formeln (Linsenmacherformel, Hauptebenen,
// Abbildungsgleichung, Petzvalsumme pro Element) und der Grenzwert realer Strahlen für h -> 0.

#include "mikroptim/glass/index_table.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "mikroptim/paraxial/ynu_trace.hpp"
#include "mikroptim/trace/surface_ops.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <vector>

using mko::compute_paraxial;
using mko::GlassCatalog;
using mko::ParaxialData;
using mko::SystemDescription;
using mko::test::kAberrationScaling;
using mko::test::kDocumentedValue;
using mko::test::kLengthFp64;
using mko::test::kParaxialLimit;
using mko::test::kParaxialRelative;
using mko::test::kUnitFp64;

namespace {

namespace fs = std::filesystem;

// Flächendaten als Arrays (Fläche 0 = Blende).
struct Arrays {
    std::vector<double> c, t, n;
};

// Dicke Linse in Luft: Brechzahl n, Krümmungen c1, c2, Mittendicke t.
struct ThickLens {
    double n, c1, c2, t;

    // Linsenmacherformel 1/f = (n-1) [c1 - c2 + (n-1) t c1 c2 / n]
    double focal_length() const {
        return 1.0 / ((n - 1.0) * (c1 - c2 + (n - 1.0) * t * c1 * c2 / n));
    }
    // Lage der Hauptebenen relativ zum ersten bzw. zweiten Scheitel (+z):
    // h1 = -f (n-1) t c2 / n, h2 = -f (n-1) t c1 / n
    double h1() const { return -focal_length() * (n - 1.0) * t * c2 / n; }
    double h2() const { return -focal_length() * (n - 1.0) * t * c1 / n; }
    // Hintere Schnittweite BFL = f (1 - (n-1) t c1 / n) = f + h2
    double bfl() const { return focal_length() * (1.0 - (n - 1.0) * t * c1 / n); }
};

// Blende im Abstand stop_distance vor der Linse, Bildebene im Abstand image_distance dahinter.
Arrays lens_arrays(const ThickLens& L, double stop_distance, double image_distance) {
    return Arrays{{0.0, L.c1, L.c2}, {stop_distance, L.t, image_distance}, {1.0, L.n, 1.0}};
}

ParaxialData paraxial(const Arrays& a, double stop_semi_diameter = 1.0) {
    return compute_paraxial(a.c, a.t, a.n, stop_semi_diameter);
}

void expect_relative(double actual, double expected, double tol) {
    EXPECT_LE(std::abs(actual - expected), tol * std::abs(expected))
        << "ist " << actual << ", erwartet " << expected;
}

SystemDescription read_system(const char* name) {
    return mko::read_system_file(fs::path(MKO_SOURCE_DIR) / "data" / "systems" / name);
}

// Lokaler Helfer: alle Kataloge einer Systemdatei laden (keine öffentliche Ladefunktion).
GlassCatalog load_catalogs(const SystemDescription& system) {
    GlassCatalog cat;
    const fs::path base = fs::path(MKO_SOURCE_DIR) / "data" / "systems";
    for (const fs::path& p : mko::resolve_glass_catalogs(system, base)) {
        cat.merge(GlassCatalog::from_file(p));
    }
    return cat;
}

// Exakter achsparalleler meridionaler Strahl der Höhe h ab Fläche 0, mit den Flächen-
// operationen aus M0. Rückgabe: Achsschnittpunkt relativ zum Scheitel der letzten Fläche.
double real_axis_crossing(const Arrays& a, double h) {
    mko::Ray<double> r{{0.0, h, 0.0}, {0.0, 0.0, 1.0}, 0.0};
    double n_before = 1.0;
    for (std::size_t s = 0; s < a.c.size(); ++s) {
        if (s > 0) {
            mko::transfer_to_tangent_plane(r, a.t[s - 1]);
        }
        double path = 0.0;
        EXPECT_EQ(mko::intersect_sphere(r, a.c[s], path), mko::TraceStatus::Ok);
        EXPECT_EQ(mko::refract_sphere(r, a.c[s], n_before / a.n[s]), mko::TraceStatus::Ok);
        n_before = a.n[s];
    }
    return mko::meridional_axis_crossing(r);
}

const ThickLens kForms[] = {
    {1.5168, 0.02, -0.015, 5.0}, // bikonvex
    {1.6, 0.05, 0.02, 4.0},      // Meniskus, positiv
    {1.5168, 0.02, 0.0, 3.0},    // plankonvex
    {1.5168, 0.0, -0.02, 3.0},   // konvex-plan
    {1.7, -0.02, 0.025, 2.0},    // bikonkav
    {1.45, 0.01, 0.03, 6.0},     // Meniskus, negativ
    {1.8, 0.1, -0.1, 18.0},      // sehr dicke Kugellinse-ähnliche Form
};

} // namespace

// --- y-nu-Kern ---------------------------------------------------------------------------

TEST(ParaxialKernel, SingleSurfaceRefraction) {
    // n' u' = n u - y c (n' - n)
    mko::ParaxialRay<double> r{2.0, 0.1};
    mko::paraxial_refract(r, 0.05, 1.2, 1.6);
    EXPECT_NEAR(r.u, (1.2 * 0.1 - 2.0 * 0.05 * 0.4) / 1.6, kUnitFp64);
    EXPECT_EQ(r.y, 2.0);
    mko::paraxial_transfer(r, 10.0);
    EXPECT_NEAR(r.y, 2.0 + 10.0 * (1.2 * 0.1 - 2.0 * 0.05 * 0.4) / 1.6, kLengthFp64);
}

TEST(ParaxialKernel, PerSurfaceOutputMatchesFinalRay) {
    const Arrays a = lens_arrays(kForms[0], 7.0, 50.0);
    mko::ParaxialRay<double> at[3];
    const auto end = mko::paraxial_trace(a.c.data(), a.t.data(), a.n.data(), 3,
                                         mko::ParaxialRay<double>{1.5, 0.01}, at);
    EXPECT_EQ(at[2].y, end.y);
    EXPECT_EQ(at[2].u, end.u);
    EXPECT_EQ(at[0].y, 1.5); // Blende bricht nicht
    EXPECT_EQ(at[0].u, 0.01);
    EXPECT_NEAR(at[1].y, 1.5 + 7.0 * 0.01, kLengthFp64);
}

// --- Brennweite, Schnittweite, Hauptebenen -------------------------------------------------

TEST(Paraxial, ThickLensMatchesLensmakerFormula) {
    for (const ThickLens& L : kForms) {
        SCOPED_TRACE(::testing::Message()
                     << "n=" << L.n << " c1=" << L.c1 << " c2=" << L.c2 << " t=" << L.t);
        const double t0 = 7.0;
        const double t_image = 50.0;
        const ParaxialData p = paraxial(lens_arrays(L, t0, t_image));
        const double f = L.focal_length();
        ASSERT_FALSE(p.afocal);
        expect_relative(p.focal_length, f, kParaxialRelative);
        expect_relative(p.power, 1.0 / f, kParaxialRelative);
        EXPECT_NEAR(p.back_focal_distance, L.bfl(), kLengthFp64);
        EXPECT_NEAR(p.rear_principal_plane, L.h2(), kLengthFp64);
        EXPECT_NEAR(p.front_principal_plane, t0 + L.h1(), kLengthFp64);
        // F liegt um f vor H (Luft auf beiden Seiten)
        EXPECT_NEAR(p.front_focal_distance, t0 + L.h1() - f, kLengthFp64);
        EXPECT_NEAR(p.image_defocus, L.bfl() - t_image, kLengthFp64);
        // reduzierte Winkel: det M = 1
        const auto [A, B, C, D] = p.abcd;
        EXPECT_NEAR(A * D - B * C, 1.0, kUnitFp64);
    }
}

TEST(Paraxial, StopDistanceDoesNotChangeFocalData) {
    const ThickLens L = kForms[1];
    const ParaxialData p1 = paraxial(lens_arrays(L, 1.0, 20.0));
    const ParaxialData p2 = paraxial(lens_arrays(L, 30.0, 20.0));
    EXPECT_EQ(p1.focal_length, p2.focal_length);
    EXPECT_EQ(p1.back_focal_distance, p2.back_focal_distance);
    EXPECT_NEAR(p2.front_principal_plane - p1.front_principal_plane, 29.0, kLengthFp64);
}

TEST(Paraxial, ImageSpaceMediumScalesRearFocalLength) {
    // Bildraum mit n' = 1,33: EFL = 1/phi, der hintere Brennpunkt liegt n' f hinter H'.
    const double n_image = 1.33;
    const Arrays a{{0.0, 0.02, -0.01}, {5.0, 4.0, 30.0}, {1.0, 1.6, n_image}};
    const ParaxialData p = paraxial(a);
    const double phi = 0.02 * 0.6 + (-0.01) * (n_image - 1.6) -
                       4.0 / 1.6 * (0.02 * 0.6) * (-0.01) * (n_image - 1.6);
    expect_relative(p.focal_length, 1.0 / phi, kParaxialRelative);
    EXPECT_NEAR(p.back_focal_distance - p.rear_principal_plane, n_image / phi, kLengthFp64);
    EXPECT_NEAR(p.front_principal_plane - p.front_focal_distance, 1.0 / phi, kLengthFp64);
}

TEST(Paraxial, EinzellinseBk7File) {
    const SystemDescription sys = read_system("einzellinse_bk7.json");
    const GlassCatalog cat = load_catalogs(sys);
    const ParaxialData p = compute_paraxial(sys, cat);

    const double n_d = cat.at("N-BK7").index(sys.spectrum.reference_nm);
    const ThickLens L{n_d, sys.surfaces[1].curvature, sys.surfaces[2].curvature,
                      sys.surfaces[1].thickness};
    expect_relative(p.focal_length, L.focal_length(), kParaxialRelative);
    EXPECT_NEAR(p.back_focal_distance, L.bfl(), kLengthFp64);
    EXPECT_NEAR(p.image_defocus, L.bfl() - sys.surfaces[2].thickness, kLengthFp64);
    // gerundete Werte aus dem Plan
    EXPECT_NEAR(p.focal_length, 101.05, kDocumentedValue);
    EXPECT_NEAR(p.back_focal_distance, 99.05, kDocumentedValue);
    EXPECT_NEAR(p.image_defocus, 0.0, kDocumentedValue);
}

TEST(Paraxial, WavelengthSelection) {
    const SystemDescription sys = read_system("einzellinse_bk7.json");
    const GlassCatalog cat = load_catalogs(sys);
    std::vector<mko::MaterialSpec> media;
    for (const auto& s : sys.surfaces) {
        media.push_back(s.material);
    }
    const mko::IndexTable table = mko::make_index_table(media, sys.spectrum.wavelengths_nm, cat);
    // reference_nm = wavelengths_nm[1]: beide Wege liefern bitgleiche Ergebnisse
    const ParaxialData ref = compute_paraxial(sys, cat);
    const ParaxialData tab = compute_paraxial(sys, table, 1);
    EXPECT_EQ(ref.focal_length, tab.focal_length);
    EXPECT_EQ(ref.back_focal_distance, tab.back_focal_distance);
    EXPECT_EQ(ref.petzval_sum, tab.petzval_sum);
    const ParaxialData blue = compute_paraxial(sys, table, 0);
    EXPECT_EQ(blue.focal_length, compute_paraxial(sys, cat, 486.13).focal_length);
    // normale Dispersion: positive Linse ist im Blauen kürzer
    EXPECT_LT(blue.focal_length, ref.focal_length);
    EXPECT_LT(ref.focal_length, compute_paraxial(sys, table, 2).focal_length);
}

TEST(Paraxial, PlaneParallelPlateIsAfocal) {
    const SystemDescription sys = read_system("planplatte_bk7.json");
    const GlassCatalog cat = load_catalogs(sys);
    const ParaxialData p = compute_paraxial(sys, cat);
    const double n = cat.at("N-BK7").index(sys.spectrum.reference_nm);
    EXPECT_TRUE(p.afocal);
    EXPECT_EQ(p.power, 0.0);
    EXPECT_FALSE(std::signbit(p.power));
    EXPECT_EQ(p.focal_length, std::numeric_limits<double>::infinity());
    EXPECT_TRUE(std::isnan(p.back_focal_distance));
    EXPECT_TRUE(std::isnan(p.front_focal_distance));
    EXPECT_TRUE(std::isnan(p.front_principal_plane));
    EXPECT_TRUE(std::isnan(p.rear_principal_plane));
    EXPECT_TRUE(std::isnan(p.image_defocus));
    EXPECT_EQ(p.petzval_sum, 0.0);
    EXPECT_EQ(p.petzval_radius, std::numeric_limits<double>::infinity());
    // Blendenbild durch die Platte: scheinbarer Abstand 2 + 10/n vor der letzten Fläche
    ASSERT_FALSE(p.exit_pupil_at_infinity);
    EXPECT_NEAR(p.exit_pupil_position, -(2.0 + 10.0 / n), kLengthFp64);
    EXPECT_NEAR(p.exit_pupil_magnification, 1.0, kUnitFp64);
    EXPECT_NEAR(p.exit_pupil_semi_diameter, sys.surfaces[0].semi_diameter, kLengthFp64);
}

// --- Petzval -------------------------------------------------------------------------------

TEST(Paraxial, PetzvalSumOfMultiLensSystem) {
    const double n1 = 1.6, n2 = 1.5, n3 = 1.75, n4 = 1.45;
    // Blende, Einzellinse, Kittglied, negative Einzellinse
    Arrays a{{0.0, 0.02, -0.005, 0.03, -0.04, -0.01, -0.03, 0.01},
             {10.0, 5.0, 3.0, 6.0, 2.0, 8.0, 2.0, 40.0},
             {1.0, n1, 1.0, n2, n3, 1.0, n4, 1.0}};
    // pro Element: Einzellinse in Luft (n-1)(c1-c2)/n, Kittglied Fläche für Fläche
    const double singlet1 = (n1 - 1.0) * (0.02 - -0.005) / n1;
    const double doublet =
        0.03 * (n2 - 1.0) / n2 + -0.04 * (n3 - n2) / (n2 * n3) + -0.01 * (1.0 - n3) / n3;
    const double singlet2 = (n4 - 1.0) * (-0.03 - 0.01) / n4;
    const double expected = singlet1 + doublet + singlet2;
    const double scale = std::abs(singlet1) + std::abs(doublet) + std::abs(singlet2);

    const ParaxialData p = paraxial(a);
    EXPECT_NEAR(p.petzval_sum, expected, kParaxialRelative * scale);
    EXPECT_EQ(p.petzval_curvature, -p.petzval_sum); // Bildraum Luft
    EXPECT_EQ(p.petzval_radius, 1.0 / p.petzval_curvature);

    // P hängt nicht von Dicken und Abständen ab (bitgleich, gleiche Summationsreihenfolge)
    for (double& t : a.t) {
        t *= 1.7;
    }
    EXPECT_EQ(paraxial(a).petzval_sum, p.petzval_sum);

    // Bildraum Immersion (n' = 1,33): letzter Flächenbeitrag ändert sich, kappa_P = -n' P
    a.n.back() = 1.33;
    const double expected_imm =
        singlet1 + doublet + (n4 - 1.0) * (-0.03) / n4 + 0.01 * (1.33 - n4) / (n4 * 1.33);
    const ParaxialData q = paraxial(a);
    EXPECT_NEAR(q.petzval_sum, expected_imm, kParaxialRelative * scale);
    EXPECT_EQ(q.petzval_curvature, -1.33 * q.petzval_sum);
}

TEST(Paraxial, ThinLensPetzvalSumIsOneOverNf) {
    for (const ThickLens& form : kForms) {
        ThickLens L = form;
        L.t = 0.0;
        const ParaxialData p = paraxial(lens_arrays(L, 3.0, 10.0));
        expect_relative(p.petzval_sum * L.n * p.focal_length, 1.0, kParaxialRelative);
        // Petzvalschale einer positiven Linse ist zur Linse hin gekrümmt (kappa < 0)
        EXPECT_EQ(std::signbit(p.petzval_curvature), p.focal_length > 0.0);
    }
}

// --- Grenzwert realer Strahlen -------------------------------------------------------------

TEST(Paraxial, RealRaysConvergeToParaxialFocus) {
    const SystemDescription sys = read_system("einzellinse_bk7.json");
    const GlassCatalog cat = load_catalogs(sys);
    const ParaxialData p = compute_paraxial(sys, cat);
    const double n = cat.at("N-BK7").index(sys.spectrum.reference_nm);
    Arrays a{{0.0, sys.surfaces[1].curvature, sys.surfaces[2].curvature},
             {sys.surfaces[0].thickness, sys.surfaces[1].thickness, sys.surfaces[2].thickness},
             {1.0, n, 1.0}};

    // Längsaberration L(h) - s' = k h² + O(h⁴): Verhältnis bei h und h/2 gegen 4
    double previous = real_axis_crossing(a, 1.0) - p.back_focal_distance;
    EXPECT_LT(previous, 0.0); // unterkorrigierte sphärische Aberration einer Sammellinse
    for (const double h : {0.5, 0.25, 0.125}) {
        const double diff = real_axis_crossing(a, h) - p.back_focal_distance;
        EXPECT_NEAR(previous / diff / 4.0, 1.0, kAberrationScaling) << "h = " << h;
        previous = diff;
    }
    EXPECT_LT(std::abs(real_axis_crossing(a, 1e-4) - p.back_focal_distance), kParaxialLimit);
}

// --- Austrittspupille ----------------------------------------------------------------------

TEST(Paraxial, ExitPupilFollowsImagingEquation) {
    for (const ThickLens& L : {kForms[0], kForms[1], kForms[4]}) {
        for (const double t0 : {5.0, 15.0, 80.0, 150.0}) {
            SCOPED_TRACE(::testing::Message() << "c1=" << L.c1 << " t0=" << t0);
            const double a = 2.5;
            const ParaxialData p = paraxial(lens_arrays(L, t0, 40.0), a);
            // Gaußsche Abbildungsgleichung 1/s' - 1/s = 1/f, Gegenstand (Blende) bei
            // s = -(t0 + h1) vor H, Bild s' hinter H'
            const double f = L.focal_length();
            const double s = -(t0 + L.h1());
            const double s_img = 1.0 / (1.0 / f + 1.0 / s);
            ASSERT_FALSE(p.exit_pupil_at_infinity);
            EXPECT_NEAR(p.exit_pupil_position, L.h2() + s_img, kLengthFp64);
            expect_relative(p.exit_pupil_magnification, s_img / s, kParaxialRelative);
            EXPECT_NEAR(p.exit_pupil_semi_diameter, a * std::abs(s_img / s), kLengthFp64);
        }
    }
}

TEST(Paraxial, EinzellinseExitPupil) {
    const SystemDescription sys = read_system("einzellinse_bk7.json");
    const GlassCatalog cat = load_catalogs(sys);
    const ParaxialData p = compute_paraxial(sys, cat);
    const ThickLens L{cat.at("N-BK7").index(sys.spectrum.reference_nm), sys.surfaces[1].curvature,
                      sys.surfaces[2].curvature, sys.surfaces[1].thickness};
    const double s = -(sys.surfaces[0].thickness + L.h1());
    const double s_img = 1.0 / (1.0 / L.focal_length() + 1.0 / s);
    EXPECT_NEAR(p.exit_pupil_position, L.h2() + s_img, kLengthFp64);
    EXPECT_NEAR(p.exit_pupil_semi_diameter, sys.surfaces[0].semi_diameter * s_img / s, kLengthFp64);
    EXPECT_GT(p.exit_pupil_magnification, 1.0); // virtuelles, vergrößertes Blendenbild
}

TEST(Paraxial, StopInFrontFocusGivesTelecentricImageSpace) {
    // n = 2, c = ±0,25, t = 0: alle Zwischenwerte exakt darstellbar, f = 2 = Blendenabstand
    const Arrays a{{0.0, 0.25, -0.25}, {2.0, 0.0, 4.0}, {1.0, 2.0, 1.0}};
    const ParaxialData p = paraxial(a, 1.0);
    EXPECT_EQ(p.focal_length, 2.0);
    EXPECT_EQ(p.front_focal_distance, 0.0); // F liegt in der Blende
    EXPECT_TRUE(p.exit_pupil_at_infinity);
    EXPECT_TRUE(std::isnan(p.exit_pupil_position));
    EXPECT_TRUE(std::isnan(p.exit_pupil_magnification));
    EXPECT_EQ(p.exit_pupil_semi_diameter, std::numeric_limits<double>::infinity());
}

// --- Fehlerbehandlung ----------------------------------------------------------------------

TEST(Paraxial, RejectsInvalidArrays) {
    const std::vector<double> c{0.0, 0.01}, t{1.0, 2.0}, n{1.0, 1.5};
    const std::vector<double> short_t{1.0};
    const std::vector<double> empty;
    const std::vector<double> bad_n{1.0, 0.0};
    const std::vector<double> nan_c{0.0, std::numeric_limits<double>::quiet_NaN()};
    EXPECT_THROW((void)compute_paraxial(empty, empty, empty, 1.0), std::invalid_argument);
    EXPECT_THROW((void)compute_paraxial(c, short_t, n, 1.0), std::invalid_argument);
    EXPECT_THROW((void)compute_paraxial(c, t, bad_n, 1.0), std::invalid_argument);
    EXPECT_THROW((void)compute_paraxial(nan_c, t, n, 1.0), std::invalid_argument);
    EXPECT_THROW((void)compute_paraxial(c, t, n, 0.0), std::invalid_argument);
    EXPECT_NO_THROW((void)compute_paraxial(c, t, n, 1.0));
}

TEST(Paraxial, RejectsMismatchedIndexTableAndWavelength) {
    const SystemDescription sys = read_system("einzellinse_bk7.json");
    const GlassCatalog cat = load_catalogs(sys);
    std::vector<mko::MaterialSpec> media;
    for (const auto& s : sys.surfaces) {
        media.push_back(s.material);
    }
    const mko::IndexTable table = mko::make_index_table(media, sys.spectrum.wavelengths_nm, cat);
    EXPECT_THROW((void)compute_paraxial(sys, table, 3), std::out_of_range);
    EXPECT_THROW((void)compute_paraxial(sys, table, -1), std::out_of_range);
    media.pop_back();
    const mko::IndexTable short_table =
        mko::make_index_table(media, sys.spectrum.wavelengths_nm, cat);
    EXPECT_THROW((void)compute_paraxial(sys, short_table, 0), std::invalid_argument);
    // N-BK7 ist unterhalb von 300 nm nicht definiert
    EXPECT_THROW((void)compute_paraxial(sys, cat, 200.0), mko::GlassError);
}
