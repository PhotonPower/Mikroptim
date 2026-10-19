// Vektorielle PSF (M8-T5, Richards-Wolf) auf der CPU: Orakel (Besselfunktionen), Geometrie der
// Polarisationsdrehung, Eingangsfelder, und die Abnahmeprüfungen aus vector_psf_test_systems.hpp
// (Grenzfall kleiner NA gegen die skalare PSF, hohe NA gegen die Quadratur des
// Richards-Wolf-Integrals, Defokus, Energie, Symmetrien, Strehl bei kleinen Aberrationen,
// aplanatischer Punkt über EVAL).

#include "eval_test_systems.hpp"
#include "mikroptim/eval/eval.hpp"
#include "mikroptim/eval/psf.hpp"
#include "mikroptim/eval/vector_psf.hpp"
#include "mikroptim/eval/vector_psf_ops.hpp"
#include "psf_test_systems.hpp"
#include "tolerances.hpp"
#include "vector_psf_test_systems.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

using mko::Polarization;
using mko::VectorPsfPupil;
using mko::VectorPsfResult;
using mko::VectorPsfSettings;

namespace {

mko::test::EvalFn cpu_eval() {
    return [](const mko::SystemBatchHost& b, const mko::EvalSettings& s) {
        return mko::eval_cpu(b, s);
    };
}

} // namespace

TEST(VectorPsfOracle, BesselAgainstKnownValues) {
    // Abramowitz und Stegun, Tafel 9.1
    EXPECT_NEAR(mko::test::bessel_jn(0, 1.0), 0.7651976865579666, 1e-15);
    EXPECT_NEAR(mko::test::bessel_jn(1, 1.0), 0.4400505857449335, 1e-15);
    EXPECT_NEAR(mko::test::bessel_jn(2, 1.0), 0.1149034849319005, 1e-15);
    EXPECT_NEAR(mko::test::bessel_jn(0, 10.0), -0.2459357644513483, 1e-15);
    EXPECT_NEAR(mko::test::bessel_jn(1, 10.0), 0.04347274616886144, 1e-15);
    EXPECT_NEAR(mko::test::bessel_jn(2, 10.0), 0.2546303136851206, 1e-15);
    // gegen die Potenzreihe aus M8-T2 (die bei x ≈ 12 durch Auslöschung etwa 3 Stellen verliert)
    for (double x = 0.0; x <= 12.0; x += 0.37) {
        EXPECT_NEAR(mko::test::bessel_jn(1, x), mko::test::bessel_j1(x), 1e-12) << x;
    }
    EXPECT_NEAR(mko::test::bessel_jn(1, mko::kBesselJ1FirstZero), 0.0, 1e-15);
}

TEST(VectorPsfOps, BasisIsRotationOntoRayDirection) {
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    for (int q = 0; q < 200; ++q) {
        mko::VectorPsfParams p{};
        p.lambda_mm = 5e-4;
        p.scale = 1.0;
        p.index = q % 2 == 0 ? 1.0 : 1.33;
        p.a_x = -0.7 * p.index;
        p.a_y = 0.6 * p.index;
        p.chief_x = 0.1 * uni(rng);
        p.chief_y = 0.2 * uni(rng);
        const double theta = q < 100 ? 0.0 : 0.1 * uni(rng);
        p.input_y = std::sin(theta);
        p.input_z = std::cos(theta);
        p.defocus = 1e-3 * uni(rng);
        const double x = uni(rng) * 0.7;
        const double y = uni(rng) * 0.7;
        double ex[3];
        double ey[3];
        double w = 0.0;
        double dz = 0.0;
        ASSERT_TRUE(mko::vector_psf_basis(p, x, y, ex, ey, w, dz));
        const double sx = p.chief_x + p.a_x * x / p.index;
        const double sy = p.chief_y + p.a_y * y / p.index;
        const double sz = std::sqrt(1.0 - sx * sx - sy * sy);
        // orthonormal, rechtshändig mit e_x × e_y = s (R u = s)
        EXPECT_NEAR(ex[0] * ex[0] + ex[1] * ex[1] + ex[2] * ex[2], 1.0, 1e-15);
        EXPECT_NEAR(ey[0] * ey[0] + ey[1] * ey[1] + ey[2] * ey[2], 1.0, 1e-15);
        EXPECT_NEAR(ex[0] * ey[0] + ex[1] * ey[1] + ex[2] * ey[2], 0.0, 1e-15);
        EXPECT_NEAR(ex[1] * ey[2] - ex[2] * ey[1], sx, 1e-15);
        EXPECT_NEAR(ex[2] * ey[0] - ex[0] * ey[2], sy, 1e-15);
        EXPECT_NEAR(ex[0] * ey[1] - ex[1] * ey[0], sz, 1e-15);
        EXPECT_NEAR(w, std::pow(1.0 - sx * sx - sy * sy, -0.25), 1e-15);
        EXPECT_NEAR(dz, p.index * p.defocus * (sz - 1.0), 1e-17);
        if (theta == 0.0) {
            // Richards-Wolf-Matrix für u = z
            EXPECT_NEAR(ex[0], 1.0 - sx * sx / (1.0 + sz), 1e-15);
            EXPECT_NEAR(ex[1], -sx * sy / (1.0 + sz), 1e-15);
            EXPECT_NEAR(ex[2], -sx, 1e-15);
            EXPECT_NEAR(ey[0], -sx * sy / (1.0 + sz), 1e-15);
            EXPECT_NEAR(ey[1], 1.0 - sy * sy / (1.0 + sz), 1e-15);
            EXPECT_NEAR(ey[2], -sy, 1e-15);
        }
    }
    // nicht ausbreitungsfähig
    mko::VectorPsfParams p{};
    p.index = 1.0;
    p.a_x = -0.95;
    p.a_y = -0.95;
    p.chief_y = 0.2;
    p.input_z = 1.0;
    double ex[3];
    double ey[3];
    double w = 0.0;
    double dz = 0.0;
    EXPECT_FALSE(mko::vector_psf_basis(p, 0.0, -1.0, ex, ey, w, dz));
    double re[3];
    double im[3];
    const mko::VectorPsfJones j{{1.0, 0.0}, {0.0, 0.0}, 1.0};
    EXPECT_FALSE(mko::vector_psf_pupil_value(p, j, 0.0, -1.0, 0.0, re, im));
    EXPECT_FALSE(mko::vector_psf_pupil_value(p, j, 0.0, 0.0, mko::PsfResult::kNaN, re, im));
    EXPECT_EQ(re[0], 0.0);
}

TEST(VectorPsf, InputsSettingsAndErrors) {
    for (const Polarization p :
         {Polarization::X, Polarization::Y, Polarization::Circular, Polarization::Unpolarized}) {
        EXPECT_EQ(mko::parse_polarization(mko::polarization_name(p)), p);
        const mko::VectorPsfInputs in = mko::vector_psf_inputs(p);
        double power = 0.0;
        for (int k = 0; k < in.count; ++k) {
            const mko::VectorPsfJones& j = in.jones[k];
            power += j.weight * (j.re[0] * j.re[0] + j.im[0] * j.im[0] + j.re[1] * j.re[1] +
                                 j.im[1] * j.im[1]);
        }
        EXPECT_NEAR(power, 1.0, 1e-15);
    }
    EXPECT_EQ(mko::vector_psf_inputs(Polarization::Unpolarized).count, 2);
    EXPECT_THROW((void)mko::parse_polarization("z"), std::invalid_argument);
    VectorPsfSettings s;
    s.defocus = std::nan("");
    EXPECT_THROW(mko::check_vector_psf_settings(s), std::invalid_argument);
    VectorPsfPupil vp =
        mko::test::synthetic_vector_pupil(16, 500.0, 0.5, [](double, double) { return 0.0; });
    EXPECT_THROW((void)mko::vector_psf_cpu(vp, 16), std::invalid_argument); // M < 2N
    VectorPsfPupil bad = vp;
    bad.input_y = 0.5;
    EXPECT_THROW((void)mko::vector_psf_cpu(bad, 32), std::invalid_argument);
    bad = vp;
    bad.chief_y = 1.0;
    EXPECT_THROW((void)mko::vector_psf_cpu(bad, 32), std::invalid_argument);
    bad = vp;
    bad.pupil.na_x = 0.0;
    EXPECT_THROW((void)mko::vector_psf_cpu(bad, 32), std::invalid_argument);
    bad = vp;
    bad.index = 0.0;
    EXPECT_THROW((void)mko::vector_psf_cpu(bad, 32), std::invalid_argument);
    EXPECT_THROW((void)mko::make_vector_psf_device_layout(16, 32, 3, 1), std::invalid_argument);
    EXPECT_GT(mko::make_vector_psf_device_layout(16, 32, 2, 3).total_bytes, 0u);
}

TEST(VectorPsf, EvanescentPointsAreExcluded) {
    // Hauptstrahl geneigt: ein Teil der Pupille liegt jenseits |s_⊥| = 1 und zählt nicht.
    VectorPsfPupil vp =
        mko::test::synthetic_vector_pupil(32, 500.0, 0.95, [](double, double) { return 0.0; });
    vp.chief_y = 0.2;
    const VectorPsfResult r = mko::vector_psf_cpu(vp, 64);
    EXPECT_GT(r.evanescent, 0);
    EXPECT_EQ(r.transmitted + r.evanescent, vp.pupil.full_count);
    EXPECT_NEAR(r.energy, 1.0, mko::test::kVectorPsfIdentity);
    EXPECT_NEAR(r.strehl_origin, 1.0, mko::test::kVectorPsfIdentity);
}

TEST(VectorPsf, SmallNaApproachesScalarPsf) {
    mko::test::check_vector_scalar_limit(mko::test::vector_psf_cpu_all, "CPU");
}

TEST(VectorPsf, HighNaMatchesRichardsWolfQuadrature) {
    mko::test::check_vector_richards_wolf(mko::test::vector_psf_cpu_all, "CPU");
}

TEST(VectorPsf, DefocusCompensatesPupilPhase) {
    mko::test::check_vector_defocus_sign(mko::test::vector_psf_cpu_all, "CPU");
}

TEST(VectorPsf, ParsevalAndDirectEvaluation) {
    mko::test::check_vector_parseval(mko::test::vector_psf_cpu_all, "CPU");
}

TEST(VectorPsf, SymmetryAndPolarizationRotation) {
    mko::test::check_vector_symmetry(mko::test::vector_psf_cpu_all, "CPU");
}

TEST(VectorPsf, StrehlFollowsWeightedMarechal) {
    mko::test::check_vector_marechal(mko::test::vector_psf_cpu_all, "CPU");
}

TEST(VectorPsf, AplanaticPointThroughEvalMatchesQuadrature) {
    mko::test::check_vector_aplanatic(cpu_eval(), mko::test::vector_psf_cpu_all, "CPU");
}

TEST(VectorPsf, PupilFromEvalWithFieldAngle) {
    // Einzellinse mit Feld 1°: Eingangsbündel (0, sin θ_f, cos θ_f), Hauptstrahlrichtung aus dem
    // Feldstrahl, n' = 1; alle Tupel rechenbar.
    const mko::test::LensCase lc = mko::test::lens_case(2.0, {0.0, 1.0});
    mko::EvalSettings s;
    s.grid = 32;
    const mko::EvalResult er = mko::eval_cpu(lc.batch, s);
    const std::vector<VectorPsfPupil> pupils = mko::make_vector_psf_pupils(lc.batch, er, 1);
    ASSERT_EQ(pupils.size(), static_cast<std::size_t>(2 * lc.batch.wavelengths()));
    const VectorPsfPupil& axis = pupils[0];
    const VectorPsfPupil& field = pupils[static_cast<std::size_t>(lc.batch.wavelengths())];
    EXPECT_EQ(axis.input_y, 0.0);
    EXPECT_EQ(axis.chief_y, er.field_ray(1, 0, 0, mko::kEvalChiefRay, 3));
    EXPECT_NEAR(field.input_y, std::sin(lc.batch.field_angle(1, 1)), 1e-15);
    EXPECT_EQ(field.chief_y, er.field_ray(1, 1, 0, mko::kEvalChiefRay, 3));
    EXPECT_GT(field.chief_y, 0.0);
    EXPECT_EQ(field.index, lc.batch.index(1, lc.batch.surfaces() - 1, 0));
    mko::PsfSettings ps;
    ps.padding = 2.0;
    const std::vector<VectorPsfResult> r =
        mko::vector_psf_cpu(lc.batch, er, 1, ps, mko::test::polarization_settings(Polarization::Y));
    ASSERT_EQ(r.size(), pupils.size());
    for (const VectorPsfResult& t : r) {
        EXPECT_GT(t.strehl, 0.0);
        EXPECT_LE(t.peak, 1.0 + 1e-12);
        EXPECT_NEAR(t.energy, 1.0, mko::test::kVectorPsfIdentity);
        EXPECT_EQ(t.polarization, Polarization::Y);
    }
}
