// Abzug von Kolben und Kippung (M8-T4, Kippungsabzug der toleranzbewussten Merit M11-T6b):
// PistonTiltFit gegen exakt lineare Daten, TiltProjection der Strahlresiduen gegen den Rest der
// Tolerierung (tuple_quality) an einem dezentrierten Mehrlinser (drei Felder, vier Wellenlängen,
// volle Pupille), Projektion idempotent und orthogonal zu √W (1, x, y), Achsfall unverändert.

#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/tolerance/piston_tilt.hpp"
#include "mikroptim/tolerance/tolerance_analysis.hpp"
#include "mikroptim/trace/surface_ops.hpp"
#include "residual_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

// Exakt lineare Daten v = c₀ + t_x x + t_y y werden exakt angepasst; bei einer Speiche (x = 0)
// nur der Kolben (Mittel), wie in tuple_quality.
TEST(PistonTiltFit, SolvesLinearDataAndFallsBackToPiston) {
    const mko::PupilQuadrature<double> q =
        mko::make_pupil_quadrature<double>(10, 8, mko::PupilMode::Full);
    const int n = mko::sample_count(q);
    std::vector<std::uint8_t> use(static_cast<std::size_t>(n), 1);
    use[3] = 0; // eine Stützstelle ausgeschlossen
    const mko::PistonTiltFit fit(q, use);
    ASSERT_TRUE(fit.valid());
    ASSERT_TRUE(fit.tilt());
    double b0 = 0.0, b1 = 0.0, b2 = 0.0;
    for (int k = 0; k < n; ++k) {
        if (!fit.used(k)) {
            continue;
        }
        const double v = 0.7 + 2.5 * fit.x(k) - 1.25 * fit.y(k);
        b0 += fit.weight(k) * v;
        b1 += fit.weight(k) * (fit.x(k) * v);
        b2 += fit.weight(k) * (fit.y(k) * v);
    }
    const std::array<double, 3> c = fit.solve(b0, b1, b2);
    EXPECT_NEAR(c[0], 0.7, 1e-14);
    EXPECT_NEAR(c[1], 2.5, 1e-14);
    EXPECT_NEAR(c[2], -1.25, 1e-14);
    // Achse: eine Speiche, x = 0 → nur der Kolben
    const mko::PupilQuadrature<double> axis =
        mko::make_pupil_quadrature<double>(10, 1, mko::PupilMode::Axis);
    const std::vector<std::uint8_t> all(static_cast<std::size_t>(mko::sample_count(axis)), 1);
    const mko::PistonTiltFit a(axis, all);
    EXPECT_TRUE(a.valid());
    EXPECT_FALSE(a.tilt());
    const std::array<double, 3> p = a.solve(2.0 * a.fit_weight(), 1.0, 1.0);
    EXPECT_EQ(p[0], 2.0);
    EXPECT_EQ(p[1], 0.0);
    EXPECT_EQ(p[2], 0.0);
    EXPECT_THROW(mko::PistonTiltFit(q, std::span<const std::uint8_t>(all)), std::invalid_argument);
}

// Die Projektion der Strahlresiduen ist der Rest der Tolerierung: P r = s √W e mit e aus
// tuple_quality (gleiche Anpassung über die gültigen Strahlen), also Σ (P r)² = Σ ŵ ŵ (RMS/λ)²
// = Q der Tolerierung und der Empfindlichkeiten. Hart und weich, ohne Versagen K = Ok.
TEST(TiltProjection, MatchesToleranceQualityRest) {
    const mko::LoadedSystem loaded =
        mko::test::load_residual_system("testsystem_mehrlinser_dezentriert.json");
    const mko::SystemBatchHost batch = mko::make_non_coaxial(mko::make_system_batch(loaded, 1));
    const std::vector<double> tuple_weight = mko::tolerance_tuple_weights(batch);
    const double lambda_ref = batch.reference_wavelength_nm * 1e-6;
    double worst_rest = 0.0;
    double worst_q = 0.0;
    double worst_idem = 0.0;
    double worst_orth = 0.0;
    for (const mko::FailureMode mode : {mko::FailureMode::Hard, mko::FailureMode::Soft}) {
        mko::ResidualSettings rs;
        rs.failure = mode;
        const mko::ResidualLayout layout(batch.topology(), rs);
        const mko::ResidualResult rr = mko::residuals_cpu(batch, 0, layout);
        ASSERT_EQ(rr.failed_count(), 0U);
        std::vector<double> r(rr.residuals.begin(), rr.residuals.end());
        const mko::TiltProjection proj(layout, rr.status);
        proj.apply(r);
        // Strafresiduen (weich) unverändert
        for (std::int64_t i = layout.rays_per_variant(); i < layout.trace_rows(); ++i) {
            EXPECT_EQ(r[static_cast<std::size_t>(i)], rr.residuals[static_cast<std::size_t>(i)]);
        }
        double q = 0.0;
        double projected = 0.0;
        for (int f = 0; f < layout.fields(); ++f) {
            const mko::PupilQuadrature<double>& pq = layout.quadrature(f);
            const auto n = static_cast<std::size_t>(layout.samples(f));
            for (int l = 0; l < layout.wavelengths(); ++l) {
                const auto o = static_cast<std::size_t>(layout.offset(f, l));
                std::vector<double> rest(n);
                const mko::TupleQuality tq =
                    mko::tuple_quality(pq, std::span<const double>(rr.opd).subspan(o, n),
                                       std::span<const std::uint8_t>(rr.status).subspan(o, n),
                                       batch.wavelength_nm(l) * 1e-6, rest);
                const double w =
                    tuple_weight[static_cast<std::size_t>(f * layout.wavelengths() + l)];
                q += w * (tq.rms / lambda_ref) * (tq.rms / lambda_ref);
                const double s = std::sqrt(w) / lambda_ref;
                double scale = 0.0;
                double sum = 0.0, sx = 0.0, sy = 0.0;
                for (std::size_t k = 0; k < n; ++k) {
                    scale = std::max(scale, std::abs(rr.residuals[o + k]));
                }
                for (std::size_t k = 0; k < n; ++k) {
                    const mko::PupilSample<double> smp = mko::sample(pq, static_cast<int>(k));
                    const double root = std::sqrt(smp.weight);
                    const double expected = s * root * rest[k];
                    worst_rest = std::max(worst_rest, std::abs(r[o + k] - expected) / scale);
                    projected += r[o + k] * r[o + k];
                    sum += root * r[o + k];
                    sx += root * smp.rho * std::sin(smp.phi) * r[o + k];
                    sy += root * smp.rho * std::cos(smp.phi) * r[o + k];
                }
                worst_orth = std::max({worst_orth, std::abs(sum) / scale, std::abs(sx) / scale,
                                       std::abs(sy) / scale});
            }
        }
        worst_q = std::max(worst_q, std::abs(projected - q) / q);
        // idempotent
        std::vector<double> twice = r;
        proj.apply(twice);
        double scale = 0.0;
        for (const double v : r) {
            scale = std::max(scale, std::abs(v));
        }
        for (std::size_t i = 0; i < r.size(); ++i) {
            worst_idem = std::max(worst_idem, std::abs(twice[i] - r[i]) / scale);
        }
    }
    EXPECT_LE(worst_rest, mko::test::kTiltProjectionRest);
    EXPECT_LE(worst_q, mko::test::kTiltProjectionRest);
    EXPECT_LE(worst_idem, mko::test::kTiltProjectionRest);
    EXPECT_LE(worst_orth, mko::test::kTiltProjectionRest);
    std::printf("[ OK ] Projektion gegen Rest der Tolerierung %.2e, Q %.2e, idempotent %.2e, "
                "orthogonal %.2e (relativ)\n",
                worst_rest, worst_q, worst_idem, worst_orth);
}

// Achsfelder (eine Speiche) haben keine anpassbare Kippung: die Residuen bleiben bitgleich.
TEST(TiltProjection, AxisFieldsUnchanged) {
    const mko::LoadedSystem loaded = mko::test::load_residual_system("einzellinse_bk7.json");
    const mko::SystemBatchHost batch = mko::make_system_batch(loaded, 1);
    const mko::ResidualLayout layout(batch.topology(), {});
    ASSERT_EQ(layout.pupil_mode(0), mko::PupilMode::Axis);
    const mko::ResidualResult rr = mko::residuals_cpu(batch, 0, layout);
    std::vector<double> r(rr.residuals.begin(), rr.residuals.end());
    const mko::TiltProjection proj(layout, rr.status);
    proj.apply(r);
    for (int f = 0; f < layout.fields(); ++f) {
        if (layout.pupil_mode(f) != mko::PupilMode::Axis) {
            continue;
        }
        for (int l = 0; l < layout.wavelengths(); ++l) {
            const auto o = static_cast<std::size_t>(layout.offset(f, l));
            for (std::size_t k = 0; k < static_cast<std::size_t>(layout.samples(f)); ++k) {
                EXPECT_EQ(r[o + k], rr.residuals[o + k]);
            }
        }
    }
    EXPECT_THROW(mko::TiltProjection(layout, std::span<const std::uint8_t>(rr.status).subspan(1)),
                 std::invalid_argument);
}

} // namespace
