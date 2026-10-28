// Glattes Maximum der Strahl-Extrema (M11-T1a, ADR-0013): Formeln, CPU-Reduktion, Dateiformat,
// mechanische Nebenbedingungen und Schutz der GPU-Pfade.

#include "mikroptim/batch/batch_extrema.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/optim/mechanical_constraints.hpp"
#include "mikroptim/optim/trace_extra_residuals.hpp"
#include "mikroptim/trace/surface_extrema.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using mko::test::kSmoothMaxFormula;

namespace {

const char* system_file() {
    return MKO_SOURCE_DIR "/data/systems/zweilinser_mechanik.json";
}

// Log-Sum-Exp direkt in long double
double direct_lse(const std::vector<double>& x, double beta) {
    const long double m = *std::max_element(x.begin(), x.end());
    long double s = 0.0L;
    for (double v : x) {
        s += std::exp(static_cast<long double>(beta) * (v - m));
    }
    return static_cast<double>(m + std::log(s) / beta);
}

} // namespace

TEST(SmoothMax, AccumulatorMatchesDirectFormulaAndBounds) {
    std::vector<double> x;
    for (int i = 0; i < 200; ++i) {
        x.push_back(2.0 + 0.5 * std::sin(0.37 * i) + 1e-3 * i); // Maximum nicht am Anfang
    }
    for (double beta : {1.0, 30.0, 1e3, 1e6}) {
        double m = 0.0, s = 0.0;
        for (double v : x) {
            mko::smooth_max_add(m, s, v, beta);
        }
        const double value = mko::smooth_max_value(m, s, beta, -1.0);
        const double hard = *std::max_element(x.begin(), x.end());
        EXPECT_NEAR(value, direct_lse(x, beta), kSmoothMaxFormula * (1.0 + std::abs(value)))
            << beta;
        EXPECT_GE(value, hard);
        EXPECT_LE(value, hard + std::log(static_cast<double>(x.size())) / beta + 1e-15);
        // Zusammenführen zweier Hälften = fortlaufend
        double m1 = 0.0, s1 = 0.0, m2 = 0.0, s2 = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            mko::smooth_max_add(i < 77 ? m1 : m2, i < 77 ? s1 : s2, x[i], beta);
        }
        mko::smooth_max_merge(m1, s1, m2, s2, beta);
        EXPECT_NEAR(mko::smooth_max_value(m1, s1, beta, -1.0), value,
                    kSmoothMaxFormula * (1.0 + std::abs(value)));
    }
    double m = 0.0, s = 0.0;
    EXPECT_EQ(mko::smooth_max_value(m, s, 10.0, 42.0), 42.0); // leer: neutrales Element
}

TEST(SmoothMax, SurfaceExtremaBoundsAndLimit) {
    const mko::LoadedSystem l = mko::load_system(system_file());
    const mko::SystemBatchHost batch = mko::make_system_batch(l, 1);
    const mko::ResidualLayout layout(batch.topology());
    const mko::SurfaceExtremaResult hard = mko::surface_extrema_cpu(batch, 0, layout);
    const double logN = std::log(static_cast<double>(layout.rays_per_variant()));
    for (double beta : {50.0, 1e3}) {
        const mko::SurfaceExtremaResult smooth =
            mko::surface_extrema_cpu(batch, 0, layout, mko::ExtremaSmoothing{beta, beta});
        for (int s = 0; s < hard.surfaces; ++s) {
            EXPECT_GE(smooth.max_height(0, s), hard.max_height(0, s) - 1e-15) << s;
            EXPECT_LE(smooth.max_height(0, s), hard.max_height(0, s) + logN / beta + 1e-12) << s;
            EXPECT_LE(smooth.min_cos_incidence(0, s), hard.min_cos_incidence(0, s) + 1e-15) << s;
            EXPECT_GE(smooth.min_cos_incidence(0, s),
                      hard.min_cos_incidence(0, s) - logN / beta - 1e-12)
                << s;
        }
    }
    // Grenzwert β → ∞: hartes Maximum
    const mko::SurfaceExtremaResult limit =
        mko::surface_extrema_cpu(batch, 0, layout, mko::ExtremaSmoothing{1e12, 1e12});
    for (int s = 0; s < hard.surfaces; ++s) {
        EXPECT_NEAR(limit.max_height(0, s), hard.max_height(0, s), 1e-10);
        EXPECT_NEAR(limit.min_cos_incidence(0, s), hard.min_cos_incidence(0, s), 1e-10);
    }
    // Nur eine Größe geglättet: die andere bleibt bitgleich hart
    const mko::SurfaceExtremaResult only_height =
        mko::surface_extrema_cpu(batch, 0, layout, mko::ExtremaSmoothing{100.0, 0.0});
    for (int s = 0; s < hard.surfaces; ++s) {
        EXPECT_EQ(only_height.min_cos_incidence(0, s), hard.min_cos_incidence(0, s));
    }
}

TEST(SmoothMax, FileFormatAndMechanicalConstraints) {
    mko::SystemDescription sys = mko::read_system_file(system_file());
    EXPECT_FALSE(sys.constraints->extrema_smoothing.has_value());
    EXPECT_FALSE(mko::make_mechanical_constraints(sys).smoothing.enabled());
    sys.constraints->extrema_smoothing = mko::ExtremaSmoothingSpec{2000.0, 500.0};
    const std::string text = mko::write_system_json(sys);
    EXPECT_NE(text.find("extrema_smoothing"), std::string::npos);
    const mko::SystemDescription back = mko::read_system_json(text);
    ASSERT_TRUE(back.constraints->extrema_smoothing.has_value());
    EXPECT_EQ(*back.constraints->extrema_smoothing, (mko::ExtremaSmoothingSpec{2000.0, 500.0}));
    const mko::MechanicalConstraints mc = mko::make_mechanical_constraints(back);
    EXPECT_EQ(mc.smoothing, (mko::ExtremaSmoothing{2000.0, 500.0}));
    const mko::MechanicalResiduals mr = mko::make_mechanical_residuals(mc);
    ASSERT_GT(mr.traced.count, 0);
    EXPECT_EQ(mr.traced.smoothing, mc.smoothing);
    // GPU-Pfade lehnen geglättete Extrema bis M11-T1b ab
    EXPECT_THROW(mko::require_hard_extrema(mr.traced, "Test"), std::invalid_argument);
    EXPECT_NO_THROW(mko::require_hard_extrema(mko::TraceExtraResiduals{}, "Test"));
    // Ungültige Werte
    std::string bad = text;
    bad.replace(bad.find("2000.0"), 6, "-1.0");
    EXPECT_THROW((void)mko::read_system_json(bad), std::exception);
}
