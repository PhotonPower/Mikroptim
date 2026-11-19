// Monte-Carlo-Tolerierung (M8-T4), CPU: Bewertung eines Tupels gegen analytische Orakel
// (Kolben und Kippung, Defokus mit RMS A/√12 und Strehl sinc², versagte Strahlen), Kennzahlen
// der Verteilung gegen exakte Ränge, Kompensator gegen die exakte Kompensation einer
// verschobenen planparallelen Platte, Strehl der Quadratur gegen die direkte Summe der PSF
// (M8-T2) auf dem dichten Raster, und die Abnahme: Verteilung von Strehl und RMS über 10.000
// Stichproben, reproduzierbar mit festem Startwert (bitgleich für jede Tranchengröße und
// Threadzahl). GPU-Teil: test_cuda_tolerance.cpp.

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/eval/eval.hpp"
#include "mikroptim/eval/psf.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/tolerance/tolerance_analysis.hpp"
#include "mikroptim/trace/surface_ops.hpp"
#include "tolerance_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

using mko::PupilMode;
using mko::SystemBatchHost;
using mko::ToleranceResult;
using mko::ToleranceSampleResult;
using mko::ToleranceSettings;
using nlohmann::json;

namespace {

constexpr double kLambda = 546.07e-6; // [mm]

struct Pupil {
    mko::PupilQuadrature<double> q;
    std::vector<double> x, y, rho2;
};

Pupil pupil(int rings, int spokes, PupilMode mode) {
    Pupil p;
    p.q = mko::make_pupil_quadrature<double>(rings, spokes, mode);
    for (int k = 0; k < mko::sample_count(p.q); ++k) {
        const mko::PupilSample<double> s = mko::sample(p.q, k);
        p.x.push_back(s.rho * std::sin(s.phi));
        p.y.push_back(s.rho * std::cos(s.phi));
        p.rho2.push_back(s.rho * s.rho);
    }
    return p;
}

bool same_double(double a, double b) {
    return std::memcmp(&a, &b, sizeof a) == 0;
}

// Bitgleiche Ergebnisse zweier Stichproben.
bool same_sample(const ToleranceSampleResult& a, const ToleranceSampleResult& b) {
    if (a.rms.size() != b.rms.size() || a.strehl.size() != b.strehl.size()) {
        return false;
    }
    for (std::size_t t = 0; t < a.rms.size(); ++t) {
        if (!same_double(a.rms[t], b.rms[t]) || !same_double(a.strehl[t], b.strehl[t])) {
            return false;
        }
    }
    return same_double(a.rms_composite, b.rms_composite) &&
           same_double(a.strehl_min, b.strehl_min) &&
           same_double(a.working_distance, b.working_distance) &&
           same_double(a.compensation, b.compensation) && a.failed_rays == b.failed_rays &&
           a.compensator_status == b.compensator_status &&
           a.compensator_iterations == b.compensator_iterations;
}

struct Analysis {
    mko::LoadedSystem loaded;
    SystemBatchHost batch;
};

Analysis prepare(const json& j) {
    Analysis a;
    a.loaded = mko::test::load_tolerance_json(j);
    a.batch = mko::make_system_batch(a.loaded, 1);
    return a;
}

ToleranceResult run(const Analysis& a, const ToleranceSettings& s) {
    return mko::run_tolerance_analysis(a.loaded.system, a.batch, s);
}

} // namespace

TEST(ToleranceQuality, PistonAndTiltAreRemoved) {
    const Pupil p = pupil(10, 8, PupilMode::Full);
    const std::size_t n = p.x.size();
    std::vector<double> opd(n);
    const std::vector<std::uint8_t> status(n, 0);
    for (std::size_t k = 0; k < n; ++k) {
        opd[k] = 0.3 + 2e-4 * p.x[k] - 7e-4 * p.y[k];
    }
    const mko::TupleQuality t = mko::tuple_quality(p.q, opd, status, kLambda);
    EXPECT_NEAR(t.rms, 0.0, mko::test::kLengthFp64);
    EXPECT_NEAR(t.strehl, 1.0, mko::test::kUnitFp64);
    EXPECT_NEAR(t.tilt_x, 2e-4, mko::test::kLengthFp64);
    EXPECT_NEAR(t.tilt_y, -7e-4, mko::test::kLengthFp64);
    EXPECT_EQ(t.failed, 0);
}

TEST(ToleranceQuality, DefocusMatchesClosedForm) {
    // OPD = A ρ² (+ Kippung): mit u = ρ² gleichverteilt auf [0, 1] ist RMS = A/√12 und
    // Strehl = |∫₀¹ e^{iφu} du|² = sinc²(φ/2), φ = 2πA/λ. Die radiale Gauß-Regel ist für
    // Polynome in u exakt; für e^{iφu} ist ihr Fehler bei 10 Ringen ≈ φ²⁰/(20!)².
    for (const PupilMode mode : {PupilMode::Full, PupilMode::Axis}) {
        const Pupil p = pupil(10, 8, mode);
        const std::size_t n = p.x.size();
        const std::vector<std::uint8_t> status(n, 0);
        for (const double waves : {0.05, 0.2, 0.5, 1.0}) {
            const double a = waves * kLambda;
            std::vector<double> opd(n);
            for (std::size_t k = 0; k < n; ++k) {
                // Kippung nur bei voller Pupille (im Achsfall ist x = 0 und die Kippung nicht
                // bestimmbar; die Bewertung zieht dort nur den Kolben ab)
                opd[k] = a * p.rho2[k] + (mode == PupilMode::Full ? 1e-3 * p.y[k] : 0.0);
            }
            const mko::TupleQuality t = mko::tuple_quality(p.q, opd, status, kLambda);
            const double half = std::numbers::pi * waves;
            const double sinc = std::sin(half) / half;
            EXPECT_NEAR(t.rms / (a / std::sqrt(12.0)), 1.0, mko::test::kQuadrature * 10) << waves;
            EXPECT_NEAR(t.strehl, sinc * sinc, mko::test::kQuadrature * 10) << waves;
            if (mode == PupilMode::Full) {
                EXPECT_NEAR(t.tilt_y, 1e-3, mko::test::kLengthFp64);
            }
        }
    }
}

TEST(ToleranceQuality, FailedRaysAreOpaque) {
    const Pupil p = pupil(6, 8, PupilMode::Full);
    const std::size_t n = p.x.size();
    std::vector<double> opd(n, 0.0);
    std::vector<std::uint8_t> status(n, 0);
    // Äußerster Ring versagt (Gewicht W_5): Strehl = (1 − W_5)², RMS 0
    double transmitted = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        if (static_cast<int>(k) / 8 == 5) {
            status[k] = static_cast<std::uint8_t>(mko::TraceStatus::Missed);
            opd[k] = std::numeric_limits<double>::quiet_NaN();
        } else {
            transmitted += mko::sample(p.q, static_cast<int>(k)).weight;
        }
    }
    mko::TupleQuality t = mko::tuple_quality(p.q, opd, status, kLambda);
    EXPECT_EQ(t.failed, 8);
    EXPECT_NEAR(t.rms, 0.0, mko::test::kLengthFp64);
    EXPECT_NEAR(t.strehl, transmitted * transmitted, mko::test::kUnitFp64);
    // Alle versagt: RMS NaN, Strehl 0
    std::fill(status.begin(), status.end(), static_cast<std::uint8_t>(mko::TraceStatus::TIR));
    t = mko::tuple_quality(p.q, opd, status, kLambda);
    EXPECT_TRUE(std::isnan(t.rms));
    EXPECT_EQ(t.strehl, 0.0);
    EXPECT_EQ(t.failed, static_cast<int>(n));
    // Fehler: Längen, Wellenlänge
    EXPECT_THROW(
        (void)mko::tuple_quality(p.q, std::span<const double>(opd).first(3), status, kLambda),
        std::invalid_argument);
    EXPECT_THROW((void)mko::tuple_quality(p.q, opd, status, 0.0), std::invalid_argument);
}

TEST(ToleranceStats, NearestRankPercentiles) {
    std::vector<double> v;
    for (int i = 100; i >= 1; --i) {
        v.push_back(i);
    }
    const mko::DistributionStats rms = mko::distribution_stats(v, true);
    EXPECT_EQ(rms.count, 100);
    EXPECT_EQ(rms.mean, 50.5);
    EXPECT_NEAR(rms.std, std::sqrt(100.0 * 101.0 / 12.0), 1e-12); // (n² − 1)/12 · n/(n − 1)
    EXPECT_EQ(rms.min, 1.0);
    EXPECT_EQ(rms.max, 100.0);
    EXPECT_EQ(rms.p50, 50.0);
    EXPECT_EQ(rms.p90, 90.0);
    EXPECT_EQ(rms.p95, 95.0);
    EXPECT_EQ(rms.p99, 99.0);
    const mko::DistributionStats strehl = mko::distribution_stats(v, false);
    EXPECT_EQ(strehl.p50, 51.0);
    EXPECT_EQ(strehl.p90, 11.0);
    EXPECT_EQ(strehl.p95, 6.0);
    EXPECT_EQ(strehl.p99, 2.0);
    // NaN zählt als schlechtester Wert in den Perzentilen, nicht in Mittel und Std
    v[0] = std::numeric_limits<double>::quiet_NaN(); // war 100
    const mko::DistributionStats with_nan = mko::distribution_stats(v, true);
    EXPECT_EQ(with_nan.count, 99);
    EXPECT_EQ(with_nan.max, 99.0);
    EXPECT_EQ(with_nan.p99, 99.0);
    const mko::DistributionStats two =
        mko::distribution_stats(std::vector<double>{3.0, std::nan("")}, true);
    EXPECT_EQ(two.p50, 3.0);
    EXPECT_TRUE(std::isnan(two.p99)); // schlechtester Wert NaN
    const mko::DistributionStats empty = mko::distribution_stats({}, true);
    EXPECT_EQ(empty.count, 0);
    EXPECT_TRUE(std::isnan(empty.mean));
    EXPECT_TRUE(std::isnan(empty.p50));
}

TEST(ToleranceAnalysis, CompensatorRecoversShiftedPlate) {
    // Orakel: Eine planparallele Platte im konvergenten Strahl ist längs z verschiebungs-
    // invariant. Ein um δ_i = 0,05 mm · ξ_i längerer Luftspalt davor wird durch den Arbeitsabstand
    // exakt ausgeglichen: Kompensation_i = Kompensation_nominal − δ_i, RMS und Strehl wie nominal.
    const Analysis a = prepare(mko::test::plate_system_json());
    ToleranceSettings s;
    s.samples = 300;
    s.seed = 11;
    const ToleranceResult r = run(a, s);
    ASSERT_EQ(r.results.size(), 300u);
    EXPECT_EQ(r.compensator, mko::CompensatorType::WorkingDistance);
    const mko::ToleranceModel model(a.loaded.system, mko::make_non_coaxial(a.batch).topology());
    double worst_comp = 0.0;
    double worst_rms = 0.0;
    for (std::size_t i = 0; i < r.results.size(); ++i) {
        const ToleranceSampleResult& x = r.results[i];
        const double delta = 0.05 * model.deviates(11, i)[0];
        worst_comp =
            std::max(worst_comp, std::abs(x.compensation - (r.nominal.compensation - delta)));
        worst_rms = std::max(worst_rms, std::abs(x.rms_composite / r.nominal.rms_composite - 1.0));
        EXPECT_EQ(x.failed_rays, 0);
    }
    EXPECT_LT(worst_comp, mko::test::kToleranceCompensator);
    EXPECT_LT(worst_rms, mko::test::kToleranceCompensatedRms);
    // Ohne Kompensator: Arbeitsabstand aus der Datei, keine Kompensation
    s.compensator = mko::CompensatorType::None;
    s.samples = 20;
    const ToleranceResult none = run(a, s);
    EXPECT_EQ(none.compensator, mko::CompensatorType::None);
    for (const ToleranceSampleResult& x : none.results) {
        EXPECT_EQ(x.compensation, 0.0);
        EXPECT_EQ(x.working_distance, a.batch.image_distance(0));
        EXPECT_EQ(x.compensator_status, mko::LmStatus::Running);
    }
}

TEST(ToleranceAnalysis, QuadratureStrehlMatchesDirectPsfSum) {
    // Strehl der Quadratur (Kippung abgezogen) gegen PsfResult::strehl (direkte Summe über das
    // dichte EVAL-Raster, ebenfalls nach Abzug der Kippung, M8-T2) für gestörte Stichproben ohne
    // Kompensator, Toleranzen dreifach (Strehl bis unter 0,5). Beide sind Näherungen desselben
    // Integrals über die Pupille.
    json j = mko::test::tolerance_system_json();
    for (json& item : j["tolerances"]["items"]) {
        for (const char* key : {"delta", "relative", "fringes", "arcmin"}) {
            if (item.contains(key)) {
                item[key] = 3.0 * item[key].get<double>();
            }
        }
    }
    const Analysis a = prepare(j);
    ToleranceSettings s;
    s.samples = 12;
    s.seed = 5;
    s.compensator = mko::CompensatorType::None; // Standardquadratur 10 Ringe × 8 Speichen
    const ToleranceResult r = run(a, s);
    const SystemBatchHost base = mko::make_non_coaxial(a.batch);
    const mko::ToleranceModel model(a.loaded.system, base.topology());
    SystemBatchHost samples = base;
    samples.resize(s.samples);
    model.write_samples(base, 0, s.seed, 0, samples, 0, s.samples);
    mko::EvalSettings es;
    es.grid = 256;
    const mko::EvalResult ev = mko::eval_cpu(samples, es);
    double worst = 0.0;
    double smallest = 1.0;
    for (int i = 0; i < s.samples; ++i) {
        const std::vector<mko::PsfResult> psf = mko::psf_cpu(samples, ev, i);
        ASSERT_EQ(psf.size(), r.results[static_cast<std::size_t>(i)].strehl.size());
        for (std::size_t t = 0; t < psf.size(); ++t) {
            worst = std::max(
                worst, std::abs(psf[t].strehl - r.results[static_cast<std::size_t>(i)].strehl[t]));
            smallest = std::min(smallest, psf[t].strehl);
        }
    }
    std::printf("Strehl Quadratur gegen PSF: max. %.3e (kleinster Strehl %.3f)\n", worst, smallest);
    EXPECT_LT(worst, mko::test::kToleranceStrehlVsPsf);
    EXPECT_LT(smallest, 0.5); // auch deutlich gestörte Stichproben
}

TEST(ToleranceAnalysis, ZeroTolerancesReproduceNominal) {
    json j = mko::test::tolerance_system_json();
    for (json& item : j["tolerances"]["items"]) {
        for (const char* key : {"delta", "relative", "fringes", "arcmin"}) {
            if (item.contains(key)) {
                item[key] = 0.0;
            }
        }
    }
    const Analysis a = prepare(j);
    ToleranceSettings s;
    s.samples = 5;
    const ToleranceResult r = run(a, s);
    for (const ToleranceSampleResult& x : r.results) {
        EXPECT_TRUE(same_sample(x, r.nominal));
    }
    EXPECT_EQ(r.rms_composite.std, 0.0);
    EXPECT_EQ(r.rms_composite.p99, r.nominal.rms_composite);
}

TEST(ToleranceAnalysis, InvalidSettings) {
    const Analysis a = prepare(mko::test::tolerance_system_json());
    ToleranceSettings s;
    s.samples = 0;
    EXPECT_THROW((void)run(a, s), std::invalid_argument);
    s.samples = 1;
    s.chunk = 0;
    EXPECT_THROW((void)run(a, s), std::invalid_argument);
    s.chunk = 1;
    Analysis two = prepare(mko::test::tolerance_system_json());
    two.batch.resize(2);
    EXPECT_THROW((void)run(two, s), std::invalid_argument);
}

// Abnahme M8-T4: Verteilung von Strehl und RMS-Wellenfront über 10.000 Stichproben des
// Testsystems mit allen Toleranzarten und Kompensator Arbeitsabstand; Ergebnis reproduzierbar
// mit festem Startwert. Ein zweiter Lauf mit anderer Tranchengröße und einem Host-Thread liefert
// für die ersten 2.500 Stichproben bitgleiche Ergebnisse (Stichprobe i hängt nur von Startwert
// und i ab), ein anderer Startwert andere.
TEST(ToleranceAnalysis, TenThousandSamplesReproducible) {
    const Analysis a = prepare(mko::test::tolerance_system_json());
    ToleranceSettings s;
    s.samples = 10000;
    s.seed = 2026;
    const ToleranceResult r = run(a, s);
    ASSERT_EQ(r.results.size(), 10000u);
    EXPECT_EQ(r.chunks, 3);
    EXPECT_EQ(r.draws, 26);
    EXPECT_EQ(r.failed_samples, 0);
    // Verteilung: alle Stichproben endlich, Perzentile monoton, Störungen verschlechtern im
    // Mittel; Kennzahlen konsistent mit den Werten pro Tupel
    EXPECT_EQ(r.rms_composite.count, 10000);
    EXPECT_EQ(r.strehl_min.count, 10000);
    EXPECT_LE(r.rms_composite.p50, r.rms_composite.p90);
    EXPECT_LE(r.rms_composite.p90, r.rms_composite.p95);
    EXPECT_LE(r.rms_composite.p95, r.rms_composite.p99);
    EXPECT_GE(r.strehl_min.p50, r.strehl_min.p90);
    EXPECT_GE(r.strehl_min.p90, r.strehl_min.p95);
    EXPECT_GE(r.strehl_min.p95, r.strehl_min.p99);
    EXPECT_GT(r.rms_composite.mean, r.nominal.rms_composite);
    EXPECT_LT(r.strehl_min.mean, r.nominal.strehl_min);
    EXPECT_GT(r.yield, 0.0);
    EXPECT_LT(r.yield, 1.0);
    for (const ToleranceSampleResult& x : r.results) {
        double sum = 0.0;
        double smin = 2.0;
        for (std::size_t t = 0; t < x.rms.size(); ++t) {
            sum += r.tuple_weight[t] * (x.rms[t] * x.rms[t]);
            smin = std::min(smin, x.strehl[t]);
        }
        ASSERT_EQ(x.rms_composite, std::sqrt(sum));
        ASSERT_EQ(x.strehl_min, smin);
        ASSERT_GT(x.strehl_min, 0.0);
        ASSERT_LE(x.strehl_min, 1.0);
    }
    // Reproduzierbarkeit: andere Tranchen und ein Thread
    ToleranceSettings t = s;
    t.samples = 2500;
    t.chunk = 777;
    t.threads = 1;
    t.lm.threads = 1;
    const ToleranceResult q = run(a, t);
    EXPECT_TRUE(same_sample(q.nominal, r.nominal));
    for (std::size_t i = 0; i < q.results.size(); ++i) {
        ASSERT_TRUE(same_sample(q.results[i], r.results[i])) << i;
    }
    // Anderer Startwert: andere Stichproben
    t.samples = 3;
    t.seed = 2027;
    const ToleranceResult o = run(a, t);
    EXPECT_FALSE(same_sample(o.results[0], r.results[0]));
}
