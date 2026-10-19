// Mehrere GPUs (M7-T3, nur mit MIKROPTIM_ENABLE_CUDA). Mit einer GPU wird die Verteilung über
// "logische Geräte" geprüft: dasselbe Gerät mehrfach in der Liste ({0, 0}, {0, 0, 0}), jeder
// Eintrag mit eigener Engine und eigenem Host-Thread. Abnahme: Ergebnisse pro Variante bitgleich
// zum Lauf auf einer Engine (alle Modi, Batchgrößen, ungleiche Anteile, Granularität, Tranchen
// in den Slots), Verteilung nach gemessenem Durchsatz (Kalibrierung, gleitender Mittelwert).
// Echte Mehrgerätetests (RTX 3090 + Tesla P100) laufen nur, wenn zwei Geräte sichtbar sind:
// MultiTraceEngine.RealDevicesBitIdenticalToSingle und MultiTraceEngine.ResultFingerprint.

#include "lm_test_systems.hpp"
#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/cuda/cuda_error.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/levenberg_marquardt.hpp"
#include "mikroptim/cuda/merit.hpp"
#include "mikroptim/cuda/multi_levenberg_marquardt.hpp"
#include "mikroptim/cuda/multi_trace_engine.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/cuda/tuple_packs.hpp"
#include "prefilter_test_systems.hpp"
#include "residual_test_systems.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using mko::BatchTopology;
using mko::OutputRequest;
using mko::PrefilterResult;
using mko::ResidualLayout;
using mko::SystemBatchHost;
using mko::cuda::CudaError;
using mko::cuda::DeviceRange;
using mko::cuda::DeviceTranche;
using mko::cuda::EngineOptions;
using mko::cuda::MultiEngineOptions;
using mko::cuda::MultiTraceEngine;
using mko::cuda::Precision;
using mko::cuda::PrefilterSettings;
using mko::cuda::TraceEngine;
using mko::cuda::TraceMode;
using mko::cuda::TraceOutput;
using mko::cuda::TraceSettings;
using mko::cuda::WorkKind;

namespace {

#define MKO_REQUIRE_GPU()                                                                          \
    do {                                                                                           \
        if (mko::cuda::device_count() == 0) {                                                      \
            GTEST_SKIP() << "Keine CUDA-GPU gefunden";                                             \
        }                                                                                          \
    } while (0)

template <class T>
bool same_bytes(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

bool same_extrema(const mko::SurfaceExtremaResult& a, const mko::SurfaceExtremaResult& b) {
    return a.variants == b.variants && a.surfaces == b.surfaces && same_bytes(a.values, b.values);
}

// Alle Ergebnisfelder bitgleich (Modus, Größen, Residuen, OPD, Status, Merit, Extrema).
::testing::AssertionResult identical(const TraceOutput& a, const TraceOutput& b) {
    const mko::ResidualResult& ra = a.residuals;
    const mko::ResidualResult& rb = b.residuals;
    const mko::MeritResult& ma = a.merit;
    const mko::MeritResult& mb = b.merit;
    if (a.mode != b.mode || a.layout.rays_per_variant() != b.layout.rays_per_variant() ||
        a.info.variants != b.info.variants) {
        return ::testing::AssertionFailure() << "Modus oder Größen verschieden";
    }
    if (ra.variants != rb.variants || ra.rays_per_variant != rb.rays_per_variant ||
        ra.penalty_rows != rb.penalty_rows || ra.stride != rb.stride ||
        !same_bytes(ra.residuals, rb.residuals) || !same_bytes(ra.opd, rb.opd) ||
        !same_bytes(ra.status, rb.status) || !same_extrema(ra.extrema, rb.extrema)) {
        return ::testing::AssertionFailure() << "RESIDUALS verschieden";
    }
    if (ma.variants != mb.variants || ma.rays_per_variant != mb.rays_per_variant ||
        !same_bytes(ma.merit, mb.merit) || !same_bytes(ma.failed, mb.failed) ||
        !same_bytes(ma.hard_failed, mb.hard_failed) || !same_extrema(ma.extrema, mb.extrema)) {
        return ::testing::AssertionFailure() << "MERIT verschieden";
    }
    return ::testing::AssertionSuccess();
}

::testing::AssertionResult identical(const PrefilterResult& a, const PrefilterResult& b) {
    const bool same = a.stats.variants == b.stats.variants &&
                      a.stats.rays_per_variant == b.stats.rays_per_variant &&
                      same_bytes(a.stats.merit, b.stats.merit) &&
                      same_bytes(a.stats.optical_length, b.stats.optical_length) &&
                      same_bytes(a.stats.failed, b.stats.failed) &&
                      same_bytes(a.stats.uncertain_ok, b.stats.uncertain_ok) &&
                      same_bytes(a.stats.uncertain_failed, b.stats.uncertain_failed) &&
                      same_bytes(a.merit_lower, b.merit_lower) &&
                      same_bytes(a.merit_upper, b.merit_upper) && same_bytes(a.reject, b.reject) &&
                      std::memcmp(&a.reference_merit, &b.reference_merit, sizeof(double)) == 0 &&
                      a.reference_variant == b.reference_variant && a.kept == b.kept;
    return same ? ::testing::AssertionSuccess()
                : ::testing::AssertionFailure() << "Vorfilterung verschieden";
}

// Gerätespeicher eines Laufs mit `variants` Varianten (wie test_cuda_trace_engine.cpp).
std::size_t run_bytes(const BatchTopology& t, int variants, const ResidualLayout& layout,
                      TraceMode mode, const TraceSettings& s) {
    OutputRequest r;
    r.rays_per_variant = layout.rays_per_variant();
    r.penalty_residuals = layout.penalty_rows();
    if (mode == TraceMode::Merit) {
        r.residuals = false;
        r.ray_status = false;
        r.opd = false;
        r.merit = true;
    } else {
        r.extra_residuals = s.extra_residuals;
        r.residuals = true;
        r.ray_status = true;
        r.opd = s.opd;
        r.merit = false;
    }
    r.surface_max = s.extrema;
    const std::size_t extra =
        mode == TraceMode::Merit && !s.extra_merit.empty() ? sizeof(double) : 0;
    return mko::estimate_device_bytes(t, variants, r, extra) +
           mko::align_bytes(static_cast<std::size_t>(t.fields) * sizeof(mko::ResidualFieldPlan)) +
           mko::align_bytes(static_cast<std::size_t>(t.fields) * t.wavelengths *
                            sizeof(mko::cuda::TuplePack)); // Tupelpakete (M7-T4)
}

EngineOptions limited_to(std::size_t bytes) {
    EngineOptions o;
    o.memory_limit_bytes = bytes;
    o.reserve = mko::MemoryReserve{0, 0.0};
    return o;
}

MultiEngineOptions logical(int slots, std::vector<double> weights = {}) {
    MultiEngineOptions o;
    o.devices.assign(static_cast<std::size_t>(slots), 0);
    o.weights = std::move(weights);
    return o;
}

std::vector<double> extra_merit_values(int variants) {
    std::vector<double> e(static_cast<std::size_t>(variants));
    for (int v = 0; v < variants; ++v) {
        e[static_cast<std::size_t>(v)] = 1e-9 * (v % 13);
    }
    return e;
}

// Bereiche des letzten Laufs: lückenlos über [0, V) und, bei festen Gewichten, wie
// split_variants.
void expect_ranges(const MultiTraceEngine& m, int variants, int granularity = 1) {
    const mko::cuda::MultiRunInfo& info = m.last_run();
    ASSERT_EQ(static_cast<int>(info.devices.size()), m.size());
    int next = 0;
    for (const mko::cuda::DeviceRunInfo& d : info.devices) {
        EXPECT_EQ(d.range.first, next);
        next += d.range.count;
    }
    EXPECT_EQ(next, variants);
    if (!m.options().weights.empty()) {
        const std::vector<DeviceRange> expect =
            mko::cuda::split_variants(variants, granularity, m.options().weights);
        for (int k = 0; k < m.size(); ++k) {
            EXPECT_EQ(info.devices[static_cast<std::size_t>(k)].range,
                      expect[static_cast<std::size_t>(k)]);
        }
    }
}

// FNV-1a (64 Bit) über die Bytes eines Vektors.
template <class T>
std::uint64_t fnv1a(const std::vector<T>& v, std::uint64_t h = 1469598103934665603ull) {
    const auto* p = reinterpret_cast<const unsigned char*>(v.data());
    for (std::size_t i = 0; i < v.size() * sizeof(T); ++i) {
        h = (h ^ p[i]) * 1099511628211ull;
    }
    return h;
}

} // namespace

// Gerätewahl und Optionen: ohne GPU CudaError (läuft auch in der CI), sonst Prüfung von
// Index, Gewichten und Einstellungen; Standardliste = alle Geräte ab sm_60.
TEST(MultiTraceEngine, DeviceSelectionAndOptions) {
    const int n = mko::cuda::device_count();
    if (n == 0) {
        EXPECT_THROW(MultiTraceEngine{}, CudaError);
        EXPECT_TRUE(mko::cuda::usable_devices().empty());
        return;
    }
    const MultiTraceEngine all;
    EXPECT_EQ(all.size(), static_cast<int>(mko::cuda::usable_devices().size()));
    EXPECT_GE(all.size(), 1);
    for (int k = 0; k < all.size(); ++k) {
        std::printf("[ Info     ] Slot %d: Gerät %d (%s)\n", k, all.device(k),
                    all.engine(k).device_name().c_str());
    }
    EXPECT_THROW((void)MultiTraceEngine{logical(1, {})}.engine(1), std::out_of_range);
    MultiEngineOptions bad = logical(2);
    bad.devices[1] = n;
    EXPECT_THROW(MultiTraceEngine{bad}, std::invalid_argument);
    bad.devices[1] = -1;
    EXPECT_THROW(MultiTraceEngine{bad}, std::invalid_argument);
    EXPECT_THROW(MultiTraceEngine{logical(2, {1.0})}, std::invalid_argument);
    EXPECT_THROW((MultiTraceEngine{logical(2, {1.0, -1.0})}), std::invalid_argument);
    EXPECT_THROW((MultiTraceEngine{logical(2, {0.0, 0.0})}), std::invalid_argument);
    MultiEngineOptions o = logical(2);
    o.smoothing = 0.0;
    EXPECT_THROW(MultiTraceEngine{o}, std::invalid_argument);
    o = logical(2);
    o.calibration_variants = -1;
    EXPECT_THROW(MultiTraceEngine{o}, std::invalid_argument);
    o = logical(2);
    o.slot_engines.resize(3);
    EXPECT_THROW(MultiTraceEngine{o}, std::invalid_argument);
    o = logical(2);
    o.engine.reserve.fraction = 1.5; // wie TraceEngine
    EXPECT_THROW(MultiTraceEngine{o}, std::invalid_argument);

    // Logische Geräte teilen sich den Speicher: Grenze je Slot höchstens die Hälfte
    const MultiTraceEngine two(logical(2));
    ASSERT_EQ(two.size(), 2);
    const std::size_t total = mko::cuda::device_memory_info(0).total_bytes;
    for (int k = 0; k < 2; ++k) {
        ASSERT_TRUE(two.engine(k).options().memory_limit_bytes.has_value());
        EXPECT_LE(*two.engine(k).options().memory_limit_bytes, total / 2);
    }
    // Anteile: gleich ohne Schätzung, feste Gewichte normiert
    EXPECT_EQ(two.shares(WorkKind::Merit), (std::vector<double>{0.5, 0.5}));
    EXPECT_FALSE(two.has_estimate(WorkKind::Merit));
    const MultiTraceEngine fixed(logical(2, {3.0, 1.0}));
    EXPECT_EQ(fixed.shares(WorkKind::Residuals, Precision::Df64),
              (std::vector<double>{0.75, 0.25}));
    EXPECT_TRUE(fixed.has_estimate(WorkKind::Residuals));
}

// Abnahme M7-T3 (Bitgleichheit), Modus RESIDUALS mit OPD, Status, weiteren Residuen und
// Max-Reduktionen: {0, 0} und {0, 0, 0}, gleiche, ungleiche, gemessene Anteile und ein Anteil 0,
// Batchgrößen 1, 2, 7 und 700; jeweils bitgleich zum Lauf auf einer Engine.
TEST(MultiTraceEngine, ResidualsBitIdenticalOnLogicalDevices) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    TraceSettings s;
    s.opd = true;
    s.extra_residuals = 3;
    s.extrema = true;
    TraceEngine single(0);
    const std::vector<std::vector<std::vector<double>>> weights = {
        {{}, {1.0, 1.0}, {3.0, 1.0}, {0.0, 1.0}},
        {{}, {1.0, 2.0, 5.0}, {1.0, 0.0, 1.0}},
    };
    for (const int variants : {1, 2, 7, 700}) {
        const SystemBatchHost batch = mko::test::perturbed_batch(ls, variants, 4242);
        const TraceOutput reference = single.run(batch, TraceMode::Residuals, s);
        for (std::size_t list = 0; list < weights.size(); ++list) {
            for (const std::vector<double>& w : weights[list]) {
                MultiTraceEngine multi(logical(static_cast<int>(list) + 2, w));
                TraceOutput out;
                multi.run(batch, TraceMode::Residuals, s, out);
                EXPECT_TRUE(identical(out, reference))
                    << variants << " Varianten, " << multi.size() << " Slots, Gewichte "
                    << (w.empty() ? std::string("gemessen") : std::to_string(w.size()));
                expect_ranges(multi, variants);
                EXPECT_EQ(out.info.variants, variants);
                // Wiederverwendung der Ausgabe und der Puffer: zweiter Lauf ebenfalls bitgleich
                multi.run(batch, TraceMode::Residuals, s, out);
                EXPECT_TRUE(identical(out, reference));
            }
        }
    }
    // Ohne Status und OPD, ohne weitere Residuen
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 300, 99);
    TraceSettings plain;
    plain.status = false;
    MultiTraceEngine multi(logical(3, {2.0, 1.0, 1.0}));
    EXPECT_TRUE(identical(multi.run(batch, TraceMode::Residuals, plain),
                          single.run(batch, TraceMode::Residuals, plain)));
}

// Modus MERIT mit extra_merit und Max-Reduktionen sowie beide Modi in df64: bitgleich.
TEST(MultiTraceEngine, MeritAndDf64BitIdenticalOnLogicalDevices) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 701, 31337);
    const std::vector<double> extra = extra_merit_values(batch.variants());
    TraceEngine single(0);
    for (const Precision p : {Precision::Fp64, Precision::Df64}) {
        TraceSettings sm;
        sm.extrema = true;
        sm.extra_merit = extra;
        sm.precision = p;
        TraceSettings sr;
        sr.opd = true;
        sr.precision = p;
        const TraceOutput ref_m = single.run(batch, TraceMode::Merit, sm);
        const TraceOutput ref_r = single.run(batch, TraceMode::Residuals, sr);
        for (const auto& w : std::vector<std::vector<double>>{{}, {1.0, 3.0}, {5.0, 1.0, 2.0}}) {
            MultiTraceEngine multi(logical(w.empty() ? 3 : static_cast<int>(w.size()), w));
            EXPECT_TRUE(identical(multi.run(batch, TraceMode::Merit, sm), ref_m))
                << mko::cuda::to_string(p);
            expect_ranges(multi, batch.variants());
            EXPECT_EQ(multi.last_run().kind, WorkKind::Merit);
            EXPECT_EQ(multi.last_run().precision, p);
            EXPECT_TRUE(identical(multi.run(batch, TraceMode::Residuals, sr), ref_r))
                << mko::cuda::to_string(p);
        }
    }
    // Leerer Batch
    SystemBatchHost empty = batch;
    empty.resize(0);
    MultiTraceEngine multi(logical(2));
    const TraceOutput out = multi.run(empty, TraceMode::Merit);
    EXPECT_EQ(out.merit.variants, 0);
    EXPECT_EQ(out.info.tranches, 0);
}

// Tranchen innerhalb der Slots (Speichergrenze für 64 Varianten pro Engine): bitgleich, die
// Statistik summiert die Tranchen der Slots.
TEST(MultiTraceEngine, TranchesInsideSlots) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 700, 8008);
    TraceSettings s;
    s.opd = true;
    const ResidualLayout layout(batch.topology(), s.quadrature);
    TraceEngine single(0);
    const TraceOutput reference = single.run(batch, TraceMode::Residuals, s);
    MultiEngineOptions o = logical(3, {1.0, 1.0, 1.0});
    o.engine = limited_to(run_bytes(batch.topology(), 64, layout, TraceMode::Residuals, s));
    MultiTraceEngine multi(o);
    const TraceOutput out = multi.run(batch, TraceMode::Residuals, s);
    EXPECT_TRUE(identical(out, reference));
    int tranches = 0;
    for (const mko::cuda::DeviceRunInfo& d : multi.last_run().devices) {
        EXPECT_EQ(d.info.tranche_variants, 64);
        EXPECT_EQ(d.info.variants, d.range.count);
        tranches += d.info.tranches;
    }
    EXPECT_EQ(out.info.tranches, tranches);
    EXPECT_EQ(out.info.tranches, 4 + 4 + 4); // 233/234/233 Varianten in Tranchen zu 64
    EXPECT_EQ(out.info.tranche_variants, 64);
}

// run_on_device: Bereiche aus ganzen Blöcken (granularity), Tranchen jedes Slots innerhalb
// seines Bereichs mit `first` bezogen auf den ganzen Batch; Rückrufe aus mehreren Threads.
// Ausnahmen aus dem Rückruf werden unverändert weitergereicht.
TEST(MultiTraceEngine, RunOnDeviceKeepsBlocksTogether) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const int g = 27;
    const int V = g * 9 + 5;
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, V, 77);
    const ResidualLayout layout(batch.topology());
    TraceSettings s;
    s.extra_residuals = 2;
    MultiEngineOptions o = logical(3, {1.0, 2.0, 1.0});
    // höchstens 2 Blöcke pro Tranche
    o.engine = limited_to(run_bytes(batch.topology(), 2 * g + 3, layout, TraceMode::Residuals, s));
    MultiTraceEngine multi(o);
    struct Seen {
        int slot, first, count;
        bool ok;
    };
    std::mutex mutex;
    std::vector<Seen> seen;
    const mko::cuda::TraceRunInfo info =
        multi.run_on_device(batch, s, g, [&](int slot, const DeviceTranche& t) {
            const bool ok = t.residuals != nullptr && t.status != nullptr &&
                            t.rays_per_variant == layout.rays_per_variant() &&
                            t.penalty_rows == layout.penalty_rows() &&
                            t.stride == layout.trace_rows() + 2;
            const std::lock_guard<std::mutex> lock(mutex);
            seen.push_back({slot, t.first, t.count, ok});
        });
    EXPECT_EQ(info.variants, V);
    expect_ranges(multi, V, g);
    std::sort(seen.begin(), seen.end(),
              [](const Seen& a, const Seen& b) { return a.first < b.first; });
    int next = 0;
    for (const Seen& t : seen) {
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(t.first, next);
        EXPECT_EQ(t.first % g, 0);
        if (t.first + t.count < V) {
            EXPECT_EQ(t.count % g, 0);
        }
        EXPECT_LE(t.count, 2 * g);
        const DeviceRange r = multi.last_run().devices[static_cast<std::size_t>(t.slot)].range;
        EXPECT_GE(t.first, r.first);
        EXPECT_LE(t.first + t.count, r.first + r.count);
        next += t.count;
    }
    EXPECT_EQ(next, V);
    EXPECT_EQ(info.tranches, static_cast<int>(seen.size()));

    // Ausnahme aus dem Rückruf eines Slots: Typ bleibt erhalten, Engine bleibt benutzbar
    struct CallerFailure {
        int slot;
    };
    try {
        (void)multi.run_on_device(batch, s, g, [](int slot, const DeviceTranche&) {
            if (slot == 1) {
                throw CallerFailure{slot};
            }
        });
        ADD_FAILURE() << "keine Ausnahme";
    } catch (const CallerFailure& e) {
        EXPECT_EQ(e.slot, 1);
    }
    const auto noop = [](int, const DeviceTranche&) {};
    EXPECT_NO_THROW((void)multi.run_on_device(batch, s, g, noop));
    EXPECT_THROW((void)multi.run_on_device(batch, s, 0, noop), std::invalid_argument);
    EXPECT_THROW((void)multi.run_on_device(batch, s, g, {}), std::invalid_argument);
    SystemBatchHost empty = batch;
    empty.resize(0);
    EXPECT_EQ(multi.run_on_device(empty, s, g, noop).tranches, 0);
}

// FP32-Vorfilterung: Rohdaten pro Slot, Entscheidung über den ganzen Batch – bitgleich zur
// Vorfilterung auf einer Engine (auch Kriterium (c), das die beste Variante aller Slots
// braucht, und mit extra_merit).
TEST(MultiTraceEngine, PrefilterBitIdenticalOnLogicalDevices) {
    MKO_REQUIRE_GPU();
    const SystemBatchHost batch = mko::test::multistart_batch(
        mko::test::load_residual_system("testsystem_mehrlinser.json"), 700, 5005);
    const std::vector<double> extra = extra_merit_values(batch.variants());
    PrefilterSettings settings;
    settings.options.criteria.max_merit_factor = 100.0;
    settings.extra_merit = extra;
    TraceEngine single(0);
    const PrefilterResult reference = single.prefilter_variants(batch, settings);
    EXPECT_GT(reference.kept, 0);
    EXPECT_LT(reference.kept, batch.variants());
    for (const auto& w : std::vector<std::vector<double>>{{}, {1.0, 3.0}, {1.0, 1.0, 1.0}}) {
        MultiTraceEngine multi(logical(w.empty() ? 2 : static_cast<int>(w.size()), w));
        EXPECT_TRUE(identical(multi.prefilter_variants(batch, settings), reference));
        expect_ranges(multi, batch.variants());
        EXPECT_EQ(multi.last_run().kind, WorkKind::Prefilter);
    }
}

// Abnahme M7-T3 (Verteilung nach gemessenem Durchsatz): Slot 1 ist künstlich langsam (Tranchen
// zu 8 Varianten, also viele Kernelstarts und Kopien), Slot 0 nicht. Die Kalibrierung beim
// ersten Lauf gibt Slot 0 den größeren Anteil, die Aufteilung folgt ihm, weitere Läufe gehen in
// den gleitenden Mittelwert ein; die Ergebnisse bleiben bitgleich.
TEST(MultiTraceEngine, DistributionFollowsMeasuredThroughput) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 3000, 1234);
    const ResidualLayout layout(batch.topology());
    const TraceSettings s;
    TraceEngine single(0);
    const TraceOutput reference = single.run(batch, TraceMode::Merit, s);
    MultiEngineOptions o = logical(2);
    o.slot_engines = {EngineOptions{},
                      limited_to(run_bytes(batch.topology(), 8, layout, TraceMode::Merit, s))};
    MultiTraceEngine multi(o);
    EXPECT_FALSE(multi.has_estimate(WorkKind::Merit));
    TraceOutput out;
    multi.run(batch, TraceMode::Merit, s, out);
    EXPECT_TRUE(identical(out, reference));
    const mko::cuda::MultiRunInfo first = multi.last_run();
    EXPECT_TRUE(first.calibrated);
    EXPECT_TRUE(multi.has_estimate(WorkKind::Merit));
    EXPECT_FALSE(multi.has_estimate(WorkKind::Residuals));
    ASSERT_EQ(first.shares.size(), 2u);
    std::printf("[ Info     ] Kalibrierung: Anteile %.3f / %.3f, Bereiche %d / %d, Durchsatz "
                "%.3g / %.3g Varianten/s\n",
                first.shares[0], first.shares[1], first.devices[0].range.count,
                first.devices[1].range.count, first.devices[0].throughput,
                first.devices[1].throughput);
    EXPECT_GT(first.shares[0], first.shares[1]);
    EXPECT_GT(first.devices[0].range.count, first.devices[1].range.count);
    EXPECT_NEAR(first.shares[0] + first.shares[1], 1.0, 1e-12);
    for (const mko::cuda::DeviceRunInfo& d : first.devices) {
        EXPECT_GT(d.seconds, 0.0);
        EXPECT_GT(d.throughput, 0.0);
        EXPECT_EQ(d.device, 0);
    }
    EXPECT_GT(first.devices[1].info.tranches, 1);

    bool updated = false;
    for (int run = 0; run < 5; ++run) {
        multi.run(batch, TraceMode::Merit, s, out);
        EXPECT_TRUE(identical(out, reference));
        EXPECT_FALSE(multi.last_run().calibrated);
        updated = updated || multi.last_run().shares_updated;
        const std::vector<double> now = multi.shares(WorkKind::Merit);
        std::printf("[ Info     ] Lauf %d: Anteile danach %.3f / %.3f, Zeiten %.2f / %.2f ms\n",
                    run, now[0], now[1], 1e3 * multi.last_run().devices[0].seconds,
                    1e3 * multi.last_run().devices[1].seconds);
        EXPECT_GT(now[0], now[1]);
    }
    EXPECT_TRUE(updated);

    // Ohne Kalibrierung: gleiche Anteile im ersten Lauf, danach gleitender Mittelwert
    MultiEngineOptions no_cal = o;
    no_cal.calibration_variants = 0;
    MultiTraceEngine plain(no_cal);
    plain.run(batch, TraceMode::Merit, s, out);
    EXPECT_FALSE(plain.last_run().calibrated);
    EXPECT_EQ(plain.last_run().shares, (std::vector<double>{0.5, 0.5}));
    EXPECT_TRUE(identical(out, reference));
    // Explizite Kalibrierung (mit calibration_variants = 0 über den ganzen Batch)
    plain.calibrate(batch, TraceMode::Merit);
    EXPECT_GT(plain.shares(WorkKind::Merit)[0], 0.5);
    EXPECT_FALSE(plain.has_estimate(WorkKind::Residuals));
    // Gesetzte Anteile; Jacobian übernimmt ohne eigene Schätzung die von Residuals
    plain.set_shares(WorkKind::Residuals, Precision::Fp64, std::vector<double>{1.0, 3.0});
    EXPECT_EQ(plain.shares(WorkKind::Residuals), (std::vector<double>{0.25, 0.75}));
    EXPECT_EQ(plain.shares(WorkKind::Jacobian), (std::vector<double>{0.25, 0.75}));
    EXPECT_THROW(plain.set_shares(WorkKind::Residuals, Precision::Fp64, std::vector<double>{1.0}),
                 std::invalid_argument);
}

// Fehler: ungültige Werte im Batch werfen vor jeder Gerätearbeit (Variantennummer des ganzen
// Batches, ohne Geräteangabe); Fehler auf den Geräten tragen Gerät und Slot im Text und
// behalten ihren Typ; danach ist die Engine weiter benutzbar.
TEST(MultiTraceEngine, ErrorsCarryDeviceInfo) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    SystemBatchHost batch = mko::test::perturbed_batch(ls, 600, 4);
    MultiTraceEngine multi(logical(2, {1.0, 1.0}));

    SystemBatchHost bad = batch;
    bad.pupil_radius(500) = -1.0;
    try {
        multi.run(bad, TraceMode::Residuals);
        ADD_FAILURE() << "keine Ausnahme";
    } catch (const std::invalid_argument& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("500"), std::string::npos) << what;
        EXPECT_EQ(what.find("Slot"), std::string::npos) << what;
    }

    // Zu viele Stützstellen pro Feld im Modus MERIT: jede Engine wirft std::invalid_argument
    TraceSettings dense;
    dense.quadrature = mko::ResidualSettings{16, mko::cuda::merit_max_samples(false) / 16 + 1};
    try {
        multi.run(batch, TraceMode::Merit, dense);
        ADD_FAILURE() << "keine Ausnahme";
    } catch (const std::invalid_argument& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("Gerät 0 ("), std::string::npos) << what;
        EXPECT_NE(what.find("Slot 0"), std::string::npos) << what;
        std::printf("[ Info     ] %s\n", what.c_str());
    }
    const TraceOutput ok = multi.run(batch, TraceMode::Merit);
    EXPECT_EQ(ok.merit.variants, 600);

    // Speichermangel auf den Geräten: CudaError mit Geräteangabe
    MultiEngineOptions tiny = logical(2, {1.0, 1.0});
    tiny.engine = limited_to(1024);
    MultiTraceEngine small(tiny);
    try {
        small.run(batch, TraceMode::Residuals);
        ADD_FAILURE() << "keine Ausnahme";
    } catch (const CudaError& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("Slot 0"), std::string::npos) << what;
    }
    // Nur ein Slot beteiligt (Gewicht 0 für Slot 0): Geräteangabe des beteiligten Slots
    tiny.weights = {0.0, 1.0};
    MultiTraceEngine one(tiny);
    try {
        one.run(batch, TraceMode::Residuals);
        ADD_FAILURE() << "keine Ausnahme";
    } catch (const CudaError& e) {
        EXPECT_NE(std::string(e.what()).find("Slot 1"), std::string::npos) << e.what();
    }
}

// Gebündelter LM auf mehreren (logischen) Geräten: alle Läufe bitgleich zu optimize_lm_gpu auf
// einer Engine, mit festen ungleichen und mit gemessenen Anteilen.
TEST(MultiTraceEngine, LmBitIdenticalToSingleGpu) {
    MKO_REQUIRE_GPU();
    const mko::test::FdProblem p = mko::test::lens_bending_problem(mko::test::kLensFocalWeight);
    const int runs = 96;
    const std::vector<double> x0 = mko::test::perturbed_starts(p.set, p.x0, runs, 0.05, 2026);
    mko::LmOptions o;
    o.jacobian.extra = mko::test::paraxial_extra(p);
    o.max_iterations = 80;
    TraceEngine engine(0);
    const mko::LmResult reference = mko::cuda::optimize_lm_gpu(engine, p.set, p.base, x0, o);
    for (const auto& w : std::vector<std::vector<double>>{{1.0, 3.0}, {}}) {
        MultiTraceEngine multi(logical(w.empty() ? 3 : 2, w));
        std::vector<mko::cuda::LmGpuStats> stats;
        const mko::LmResult result =
            mko::cuda::optimize_lm_multi_gpu(multi, p.set, p.base, x0, o, &stats);
        ASSERT_EQ(result.runs.size(), reference.runs.size());
        EXPECT_EQ(result.cycles, reference.cycles);
        for (std::size_t i = 0; i < reference.runs.size(); ++i) {
            EXPECT_TRUE(mko::test::same_run(result.runs[i], reference.runs[i])) << "Lauf " << i;
        }
        ASSERT_EQ(static_cast<int>(stats.size()), multi.size());
        std::int64_t jacobian_runs = 0;
        for (const mko::cuda::LmGpuStats& st : stats) {
            EXPECT_GT(st.jacobian_runs, 0); // jeder Slot hat gerechnet
            jacobian_runs += st.jacobian_runs;
        }
        EXPECT_GE(jacobian_runs, runs);
        std::printf("[ Info     ] LM auf %d Slots (%s): Läufe für J pro Slot", multi.size(),
                    w.empty() ? "gemessen" : "fest 1:3");
        for (const mko::cuda::LmGpuStats& st : stats) {
            std::printf(" %lld", static_cast<long long>(st.jacobian_runs));
        }
        std::printf("\n");
    }
}

// Echte Mehrgerätekonfiguration (RTX 3090 + Tesla P100): jedes sichtbare Gerät liefert für
// denselben Batch bitgleiche Ergebnisse (FP64 und df64, beide Modi, Vorfilterung), und die
// Verteilung auf alle Geräte ist bitgleich zum Lauf auf Gerät 0. Mit nur einem Gerät
// übersprungen; der Maintainer führt den Test auf dem Rechner mit beiden Karten aus:
//   ctest --preset release-cuda -R "MultiTraceEngine\." --output-on-failure
TEST(MultiTraceEngine, RealDevicesBitIdenticalToSingle) {
    MKO_REQUIRE_GPU();
    const std::vector<int> devices = mko::cuda::usable_devices();
    if (devices.size() < 2) {
        GTEST_SKIP() << "Nur ein CUDA-Gerät sichtbar; auf RTX 3090 + Tesla P100 ausführen";
    }
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 2000, 4242);
    const std::vector<double> extra = extra_merit_values(batch.variants());
    PrefilterSettings ps;
    ps.options.criteria.max_merit_factor = 100.0;
    struct Case {
        TraceMode mode;
        TraceSettings settings;
    };
    std::vector<Case> cases;
    for (const Precision p : {Precision::Fp64, Precision::Df64}) {
        TraceSettings r;
        r.opd = true;
        r.extrema = true;
        r.precision = p;
        cases.push_back({TraceMode::Residuals, r});
        TraceSettings m;
        m.extrema = true;
        m.extra_merit = extra;
        m.precision = p;
        cases.push_back({TraceMode::Merit, m});
    }
    TraceEngine first(devices.front());
    std::vector<TraceOutput> reference;
    for (const Case& c : cases) {
        reference.push_back(first.run(batch, c.mode, c.settings));
    }
    const PrefilterResult ref_pre = first.prefilter_variants(batch, ps);
    for (std::size_t k = 1; k < devices.size(); ++k) {
        TraceEngine other(devices[k]);
        for (std::size_t c = 0; c < cases.size(); ++c) {
            EXPECT_TRUE(identical(other.run(batch, cases[c].mode, cases[c].settings), reference[c]))
                << "Gerät " << devices[k] << " (" << other.device_name() << ") gegen Gerät "
                << devices.front() << " (" << first.device_name() << "), Fall " << c << " ("
                << mko::cuda::to_string(cases[c].mode) << ", "
                << mko::cuda::to_string(cases[c].settings.precision) << ")";
        }
        EXPECT_TRUE(identical(other.prefilter_variants(batch, ps), ref_pre))
            << "Vorfilterung, Gerät " << devices[k];
    }
    MultiTraceEngine multi; // alle Geräte, gemessene Anteile
    for (int run = 0; run < 3; ++run) {
        for (std::size_t c = 0; c < cases.size(); ++c) {
            EXPECT_TRUE(identical(multi.run(batch, cases[c].mode, cases[c].settings), reference[c]))
                << "Verteilung auf alle Geräte, Fall " << c;
            const mko::cuda::MultiRunInfo& info = multi.last_run();
            std::printf("[ Info     ] Fall %zu: Anteile", c);
            for (std::size_t k = 0; k < info.devices.size(); ++k) {
                std::printf(" %s %.3f (%d Varianten, %.3g/s)",
                            multi.engine(static_cast<int>(k)).device_name().c_str(), info.shares[k],
                            info.devices[k].range.count, info.devices[k].throughput);
            }
            std::printf("\n");
        }
        EXPECT_TRUE(identical(multi.prefilter_variants(batch, ps), ref_pre));
    }
}

// Fingerabdruck (FNV-1a über die Ergebnisbits) eines festen Batches pro sichtbarem Gerät, zum
// Vergleich zwischen Rechnern, Architekturen und Builds (z. B. JIT aus compute_60-PTX auf der
// RTX 3090 gegen den nativen sm_86-Build, docs/architektur.md, "Mehrere GPUs (M7-T3)"). Prüft
// nichts; gleiche Zahlen bedeuten bitgleiche Ergebnisse.
TEST(MultiTraceEngine, ResultFingerprint) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 1000, 4242);
    for (const int d : mko::cuda::usable_devices()) {
        TraceEngine e(d);
        std::printf("[ Info     ] Fingerabdruck Gerät %d (%s):", d, e.device_name().c_str());
        for (const Precision p : {Precision::Fp64, Precision::Df64}) {
            TraceSettings r;
            r.opd = true;
            r.extrema = true;
            r.precision = p;
            const TraceOutput res = e.run(batch, TraceMode::Residuals, r);
            std::uint64_t h = fnv1a(res.residuals.residuals);
            h = fnv1a(res.residuals.opd, h);
            h = fnv1a(res.residuals.status, h);
            h = fnv1a(res.residuals.extrema.values, h);
            TraceSettings m;
            m.precision = p;
            const TraceOutput mer = e.run(batch, TraceMode::Merit, m);
            const std::uint64_t hm =
                fnv1a(mer.merit.hard_failed, fnv1a(mer.merit.failed, fnv1a(mer.merit.merit)));
            std::printf(" %s RESIDUALS %016llx MERIT %016llx", mko::cuda::to_string(p),
                        static_cast<unsigned long long>(h), static_cast<unsigned long long>(hm));
        }
        const PrefilterResult pre = e.prefilter_variants(batch);
        std::printf(" PREFILTER %016llx\n", static_cast<unsigned long long>(fnv1a(
                                                pre.stats.optical_length, fnv1a(pre.stats.merit))));
    }
}
