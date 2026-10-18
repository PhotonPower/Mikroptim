// Aufteilung auf mehrere GPUs (M7-T3), ohne CUDA: split_variants (zusammenhängende Bereiche,
// Granularität, Anteile) und update_shares (gleitender Mittelwert nach gemessenem Durchsatz).
// Der Header der MultiTraceEngine ist reines C++ und läuft auch im CPU-Build.

#include "mikroptim/cuda/multi_trace_engine.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

using mko::cuda::DeviceRange;
using mko::cuda::MultiEngineOptions;
using mko::cuda::MultiTraceEngine;
using mko::cuda::split_variants;
using mko::cuda::update_shares;

static_assert(!std::is_copy_constructible_v<MultiTraceEngine>);
static_assert(std::is_nothrow_move_constructible_v<MultiTraceEngine>);

namespace {

// Bereiche lückenlos, in Slot-Reihenfolge, über [0, V); Grenzen Vielfache von g.
void expect_partition(const std::vector<DeviceRange>& r, int variants, int granularity) {
    int next = 0;
    for (const DeviceRange& d : r) {
        EXPECT_EQ(d.first, next);
        EXPECT_GE(d.count, 0);
        if (d.first < variants) {
            EXPECT_EQ(d.first % granularity, 0); // leere Bereiche am Ende beginnen bei V
        }
        if (d.first + d.count < variants) {
            EXPECT_EQ(d.count % granularity, 0);
        }
        next = d.first + d.count;
    }
    EXPECT_EQ(next, variants);
}

} // namespace

TEST(MultiGpuSplit, ProportionalContiguousRanges) {
    const std::vector<double> equal{1.0, 1.0};
    EXPECT_EQ(split_variants(10, 1, equal), (std::vector<DeviceRange>{{0, 5}, {5, 5}}));
    const std::vector<double> three_to_one{3.0, 1.0};
    EXPECT_EQ(split_variants(100, 1, three_to_one), (std::vector<DeviceRange>{{0, 75}, {75, 25}}));
    // nicht normierte Anteile, drei Slots
    const std::vector<double> w{0.2, 0.5, 0.3};
    EXPECT_EQ(split_variants(1000, 1, w),
              (std::vector<DeviceRange>{{0, 200}, {200, 500}, {700, 300}}));
    // Anteil 0 schließt einen Slot aus; Grenze round(4,5) = 5 (halb von null weg)
    const std::vector<double> zero{1.0, 0.0, 1.0};
    EXPECT_EQ(split_variants(9, 1, zero), (std::vector<DeviceRange>{{0, 5}, {5, 0}, {5, 4}}));
    // leerer Batch und ein Slot
    EXPECT_EQ(split_variants(0, 3, equal), (std::vector<DeviceRange>{{0, 0}, {0, 0}}));
    const std::vector<double> one{2.5};
    EXPECT_EQ(split_variants(17, 4, one), (std::vector<DeviceRange>{{0, 17}}));
    // weniger Varianten als Slots
    const std::vector<double> four{1.0, 1.0, 1.0, 1.0};
    expect_partition(split_variants(2, 1, four), 2, 1);
}

// Blöcke aus g Varianten (z. B. ein FD-Lauf mit 2P + 1 Varianten) werden nie geteilt; nur der
// letzte Block darf kürzer sein.
TEST(MultiGpuSplit, GranularityKeepsBlocksTogether) {
    const std::vector<double> w{1.0, 2.0, 1.0};
    const std::vector<DeviceRange> r = split_variants(27 * 10 + 5, 27, w);
    expect_partition(r, 27 * 10 + 5, 27);
    // 11 Blöcke: Grenzen round(11/4) = 3, round(33/4) = 8
    EXPECT_EQ(r[0], (DeviceRange{0, 81}));
    EXPECT_EQ(r[1], (DeviceRange{81, 135}));
    EXPECT_EQ(r[2], (DeviceRange{216, 59}));

    std::mt19937_64 rng(2026);
    std::uniform_real_distribution<double> share(0.0, 1.0);
    for (int trial = 0; trial < 500; ++trial) {
        const int n = 1 + static_cast<int>(rng() % 5);
        std::vector<double> s(static_cast<std::size_t>(n));
        for (double& x : s) {
            x = share(rng) + 1e-3;
        }
        const int g = 1 + static_cast<int>(rng() % 30);
        const int v = static_cast<int>(rng() % 2000);
        expect_partition(split_variants(v, g, s), v, g);
    }
}

TEST(MultiGpuSplit, InvalidArgumentsThrow) {
    const std::vector<double> ok{1.0, 1.0};
    EXPECT_THROW((void)split_variants(-1, 1, ok), std::invalid_argument);
    EXPECT_THROW((void)split_variants(10, 0, ok), std::invalid_argument);
    EXPECT_THROW((void)split_variants(10, 1, std::vector<double>{}), std::invalid_argument);
    EXPECT_THROW((void)split_variants(10, 1, std::vector<double>{1.0, -1.0}),
                 std::invalid_argument);
    EXPECT_THROW((void)split_variants(10, 1, std::vector<double>{0.0, 0.0}), std::invalid_argument);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void)split_variants(10, 1, std::vector<double>{1.0, nan}), std::invalid_argument);
}

// Gleitender Mittelwert: Mit α = 1 sind die neuen Anteile proportional zum Durchsatz; die Summe
// bleibt 1; Läufe mit zu kurzen Zeiten oder nur einem beteiligten Slot ändern nichts.
TEST(MultiGpuSplit, UpdateSharesFollowsThroughput) {
    std::vector<double> s{0.5, 0.5};
    const std::vector<DeviceRange> r{{0, 100}, {100, 100}};
    // Slot 0 doppelt so schnell (gleich viele Varianten in der halben Zeit)
    EXPECT_TRUE(update_shares(s, r, std::vector<double>{0.05, 0.1}, 1.0, 0.005));
    EXPECT_DOUBLE_EQ(s[0], 2.0 / 3.0);
    EXPECT_DOUBLE_EQ(s[1], 1.0 / 3.0);
    // Mit α = 0,5 halber Schritt zu (0,5; 0,5) zurück, wenn beide gleich schnell sind
    EXPECT_TRUE(update_shares(s, r, std::vector<double>{0.1, 0.1}, 0.5, 0.005));
    EXPECT_DOUBLE_EQ(s[0], 0.5 * (2.0 / 3.0) + 0.25);
    EXPECT_DOUBLE_EQ(s[0] + s[1], 1.0);
    // Stationär: Aufteilung nach Durchsatz führt zu gleichen Zeiten und ändert nichts mehr
    std::vector<double> st{2.0 / 3.0, 1.0 / 3.0};
    const std::vector<DeviceRange> rs{{0, 200}, {200, 100}};
    EXPECT_TRUE(update_shares(st, rs, std::vector<double>{0.1, 0.1}, 0.3, 0.005));
    EXPECT_NEAR(st[0], 2.0 / 3.0, 1e-15);

    // zu kurz gemessen
    std::vector<double> u{0.5, 0.5};
    EXPECT_FALSE(update_shares(u, r, std::vector<double>{0.001, 0.1}, 1.0, 0.005));
    EXPECT_EQ(u, (std::vector<double>{0.5, 0.5}));
    // nur ein Slot mit Arbeit
    const std::vector<DeviceRange> single{{0, 100}, {100, 0}};
    EXPECT_FALSE(update_shares(u, single, std::vector<double>{0.1, 0.0}, 1.0, 0.005));
    // nicht beteiligter Slot behält seinen Anteil, die Summe bleibt 1
    std::vector<double> t{0.25, 0.25, 0.5};
    const std::vector<DeviceRange> rt{{0, 50}, {50, 0}, {50, 100}};
    EXPECT_TRUE(update_shares(t, rt, std::vector<double>{0.1, 0.0, 0.1}, 1.0, 0.005));
    EXPECT_DOUBLE_EQ(t[1], 0.25);
    EXPECT_DOUBLE_EQ(t[0] + t[2], 0.75);
    EXPECT_DOUBLE_EQ(t[0], 0.25); // Durchsatz 500/s gegen 1.000/s: 0,75 · 1/3
    // Fehler
    EXPECT_THROW(update_shares(u, r, std::vector<double>{0.1}, 1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(update_shares(u, r, std::vector<double>{0.1, 0.1}, 0.0, 0.0),
                 std::invalid_argument);
    EXPECT_THROW(update_shares(u, r, std::vector<double>{0.1, 0.1}, 1.5, 0.0),
                 std::invalid_argument);
}

TEST(MultiGpuSplit, OptionDefaults) {
    const MultiEngineOptions o;
    EXPECT_TRUE(o.devices.empty());
    EXPECT_TRUE(o.weights.empty());
    EXPECT_TRUE(o.slot_engines.empty());
    EXPECT_EQ(o.calibration_variants, 256);
    EXPECT_GT(o.smoothing, 0.0);
    EXPECT_LE(o.smoothing, 1.0);
    EXPECT_STREQ(mko::cuda::to_string(mko::cuda::WorkKind::Jacobian), "JACOBIAN");
}
