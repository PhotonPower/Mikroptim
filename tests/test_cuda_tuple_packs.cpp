// Tupelpakete der Kernel RESIDUALS und MERIT (M7-T4, nur mit MIKROPTIM_ENABLE_CUDA): Die
// Ergebnisse sind für jede Paketgröße bitgleich, insbesondere zu pack_rays = 1 (ein Tupel pro
// Block à 128 Threads wie vor M7-T4), in FP64 und df64, mit hartem und weichem Versagen,
// Geisterplatten, Max-Reduktionen und Quadraturen mit mehr als 128 Stützstellen pro Feld.
// Planung: test_tuple_packs.cpp.

#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/device_system_batch.hpp"
#include "mikroptim/cuda/merit.hpp"
#include "mikroptim/cuda/precision.hpp"
#include "mikroptim/cuda/residuals.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "residual_test_systems.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using mko::MeritResult;
using mko::ResidualLayout;
using mko::ResidualResult;
using mko::ResidualSettings;
using mko::SystemBatchHost;
using mko::cuda::DeviceMerit;
using mko::cuda::DeviceResiduals;
using mko::cuda::DeviceSystemBatch;
using mko::cuda::MeritRunOptions;
using mko::cuda::Precision;
using mko::cuda::ResidualRunOptions;
using mko::cuda::TraceEngine;
using mko::cuda::TraceMode;
using mko::cuda::TraceSettings;

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

} // namespace

// Abnahme M7-T4: Paketgrößen 1 (Abbildung vor M7-T4), 128, 512 (Standard) und 4096 liefern
// bitgleiche Residuen, OPD, Status, Merit, Zähler und Max-Reduktionen.
TEST(CudaTuplePacks, ResultsAreBitIdenticalForAllPackSizes) {
    MKO_REQUIRE_GPU();
    const char* systems[] = {"einzellinse_bk7.json", "testsystem_mehrlinser.json",
                             "einzellinse_geisterplatte.json"};
    const ResidualSettings quads[] = {{10, 8, mko::FailureMode::Soft},
                                      {6, 5, mko::FailureMode::Hard},
                                      {12, 40, mko::FailureMode::Soft}};
    const int sizes[] = {1, 128, mko::cuda::kDefaultPackRays, 4096};
    std::uint64_t seed = 7400;
    int compared = 0;
    for (const char* file : systems) {
        const mko::LoadedSystem ls = mko::test::load_residual_system(file);
        const SystemBatchHost batch = mko::test::perturbed_batch(ls, 200, ++seed);
        const DeviceSystemBatch device(batch);
        for (const ResidualSettings& q : quads) {
            const ResidualLayout layout(batch.topology(), q);
            for (const Precision p : {Precision::Fp64, Precision::Df64}) {
                SCOPED_TRACE(std::string(file) + " " + std::to_string(q.rings) + "x" +
                             std::to_string(q.spokes) + (p == Precision::Df64 ? " df64" : " fp64"));
                ResidualResult r0;
                MeritResult m0;
                for (const int size : sizes) {
                    ResidualRunOptions ro;
                    ro.opd = true;
                    ro.extrema = true;
                    ro.precision = p;
                    ro.pack_rays = size;
                    DeviceResiduals residuals;
                    residuals.run(device, layout, ro);
                    const ResidualResult r = residuals.download();
                    MeritRunOptions mo;
                    mo.extrema = true;
                    mo.precision = p;
                    mo.pack_rays = size;
                    DeviceMerit merit;
                    merit.run(device, layout, mo);
                    const MeritResult m = merit.download();
                    if (size == 1) {
                        r0 = r;
                        m0 = m;
                        EXPECT_GT(r.failed_count(), 0u) << "Versagenspfad nicht geprüft";
                        continue;
                    }
                    SCOPED_TRACE("pack_rays = " + std::to_string(size));
                    EXPECT_TRUE(same_bytes(r.residuals, r0.residuals));
                    EXPECT_TRUE(same_bytes(r.opd, r0.opd));
                    EXPECT_TRUE(same_bytes(r.status, r0.status));
                    EXPECT_TRUE(same_bytes(r.extrema.values, r0.extrema.values));
                    EXPECT_TRUE(same_bytes(m.merit, m0.merit));
                    EXPECT_TRUE(same_bytes(m.failed, m0.failed));
                    EXPECT_TRUE(same_bytes(m.hard_failed, m0.hard_failed));
                    EXPECT_TRUE(same_bytes(m.extrema.values, m0.extrema.values));
                    EXPECT_TRUE(same_bytes(m.extrema.values, r.extrema.values));
                    ++compared;
                }
            }
        }
    }
    std::printf("[ Info     ] %d Läufe bitgleich zu pack_rays = 1\n", compared);
}

// Ungültige Paketgröße: std::invalid_argument in DeviceResiduals, DeviceMerit und TraceEngine.
TEST(CudaTuplePacks, InvalidPackSizeThrows) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("einzellinse_bk7.json");
    const SystemBatchHost batch = mko::make_system_batch(ls, 2);
    const ResidualLayout layout(batch.topology());
    const DeviceSystemBatch device(batch);
    ResidualRunOptions ro;
    ro.pack_rays = 0;
    DeviceResiduals residuals;
    EXPECT_THROW(residuals.run(device, layout, ro), std::invalid_argument);
    MeritRunOptions mo;
    mo.pack_rays = -3;
    DeviceMerit merit;
    EXPECT_THROW(merit.run(device, layout, mo), std::invalid_argument);
    TraceEngine engine(0);
    TraceSettings s;
    s.pack_rays = 0;
    EXPECT_THROW((void)engine.run(batch, TraceMode::Residuals, s), std::invalid_argument);
    EXPECT_THROW((void)engine.run(batch, TraceMode::Merit, s), std::invalid_argument);
    // Nach den Fehlern funktionieren die Puffer weiter
    EXPECT_NO_THROW(residuals.run(device, layout));
    EXPECT_NO_THROW(merit.run(device, layout));
}

// merit_max_samples (M7-T4): Im weichen Modus liegen zusätzlich die Strafresiduen im Shared
// Memory (17 statt 9 Byte pro Strahl); beide Grenzen sind erreichbar und werden durchgesetzt.
TEST(CudaTuplePacks, MeritSampleLimitDependsOnFailureMode) {
    MKO_REQUIRE_GPU();
    const int soft =
        mko::cuda::merit_max_samples(true, 0, 0, Precision::Fp64, mko::FailureMode::Soft);
    const int hard =
        mko::cuda::merit_max_samples(true, 0, 0, Precision::Fp64, mko::FailureMode::Hard);
    EXPECT_GT(soft, 1000);
    EXPECT_GT(hard, soft);
    std::printf("[ Info     ] MERIT mit Extrema: höchstens %d (weich) bzw. %d (hart) Stützstellen "
                "pro Feld\n",
                soft, hard);
    const mko::LoadedSystem ls = mko::test::load_residual_system("einzellinse_bk7.json");
    const SystemBatchHost batch = mko::make_system_batch(ls, 2);
    const DeviceSystemBatch device(batch);
    MeritRunOptions mo;
    mo.extrema = true;
    DeviceMerit merit;
    for (const auto mode : {mko::FailureMode::Soft, mko::FailureMode::Hard}) {
        const int limit = mode == mko::FailureMode::Soft ? soft : hard;
        EXPECT_NO_THROW(
            merit.run(device, ResidualLayout(batch.topology(), {16, limit / 16, mode}), mo));
        EXPECT_THROW(
            merit.run(device, ResidualLayout(batch.topology(), {16, limit / 16 + 1, mode}), mo),
            std::invalid_argument);
    }
}
