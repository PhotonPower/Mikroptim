// Modus ADJOINT, Grundgerüst auf der GPU (M5-T1): Vorwärtsdurchlauf mit Aufzeichnung im Local
// Memory. Residuen, Status und OPD bitgleich zum Kernel RESIDUALS (mit und ohne Aufzeichnung),
// Aufzeichnung gegen die CPU-Referenz, vollständige Aufzeichnung, Checkpoints und
// Kapazitätsklassen untereinander bitgleich, wiederholte Läufe bitgleich, Prüfsumme gegen die
// heruntergeladene Aufzeichnung, Ressourcen der Kernel-Varianten. CPU-Teil:
// test_adjoint_record.cpp.

#include "adjoint_test_tools.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cuda/adjoint.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/device_system_batch.hpp"
#include "mikroptim/cuda/residuals.hpp"
#include "mikroptim/trace/adjoint_record.hpp"
#include "residual_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using mko::FailureMode;
using mko::ResidualLayout;
using mko::ResidualResult;
using mko::ResidualSettings;
using mko::SystemBatchHost;
using mko::TraceStatus;
using mko::cuda::AdjointForwardOptions;
using mko::cuda::AdjointForwardResult;
using mko::cuda::DeviceAdjointForward;
using mko::cuda::DeviceResiduals;
using mko::cuda::DeviceSystemBatch;
using mko::cuda::Precision;
using mko::cuda::RecordStorage;

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

ResidualSettings settings_with(FailureMode mode) {
    ResidualSettings s;
    s.rings = 6;
    s.spokes = 6;
    s.failure = mode;
    return s;
}

// Prüffälle: gestörte Varianten (auch extreme mit hart und weich versagten Strahlen),
// Geisterplatten, das Startsystem des weichen Versagens und Stapel aus S = 12, 24, 64 Flächen
// (alle drei Kapazitätsklassen).
struct GpuCase {
    std::string name;
    SystemBatchHost batch;
    ResidualSettings settings;
};

std::vector<GpuCase> gpu_cases() {
    std::vector<GpuCase> c;
    const auto mehrlinser = mko::test::load_residual_system("testsystem_mehrlinser.json");
    c.push_back({"Mehrlinser weich", mko::test::perturbed_batch(mehrlinser, 120, 20261022),
                 settings_with(FailureMode::Soft)});
    c.push_back({"Mehrlinser hart", mko::test::perturbed_batch(mehrlinser, 120, 20261022),
                 settings_with(FailureMode::Hard)});
    c.push_back(
        {"Geisterplatten weich",
         mko::test::perturbed_batch(
             mko::test::load_residual_system("einzellinse_geisterplatte.json"), 80, 20261023),
         settings_with(FailureMode::Soft)});
    c.push_back(
        {"weiches_versagen_start weich",
         mko::make_system_batch(mko::test::load_residual_system("weiches_versagen_start.json"), 1),
         settings_with(FailureMode::Soft)});
    c.push_back({"S = 12", mko::test::many_surface_batch(30, 20261024, 12),
                 settings_with(FailureMode::Soft)});
    c.push_back({"S = 24", mko::test::many_surface_batch(30, 20261025, 24),
                 settings_with(FailureMode::Hard)});
    c.push_back({"S = 64", mko::test::many_surface_batch(30, 20261026, 64),
                 settings_with(FailureMode::Soft)});
    return c;
}

AdjointForwardResult run_adjoint(const DeviceSystemBatch& batch, const ResidualLayout& layout,
                                 RecordStorage storage, Precision precision, bool states,
                                 int capacity = 0) {
    AdjointForwardOptions o;
    o.storage = storage;
    o.precision = precision;
    o.states = states;
    o.opd = true;
    o.capacity = capacity;
    return mko::cuda::run_adjoint_forward(batch, layout, o);
}

const char* precision_name(Precision p) {
    return mko::cuda::to_string(p);
}

} // namespace

// Die Aufzeichnung ändert die Numerik im Kernel nicht: Residuen (Wellenfront- und
// Strafresiduen), Status, OPD und OPL sind mit vollständiger Aufzeichnung und mit Checkpoints
// bitgleich zum Lauf ohne Aufzeichnung, in FP64 und df64; die Prüfsummen von vollständiger
// Aufzeichnung und Checkpoints ebenfalls. Gegen den Kernel RESIDUALS (mit FMA-Kontraktion; ADJOINT
// rechnet ohne, docs/architektur.md) gleicher Status und OPD innerhalb der Toleranzen CPU/GPU,
// in df64 (nie kontrahiert) bitgleich.
TEST(CudaAdjoint, RecordingLeavesResultsBitIdentical) {
    MKO_REQUIRE_GPU();
    std::size_t rays = 0;
    std::size_t failed = 0;
    double worst_ok = 0.0;
    double worst_soft = 0.0;
    double worst_penalty = 0.0;
    for (const GpuCase& c : gpu_cases()) {
        const DeviceSystemBatch device(c.batch);
        const ResidualLayout layout(c.batch.topology(), c.settings);
        for (const Precision precision : {Precision::Fp64, Precision::Df64}) {
            DeviceResiduals residuals;
            mko::cuda::ResidualRunOptions ro;
            ro.opd = true;
            ro.precision = precision;
            residuals.run(device, layout, ro);
            const ResidualResult ref = residuals.download();
            const AdjointForwardResult none =
                run_adjoint(device, layout, RecordStorage::None, precision, false);
            const AdjointForwardResult full =
                run_adjoint(device, layout, RecordStorage::Full, precision, false);
            const AdjointForwardResult check =
                run_adjoint(device, layout, RecordStorage::Checkpoint, precision, false);
            for (const AdjointForwardResult* r : {&full, &check}) {
                const std::string what = c.name + " " + precision_name(precision) + " " +
                                         mko::cuda::to_string(r->storage);
                EXPECT_TRUE(same_bytes(r->residuals.residuals, none.residuals.residuals)) << what;
                EXPECT_TRUE(same_bytes(r->residuals.status, none.residuals.status)) << what;
                EXPECT_TRUE(same_bytes(r->residuals.opd, none.residuals.opd)) << what;
                EXPECT_TRUE(same_bytes(r->opl, none.opl)) << what;
            }
            EXPECT_TRUE(same_bytes(full.checksum, check.checksum)) << c.name;
            EXPECT_EQ(none.capacity, 0);
            EXPECT_EQ(full.capacity, mko::record_capacity_class(c.batch.surfaces()));
            // gegen RESIDUALS
            ASSERT_TRUE(same_bytes(none.residuals.status, ref.status)) << c.name;
            if (precision == Precision::Df64) {
                EXPECT_TRUE(same_bytes(none.residuals.residuals, ref.residuals)) << c.name;
                EXPECT_TRUE(same_bytes(none.residuals.opd, ref.opd)) << c.name;
            }
            const std::int64_t R = layout.rays_per_variant();
            for (std::size_t i = 0; i < ref.opd.size(); ++i) {
                const auto st = static_cast<TraceStatus>(ref.status[i]);
                const double e = std::abs(none.residuals.opd[i] - ref.opd[i]);
                if (st == TraceStatus::Ok) {
                    const double tol =
                        std::max(mko::test::kCpuGpuFp64,
                                 mko::test::kCpuGpuFp64Relative * std::abs(ref.opd[i]));
                    worst_ok = std::max(worst_ok, e / tol);
                    ASSERT_LE(e, tol) << c.name << " Strahl " << i;
                } else if (mko::is_soft_failure(st)) {
                    worst_soft = std::max(worst_soft, e / mko::test::kCpuGpuSoftTuple);
                    ASSERT_LE(e, mko::test::kCpuGpuSoftTuple) << c.name << " Strahl " << i;
                }
            }
            if (layout.soft_failure()) {
                for (int v = 0; v < ref.variants; ++v) {
                    for (std::int64_t i = 0; i < R; ++i) {
                        const double a = none.residuals.penalty(v, i);
                        const double b = ref.penalty(v, i);
                        const double e = std::abs(a - b) / std::max(1e-300, std::abs(b));
                        if (b != 0.0) {
                            worst_penalty = std::max(worst_penalty, e);
                            ASSERT_LE(e, mko::test::kCpuGpuPenaltyRelative) << c.name;
                        } else {
                            ASSERT_EQ(a, 0.0) << c.name;
                        }
                    }
                }
            }
            rays += ref.status.size();
            failed += ref.failed_count();
        }
    }
    std::printf("[ Info     ] %zu Strahlen (FP64 und df64), davon %zu versagt: mit und ohne "
                "Aufzeichnung bitgleich; gegen RESIDUALS FP64 OPD / Toleranz max %.3f (gültig), "
                "%.3f (weich versagt), Strafen relativ %.2e\n",
                rays, failed, worst_ok, worst_soft, worst_penalty);
}

// Aufzeichnung auf der GPU (FP64) gegen die CPU-Referenz (AdjointRecord über batch_ray_opl):
// Status, Zahl der Zustände und Zweige gleich, Zustände und s_ref innerhalb der Toleranzen
// CPU/GPU (gültige Strahlen kCpuGpuFp64 bzw. relativ kCpuGpuFp64Relative; versagte Strahlen sind
// fortgesetzt und schlechter konditioniert: kCpuGpuSoftTuple).
TEST(CudaAdjoint, RecordMatchesCpu) {
    MKO_REQUIRE_GPU();
    double worst_ok = 0.0;
    double worst_failed = 0.0;
    std::size_t states_checked = 0;
    for (const GpuCase& c : gpu_cases()) {
        const DeviceSystemBatch device(c.batch);
        const ResidualLayout layout(c.batch.topology(), c.settings);
        const mko::test::CpuRecords cpu = mko::test::cpu_records(c.batch, layout);
        for (const RecordStorage storage : {RecordStorage::Full, RecordStorage::Checkpoint}) {
            const AdjointForwardResult gpu =
                run_adjoint(device, layout, storage, Precision::Fp64, true);
            const auto S = static_cast<std::size_t>(gpu.surfaces);
            ASSERT_EQ(gpu.rays(), cpu.status.size());
            for (std::size_t i = 0; i < gpu.rays(); ++i) {
                ASSERT_EQ(gpu.residuals.status[i], cpu.status[i]) << c.name << " Strahl " << i;
                ASSERT_EQ(gpu.count[i], cpu.count[i]) << c.name << " Strahl " << i;
                const bool ok = static_cast<TraceStatus>(cpu.status[i]) == TraceStatus::Ok;
                const double abs_tol = ok ? mko::test::kCpuGpuFp64 : mko::test::kCpuGpuSoftTuple;
                const double rel_tol =
                    ok ? mko::test::kCpuGpuFp64Relative : mko::test::kCpuGpuSoftTuple;
                for (std::size_t s = 0; s < static_cast<std::size_t>(cpu.count[i]); ++s) {
                    ASSERT_EQ(gpu.branch[i * S + s], cpu.branch[i * S + s])
                        << c.name << " Strahl " << i << " s=" << s;
                    for (std::size_t j = 0; j < 6; ++j) {
                        const double a = gpu.states[(i * S + s) * 6 + j];
                        const double b = cpu.states[(i * S + s) * 6 + j];
                        const double e = std::abs(a - b);
                        const double tol = std::max(abs_tol, rel_tol * std::abs(b));
                        double& worst = ok ? worst_ok : worst_failed;
                        worst = std::max(worst, e / tol);
                        ASSERT_LE(e, tol) << c.name << " Strahl " << i << " s=" << s << " j=" << j;
                    }
                    ++states_checked;
                }
                ASSERT_EQ(gpu.image_branch[i], cpu.image_branch[i]) << c.name;
                const double e = std::abs(gpu.image_s[i] - cpu.image_s[i]);
                ASSERT_LE(e, std::max(abs_tol, rel_tol * std::abs(cpu.image_s[i]))) << c.name;
            }
        }
    }
    std::printf("[ Info     ] %zu Zustände gegen die CPU: größte Abweichung / Toleranz %.3f "
                "(gültige Strahlen), %.3f (versagte)\n",
                states_checked, worst_ok, worst_failed);
}

// Vollständige Aufzeichnung (automatische Kapazitätsklasse und C = 64) und Checkpoints
// (automatisch und C = 64) liefern bitgleiche Zustände, Zweige, Zähler, Referenzschritte und
// Prüfsummen, in FP64 und df64: Das Neu-Vorwärtsrechnen der Abschnitte im Kernel ist bitgleich
// zum ursprünglichen Durchlauf.
TEST(CudaAdjoint, CheckpointsAndCapacityClassesBitIdentical) {
    MKO_REQUIRE_GPU();
    for (const GpuCase& c : gpu_cases()) {
        const DeviceSystemBatch device(c.batch);
        const ResidualLayout layout(c.batch.topology(), c.settings);
        for (const Precision precision : {Precision::Fp64, Precision::Df64}) {
            const AdjointForwardResult ref =
                run_adjoint(device, layout, RecordStorage::Full, precision, true);
            const AdjointForwardResult variants[] = {
                run_adjoint(device, layout, RecordStorage::Full, precision, true, 64),
                run_adjoint(device, layout, RecordStorage::Checkpoint, precision, true),
                run_adjoint(device, layout, RecordStorage::Checkpoint, precision, true, 64)};
            for (const AdjointForwardResult& r : variants) {
                const std::string what = c.name + " " + precision_name(precision) + " " +
                                         mko::cuda::to_string(r.storage) +
                                         " C = " + std::to_string(r.capacity);
                EXPECT_TRUE(same_bytes(r.states, ref.states)) << what;
                EXPECT_TRUE(same_bytes(r.branch, ref.branch)) << what;
                EXPECT_TRUE(same_bytes(r.count, ref.count)) << what;
                EXPECT_TRUE(same_bytes(r.image_s, ref.image_s)) << what;
                EXPECT_TRUE(same_bytes(r.image_branch, ref.image_branch)) << what;
                EXPECT_TRUE(same_bytes(r.checksum, ref.checksum)) << what;
                EXPECT_TRUE(same_bytes(r.residuals.residuals, ref.residuals.residuals)) << what;
            }
        }
    }
}

// Determinismus: Zwei Läufe mit gleichen Eingaben sind bitgleich (Residuen, OPL, Prüfsummen,
// Aufzeichnung), für beide Speicherformen und beide Zahlentypen; die Prüfsumme stimmt mit der
// auf dem Host aus der heruntergeladenen Aufzeichnung gebildeten überein.
TEST(CudaAdjoint, RepeatedRunsBitIdenticalAndChecksumMatchesRecord) {
    MKO_REQUIRE_GPU();
    for (const GpuCase& c : gpu_cases()) {
        const DeviceSystemBatch device(c.batch);
        const ResidualLayout layout(c.batch.topology(), c.settings);
        for (const Precision precision : {Precision::Fp64, Precision::Df64}) {
            for (const RecordStorage storage : {RecordStorage::Full, RecordStorage::Checkpoint}) {
                DeviceAdjointForward d;
                AdjointForwardOptions o;
                o.storage = storage;
                o.precision = precision;
                o.states = true;
                d.run(device, layout, o);
                const AdjointForwardResult a = d.download();
                d.run(device, layout, o); // Puffer wiederverwendet
                const AdjointForwardResult b = d.download();
                EXPECT_TRUE(same_bytes(a.residuals.residuals, b.residuals.residuals)) << c.name;
                EXPECT_TRUE(same_bytes(a.opl, b.opl)) << c.name;
                EXPECT_TRUE(same_bytes(a.checksum, b.checksum)) << c.name;
                EXPECT_TRUE(same_bytes(a.states, b.states)) << c.name;
                const auto S = static_cast<std::size_t>(a.surfaces);
                for (std::size_t i = 0; i < a.rays(); ++i) {
                    const std::uint64_t h = mko::test::host_record_checksum(
                        a.states.data() + i * S * 6, a.branch.data() + i * S, a.count[i],
                        a.image_s[i], a.image_branch[i]);
                    ASSERT_EQ(h, a.checksum[i]) << c.name << " Strahl " << i;
                }
            }
        }
    }
}

// Ungültige Optionen werden abgewiesen.
TEST(CudaAdjoint, RejectsInvalidOptions) {
    MKO_REQUIRE_GPU();
    const SystemBatchHost batch = mko::test::many_surface_batch(2, 1, 24);
    const DeviceSystemBatch device(batch);
    const ResidualLayout layout(batch.topology(), settings_with(FailureMode::Soft));
    AdjointForwardOptions o;
    o.capacity = 16; // < S = 24
    EXPECT_THROW(mko::cuda::run_adjoint_forward(device, layout, o), std::invalid_argument);
    o.capacity = 40; // keine Klasse
    EXPECT_THROW(mko::cuda::run_adjoint_forward(device, layout, o), std::invalid_argument);
    o.capacity = 0;
    o.storage = static_cast<RecordStorage>(7);
    EXPECT_THROW(mko::cuda::run_adjoint_forward(device, layout, o), std::invalid_argument);
    o.storage = RecordStorage::Full;
    o.capacity = 32;
    EXPECT_EQ(mko::cuda::run_adjoint_forward(device, layout, o).capacity, 32);
    DeviceAdjointForward empty;
    EXPECT_THROW(empty.download(), std::logic_error);
    EXPECT_THROW(mko::cuda::adjoint_kernel_info(RecordStorage::Full, 20), std::invalid_argument);
}

// Ressourcen der Kernel-Varianten (cudaFuncGetAttributes; Zahlen für docs/architektur.md): Die
// vollständige Aufzeichnung liegt im Local Memory (mindestens 49·C Byte pro Thread).
TEST(CudaAdjoint, KernelResources) {
    MKO_REQUIRE_GPU();
    for (const Precision precision : {Precision::Fp64, Precision::Df64}) {
        const auto none = mko::cuda::adjoint_kernel_info(RecordStorage::None, 0, precision);
        std::printf("[ Info     ] %s none: %d Register, Local %zu B, Shared %zu B\n",
                    precision_name(precision), none.registers, none.local_bytes, none.shared_bytes);
        for (const RecordStorage storage : {RecordStorage::Full, RecordStorage::Checkpoint}) {
            for (const int capacity : {16, 32, 64}) {
                const auto info = mko::cuda::adjoint_kernel_info(storage, capacity, precision);
                std::printf("[ Info     ] %s %s C = %d: %d Register, Local %zu B, Shared %zu B\n",
                            precision_name(precision), mko::cuda::to_string(storage), capacity,
                            info.registers, info.local_bytes, info.shared_bytes);
                EXPECT_GE(info.max_threads_per_block, mko::cuda::kAdjointThreads);
                if (storage == RecordStorage::Full) {
                    EXPECT_GE(info.local_bytes, static_cast<std::size_t>(49 * capacity));
                }
            }
        }
    }
}
