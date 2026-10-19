// Glas-Einrasten (M6-T5) auf der GPU (nur mit MIKROPTIM_ENABLE_CUDA): dieselbe Abnahme wie
// test_glass_snap.cpp mit cuda::optimize_lm_gpu; die GPU wählt dieselben Gläser wie die CPU,
// die Merit stimmt bis auf die Rundung der Reoptimierung, zwei GPU-Läufe sind bitgleich.

#include "glass_snap_test_systems.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/levenberg_marquardt.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace mko::test {
namespace {

TEST(CudaGlassSnap, GpuMatchesCpuSelection) {
    if (cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    cuda::TraceEngine engine(0);
    const GlassSnapOptimizer gpu = [&engine](const VariableSet& set, const SystemBatchHost& base,
                                             std::span<const double> x0, const LmOptions& o) {
        return cuda::optimize_lm_gpu(engine, set, base, x0, o);
    };
    const GlassSnapOptimizer cpu = [](const VariableSet& set, const SystemBatchHost& base,
                                      std::span<const double> x0, const LmOptions& o) {
        return optimize_lm_cpu(set, base, x0, o);
    };
    // Kontinuierliches Optimum auf der CPU, damit beide Backends vom selben Punkt einrasten
    const ContinuousOptimum c = continuous_optimum(cpu);
    ASSERT_TRUE(lm_converged(c.run.status));
    const GlassSnapOptions o = snap_options(c.loaded.system, 3, 64);
    const GlassSnapResult g = snap_glasses(c.loaded, c.hull, o, gpu);
    const GlassSnapResult r = snap_glasses(c.loaded, c.hull, o, cpu);
    ASSERT_EQ(g.runs.size(), r.runs.size());
    for (std::size_t i = 0; i < g.runs.size(); ++i) {
        std::printf("[GPU] %2zu %-8s %-8s eingerastet %.10e (CPU %.10e)  reoptimiert %.10e (CPU "
                    "%.10e)  %4d Zyklen (CPU %4d) %s\n",
                    i, g.runs[i].glasses[0].c_str(), g.runs[i].glasses[1].c_str(),
                    g.runs[i].snapped_merit, r.runs[i].snapped_merit, g.runs[i].run.merit,
                    r.runs[i].run.merit, g.runs[i].run.iterations, r.runs[i].run.iterations,
                    static_cast<int>(i) == g.best ? "bester" : "");
        // Kandidaten und Kombinationen entstehen auf dem Host: identisch
        EXPECT_EQ(g.runs[i].glasses, r.runs[i].glasses);
        EXPECT_EQ(g.runs[i].combination, r.runs[i].combination);
        EXPECT_LE(std::abs(g.runs[i].snapped_merit / r.runs[i].snapped_merit - 1.0),
                  kLmReturnRelative);
        EXPECT_EQ(g.runs[i].run.failed_rays, 0);
    }
    // dieselbe Auswahl, Merit bis auf die Rundung
    EXPECT_EQ(g.best, r.best);
    EXPECT_EQ(g.best_run().glasses, r.best_run().glasses);
    EXPECT_LE(std::abs(g.best_run().run.merit / r.best_run().run.merit - 1.0), kLmReturnRelative);
    EXPECT_LE(std::abs(g.continuous_merit / r.continuous_merit - 1.0), kLmReturnRelative);
    EXPECT_GE(g.degradation(), -kGlassSnapMeritSlack * g.continuous_merit);
    std::printf("[GPU] bester Lauf %d (%s, %s): F %.10e (CPU %.10e), Verschlechterung %.4e\n",
                g.best, g.best_run().glasses[0].c_str(), g.best_run().glasses[1].c_str(),
                g.best_run().run.merit, r.best_run().run.merit, g.degradation());
    // Ergebnis: nur Katalog-Gläser
    for (const Surface& s : g.result.system.surfaces) {
        EXPECT_NE(s.material.kind, MaterialKind::Buchdahl);
    }
    // zwei GPU-Läufe bitgleich
    const GlassSnapResult again = snap_glasses(c.loaded, c.hull, o, gpu);
    EXPECT_EQ(again.best, g.best);
    for (std::size_t i = 0; i < g.runs.size(); ++i) {
        EXPECT_EQ(again.runs[i].run.x, g.runs[i].run.x);
        EXPECT_EQ(again.runs[i].run.merit, g.runs[i].run.merit);
    }
}

} // namespace
} // namespace mko::test
