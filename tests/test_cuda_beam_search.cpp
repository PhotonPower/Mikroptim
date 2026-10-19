// Strahlsuche (M6-T4) mit dem GPU-Optimierer (optimize_lm_gpu; nur mit MIKROPTIM_ENABLE_CUDA):
// gemeinsame LM-Aufrufe pro Topologie bitgleich zu Einzelaufrufen, Determinismus und der längere
// Lauf NA 0,3 → 0,6 (docs/optimierung.md, Abschnitt 8).

#include "beam_test_tools.hpp"
#include "homotopy_test_systems.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/levenberg_marquardt.hpp"
#include "mikroptim/cuda/trace_engine.hpp"

#include <gtest/gtest.h>

#include <iostream>
#include <span>

namespace mko::test {
namespace {

HomotopyOptimizer beam_gpu_optimizer(cuda::TraceEngine& engine) {
    return
        [&engine](const VariableSet& set, const SystemBatchHost& base, std::span<const double> x0,
                  const LmOptions& o) { return cuda::optimize_lm_gpu(engine, set, base, x0, o); };
}

// Kurzlauf B = 2 auf der GPU: gebündelt gegen einzeln bitgleich, zwei Läufe bitgleich.
TEST(CudaBeamSearch, GroupingAndRepeatBitwiseGpu) {
    if (cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    cuda::TraceEngine engine(0);
    BeamSmokeSettings st;
    st.width = 2;
    const BeamResult grouped = beam_smoke_run(st, beam_gpu_optimizer(engine));
    const BeamResult again = beam_smoke_run(st, beam_gpu_optimizer(engine));
    st.group = false;
    const BeamResult single = beam_smoke_run(st, beam_gpu_optimizer(engine));
    EXPECT_EQ(grouped.status, HomotopyStatus::MaxElements) << grouped.message;
    EXPECT_EQ(beam_json_without_groups(grouped), beam_json_without_groups(single));
    EXPECT_EQ(beam_json_without_seconds(grouped), beam_json_without_seconds(again));
    EXPECT_TRUE(grouped.system == single.system);
    EXPECT_LT(grouped.groups, single.groups);
    std::cout << "gebündelt: " << grouped.solves << " Optimierungen in " << grouped.groups
              << " Aufrufen, " << grouped.seconds << " s; einzeln: " << single.groups
              << " Aufrufe, " << single.seconds << " s\n";
}

// Längerer Lauf auf der GPU (lokal mit --gtest_also_run_disabled_tests).
TEST(CudaBeamSearch, DISABLED_ApertureToNa06BeamGpu) {
    if (cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    cuda::TraceEngine engine(0);
    const BeamResult r = beam_acceptance_run(2, 2, beam_gpu_optimizer(engine), true);
    write_beam_summary(r, std::cout);
    write_beam_tree(r, std::cout);
    std::cout << "Strahlsuche (GPU): RMS " << r.goal.max_rms << " Wellen, " << r.elements
              << " Elemente, " << r.solves << " Optimierungen in " << r.groups << " Aufrufen, "
              << r.seconds << " s\n";
    EXPECT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    EXPECT_TRUE(r.goal.reached);
    EXPECT_LE(r.goal.max_rms, 1.0 / 14.0);
}

} // namespace
} // namespace mko::test
