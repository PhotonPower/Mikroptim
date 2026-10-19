// Einfüge-Operatoren (M6-T2) auf der GPU (nur mit MIKROPTIM_ENABLE_CUDA): Merit-Invarianz mit
// dem Kernel MERIT für alle Fälle der CPU-Abnahme, Abnahme B (Achromat) mit optimize_lm_gpu und
// der Ablauf des Operators A (docs/optimierung.md, Abschnitt 6).

#include "insertion_optimization.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/levenberg_marquardt.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

namespace mko::test {
namespace {

// F wie system_merit, Strahlresiduen im Kernel MERIT, Zusatzresiduen auf dem Host.
double gpu_merit(cuda::TraceEngine& engine, const SystemDescription& d) {
    const LoadedSystem l = prepare_test_system(d);
    const SystemBatchHost b = make_system_batch(l, 1);
    const ExtraResiduals extra = insertion_extra(l.system);
    std::vector<double> e(static_cast<std::size_t>(extra.count));
    evaluate_extra_residuals(extra, b, e);
    double em = 0.0;
    for (const double v : e) {
        em += v * v;
    }
    const TraceExtraResiduals traced = insertion_trace_extra(l.system);
    cuda::TraceSettings s;
    s.extra_merit = std::span<const double>(&em, 1);
    s.extrema = traced.count > 0; // Max-Reduktionen für die trace-abhängigen Residuen
    const cuda::TraceOutput out = engine.run(b, cuda::TraceMode::Merit, s);
    double f = out.merit.merit.at(0);
    if (traced.count > 0) {
        f += trace_extra_merit_values(traced, b, out.merit.extrema).at(0);
    }
    return f;
}

TEST(CudaInsertion, MeritUnchangedOnGpu) {
    if (cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    cuda::TraceEngine engine(0);
    double worst = 0.0;
    double worst_cpu = 0.0;
    int within_plan = 0;
    const std::vector<InsertionCase> cases = insertion_cases();
    for (const InsertionCase& c : cases) {
        const double before = gpu_merit(engine, c.before);
        const double after = gpu_merit(engine, c.after);
        const double rel = relative_change(before, after);
        const double cpu = relative_change(system_merit(c.after), after);
        worst = std::max(worst, rel);
        worst_cpu = std::max(worst_cpu, cpu);
        within_plan += rel <= kInsertionMeritPlan ? 1 : 0;
        std::printf("[GPU] %-68s F = %.15e  |dF|/F = %.2e  GPU/CPU %.2e\n", c.name.c_str(), before,
                    rel, cpu);
        EXPECT_LE(rel, kInsertionMeritRelative) << c.name;
        EXPECT_LE(cpu, kCpuGpuFp64Relative) << c.name;
    }
    std::printf("[GPU] größte relative Änderung %.2e (Toleranz %.0e); %d von %zu Fällen <= %.0e; "
                "GPU gegen CPU höchstens %.2e\n",
                worst, kInsertionMeritRelative, within_plan, cases.size(), kInsertionMeritPlan,
                worst_cpu);
}

TEST(CudaInsertion, OperatorBMakesAchromatGpu) {
    if (cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    cuda::TraceEngine engine(0);
    const auto gpu_optimize = [&](const VariableSet& set, const SystemBatchHost& base,
                                  std::span<const double> x0, const LmOptions& o) {
        return cuda::optimize_lm_gpu(engine, set, base, x0, o);
    };
    const AchromatRun a = achromat_run(gpu_optimize);
    print_achromat("GPU", a);
    check_achromat(a);
}

TEST(CudaInsertion, OperatorAFromDiagnosisGpu) {
    if (cuda::device_count() == 0) {
        GTEST_SKIP() << "Keine CUDA-GPU gefunden";
    }
    cuda::TraceEngine engine(0);
    const auto gpu_optimize = [&](const VariableSet& set, const SystemBatchHost& base,
                                  std::span<const double> x0, const LmOptions& o) {
        return cuda::optimize_lm_gpu(engine, set, base, x0, o);
    };
    const MeniscusRun m = meniscus_run(gpu_optimize);
    print_meniscus("GPU", m);
    check_meniscus(m);
}

} // namespace
} // namespace mko::test
