// Mechanische Nebenbedingungen (M4-T3) auf der GPU (nur mit MIKROPTIM_ENABLE_CUDA): trace-abhängige
// Residuen aus den Extrema der Engine gegen die CPU, FD-Jacobi mit diesen Residuen gegen die CPU
// und bitgleich über Tranchen, Merit des GPU-Backends gegen die CPU und die Abnahme
// (Penalty-Fortsetzung mit optimize_lm_gpu) wie test_mechanical_optimization.cpp.

#include "mechanical_test_systems.hpp"
#include "residual_test_systems.hpp"
#include "tolerances.hpp"

#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/fd_jacobian.hpp"
#include "mikroptim/cuda/levenberg_marquardt.hpp"
#include "mikroptim/cuda/trace_engine.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace mko::test {
namespace {

#define MKO_REQUIRE_GPU()                                                                          \
    do {                                                                                           \
        if (mko::cuda::device_count() == 0) {                                                      \
            GTEST_SKIP() << "Keine CUDA-GPU gefunden";                                             \
        }                                                                                          \
    } while (0)

bool same_bits(std::span<const double> a, std::span<const double> b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

// Abweichung eines trace-abhängigen Residuums GPU gegen CPU: Die Extrema stimmen auf
// max(kCpuGpuFp64, kCpuGpuFp64Relative · h) überein (Höhen bis etwa 5 mm, cos I absolut), die
// Residuen sind darin mit Steigung ≤ 1 (Höhe, cos) bzw. |d sag/dh| = |c| h/√(1 − c²h²) < 2
// (Randdicke im Testsystem) linear, mal √w.
double traced_residual_tolerance(double weight) {
    return std::sqrt(weight) * 2.0 * std::max(kCpuGpuFp64, kCpuGpuFp64Relative * 5.0);
}

// Startpunkte für mehrere Läufe: x₀ und um ±2 % gestörte Punkte (zulässig).
std::vector<double> run_points(const FdProblem& p, int runs) {
    PerturbationOptions o;
    o.curvature = {0.02, 0.0};
    o.thickness = {0.02, 0.0};
    o.working_distance = {0.02, 0.0};
    o.seed = 99;
    o.first_is_base = true;
    std::vector<double> x;
    for (int j = 0; j < runs; ++j) {
        const std::vector<double> y =
            perturbed_vector(p.set, p.x0, o, static_cast<std::uint64_t>(j));
        x.insert(x.end(), y.begin(), y.end());
    }
    return x;
}

// Extrema der Engine (Modus RESIDUALS) → trace-abhängige Residuen gleich denen aus den
// CPU-Extrema, für gestörte Varianten des Testsystems.
TEST(CudaMechanical, TraceResidualsFromEngineMatchCpu) {
    MKO_REQUIRE_GPU();
    const LoadedSystem loaded = load_residual_system("zweilinser_mechanik.json");
    const SystemBatchHost b = perturbed_batch(loaded, 64, 314);
    const MechanicalConstraints m = make_mechanical_constraints(loaded.system);
    const MechanicalResiduals r = make_mechanical_residuals(m);
    cuda::TraceEngine engine(0);
    cuda::TraceSettings settings;
    settings.extrema = true;
    const cuda::TraceOutput out = engine.run(b, cuda::TraceMode::Residuals, settings);
    const SurfaceExtremaResult cpu = surface_extrema_cpu(b, out.layout);
    std::vector<double> rg(static_cast<std::size_t>(b.variants() * r.traced.count));
    std::vector<double> rc(rg.size());
    evaluate_trace_extra_residuals(r.traced, b, out.residuals.extrema, 0, rg);
    evaluate_trace_extra_residuals(r.traced, b, cpu, 0, rc);
    const double tol = traced_residual_tolerance(m.items[0].weight);
    double worst = 0.0;
    int active = 0;
    for (std::size_t i = 0; i < rg.size(); ++i) {
        worst = std::max(worst, std::abs(rg[i] - rc[i]));
        EXPECT_NEAR(rg[i], rc[i], tol) << i;
        active += rc[i] > 0.0 ? 1 : 0;
    }
    std::printf("[GPU] trace-abhängige Residuen: %zu Werte, %d aktiv, max |GPU − CPU| = %.3e "
                "(Toleranz %.3e)\n",
                rg.size(), active, worst, tol);
    EXPECT_GT(active, 20);
}

// FD-Jacobi mit trace-abhängigen Residuen: GPU gegen CPU (r₀ und J der Trace-Zeilen innerhalb
// der aus der Residuen-Toleranz abgeleiteten Schranke, alle übrigen Zeilen wie in M3-T2), drei
// Läufe; Tranchen aus einzelnen Läufen bitgleich zu einer Tranche.
TEST(CudaMechanical, FdJacobianMatchesCpuAndTranches) {
    MKO_REQUIRE_GPU();
    const FdProblem p = mechanical_problem();
    const LmOptions o = mechanical_options(p.loaded.system, 1);
    const std::vector<double> x0 = run_points(p, 3);
    cuda::TraceEngine engine(0);
    const FdJacobian gpu = cuda::fd_jacobian_gpu(engine, p.set, p.base, x0, o.jacobian);
    const FdJacobian cpu = fd_jacobian_cpu(p.set, p.base, x0, o.jacobian);
    ASSERT_EQ(gpu.shape, cpu.shape);
    const std::int64_t first = cpu.shape.ray_rows + o.jacobian.extra.count;
    ASSERT_EQ(cpu.shape.rows() - first, o.jacobian.trace_extra.count);
    const double delta = traced_residual_tolerance(1e6); // Gewichte der Datei
    double worst = 0.0;
    for (int i = 0; i < 3; ++i) {
        const FdPattern& pat = cpu.patterns[static_cast<std::size_t>(i)];
        for (std::int64_t row = first; row < cpu.shape.rows(); ++row) {
            const double r0 = cpu.residual(i)[static_cast<std::size_t>(row)];
            EXPECT_NEAR(gpu.residual(i)[static_cast<std::size_t>(row)], r0, delta) << row;
            for (int k = 0; k < p.set.size(); ++k) {
                const auto kk = static_cast<std::size_t>(k);
                const double denom = std::abs(pat.step_plus[kk] - pat.step_minus[kk]);
                const double jc = cpu.at(i, row, k);
                const double bound = 4.0 * delta / denom + 1e-12 * std::abs(jc);
                const double diff = std::abs(gpu.at(i, row, k) - jc);
                worst = std::max(worst, diff / bound);
                EXPECT_LE(diff, bound) << "Lauf " << i << " Zeile " << row << " Spalte " << k;
            }
        }
    }
    std::printf("[GPU] FD-Jacobi Trace-Zeilen: max |ΔJ|/Schranke %.3e\n", worst);

    // Tranchen aus einzelnen Läufen (Speichergrenze) bitgleich zu einer Tranche
    const int n = fd_variant_count(FdScheme::Central, p.set.size());
    const ResidualLayout layout(p.base.topology(), o.jacobian.quadrature);
    OutputRequest req;
    req.rays_per_variant = layout.rays_per_variant();
    req.penalty_residuals = layout.penalty_rows();
    req.residuals = true;
    req.ray_status = true;
    req.opd = false;
    req.merit = false;
    req.surface_max = true;
    cuda::EngineOptions limited;
    limited.memory_limit_bytes =
        estimate_device_bytes(p.base.topology(), n + n / 2, req) +
        align_bytes(static_cast<std::size_t>(p.base.fields()) * sizeof(ResidualFieldPlan));
    limited.reserve = MemoryReserve{0, 0.0};
    cuda::TraceEngine small(0, limited);
    cuda::DeviceFdJacobian a;
    a.compute(engine, p.set, p.base, x0, o.jacobian);
    cuda::DeviceFdJacobian bdev;
    bdev.compute(small, p.set, p.base, x0, o.jacobian);
    EXPECT_EQ(a.run_info().tranches, 1);
    EXPECT_EQ(bdev.run_info().tranches, 3);
    const FdJacobian ja = a.download();
    const FdJacobian jb = bdev.download();
    EXPECT_TRUE(same_bits(ja.jacobian, jb.jacobian));
    EXPECT_TRUE(same_bits(ja.residuals, jb.residuals));
    EXPECT_TRUE(same_bits(ja.jacobian, gpu.jacobian));
}

// Merit des GPU-Backends (MERIT mit Extrema, trace-abhängige Residuen addiert) gegen die CPU an
// mehreren Punkten, und der Anteil der trace-abhängigen Residuen gegen die CPU.
TEST(CudaMechanical, GpuMeritMatchesCpu) {
    MKO_REQUIRE_GPU();
    const FdProblem p = mechanical_problem();
    const LmOptions with = mechanical_options(p.loaded.system, 1);
    LmOptions without = with;
    without.jacobian.trace_extra = {};
    const int runs = 6;
    const std::vector<double> x = run_points(p, runs);
    std::vector<int> ids(runs);
    for (int j = 0; j < runs; ++j) {
        ids[static_cast<std::size_t>(j)] = j;
    }
    SystemBatchHost base = p.base; // eine Vorlage für alle Läufe
    cuda::TraceEngine engine(0);
    const auto merit = [&](LmBackend& backend) {
        std::vector<double> f(runs);
        std::vector<std::int64_t> failed(runs);
        std::vector<std::int64_t> hard(runs);
        backend.merit(ids, x, f, failed, hard);
        return f;
    };
    cuda::GpuLmBackend gw(engine, p.set, base, with);
    cuda::GpuLmBackend go(engine, p.set, base, without);
    CpuLmBackend cw(p.set, base, with);
    CpuLmBackend co(p.set, base, without);
    const std::vector<double> fgw = merit(gw);
    const std::vector<double> fgo = merit(go);
    const std::vector<double> fcw = merit(cw);
    const std::vector<double> fco = merit(co);
    for (int j = 0; j < runs; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        std::printf("[GPU] Punkt %d: F GPU %.12e, CPU %.12e, Trace-Anteil GPU %.6e, CPU %.6e\n", j,
                    fgw[jj], fcw[jj], fgw[jj] - fgo[jj], fcw[jj] - fco[jj]);
        EXPECT_NEAR(fgw[jj], fcw[jj], kMechanicalMeritCpuGpu * fcw[jj]);
        EXPECT_GT(fcw[jj] - fco[jj], 0.0);
        EXPECT_NEAR(fgw[jj] - fgo[jj], fcw[jj] - fco[jj], kMechanicalMeritCpuGpu * fcw[jj]);
    }
}

// Abnahme auf der GPU: dieselbe Fortsetzung mit optimize_lm_gpu hält alle Grenzen ein.
TEST(CudaMechanical, GpuKeepsAllLimits) {
    MKO_REQUIRE_GPU();
    const FdProblem p = mechanical_problem();
    cuda::TraceEngine engine(0);
    const std::vector<MechanicalStage> stages =
        mechanical_continuation(p, [&](const LmOptions& o, std::span<const double> x0) {
            return cuda::optimize_lm_gpu(engine, p.set, p.base, x0, o);
        });
    ASSERT_EQ(stages.size(), static_cast<std::size_t>(kMechanicalStages));
    for (const MechanicalStage& s : stages) {
        std::printf("[GPU]   w = %8.1e: F %.6e -> %.10e, %3d Zyklen, %-16s", s.weight,
                    s.run.history.front(), s.run.merit, s.run.iterations, to_string(s.run.status));
        for (const ConstraintType t : mechanical_types()) {
            std::printf("  %s %+.2e", std::string(to_string(t)).c_str(), s.state.worst(t));
        }
        std::printf("\n");
        EXPECT_TRUE(lm_converged(s.run.status)) << s.weight << ": " << to_string(s.run.status);
        EXPECT_EQ(s.run.failed_rays, 0);
    }
    const MechanicalStage& last = stages.back();
    for (std::size_t k = 0; k < last.state.violation.size(); ++k) {
        const ConstraintType t = last.state.constraints.items[k].type;
        EXPECT_LE(last.state.violation[k],
                  mechanical_tolerance(t, kMechanicalPenaltyLength, kMechanicalPenaltyCos))
            << to_string(t) << " Fläche " << last.state.constraints.items[k].file_surface;
    }
    for (const ConstraintType t :
         {ConstraintType::MaxCenterThickness, ConstraintType::MinEdgeThickness,
          ConstraintType::MaxIncidenceAngle}) {
        EXPECT_GT(last.state.worst(t), -kMechanicalPenaltyLength) << to_string(t);
    }
    EXPECT_LT(std::abs(focal_length_at(p, last.run.x) - 10.0), kMechanicalFocalLength);
}

} // namespace
} // namespace mko::test
