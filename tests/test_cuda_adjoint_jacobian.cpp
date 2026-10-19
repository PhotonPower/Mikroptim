// Adjungierte Jacobi-Matrix auf der GPU (M5-T2, nur mit MIKROPTIM_ENABLE_CUDA): GPU gegen
// CPU-Referenz, Abnahme gegen die Richardson-Referenz und die FD-Jacobi der GPU für alle
// Variablentypen, df64 gegen FP64, Determinismus und Unabhängigkeit von der Lage im Batch,
// Ressourcen der Kernel, erste Laufzeitzahl.

#include "adjoint_test_systems.hpp"
#include "fd_test_systems.hpp"
#include "mikroptim/cuda/adjoint.hpp"
#include "mikroptim/cuda/adjoint_jacobian.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/fd_jacobian.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/optim/adjoint_jacobian.hpp"
#include "mikroptim/optim/fd_jacobian.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

using mko::AdjointJacobian;
using mko::ResidualLayout;
using mko::cuda::Precision;
using mko::cuda::TraceEngine;

namespace {

#define MKO_REQUIRE_GPU()                                                                          \
    do {                                                                                           \
        if (mko::cuda::device_count() == 0) {                                                      \
            GTEST_SKIP() << "Keine CUDA-GPU gefunden";                                             \
        }                                                                                          \
    } while (0)

template <class T>
bool same_bytes(std::span<const T> a, std::span<const T> b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

// Relative Abweichung pro Spalte (größte) zweier Jacobi-Matrizen eines Laufs.
double max_column_deviation(std::span<const double> a, std::span<const double> b, int P) {
    return mko::test::jacobian_deviation(a, b, P).max_column;
}

} // namespace

// GPU (FP64) gegen die CPU-Referenz auf allen Testsystemen: Status gleich, r₀ der Strahlzeilen
// innerhalb der Residuen-Toleranzen, Trace-Zeilen von J relativ pro Spalte ≤ kAdjointCpuGpu
// (dieselbe Pro-Strahl-Rechnung; Unterschiede aus sin/cos der CUDA-Bibliothek und der Reihenfolge
// der Kolbensummen), Zusatzzeilen ohne trace-abhängige Residuen bitgleich.
TEST(CudaAdjointJacobian, MatchesCpu) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    const auto cases = mko::test::adjoint_cases();
    for (const auto& c : cases) {
        const mko::test::FdProblem& p = c->p;
        const int P = p.set.size();
        const AdjointJacobian cpu = mko::adjoint_jacobian_cpu(p.set, p.base, p.x0, c->options);
        const AdjointJacobian gpu =
            mko::cuda::adjoint_jacobian_gpu(engine, p.set, p.base, p.x0, c->options);
        ASSERT_EQ(gpu.shape, cpu.shape) << c->name;
        EXPECT_TRUE(same_bytes<std::uint8_t>(gpu.status, cpu.status)) << c->name;
        const std::int64_t rv = cpu.shape.ray_rows;
        const std::int64_t R = cpu.shape.rows();
        double worst_r = 0.0;
        for (std::int64_t r = 0; r < rv; ++r) {
            const double a = gpu.residuals[static_cast<std::size_t>(r)];
            const double b = cpu.residuals[static_cast<std::size_t>(r)];
            worst_r = std::max(worst_r, std::abs(a - b) / std::max(std::abs(b), 1.0));
        }
        const double dev =
            max_column_deviation(mko::test::row_block(mko::test::run_columns(gpu, 0), P, 0, rv),
                                 mko::test::row_block(mko::test::run_columns(cpu, 0), P, 0, rv), P);
        const auto ga = mko::test::row_block(mko::test::run_columns(gpu, 0), P, rv, R);
        const auto ca = mko::test::row_block(mko::test::run_columns(cpu, 0), P, rv, R);
        std::printf("[GPU gegen CPU] %s: J größte Spalte relativ %.3e, r₀ max %.3e\n",
                    c->name.c_str(), dev, worst_r);
        EXPECT_LT(dev, mko::test::kAdjointCpuGpu) << c->name;
        EXPECT_LT(worst_r, mko::test::kAdjointCpuGpu) << c->name;
        if (c->options.trace_extra.count == 0) {
            EXPECT_TRUE(same_bytes<double>(ga, ca)) << c->name << ": Zusatzzeilen";
            for (std::int64_t r = rv; r < R; ++r) {
                EXPECT_EQ(gpu.residuals[static_cast<std::size_t>(r)],
                          cpu.residuals[static_cast<std::size_t>(r)]);
            }
        } else {
            // Trace-abhängige Zeilen: finite Differenzen der Extrema, die auf CPU und GPU um
            // Rundungen (kCpuGpuFp64) abweichen; geteilt durch h ≈ 6·10⁻⁶·x wie in der FD-Jacobi
            const double extra_dev = max_column_deviation(ga, ca, P);
            std::printf("    trace-abhängige Zusatzzeilen: größte Spalte relativ %.3e\n",
                        extra_dev);
            EXPECT_LT(extra_dev, mko::test::kAdjointJacobian) << c->name;
        }
    }
}

// Abnahme M5-T2 auf der GPU: Trace-Zeilen gegen die Richardson-Referenz (adaptiv, CPU) pro Spalte
// ≤ 1·10⁻⁶ für alle Variablentypen; die ganze Matrix gegen die FD-Jacobi der GPU spaltenskaliert
// ≤ 1·10⁻⁶ plus deren eigener Fehler gegen die Referenz.
TEST(CudaAdjointJacobian, AcceptanceAgainstRichardsonAndFd) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    const auto cases = mko::test::adjoint_cases();
    for (const auto& c : cases) {
        const mko::test::FdProblem& p = c->p;
        const int P = p.set.size();
        const ResidualLayout layout(p.base.topology(), c->options.quadrature);
        const std::int64_t rv = layout.trace_rows();
        const AdjointJacobian gpu =
            mko::cuda::adjoint_jacobian_gpu(engine, p.set, p.base, p.x0, c->options);
        const mko::FdJacobian fd =
            mko::cuda::fd_jacobian_gpu(engine, p.set, p.base, p.x0, mko::fd_options_of(c->options));
        const auto jg = mko::test::run_columns(gpu, 0);
        const auto jf = mko::test::run_jacobian(fd, 0);
        double fd_error = 0.0;
        if (c->richardson) {
            const auto ref = mko::test::adaptive_reference(p, layout, c->options.extra, rv);
            ASSERT_FALSE(ref.r2.empty()) << c->name;
            const auto d = mko::test::adjoint_deviation(mko::test::row_block(jg, P, 0, rv),
                                                        mko::test::row_block(ref.r2, P, 0, rv), p);
            mko::test::print_deviation("GPU gegen Richardson, Trace-Zeilen", c->name, p, d);
            for (int k = 0; k < P; ++k) {
                EXPECT_LT(d.plain.column[static_cast<std::size_t>(k)], mko::test::kAdjointJacobian)
                    << c->name << ", " << p.set[k].name;
            }
            EXPECT_LT(d.scaled.frobenius, mko::test::kAdjointJacobian) << c->name;
            fd_error = mko::test::adjoint_deviation(mko::test::row_block(jf, P, 0, rv),
                                                    mko::test::row_block(ref.r2, P, 0, rv), p)
                           .scaled.frobenius;
        }
        const auto dfd = mko::test::adjoint_deviation(jg, jf, p);
        std::printf("[GPU gegen FD-GPU] %s: skaliert %.3e (FD gegen Richardson %.3e)\n",
                    c->name.c_str(), dfd.scaled.frobenius, fd_error);
        EXPECT_LT(dfd.scaled.frobenius, mko::test::kAdjointJacobian + fd_error) << c->name;
    }
}

// df64 gegen FP64 (beide GPU): Status gleich, Trace-Zeilen von J relativ pro Spalte
// ≤ kAdjointDf64 (die trace-abhängigen Zusatzzeilen sind finite Differenzen der df64-Extrema).
TEST(CudaAdjointJacobian, Df64MatchesFp64) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    const auto cases = mko::test::adjoint_cases();
    for (const auto& c : cases) {
        const mko::test::FdProblem& p = c->p;
        const AdjointJacobian a =
            mko::cuda::adjoint_jacobian_gpu(engine, p.set, p.base, p.x0, c->options);
        const AdjointJacobian b = mko::cuda::adjoint_jacobian_gpu(engine, p.set, p.base, p.x0,
                                                                  c->options, Precision::Df64);
        EXPECT_TRUE(same_bytes<std::uint8_t>(a.status, b.status)) << c->name;
        const int P = p.set.size();
        const std::int64_t rv = a.shape.ray_rows;
        const double dev =
            max_column_deviation(mko::test::row_block(mko::test::run_columns(b, 0), P, 0, rv),
                                 mko::test::row_block(mko::test::run_columns(a, 0), P, 0, rv), P);
        std::printf("[df64 gegen FP64] %s: J größte Spalte relativ %.3e\n", c->name.c_str(), dev);
        EXPECT_LT(dev, mko::test::kAdjointDf64) << c->name;
    }
}

// Determinismus (ADR-0007): wiederholte Rechnungen bitgleich; ein Lauf ist unabhängig von seiner
// Lage im Batch und der Zahl der Läufe bitgleich (Block pro Tupel, feste Reduktionsreihenfolge).
TEST(CudaAdjointJacobian, RepeatedRunsAndBatchPositionBitIdentical) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    const mko::test::FdProblem p = mko::test::ghost_gap_problem();
    const int P = p.set.size();
    std::vector<double> x0;
    for (int i = 0; i < 4; ++i) {
        for (int k = 0; k < P; ++k) {
            x0.push_back(p.x0[static_cast<std::size_t>(k)] *
                         (1.0 + 0.01 * (i + 1) * ((k % 3) - 1)));
        }
    }
    for (const Precision prec : {Precision::Fp64, Precision::Df64}) {
        mko::cuda::DeviceAdjointJacobian d;
        d.compute(engine, p.set, p.base, x0, {}, prec);
        const AdjointJacobian first = d.download();
        for (int rep = 0; rep < 5; ++rep) {
            d.compute(engine, p.set, p.base, x0, {}, prec);
            const AdjointJacobian again = d.download();
            EXPECT_TRUE(same_bytes<double>(first.jacobian, again.jacobian));
            EXPECT_TRUE(same_bytes<double>(first.residuals, again.residuals));
            EXPECT_TRUE(same_bytes<std::uint8_t>(first.status, again.status));
        }
        for (int i = 0; i < 4; ++i) {
            const auto xi = std::span<const double>(x0).subspan(static_cast<std::size_t>(i * P),
                                                                static_cast<std::size_t>(P));
            const AdjointJacobian one =
                mko::cuda::adjoint_jacobian_gpu(engine, p.set, p.base, xi, {}, prec);
            EXPECT_TRUE(same_bytes<double>(mko::test::run_columns(first, i),
                                           mko::test::run_columns(one, 0)))
                << "Lauf " << i;
        }
    }
}

// Ressourcen der Kernel (Register, Local Memory für AdjointRecord und Rückwärtsdurchlauf,
// statischer Shared Memory) für alle Kapazitätsklassen und Zahlentypen; Ausgabe für die Doku.
TEST(CudaAdjointJacobian, KernelResources) {
    MKO_REQUIRE_GPU();
    for (const Precision prec : {Precision::Fp64, Precision::Df64}) {
        for (const int cap :
             {mko::kRecordCapacitySmall, mko::kRecordCapacityMedium, mko::kRecordCapacityLarge}) {
            const mko::cuda::AdjointKernelInfo info =
                mko::cuda::adjoint_jacobian_kernel_info(cap, prec);
            std::printf("adjoint_jacobian_kernel<%s, %d>: %d Register, Stack %zu B, Shared %zu B\n",
                        mko::cuda::to_string(prec), cap, info.registers, info.local_bytes,
                        info.shared_bytes);
            EXPECT_GE(info.max_threads_per_block, mko::cuda::kAdjointJacobianThreads);
            EXPECT_LE(info.shared_bytes + 17 * 1024, 48u * 1024u);
        }
    }
}

// Erste Laufzeitzahl (kein Benchmark, keine Abnahme; die Abnahme der Geschwindigkeit ist M5-T3):
// Mehrlinser, 13 Variablen, 200 Läufe – adjungiert gegen FD-Jacobi auf derselben GPU.
TEST(CudaAdjointJacobian, FirstTimingAgainstFd) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    const mko::test::FdProblem p = mko::test::mehrlinser_problem();
    std::vector<double> x0;
    for (int i = 0; i < 200; ++i) {
        x0.insert(x0.end(), p.x0.begin(), p.x0.end());
    }
    mko::cuda::DeviceAdjointJacobian adj;
    mko::cuda::DeviceFdJacobian fd;
    adj.compute(engine, p.set, p.base, x0); // Aufwärmen
    fd.compute(engine, p.set, p.base, x0);
    double t_adj = 1e30;
    double t_fd = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        adj.compute(engine, p.set, p.base, x0);
        auto t1 = std::chrono::steady_clock::now();
        fd.compute(engine, p.set, p.base, x0);
        auto t2 = std::chrono::steady_clock::now();
        t_adj = std::min(t_adj, std::chrono::duration<double, std::milli>(t1 - t0).count());
        t_fd = std::min(t_fd, std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
    std::printf("Mehrlinser, P = 13, 200 Läufe: adjungiert %.1f ms, FD %.1f ms (Faktor %.1f), "
                "J %.1f MB\n",
                t_adj, t_fd, t_fd / t_adj,
                static_cast<double>(adj.shape().jacobian_size()) * 8.0 / 1e6);
    EXPECT_TRUE(adj.ready());
}
