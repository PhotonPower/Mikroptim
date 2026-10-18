// Finite-Differenzen-Jacobi auf der GPU (M3-T2, nur mit MIKROPTIM_ENABLE_CUDA): Abgleich mit
// dem CPU-Pfad (Toleranz aus den Residuen-Toleranzen abgeleitet), Richardson, mehrere Läufe
// und Tranchen bitgleich, Gerätepfad gegen die Differenzbildung auf dem Host, Status-Wechsel,
// TraceEngine::run_on_device.

#include "fd_test_systems.hpp"
#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cuda/cuda_error.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/fd_jacobian.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/optim/fd_formula.hpp"
#include "mikroptim/optim/fd_jacobian.hpp"
#include "residual_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using mko::FdJacobian;
using mko::FdJacobianOptions;
using mko::FdScheme;
using mko::FdStencil;
using mko::ResidualLayout;
using mko::SystemBatchHost;
using mko::cuda::CudaError;
using mko::cuda::DeviceFdJacobian;
using mko::cuda::DeviceTranche;
using mko::cuda::EngineOptions;
using mko::cuda::TraceEngine;
using mko::cuda::TraceSettings;
using mko::test::FdProblem;
using mko::test::kCpuGpuFp64;
using mko::test::kCpuGpuFp64Relative;

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

bool same_bits(std::span<const double> a, std::span<const double> b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

bool identical(const FdJacobian& a, const FdJacobian& b) {
    return a.shape == b.shape && a.scheme == b.scheme && same_bytes(a.jacobian, b.jacobian) &&
           same_bytes(a.residuals, b.residuals) && same_bytes(a.status_changes, b.status_changes) &&
           a.patterns == b.patterns;
}

// Startvektoren für `runs` Läufe: x₀ und gestörte Vektoren (1 % relativ, fester Startwert).
std::vector<double> run_vectors(const FdProblem& p, int runs) {
    mko::PerturbationOptions po;
    po.seed = 23;
    std::vector<double> x = p.x0;
    for (int i = 1; i < runs; ++i) {
        const std::vector<double> xi =
            mko::perturbed_vector(p.set, p.x0, po, static_cast<std::uint64_t>(i));
        x.insert(x.end(), xi.begin(), xi.end());
    }
    return x;
}

// Abweichung eines Residuums zwischen GPU und CPU (Toleranz von RESIDUALS, M2-T2).
double residual_tolerance(double r) {
    return std::max(kCpuGpuFp64, kCpuGpuFp64Relative * std::abs(r));
}

// Schranke für |J_GPU − J_CPU| eines Elements, abgeleitet aus den Residuen-Toleranzen: Die
// Formel ist linear in den Residuen, J = Σ_x c_x·r_x (zentral/gekürzt c_a = −c_b = 1/(d_a − d_b),
// einseitig c_a = d_b²/N, c_b = −d_a²/N, c₀ = −(c_a + c_b) mit N = d_a·d_b·(d_b − d_a),
// vorwärts c_a = −c₀ = 1/d_a), also |ΔJ| ≤ Σ_x |c_x|·δ(r_x) mit δ = residual_tolerance (0 für
// Zusatzresiduen, die auf dem Host gleich ausgewertet werden). Dazu die Rundung der Formel
// selbst (FMA-Kontraktion im Kernel): 8ε·Σ_x |c_x·r_x|.
double element_bound(FdStencil s, double da, double db, double r0, double ra, double rb,
                     bool ray_row) {
    double c0 = 0.0;
    double ca = 0.0;
    double cb = 0.0;
    switch (s) {
    case FdStencil::OneSided: {
        const double n = da * db * (db - da);
        ca = db * db / n;
        cb = -da * da / n;
        c0 = -(ca + cb);
        break;
    }
    case FdStencil::Forward:
        ca = 1.0 / da;
        c0 = -ca;
        break;
    default:
        ca = 1.0 / (da - db);
        cb = -ca;
        break;
    }
    const double formula =
        8.0 * DBL_EPSILON * (std::abs(c0 * r0) + std::abs(ca * ra) + std::abs(cb * rb));
    if (!ray_row) {
        return formula;
    }
    return std::abs(c0) * residual_tolerance(r0) + std::abs(ca) * residual_tolerance(ra) +
           std::abs(cb) * residual_tolerance(rb) + formula;
}

// Vergleicht GPU- und CPU-Jacobi elementweise mit element_bound; Rückgabe: größtes Verhältnis
// |ΔJ|/Schranke.
double compare_gpu_cpu(const FdProblem& p, std::span<const double> x0, const FdJacobianOptions& opt,
                       const FdJacobian& gpu, const FdJacobian& cpu) {
    EXPECT_EQ(gpu.shape, cpu.shape);
    EXPECT_TRUE(same_bytes(gpu.status_changes, cpu.status_changes));
    const mko::FdJacobianBatch fd = mko::make_fd_jacobian_batch(p.set, p.base, x0, opt.scheme);
    const mko::ResidualResult res =
        mko::residuals_cpu(fd.batch, ResidualLayout(p.base.topology(), opt.quadrature));
    const std::vector<double> extra = mko::test::extra_values(fd.batch, opt.extra);
    const std::int64_t rv = cpu.shape.ray_rows;
    const std::int64_t R = cpu.shape.rows();
    const auto value = [&](int v, std::int64_t r) {
        return r < rv
                   ? res.residual(v, r)
                   : extra[static_cast<std::size_t>(v) * static_cast<std::size_t>(opt.extra.count) +
                           static_cast<std::size_t>(r - rv)];
    };
    const int n = fd.variants_per_run;
    const bool central = opt.scheme == FdScheme::Central;
    double worst = 0.0;
    for (int i = 0; i < cpu.shape.runs; ++i) {
        const mko::FdPattern& pat = cpu.patterns[static_cast<std::size_t>(i)];
        for (std::int64_t r = 0; r < R; ++r) {
            const double tol = residual_tolerance(value(i * n, r));
            EXPECT_LE(std::abs(gpu.residual(i)[static_cast<std::size_t>(r)] -
                               cpu.residual(i)[static_cast<std::size_t>(r)]),
                      r < rv ? tol : 0.0);
        }
        for (int k = 0; k < cpu.shape.variables; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            const int va = i * n + mko::fd_plus_variant(opt.scheme, k);
            const int vb = central ? i * n + mko::fd_minus_variant(k) : va;
            const double da = pat.step_plus[kk];
            const double db = central ? pat.step_minus[kk] : 0.0;
            for (std::int64_t r = 0; r < R; ++r) {
                const double bound = element_bound(pat.stencil[kk], da, db, value(i * n, r),
                                                   value(va, r), value(vb, r), r < rv);
                const double diff = std::abs(gpu.at(i, r, k) - cpu.at(i, r, k));
                EXPECT_LE(diff, bound) << "Lauf " << i << ", " << p.set[k].name << ", Zeile " << r;
                if (bound > 0.0) {
                    worst = std::max(worst, diff / bound);
                }
            }
        }
    }
    return worst;
}

// Engine, deren Speicher genau für `bytes` reicht (ohne Reserve).
EngineOptions limited_to(std::size_t bytes) {
    EngineOptions o;
    o.memory_limit_bytes = bytes;
    o.reserve = mko::MemoryReserve{0, 0.0};
    return o;
}

// Gerätespeicher eines Laufs der Engine (Modus RESIDUALS ohne OPD) mit `variants` Varianten:
// Batch + Ausgaben + Feldablage (docs/architektur.md, „Host-API TraceEngine“).
std::size_t run_bytes(const mko::BatchTopology& t, int variants, const ResidualLayout& layout,
                      std::int64_t extra_residuals = 0) {
    mko::OutputRequest r;
    r.rays_per_variant = layout.rays_per_variant();
    r.extra_residuals = extra_residuals;
    r.residuals = true;
    r.ray_status = true;
    r.opd = false;
    r.merit = false;
    r.surface_max = false;
    return mko::estimate_device_bytes(t, variants, r) +
           mko::align_bytes(static_cast<std::size_t>(t.fields) * sizeof(mko::ResidualFieldPlan));
}

} // namespace

// GPU gegen CPU: Mehrlinser mit Zusatzresiduen, drei Läufe in einem Batch. Jedes Element
// innerhalb der aus den Residuen-Toleranzen abgeleiteten Schranke (element_bound), r₀ innerhalb
// der Residuen-Toleranz, Status-Wechsel gleich; Lauf 0 außerdem gegen Richardson wie auf der
// CPU (skalierte Frobeniusnorm < 1·10⁻⁶).
TEST(CudaFdJacobian, MatchesCpu) {
    MKO_REQUIRE_GPU();
    const FdProblem p = mko::test::mehrlinser_problem();
    const std::vector<double> x0 = run_vectors(p, 3);
    FdJacobianOptions opt;
    opt.extra = mko::test::mehrlinser_extra(p.set);
    TraceEngine engine(0);
    const FdJacobian gpu = mko::cuda::fd_jacobian_gpu(engine, p.set, p.base, x0, opt);
    const FdJacobian cpu = mko::fd_jacobian_cpu(p.set, p.base, x0, opt);
    ASSERT_EQ(gpu.shape.runs, 3);
    ASSERT_EQ(gpu.shape.rows(), 680 + 2);
    const double worst = compare_gpu_cpu(p, x0, opt, gpu, cpu);
    double rel = 0.0; // größte relative Abweichung GPU/CPU pro Spalte (2-Norm)
    for (int i = 0; i < 3; ++i) {
        const auto d = mko::test::jacobian_deviation(mko::test::run_jacobian(gpu, i),
                                                     mko::test::run_jacobian(cpu, i), p.set.size());
        rel = std::max(rel, d.max_column);
    }

    // Lauf 0 ohne Zusatzresiduen gegen Richardson
    const ResidualLayout layout(p.base.topology());
    const FdJacobian g0 = mko::cuda::fd_jacobian_gpu(engine, p.set, p.base, p.x0);
    const auto ref = mko::test::richardson_reference(p, p.x0, layout);
    const auto scaled = mko::test::jacobian_deviation(
        mko::test::scale_columns(mko::test::run_jacobian(g0, 0), p, p.x0),
        mko::test::scale_columns(ref.r2, p, p.x0), p.set.size());
    std::printf("[GPU] max |dJ|/Schranke %.3e, max. relative Spaltenabweichung GPU/CPU %.3e, "
                "gegen Richardson (skaliert) %.3e\n",
                worst, rel, scaled.frobenius);
    EXPECT_LT(scaled.frobenius, mko::test::kFdJacobianRichardson);
}

// Mit echten paraxialen Zusatzresiduen (M3-T3): Mehrlinser mit Brennweite, Arbeitsabstand und
// Bildebene im paraxialen Fokus, drei Läufe; GPU gegen CPU wie oben und gegen Richardson.
TEST(CudaFdJacobian, ParaxialResidualsMatchCpu) {
    MKO_REQUIRE_GPU();
    const FdProblem p = mko::test::mehrlinser_constraints_problem();
    const std::vector<double> x0 = run_vectors(p, 3);
    FdJacobianOptions opt;
    opt.extra = mko::test::paraxial_extra(p);
    ASSERT_EQ(opt.extra.count, 3);
    TraceEngine engine(0);
    const FdJacobian gpu = mko::cuda::fd_jacobian_gpu(engine, p.set, p.base, x0, opt);
    const FdJacobian cpu = mko::fd_jacobian_cpu(p.set, p.base, x0, opt);
    ASSERT_EQ(gpu.shape.extra_rows, 3);
    const double worst = compare_gpu_cpu(p, x0, opt, gpu, cpu);
    const ResidualLayout layout(p.base.topology());
    const auto ref = mko::test::richardson_reference(p, p.x0, layout, opt.extra);
    const auto scaled = mko::test::jacobian_deviation(
        mko::test::scale_columns(mko::test::run_jacobian(gpu, 0), p, p.x0),
        mko::test::scale_columns(ref.r2, p, p.x0), p.set.size());
    std::printf("[GPU + paraxial] max |dJ|/Schranke %.3e, gegen Richardson (skaliert) %.3e\n",
                worst, scaled.frobenius);
    EXPECT_LT(scaled.frobenius, mko::test::kFdJacobianRichardson);
}

// Mehrere Läufe in einem Batch = Läufe einzeln (bitgleich, auch die Muster), und zwei
// Berechnungen mit gleichen Eingaben sind bitgleich (deterministische Kernel, ADR-0007).
TEST(CudaFdJacobian, MultipleRunsMatchSingleRuns) {
    MKO_REQUIRE_GPU();
    const FdProblem p = mko::test::mehrlinser_problem();
    const int P = p.set.size();
    const std::vector<double> x0 = run_vectors(p, 4);
    FdJacobianOptions opt;
    opt.extra = mko::test::mehrlinser_extra(p.set);
    TraceEngine engine(0);
    DeviceFdJacobian dev;
    dev.compute(engine, p.set, p.base, x0, opt);
    const FdJacobian all = dev.download();
    dev.compute(engine, p.set, p.base, x0, opt); // Puffer wiederverwendet
    EXPECT_TRUE(identical(all, dev.download()));
    for (int i = 0; i < 4; ++i) {
        const std::span<const double> xi(x0.data() + static_cast<std::size_t>(i * P),
                                         static_cast<std::size_t>(P));
        const FdJacobian one = mko::cuda::fd_jacobian_gpu(engine, p.set, p.base, xi, opt);
        EXPECT_TRUE(same_bits(mko::test::run_jacobian(all, i), one.jacobian)) << i;
        EXPECT_TRUE(same_bits(all.residual(i), one.residuals)) << i;
        EXPECT_EQ(all.patterns[static_cast<std::size_t>(i)], one.patterns[0]);
        for (int k = 0; k < P; ++k) {
            EXPECT_EQ(all.status_change(i, k), one.status_changes[static_cast<std::size_t>(k)]);
        }
    }
}

// Tranchen aus ganzen Läufen (Speichergrenze für 1,5 Läufe → ein Lauf pro Tranche) liefern
// bitgleich dasselbe wie ein Lauf in einer Tranche.
TEST(CudaFdJacobian, TranchesMatchSingleTranche) {
    MKO_REQUIRE_GPU();
    const FdProblem p = mko::test::mehrlinser_problem();
    const std::vector<double> x0 = run_vectors(p, 3);
    FdJacobianOptions opt;
    opt.extra = mko::test::mehrlinser_extra(p.set);
    const int n = mko::fd_variant_count(FdScheme::Central, p.set.size());
    const ResidualLayout layout(p.base.topology());
    TraceEngine big(0);
    TraceEngine small(0, limited_to(run_bytes(p.base.topology(), n + n / 2, layout)));
    DeviceFdJacobian a;
    a.compute(big, p.set, p.base, x0, opt);
    EXPECT_EQ(a.run_info().tranches, 1);
    DeviceFdJacobian b;
    b.compute(small, p.set, p.base, x0, opt);
    EXPECT_EQ(b.run_info().tranches, 3);
    EXPECT_EQ(b.run_info().tranche_variants, n);
    EXPECT_TRUE(identical(a.download(), b.download()));
    // Nicht einmal ein Lauf passt: CudaError, danach wieder benutzbar
    TraceEngine tiny(0, limited_to(run_bytes(p.base.topology(), n - 1, layout)));
    EXPECT_THROW(b.compute(tiny, p.set, p.base, x0, opt), CudaError);
    EXPECT_FALSE(b.ready());
    EXPECT_EQ(b.device_jacobian(), nullptr);
    b.compute(big, p.set, p.base, x0, opt);
    EXPECT_TRUE(identical(a.download(), b.download()));
}

// Gerätepfad gegen die Differenzbildung auf dem Host aus den Residuen der Engine (run mit
// Download): Bei allen vier Schablonen (Schranken) und Zusatzresiduen sind zentrale und
// gekürzte Spalten bitgleich ((r_a − r_b)/(d_a − d_b) lässt keine FMA-Kontraktion zu),
// einseitige innerhalb der Rundung der Formel.
TEST(CudaFdJacobian, DevicePathMatchesHostDifferences) {
    MKO_REQUIRE_GPU();
    const FdProblem p = mko::test::einzellinse_bound_problem();
    FdJacobianOptions opt;
    opt.extra = mko::test::einzellinse_extra(p.set);
    TraceEngine engine(0);
    const FdJacobian dev = mko::cuda::fd_jacobian_gpu(engine, p.set, p.base, p.x0, opt);

    const mko::FdJacobianBatch fd = mko::make_fd_jacobian_batch(p.set, p.base, p.x0);
    TraceSettings s;
    s.extra_residuals = opt.extra.count; // Platz für Zusatzresiduen wie dokumentiert
    const mko::cuda::TraceOutput out = engine.run(fd.batch, mko::cuda::TraceMode::Residuals, s);
    const std::vector<double> extra = mko::test::extra_values(fd.batch, opt.extra);
    FdJacobian host;
    mko::fd_jacobian_from_residuals(fd, out.residuals, extra, opt.extra.count, host);

    EXPECT_TRUE(same_bytes(dev.residuals, host.residuals));
    EXPECT_TRUE(same_bytes(dev.status_changes, host.status_changes));
    EXPECT_EQ(dev.patterns, host.patterns);
    const mko::FdPattern& pat = host.patterns[0];
    for (int k = 0; k < p.set.size(); ++k) {
        const auto kk = static_cast<std::size_t>(k);
        if (pat.stencil[kk] != FdStencil::OneSided) {
            EXPECT_TRUE(same_bits(dev.column(0, k), host.column(0, k))) << p.set[k].name;
            continue;
        }
        // Nur die Rundung der Formel (gleiche Residuen): element_bound ohne Residuen-Toleranz
        const std::int64_t rv = host.shape.ray_rows;
        const auto value = [&](int v, std::int64_t r) {
            return r < rv ? out.residuals.residual(v, r)
                          : extra[static_cast<std::size_t>(v) *
                                      static_cast<std::size_t>(opt.extra.count) +
                                  static_cast<std::size_t>(r - rv)];
        };
        for (std::int64_t r = 0; r < host.shape.rows(); ++r) {
            const double bound =
                element_bound(FdStencil::OneSided, pat.step_plus[kk], pat.step_minus[kk],
                              value(0, r), value(2 * k + 1, r), value(2 * k + 2, r), false);
            EXPECT_LE(std::abs(dev.at(0, r, k) - host.at(0, r, k)), bound)
                << p.set[k].name << ", Zeile " << r;
        }
    }
}

// Gebündelter Fall: 1.000 Läufe des Mehrlinsers (27.000 Varianten, 680 Strahlen) in einem
// Aufruf; Lauf 500 bitgleich zum Einzellauf. Gibt die Laufzeit aus (kein Benchmark).
TEST(CudaFdJacobian, LargeBatch) {
    MKO_REQUIRE_GPU();
    const FdProblem p = mko::test::mehrlinser_problem();
    const int P = p.set.size();
    const int runs = 1000;
    const std::vector<double> x0 = run_vectors(p, runs);
    TraceEngine engine(0);
    DeviceFdJacobian dev;
    dev.compute(engine, p.set, p.base, x0); // Aufwärmen (Puffer, Kontext)
    const auto t0 = std::chrono::steady_clock::now();
    dev.compute(engine, p.set, p.base, x0);
    const auto t1 = std::chrono::steady_clock::now();
    const FdJacobian all = dev.download();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("[GPU] %d Läufe × %d Varianten (P = %d, R = %lld): %.1f ms für compute "
                "(Batch auf dem Host, Zusatzresiduen, Trace, Differenzen), %d Tranche(n), "
                "J = %.1f MB\n",
                runs, 2 * P + 1, P, static_cast<long long>(all.shape.rows()), ms,
                dev.run_info().tranches,
                static_cast<double>(all.jacobian.size() * sizeof(double)) / 1e6);
    for (const double v : all.jacobian) {
        ASSERT_TRUE(std::isfinite(v));
    }
    const std::span<const double> x500(x0.data() + static_cast<std::size_t>(500 * P),
                                       static_cast<std::size_t>(P));
    const FdJacobian one = mko::cuda::fd_jacobian_gpu(engine, p.set, p.base, x500);
    EXPECT_TRUE(same_bits(mko::test::run_jacobian(all, 500), one.jacobian));
}

// Status-Wechsel an der Totalreflexion wie auf der CPU (genau ein Strahl in der Spalte c[2]).
TEST(CudaFdJacobian, StatusChangesAtTotalReflection) {
    MKO_REQUIRE_GPU();
    const FdProblem p = mko::test::tir_problem();
    TraceEngine engine(0);
    const FdJacobian gpu = mko::cuda::fd_jacobian_gpu(engine, p.set, p.base, p.x0);
    const FdJacobian cpu = mko::fd_jacobian_cpu(p.set, p.base, p.x0);
    EXPECT_EQ(gpu.status_change(0, 0), 1);
    EXPECT_EQ(gpu.status_change(0, 1), 0);
    EXPECT_TRUE(same_bytes(gpu.status_changes, cpu.status_changes));
}

// Vorwärtsschema auf der GPU wie auf der CPU.
TEST(CudaFdJacobian, ForwardScheme) {
    MKO_REQUIRE_GPU();
    const FdProblem p = mko::test::mehrlinser_problem();
    const std::vector<double> x0 = run_vectors(p, 2);
    FdJacobianOptions opt;
    opt.scheme = FdScheme::Forward;
    TraceEngine engine(0);
    const FdJacobian gpu = mko::cuda::fd_jacobian_gpu(engine, p.set, p.base, x0, opt);
    const FdJacobian cpu = mko::fd_jacobian_cpu(p.set, p.base, x0, opt);
    EXPECT_EQ(gpu.scheme, FdScheme::Forward);
    compare_gpu_cpu(p, x0, opt, gpu, cpu);
}

TEST(CudaFdJacobian, InvalidUse) {
    MKO_REQUIRE_GPU();
    const FdProblem p = mko::test::einzellinse_problem();
    DeviceFdJacobian dev;
    EXPECT_FALSE(dev.ready());
    EXPECT_EQ(dev.device_jacobian(), nullptr);
    EXPECT_EQ(dev.device_residuals(), nullptr);
    EXPECT_EQ(dev.device_status_changes(), nullptr);
    EXPECT_THROW((void)dev.download(), std::logic_error);
    TraceEngine engine(0);
    std::vector<double> bad = p.x0;
    bad.pop_back();
    EXPECT_THROW(dev.compute(engine, p.set, p.base, bad), std::invalid_argument);
    FdJacobianOptions opt;
    opt.extra.count = 1; // ohne Auswertungsfunktion
    EXPECT_THROW(dev.compute(engine, p.set, p.base, p.x0, opt), std::invalid_argument);
    dev.compute(engine, p.set, p.base, p.x0);
    EXPECT_TRUE(dev.ready());
    EXPECT_NE(dev.device_jacobian(), nullptr);
    EXPECT_GE(dev.capacity_bytes(),
              DeviceFdJacobian::buffer_bytes(dev.shape(), 2 * p.set.size() + 1));
    DeviceFdJacobian moved = std::move(dev);
    EXPECT_TRUE(moved.ready());
    EXPECT_FALSE(dev.ready()); // NOLINT(bugprone-use-after-move): Zustand nach dem Verschieben
    moved.release();
    EXPECT_FALSE(moved.ready());
    EXPECT_EQ(moved.capacity_bytes(), 0u);
}

// run_on_device: Tranchen als Vielfache von granularity (die letzte darf kleiner sein, wenn V
// kein Vielfaches ist), lückenlos und in Reihenfolge, Gerätezeiger und Ablage wie
// DeviceResiduals; Fehlerfälle.
TEST(TraceEngineOnDevice, TranchesAreMultiplesOfGranularity) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 10, 77);
    const ResidualLayout layout(batch.topology());
    TraceSettings s;
    s.extra_residuals = 2;
    TraceEngine engine(0, limited_to(run_bytes(batch.topology(), 7, layout, 2)));
    struct Seen {
        int first, count;
        std::int64_t rays, stride;
        bool pointers;
    };
    std::vector<Seen> seen;
    const mko::cuda::TraceRunInfo info =
        engine.run_on_device(batch, s, 3, [&](const DeviceTranche& t) {
            seen.push_back({t.first, t.count, t.rays_per_variant, t.stride,
                            t.residuals != nullptr && t.status != nullptr && t.opd == nullptr &&
                                t.extrema == nullptr});
        });
    EXPECT_EQ(info.variants, 10);
    EXPECT_EQ(info.tranche_variants, 6); // Kapazität 7, abgerundet auf ein Vielfaches von 3
    EXPECT_EQ(info.tranches, 2);
    ASSERT_EQ(seen.size(), 2u);
    EXPECT_EQ(seen[0].first, 0);
    EXPECT_EQ(seen[0].count, 6);
    EXPECT_EQ(seen[1].first, 6);
    EXPECT_EQ(seen[1].count, 4);
    for (const Seen& t : seen) {
        EXPECT_EQ(t.rays, layout.rays_per_variant());
        EXPECT_EQ(t.stride, layout.rays_per_variant() + 2);
        EXPECT_TRUE(t.pointers);
    }
    const auto noop = [](const DeviceTranche&) {};
    EXPECT_THROW((void)engine.run_on_device(batch, s, 0, noop), std::invalid_argument);
    EXPECT_THROW((void)engine.run_on_device(batch, s, 3, {}), std::invalid_argument);
    EXPECT_THROW((void)engine.run_on_device(batch, s, 8, noop), CudaError); // 8 > Kapazität 7
    // Ausnahme aus dem Rückruf wird weitergereicht, die Engine bleibt benutzbar
    EXPECT_THROW((void)engine.run_on_device(
                     batch, s, 1, [](const DeviceTranche&) { throw std::runtime_error("x"); }),
                 std::runtime_error);
    EXPECT_NO_THROW((void)engine.run_on_device(batch, s, 1, noop));
    SystemBatchHost empty = batch;
    empty.resize(0);
    EXPECT_EQ(engine.run_on_device(empty, s, 3, noop).tranches, 0);
}
