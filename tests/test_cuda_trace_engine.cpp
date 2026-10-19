// Host-API TraceEngine (M2-T4, nur mit MIKROPTIM_ENABLE_CUDA): Gerätewahl, Tranchen bei
// künstlich begrenztem Speicher (bitgleich zum Lauf in einer Tranche), Ausgabewahl,
// Fehlerbehandlung, RMS pro Tupel für die CLI.

#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/cuda/cuda_error.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/cuda/tuple_packs.hpp"
#include "mikroptim/merit/wavefront.hpp"
#include "residual_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using mko::BatchTopology;
using mko::OutputRequest;
using mko::ResidualLayout;
using mko::ResidualResult;
using mko::ResidualSettings;
using mko::SystemBatchHost;
using mko::TraceStatus;
using mko::cuda::CudaError;
using mko::cuda::EngineOptions;
using mko::cuda::TraceEngine;
using mko::cuda::TraceMode;
using mko::cuda::TraceOutput;
using mko::cuda::TraceSettings;
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

bool identical(const ResidualResult& a, const ResidualResult& b) {
    return a.variants == b.variants && a.rays_per_variant == b.rays_per_variant &&
           a.stride == b.stride && same_bytes(a.residuals, b.residuals) &&
           same_bytes(a.opd, b.opd) && same_bytes(a.status, b.status);
}

// Gerätespeicher eines Laufs mit `variants` Varianten im Modus RESIDUALS nach
// docs/architektur.md ("Host-API TraceEngine"): Batch + Ausgaben + Feldablage + Tupelpakete.
std::size_t run_bytes(const BatchTopology& t, int variants, const ResidualLayout& layout,
                      const TraceSettings& s) {
    OutputRequest r;
    r.rays_per_variant = layout.rays_per_variant();
    r.penalty_residuals = layout.penalty_rows();
    r.extra_residuals = s.extra_residuals;
    r.residuals = true;
    r.ray_status = true;
    r.opd = s.opd;
    r.merit = false;
    r.surface_max = false;
    return mko::estimate_device_bytes(t, variants, r) +
           mko::align_bytes(static_cast<std::size_t>(t.fields) * sizeof(mko::ResidualFieldPlan)) +
           mko::align_bytes(static_cast<std::size_t>(t.fields) * t.wavelengths *
                            sizeof(mko::cuda::TuplePack)); // Tupelpakete (M7-T4)
}

// Dasselbe im Modus MERIT: Merit, Teilsummen, versagte Strahlen, optional Max-Reduktionen
// und extra_merit (8 Byte pro Variante) + Feldablage + Tupelpakete.
std::size_t merit_run_bytes(const BatchTopology& t, int variants, const ResidualLayout& layout,
                            const TraceSettings& s) {
    OutputRequest r;
    r.rays_per_variant = layout.rays_per_variant();
    r.penalty_residuals = layout.penalty_rows();
    r.residuals = false;
    r.ray_status = false;
    r.opd = false;
    r.merit = true;
    r.surface_max = s.extrema;
    return mko::estimate_device_bytes(t, variants, r, s.extra_merit.empty() ? 0 : sizeof(double)) +
           mko::align_bytes(static_cast<std::size_t>(t.fields) * sizeof(mko::ResidualFieldPlan)) +
           mko::align_bytes(static_cast<std::size_t>(t.fields) * t.wavelengths *
                            sizeof(mko::cuda::TuplePack)); // Tupelpakete (M7-T4)
}

// Engine, deren Speicher genau für `tranche` Varianten reicht (ohne Reserve).
EngineOptions limited_to(std::size_t bytes) {
    EngineOptions o;
    o.memory_limit_bytes = bytes;
    o.reserve = mko::MemoryReserve{0, 0.0};
    return o;
}

} // namespace

// Gerätewahl: ohne GPU CudaError (läuft auch in der CI), sonst Prüfung des Index.
TEST(TraceEngine, DeviceSelection) {
    const int n = mko::cuda::device_count();
    if (n == 0) {
        EXPECT_THROW(TraceEngine{0}, CudaError);
        return;
    }
    EXPECT_THROW(TraceEngine{-1}, std::invalid_argument);
    EXPECT_THROW(TraceEngine{n}, std::invalid_argument);
    EngineOptions bad;
    bad.reserve.fraction = 1.5;
    EXPECT_THROW((TraceEngine{0, bad}), std::invalid_argument);
    const TraceEngine engine(0);
    EXPECT_EQ(engine.device(), 0);
    EXPECT_FALSE(engine.device_name().empty());
    EXPECT_EQ(engine.device_bytes(), 0u);
    std::printf("[ Info     ] TraceEngine auf Gerät 0: %s\n", engine.device_name().c_str());
}

// Abnahme M2-T4: Ein Batch, der nicht in den (künstlich begrenzten) Speicher passt, wird in
// Tranchen verarbeitet; die Ergebnisse sind bitgleich zum Lauf in einer Tranche, die Puffer
// bleiben innerhalb der Grenze und werden über Tranchen und Läufe wiederverwendet.
TEST(TraceEngine, TranchesMatchSingleRun) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 700, 8008);
    const BatchTopology& t = batch.topology();
    TraceSettings s;
    s.opd = true;
    s.extra_residuals = 3;
    const ResidualLayout layout(t, s.quadrature);

    TraceEngine full(0);
    const TraceOutput reference = full.run(batch, TraceMode::Residuals, s);
    ASSERT_EQ(reference.info.tranches, 1);
    EXPECT_EQ(reference.info.tranche_variants, 700);
    EXPECT_GT(reference.residuals.failed_count(), 0u);

    // Grenze für genau 200 Varianten: 700 = 3,5 Tranchen → 4 Kernelstarts (200, 200, 200, 100)
    const std::size_t limit = run_bytes(t, 200, layout, s);
    TraceEngine limited(0, limited_to(limit));
    EXPECT_EQ(limited.max_variants(t, TraceMode::Residuals, s), 200);
    EXPECT_EQ(TraceEngine(0, limited_to(limit - 1)).max_variants(t, TraceMode::Residuals, s), 199);
    TraceOutput out;
    limited.run(batch, TraceMode::Residuals, s, out);
    EXPECT_EQ(out.info.variants, 700);
    EXPECT_EQ(out.info.tranches, 4);
    EXPECT_EQ(out.info.tranche_variants, 200);
    EXPECT_EQ(out.info.budget_bytes, limit);
    EXPECT_EQ(out.info.device_bytes, limit); // Puffer genau für eine volle Tranche
    EXPECT_EQ(limited.device_bytes(), limit);
    EXPECT_TRUE(identical(out.residuals, reference.residuals));
    EXPECT_EQ(out.layout.rays_per_variant(), layout.rays_per_variant());

    // Gegen die CPU-Referenz (Status gleich, Residuen innerhalb der Toleranz von M2-T2, für
    // Residuen in Wellen umgerechnet mit r = u·OPD, M4-T1)
    const ResidualResult cpu = mko::residuals_cpu(batch, layout);
    const std::vector<double> units = mko::test::residual_units(batch, layout);
    std::size_t violations = 0;
    for (int v = 0; v < 700; ++v) {
        for (std::int64_t i = 0; i < layout.rays_per_variant(); ++i) {
            const double d = std::abs(out.residuals.residual(v, i) - cpu.residual(v, i));
            const double bound =
                mko::test::residual_bound(units[static_cast<std::size_t>(i)], cpu.residual(v, i),
                                          kCpuGpuFp64, kCpuGpuFp64Relative);
            violations +=
                (out.residuals.status_at(v, i) != cpu.status_at(v, i) || !(d <= bound)) ? 1 : 0;
        }
        for (std::int64_t e = layout.trace_rows(); e < out.residuals.stride; ++e) {
            EXPECT_EQ(out.residuals.residual(v, e), 0.0);
        }
    }
    EXPECT_EQ(violations, 0u);

    // Zweiter Lauf mit derselben Engine und Ausgabe: 450 Varianten (200, 200, 50), gleiche
    // Puffer; die ersten 450 Varianten sind dieselben wie oben (gleicher Seed).
    const SystemBatchHost prefix = mko::test::perturbed_batch(ls, 450, 8008);
    limited.run(prefix, TraceMode::Residuals, s, out);
    EXPECT_EQ(out.info.tranches, 3);
    EXPECT_EQ(limited.device_bytes(), limit);
    ASSERT_EQ(out.residuals.variants, 450);
    const auto rays = static_cast<std::size_t>(layout.rays_per_variant());
    const auto stride = static_cast<std::size_t>(out.residuals.stride);
    EXPECT_EQ(std::memcmp(out.residuals.residuals.data(), reference.residuals.residuals.data(),
                          450 * stride * sizeof(double)),
              0);
    EXPECT_EQ(std::memcmp(out.residuals.opd.data(), reference.residuals.opd.data(),
                          450 * rays * sizeof(double)),
              0);
    EXPECT_EQ(
        std::memcmp(out.residuals.status.data(), reference.residuals.status.data(), 450 * rays), 0);

    // Ohne OPD passen mehr Varianten in dieselbe Grenze; Residuen unverändert.
    TraceSettings lean = s;
    lean.opd = false;
    lean.status = false;
    const int lean_capacity = limited.max_variants(t, TraceMode::Residuals, lean);
    EXPECT_GT(lean_capacity, 200);
    EXPECT_LT(lean_capacity, 700);
    limited.run(batch, TraceMode::Residuals, lean, out);
    EXPECT_EQ(out.info.tranche_variants, lean_capacity);
    EXPECT_EQ(out.info.tranches, (700 + lean_capacity - 1) / lean_capacity);
    EXPECT_LE(limited.device_bytes(), limit);
    EXPECT_TRUE(out.residuals.opd.empty());
    EXPECT_TRUE(out.residuals.status.empty());
    EXPECT_TRUE(same_bytes(out.residuals.residuals, reference.residuals.residuals));
    std::printf("[ Info     ] Grenze %zu Byte: 200 Varianten pro Tranche mit OPD (4 Tranchen), "
                "%d ohne OPD und Status (%d Tranchen)\n",
                limit, lean_capacity, out.info.tranches);
}

// MERIT (M2-T3) über mehrere Tranchen ist bitgleich zum Lauf in einer Tranche: Merit,
// versagte Strahlen, Max-Reduktionen und der Anteil weiterer Residuen (extra_merit wird pro
// Tranche an der Stelle `first` gelesen). Dasselbe für die Max-Reduktionen im Modus
// RESIDUALS; Wechsel der Modi mit derselben Engine.
TEST(TraceEngine, MeritTranchesMatchSingleRun) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 700, 8008);
    const BatchTopology& t = batch.topology();
    std::vector<double> extra(700);
    for (std::size_t v = 0; v < extra.size(); ++v) {
        extra[v] = 1e-6 * static_cast<double>(v % 17);
    }
    TraceSettings s;
    s.extrema = true;
    s.extra_merit = extra;
    const ResidualLayout layout(t, s.quadrature);

    TraceEngine full(0);
    const TraceOutput reference = full.run(batch, TraceMode::Merit, s);
    ASSERT_EQ(reference.info.tranches, 1);
    EXPECT_EQ(reference.mode, TraceMode::Merit);
    EXPECT_EQ(reference.merit.variants, 700);
    EXPECT_EQ(reference.merit.extrema.surfaces, t.surfaces);
    EXPECT_EQ(reference.residuals.variants, 0); // Ergebnis des anderen Modus leer
    EXPECT_TRUE(reference.residuals.residuals.empty());
    std::int64_t failed = 0;
    for (const std::int64_t f : reference.merit.failed) {
        failed += f;
    }
    EXPECT_GT(failed, 0);

    // Grenze für genau 200 Varianten: 4 Kernelstarts (200, 200, 200, 100)
    const std::size_t limit = merit_run_bytes(t, 200, layout, s);
    TraceEngine limited(0, limited_to(limit));
    EXPECT_EQ(limited.max_variants(t, TraceMode::Merit, s), 200);
    EXPECT_EQ(TraceEngine(0, limited_to(limit - 1)).max_variants(t, TraceMode::Merit, s), 199);
    TraceOutput out;
    limited.run(batch, TraceMode::Merit, s, out);
    EXPECT_EQ(out.info.tranches, 4);
    EXPECT_EQ(out.info.tranche_variants, 200);
    EXPECT_EQ(limited.device_bytes(), limit);
    EXPECT_TRUE(same_bytes(out.merit.merit, reference.merit.merit));
    EXPECT_TRUE(same_bytes(out.merit.failed, reference.merit.failed));
    EXPECT_TRUE(same_bytes(out.merit.extrema.values, reference.merit.extrema.values));
    // Der Anteil weiterer Residuen wird zuletzt addiert
    TraceSettings plain = s;
    plain.extra_merit = {};
    const TraceOutput without = full.run(batch, TraceMode::Merit, plain);
    for (std::size_t v = 0; v < extra.size(); ++v) {
        const double expected = without.merit.merit[v] + extra[v];
        EXPECT_EQ(std::memcmp(&out.merit.merit[v], &expected, sizeof(double)), 0) << v;
    }

    // RESIDUALS mit Max-Reduktionen in derselben Grenze (mehr Speicher pro Variante, also mehr
    // Tranchen); bitgleich zum Lauf in einer Tranche und zu den Extrema aus MERIT.
    TraceSettings rs;
    rs.extrema = true;
    const TraceOutput r_full = full.run(batch, TraceMode::Residuals, rs);
    limited.run(batch, TraceMode::Residuals, rs, out);
    EXPECT_GT(out.info.tranches, 4);
    EXPECT_LE(limited.device_bytes(), limit);
    EXPECT_TRUE(out.merit.merit.empty());
    EXPECT_TRUE(identical(out.residuals, r_full.residuals));
    EXPECT_TRUE(same_bytes(out.residuals.extrema.values, r_full.residuals.extrema.values));
    EXPECT_TRUE(same_bytes(out.residuals.extrema.values, reference.merit.extrema.values));

    // Wechsel der Modi ohne Grenze: beide Ausgabepuffer bleiben erhalten, Ergebnisse gleich
    const std::size_t both = full.device_bytes();
    for (int i = 0; i < 2; ++i) {
        EXPECT_TRUE(
            same_bytes(full.run(batch, TraceMode::Merit, s).merit.merit, reference.merit.merit));
        EXPECT_TRUE(
            identical(full.run(batch, TraceMode::Residuals, rs).residuals, r_full.residuals));
        EXPECT_EQ(full.device_bytes(), both);
    }
    std::printf("[ Info     ] MERIT mit Extrema und extra_merit: Grenze %zu Byte = 200 Varianten "
                "pro Tranche; RESIDUALS mit Extrema in derselben Grenze: %d Tranchen\n",
                limit, out.info.tranches);
}

// Eine Tranche pro Variante (kleinste Grenze) ist ebenfalls bitgleich.
TEST(TraceEngine, OneVariantPerTranche) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("einzellinse_bk7.json");
    const SystemBatchHost batch = mko::test::perturbed_batch(ls, 7, 9009);
    TraceSettings s;
    s.opd = true;
    const ResidualLayout layout(batch.topology(), s.quadrature);
    TraceEngine single(0, limited_to(run_bytes(batch.topology(), 1, layout, s)));
    const TraceOutput out = single.run(batch, TraceMode::Residuals, s);
    EXPECT_EQ(out.info.tranches, 7);
    EXPECT_EQ(out.info.tranche_variants, 1);
    EXPECT_TRUE(
        identical(out.residuals, TraceEngine(0).run(batch, TraceMode::Residuals, s).residuals));
}

// Passt nicht einmal eine Variante, wirft der Lauf CudaError; die Engine bleibt benutzbar.
TEST(TraceEngine, TooLittleMemoryThrows) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("einzellinse_bk7.json");
    const SystemBatchHost batch = mko::make_system_batch(ls, 3);
    const TraceSettings s;
    const ResidualLayout layout(batch.topology(), s.quadrature);
    TraceEngine tiny(0, limited_to(run_bytes(batch.topology(), 1, layout, s) - 1));
    EXPECT_EQ(tiny.max_variants(batch.topology(), TraceMode::Residuals, s), 0);
    TraceOutput out;
    EXPECT_THROW(tiny.run(batch, TraceMode::Residuals, s, out), CudaError);
    // Die Standardreserve (mindestens 256 MiB) bleibt auch bei einer Grenze frei.
    EngineOptions small;
    small.memory_limit_bytes = std::size_t{200} << 20;
    TraceEngine reserved(0, small);
    EXPECT_EQ(reserved.max_variants(batch.topology(), TraceMode::Residuals, s), 0);
    EXPECT_THROW(reserved.run(batch, TraceMode::Residuals, s, out), CudaError);
    // Leerer Batch braucht keinen Speicher
    SystemBatchHost empty = batch;
    empty.resize(0);
    EXPECT_NO_THROW(tiny.run(empty, TraceMode::Residuals, s, out));
    EXPECT_EQ(out.info.tranches, 0);
    EXPECT_EQ(out.residuals.variants, 0);
    EXPECT_TRUE(out.residuals.residuals.empty());
    EXPECT_TRUE(out.residuals.status.empty());
}

// Freier Speicher ohne Grenze: max_variants folgt cudaMemGetInfo mit Standardreserve.
TEST(TraceEngine, MaxVariantsFromFreeMemory) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const BatchTopology t = mko::make_system_batch(ls, 1).topology();
    const TraceEngine engine(0);
    const TraceSettings s;
    const int n = engine.max_variants(t, TraceMode::Residuals, s);
    const mko::cuda::DeviceMemoryInfo mem = mko::cuda::device_memory_info(0);
    const ResidualLayout layout(t, s.quadrature);
    EXPECT_GT(n, 1000);
    EXPECT_LE(run_bytes(t, n, layout, s), mko::usable_bytes(mem.free_bytes));
    std::printf("[ Info     ] %.2f GiB frei: bis %d Varianten von testsystem_mehrlinser.json pro "
                "Tranche (%lld Strahlen pro Variante)\n",
                static_cast<double>(mem.free_bytes) / (1024.0 * 1024.0 * 1024.0), n,
                static_cast<long long>(layout.rays_per_variant()));
}

// Ungültige Eingaben: std::invalid_argument vor jeder Gerätearbeit; die Engine bleibt
// benutzbar.
TEST(TraceEngine, InvalidInputThrows) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem ls = mko::test::load_residual_system("einzellinse_bk7.json");
    const SystemBatchHost batch = mko::make_system_batch(ls, 2);
    TraceEngine engine(0);
    TraceOutput out;
    // MERIT: extra_merit mit falscher Länge, zu viele Stützstellen für das Shared Memory
    const std::vector<double> extra(3, 0.0);
    TraceSettings merit_bad;
    merit_bad.extra_merit = extra;
    EXPECT_THROW(engine.run(batch, TraceMode::Merit, merit_bad, out), std::invalid_argument);
    merit_bad = TraceSettings{};
    merit_bad.quadrature = ResidualSettings{16, mko::cuda::merit_max_samples(false) / 16 + 1};
    EXPECT_THROW(engine.run(batch, TraceMode::Merit, merit_bad, out), std::invalid_argument);
    EXPECT_STREQ(mko::cuda::to_string(TraceMode::Merit), "MERIT");
    EXPECT_STREQ(mko::cuda::to_string(TraceMode::Residuals), "RESIDUALS");
    TraceSettings bad;
    bad.extra_residuals = -1;
    EXPECT_THROW(engine.run(batch, TraceMode::Residuals, bad, out), std::invalid_argument);
    bad = TraceSettings{};
    bad.quadrature.rings = 0;
    EXPECT_THROW(engine.run(batch, TraceMode::Residuals, bad, out), std::invalid_argument);
    bad.quadrature = ResidualSettings{17, 8};
    EXPECT_THROW(engine.run(batch, TraceMode::Residuals, bad, out), std::invalid_argument);
    // Nicht endlicher Wert, Variante verlässt das Achsfeld
    SystemBatchHost nan = batch;
    nan.thickness(1, 1) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(engine.run(nan, TraceMode::Residuals, {}, out), std::invalid_argument);
    SystemBatchHost off_axis = batch;
    off_axis.field_angle(1, 0) = 0.01;
    EXPECT_THROW(engine.run(off_axis, TraceMode::Residuals, {}, out), std::invalid_argument);
    // Pupillenarten nicht festgelegt
    BatchTopology open = batch.topology();
    open.pupil_mode.clear();
    EXPECT_THROW(engine.run(SystemBatchHost(open, 1), TraceMode::Residuals, {}, out),
                 std::invalid_argument);
    EXPECT_EQ(engine.device_bytes(), 0u); // keine Gerätearbeit vor der Prüfung
    // Danach läuft die Engine normal
    engine.run(batch, TraceMode::Residuals, {}, out);
    EXPECT_EQ(out.residuals.variants, 2);
    EXPECT_EQ(out.residuals.failed_count(), 0u);
}

// Kein globaler Zustand: zwei Engines auf demselben Gerät im Wechsel, eine verschobene Engine
// und release() liefern dieselben Ergebnisse.
TEST(TraceEngine, EnginesAreIndependent) {
    MKO_REQUIRE_GPU();
    const mko::LoadedSystem a = mko::test::load_residual_system("einzellinse_bk7.json");
    const mko::LoadedSystem b = mko::test::load_residual_system("testsystem_mehrlinser.json");
    const SystemBatchHost batch_a = mko::test::perturbed_batch(a, 50, 1);
    const SystemBatchHost batch_b = mko::test::perturbed_batch(b, 80, 2);
    TraceSettings s;
    s.opd = true;
    TraceEngine first(0);
    TraceEngine second(0);
    const TraceOutput ra = first.run(batch_a, TraceMode::Residuals, s);
    const TraceOutput rb = second.run(batch_b, TraceMode::Residuals, s);
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(
            identical(second.run(batch_a, TraceMode::Residuals, s).residuals, ra.residuals));
        EXPECT_TRUE(identical(first.run(batch_b, TraceMode::Residuals, s).residuals, rb.residuals));
    }
    TraceEngine moved(std::move(first));
    EXPECT_EQ(moved.device(), 0);
    EXPECT_TRUE(identical(moved.run(batch_a, TraceMode::Residuals, s).residuals, ra.residuals));
    moved.release();
    EXPECT_EQ(moved.device_bytes(), 0u);
    EXPECT_TRUE(identical(moved.run(batch_b, TraceMode::Residuals, s).residuals, rb.residuals));
}

// Die Ausgabe reicht für `mikroptim trace --gpu` (M2-T5): RMS und versagte Strahlen pro Feld
// und Wellenlänge aus OPD und Status (tuple_wavefronts) stimmen mit evaluate_wavefront überein.
TEST(TraceEngine, TupleWavefrontsMatchEvaluateWavefront) {
    MKO_REQUIRE_GPU();
    TraceEngine engine(0);
    for (const char* file : {"einzellinse_bk7.json", "testsystem_mehrlinser.json"}) {
        SCOPED_TRACE(file);
        const mko::LoadedSystem ls = mko::test::load_residual_system(file);
        const mko::WavefrontResult wf = mko::evaluate_wavefront(ls);
        TraceSettings s;
        s.opd = true;
        const TraceOutput out = engine.run(mko::make_system_batch(ls, 1), TraceMode::Residuals, s);
        const std::vector<mko::TupleWavefront> t =
            mko::tuple_wavefronts(out.residuals, out.layout, 0);
        ASSERT_EQ(t.size(), wf.maps.size());
        double max_diff = 0.0;
        for (std::size_t i = 0; i < t.size(); ++i) {
            EXPECT_EQ(t[i].failed, wf.maps[i].failed);
            ASSERT_TRUE(t[i].valid());
            const double d = std::abs(t[i].rms - wf.maps[i].rms);
            EXPECT_LE(d, kCpuGpuFp64);
            max_diff = std::max(max_diff, d);
        }
        std::printf("[ Info     ] %s: %zu Tupel, max |RMS_GPU - RMS_CPU| = %.3e mm\n", file,
                    t.size(), max_diff);
    }
}
