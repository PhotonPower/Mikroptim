// Toleranzbewusste Optimierung (M11-T6), CPU. Abnahme am konstruierten Orakel
// data/systems/kittblock_toleranzbewusst.json: Glasblock aus zwei Teilen desselben Glases mit
// gekrümmter Kittfläche c im kollimierten Strahl vor einer Sammellinse. Nominal ist die Kittfläche
// unsichtbar (gleiche Brechzahl auf beiden Seiten: keine Brechung, gleiche optische Weglänge),
// also F(c) = F(0) für jede Krümmung. Mit der Brechzahltoleranz Δn des hinteren Teils hat die
// Kittfläche die Brechkraft Δn·c; bei c = 0 bleibt nur die ebene Rückfläche im kollimierten
// Strahl, deren Wirkung ein Kolben ist (F(0, ±Δn) = F(0)). Lösung A (c = 0,05/mm) und Lösung B
// (c = 0) sind nominal gleichwertig, B ist unempfindlich. Dazu: Einzelrechnung der gestörten
// Systemdateien, Kettenregel für Störungen von Variablen (Radius relativ, Dicke, Arbeitsabstand)
// gegen finite Differenzen der erwarteten Merit, Bitgleichheit gebündelter und einzelner Läufe,
// Bestätigung durch die Monte-Carlo-Tolerierung (M8-T4). GPU-Teil: test_cuda_tolerance.cpp.

#include "mikroptim/batch/batch_merit.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "mikroptim/tolerance/tolerance_analysis.hpp"
#include "mikroptim/tolerance/tolerance_aware.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using nlohmann::json;

namespace {

const std::string kBlockFile =
    std::string(MKO_SOURCE_DIR) + "/data/systems/kittblock_toleranzbewusst.json";

json block_json() {
    std::ifstream in(kBlockFile);
    std::stringstream ss;
    ss << in.rdbuf();
    return json::parse(ss.str());
}

mko::LoadedSystem load_json(const json& j) {
    return mko::prepare_system(mko::read_system_json(j.dump()),
                               std::filesystem::path(kBlockFile).parent_path());
}

// Geladenes System mit Ausgangsbatch und Variablen (Backends halten Referenzen darauf).
struct Problem {
    explicit Problem(const json& j)
        : loaded(load_json(j)), base(mko::make_system_batch(loaded, 1)),
          set(loaded.system, base.topology()) {}
    mko::LoadedSystem loaded;
    mko::SystemBatchHost base;
    mko::VariableSet set;
};

mko::LmOptions lm_options() {
    mko::LmOptions o;
    o.max_iterations = 200;
    o.threads = 1;
    o.jacobian.extra_threads = 1;
    return o;
}

mko::ToleranceAwareSettings plus_minus(double alpha) {
    mko::ToleranceAwareSettings s;
    s.alpha = alpha;
    return s;
}

constexpr double kSolutionA = 0.05; // c der Kittfläche [1/mm], empfindlich
constexpr double kSolutionB = 0.0;  // unempfindlich

// Merit der Teilsysteme (nominal, gestört) an einem Punkt.
mko::ToleranceAwareMerits parts(const mko::ToleranceAwareModel& model, const Problem& p,
                                const mko::LmOptions& o, std::span<const double> x) {
    mko::ToleranceAwareBackend backend(model, mko::cpu_tolerance_inner(), p.base, o);
    const int run0 = 0;
    return backend.merits(std::span<const int>(&run0, 1), x);
}

// F einer einzeln geladenen Systemdatei (nicht koaxial, volle Pupille, wie die inneren Läufe).
double single_merit(const json& j, const mko::LmOptions& o) {
    const mko::LoadedSystem loaded = load_json(j);
    const mko::SystemBatchHost batch = mko::make_non_coaxial(mko::make_system_batch(loaded, 1));
    const mko::ResidualLayout layout(batch.topology(), o.jacobian.quadrature);
    return mko::merit_cpu(batch, 0, layout, false).merit[0];
}

double relative(double a, double b) {
    return std::abs(a - b) / std::max(std::abs(b), 1e-300);
}

} // namespace

// Orakel: nominal gleichwertig (unsichtbare Kittfläche), B unempfindlich, A nicht; die gestörten
// Teilsysteme stimmen mit Einzelrechnungen der gestörten Dateien (n₀ ± Δn) überein.
TEST(ToleranceAware, OracleNominallyEqualDifferentSensitivity) {
    const Problem p(block_json());
    const mko::LmOptions o = lm_options();
    const mko::ToleranceAwareModel model(p.loaded.system, p.base, p.set, plus_minus(0.5));
    ASSERT_EQ(model.samples(), 2);
    std::vector<double> xa = p.set.extract(p.base, 0);
    ASSERT_EQ(xa.size(), 2u);
    ASSERT_EQ(xa[0], kSolutionA);
    std::vector<double> xb = xa;
    xb[0] = kSolutionB;
    const mko::ToleranceAwareMerits ma = parts(model, p, o, xa);
    const mko::ToleranceAwareMerits mb = parts(model, p, o, xb);
    ASSERT_EQ(ma.merit.size(), 3u);
    // nominal gleich (bis auf Rundung)
    const double nominal = relative(ma.merit[0], mb.merit[0]);
    EXPECT_LE(nominal, mko::test::kToleranceAwareInvisible) << "F(A) = " << ma.merit[0];
    // B: gestört wie nominal; A: deutlich schlechter
    for (std::size_t s = 1; s < 3; ++s) {
        EXPECT_LE(relative(mb.merit[s], mb.merit[0]), mko::test::kToleranceAwareInvisible);
        EXPECT_GT(ma.merit[s], 2.0 * ma.merit[0]) << "s = " << s;
    }
    std::printf("[ OK ] Orakel: F nominal A %.12e, B %.12e (rel. %.2e); gestört A %.6e / %.6e, "
                "B %.12e / %.12e\n",
                ma.merit[0], mb.merit[0], nominal, ma.merit[1], ma.merit[2], mb.merit[1],
                mb.merit[2]);
    // Einzelrechnung: Datei mit n₀ ± Δn des hinteren Teils
    for (int sign = -1; sign <= 1; sign += 2) {
        json j = block_json();
        const double n0 = j["surfaces"][2]["material"]["n0"].get<double>();
        j["surfaces"][2]["material"]["n0"] = n0 + sign * 0.002;
        const double single = single_merit(j, o);
        const double batch = ma.merit[sign > 0 ? 1 : 2];
        EXPECT_LE(relative(batch, single), mko::test::kSensitivitySingleSystem)
            << "Vorzeichen " << sign << ": Batch " << batch << ", Einzelrechnung " << single;
    }
}

// Abnahme: Die toleranzbewusste Optimierung wählt von den nominal gleichwertigen Lösungen A und B
// die unempfindlichere. (a) F_tol(A) > F_tol(B), also wählt die Auswahl unter Läufen B; (b) vom
// Start A aus läuft sie nach B (c → 0), während die gewöhnliche Optimierung bei A bleibt.
TEST(ToleranceAware, PicksInsensitiveSolution) {
    const Problem p(block_json());
    const mko::LmOptions o = lm_options();
    const mko::ToleranceAwareModel model(p.loaded.system, p.base, p.set, plus_minus(0.5));
    const std::vector<double> xa = p.set.extract(p.base, 0);
    std::vector<double> xb = xa;
    xb[0] = kSolutionB;
    // (a) beide Lösungen als Läufe, nur auswerten
    std::vector<double> both = xa;
    both.insert(both.end(), xb.begin(), xb.end());
    mko::LmOptions eval = o;
    eval.max_iterations = 0;
    const mko::LmResult r0 = mko::optimize_lm_tolerance_aware(model, mko::cpu_tolerance_inner(),
                                                              p.set, p.base, both, eval);
    ASSERT_EQ(r0.runs.size(), 2u);
    EXPECT_GT(r0.runs[0].merit, 1.5 * r0.runs[1].merit);
    // (b) Optimierung ab A
    const mko::LmResult plain = mko::optimize_lm_cpu(p.set, p.base, xa, o);
    const mko::LmResult tol =
        mko::optimize_lm_tolerance_aware(model, mko::cpu_tolerance_inner(), p.set, p.base, xa, o);
    const mko::LmRunResult& rp = plain.runs.at(0);
    const mko::LmRunResult& rt = tol.runs.at(0);
    EXPECT_TRUE(mko::lm_converged(rt.status)) << mko::to_string(rt.status);
    EXPECT_LE(std::abs(rp.x[0] - kSolutionA), mko::test::kToleranceAwareStays * kSolutionA)
        << "gewöhnlich: c = " << rp.x[0];
    EXPECT_LE(std::abs(rt.x[0]), mko::test::kToleranceAwareChosen * kSolutionA)
        << "toleranzbewusst: c = " << rt.x[0];
    // Am Ende: gestörte Teilsysteme wie das nominelle, nominal so gut wie A
    const mko::ToleranceAwareMerits m = parts(model, p, o, rt.x);
    const mko::ToleranceAwareMerits ma = parts(model, p, o, xa);
    EXPECT_LE(relative(m.merit[0], ma.merit[0]), mko::test::kToleranceAwareNominal);
    for (std::size_t s = 1; s < m.merit.size(); ++s) {
        EXPECT_LE(relative(m.merit[s], m.merit[0]), mko::test::kToleranceAwareNominal);
    }
    std::printf("[ OK ] ab A: gewöhnlich c = %.3e (%d Zyklen, %s), toleranzbewusst c = %.3e "
                "(%d Zyklen, %s), F_tol %.6e -> %.6e; F_tol(A) %.6e, F_tol(B) %.6e\n",
                rp.x[0], rp.iterations, mko::to_string(rp.status), rt.x[0], rt.iterations,
                mko::to_string(rt.status), rt.history.front(), rt.merit, r0.runs[0].merit,
                r0.runs[1].merit);
}

// Die Monte-Carlo-Tolerierung (M8-T4, Toleranzsatz der Datei, ohne Kompensator) bestätigt die
// bessere Verteilung des toleranzbewussten Ergebnisses.
TEST(ToleranceAware, MonteCarloConfirmsBetterDistribution) {
    const Problem p(block_json());
    const mko::LmOptions o = lm_options();
    const mko::ToleranceAwareModel model(p.loaded.system, p.base, p.set, plus_minus(0.5));
    const std::vector<double> xa = p.set.extract(p.base, 0);
    const mko::LmResult tol =
        mko::optimize_lm_tolerance_aware(model, mko::cpu_tolerance_inner(), p.set, p.base, xa, o);
    const auto analyse = [&](std::span<const double> x) {
        mko::SystemDescription sys = p.loaded.system;
        p.set.apply_to_description(sys, x);
        const mko::LoadedSystem loaded =
            mko::prepare_system(sys, std::filesystem::path(kBlockFile).parent_path());
        mko::ToleranceSettings ts;
        ts.samples = 200;
        ts.seed = 2026;
        ts.threads = 1;
        return mko::run_tolerance_analysis(loaded.system, mko::make_system_batch(loaded, 1), ts);
    };
    const mko::ToleranceResult ra = analyse(xa);
    const mko::ToleranceResult rb = analyse(tol.runs.at(0).x);
    EXPECT_GT(ra.rms_composite.mean, 1.1 * rb.rms_composite.mean);
    EXPECT_LE(rb.rms_composite.max, rb.nominal.rms_composite * (1.0 + 1e-6));
    std::printf("[ OK ] Monte Carlo (200): RMS Mittel A %.6e, Ergebnis %.6e (nominal %.6e), "
                "95 %% A %.6e, Ergebnis %.6e\n",
                ra.rms_composite.mean, rb.rms_composite.mean, rb.nominal.rms_composite,
                ra.rms_composite.p95, rb.rms_composite.p95);
}

// Störungen von Variablen (relativer Radius an variablen Krümmungen, Dicke eines variablen
// Luftspalts, axialer Anteil der Elementverkippung in Luftspalt und Arbeitsabstand):
// Teilsysteme gleich dem direkt gestörten Batch (nominell mit x, dann ToleranceModel::apply), und
// der Gradient Jᵀr der Zusammensetzung (Kettenregel D_s) gleich der Ableitung von F_tol/2.
TEST(ToleranceAware, ChainRuleForPerturbedVariables) {
    json j = block_json();
    j["variables"]["items"] =
        json::array({{{"type", "curvature"}, {"surfaces", {2, 4, 5}}},
                     {{"type", "thickness"}, {"surface", 3}},
                     {{"type", "working_distance"}, {"min", 10.0}, {"max", 30.0}}});
    j["tolerances"]["items"] =
        json::array({{{"type", "radius"}, {"surfaces", {4, 5}}, {"relative", 0.05}},
                     {{"type", "thickness"}, {"surfaces", {3, 4}}, {"delta", 0.2}},
                     {{"type", "index"}, {"surface", 2}, {"delta", 0.002}},
                     {{"type", "element_tilt"}, {"element", {4, 5}}, {"arcmin", 20.0}}});
    const Problem p(j);
    mko::LmOptions o = lm_options();
    o.jacobian.quadrature.failure = mko::FailureMode::Hard;
    double max_direct = 0.0;
    double max_gradient = 0.0;
    for (const mko::ToleranceAwareScheme scheme :
         {mko::ToleranceAwareScheme::PlusMinus, mko::ToleranceAwareScheme::Samples}) {
        mko::ToleranceAwareSettings ts = plus_minus(0.7);
        ts.scheme = scheme;
        ts.samples = 5;
        ts.seed = 11;
        const mko::ToleranceAwareModel model(p.loaded.system, p.base, p.set, ts);
        const std::vector<double> x = p.set.extract(p.base, 0);
        const auto P = static_cast<std::size_t>(p.set.size());
        mko::ToleranceAwareBackend backend(model, mko::cpu_tolerance_inner(), p.base, o);
        const int run0 = 0;
        const std::span<const int> run(&run0, 1);
        // (a) Teilsysteme gegen den direkt gestörten Batch
        const mko::ToleranceAwareMerits m = backend.merits(run, x);
        const mko::SystemBatchHost nominal = mko::make_non_coaxial(p.base);
        const mko::ResidualLayout layout(nominal.topology(), o.jacobian.quadrature);
        for (int s = 0; s <= model.samples(); ++s) {
            mko::SystemBatchHost direct = nominal;
            model.inner_set().apply(direct, 0, x);
            if (s > 0) {
                model.model().apply(direct, 0, model.deviates(s));
            }
            const double f = mko::merit_cpu(direct, 0, layout, false).merit[0];
            max_direct = std::max(max_direct, relative(m.merit[static_cast<std::size_t>(s)], f));
            EXPECT_LE(relative(m.merit[static_cast<std::size_t>(s)], f),
                      mko::test::kToleranceAwareDirect)
                << to_string(scheme) << " s = " << s;
        }
        // (b) Gradient: ∂F_tol/∂x_k = 2 g_k, zentrale Differenz von F_tol
        std::vector<double> a(P * P);
        std::vector<double> g(P);
        backend.normal_equations(run, x, a, g);
        for (std::size_t k = 0; k < P; ++k) {
            const double h = 1e-5 * std::max(std::abs(x[k]), p.set[static_cast<int>(k)].typical);
            std::vector<double> up = x;
            std::vector<double> down = x;
            up[k] += h;
            down[k] -= h;
            std::vector<double> fu(1);
            std::vector<double> fd(1);
            std::vector<std::int64_t> n1(1);
            std::vector<std::int64_t> n2(1);
            backend.merit(run, up, fu, n1, n2);
            backend.merit(run, down, fd, n1, n2);
            const double fd_grad = (fu[0] - fd[0]) / (up[k] - down[k]);
            double scale = 0.0;
            for (std::size_t l = 0; l < P; ++l) {
                scale = std::max(scale,
                                 std::abs(2.0 * g[l]) *
                                     std::max(std::abs(x[l]), p.set[static_cast<int>(l)].typical));
            }
            const double s_k = std::max(std::abs(x[k]), p.set[static_cast<int>(k)].typical);
            max_gradient = std::max(max_gradient, std::abs(2.0 * g[k] - fd_grad) * s_k / scale);
            EXPECT_LE(std::abs(2.0 * g[k] - fd_grad) * s_k,
                      mko::test::kToleranceAwareGradient * scale)
                << to_string(scheme) << " Variable " << p.set[static_cast<int>(k)].name
                << ": 2g = " << 2.0 * g[k] << ", FD " << fd_grad;
        }
    }
    std::printf("[ OK ] Teilsysteme gegen direkt gestörten Batch: %.2e relativ; Gradient gegen FD "
                "von F_tol: %.2e\n",
                max_direct, max_gradient);
}

// Gebündelte Läufe bitgleich zu einzelnen (beide Schemata), Stichproben nur von (seed, s)
// abhängig.
TEST(ToleranceAware, JointRunsBitwiseEqualToSingleRuns) {
    const Problem p(block_json());
    mko::LmOptions o = lm_options();
    o.max_iterations = 15;
    o.threads = 0;
    const std::vector<double> x0 = p.set.extract(p.base, 0);
    std::vector<double> starts;
    for (const double c : {0.05, -0.03, 0.08}) {
        starts.push_back(c);
        starts.push_back(x0[1] + c);
    }
    for (const mko::ToleranceAwareScheme scheme :
         {mko::ToleranceAwareScheme::PlusMinus, mko::ToleranceAwareScheme::Samples}) {
        mko::ToleranceAwareSettings ts = plus_minus(0.5);
        ts.scheme = scheme;
        ts.samples = 3;
        ts.seed = 5;
        const mko::ToleranceAwareModel model(p.loaded.system, p.base, p.set, ts);
        if (scheme == mko::ToleranceAwareScheme::Samples) {
            for (int s = 1; s <= 3; ++s) {
                const std::vector<double> xi =
                    model.model().deviates(5, static_cast<std::uint64_t>(s - 1));
                const std::span<const double> got = model.deviates(s);
                EXPECT_TRUE(std::equal(got.begin(), got.end(), xi.begin(), xi.end()));
            }
        }
        const mko::LmResult joint = mko::optimize_lm_tolerance_aware(
            model, mko::cpu_tolerance_inner(), p.set, p.base, starts, o);
        for (std::size_t i = 0; i < 3; ++i) {
            const mko::LmResult one = mko::optimize_lm_tolerance_aware(
                model, mko::cpu_tolerance_inner(), p.set, p.base,
                std::span<const double>(starts).subspan(2 * i, 2), o);
            const mko::LmRunResult& a = joint.runs[i];
            const mko::LmRunResult& b = one.runs[0];
            EXPECT_EQ(a.x, b.x) << to_string(scheme) << " Lauf " << i;
            EXPECT_EQ(a.history, b.history) << to_string(scheme) << " Lauf " << i;
            EXPECT_EQ(a.iterations, b.iterations);
            EXPECT_EQ(a.status, b.status);
        }
    }
}

TEST(ToleranceAware, RejectsInvalidSettings) {
    const Problem p(block_json());
    EXPECT_THROW(mko::ToleranceAwareModel(p.loaded.system, p.base, p.set, plus_minus(0.0)),
                 std::invalid_argument);
    EXPECT_THROW(mko::ToleranceAwareModel(p.loaded.system, p.base, p.set, plus_minus(1.5)),
                 std::invalid_argument);
    mko::ToleranceAwareSettings none = plus_minus(0.5);
    none.scheme = mko::ToleranceAwareScheme::Samples;
    none.samples = 0;
    EXPECT_THROW(mko::ToleranceAwareModel(p.loaded.system, p.base, p.set, none),
                 std::invalid_argument);
    // ohne Toleranzsatz
    json j = block_json();
    j.erase("tolerances");
    const Problem q(j);
    EXPECT_THROW(mko::ToleranceAwareModel(q.loaded.system, q.base, q.set, plus_minus(0.5)),
                 std::invalid_argument);
    // Brechzahltoleranz an einem Glas mit Glasvariablen: nicht darstellbar
    json k = block_json();
    k["variables"]["items"].push_back({{"type", "glass"}, {"surface", 2}});
    const Problem r(k);
    EXPECT_THROW(mko::ToleranceAwareModel(r.loaded.system, r.base, r.set, plus_minus(0.5)),
                 std::invalid_argument);
    EXPECT_EQ(mko::tolerance_aware_scheme_from_string("plus_minus"),
              mko::ToleranceAwareScheme::PlusMinus);
    EXPECT_EQ(mko::tolerance_aware_scheme_from_string("samples"),
              mko::ToleranceAwareScheme::Samples);
    EXPECT_FALSE(mko::tolerance_aware_scheme_from_string("x"));
}
