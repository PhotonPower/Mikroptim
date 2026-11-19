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
// Bestätigung durch die Monte-Carlo-Tolerierung (M8-T4). M11-T6b: Kippungsabzug (Dezentrierung
// der Sammellinse verschiebt nur das Bild; Teilsysteme gleich der Bewertung der Tolerierung und
// der Empfindlichkeiten; Gradient gegen FD; Zeilen des CPU-Backends), Standardsatz und Anbindung
// an die Synthese. GPU-Teil: test_cuda_tolerance.cpp.

#include "mikroptim/batch/batch_merit.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/normal_equations.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "mikroptim/synth/optical_homotopy.hpp"
#include "mikroptim/tolerance/sensitivity.hpp"
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
#include <memory>
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

// ---- M11-T6b: Kippungsabzug, Anbindung an die Synthese ----------------------------------------

namespace {

mko::ToleranceAwareSettings tilt_settings(double alpha, mko::ToleranceAwareTilt tilt) {
    mko::ToleranceAwareSettings s = plus_minus(alpha);
    s.tilt = tilt;
    return s;
}

mko::LmOptions hard_options() {
    mko::LmOptions o = lm_options();
    o.jacobian.quadrature.failure = mko::FailureMode::Hard;
    return o;
}

// Kittblock mit einem Toleranzsatz aus Arten, die Lage und Güte des Bildes ändern (Dezentrierung
// und Verkippung der Sammellinse, Radien, Brechzahl), ohne Kompensator.
json mixed_tolerances() {
    json j = block_json();
    j["tolerances"]["items"] =
        json::array({{{"type", "element_decenter"}, {"element", {4, 5}}, {"delta", 0.01}},
                     {{"type", "element_tilt"}, {"element", {4, 5}}, {"arcmin", 2.0}},
                     {{"type", "radius"}, {"surfaces", {4, 5}}, {"relative", 0.002}},
                     {{"type", "index"}, {"surface", 2}, {"delta", 0.002}}});
    return j;
}

double weighted(const mko::ToleranceAwareMerits& m, const mko::ToleranceAwareModel& model) {
    double f = 0.0;
    for (std::size_t s = 0; s < m.merit.size(); ++s) {
        f += model.weight(static_cast<int>(s)) * m.merit[s];
    }
    return f;
}

} // namespace

// Abnahme (1): Eine reine Dezentrierung der Sammellinse (Linse mit Brechkraft) verschiebt vor
// allem das Bild. Mit Kippung (keep, M11-T6) zählt die Verschiebung als Bildfehler: F(±δ) ist ein
// Vielfaches von F₀. Mit Kippungsabzug (remove) verschlechtert sie F_tol nicht: Die gestörten
// Teilsysteme bleiben bei F₀ bis auf die echte Änderung der Bildgüte (Koma der dezentrierten
// Linse), und diese ist genau Q± der Empfindlichkeiten (M10-T3).
TEST(ToleranceAware, TiltRemovalDecenterOnlyShiftsImage) {
    json j = block_json();
    j["tolerances"]["items"] =
        json::array({{{"type", "element_decenter"}, {"element", {4, 5}}, {"delta", 0.01}}});
    const Problem p(j);
    const mko::LmOptions o = hard_options();
    const std::vector<double> x = p.set.extract(p.base, 0);
    const mko::ToleranceAwareModel keep(p.loaded.system, p.base, p.set,
                                        tilt_settings(0.5, mko::ToleranceAwareTilt::Keep));
    const mko::ToleranceAwareModel remove(p.loaded.system, p.base, p.set,
                                          tilt_settings(0.5, mko::ToleranceAwareTilt::Remove));
    ASSERT_EQ(remove.samples(), 4); // Δx, Δy je ±δ
    const mko::ToleranceAwareMerits mk = parts(keep, p, o, x);
    const mko::ToleranceAwareMerits mr = parts(remove, p, o, x);
    const double f0 = mk.merit[0];
    EXPECT_LE(relative(mr.merit[0], f0), mko::test::kToleranceAwareDirect); // nominal unverändert
    double keep_max = 0.0;
    double remove_max = 0.0;
    for (std::size_t s = 1; s < mk.merit.size(); ++s) {
        EXPECT_GT(mk.merit[s], 10.0 * f0) << "keep, s = " << s;
        keep_max = std::max(keep_max, (mk.merit[s] - f0) / f0);
        remove_max = std::max(remove_max, std::abs(mr.merit[s] - f0) / f0);
    }
    EXPECT_LE(remove_max, mko::test::kToleranceAwareShiftOnly);
    const double tol_keep = weighted(mk, keep);
    const double tol_remove = weighted(mr, remove);
    EXPECT_LE(std::abs(tol_remove - f0) / f0, mko::test::kToleranceAwareShiftOnly);
    // Die verbleibende Änderung ist die der Empfindlichkeiten (gleiche Bewertung, ohne
    // Kompensator)
    mko::SensitivitySettings ss;
    ss.threads = 1;
    const mko::SensitivityResult sr = mko::run_sensitivity_analysis(p.loaded.system, p.base, ss);
    EXPECT_LE(relative(sr.q0, f0), mko::test::kToleranceAwareVsTolerancing);
    ASSERT_EQ(sr.components.size(), 2U);
    double worst = 0.0;
    for (std::size_t d = 0; d < 2; ++d) {
        const mko::SensitivityComponent& c = sr.components[d];
        worst = std::max({worst, relative(mr.merit[2 * d + 1], c.q_plus),
                          relative(mr.merit[2 * d + 2], c.q_minus)});
    }
    EXPECT_LE(worst, mko::test::kToleranceAwareVsTolerancing);
    std::printf("[ OK ] Dezentrierung 10 um der Sammellinse: F0 %.6e; keep: F(+-d) bis "
                "F0 (1 + %.3e), F_tol %.6e; remove: |F(+-d) - F0|/F0 <= %.3e, F_tol %.6e; gegen "
                "Q+- der Empfindlichkeiten %.2e\n",
                f0, keep_max, tol_keep, remove_max, tol_remove, worst);
}

// Abnahme (2): F_tol stimmt mit der Bewertung der Tolerierung überein. Mit Kippungsabzug ist
// jedes gestörte Teilsystem s das Q der Monte-Carlo-Tolerierung (M8-T4, ohne Kompensator, Schema
// samples mit demselben Startwert: Stichprobe s − 1) bzw. Q± der Empfindlichkeiten (M10-T3,
// Schema plus_minus). Hartes Versagen wie die Bewertung der Tolerierung.
TEST(ToleranceAware, TiltRemovalMatchesTolerancing) {
    const Problem p(mixed_tolerances());
    const mko::LmOptions o = hard_options();
    const std::vector<double> x = p.set.extract(p.base, 0);
    // (a) Monte Carlo
    constexpr int kSamples = 20;
    constexpr std::uint64_t kSeed = 2026;
    mko::ToleranceAwareSettings ts = tilt_settings(0.5, mko::ToleranceAwareTilt::Remove);
    ts.scheme = mko::ToleranceAwareScheme::Samples;
    ts.samples = kSamples;
    ts.seed = kSeed;
    const mko::ToleranceAwareModel model(p.loaded.system, p.base, p.set, ts);
    const mko::ToleranceAwareMerits m = parts(model, p, o, x);
    mko::ToleranceSettings mc;
    mc.samples = kSamples;
    mc.seed = kSeed;
    mc.threads = 1;
    mc.compensator = mko::CompensatorType::None;
    const mko::ToleranceResult tr = mko::run_tolerance_analysis(p.loaded.system, p.base, mc);
    ASSERT_EQ(tr.results.size(), static_cast<std::size_t>(kSamples));
    double worst_mc = 0.0;
    double sum_q = 0.0;
    for (int s = 1; s <= kSamples; ++s) {
        const double rms = tr.results[static_cast<std::size_t>(s - 1)].rms_composite;
        sum_q += rms * rms;
        worst_mc = std::max(worst_mc, relative(m.merit[static_cast<std::size_t>(s)], rms * rms));
    }
    EXPECT_LE(worst_mc, mko::test::kToleranceAwareVsTolerancing);
    // ohne Kippungsabzug nicht (Kontrolle)
    ts.tilt = mko::ToleranceAwareTilt::Keep;
    const mko::ToleranceAwareModel keep(p.loaded.system, p.base, p.set, ts);
    const mko::ToleranceAwareMerits mk = parts(keep, p, o, x);
    double sum_keep = 0.0;
    for (int s = 1; s <= kSamples; ++s) {
        sum_keep += mk.merit[static_cast<std::size_t>(s)];
    }
    EXPECT_GT(sum_keep, 2.0 * sum_q);
    // (b) Empfindlichkeiten
    const mko::ToleranceAwareModel pm(p.loaded.system, p.base, p.set,
                                      tilt_settings(0.5, mko::ToleranceAwareTilt::Remove));
    const mko::ToleranceAwareMerits mp = parts(pm, p, o, x);
    mko::SensitivitySettings ss;
    ss.threads = 1;
    const mko::SensitivityResult sr = mko::run_sensitivity_analysis(p.loaded.system, p.base, ss);
    ASSERT_EQ(2 * sr.components.size(), static_cast<std::size_t>(pm.samples()));
    double worst_sens = 0.0;
    for (std::size_t d = 0; d < sr.components.size(); ++d) {
        const mko::SensitivityComponent& c = sr.components[d];
        worst_sens = std::max({worst_sens, relative(mp.merit[2 * d + 1], c.q_plus),
                               relative(mp.merit[2 * d + 2], c.q_minus)});
    }
    EXPECT_LE(worst_sens, mko::test::kToleranceAwareVsTolerancing);
    std::printf("[ OK ] Teilsysteme gegen Monte Carlo (%d Stichproben, ohne Kompensator) %.2e, "
                "gegen Q+- der Empfindlichkeiten (%zu Komponenten) %.2e relativ; Mittel F "
                "gestört remove %.6e, keep %.6e\n",
                kSamples, worst_mc, sr.components.size(), worst_sens, sum_q / kSamples,
                sum_keep / kSamples);
}

// Jacobi-Matrix mit Kippungsabzug: der Gradient 2 Σ w_s D_s (P J_s)ᵀ(P r_s) gegen die zentrale
// Differenz von F_tol (Störungen von Variablen wie in ChainRuleForPerturbedVariables, dazu die
// Dezentrierung), beide Schemata, FD- und adjungierte Jacobi-Matrix.
TEST(ToleranceAware, TiltRemovalGradientMatchesFiniteDifferences) {
    json j = block_json();
    j["variables"]["items"] =
        json::array({{{"type", "curvature"}, {"surfaces", {2, 4, 5}}},
                     {{"type", "thickness"}, {"surface", 3}},
                     {{"type", "working_distance"}, {"min", 10.0}, {"max", 30.0}}});
    j["tolerances"]["items"] =
        json::array({{{"type", "radius"}, {"surfaces", {4, 5}}, {"relative", 0.05}},
                     {{"type", "thickness"}, {"surfaces", {3, 4}}, {"delta", 0.2}},
                     {{"type", "element_decenter"}, {"element", {4, 5}}, {"delta", 0.05}},
                     {{"type", "element_tilt"}, {"element", {4, 5}}, {"arcmin", 20.0}}});
    const Problem p(j);
    double max_gradient = 0.0;
    for (const mko::JacobianMethod method :
         {mko::JacobianMethod::FiniteDifferences, mko::JacobianMethod::Adjoint}) {
        mko::LmOptions o = hard_options();
        o.jacobian_method = method;
        for (const mko::ToleranceAwareScheme scheme :
             {mko::ToleranceAwareScheme::PlusMinus, mko::ToleranceAwareScheme::Samples}) {
            mko::ToleranceAwareSettings ts = tilt_settings(0.7, mko::ToleranceAwareTilt::Remove);
            ts.scheme = scheme;
            ts.samples = 5;
            ts.seed = 11;
            const mko::ToleranceAwareModel model(p.loaded.system, p.base, p.set, ts);
            const std::vector<double> x = p.set.extract(p.base, 0);
            const auto P = static_cast<std::size_t>(p.set.size());
            mko::ToleranceAwareBackend backend(model, mko::cpu_tolerance_inner(), p.base, o);
            const int run0 = 0;
            const std::span<const int> run(&run0, 1);
            std::vector<double> a(P * P);
            std::vector<double> g(P);
            backend.normal_equations(run, x, a, g);
            double scale = 0.0;
            for (std::size_t l = 0; l < P; ++l) {
                scale = std::max(scale,
                                 std::abs(2.0 * g[l]) *
                                     std::max(std::abs(x[l]), p.set[static_cast<int>(l)].typical));
            }
            for (std::size_t k = 0; k < P; ++k) {
                const double s_k = std::max(std::abs(x[k]), p.set[static_cast<int>(k)].typical);
                const double h = 1e-5 * s_k;
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
                max_gradient = std::max(max_gradient, std::abs(2.0 * g[k] - fd_grad) * s_k / scale);
                EXPECT_LE(std::abs(2.0 * g[k] - fd_grad) * s_k,
                          mko::test::kToleranceAwareGradient * scale)
                    << mko::to_string(method) << ' ' << to_string(scheme) << " Variable "
                    << p.set[static_cast<int>(k)].name << ": 2g = " << 2.0 * g[k] << ", FD "
                    << fd_grad;
            }
        }
    }
    std::printf("[ OK ] Kippungsabzug: Gradient gegen FD von F_tol %.2e (FD und adjungiert)\n",
                max_gradient);
}

// Zeilen des CPU-Backends auf dem Host: A und g aus host_jacobians bitgleich zu
// normal_equations (FD und adjungiert), ‖r‖² aus host_residuals gleich der Merit, Status am
// Punkt gleich in beiden.
TEST(ToleranceAware, CpuBackendHostRows) {
    const Problem p(mixed_tolerances());
    for (const mko::JacobianMethod method :
         {mko::JacobianMethod::FiniteDifferences, mko::JacobianMethod::Adjoint}) {
        mko::LmOptions o = lm_options();
        o.jacobian_method = method;
        const mko::SystemBatchHost base = mko::make_non_coaxial(p.base);
        const mko::VariableSet set(p.loaded.system, base.topology());
        mko::CpuLmBackend backend(set, base, o);
        const std::vector<double> x = set.extract(base, 0);
        std::vector<double> xs = x;
        xs.insert(xs.end(), x.begin(), x.end());
        xs[static_cast<std::size_t>(set.size())] += 0.01;
        const std::vector<int> runs{0, 1};
        const auto P = static_cast<std::size_t>(set.size());
        std::vector<double> a(2 * P * P);
        std::vector<double> g(2 * P);
        backend.normal_equations(runs, xs, a, g);
        mko::LmHostRows rows;
        backend.host_jacobians(runs, xs, rows);
        ASSERT_EQ(rows.shape.runs, 2);
        const auto R = static_cast<std::size_t>(rows.shape.rows());
        for (std::size_t i = 0; i < 2; ++i) {
            std::vector<double> ai(P * P);
            std::vector<double> gi(P);
            mko::normal_equations_host(
                std::span<const double>(rows.jacobian).subspan(i * P * R, P * R),
                std::span<const double>(rows.residuals).subspan(i * R, R), static_cast<int>(P), ai,
                gi);
            for (std::size_t k = 0; k < P * P; ++k) {
                EXPECT_EQ(ai[k], a[i * P * P + k]);
            }
            for (std::size_t k = 0; k < P; ++k) {
                EXPECT_EQ(gi[k], g[i * P + k]);
            }
        }
        std::vector<double> f(2);
        std::vector<std::int64_t> nf(2);
        std::vector<std::int64_t> nh(2);
        backend.merit(runs, xs, f, nf, nh);
        mko::LmHostRows res;
        backend.host_residuals(runs, xs, res);
        ASSERT_EQ(res.residuals.size(), 2 * R);
        for (std::size_t i = 0; i < 2; ++i) {
            double sum = 0.0;
            double sum_j = 0.0;
            for (std::size_t k = 0; k < R; ++k) {
                sum += res.residuals[i * R + k] * res.residuals[i * R + k];
                sum_j += rows.residuals[i * R + k] * rows.residuals[i * R + k];
            }
            EXPECT_LE(relative(sum, f[i]), mko::test::kToleranceAwareDirect);
            EXPECT_LE(relative(sum_j, f[i]), mko::test::kToleranceAwareDirect);
            const auto n = static_cast<std::size_t>(res.rays_per_variant);
            EXPECT_TRUE(std::equal(res.status.begin() + static_cast<std::ptrdiff_t>(i * n),
                                   res.status.begin() + static_cast<std::ptrdiff_t>((i + 1) * n),
                                   rows.status.begin() + static_cast<std::ptrdiff_t>(i * n)));
        }
    }
}

// Mit Kippungsabzug: gebündelte Läufe bitgleich zu einzelnen; die Optimierung senkt F_tol.
TEST(ToleranceAware, TiltRemovalJointRunsBitwiseEqualToSingleRuns) {
    const Problem p(mixed_tolerances());
    mko::LmOptions o = hard_options();
    o.max_iterations = 10;
    o.threads = 0;
    const std::vector<double> x0 = p.set.extract(p.base, 0);
    std::vector<double> starts;
    for (const double c : {0.05, -0.03}) {
        starts.push_back(c);
        starts.push_back(x0[1] + c);
    }
    const mko::ToleranceAwareModel model(p.loaded.system, p.base, p.set,
                                         tilt_settings(0.5, mko::ToleranceAwareTilt::Remove));
    const mko::LmResult joint = mko::optimize_lm_tolerance_aware(model, mko::cpu_tolerance_inner(),
                                                                 p.set, p.base, starts, o);
    for (std::size_t i = 0; i < 2; ++i) {
        const mko::LmResult one =
            mko::optimize_lm_tolerance_aware(model, mko::cpu_tolerance_inner(), p.set, p.base,
                                             std::span<const double>(starts).subspan(2 * i, 2), o);
        EXPECT_EQ(joint.runs[i].x, one.runs[0].x) << "Lauf " << i;
        EXPECT_EQ(joint.runs[i].history, one.runs[0].history) << "Lauf " << i;
        EXPECT_EQ(joint.runs[i].status, one.runs[0].status);
        EXPECT_LT(one.runs[0].merit, one.runs[0].history.front());
    }
    EXPECT_EQ(mko::tolerance_aware_tilt_from_string("keep"), mko::ToleranceAwareTilt::Keep);
    EXPECT_EQ(mko::tolerance_aware_tilt_from_string("remove"), mko::ToleranceAwareTilt::Remove);
    EXPECT_FALSE(mko::tolerance_aware_tilt_from_string("x"));
    EXPECT_STREQ(mko::to_string(mko::ToleranceAwareTilt::Remove), "remove");
}

// Standardsatz der Synthese: ohne Brechzahltoleranz an Gläsern mit Glasvariablen (sonst nicht
// darstellbar), alle übrigen Einträge wie default_sensitivity_tolerances.
TEST(ToleranceAware, StandardSetSkipsGlassVariables) {
    json k = block_json();
    k.erase("tolerances");
    k["variables"]["items"].push_back({{"type", "glass"}, {"surface", 2}});
    const Problem r(k);
    const mko::Tolerances full = mko::default_sensitivity_tolerances(r.loaded.system);
    const mko::Tolerances t = mko::tolerance_aware_standard_set(r.loaded.system, r.set);
    ASSERT_EQ(t.items.size(), full.items.size());
    for (std::size_t i = 0; i < t.items.size(); ++i) {
        if (t.items[i].type == mko::ToleranceType::Index) {
            EXPECT_EQ(std::count(t.items[i].surfaces.begin(), t.items[i].surfaces.end(), 2), 0);
            EXPECT_EQ(t.items[i].surfaces.size() + 1, full.items[i].surfaces.size());
        } else {
            EXPECT_EQ(t.items[i], full.items[i]);
        }
    }
    mko::SystemDescription with = r.loaded.system;
    with.tolerances = t;
    EXPECT_NO_THROW(mko::ToleranceAwareModel(with, r.base, r.set, plus_minus(0.5)));
}

// Anbindung an die Synthese: Mit system_optimizer ist die Merit eines Schritts F_tol am
// realisierten Punkt (Standardsatz der Beschreibung), also auch die Rangfolge der Strahlsuche
// (beam_rank_before ordnet nach HomotopySolve::merit); die Bewertung F_B bleibt nominal.
TEST(ToleranceAware, SynthStepUsesExpectedMerit) {
    const mko::LoadedSystem start = mko::load_system(
        std::string(MKO_SOURCE_DIR) + "/data/systems/einzellinse_homotopie_start.json");
    const mko::ToleranceAwareSettings ts = tilt_settings(0.5, mko::ToleranceAwareTilt::Remove);
    mko::OpticalHomotopyOptions o;
    o.lm.max_iterations = 0;
    o.lm.threads = 1;
    o.lm.jacobian.extra_threads = 1;
    o.saddle_delta = 0.0;
    struct Seen {
        mko::SystemDescription realized;
        mko::SystemBatchHost base;
        std::vector<double> x0;
        mko::LmOptions options;
    };
    const auto seen = std::make_shared<Seen>();
    o.system_optimizer = [ts, seen](const mko::SystemDescription& realized,
                                    const mko::VariableSet& set, const mko::SystemBatchHost& base,
                                    std::span<const double> x0, const mko::LmOptions& lm) {
        seen->realized = realized;
        seen->base = base;
        seen->x0.assign(x0.begin(), x0.end());
        seen->options = lm;
        return mko::optimize_lm_tolerance_aware_standard(realized, ts, mko::cpu_tolerance_inner(),
                                                         set, base, x0, lm);
    };
    mko::OpticalHomotopyModel m(start, o);
    mko::SystemDescription s = m.start_system();
    const mko::HomotopySolve solve = m.solve(s, m.start_point());
    ASSERT_FALSE(seen->x0.empty());
    // F_tol am Startpunkt aus einem eigenen Backend
    const mko::VariableSet set(seen->realized, seen->base.topology());
    mko::SystemDescription with = seen->realized;
    with.tolerances = mko::tolerance_aware_standard_set(seen->realized, set);
    const mko::ToleranceAwareModel model(with, seen->base, set, ts);
    mko::ToleranceAwareBackend backend(model, mko::cpu_tolerance_inner(), seen->base,
                                       seen->options);
    const int run0 = 0;
    std::vector<double> f(1);
    std::vector<std::int64_t> n1(1);
    std::vector<std::int64_t> n2(1);
    backend.merit(std::span<const int>(&run0, 1), seen->x0, f, n1, n2);
    EXPECT_EQ(solve.merit, f[0]);
    EXPECT_GT(model.samples(), 10);
    // nominal bleibt die Bewertung F_B: kleiner als F_tol (gestörte Systeme sind schlechter)
    EXPECT_LT(solve.wavefront + solve.ghost_penalty, solve.merit);
    std::printf("[ OK ] Synthese-Schritt: F_tol %.6e (S = %d), F_B nominal %.6e\n", solve.merit,
                model.samples(), solve.wavefront + solve.ghost_penalty);
}
