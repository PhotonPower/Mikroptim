#pragma once

// Abnahme der L1-Strafe der Geisterkoeffizienten und des Diagnoseberichts (M4-T6, CPU und GPU):
// Testsysteme mit Geisterplatten-Wörterbuch, Zusatzresiduen (paraxiale Nebenbedingungen plus
// Strafe) und Hilfen für die Auswertung (docs/optimierung.md, Abschnitte 3 und 5).

#include "fd_test_systems.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/optim/ghost_diagnosis.hpp"
#include "mikroptim/optim/ghost_penalty.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mko::test {

// Exaktheit: einzellinse_geisterplatten_exakt.json (Biegung der Einzellinse, f = 100 mm,
// NA 0,025; vier Geisterplatten mit je K = 4 und Startkoeffizienten ≠ 0; μ = 100).
inline FdProblem ghost_exact_problem() {
    return make_fd_problem(load_residual_system("einzellinse_geisterplatten_exakt.json"));
}

// Bedarf: einzellinse_na05_geisterplatten.json (Einzellinse bei NA 0,5, Achse, eine
// Wellenlänge; Wörterbuch aus drei Lagen mit je drei Gläsern; μ aus dem Beugungsziel).
inline FdProblem ghost_need_problem() {
    return make_fd_problem(load_residual_system("einzellinse_na05_geisterplatten.json"));
}

// Strafe aus der Datei, optional mit anderem μ (Gewicht aller ghost_penalty-Einträge) und ε.
inline GhostPenalty ghost_penalty_of(const SystemDescription& system, double mu = -1.0,
                                     double smoothing = -1.0) {
    SystemDescription d = system;
    for (ConstraintSpec& c : d.constraints.value().items) {
        if (c.type == ConstraintType::GhostPenalty) {
            if (mu >= 0.0) {
                c.weight = mu;
            }
            if (smoothing > 0.0) {
                c.smoothing = smoothing;
            }
        }
    }
    return make_ghost_penalty(d);
}

// Zusatzresiduen wie in mikroptim optimize: paraxiale Nebenbedingungen, dann die Strafe.
inline ExtraResiduals ghost_extra(const SystemDescription& system, const GhostPenalty& penalty) {
    return concat_extra_residuals({make_paraxial_residuals(make_paraxial_constraints(system)),
                                   make_ghost_penalty_residuals(penalty)});
}

// Nur die paraxialen Nebenbedingungen (Merit des realen Systems ohne Strafe).
inline ExtraResiduals paraxial_only(const SystemDescription& system) {
    return make_paraxial_residuals(make_paraxial_constraints(system));
}

// Indizes der Geisterkoeffizienten im Variablenvektor.
inline std::vector<int> ghost_variables(const VariableSet& set) {
    std::vector<int> out;
    for (int k = 0; k < set.size(); ++k) {
        if (set[k].type == VariableType::GhostCoefficient) {
            out.push_back(k);
        }
    }
    return out;
}

// Variablenvektor mit allen Geisterkoeffizienten = 0.
inline std::vector<double> without_ghosts(const VariableSet& set, std::vector<double> x) {
    for (const int k : ghost_variables(set)) {
        x[static_cast<std::size_t>(k)] = 0.0;
    }
    return x;
}

// max |a| über alle Geisterkoeffizienten von x.
inline double max_ghost(const VariableSet& set, std::span<const double> x) {
    double m = 0.0;
    for (const int k : ghost_variables(set)) {
        m = std::max(m, std::abs(x[static_cast<std::size_t>(k)]));
    }
    return m;
}

// Gradient g_k = ∂F_rest/∂a_k der Merit ohne Strafe (Wellenfront + paraxial) am Punkt x, aus
// der FD-Jacobi: g = 2 Jᵀ r über alle Zeilen.
inline std::vector<double> rest_gradient(const FdProblem& p, std::span<const double> x,
                                         const ResidualSettings& quadrature = {}) {
    FdJacobianOptions o;
    o.quadrature = quadrature;
    o.extra = paraxial_only(p.loaded.system);
    const FdJacobian j = fd_jacobian_cpu(p.set, p.base, x, o);
    const std::span<const double> r = j.residual(0);
    std::vector<double> g(static_cast<std::size_t>(p.set.size()), 0.0);
    for (int k = 0; k < p.set.size(); ++k) {
        const std::span<const double> col = j.column(0, k);
        double s = 0.0;
        for (std::size_t i = 0; i < r.size(); ++i) {
            s += col[i] * r[i];
        }
        g[static_cast<std::size_t>(k)] = 2.0 * s;
    }
    return g;
}

// Strafterm des Geisterkoeffizienten (Platte, Ordnung); wirft, wenn er fehlt.
inline const GhostPenaltyTerm& penalty_term(const GhostPenalty& pen, int plate, int order) {
    for (const GhostPenaltyTerm& t : pen.terms) {
        if (t.plate == plate && t.order == order) {
            return t;
        }
    }
    throw std::logic_error("Geisterkoeffizient ohne Strafterm");
}

// Systembeschreibung ohne Geistervariablen (Eintrag ghost_coefficient entfernt) und mit allen
// Koeffizienten 0: das reale System allein.
inline FdProblem plain_problem(const FdProblem& p) {
    SystemDescription d = p.loaded.system;
    std::vector<VariableSpec>& items = d.variables.value().items;
    items.erase(std::remove_if(
                    items.begin(), items.end(),
                    [](const VariableSpec& s) { return s.type == VariableType::GhostCoefficient; }),
                items.end());
    for (GhostPlate& g : d.ghost_plates) {
        std::fill(g.coefficients.begin(), g.coefficients.end(), 0.0);
    }
    return make_fd_problem(prepare_system(d, std::filesystem::path(kResidualSystems)));
}

// RMS der Wellenfront [Wellen] am Punkt x (ein Feld, eine Wellenlänge: √F_Wellenfront).
inline double wavefront_rms(const FdProblem& p, std::span<const double> x) {
    const int run0 = 0;
    const SystemBatchHost b = lm_point_batch(p.set, p.base, std::span<const int>(&run0, 1), x);
    return std::sqrt(make_cpu_merit({}, {})(b)[0]);
}

struct ExactnessResult {
    LmRunResult run;                     // mit Geisterplatten und Strafe
    std::vector<double> rest;            // g_k = ∂F_rest/∂x_k am Ende (ohne Strafe)
    double threshold_ratio = 0.0;        // max_k |g_k|/(μ w_k) über die Geisterkoeffizienten
    double max_stationarity_error = 0.0; // max_k |a_k − a_k,Vorhersage| / ε
    std::vector<double> predicted;       // a_k,Vorhersage = −ε s/√(1 − s²) (0 für andere Variablen)
    double rest_merit = 0.0;             // F ohne Strafe am Ende
    double plain_merit = 0.0;            // F des realen Systems allein (LM ohne Geistervariablen)
    // Gewinn der geglätteten Strafe gegenüber dem realen System allein, F_allein − F, und seine
    // Vorhersage Σ_k μ w_k ε (1 − √(1 − s_k²)) ≈ Σ_k ε g_k²/(2 μ w_k) (Minimum von
    // g a + μ w (√(a² + ε²) − ε) über a).
    double smoothing_gain = 0.0;
    double predicted_gain = 0.0;
    LmRunResult plain;
};

// Lauf mit Strafe `pen` (Optimierer: LmRunResult optimize(const LmOptions&) vom Start der
// Datei), Auswertung der Stationarität der geglätteten Strafe und Vergleich mit dem realen
// System allein (CPU).
template <class Optimize>
ExactnessResult ghost_exactness(const FdProblem& p, const GhostPenalty& pen, Optimize&& optimize) {
    ExactnessResult e;
    LmOptions o;
    o.max_iterations = 1000;
    o.jacobian.extra = ghost_extra(p.loaded.system, pen);
    e.run = optimize(o);
    e.rest = rest_gradient(p, e.run.x);
    e.predicted.assign(e.rest.size(), 0.0);
    for (const int k : ghost_variables(p.set)) {
        const auto kk = static_cast<std::size_t>(k);
        const GhostPenaltyTerm& t = penalty_term(pen, p.set[k].ghost_plate, p.set[k].ghost_order);
        // Stationarität von F_rest + μ w (√(a² + ε²) − ε): g + μ w a/√(a² + ε²) = 0
        const double s = e.rest[kk] / t.weight;
        e.threshold_ratio = std::max(e.threshold_ratio, std::abs(s));
        e.predicted_gain += t.weight * t.smoothing * (1.0 - std::sqrt(std::max(0.0, 1.0 - s * s)));
        const double pred = std::abs(s) < 1.0 ? -t.smoothing * s / std::sqrt(1.0 - s * s)
                                              : std::numeric_limits<double>::infinity();
        e.predicted[kk] = pred;
        e.max_stationarity_error =
            std::max(e.max_stationarity_error, std::abs(e.run.x[kk] - pred) / t.smoothing);
    }
    const int run0 = 0;
    const SystemBatchHost b =
        lm_point_batch(p.set, p.base, std::span<const int>(&run0, 1), e.run.x);
    e.rest_merit = make_cpu_merit({}, paraxial_only(p.loaded.system))(b)[0];

    const FdProblem q = plain_problem(p);
    LmOptions po;
    po.max_iterations = 1000;
    po.jacobian.extra = paraxial_only(q.loaded.system);
    e.plain = optimize_lm_cpu(q.set, q.base, q.x0, po).runs.at(0);
    e.plain_merit = e.plain.merit;
    e.smoothing_gain = e.plain_merit - e.run.merit;
    return e;
}

inline void print_exactness(const char* tag, const FdProblem& p, const ExactnessResult& e) {
    std::printf("[%s] mit Geisterplatten: F %.6e -> %.10e (ohne Strafe %.10e), %d Zyklen, %s\n",
                tag, e.run.history.front(), e.run.merit, e.rest_merit, e.run.iterations,
                to_string(e.run.status));
    std::printf("[%s] reales System allein: F -> %.10e, %d Zyklen; F_allein − F = %.4e "
                "(Vorhersage Σ ε g²/(2 μ w) = %.4e)\n",
                tag, e.plain_merit, e.plain.iterations, e.smoothing_gain, e.predicted_gain);
    std::printf("[%s] max|a| = %.3e, max|g|/(μ w) = %.3e, Stationarität %.2e\n", tag,
                max_ghost(p.set, e.run.x), e.threshold_ratio, e.max_stationarity_error);
    for (const int k : ghost_variables(p.set)) {
        const auto kk = static_cast<std::size_t>(k);
        std::printf("    %-8s a = %+.4e  Vorhersage %+.4e  g = %+.4e\n", p.set[k].name.c_str(),
                    e.run.x[kk], e.predicted[kk], e.rest[kk]);
    }
}

struct NeedResult {
    LmRunResult plain; // reales System allein
    LmRunResult run;   // mit Geisterplatten
    double rms_plain = 0.0;
    double rms_ghost = 0.0;
    GhostPenalty penalty;
    GhostDiagnosis diagnosis;
    std::string text;
    std::string json;
};

// Optimierer: LmRunResult optimize(const FdProblem&, const LmOptions&) vom Start der Datei.
template <class Optimize>
NeedResult ghost_need(const FdProblem& p, Optimize&& optimize) {
    NeedResult n;
    const FdProblem q = plain_problem(p);
    LmOptions po;
    po.max_iterations = 1000;
    po.jacobian.extra = paraxial_only(q.loaded.system);
    n.plain = optimize(q, po);
    n.rms_plain = wavefront_rms(q, n.plain.x);

    n.penalty = ghost_penalty_of(p.loaded.system);
    LmOptions o;
    o.max_iterations = 1000;
    o.jacobian.extra = ghost_extra(p.loaded.system, n.penalty);
    n.run = optimize(p, o);
    n.rms_ghost = wavefront_rms(p, n.run.x);

    const int run0 = 0;
    const SystemBatchHost b =
        lm_point_batch(p.set, p.base, std::span<const int>(&run0, 1), n.run.x);
    n.diagnosis = diagnose_ghost_plates(p.loaded.system, p.loaded.catalog, b, 0, n.penalty,
                                        make_cpu_merit({}, o.jacobian.extra));
    std::ostringstream os;
    write_ghost_diagnosis_text(n.diagnosis, os);
    n.text = os.str();
    n.json = ghost_diagnosis_json(n.diagnosis);
    return n;
}

inline void print_need(const char* tag, const FdProblem& p, const NeedResult& n) {
    std::printf("[%s] reales System allein: F %.6e -> %.10e, %d Zyklen, %s, RMS %.4e Wellen\n", tag,
                n.plain.history.front(), n.plain.merit, n.plain.iterations,
                to_string(n.plain.status), n.rms_plain);
    std::printf("[%s] mit Geisterplatten: F %.6e -> %.10e, %d Zyklen, %s, RMS %.4e Wellen "
                "(λ/14 = %.4e), max|a| = %.3e mm\n",
                tag, n.run.history.front(), n.run.merit, n.run.iterations, to_string(n.run.status),
                n.rms_ghost, 1.0 / 14.0, max_ghost(p.set, n.run.x));
    std::printf("%s", n.text.c_str());
}

// Index der aktivsten Platte (größte Σ|a|).
inline int most_active_plate(const GhostDiagnosis& d) {
    int best = 0;
    for (const GhostPlateReport& r : d.plates) {
        if (r.sum_abs > d.plates[static_cast<std::size_t>(best)].sum_abs) {
            best = r.plate;
        }
    }
    return best;
}

// Prüfungen der Abnahme „Bedarf“ (CPU und GPU).
inline void check_need(const FdProblem& p, const NeedResult& n) {
    EXPECT_TRUE(lm_converged(n.plain.status)) << to_string(n.plain.status);
    EXPECT_TRUE(lm_converged(n.run.status)) << to_string(n.run.status);
    EXPECT_EQ(n.plain.failed_rays, 0);
    EXPECT_EQ(n.run.failed_rays, 0);
    // Bedarf: das reale System allein bleibt weit über der Beugungsgrenze, die Geisterplatten
    // senken den Fehler um mehr als den Faktor 10.
    EXPECT_GT(n.rms_plain, 10.0 / 14.0);
    EXPECT_LT(n.rms_ghost, 0.1 * n.rms_plain);

    const GhostDiagnosis& d = n.diagnosis;
    ASSERT_EQ(d.plates.size(), p.loaded.system.ghost_plates.size());
    const int b = most_active_plate(d);
    const GhostPlateReport& r = d.plates[static_cast<std::size_t>(b)];
    const GhostPlate& gp = p.loaded.system.ghost_plates[static_cast<std::size_t>(b)];
    // Koeffizienten aktiv, dominante Ordnung 2 (sphärische Aberration ∝ ρ⁴ − ρ² ∝ P_2 nach
    // Fokussierung über den Arbeitsabstand)
    EXPECT_TRUE(r.active);
    ASSERT_FALSE(r.dominant_orders.empty());
    EXPECT_EQ(r.dominant_orders.front(), 2);
    EXPECT_GT(r.max_abs, 100.0 * r.activity_threshold);
    // Lage: Fläche und Abstand der Platte aus der Datei, Lage der Empfehlung
    EXPECT_EQ(r.after_surface, gp.after_surface);
    EXPECT_EQ(r.distance, gp.distance);
    ASSERT_GE(d.recommended_position, 0);
    const GhostPositionReport& pos = d.positions[static_cast<std::size_t>(d.recommended_position)];
    EXPECT_EQ(pos.after_surface, r.after_surface);
    EXPECT_EQ(pos.distance, r.distance);
    EXPECT_NE(std::find(pos.active_plates.begin(), pos.active_plates.end(), b),
              pos.active_plates.end());
    // Glas: die L1-Strafe wählt bei monochromatischer Korrektur das Glas mit dem größten
    // n − 1 der Lage (kleinstes |a| für dieselbe Phase) – hier das Flint.
    for (const int g : pos.plates) {
        EXPECT_LE(d.plates[static_cast<std::size_t>(g)].glass.n_ref, r.glass.n_ref) << g;
    }
    EXPECT_EQ(r.glass.name, "buchdahl(n0=1.76, nu=-0.18/0.112)");
    // Klassifikation: eine Wellenlänge → Operator A
    EXPECT_EQ(pos.recommendation, GhostOperator::A);
    EXPECT_EQ(d.recommendation, GhostOperator::A);
    // Beitrag zur Merit: die Platte senkt F netto (Gewinn größer als ihre Strafe)
    EXPECT_GT(r.merit_gain, r.penalty);
    EXPECT_GT(r.penalty, 0.0);
    EXPECT_NEAR(r.penalty,
                ghost_penalty_merit(
                    n.penalty, lm_point_batch(p.set, p.base, std::vector<int>{0}, n.run.x), 0, b),
                1e-15 * r.penalty);
    // Text und JSON nennen Lage, Ordnung und Glas
    EXPECT_NE(n.text.find("Operator A"), std::string::npos);
    EXPECT_NE(n.text.find(r.glass.name), std::string::npos);
    const nlohmann::json j = nlohmann::json::parse(n.json);
    const nlohmann::json& jp = j.at("plates").at(static_cast<std::size_t>(b));
    EXPECT_EQ(jp.at("after_surface").get<int>(), gp.after_surface);
    EXPECT_EQ(jp.at("distance").get<double>(), gp.distance);
    EXPECT_EQ(jp.at("dominant_orders").at(0).get<int>(), 2);
    EXPECT_EQ(jp.at("glass").at("name").get<std::string>(), r.glass.name);
    EXPECT_TRUE(jp.at("active").get<bool>());
    EXPECT_EQ(j.at("recommendation").get<std::string>(), "A");
}

} // namespace mko::test
