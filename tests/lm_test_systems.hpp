#pragma once

// Testprobleme für den gebündelten Levenberg-Marquardt (M3-T4, CPU und GPU): analytische
// Residuenfunktionen als Backend und die Einzellinse für die Abnahme.

#include "fd_test_systems.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/normal_equations.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mko::test {

// Residuenfunktion r(x) ∈ ℝ^m mit analytischer Jacobi-Matrix (m × P spaltenweise; `jac` ist leer,
// wenn nur r gebraucht wird) und optionaler Zahl „versagter Strahlen“ an x, davon hart versagt
// (ohne Funktion: alle versagten).
struct DenseProblem {
    int rows = 0;
    int variables = 0;
    std::function<void(std::span<const double> x, std::span<double> r, std::span<double> jac)>
        evaluate;
    std::function<std::int64_t(std::span<const double> x)> failed;
    std::function<std::int64_t(std::span<const double> x)> hard_failed;
};

// Backend aus einer DenseProblem: A und g mit normal_equations_host, F = Σ r² aufsteigend.
class DenseBackend final : public LmBackend {
public:
    explicit DenseBackend(DenseProblem problem) : problem_(std::move(problem)) {}

    void normal_equations(std::span<const int> runs, std::span<const double> x, std::span<double> a,
                          std::span<double> g) override {
        const auto P = static_cast<std::size_t>(problem_.variables);
        const auto m = static_cast<std::size_t>(problem_.rows);
        std::vector<double> r(m);
        std::vector<double> j(m * P);
        for (std::size_t i = 0; i < runs.size(); ++i) {
            problem_.evaluate(x.subspan(i * P, P), r, j);
            normal_equations_host(j, r, problem_.variables, a.subspan(i * P * P, P * P),
                                  g.subspan(i * P, P));
            ++jacobian_points;
        }
    }

    void merit(std::span<const int> runs, std::span<const double> x, std::span<double> merit,
               std::span<std::int64_t> failed, std::span<std::int64_t> hard_failed) override {
        const auto P = static_cast<std::size_t>(problem_.variables);
        std::vector<double> r(static_cast<std::size_t>(problem_.rows));
        for (std::size_t i = 0; i < runs.size(); ++i) {
            const auto xi = x.subspan(i * P, P);
            problem_.evaluate(xi, r, {});
            double f = 0.0;
            for (const double v : r) {
                f += v * v;
            }
            merit[i] = f;
            failed[i] = problem_.failed ? problem_.failed(xi) : 0;
            hard_failed[i] = problem_.hard_failed ? problem_.hard_failed(xi) : failed[i];
            ++merit_points;
        }
    }

    int jacobian_points = 0;
    int merit_points = 0;

private:
    DenseProblem problem_;
};

// Lineares Problem r = M x − b (M: m × P spaltenweise).
inline DenseProblem linear_problem(std::vector<double> m, std::vector<double> b, int variables) {
    DenseProblem p;
    p.rows = static_cast<int>(b.size());
    p.variables = variables;
    p.evaluate = [m = std::move(m), b = std::move(b), variables](
                     std::span<const double> x, std::span<double> r, std::span<double> jac) {
        const std::size_t rows = b.size();
        for (std::size_t i = 0; i < rows; ++i) {
            double s = -b[i];
            for (std::size_t k = 0; k < static_cast<std::size_t>(variables); ++k) {
                s += m[k * rows + i] * x[k];
            }
            r[i] = s;
        }
        if (!jac.empty()) {
            std::copy(m.begin(), m.end(), jac.begin());
        }
    };
    return p;
}

// Rosenbrock als Kleinste-Quadrate-Problem: r = (10 (x₂ − x₁²), 1 − x₁), Minimum (1, 1), F = 0.
inline DenseProblem rosenbrock_problem() {
    DenseProblem p;
    p.rows = 2;
    p.variables = 2;
    p.evaluate = [](std::span<const double> x, std::span<double> r, std::span<double> jac) {
        r[0] = 10.0 * (x[1] - x[0] * x[0]);
        r[1] = 1.0 - x[0];
        if (!jac.empty()) {
            jac[0] = -20.0 * x[0]; // ∂r₀/∂x₁
            jac[1] = -1.0;         // ∂r₁/∂x₁
            jac[2] = 10.0;         // ∂r₀/∂x₂
            jac[3] = 0.0;          // ∂r₁/∂x₂
        }
    };
    return p;
}

// Schranken ohne Grenzen mit x_typ = 1.
struct DenseBounds {
    std::vector<double> lower, upper, typical;
    LmBounds view() const { return LmBounds{lower, upper, typical}; }
};

inline DenseBounds open_bounds(int variables) {
    const auto n = static_cast<std::size_t>(variables);
    return DenseBounds{std::vector<double>(n, -std::numeric_limits<double>::infinity()),
                       std::vector<double>(n, std::numeric_limits<double>::infinity()),
                       std::vector<double>(n, 1.0)};
}

// Bitgleichheit zweier Gleitkommazahlen (auch NaN-Muster).
inline bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

// Ergebnisse zweier Läufe bitgleich (x, F, Verlauf, Zähler, Status).
inline bool same_run(const LmRunResult& a, const LmRunResult& b) {
    if (a.x.size() != b.x.size() || a.history.size() != b.history.size() || a.status != b.status ||
        a.iterations != b.iterations || a.accepted != b.accepted || a.jacobians != b.jacobians ||
        a.failed_rays != b.failed_rays || !same_bits(a.merit, b.merit) ||
        !same_bits(a.damping, b.damping) || !same_bits(a.gradient_norm, b.gradient_norm)) {
        return false;
    }
    for (std::size_t k = 0; k < a.x.size(); ++k) {
        if (!same_bits(a.x[k], b.x[k])) {
            return false;
        }
    }
    for (std::size_t k = 0; k < a.history.size(); ++k) {
        if (!same_bits(a.history[k], b.history[k])) {
            return false;
        }
    }
    return true;
}

// Gewicht der Brennweitenbedingung [1/mm²] in den Abnahmetests. Die Strahlresiduen sind seit
// M4-T1 in Wellen bei λ_ref mit auf Summe 1 normierten Feld- und Spektralgewichten
// (r = √(ŵ_f ŵ_λ W_i V_j) · OPD/λ_ref, bei der Einzellinse ≈ 1e-3 Wellen). M3-T4 hat die
// Gewichte für Residuen in mm ohne Gewichte gewählt (1e-8 bzw. 1e-6 /mm²). Hier sind sie exakt
// umgerechnet: Bei gleichen Gewichten ist jedes Strahlresiduum mit 1/(λ_ref √(F·L)) skaliert,
// die Wellenfrontmerit also mit 1/(F·L·λ_ref²) (λ_ref in mm); dasselbe Gewicht der Bedingung
// ist w_M3 / (F·L·λ_ref²), und das Problem ist bis auf einen gemeinsamen Faktor von F dasselbe
// (LM ist dagegen invariant). Einzellinse (ein Tupel, λ_ref = 587,56 nm): ≈ 0,029 /mm²;
// Mehrlinser (3 Felder, 4 Wellenlängen, λ_ref = 546,07 nm): ≈ 0,28 /mm². Ein viel höheres
// Gewicht macht JᵀJ schlecht konditioniert (die Biegung, die die Brennweite kaum ändert,
// verschwindet in der Rundung der Normalgleichungen) und LM langsam (docs/optimierung.md,
// Abschnitt 2.2). Mit 0,029 /mm² hält die Bedingung f bei der Einzellinse auf ~1e-6 mm.
inline constexpr double kLensFocalWeight = 1e-8 / (587.56e-6 * 587.56e-6);
inline constexpr double kMehrlinserFocalWeight = 1e-6 / (12.0 * 546.07e-6 * 546.07e-6);

// Einzellinse N-BK7 (einzellinse_bk7_variablen.json: f ≈ 100 mm, Blende mit Radius 2,5 mm, also
// NA ≈ 0,025, eine Wellenlänge 587,56 nm, nur Achsfeld) für die Abnahme „minimale sphärische
// Aberration“: Variablen c₁, c₂ ∈ [−0,05; 0,05] und Arbeitsabstand w ∈ [50; 150] (x_typ 10),
// Mittendicke 6 mm fest, Brennweite 100 mm als paraxiale Gleichung mit Gewicht `focal_weight`.
// Bei dieser kleinen Apertur dominiert W₀₄₀ (Terme höherer Ordnung relativ ~NA²), die
// Wellenfront wird über den Arbeitsabstand fokussiert; das Optimum ist die Biegung minimaler
// sphärischer Aberration bei fester Brennweite.
inline FdProblem lens_bending_problem(double focal_weight = 100.0) {
    SystemDescription d = load_residual_system("einzellinse_bk7_variablen.json").system;
    d.name = "Einzellinse N-BK7, Biegung auf minimale sphärische Aberration";
    auto c = variable_spec(VariableType::Curvature, SurfaceSelection::All);
    c.min = -0.05;
    c.max = 0.05;
    auto w = variable_spec(VariableType::WorkingDistance, SurfaceSelection::None);
    w.min = 50.0;
    w.max = 150.0;
    w.typical = 10.0;
    d.variables = Variables{{c, w}};
    ConstraintSpec f;
    f.type = ConstraintType::FocalLength;
    f.weight = focal_weight;
    d.constraints = Constraints{{f}};
    return make_fd_problem(prepare_system(d, std::filesystem::path(kResidualSystems)));
}

// Mehrlinser (testsystem_mehrlinser.json, f ≈ 20,6 mm, NA ≈ 0,19, 3 Felder, 4 Wellenlängen, 680
// Strahlen) mit P = 13 Variablen und Schranken: Krümmungen aller Sphären in [−0,2; 0,2],
// Dicken der Flächen 1 … 6 in [0,5; 10], Arbeitsabstand in [5; 30] (x_typ 1); Brennweite
// 20,6 mm als paraxiale Gleichung mit Gewicht `focal_weight`.
inline FdProblem mehrlinser_lm_problem(double focal_weight) {
    SystemDescription d = load_residual_system("testsystem_mehrlinser.json").system;
    auto c = variable_spec(VariableType::Curvature, SurfaceSelection::All);
    c.min = -0.2;
    c.max = 0.2;
    auto t = variable_spec(VariableType::Thickness, SurfaceSelection::List, {1, 2, 3, 4, 5, 6});
    t.min = 0.5;
    t.max = 10.0;
    auto w = variable_spec(VariableType::WorkingDistance, SurfaceSelection::None);
    w.min = 5.0;
    w.max = 30.0;
    w.typical = 1.0;
    d.variables = Variables{{c, t, w}};
    d.target = Target{};
    d.target->focal_length = 20.6;
    ConstraintSpec f;
    f.type = ConstraintType::FocalLength;
    f.weight = focal_weight;
    d.constraints = Constraints{{f}};
    return make_fd_problem(prepare_system(d, std::filesystem::path(kResidualSystems)));
}

// Stapel aus `lenses` Kopien der Einzellinse (einzellinse_bk7_variablen.json) mit Luftspalten
// von 2 mm und Arbeitsabstand 10 mm, alle Krümmungen, alle Dicken außer dem Arbeitsabstand und
// der Arbeitsabstand als Variablen (P = 4·lenses + 1, ohne Schranken). Nur für Tests der
// Normalgleichungen mit P + 1 > kGramTile (mehrere Kacheln); optisch ohne Bedeutung.
inline FdProblem lens_stack_problem(int lenses) {
    SystemDescription d = load_residual_system("einzellinse_bk7_variablen.json").system;
    d.name = "Linsenstapel";
    d.target.reset();
    const Surface front = d.surfaces[1];
    Surface back = d.surfaces[2];
    back.thickness = 2.0;
    d.surfaces.resize(1);
    for (int i = 0; i < lenses; ++i) {
        d.surfaces.push_back(front);
        d.surfaces.push_back(back);
    }
    d.surfaces.back().thickness = 10.0;
    std::vector<int> thick;
    for (int s = 0; s + 1 < static_cast<int>(d.surfaces.size()); ++s) {
        thick.push_back(s);
    }
    d.variables = Variables{{variable_spec(VariableType::Curvature, SurfaceSelection::All),
                             variable_spec(VariableType::Thickness, SurfaceSelection::List, thick),
                             variable_spec(VariableType::WorkingDistance, SurfaceSelection::None)}};
    return make_fd_problem(prepare_system(d, std::filesystem::path(kResidualSystems)));
}

// Plankonvexlinse an der Totalreflexion (tir_problem aus fd_test_systems.hpp), Start mit
// stärkerer Krümmung c₂ = `factor`·c₂, sodass die äußeren Ringe totalreflektieren; Variablen
// c[2] (Schranken [−1; 1]) und w (Schranken [1; 100]). Für die Robustheit bei versagten
// Strahlen.
inline FdProblem tir_start_problem(double factor) {
    FdProblem p = tir_problem();
    SystemDescription d = p.loaded.system;
    d.surfaces[2].curvature *= factor;
    auto c = variable_spec(VariableType::Curvature, SurfaceSelection::One, {2});
    c.min = -1.0;
    c.max = 1.0;
    auto w = variable_spec(VariableType::WorkingDistance, SurfaceSelection::None);
    w.min = 1.0;
    w.max = 100.0;
    d.variables = Variables{{c, w}};
    return make_fd_problem(prepare_system(d, std::filesystem::path(kResidualSystems)));
}

// Formfaktor q = (c₁ + c₂)/(c₁ − c₂) (Coddington) aus x = (c₁, c₂, w).
inline double shape_factor(std::span<const double> x) {
    return (x[0] + x[1]) / (x[0] - x[1]);
}

// Dünnlinsentheorie, Objekt im Unendlichen: q_min = 2(n² − 1)/(n + 2).
inline double thin_lens_best_shape(double n) {
    return 2.0 * (n * n - 1.0) / (n + 2.0);
}

// Startpunkte, in denen jede Variable um genau 1 % ihres Betrags gestört ist, nach einem festen
// Vorzeichenmuster: Muster 0 alle +, 1 alle −, 2 abwechselnd (+ zuerst), 3 abwechselnd
// (− zuerst). Würde die Störung eine Schranke verletzen (Variable auf oder nahe der Schranke),
// wird das Vorzeichen umgedreht, sodass wirklich alle Variablen gestört sind.
inline std::vector<double> one_percent_starts(const VariableSet& set, std::span<const double> x,
                                              int patterns = 4) {
    const auto P = static_cast<std::size_t>(set.size());
    std::vector<double> out;
    for (int m = 0; m < patterns; ++m) {
        std::vector<double> y(x.begin(), x.end());
        for (std::size_t k = 0; k < P; ++k) {
            double s = 1.0;
            if (m == 1) {
                s = -1.0;
            } else if (m == 2) {
                s = k % 2 == 0 ? 1.0 : -1.0;
            } else if (m == 3) {
                s = k % 2 == 0 ? -1.0 : 1.0;
            }
            const double lo = set.lower()[k];
            const double hi = set.upper()[k];
            double v = x[k] + 0.01 * s * std::abs(x[k]);
            if (v < lo || v > hi) {
                v = x[k] - 0.01 * s * std::abs(x[k]);
            }
            y[k] = v;
        }
        if (!set.feasible(y)) {
            throw std::logic_error("one_percent_starts: Störung passt nicht in die Schranken");
        }
        out.insert(out.end(), y.begin(), y.end());
    }
    return out;
}

// `runs` zufällig gestörte Startpunkte (perturbed_vector, gleichverteilt, `relative` pro
// Variable, Startwert `seed`), hintereinander.
inline std::vector<double> perturbed_starts(const VariableSet& set, std::span<const double> x0,
                                            int runs, double relative, std::uint64_t seed) {
    PerturbationOptions o;
    o.curvature = {relative, 0.0};
    o.thickness = {relative, 0.0};
    o.working_distance = {relative, 0.0};
    o.seed = seed;
    std::vector<double> out;
    for (int j = 0; j < runs; ++j) {
        const std::vector<double> y = perturbed_vector(set, x0, o, static_cast<std::uint64_t>(j));
        out.insert(out.end(), y.begin(), y.end());
    }
    return out;
}

// Die Startpunkte eines Laufs.
inline std::span<const double> run_start(std::span<const double> x0, int P, int run) {
    return x0.subspan(static_cast<std::size_t>(run) * static_cast<std::size_t>(P),
                      static_cast<std::size_t>(P));
}

} // namespace mko::test
