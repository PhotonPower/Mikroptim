#pragma once

// Testsysteme mit Variablen, Richardson-Referenz und Fehlermaße für die Tests der
// finite-Differenzen-Jacobi (CPU und GPU, M3-T2).

#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/model/variables.hpp"
#include "mikroptim/optim/extra_residuals.hpp"
#include "mikroptim/optim/fd_jacobian.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "mikroptim/optim/variant_generator.hpp"
#include "residual_test_systems.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace mko::test {

// System, Basisbatch (eine Variante), Variablen und Startvektor.
struct FdProblem {
    LoadedSystem loaded;
    SystemBatchHost base;
    VariableSet set;
    std::vector<double> x0;
};

// Zusatzresiduen aller Varianten, Ablage [v·E + i] (evaluate_extra_residuals, M3-T3).
inline std::vector<double> extra_values(const SystemBatchHost& batch, const ExtraResiduals& extra) {
    std::vector<double> out(static_cast<std::size_t>(batch.variants()) *
                            static_cast<std::size_t>(extra.count));
    evaluate_extra_residuals(extra, batch, out);
    return out;
}

inline FdProblem make_fd_problem(LoadedSystem loaded) {
    FdProblem p{std::move(loaded), {}, {}, {}};
    p.base = make_system_batch(p.loaded, 1);
    p.set = VariableSet(p.loaded.system, p.base.topology());
    p.x0 = p.set.extract(p.base, 0);
    return p;
}

inline VariableSpec variable_spec(VariableType type, SurfaceSelection selection,
                                  std::vector<int> surfaces = {}) {
    VariableSpec s;
    s.type = type;
    s.selection = selection;
    s.surfaces = std::move(surfaces);
    return s;
}

// Mehrlinser (testsystem_mehrlinser.json, 3 Felder, 4 Wellenlängen, 680 Strahlen) mit
// P = 13 Variablen ohne Schranken: Krümmungen aller Sphären (Dateiflächen 1, 2, 3, 4, 6, 7),
// alle Dicken der Flächen 1 … 6 und der Arbeitsabstand.
inline FdProblem mehrlinser_problem() {
    SystemDescription d = load_residual_system("testsystem_mehrlinser.json").system;
    Variables v;
    v.items.push_back(variable_spec(VariableType::Curvature, SurfaceSelection::All));
    v.items.push_back(
        variable_spec(VariableType::Thickness, SurfaceSelection::List, {1, 2, 3, 4, 5, 6}));
    v.items.push_back(variable_spec(VariableType::WorkingDistance, SurfaceSelection::None));
    d.variables = v;
    return make_fd_problem(prepare_system(d, std::filesystem::path(kResidualSystems)));
}

// Einzellinse mit Variablen (einzellinse_bk7_variablen.json): c[1], c[2] in [−0,05; 0,05],
// t[1] in [2; 12], w in [50; 150] (x_typ 10); x₀ = (0,00967; −0,00967; 6; 99,05).
inline FdProblem einzellinse_problem() {
    return make_fd_problem(load_residual_system("einzellinse_bk7_variablen.json"));
}

// Echte paraxiale Zusatzresiduen (M3-T3) aus dem Abschnitt `constraints` der Systemdatei.
inline ExtraResiduals paraxial_extra(const FdProblem& p) {
    return make_paraxial_residuals(make_paraxial_constraints(p.loaded.system));
}

// Einzellinse mit Nebenbedingungen (einzellinse_bk7_nebenbedingungen.json): Variablen wie
// einzellinse_problem, Zusatzresiduen Brennweite (Gleichung, Gewicht 100), Abgleichlänge und
// Arbeitsabstand als Untergrenze (bei x₀ inaktiv, Residuum und Ableitung 0).
inline FdProblem einzellinse_constraints_problem() {
    return make_fd_problem(load_residual_system("einzellinse_bk7_nebenbedingungen.json"));
}

// Mehrlinser mit den Variablen von mehrlinser_problem und paraxialen Nebenbedingungen:
// Brennweite 20,6 mm und Arbeitsabstand 14,45 mm als Gleichungen, Bildebene im paraxialen Fokus.
inline FdProblem mehrlinser_constraints_problem() {
    FdProblem p = mehrlinser_problem();
    SystemDescription d = p.loaded.system;
    d.target = Target{};
    d.target->focal_length = 20.6;
    d.target->working_distance = 14.45;
    Constraints c;
    ConstraintSpec f;
    f.type = ConstraintType::FocalLength;
    ConstraintSpec w;
    w.type = ConstraintType::WorkingDistance;
    ConstraintSpec focus;
    focus.type = ConstraintType::ParaxialFocus;
    c.items = {f, w, focus};
    d.constraints = c;
    return make_fd_problem(prepare_system(d, std::filesystem::path(kResidualSystems)));
}

// Einzellinse mit Schranken an x₀ (alle vier Schablonen des zentralen Schemas): c[1] auf der
// oberen Schranke → einseitig (x₀ − h, x₀ − 2h); c[2] in einem Bereich von 1,3 h → gekürzt;
// t[1] 0,25 h über der unteren Schranke → einseitig (x₀ + h, x₀ + 2h); w im Inneren → zentral.
inline FdProblem einzellinse_bound_problem() {
    FdProblem p = einzellinse_problem();
    const double h1 = fd_step(p.x0[1], p.set[1].typical);
    const double h2 = fd_step(p.x0[2], p.set[2].typical);
    SystemDescription d = p.loaded.system;
    auto c1 = variable_spec(VariableType::Curvature, SurfaceSelection::One, {1});
    c1.min = -0.05;
    c1.max = p.x0[0];
    auto c2 = variable_spec(VariableType::Curvature, SurfaceSelection::One, {2});
    c2.min = p.x0[1] - 0.5 * h1;
    c2.max = p.x0[1] + 0.8 * h1;
    auto t1 = variable_spec(VariableType::Thickness, SurfaceSelection::One, {1});
    t1.min = p.x0[2] - 0.25 * h2;
    t1.max = 12.0;
    auto w = variable_spec(VariableType::WorkingDistance, SurfaceSelection::None);
    w.min = 50.0;
    w.max = 150.0;
    w.typical = 10.0;
    d.variables = Variables{{c1, c2, t1, w}};
    p.set = VariableSet(d, p.base.topology());
    p.loaded.system = d;
    return p;
}

// Plankonvexlinse (Planseite vorn, N-BK7, kollimiert auf der Achse), deren hintere Krümmung c₂
// so gewählt ist, dass der äußerste Ring (Höhe y in der Linse) knapp unter dem Grenzwinkel
// austritt: n·y·|c₂| = 1 − 3·10⁻⁶. Der FD-Punkt c₂ − h (|c₂| um 6·10⁻⁶ relativ größer)
// totalreflektiert diesen Strahl, alle übrigen bleiben gültig. Variablen c[2] und w.
inline FdProblem tir_problem(const ResidualSettings& quadrature = {}) {
    SystemDescription d = load_residual_system("einzellinse_bk7_variablen.json").system;
    d.name = "Plankonvexlinse an der Totalreflexion";
    d.target.reset();
    d.surfaces[1].type = SurfaceType::Plane;
    d.surfaces[1].curvature = 0.0;
    d.surfaces[1].semi_diameter = 3.0;
    d.surfaces[2].curvature = -0.25; // vorläufig, unten ersetzt
    d.surfaces[2].semi_diameter = 3.0;
    d.surfaces[2].thickness = 7.0;
    Variables v;
    v.items.push_back(variable_spec(VariableType::Curvature, SurfaceSelection::One, {2}));
    v.items.push_back(variable_spec(VariableType::WorkingDistance, SurfaceSelection::None));
    d.variables = v;
    LoadedSystem probe = prepare_system(d, std::filesystem::path(kResidualSystems));
    const SystemBatchHost b = make_system_batch(probe, 1);
    const ResidualLayout layout(b.topology(), quadrature);
    const PupilQuadrature<double>& q = layout.quadrature(0);
    const double y = b.pupil_radius(0) * std::sqrt(q.u[q.rings - 1]); // äußerster Ring
    const double n = b.index(0, 1, 0);                                // Glas nach Fläche 1
    d.surfaces[2].curvature = -(1.0 - 3e-6) / (n * y);
    return make_fd_problem(prepare_system(d, std::filesystem::path(kResidualSystems)));
}

// Zentrale Differenzen D(H) aller Spalten mit H_k = factor·max(|x₀_k|, x_typ,k) ohne Rücksicht
// auf Schranken (CPU, residuals_cpu), Ablage [P][R] wie eine Spalte von FdJacobian. Die
// Funktion r(x) ist auch außerhalb der Schranken definiert. failed: versagte Strahlen in den
// gestörten Varianten.
inline std::vector<double> central_differences(const FdProblem& p, std::span<const double> x0,
                                               const ResidualLayout& layout,
                                               const ExtraResiduals& extra, double factor,
                                               std::size_t* failed = nullptr) {
    const int P = p.set.size();
    SystemBatchHost b = p.base;
    b.resize(2 * P);
    std::vector<double> xp(P);
    std::vector<double> xm(P);
    for (int k = 0; k < P; ++k) {
        const auto kk = static_cast<std::size_t>(k);
        const double h = factor * std::max(std::abs(x0[kk]), p.set[k].typical);
        std::vector<double> y(x0.begin(), x0.end());
        y[kk] = x0[kk] + h;
        xp[kk] = y[kk];
        b.copy_variant(0, 2 * k);
        p.set.apply(b, 2 * k, y);
        y[kk] = x0[kk] - h;
        xm[kk] = y[kk];
        b.copy_variant(0, 2 * k + 1);
        p.set.apply(b, 2 * k + 1, y);
    }
    const ResidualResult res = residuals_cpu(b, layout);
    if (failed != nullptr) {
        *failed = res.failed_count();
    }
    const std::vector<double> e = extra_values(b, extra);
    const std::int64_t rv = res.rays_per_variant;
    const std::int64_t R = rv + extra.count;
    std::vector<double> d(static_cast<std::size_t>(P) * static_cast<std::size_t>(R));
    const auto value = [&](int v, std::int64_t r) {
        return r < rv ? res.residual(v, r)
                      : e[static_cast<std::size_t>(v) * static_cast<std::size_t>(extra.count) +
                          static_cast<std::size_t>(r - rv)];
    };
    for (int k = 0; k < P; ++k) {
        const auto kk = static_cast<std::size_t>(k);
        const double step = xp[kk] - xm[kk];
        for (std::int64_t r = 0; r < R; ++r) {
            d[kk * static_cast<std::size_t>(R) + static_cast<std::size_t>(r)] =
                (value(2 * k, r) - value(2 * k + 1, r)) / step;
        }
    }
    return d;
}

// Richardson-Extrapolation zweistufig aus D(H), D(H/2), D(H/4):
// R₁(H) = (4 D(H/2) − D(H))/3 (Fehler O(H⁴)), R₂ = (16 R₁(H/2) − R₁(H))/15
//       = (64 D(H/4) − 20 D(H/2) + D(H))/45 (Fehler O(H⁶)).
struct RichardsonReference {
    std::vector<double> r1; // R₁(H/2), zur Kontrolle der Konvergenz
    std::vector<double> r2; // Referenz
    std::size_t failed = 0; // versagte Strahlen in allen gestörten Varianten
};

inline RichardsonReference richardson_reference(const FdProblem& p, std::span<const double> x0,
                                                const ResidualLayout& layout,
                                                const ExtraResiduals& extra = {},
                                                double factor = 1e-3) {
    std::size_t f1 = 0;
    std::size_t f2 = 0;
    std::size_t f4 = 0;
    const std::vector<double> d1 = central_differences(p, x0, layout, extra, factor, &f1);
    const std::vector<double> d2 = central_differences(p, x0, layout, extra, factor / 2, &f2);
    const std::vector<double> d4 = central_differences(p, x0, layout, extra, factor / 4, &f4);
    RichardsonReference ref;
    ref.failed = f1 + f2 + f4;
    ref.r1.resize(d1.size());
    ref.r2.resize(d1.size());
    for (std::size_t i = 0; i < d1.size(); ++i) {
        ref.r1[i] = (4.0 * d4[i] - d2[i]) / 3.0;
        ref.r2[i] = (64.0 * d4[i] - 20.0 * d2[i] + d1[i]) / 45.0;
    }
    return ref;
}

// Relative Abweichung einer Jacobi-Matrix J (Spalten zusammenhängend, [P][R]) von einer
// Referenz D: pro Spalte ‖J_p − D_p‖₂/‖D_p‖₂ (unabhängig von der Einheit der Variable) und
// gesamt ‖J − D‖_F/‖D‖_F.
struct JacobianDeviation {
    std::vector<double> column;
    double max_column = 0.0;
    int worst_column = -1;
    double frobenius = 0.0;
};

inline JacobianDeviation jacobian_deviation(std::span<const double> j, std::span<const double> d,
                                            int columns) {
    if (j.size() != d.size() || columns < 1 || d.size() % static_cast<std::size_t>(columns) != 0) {
        throw std::invalid_argument("jacobian_deviation: Größen passen nicht");
    }
    const std::size_t rows = d.size() / static_cast<std::size_t>(columns);
    JacobianDeviation out;
    double diff_all = 0.0;
    double ref_all = 0.0;
    for (int p = 0; p < columns; ++p) {
        double diff = 0.0;
        double ref = 0.0;
        for (std::size_t r = 0; r < rows; ++r) {
            const std::size_t i = static_cast<std::size_t>(p) * rows + r;
            diff += (j[i] - d[i]) * (j[i] - d[i]);
            ref += d[i] * d[i];
        }
        diff_all += diff;
        ref_all += ref;
        const double rel = ref > 0.0 ? std::sqrt(diff / ref)
                                     : (diff > 0.0 ? std::numeric_limits<double>::infinity() : 0.0);
        out.column.push_back(rel);
        if (rel > out.max_column || out.worst_column < 0) {
            out.max_column = std::max(out.max_column, rel);
            out.worst_column = p;
        }
    }
    out.frobenius = std::sqrt(diff_all / ref_all);
    return out;
}

// Spalten skaliert mit der natürlichen Größe der Variablen s_k = max(|x₀_k|, x_typ,k) (dieselbe
// Größe, die die FD-Schrittweite festlegt): Ĵ = J·diag(s). Jede Spalte hat dann die Einheit
// des Residuums (Änderung pro relativer Änderung der Variable), die Frobeniusnorm von Ĵ hängt
// nicht von den Einheiten der Variablen ab.
inline std::vector<double> scale_columns(std::span<const double> j, const FdProblem& p,
                                         std::span<const double> x0) {
    const auto P = static_cast<std::size_t>(p.set.size());
    const std::size_t rows = j.size() / P;
    std::vector<double> out(j.begin(), j.end());
    for (std::size_t k = 0; k < P; ++k) {
        const double s = std::max(std::abs(x0[k]), p.set[static_cast<int>(k)].typical);
        for (std::size_t r = 0; r < rows; ++r) {
            out[k * rows + r] *= s;
        }
    }
    return out;
}

// Schätzung des Rundungsanteils im relativen Spaltenfehler einer FD-Spalte (nur
// Strahlresiduen): Jedes Residuum r = √W·OPD trägt einen Rundungsfehler ≈ √S·ε·L·√W mit der
// optischen Länge L des Systems (die OPL vor dem Kolbenabzug, nicht |OPD|, bestimmt die
// Rundung; jede der S Flächen trägt etwa ε·L bei). Über die Zeilen ergibt das
// ‖δJ_p‖₂ ≈ √S·ε·L·√(Σ W)·Σ|c_x| mit den Gewichten c_x der Formel (zentral 2/|d_a − d_b| ≈ 1/h,
// einseitig ≈ 4/h, vorwärts 2/|d_a|), Σ W = Zahl der Tupel (f, l). Relativ zu ‖D_p‖₂.
inline double column_rounding_estimate(const FdProblem& p, const ResidualLayout& layout,
                                       const FdPattern& pattern, int k, double column_norm) {
    double length = 0.0; // optische Länge auf der Achse, größte Brechzahl über λ
    for (int s = 0; s < p.base.surfaces(); ++s) {
        double n = 0.0;
        for (int l = 0; l < p.base.wavelengths(); ++l) {
            n = std::max(n, p.base.index(0, s, l));
        }
        length += n * std::abs(p.base.thickness(0, s));
    }
    const auto kk = static_cast<std::size_t>(k);
    const double da = pattern.step_plus[kk];
    double coef = 0.0;
    switch (pattern.stencil[kk]) {
    case FdStencil::OneSided: {
        const double db = pattern.step_minus[kk];
        const double ca = std::abs(db * db / (da * db * (db - da)));
        const double cb = std::abs(da * da / (da * db * (db - da)));
        coef = ca + cb + std::abs(ca - cb);
        break;
    }
    case FdStencil::Forward:
        coef = 2.0 / std::abs(da);
        break;
    default:
        coef = 2.0 / std::abs(da - pattern.step_minus[kk]);
        break;
    }
    const double tuples = static_cast<double>(layout.fields() * layout.wavelengths());
    const double surfaces = static_cast<double>(p.base.surfaces());
    return std::sqrt(surfaces) * std::numeric_limits<double>::epsilon() * length *
           std::sqrt(tuples) * coef / column_norm;
}

// Euklidische Norm der Spalte k einer Matrix [P][R].
inline double column_norm(std::span<const double> d, int columns, int k) {
    const std::size_t rows = d.size() / static_cast<std::size_t>(columns);
    double s = 0.0;
    for (std::size_t r = 0; r < rows; ++r) {
        const double v = d[static_cast<std::size_t>(k) * rows + r];
        s += v * v;
    }
    return std::sqrt(s);
}

// Spalten eines Laufs als Spanne.
inline std::span<const double> run_jacobian(const FdJacobian& j, int run) {
    const std::size_t n =
        static_cast<std::size_t>(j.shape.variables) * static_cast<std::size_t>(j.shape.rows());
    return std::span<const double>(j.jacobian).subspan(static_cast<std::size_t>(run) * n, n);
}

// Zusatzresiduen als analytische Funktion der Variablen der Einzellinse (x = c₁, c₂, t₁, w):
// e₀ = c₁² + 2c₂, e₁ = t₁·w/100, e₂ = 1/w. Ableitungen exakt bekannt (analytic_extra_derivative).
inline ExtraResiduals einzellinse_extra(const VariableSet& set) {
    ExtraResiduals e;
    e.count = 3;
    e.evaluate = [&set](const SystemBatchHost& batch, int v, std::span<double> out) {
        const std::vector<double> x = set.extract(batch, v);
        out[0] = x[0] * x[0] + 2.0 * x[1];
        out[1] = x[2] * x[3] / 100.0;
        out[2] = 1.0 / x[3];
    };
    return e;
}

// ∂e_i/∂x_k von einzellinse_extra in x.
inline double analytic_extra_derivative(int i, int k, std::span<const double> x) {
    switch (i) {
    case 0:
        return k == 0 ? 2.0 * x[0] : (k == 1 ? 2.0 : 0.0);
    case 1:
        return k == 2 ? x[3] / 100.0 : (k == 3 ? x[2] / 100.0 : 0.0);
    default:
        return k == 3 ? -1.0 / (x[3] * x[3]) : 0.0;
    }
}

// Zusatzresiduen des Mehrlinsers: e₀ = Σ_k x_k² (alle Variablen), e₁ = x₀·x₁.
inline ExtraResiduals mehrlinser_extra(const VariableSet& set) {
    ExtraResiduals e;
    e.count = 2;
    e.evaluate = [&set](const SystemBatchHost& batch, int v, std::span<double> out) {
        const std::vector<double> x = set.extract(batch, v);
        double s = 0.0;
        for (const double xi : x) {
            s += xi * xi;
        }
        out[0] = s;
        out[1] = x[0] * x[1];
    };
    return e;
}

} // namespace mko::test
