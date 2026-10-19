#pragma once

// Testsysteme und Vergleiche für die adjungierte Jacobi-Matrix (M5-T2, CPU und GPU): alle
// Variablentypen (Krümmung, Dicke auch im geteilten Spalt mit Geisterplatten, Arbeitsabstand,
// Geisterkoeffizient, Glas), Zusatzresiduen und trace-abhängige Zusatzresiduen, weiches Versagen,
// Punktquelle; Fehlermaß wie in der Abnahme M3-T2 (fd_test_systems.hpp).

#include "fd_test_systems.hpp"
#include "ghost_test_systems.hpp"
#include "glass_test_systems.hpp"
#include "mechanical_test_systems.hpp"
#include "mikroptim/model/variables.hpp"
#include "mikroptim/optim/adjoint_jacobian.hpp"
#include "mikroptim/optim/fd_jacobian.hpp"
#include "residual_test_systems.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace mko::test {

// Ein Testfall: Problem (System, Basisbatch, Variablen, x₀) und Optionen der Jacobi-Rechnung.
// `richardson`: Die Richardson-Referenz (nur Strahl- und einfache Zusatzresiduen) ist anwendbar;
// sonst wird nur gegen die FD-Jacobi verglichen (trace-abhängige Zusatzresiduen).
struct AdjointCase {
    std::string name;
    FdProblem p;
    AdjointJacobianOptions options;
    bool richardson = true;
};

// Name des Variablentyps für die Ausgabe.
inline const char* variable_type_name(VariableType t) {
    switch (t) {
    case VariableType::Curvature:
        return "Krümmung";
    case VariableType::Thickness:
        return "Dicke";
    case VariableType::WorkingDistance:
        return "Arbeitsabstand";
    case VariableType::GhostCoefficient:
        return "Geisterkoeffizient";
    case VariableType::Glass:
        return "Glas";
    }
    return "?";
}

// Einzellinse mit zwei Geisterplatten im Spalt hinter der Blende (einzellinse_geisterplatte.json,
// 3 Felder, 3 Wellenlängen): Krümmungen, Dicke des geteilten Spalts nach der Blende (wirkt auf
// das letzte Teilstück), Mittendicke, Arbeitsabstand und alle 10 Geisterkoeffizienten.
inline FdProblem ghost_gap_problem() {
    SystemDescription d = load_residual_system("einzellinse_geisterplatte.json").system;
    Variables v;
    v.items.push_back(variable_spec(VariableType::Curvature, SurfaceSelection::All));
    v.items.push_back(variable_spec(VariableType::Thickness, SurfaceSelection::List, {0, 1}));
    v.items.push_back(variable_spec(VariableType::WorkingDistance, SurfaceSelection::None));
    VariableSpec g;
    g.type = VariableType::GhostCoefficient;
    g.selection = SurfaceSelection::None;
    g.plate_selection = SurfaceSelection::All;
    v.items.push_back(g);
    d.variables = v;
    return make_fd_problem(prepare_system(d, std::filesystem::path(kResidualSystems)));
}

// Punktquelle neben der Achse (wie weighted_cases, Einzellinse N-BK7, Quelle (0; 2; −300) mm,
// NA 0,05, Bildebene bei 120 mm): Krümmungen, Mittendicke, Arbeitsabstand.
inline FdProblem point_source_problem() {
    SystemDescription d = load_residual_system("einzellinse_bk7.json").system;
    d.source.type = SourceType::Point;
    d.source.field_angles_deg.clear();
    d.source.field_weights.clear();
    d.source.numerical_aperture = 0.05;
    d.source.point = {0.0, 2.0, -300.0};
    d.surfaces.back().thickness = 120.0;
    d.target.reset();
    Variables v;
    v.items.push_back(variable_spec(VariableType::Curvature, SurfaceSelection::All));
    v.items.push_back(variable_spec(VariableType::Thickness, SurfaceSelection::One, {1}));
    v.items.push_back(variable_spec(VariableType::WorkingDistance, SurfaceSelection::None));
    d.variables = v;
    return make_fd_problem(prepare_system(d, std::filesystem::path(kResidualSystems)));
}

// Alle Testfälle (als unique_ptr, weil Zusatzresiduen Teile des Problems referenzieren können).
inline std::vector<std::unique_ptr<AdjointCase>> adjoint_cases() {
    std::vector<std::unique_ptr<AdjointCase>> out;
    const auto add = [&out](std::string name, FdProblem p) -> AdjointCase& {
        auto c = std::make_unique<AdjointCase>();
        c->name = std::move(name);
        c->p = std::move(p);
        out.push_back(std::move(c));
        return *out.back();
    };
    {
        AdjointCase& c = add("Einzellinse + paraxial", einzellinse_constraints_problem());
        c.options.extra = paraxial_extra(c.p);
    }
    {
        AdjointCase& c = add("Mehrlinser + paraxial", mehrlinser_constraints_problem());
        c.options.extra = paraxial_extra(c.p);
    }
    add("Geisterplatten im Spalt", ghost_gap_problem());
    {
        AdjointCase& c = add("Geisterplatten exakt + L1", ghost_exact_problem());
        c.options.extra = ghost_extra(c.p.loaded.system, ghost_penalty_of(c.p.loaded.system));
    }
    {
        const GlassProblem g = glass_problem();
        AdjointCase& c = add("Glasvariablen + Hülle", g.p);
        c.options.extra = glass_extra(g, 1e8);
    }
    add("Punktquelle", point_source_problem());
    add("Weiches Versagen", make_fd_problem(load_residual_system("weiches_versagen_start.json")));
    {
        AdjointCase& c = add("Mechanik (trace-abhängig)", mechanical_problem());
        const LmOptions o = mechanical_options(c.p.loaded.system, 0);
        c.options.extra = o.jacobian.extra;
        c.options.trace_extra = o.jacobian.trace_extra;
        c.richardson = false;
    }
    return out;
}

// Jacobi-Matrix eines Laufs als Spanne (Spalten zusammenhängend, [P][R]).
inline std::span<const double> run_columns(const AdjointJacobian& j, int run) {
    const std::size_t n =
        static_cast<std::size_t>(j.shape.variables) * static_cast<std::size_t>(j.shape.rows());
    return std::span<const double>(j.jacobian).subspan(static_cast<std::size_t>(run) * n, n);
}

// Zeilen [first, last) jeder Spalte einer Matrix [P][R] (z. B. nur Wellenfront- oder nur
// Strafzeilen).
inline std::vector<double> row_block(std::span<const double> j, int columns, std::int64_t first,
                                     std::int64_t last) {
    const std::size_t rows = j.size() / static_cast<std::size_t>(columns);
    std::vector<double> out;
    for (int p = 0; p < columns; ++p) {
        for (std::int64_t r = first; r < last; ++r) {
            out.push_back(j[static_cast<std::size_t>(p) * rows + static_cast<std::size_t>(r)]);
        }
    }
    return out;
}

// Vergleich der adjungierten Jacobi (ein Lauf) mit einer Referenz: relativ pro Spalte, in der
// Frobeniusnorm und spaltenskaliert (S = diag(max(|x₀|, x_typ)), Abnahmemaß aus M3-T2).
struct AdjointDeviation {
    JacobianDeviation plain;
    JacobianDeviation scaled;
};

inline AdjointDeviation adjoint_deviation(std::span<const double> j, std::span<const double> ref,
                                          const FdProblem& p) {
    const int P = p.set.size();
    AdjointDeviation d;
    d.plain = jacobian_deviation(j, ref, P);
    d.scaled = jacobian_deviation(scale_columns(j, p, p.x0), scale_columns(ref, p, p.x0), P);
    return d;
}

// Größter relativer Spaltenfehler pro Variablentyp (−1, wenn der Typ fehlt).
struct TypeDeviation {
    VariableType type;
    double max_column = -1.0;
};

inline std::vector<TypeDeviation> deviation_by_type(const FdProblem& p,
                                                    const JacobianDeviation& d) {
    std::vector<TypeDeviation> out;
    for (const VariableType t :
         {VariableType::Curvature, VariableType::Thickness, VariableType::WorkingDistance,
          VariableType::GhostCoefficient, VariableType::Glass}) {
        TypeDeviation td{t, -1.0};
        for (int k = 0; k < p.set.size(); ++k) {
            if (p.set[k].type == t) {
                td.max_column = std::max(td.max_column, d.column[static_cast<std::size_t>(k)]);
            }
        }
        out.push_back(td);
    }
    return out;
}

inline void print_deviation(const char* tag, const std::string& name, const FdProblem& p,
                            const AdjointDeviation& d) {
    std::printf("[%s] %s: P = %d, skaliert (Frobenius) %.3e, unskaliert %.3e, größte Spalte "
                "%.3e (%s)\n",
                tag, name.c_str(), p.set.size(), d.scaled.frobenius, d.plain.frobenius,
                d.plain.max_column, p.set[d.plain.worst_column].name.c_str());
    for (const TypeDeviation& t : deviation_by_type(p, d.plain)) {
        if (t.max_column >= 0.0) {
            std::printf("    %-18s größte Spalte %.3e\n", variable_type_name(t.type), t.max_column);
        }
    }
}

// Status aller Strahlen der Varianten x₀ ± H (H = factor · max(|x₀|, x_typ), wie
// central_differences) gleich dem bei x₀? Sonst springt die Ableitung zwischen den Punkten, und
// die Differenzen sind keine Referenz.
inline bool status_stable(const FdProblem& p, const ResidualLayout& layout, double factor) {
    const int P = p.set.size();
    SystemBatchHost b = p.base;
    b.resize(2 * P + 1);
    for (int k = 0; k < P; ++k) {
        const auto kk = static_cast<std::size_t>(k);
        const double h = factor * std::max(std::abs(p.x0[kk]), p.set[k].typical);
        std::vector<double> y = p.x0;
        y[kk] = p.x0[kk] + h;
        b.copy_variant(0, 2 * k + 1);
        p.set.apply(b, 2 * k + 1, y);
        y[kk] = p.x0[kk] - h;
        b.copy_variant(0, 2 * k + 2);
        p.set.apply(b, 2 * k + 2, y);
    }
    const ResidualResult res = residuals_cpu(b, layout);
    for (int v = 1; v < b.variants(); ++v) {
        for (std::int64_t i = 0; i < res.rays_per_variant; ++i) {
            if (res.status_at(v, i) != res.status_at(0, i)) {
                return false;
            }
        }
    }
    return true;
}

// Richardson-Referenz der Jacobi-Matrix mit adaptiver Schrittweite: Stufen H = factor,
// factor/10, … (`levels` Stufen, jeweils · max(|x₀|, x_typ)), nur mit stabilem Strahlstatus bei
// x₀ ± H;
// pro Spalte gilt die Stufe mit der besten Konvergenz ‖R₁ − R₂‖/‖R₂‖ über die Zeilen
// [0, row_last) (−1: alle; Abbruchfehler bei großem H, Rundung bei kleinem). r2 leer, wenn keine
// Stufe stabil ist.
struct AdaptiveReference {
    std::vector<double> r2;          // [P][R]
    std::vector<double> convergence; // pro Spalte
    std::vector<double> factor;      // gewählte Stufe pro Spalte
    double max_convergence = 0.0;
    double min_factor = 0.0;
    double max_factor = 0.0;
};

inline AdaptiveReference adaptive_reference(const FdProblem& p, const ResidualLayout& layout,
                                            const ExtraResiduals& extra, std::int64_t row_last = -1,
                                            double factor = 1e-1, int levels = 5) {
    const int P = p.set.size();
    AdaptiveReference out;
    double fac = factor * 10.0;
    for (int level = 0; level < levels; ++level) {
        fac /= 10.0;
        if (!status_stable(p, layout, fac)) {
            continue;
        }
        const RichardsonReference ref = richardson_reference(p, p.x0, layout, extra, fac);
        const std::int64_t rows_all = static_cast<std::int64_t>(ref.r2.size()) / P;
        const std::int64_t last = row_last < 0 ? rows_all : row_last;
        const JacobianDeviation conv =
            jacobian_deviation(row_block(ref.r1, P, 0, last), row_block(ref.r2, P, 0, last), P);
        if (out.r2.empty()) {
            out.r2 = ref.r2;
            out.convergence = conv.column;
            out.factor.assign(static_cast<std::size_t>(P), fac);
            continue;
        }
        const std::size_t rows = ref.r2.size() / static_cast<std::size_t>(P);
        for (int k = 0; k < P; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            if (conv.column[kk] < out.convergence[kk]) {
                out.convergence[kk] = conv.column[kk];
                out.factor[kk] = fac;
                std::copy(ref.r2.begin() + static_cast<std::ptrdiff_t>(kk * rows),
                          ref.r2.begin() + static_cast<std::ptrdiff_t>((kk + 1) * rows),
                          out.r2.begin() + static_cast<std::ptrdiff_t>(kk * rows));
            }
        }
    }
    if (!out.r2.empty()) {
        out.max_convergence = *std::max_element(out.convergence.begin(), out.convergence.end());
        out.min_factor = *std::min_element(out.factor.begin(), out.factor.end());
        out.max_factor = *std::max_element(out.factor.begin(), out.factor.end());
    }
    return out;
}

} // namespace mko::test
