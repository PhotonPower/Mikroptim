#pragma once

// Glas-Einrasten (M6-T5, CPU und GPU): kontinuierliches Optimum des frei erfundenen Achromaten
// data/systems/achromat_glasvariablen.json (zwei Glasvariablen, Testgläser
// data/glass/synthetisch.json und basis.json), danach Einrasten auf Katalog-Gläser
// (docs/optimierung.md, Abschnitt 9).

#include "fd_test_systems.hpp"
#include "mikroptim/glass/glass_hull.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/optim/glass_snap.hpp"
#include "mikroptim/optim/glass_variables.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace mko::test {

inline constexpr const char* kSnapSystem = "achromat_glasvariablen.json";
// Zyklen der kontinuierlichen Optimierung (Glas-Hülle mit dem Gewicht der Datei, 10⁶). Das
// Optimum liegt im Inneren der Hülle (Abstand 0), eine Penalty-Fortsetzung wie in M6-T1 ist
// nicht nötig; der LM konvergiert nach ~420 Zyklen (converged_step).
inline constexpr int kSnapContinuousIterations = 5000;
// Zyklen der Reoptimierung nach dem Einrasten.
inline constexpr int kSnapIterations = 500;

// Zusatzresiduen ohne Glas-Hülle (paraxiale Nebenbedingungen der Datei): dieselben für das
// kontinuierliche und das eingerastete System.
inline ExtraResiduals snap_extra(const SystemDescription& system) {
    return make_paraxial_residuals(make_paraxial_constraints(system));
}

// Kontinuierliches Optimum: System nach der Optimierung (Buchdahl-Materialien mit den
// optimierten Koordinaten, Glasvariablen bleiben), Hülle der Kataloge, Lauf.
struct ContinuousOptimum {
    LoadedSystem loaded;
    GlassHull hull;
    LmRunResult run;
};

// Optimierer: LmResult optimize(const VariableSet&, const SystemBatchHost&, span x0, LmOptions).
template <class Optimize>
ContinuousOptimum continuous_optimum(Optimize&& optimize) {
    const LoadedSystem raw = load_residual_system(kSnapSystem);
    SystemDescription d = raw.system;
    convert_glass_variables(d, raw.catalog);
    const FdProblem p = make_fd_problem(prepare_system(d, std::filesystem::path(kResidualSystems)));
    ContinuousOptimum out;
    out.hull = make_glass_hull(p.loaded.catalog);
    LmOptions o;
    o.max_iterations = kSnapContinuousIterations;
    o.jacobian.extra = concat_extra_residuals(
        {snap_extra(p.loaded.system),
         make_glass_hull_residuals(make_glass_hull_penalty(p.loaded.system, p.set, out.hull))});
    out.run = optimize(p.set, p.base, std::span<const double>(p.x0), o).runs.at(0);
    SystemDescription result = p.loaded.system;
    p.set.apply_to_description(result, out.run.x);
    out.loaded = prepare_system(result, std::filesystem::path(kResidualSystems));
    return out;
}

// Einstellungen des Einrastens für die Tests.
inline GlassSnapOptions snap_options(const SystemDescription& system, int neighbors,
                                     int max_combinations) {
    GlassSnapOptions o;
    o.neighbors = neighbors;
    o.max_combinations = max_combinations;
    o.lm.max_iterations = kSnapIterations;
    o.lm.jacobian.extra = snap_extra(system);
    return o;
}

} // namespace mko::test
