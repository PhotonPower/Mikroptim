// Python-Bindings: Optimierung mit dem gebündelten Levenberg-Marquardt wie `mikroptim optimize`
// (M9-T1). Aufbau der Merit (Variablen, Nebenbedingungen, Strafen) wie in
// src/cli/optimize_command.cpp, Ergebnis mit Verlauf als NumPy.

#include "bindings.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/optim/ghost_penalty.hpp"
#include "mikroptim/optim/glass_variables.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/mechanical_constraints.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "mikroptim/optim/variant_generator.hpp"
#include "mikroptim/pupil/quadrature.hpp"

#ifdef MIKROPTIM_HAS_CUDA
#include "mikroptim/cuda/levenberg_marquardt.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#endif

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace mko::py {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Python-Rückruf hat eine Ausnahme ausgelöst; der Python-Fehler ist gesetzt und wird nach der
// Rückkehr aus der Optimierung erneut ausgelöst.
struct CallbackAbort {};

// Iterationsprotokoll (ein Eintrag pro Zyklus und Lauf, Reihenfolge wie on_iteration).
struct IterationLog {
    std::vector<std::int64_t> run;
    std::vector<std::int64_t> iteration;
    std::vector<std::uint8_t> evaluated;
    std::vector<std::uint8_t> accepted;
    std::vector<double> merit;
    std::vector<double> trial_merit;
    std::vector<double> gain_ratio;
    std::vector<double> damping;
    std::vector<double> step_norm;
    std::vector<std::int64_t> failed_rays;
    std::vector<std::int64_t> hard_failed_rays;
    std::vector<std::string> status;
};

struct PyOptimizeResult {
    PySystem system; // Ergebnis (bester Lauf)
    bool gpu = false;
    int device = -1;
    std::vector<Variable> variables;
    std::vector<double> x0;     // Startpunkt der Datei
    std::vector<double> starts; // Startpunkte aller Läufe [runs·P]
    LmResult result;
    int best = 0;
    std::int64_t start_failed = 0;
    std::int64_t start_hard_failed = 0;
    bool hard_failure = false;
    JacobianMethod jacobian = JacobianMethod::FiniteDifferences;
    IterationLog log;
    std::vector<std::string> warnings;

    const LmRunResult& best_run() const { return result.runs[sz(best)]; }
};

// Bester Lauf wie `mikroptim optimize` (select_best): endliche Merit, dann die wenigsten
// versagten Strahlen, dann die kleinste Merit, bei Gleichstand der kleinste Index; −1 ohne
// endliche Merit.
int select_best(const LmResult& result) {
    int best = -1;
    for (std::size_t i = 0; i < result.runs.size(); ++i) {
        const LmRunResult& r = result.runs[i];
        if (!std::isfinite(r.merit)) {
            continue;
        }
        if (best < 0) {
            best = static_cast<int>(i);
            continue;
        }
        const LmRunResult& b = result.runs[sz(best)];
        if (r.failed_rays < b.failed_rays ||
            (r.failed_rays == b.failed_rays && r.merit < b.merit)) {
            best = static_cast<int>(i);
        }
    }
    return best;
}

nb::dict iteration_dict(const LmIterationInfo& info) {
    nb::dict d;
    d["run"] = info.run;
    d["iteration"] = info.iteration;
    d["status"] = to_string(info.status);
    d["evaluated"] = info.evaluated;
    d["accepted"] = info.accepted;
    d["merit"] = info.merit;
    d["trial_merit"] = info.trial_merit;
    d["gain_ratio"] = info.gain_ratio;
    d["damping"] = info.damping;
    d["step_norm"] = info.step_norm;
    d["failed_rays"] = info.failed_rays;
    d["hard_failed_rays"] = info.hard_failed_rays;
    d["x"] = to_numpy(std::vector<double>(info.x.begin(), info.x.end()));
    return d;
}

PyOptimizeResult optimize_system(const PySystem& s, int max_iterations, int rings, int spokes,
                                 int runs, double perturbation, std::uint64_t seed,
                                 std::optional<double> fail_weight, bool hard_failure, bool gpu,
                                 int device, const std::string& jacobian, nb::object callback) {
    if (max_iterations < 0) {
        throw std::invalid_argument("max_iterations muss >= 0 sein");
    }
    if (rings < 1 || rings > kMaxPupilRings) {
        throw std::invalid_argument("rings muss in [1, " + std::to_string(kMaxPupilRings) +
                                    "] liegen");
    }
    if (spokes < 1) {
        throw std::invalid_argument("spokes muss >= 1 sein");
    }
    if (runs < 1) {
        throw std::invalid_argument("runs muss >= 1 sein");
    }
    if (perturbation < 0.0 || !std::isfinite(perturbation)) {
        throw std::invalid_argument("perturbation muss endlich und >= 0 sein");
    }
    if (fail_weight && hard_failure) {
        throw std::invalid_argument(
            "fail_weight gilt nur für weiches Versagen (nicht mit hard_failure=True)");
    }
    if (fail_weight && (!(*fail_weight > 0.0) || !std::isfinite(*fail_weight))) {
        throw std::invalid_argument("fail_weight muss endlich und > 0 sein");
    }
    const std::optional<JacobianMethod> method = jacobian_method_from_string(jacobian);
    if (!method) {
        throw std::invalid_argument("jacobian erwartet \"fd\" oder \"adjoint\", nicht \"" +
                                    jacobian + "\"");
    }
    if (!callback.is_none() && !PyCallable_Check(callback.ptr())) {
        throw nb::type_error("callback muss aufrufbar sein");
    }
    if (gpu) {
        require_gpu(device);
    }

    PyOptimizeResult r;
    r.hard_failure = hard_failure;
    r.jacobian = *method;
    r.warnings = s.loaded.warnings;
    bool aborted = false;
    {
        nb::gil_scoped_release release;
#ifdef MIKROPTIM_HAS_CUDA
        std::optional<cuda::TraceEngine> engine;
        if (gpu) {
            engine.emplace(device);
            r.gpu = true;
            r.device = device;
        }
#endif
        // Glasvariablen: Startgläser als Buchdahl-Materialien (wie mikroptim optimize)
        LoadedSystem loaded = s.loaded;
        std::vector<GlassConversion> glass_conversions;
        {
            SystemDescription converted = loaded.system;
            glass_conversions = convert_glass_variables(converted, loaded.catalog);
            if (!glass_conversions.empty()) {
                std::vector<std::string> warnings = std::move(loaded.warnings);
                loaded = prepare_system(std::move(converted), s.base_dir);
                loaded.warnings = std::move(warnings);
            }
        }
        const SystemDescription& sys = loaded.system;
        if (!sys.variables || sys.variables->items.empty()) {
            throw std::invalid_argument(
                "keine Variablen (Abschnitt variables fehlt oder ist leer)");
        }
        const SystemBatchHost base = make_system_batch(loaded, 1);
        const VariableSet set(sys, base.topology());
        if (set.empty()) {
            throw std::invalid_argument("keine Variablen");
        }
        const ParaxialConstraints constraints = make_paraxial_constraints(sys);
        check_paraxial_constraints(constraints, base.topology());
        const GhostPenalty penalty = make_ghost_penalty(sys);
        check_ghost_penalty(penalty, base.topology());
        GlassHullPenalty glass_penalty;
        // Glas-Hülle nur mit Glasvariablen und Eintrag glass_hull (wie mikroptim optimize)
        if (!glass_conversions.empty() && sys.constraints) {
            for (const ConstraintSpec& c : sys.constraints->items) {
                if (c.type == ConstraintType::GlassHull) {
                    glass_penalty =
                        make_glass_hull_penalty(sys, set, make_glass_hull(loaded.catalog));
                    check_glass_hull_penalty(glass_penalty, base.topology());
                    break;
                }
            }
        }
        const MechanicalConstraints mechanical_constraints = make_mechanical_constraints(sys);
        check_mechanical_constraints(mechanical_constraints, base.topology());
        r.x0 = set.extract(base, 0);
        r.variables.assign(set.variables().begin(), set.variables().end());

        if (sys.source.type == SourceType::Collimated &&
            !(sys.target && sys.target->focal_length)) {
            for (const double angle : sys.source.field_angles_deg) {
                if (angle != 0.0) {
                    r.warnings.emplace_back(
                        "Feldziele aus der paraxialen Brennweite des Startsystems (kein "
                        "target.focal_length); beim erneuten Laden des Ergebnisses ändern sie "
                        "sich");
                    break;
                }
            }
        }

        LmOptions options;
        options.max_iterations = max_iterations;
        options.jacobian_method = *method;
        options.jacobian.quadrature.rings = rings;
        options.jacobian.quadrature.spokes = spokes;
        options.jacobian.quadrature.failure = hard_failure ? FailureMode::Hard : FailureMode::Soft;
        if (fail_weight) {
            options.jacobian.quadrature.fail_weight = *fail_weight;
        }
        MechanicalResiduals mechanical = make_mechanical_residuals(mechanical_constraints);
        options.jacobian.extra = concat_extra_residuals(
            {make_paraxial_residuals(constraints), make_ghost_penalty_residuals(penalty),
             make_glass_hull_residuals(glass_penalty), std::move(mechanical.geometric)});
        options.jacobian.trace_extra = std::move(mechanical.traced);

        // Startpunkte: Lauf 0 ist x₀, Lauf j ≥ 1 gestört (wie mikroptim optimize --runs)
        if (runs == 1) {
            r.starts = r.x0;
        } else {
            PerturbationOptions p;
            p.curvature = {perturbation, 0.0};
            p.thickness = {perturbation, 0.0};
            p.working_distance = {perturbation, 0.0};
            p.ghost_coefficient = {perturbation, 0.0};
            p.glass = {perturbation, 0.0};
            p.seed = seed;
            p.first_is_base = true;
            for (int j = 0; j < runs; ++j) {
                const std::vector<double> y =
                    perturbed_vector(set, r.x0, p, static_cast<std::uint64_t>(j));
                r.starts.insert(r.starts.end(), y.begin(), y.end());
            }
        }

        IterationLog& log = r.log;
        options.on_iteration = [&log, &callback](const LmIterationInfo& info) {
            log.run.push_back(info.run);
            log.iteration.push_back(info.iteration);
            log.evaluated.push_back(info.evaluated ? 1 : 0);
            log.accepted.push_back(info.accepted ? 1 : 0);
            log.merit.push_back(info.merit);
            log.trial_merit.push_back(info.trial_merit);
            log.gain_ratio.push_back(info.gain_ratio);
            log.damping.push_back(info.damping);
            log.step_norm.push_back(info.step_norm);
            log.failed_rays.push_back(info.failed_rays);
            log.hard_failed_rays.push_back(info.hard_failed_rays);
            log.status.emplace_back(to_string(info.status));
            if (!callback.is_none()) {
                nb::gil_scoped_acquire acquire;
                try {
                    callback(iteration_dict(info));
                } catch (nb::python_error& e) {
                    e.restore();
                    throw CallbackAbort{};
                }
            }
        };

        try {
#ifdef MIKROPTIM_HAS_CUDA
            if (engine) {
                r.result = cuda::optimize_lm_gpu(*engine, set, base, r.starts, options);
            } else {
                r.result = optimize_lm_cpu(set, base, r.starts, options);
            }
#else
            r.result = optimize_lm_cpu(set, base, r.starts, options);
#endif
        } catch (const CallbackAbort&) {
            aborted = true;
        }
        if (!aborted) {
            // Versagte Strahlen bei x₀ (wie mikroptim optimize)
            std::vector<double> f(1, 0.0);
            std::vector<std::int64_t> failed(1, 0);
            std::vector<std::int64_t> hard_failed(1, 0);
            const int run0 = 0;
#ifdef MIKROPTIM_HAS_CUDA
            if (engine) {
                cuda::GpuLmBackend backend(*engine, set, base, options);
                backend.merit(std::span<const int>(&run0, 1), r.x0, f, failed, hard_failed);
            } else {
                CpuLmBackend backend(set, base, options);
                backend.merit(std::span<const int>(&run0, 1), r.x0, f, failed, hard_failed);
            }
#else
            CpuLmBackend backend(set, base, options);
            backend.merit(std::span<const int>(&run0, 1), r.x0, f, failed, hard_failed);
#endif
            r.start_failed = failed[0];
            r.start_hard_failed = hard_failed[0];
            r.best = select_best(r.result);
            if (r.best < 0) {
                throw std::runtime_error(
                    "Merit in keinem Lauf endlich (Startpunkt nicht auswertbar)");
            }
            // Ergebnis als Systembeschreibung; Glaskataloge relativ zum Verzeichnis des Starts
            SystemDescription result_system = sys;
            set.apply_to_description(result_system, r.best_run().x);
            r.system.base_dir = s.base_dir;
            r.system.loaded = prepare_system(std::move(result_system), s.base_dir);
        }
    }
    if (aborted) {
        throw nb::python_error();
    }
    return r;
}

nb::dict log_dict(const IterationLog& log) {
    nb::dict d;
    d["run"] = to_numpy(log.run);
    d["iteration"] = to_numpy(log.iteration);
    d["evaluated"] = to_numpy(log.evaluated);
    d["accepted"] = to_numpy(log.accepted);
    d["merit"] = to_numpy(log.merit);
    d["trial_merit"] = to_numpy(log.trial_merit);
    d["gain_ratio"] = to_numpy(log.gain_ratio);
    d["damping"] = to_numpy(log.damping);
    d["step_norm"] = to_numpy(log.step_norm);
    d["failed_rays"] = to_numpy(log.failed_rays);
    d["hard_failed_rays"] = to_numpy(log.hard_failed_rays);
    d["status"] = nb::cast(log.status);
    return d;
}

nb::dict run_dict(const PyOptimizeResult& r, std::size_t i) {
    const LmRunResult& run = r.result.runs[i];
    const std::size_t p = r.variables.size();
    nb::dict d;
    d["run"] = static_cast<int>(i);
    d["status"] = to_string(run.status);
    d["converged"] = lm_converged(run.status);
    d["iterations"] = run.iterations;
    d["accepted"] = run.accepted;
    d["jacobians"] = run.jacobians;
    d["start_merit"] = run.history.empty() ? kNaN : run.history.front();
    d["merit"] = run.merit;
    d["failed_rays"] = run.failed_rays;
    d["hard_failed_rays"] = run.hard_failed_rays;
    d["penalty_rounds"] = run.penalty_rounds;
    d["fail_weight"] = r.hard_failure ? kNaN : run.fail_weight;
    d["start"] =
        to_numpy(std::vector<double>(r.starts.begin() + static_cast<std::ptrdiff_t>(i * p),
                                     r.starts.begin() + static_cast<std::ptrdiff_t>((i + 1) * p)));
    d["x"] = to_numpy(run.x);
    d["history"] = to_numpy(run.history);
    return d;
}

} // namespace

void bind_optimize(nb::module_& m) {
    nb::class_<PyOptimizeResult>(
        m, "OptimizeResult",
        "Ergebnis von optimize(): optimiertes System (bester Lauf), Status, Merit, Verlauf und "
        "Iterationsprotokoll. Arrays sind NumPy-Kopien.")
        .def_ro("system", &PyOptimizeResult::system, "Optimiertes System (bester Lauf).")
        .def_prop_ro("backend", [](const PyOptimizeResult& r) { return r.gpu ? "gpu" : "cpu"; })
        .def_prop_ro(
            "jacobian",
            [](const PyOptimizeResult& r) { return std::string(to_string(r.jacobian)); },
            "Verfahren der Jacobi-Matrix: \"fd\" oder \"adjoint\".")
        .def_prop_ro(
            "status",
            [](const PyOptimizeResult& r) { return std::string(to_string(r.best_run().status)); })
        .def_prop_ro("converged",
                     [](const PyOptimizeResult& r) { return lm_converged(r.best_run().status); })
        .def_ro("best_run", &PyOptimizeResult::best)
        .def_prop_ro(
            "merit", [](const PyOptimizeResult& r) { return r.best_run().merit; },
            "Merit des Ergebnisses [Wellen²], einschließlich Straf- und Zusatzresiduen.")
        .def_prop_ro(
            "start_merit",
            [](const PyOptimizeResult& r) { return r.result.runs.front().history.front(); },
            "Merit am Startpunkt der Datei (Lauf 0) [Wellen²].")
        .def_prop_ro("iterations",
                     [](const PyOptimizeResult& r) { return r.best_run().iterations; })
        .def_prop_ro("accepted", [](const PyOptimizeResult& r) { return r.best_run().accepted; })
        .def_prop_ro("jacobians", [](const PyOptimizeResult& r) { return r.best_run().jacobians; })
        .def_prop_ro("failed_rays",
                     [](const PyOptimizeResult& r) { return r.best_run().failed_rays; })
        .def_prop_ro("hard_failed_rays",
                     [](const PyOptimizeResult& r) { return r.best_run().hard_failed_rays; })
        .def_ro("start_failed_rays", &PyOptimizeResult::start_failed)
        .def_ro("start_hard_failed_rays", &PyOptimizeResult::start_hard_failed)
        .def_prop_ro("penalty_rounds",
                     [](const PyOptimizeResult& r) { return r.best_run().penalty_rounds; })
        .def_prop_ro(
            "fail_weight",
            [](const PyOptimizeResult& r) {
                return r.hard_failure ? kNaN : r.best_run().fail_weight;
            },
            "w_fail der letzten Runde [Wellen²]; NaN bei hartem Versagen.")
        .def_prop_ro("damping", [](const PyOptimizeResult& r) { return r.best_run().damping; })
        .def_prop_ro("gradient_norm",
                     [](const PyOptimizeResult& r) { return r.best_run().gradient_norm; })
        .def_prop_ro("model_decrease",
                     [](const PyOptimizeResult& r) { return r.best_run().model_decrease; })
        .def_prop_ro(
            "history", [](const PyOptimizeResult& r) { return to_numpy(r.best_run().history); },
            "Merit des besten Laufs: [0] am Start, [k] nach Zyklus k (mehrere Runden des "
            "Strafverfahrens aneinandergehängt).")
        .def_prop_ro(
            "x", [](const PyOptimizeResult& r) { return to_numpy(r.best_run().x); },
            "Variablenvektor des Ergebnisses.")
        .def_prop_ro(
            "x0", [](const PyOptimizeResult& r) { return to_numpy(r.x0); },
            "Variablenvektor des Startsystems.")
        .def_prop_ro("variable_names",
                     [](const PyOptimizeResult& r) {
                         std::vector<std::string> names;
                         for (const Variable& v : r.variables) {
                             names.push_back(v.name);
                         }
                         return names;
                     })
        .def_prop_ro(
            "bounds",
            [](const PyOptimizeResult& r) {
                std::vector<double> b;
                for (const Variable& v : r.variables) {
                    b.push_back(v.lower);
                    b.push_back(v.upper);
                }
                return to_numpy(std::move(b), {r.variables.size(), 2});
            },
            "Schranken (P, 2): untere, obere (±inf offen).")
        .def_prop_ro(
            "runs",
            [](const PyOptimizeResult& r) {
                nb::list out;
                for (std::size_t i = 0; i < r.result.runs.size(); ++i) {
                    out.append(run_dict(r, i));
                }
                return out;
            },
            "Alle Läufe als Liste von dict (status, merit, start, x, history, …).")
        .def_prop_ro(
            "log", [](const PyOptimizeResult& r) { return log_dict(r.log); },
            "Iterationsprotokoll als dict von Arrays (ein Eintrag pro Zyklus und Lauf, "
            "wie das Feld iterations von `mikroptim optimize --json`).")
        .def_ro("warnings", &PyOptimizeResult::warnings);

    m.def("optimize", &optimize_system, nb::arg("system"), nb::arg("max_iterations") = 100,
          nb::arg("rings") = 10, nb::arg("spokes") = 8, nb::arg("runs") = 1,
          nb::arg("perturbation") = 0.05, nb::arg("seed") = 0, nb::arg("fail_weight") = nb::none(),
          nb::arg("hard_failure") = false, nb::arg("gpu") = false, nb::arg("device") = 0,
          nb::arg("jacobian") = "fd", nb::arg("callback") = nb::none(),
          "Optimiert die Variablen des Systems wie `mikroptim optimize` (gleiche Merit, "
          "jacobian \"fd\" oder \"adjoint\" wie --jacobian, "
          "Nebenbedingungen, Strafen, Multistart und Auswahl des besten Laufs). callback(info: "
          "dict) wird nach jedem Zyklus aufgerufen; eine Ausnahme darin bricht ab.");
}

} // namespace mko::py
