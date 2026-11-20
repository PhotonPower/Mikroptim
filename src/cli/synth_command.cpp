// Unterbefehl `mikroptim synth` (M6-T3): Startdatei laden, Homotopie-Controller mit dem
// optischen Modell (CPU oder --gpu über optimize_lm_gpu), Protokoll als Text oder JSON, optional
// als Datei, Ergebnis als Systemdatei. Mit --beam B die Strahlsuche (M6-T4) mit Stammbaum der
// Kandidaten (Text, JSON, optional Graphviz-DOT). Formate: docs/cli.md.

#include "cli/synth_command.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/optim/augmented_lagrangian.hpp"
#include "mikroptim/optim/glass_snap.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/synth/beam_search.hpp"
#include "mikroptim/synth/ghost_return.hpp"
#include "mikroptim/synth/homotopy.hpp"
#include "mikroptim/synth/optical_homotopy.hpp"
#include "mikroptim/tolerance/tolerance_aware.hpp"
#include "mikroptim/version.hpp"

#ifdef MIKROPTIM_HAS_CUDA
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/levenberg_marquardt.hpp"
#include "mikroptim/cuda/tolerance.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#endif

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace mko::cli {

namespace {

namespace fs = std::filesystem;
using Json = nlohmann::ordered_json;

constexpr int kDefaultMaxIterations = 500;

// Stagnationsabbruch der Synthese (M11-T1c): Standard aus. Als Standard 20,1e-3 (#65) brach er
// auf der Zielspezifikation die Aperturstufe bei NA 0,37 ab, mit 20,1e-5 fügte der Controller
// unkontrolliert Elemente ein (docs/optimierung-erweiterungen.md, Abschnitt 1).
constexpr int kDefaultStallWindow = 0;
constexpr double kDefaultStallTolerance = 1e-3;

struct SynthArgs {
    std::string file;
    std::string output;
    std::string log;
    bool json = false;
    bool quiet = false;
    bool help = false;
    bool gpu = false;
    bool have_device = false;
    int device = 0;
    std::optional<double> target_na;
    std::optional<double> start_field;
    std::optional<double> target_field;
    double rms_target = 1.0 / 14.0;
    double merit_factor = 1.0;
    double extrema_beta = 0.0; // M11-T1b
    bool predictor = false;    // M11-T3
    // Gemeinsame Apertur-/Feld-Homotopie (M6-T8): Exponent p des Feldes, Schrittweite in t
    bool joint = false;
    double joint_exponent = 1.0;
    HomotopyStepControl joint_step = HomotopyOptions{}.joint_step;
    // Feldregel (M6-T8): Operator A in der Feldlinsenlage bei dominierenden Feldfehlern
    bool field_rule = false;
    double field_ratio = OpticalHomotopyOptions{}.field_ratio;
    double field_min_rms = OpticalHomotopyOptions{}.field_min_rms;
    std::string field_glass;
    int insertion_improve = 0; // M6-T8: Einfügen nur mit Verbesserung, höchstens N Varianten
    // Rückbau (M6-T9): nach den Phasen bzw. am Elementbudget
    bool prune = false;
    std::optional<int> element_budget;
    bool have_prune_detail = false;
    int prune_stages = HomotopyOptions{}.prune_stages;
    int prune_attempts = HomotopyOptions{}.prune_attempts;
    double prune_factor = HomotopyOptions{}.prune_merit_factor;
    double prune_tolerance = HomotopyOptions{}.prune_tolerance;
    int prune_min_elements = HomotopyOptions{}.prune_min_elements;
    bool have_max_elements = false;
    // M6-T10: Rückkehr vom realen System zu Geisterplatten vor dem Lauf
    bool start_from_real = false;
    std::vector<std::string> ghost_glasses;
    std::optional<int> ghost_order;
    std::optional<double> ghost_weight;
    std::optional<std::vector<GhostReturnLocation>> ghost_locations;
    std::optional<double> sine_condition; // M11-T4: Gewicht des Eintrags sine_condition
    std::optional<ReferenceSphereKind> reference_sphere; // ADR-0017: setzt reference_sphere
    bool no_reference_warning = false;                   // --no-reference-warning
    // Augmented Lagrangian (M11-T2): Läufe K, Toleranz T, Faktor S auf die Gewichte (K = 0: aus)
    int al_rounds = 0;
    double al_tolerance = kDefaultShiftTolerance;
    // Neustart: Verschiebungen θ aus dem Protokoll bzw. der JSON-Ausgabe eines früheren Laufs
    std::string shifts_from;
    double al_scale = 1.0;
    bool split_preselect = false;     // M11-T5: Zielglas von Operator B aus der Glasvorauswahl
    bool split_start_target = false;  // und das freigegebene Glas dort starten
    bool split_achromat = false;      // M11-T5e: Achromat-Start des geteilten Glieds
    bool color_rule = false;          // M11-T5c
    int spectrum_stages = 0;          // M11-T5d: Spektrum-Homotopie (0: aus)
    bool spectrum_continuous = false; // M11-T5f: stetige Spektrumbreite
    double color_ratio = 2.0;
    int extrema_stages = HomotopyOptions{}.extrema_stages;
    double extrema_factor = HomotopyOptions{}.extrema_factor;
    int max_elements = HomotopyOptions{}.max_elements;
    int max_steps = HomotopyOptions{}.max_steps;
    int max_iterations = kDefaultMaxIterations;
    // Stagnation des LM (M11-T1c): Standard der Synthese 20 Zyklen, 1e-3 (3-mal weniger
    // LM-Zyklen am Abnahmesystem von M6-T3 bei gleicher Güte); --stall 0 schaltet ab.
    int stall_window = kDefaultStallWindow;
    double stall_tolerance = kDefaultStallTolerance;
    int rings = 10;
    int spokes = 8;
    // Verfahren der Jacobi-Matrix der LM-Läufe (M5-T3): --jacobian fd|adjoint
    JacobianMethod jacobian = LmOptions{}.jacobian_method;
    HomotopyStepControl step{};
    OpticalHomotopyOptions optical{};
    // Strahlsuche (M6-T4)
    std::optional<int> beam;
    std::optional<std::uint64_t> seed;
    std::optional<int> alternatives;
    std::optional<int> starts;
    std::optional<double> start_perturbation;
    bool prefilter = false;
    std::optional<double> dedup;
    std::string dot;
    // Toleranzbewusste Synthese (M11-T6b): alpha > 0 schaltet ein (Standard aus); Toleranzsatz
    // ist der Mikroptim-Standardsatz der realisierten Beschreibung an jedem Punkt
    double tol_alpha = 0.0;
    int tol_samples = 0;
    std::uint64_t tol_seed = 0;
    // Standard remove (Entscheidung des Maintainers 2026-11-19, ADR-0018, Nachtrag)
    ToleranceAwareTilt tol_tilt = ToleranceAwareTilt::Remove;
    bool have_tol_detail = false;
    bool have_tol_seed = false;
};

std::string_view next_value(std::span<const std::string_view> args, std::size_t& i,
                            std::string_view option) {
    if (i + 1 >= args.size()) {
        throw UsageError(std::string(option) + " erwartet einen Wert");
    }
    return args[++i];
}

double positive(std::string_view option, std::string_view v) {
    const double x = parse_double(option, v);
    if (!(x > 0.0)) {
        throw UsageError(std::string(option) + " muss > 0 sein");
    }
    return x;
}

double non_negative(std::string_view option, std::string_view v) {
    const double x = parse_double(option, v);
    if (!(x >= 0.0)) {
        throw UsageError(std::string(option) + " muss >= 0 sein");
    }
    return x;
}

// D0,DMIN,DMAX einer Schrittweitensteuerung (--step, --joint-step); grow und shrink bleiben.
HomotopyStepControl parse_step_control(std::string_view option, std::string_view value) {
    const std::string v(value);
    const std::size_t c1 = v.find(',');
    const std::size_t c2 = c1 == std::string::npos ? c1 : v.find(',', c1 + 1);
    if (c2 == std::string::npos) {
        throw UsageError(std::string(option) + " erwartet D0,DMIN,DMAX");
    }
    HomotopyStepControl c;
    c.initial = positive(option, std::string_view(v).substr(0, c1));
    c.min = positive(option, std::string_view(v).substr(c1 + 1, c2 - c1 - 1));
    c.max = positive(option, std::string_view(v).substr(c2 + 1));
    return c;
}

SynthArgs parse_args(std::span<const std::string_view> args) {
    SynthArgs a;
    bool have_file = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--help" || arg == "-h") {
            a.help = true;
        } else if (arg == "--json") {
            a.json = true;
        } else if (arg == "--quiet" || arg == "-q") {
            a.quiet = true;
        } else if (arg == "--gpu") {
            a.gpu = true;
        } else if (arg == "--device") {
            a.device = parse_int(arg, next_value(args, i, arg));
            if (a.device < 0) {
                throw UsageError("--device muss >= 0 sein");
            }
            a.have_device = true;
        } else if (arg == "-o" || arg == "--output") {
            a.output = std::string(next_value(args, i, arg));
        } else if (arg == "--log") {
            a.log = std::string(next_value(args, i, arg));
        } else if (arg == "--target-na") {
            a.target_na = positive(arg, next_value(args, i, arg));
        } else if (arg == "--start-field") {
            a.start_field = non_negative(arg, next_value(args, i, arg));
        } else if (arg == "--target-field") {
            a.target_field = non_negative(arg, next_value(args, i, arg));
        } else if (arg == "--rms-target") {
            a.rms_target = positive(arg, next_value(args, i, arg));
        } else if (arg == "--merit-factor") {
            a.merit_factor = positive(arg, next_value(args, i, arg));
        } else if (arg == "--spectrum-continuous") {
            a.spectrum_continuous = true;
        } else if (arg == "--spectrum-stages") {
            a.spectrum_stages = parse_int(arg, next_value(args, i, arg));
            if (a.spectrum_stages < 1) {
                throw UsageError("--spectrum-stages muss >= 1 sein");
            }
        } else if (arg == "--color-rule") {
            a.color_rule = true;
        } else if (arg == "--color-ratio") {
            a.color_rule = true;
            a.color_ratio = positive(arg, next_value(args, i, arg));
        } else if (arg == "--split-preselect") {
            a.split_preselect = true;
        } else if (arg == "--split-achromat") {
            a.split_preselect = true;
            a.split_achromat = true;
        } else if (arg == "--split-start-target") {
            a.split_preselect = true;
            a.split_start_target = true;
        } else if (arg == "--predictor") {
            a.predictor = true;
        } else if (arg == "--joint-aperture-field") {
            a.joint = true;
        } else if (arg == "--joint-field-exponent") {
            a.joint = true;
            a.joint_exponent = positive(arg, next_value(args, i, arg));
        } else if (arg == "--joint-step") {
            a.joint = true;
            a.joint_step = parse_step_control(arg, next_value(args, i, arg));
        } else if (arg == "--insertion-improve") {
            a.insertion_improve = parse_int(arg, next_value(args, i, arg));
            if (a.insertion_improve < 1) {
                throw UsageError("--insertion-improve muss >= 1 sein");
            }
        } else if (arg == "--prune") {
            a.prune = true;
        } else if (arg == "--element-budget") {
            a.element_budget = parse_int(arg, next_value(args, i, arg));
            if (*a.element_budget < 1) {
                throw UsageError("--element-budget muss >= 1 sein");
            }
        } else if (arg == "--prune-stages") {
            a.have_prune_detail = true;
            a.prune_stages = parse_int(arg, next_value(args, i, arg));
            if (a.prune_stages < 1) {
                throw UsageError("--prune-stages muss >= 1 sein");
            }
        } else if (arg == "--prune-attempts") {
            a.have_prune_detail = true;
            a.prune_attempts = parse_int(arg, next_value(args, i, arg));
            if (a.prune_attempts < 1) {
                throw UsageError("--prune-attempts muss >= 1 sein");
            }
        } else if (arg == "--prune-factor") {
            a.have_prune_detail = true;
            a.prune_factor = positive(arg, next_value(args, i, arg));
            if (a.prune_factor > 1.0) {
                throw UsageError("--prune-factor muss in (0, 1] liegen");
            }
        } else if (arg == "--prune-tolerance") {
            a.have_prune_detail = true;
            a.prune_tolerance = non_negative(arg, next_value(args, i, arg));
        } else if (arg == "--prune-min-elements") {
            a.have_prune_detail = true;
            a.prune_min_elements = parse_int(arg, next_value(args, i, arg));
            if (a.prune_min_elements < 0) {
                throw UsageError("--prune-min-elements muss >= 0 sein");
            }
        } else if (arg == "--start-from-real") {
            a.start_from_real = true;
        } else if (arg == "--ghost-glass") {
            a.ghost_glasses.emplace_back(next_value(args, i, arg));
            if (a.ghost_glasses.back().empty() || a.ghost_glasses.back() == "AIR") {
                throw UsageError("--ghost-glass erwartet einen Katalognamen");
            }
        } else if (arg == "--ghost-order") {
            a.ghost_order = parse_int(arg, next_value(args, i, arg));
            if (*a.ghost_order < 1) {
                throw UsageError("--ghost-order muss >= 1 sein");
            }
        } else if (arg == "--ghost-weight") {
            a.ghost_weight = positive(arg, next_value(args, i, arg));
        } else if (arg == "--ghost-positions") {
            try {
                a.ghost_locations = parse_ghost_return_locations(next_value(args, i, arg));
            } catch (const std::invalid_argument& e) {
                throw UsageError(std::string("--ghost-positions: ") + e.what());
            }
        } else if (arg == "--field-rule") {
            a.field_rule = true;
        } else if (arg == "--field-ratio") {
            a.field_rule = true;
            a.field_ratio = positive(arg, next_value(args, i, arg));
        } else if (arg == "--field-min-rms") {
            a.field_rule = true;
            a.field_min_rms = non_negative(arg, next_value(args, i, arg));
        } else if (arg == "--field-glass") {
            a.field_rule = true;
            a.field_glass = std::string(next_value(args, i, arg));
            if (a.field_glass.empty() || a.field_glass == "AIR") {
                throw UsageError("--field-glass erwartet einen Katalognamen");
            }
        } else if (arg == "--sine-condition") {
            a.sine_condition = parse_sine_condition_weight(arg, next_value(args, i, arg));
        } else if (arg == "--reference-sphere") {
            a.reference_sphere = parse_reference_sphere(arg, next_value(args, i, arg));
        } else if (arg == kNoReferenceWarningOption) {
            a.no_reference_warning = true;
        } else if (arg == "--extrema-beta") {
            a.extrema_beta = non_negative(arg, next_value(args, i, arg));
        } else if (arg == "--extrema-stages") {
            a.extrema_stages = parse_int(arg, next_value(args, i, arg));
            if (a.extrema_stages < 0) {
                throw UsageError("--extrema-stages muss >= 0 sein");
            }
        } else if (arg == "--extrema-factor") {
            a.extrema_factor = positive(arg, next_value(args, i, arg));
            if (!(a.extrema_factor > 1.0)) {
                throw UsageError("--extrema-factor muss > 1 sein");
            }
        } else if (arg == "--max-elements") {
            a.have_max_elements = true;
            a.max_elements = parse_int(arg, next_value(args, i, arg));
            if (a.max_elements < 0) {
                throw UsageError("--max-elements muss >= 0 sein");
            }
        } else if (arg == "--max-steps") {
            a.max_steps = parse_int(arg, next_value(args, i, arg));
            if (a.max_steps < 1) {
                throw UsageError("--max-steps muss >= 1 sein");
            }
        } else if (arg == "--shifts-from") {
            a.shifts_from = std::string(next_value(args, i, arg));
            if (a.shifts_from.empty()) {
                throw UsageError("--shifts-from erwartet einen Dateinamen");
            }
        } else if (arg == "--augmented-lagrangian") {
            // K[,T[,S]]: Augmented Lagrangian statt Penalty-Fortsetzung (M11-T2)
            const std::string v(next_value(args, i, arg));
            const std::size_t c1 = v.find(',');
            a.al_rounds = parse_int(arg, std::string_view(v).substr(0, c1));
            if (c1 != std::string::npos) {
                const std::size_t c2 = v.find(',', c1 + 1);
                a.al_tolerance = parse_double(
                    arg, std::string_view(v).substr(
                             c1 + 1, c2 == std::string::npos ? std::string::npos : c2 - c1 - 1));
                if (c2 != std::string::npos) {
                    a.al_scale = parse_double(arg, std::string_view(v).substr(c2 + 1));
                }
            }
            if (a.al_rounds < 0 || !(a.al_tolerance >= 0.0) || !std::isfinite(a.al_tolerance) ||
                !(a.al_scale > 0.0) || !std::isfinite(a.al_scale)) {
                throw UsageError("--augmented-lagrangian: K >= 0, T >= 0 und S > 0 erwartet");
            }
        } else if (arg == "--stall") {
            // W oder W,T: Stagnation über W Zyklen mit relativer Abnahme ≤ T (M11-T1c)
            const std::string v(next_value(args, i, arg));
            const std::size_t comma = v.find(',');
            a.stall_window = parse_int(arg, std::string_view(v).substr(0, comma));
            if (comma != std::string::npos) {
                a.stall_tolerance = parse_double(arg, std::string_view(v).substr(comma + 1));
            }
            if (a.stall_window < 0 || !(a.stall_tolerance >= 0.0)) {
                throw UsageError("--stall: W >= 0 und T >= 0 erwartet");
            }
        } else if (arg == "--max-iterations") {
            a.max_iterations = parse_int(arg, next_value(args, i, arg));
            if (a.max_iterations < 1) {
                throw UsageError("--max-iterations muss >= 1 sein");
            }
        } else if (arg == "--rings") {
            a.rings = parse_int(arg, next_value(args, i, arg));
        } else if (arg == "--spokes") {
            a.spokes = parse_int(arg, next_value(args, i, arg));
        } else if (arg == "--step") {
            a.step = parse_step_control(arg, next_value(args, i, arg));
        } else if (arg == "--element-thickness") {
            a.optical.element_thickness = positive(arg, next_value(args, i, arg));
        } else if (arg == "--element-max-thickness") {
            a.optical.element_max_thickness = non_negative(arg, next_value(args, i, arg));
        } else if (arg == "--element-stages") {
            a.optical.element_stages = parse_int(arg, next_value(args, i, arg));
        } else if (arg == "--min-air-gap") {
            a.optical.min_air_gap = non_negative(arg, next_value(args, i, arg));
        } else if (arg == "--saddle-delta") {
            a.optical.saddle_delta = non_negative(arg, next_value(args, i, arg));
        } else if (arg == "--jacobian") {
            const std::string_view v = next_value(args, i, arg);
            const std::optional<JacobianMethod> m = jacobian_method_from_string(v);
            if (!m) {
                throw UsageError("--jacobian erwartet fd oder adjoint, nicht '" + std::string(v) +
                                 "'");
            }
            a.jacobian = *m;
        } else if (arg == "--shrink-short-gaps") {
            a.optical.meniscus.shrink_short_gap = true;
        } else if (arg == "--curvature-bound") {
            const double c = positive(arg, next_value(args, i, arg));
            a.optical.meniscus.curvature_variables.min = -c;
            a.optical.meniscus.curvature_variables.max = c;
        } else if (arg == "--beam") {
            a.beam = parse_int(arg, next_value(args, i, arg));
            if (*a.beam < 1) {
                throw UsageError("--beam muss >= 1 sein");
            }
        } else if (arg == "--seed") {
            a.seed = parse_uint64(arg, next_value(args, i, arg));
        } else if (arg == "--alternatives") {
            a.alternatives = parse_int(arg, next_value(args, i, arg));
            if (*a.alternatives < 1) {
                throw UsageError("--alternatives muss >= 1 sein");
            }
        } else if (arg == "--starts") {
            a.starts = parse_int(arg, next_value(args, i, arg));
            if (*a.starts < 0) {
                throw UsageError("--starts muss >= 0 sein");
            }
        } else if (arg == "--start-perturbation") {
            a.start_perturbation = positive(arg, next_value(args, i, arg));
        } else if (arg == "--prefilter") {
            a.prefilter = true;
        } else if (arg == "--dedup") {
            a.dedup = parse_double(arg, next_value(args, i, arg));
        } else if (arg == "--tolerance-aware") {
            a.tol_alpha = parse_double(arg, next_value(args, i, arg));
            if (!(a.tol_alpha > 0.0 && a.tol_alpha <= 1.0)) {
                throw UsageError("--tolerance-aware: Gewicht alpha in (0, 1] erwartet");
            }
        } else if (arg == "--tolerance-samples") {
            a.tol_samples = parse_int(arg, next_value(args, i, arg));
            if (a.tol_samples < 0) {
                throw UsageError("--tolerance-samples muss >= 0 sein");
            }
            a.have_tol_detail = true;
        } else if (arg == "--tolerance-seed") {
            a.tol_seed = parse_uint64(arg, next_value(args, i, arg));
            a.have_tol_detail = true;
            a.have_tol_seed = true;
        } else if (arg == "--tolerance-tilt") {
            const std::string_view v = next_value(args, i, arg);
            const std::optional<ToleranceAwareTilt> t = tolerance_aware_tilt_from_string(v);
            if (!t) {
                throw UsageError("--tolerance-tilt erwartet keep oder remove, nicht '" +
                                 std::string(v) + "'");
            }
            a.tol_tilt = *t;
            a.have_tol_detail = true;
        } else if (arg == "--dot") {
            a.dot = std::string(next_value(args, i, arg));
        } else if (arg.size() > 1 && arg.front() == '-') {
            throw UsageError("unbekannte Option " + std::string(arg));
        } else if (have_file) {
            throw UsageError("mehr als eine Systemdatei angegeben");
        } else {
            a.file = std::string(arg);
            have_file = true;
        }
    }
    if (a.help) {
        return a;
    }
    if (!have_file) {
        throw UsageError("keine Systemdatei angegeben");
    }
    if (a.have_device && !a.gpu) {
        throw UsageError("--device verlangt --gpu");
    }
    if (a.spectrum_continuous && a.spectrum_stages > 0) {
        throw UsageError("--spectrum-continuous und --spectrum-stages schließen sich aus");
    }
    if (a.beam && a.predictor) {
        throw UsageError("--predictor gibt es nur ohne --beam");
    }
    if (a.beam && a.joint) {
        throw UsageError("--joint-aperture-field gibt es nur ohne --beam");
    }
    if (a.beam && a.insertion_improve > 0) {
        throw UsageError("--insertion-improve gibt es nur ohne --beam");
    }
    if (a.beam && (a.prune || a.element_budget)) {
        throw UsageError("--prune und --element-budget gibt es nur ohne --beam");
    }
    if (a.element_budget && a.have_max_elements) {
        throw UsageError(
            "--element-budget setzt die höchste Elementzahl; nicht mit --max-elements");
    }
    if (a.element_budget) {
        a.max_elements = *a.element_budget;
    }
    if (a.have_prune_detail && !a.prune && !a.element_budget) {
        throw UsageError("--prune-stages, --prune-attempts, --prune-factor, --prune-tolerance und "
                         "--prune-min-elements verlangen --prune oder --element-budget");
    }
    if ((!a.ghost_glasses.empty() || a.ghost_order || a.ghost_weight || a.ghost_locations) &&
        !a.start_from_real) {
        throw UsageError("--ghost-glass, --ghost-order, --ghost-weight und --ghost-positions "
                         "verlangen --start-from-real");
    }
    if (a.start_from_real && a.ghost_glasses.empty()) {
        throw UsageError("--start-from-real braucht mindestens ein --ghost-glass");
    }
    if (a.beam && a.al_rounds > 0) {
        throw UsageError("--augmented-lagrangian gibt es nur ohne --beam");
    }
    if (!a.shifts_from.empty() && a.al_rounds == 0) {
        throw UsageError("--shifts-from verlangt --augmented-lagrangian K mit K >= 1");
    }
    if (!a.beam && (a.seed || a.alternatives || a.starts || a.start_perturbation || a.prefilter ||
                    a.dedup || !a.dot.empty())) {
        throw UsageError("--seed, --alternatives, --starts, --start-perturbation, --prefilter, "
                         "--dedup und --dot verlangen --beam");
    }
    if (a.have_tol_detail && !(a.tol_alpha > 0.0)) {
        throw UsageError("--tolerance-samples, --tolerance-seed und --tolerance-tilt verlangen "
                         "--tolerance-aware");
    }
    if (a.have_tol_seed && a.tol_samples == 0) {
        throw UsageError("--tolerance-seed verlangt --tolerance-samples S mit S >= 1");
    }
    for (const std::string* f : {&a.output, &a.log, &a.dot}) {
        if (!f->empty() && same_file(fs::path(*f), fs::path(a.file))) {
            throw UsageError("Ausgabedatei " + *f +
                             " ist die Eingabedatei; die Eingabe wird nie überschrieben");
        }
    }
    return a;
}

Json point_short(const HomotopyPoint& p) {
    return Json{{"aperture", p.aperture}, {"field", p.field}, {"spectrum", p.spectrum}};
}

} // namespace

void print_synth_usage(std::ostream& out) {
    out << "Aufruf: mikroptim synth <start.json> --target-na NA [-o <ergebnis.json>] [--json]\n"
           "                        [--log <protokoll.json>] [--quiet] [--gpu [--device N]]\n"
           "                        [--beam B [--seed N] [--dot <stammbaum.dot>] ...]\n"
           "Homotopie-Controller der Systemsynthese (docs/optimierung.md, Abschnitt 7): öffnet\n"
           "die Apertur (und den Feldanteil) schrittweise mit Schrittweitensteuerung, fügt bei\n"
           "Bedarf (Diagnose der Geisterplatten) Elemente ein und prüft am Ende das Beugungsziel\n"
           "am realen System.\n"
           "  --target-na NA            numerische Apertur am Ende (Standard: wie der Start)\n"
           "  --start-field F           Feldanteil am Start (Standard 1)\n"
           "  --target-field F          Feldanteil am Ende (Standard: wie der Start)\n"
           "  --rms-target W            Beugungsziel [Wellen] pro Feld und Wellenlänge (1/14)\n"
           "  --merit-factor K          Schwelle F_max = (K · Ziel)² (Standard 1)\n"
           "  --predictor               Start jedes Apertur-/Feldschritts aus der Sekante der\n"
           "                            beiden letzten Zustände (M11-T3; nicht mit --beam)\n"
           "  --joint-aperture-field    Apertur und Feldanteil gemeinsam über t = 0 … 1 öffnen:\n"
           "                            NA(t) linear, Feld(t) = F0 + t^P (F1 - F0) (M6-T8;\n"
           "                            nicht mit --beam; Standard aus)\n"
           "  --joint-field-exponent P  Exponent P des Feldes (Standard 1: Diagonale)\n"
           "  --joint-step D0,DMIN,DMAX Schrittweite in t (Standard 0.03125,0.00390625,0.25)\n"
           "  --insertion-improve N     Einfügen in den stetigen Phasen nur übernehmen, wenn F_B\n"
           "                            danach kleiner ist; sonst bis zu N Varianten (M6-T8;\n"
           "                            nicht mit --beam; Standard aus)\n"
           "  --prune                   Rückbau: nach jeder Phase mit Einfügungen und vor dem\n"
           "                            Abschluss Elemente über die Rückbau-Homotopie entfernen,\n"
           "                            solange F_B <= Faktor x F_max bleibt (M6-T9; nicht mit\n"
           "                            --beam; Standard aus)\n"
           "  --element-budget N        höchstens N Glaselemente; verlangt ein Einfügen mehr,\n"
           "                            vorher ein Rückbau-Versuch (M6-T9; statt --max-elements)\n"
           "  --prune-stages K          Stufen der Rückbau-Homotopie (Standard 3)\n"
           "  --prune-attempts N        Kandidaten pro Rückbau-Runde (Standard 4)\n"
           "  --prune-factor F          Annahme bei F_B <= F x F_max, 0 < F <= 1 (Standard 1)\n"
           "  --prune-tolerance E       zusätzlich F_B <= (1 + E) x F_B vorher (Standard aus)\n"
           "  --prune-min-elements N    nie unter N Glaselemente (Standard 1)\n"
           "  --start-from-real         Rückkehr zu Geisterplatten vor dem Lauf (M6-T10): die\n"
           "                            Startdatei ohne Platten bekommt Wörterbücher mit\n"
           "                            Koeffizienten 0 (Merit unverändert) in der Mitte der\n"
           "                            Luftspalte aus --ghost-positions; Spalte fest\n"
           "  --ghost-glass GLAS        Glas einer Platte (mehrfach, mindestens einmal)\n"
           "  --ghost-order K           Ordnungen der Platten (Standard 10)\n"
           "  --ghost-weight MU         Gewicht des neuen Eintrags ghost_penalty (Standard 47.2,\n"
           "                            kappa = 1, eps = 1e-6 mm)\n"
           "  --ghost-positions LISTE   Lagen aus stop, aperture, field, front (Standard\n"
           "                            aperture,field)\n"
           "  --field-rule              Feldregel: ohne Geisterbedarf und bei dominierenden\n"
           "                            Feldfehlern Operator A im Luftspalt mit größtem |ȳ/y|\n"
           "                            (M6-T8; Standard aus)\n"
           "  --field-ratio F           Feldfehler dominieren ab RMS Rand >= F x RMS Achse\n"
           "                            (Standard 1.5; setzt --field-rule)\n"
           "  --field-min-rms W         und größter RMS >= W Wellen (Standard 0.05)\n"
           "  --field-glass GLAS        Glas des Meniskus der Feldregel (Standard: Linse vor\n"
           "                            dem Spalt)\n"
           "  --augmented-lagrangian K[,T[,S]]\n"
           "                            einseitige Bedingungen mit Verschiebung theta statt\n"
           "                            Penalty-Fortsetzung: bis K Läufe mit Gewichten x S\n"
           "                            (Standard 1), Abbruch bei max Wurzel(w)|dtheta| <= T\n"
           "                            (1e-6); theta nach jeder Optimierung fortgeschrieben\n"
           "                            (M11-T2; Standard aus; nicht mit --beam)\n"
           "  --shifts-from DATEI       Neustart: theta aus dem Protokoll (--log) bzw. der\n"
           "                            JSON-Ausgabe eines früheren Laufs (Feld\n"
           "                            augmented_lagrangian.shifts); verlangt\n"
           "                            --augmented-lagrangian\n"
           "  --sine-condition W        Verletzung der Sinusbedingung (OSC) als Residuum mit\n"
           "                            Gewicht W (Eintrag sine_condition setzen, M11-T4)\n"
           "  --reference-sphere K      Referenzsphäre der Merit: last_surface,\n"
           "                            image_exit_pupil oder image_infinite (ADR-0017;\n"
           "                            für Geisterplatten empfohlen: image_infinite)\n"
           "  --no-reference-warning    keine Warnung auf stderr, wenn die Datei kein Feld\n"
           "                            reference_sphere hat (dann gilt last_surface)\n"
           "  --spectrum-continuous     stetige Spektrum-Homotopie: Spektrumbreite von 0 (λ_ref) "
           "bis 1\n"
           "                            mit adaptiver Schrittweite (M11-T5f)\n"
           "  --spectrum-stages N       Spektrum-Homotopie: von monochromatisch bei λ_ref in N\n"
           "                            Stufen zum Spektrum der Datei (M11-T5d)\n"
           "  --color-rule              Farbregel: Operator B (Glas aus der Vorauswahl), wenn der\n"
           "                            RMS einer Wellenlänge >= Faktor x RMS bei λ_ref (M11-T5c)\n"
           "  --color-ratio F           Faktor der Farbregel (Standard 2; setzt --color-rule)\n"
           "  --split-preselect         Operator B: Zielglas aus der Glasvorauswahl (M11-T5)\n"
           "  --split-achromat          Operator B mit Vorauswahl und Achromat-Start: Kitt- und "
           "hintere\n"
           "                            Außenkrümmung aus dem dünnen Achromat, Zielglas als Start\n"
           "  --split-start-target      ebenso, und das freigegebene Glas startet beim Zielglas\n"
           "  --extrema-beta B          geglättete Strahl-Extrema mit β = B [1/mm] (0: hart, "
           "Standard);\n"
           "                            am Ende --extrema-stages Stufen (2) mit Faktor\n"
           "                            --extrema-factor (10), danach hart (nur CPU)\n"
           "  --step D0,DMIN,DMAX       Schrittweite der Apertur (Standard 0.02,0.002,0.1)\n"
           "  --max-elements N          höchstens N Glaselemente (Standard 6)\n"
           "  --max-steps N             höchstens N Optimierungen (Standard 400)\n"
           "  --max-iterations N        LM-Zyklen pro Schritt (Standard 500)\n"
           "  --stall W[,T]             LM-Abbruch bei Stagnation: F sank in W Zyklen um <= T·F\n"
           "                            (T Standard 1e-3; Standard aus; empfohlen nur für\n"
           "                            schnelle Erkundungen, z. B. 20,1e-3)\n"
           "  --rings N, --spokes M     Pupillenquadratur (Standard 10 x 8)\n"
           "  --element-thickness T     Mittendicke eines eingefügten Meniskus [mm] (0.5)\n"
           "  --element-max-thickness T danach Variable bis T [mm] (5; <= Dicke: fest)\n"
           "  --element-stages J        Stufen der inneren Homotopie über die Dicke (5)\n"
           "  --min-air-gap G           kleinster Luftspalt hinter dem Meniskus [mm] (0.1)\n"
           "  --saddle-delta D          zwei Starts am Sattelpunkt, c0 +- D [1/mm] (0.002;\n"
           "                            0: ein Zweig)\n"
           "  --curvature-bound C       Schranken +-C [1/mm] der neuen Krümmungsvariablen\n"
           "  --shrink-short-gaps       Operator A in einem Plattenspalt unter 2 x Schranke:\n"
           "                            Schranke der Teilstücke auf (T - Dicke)/2 senken\n"
           "                            (Warnung im Protokoll) statt abzulehnen (M10-T7e)\n"
           "  --jacobian V             Jacobi-Matrix der LM-Läufe: fd (Standard) oder adjoint\n"
           "  --tolerance-aware A       toleranzbewusst (M11-T6b): jeder Schritt minimiert\n"
           "                            (1-A)·F + A·Mittel F(gestört) über feste Störungen des\n"
           "                            Mikroptim-Standardsatzes der Beschreibung am Punkt;\n"
           "                            die Strahlsuche ordnet nach dieser Merit (Standard aus)\n"
           "  --tolerance-samples S     S feste Stichproben statt +-delta (Standard 0)\n"
           "  --tolerance-seed n        Startwert der Stichproben (Standard 0)\n"
           "  --tolerance-tilt K        remove (Standard): gestörte Systeme nach Abzug von\n"
           "                            Kolben und Kippung wie die Tolerierung; keep: volle\n"
           "                            Merit\n"
           "  -o, --output <datei>      Ergebnis am Endpunkt als Systemdatei\n"
           "  --log <datei>             Protokoll als JSON (mikroptim.synth/1) in eine Datei\n"
           "  --json                    Ausgabe als JSON statt Text\n"
           "  --quiet                   ohne Schrittprotokoll (nur Zusammenfassung)\n"
           "  --gpu [--device N]        LM auf einem CUDA-Gerät (optimize_lm_gpu)\n"
           "Strahlsuche (docs/optimierung.md, Abschnitt 8):\n"
           "  --beam B                  bis zu B Kandidaten im Gleichschritt, Stammbaum im\n"
           "                            Protokoll\n"
           "  --alternatives K          Varianten pro Einfügen aus der Diagnose (Lagen, Gläser;\n"
           "                            Standard 1)\n"
           "  --starts N                gestörte Starts pro Zweig (Standard 0)\n"
           "  --start-perturbation S    Störung der neuen Krümmungen [1/mm] (Standard 0.01)\n"
           "  --seed N                  Startwert der gestörten Starts (Standard 0)\n"
           "  --prefilter               FP32-Vorfilter der Starts eines Einfügens\n"
           "  --dedup T                 Duplikat bei Abstand <= T (Standard 1e-4; < 0: aus)\n"
           "  --dot <datei>             Stammbaum als Graphviz-DOT\n";
}

int run_synth_command(std::span<const std::string_view> args, std::ostream& out,
                      std::ostream& err) {
    SynthArgs a;
    try {
        a = parse_args(args);
    } catch (const UsageError& e) {
        err << "mikroptim synth: " << e.what() << '\n';
        print_synth_usage(err);
        return kExitUsage;
    }
    if (a.help) {
        print_synth_usage(out);
        return kExitOk;
    }

#ifdef MIKROPTIM_HAS_CUDA
    std::optional<cuda::TraceEngine> engine;
    if (a.gpu) {
        const int count = cuda::device_count();
        if (count == 0) {
            err << "mikroptim synth: --gpu: kein CUDA-Gerät gefunden (GPU-Backend nicht "
                   "verfügbar)\n";
            return kExitNoGpu;
        }
        if (a.device >= count) {
            err << "mikroptim synth: Fehler: --device " << a.device
                << ": Gerät existiert nicht (siehe mikroptim --devices)\n";
            return kExitError;
        }
        try {
            engine.emplace(a.device);
        } catch (const std::exception& e) {
            err << "mikroptim synth: Fehler: GPU " << a.device << ": " << e.what() << '\n';
            return kExitError;
        }
    }
#else
    if (a.gpu) {
        err << "mikroptim synth: --gpu: ohne CUDA gebaut (MIKROPTIM_ENABLE_CUDA=OFF), "
               "GPU-Backend nicht verfügbar\n";
        return kExitNoGpu;
    }
#endif

    try {
        LoadedSystem start = load_system(fs::path(a.file));
        if (a.sine_condition) {
            apply_sine_condition_option(start.system, *a.sine_condition);
        }
        // Neustart des Augmented Lagrangian (Entscheidung des Maintainers 2026-11-19)
        if (!a.shifts_from.empty()) {
            set_shift_values(start.system, read_shift_file(fs::path(a.shifts_from)));
        }
        if (a.reference_sphere) {
            start.system.reference_sphere = *a.reference_sphere;
        }
        for (const std::string& w : start.warnings) {
            err << "mikroptim synth: Warnung: " << w << '\n';
        }
        warn_missing_reference_sphere(err, "synth", start.system, a.no_reference_warning);
        // M6-T10: Rückkehr zu Geisterplatten (Merit bis auf Rundung gleich)
        std::optional<GhostReturnResult> ghost_return;
        if (a.start_from_real) {
            GhostReturnOptions go;
            for (const std::string& g : a.ghost_glasses) {
                if (!start.catalog.contains(g)) {
                    err << "mikroptim synth: --ghost-glass: Glas \"" << g
                        << "\" nicht in den Katalogen der Startdatei\n";
                    return kExitError;
                }
                MaterialSpec m;
                m.kind = MaterialKind::Catalog;
                m.name = g;
                go.glasses.push_back(std::move(m));
            }
            go.order = a.ghost_order.value_or(go.order);
            go.penalty_weight = a.ghost_weight.value_or(go.penalty_weight);
            if (a.ghost_locations) {
                go.locations = *a.ghost_locations;
            }
            // Normierungsradien für das größte Feld des Laufs (Feldanteil der Startdatei 1)
            go.norm_field_scale =
                std::max(a.start_field.value_or(1.0), a.target_field.value_or(0.0));
            go.rings = a.rings;
            go.spokes = a.spokes;
            ghost_return = return_to_ghost_plates(start, go);
            start = rebuild_loaded_system(start, ghost_return->system);
            for (const std::string& n : ghost_return->notes) {
                err << "mikroptim synth: Hinweis (--start-from-real): " << n << '\n';
            }
        }
        OpticalHomotopyOptions o = a.optical;
        o.lm.max_iterations = a.max_iterations;
        o.split.preselect = a.split_preselect;
        o.color_rule = a.color_rule;
        if (a.spectrum_stages > 0) {
            o.spectra = spectrum_stages(start.system.spectrum, a.spectrum_stages);
        }
        o.spectrum_continuous = a.spectrum_continuous;
        o.color_ratio = a.color_ratio;
        o.field_rule = a.field_rule;
        o.field_ratio = a.field_ratio;
        o.field_min_rms = a.field_min_rms;
        if (!a.field_glass.empty()) {
            if (!start.catalog.contains(a.field_glass)) {
                err << "mikroptim synth: --field-glass: Glas \"" << a.field_glass
                    << "\" nicht in den Katalogen der Startdatei\n";
                return kExitError;
            }
            MaterialSpec g;
            g.kind = MaterialKind::Catalog;
            g.name = a.field_glass;
            o.field_glass = g;
        }
        o.split.start_at_target = a.split_start_target;
        o.split.achromat_start = a.split_achromat;
        o.lm.stall_window = a.stall_window;
        o.lm.stall_tolerance = a.stall_tolerance;
        o.lm.jacobian.quadrature.rings = a.rings;
        o.lm.jacobian.quadrature.spokes = a.spokes;
        o.lm.jacobian_method = a.jacobian;
        Json backend{{"gpu", false}};
#ifdef MIKROPTIM_HAS_CUDA
        if (engine) {
            cuda::TraceEngine* e = &*engine;
            o.optimizer = [e](const VariableSet& set, const SystemBatchHost& base,
                              std::span<const double> x0, const LmOptions& lm) {
                return cuda::optimize_lm_gpu(*e, set, base, x0, lm);
            };
            o.prefilter_fn = [e](const SystemBatchHost& batch, const ResidualSettings& quadrature,
                                 const PrefilterOptions& options, std::span<const double> extra) {
                cuda::PrefilterSettings settings;
                settings.quadrature = quadrature;
                settings.options = options;
                settings.extra_merit = extra;
                return e->prefilter_variants(batch, settings);
            };
            backend = Json{{"gpu", true},
                           {"device", engine->device()},
                           {"device_name", engine->device_name()}};
        }
#endif
        if (a.tol_alpha > 0.0) {
            // Toleranzbewusste Synthese (M11-T6b): jeder Schritt minimiert F_tol mit dem
            // Standardsatz der realisierten Beschreibung, inneres Backend CPU bzw. GPU
            ToleranceAwareSettings ts;
            ts.alpha = a.tol_alpha;
            ts.scheme =
                a.tol_samples > 0 ? ToleranceAwareScheme::Samples : ToleranceAwareScheme::PlusMinus;
            ts.samples = a.tol_samples;
            ts.seed = a.tol_seed;
            ts.tilt = a.tol_tilt;
            ToleranceInnerFactory inner = cpu_tolerance_inner();
#ifdef MIKROPTIM_HAS_CUDA
            if (engine) {
                inner = cuda::gpu_tolerance_inner(*engine);
            }
#endif
            o.system_optimizer = [ts, inner](const SystemDescription& realized,
                                             const VariableSet& set, const SystemBatchHost& base,
                                             std::span<const double> x0, const LmOptions& lm) {
                return optimize_lm_tolerance_aware_standard(realized, ts, inner, set, base, x0, lm);
            };
        }
        if (a.start_perturbation) {
            o.start_perturbation = *a.start_perturbation;
        }
        OpticalHomotopyModel model(start, o);
        HomotopyPoint p0 = model.start_point();
        if (a.start_field) {
            p0.field = *a.start_field;
        }
        HomotopyOptions h;
        h.target_aperture = a.target_na.value_or(0.0);
        h.target_field = a.target_field.value_or(-1.0);
        h.target_rms = a.rms_target;
        if (a.spectrum_stages > 0) {
            h.target_spectrum = a.spectrum_stages;
        }
        if (a.spectrum_continuous) {
            h.target_spectrum_width = 1.0;
        }
        h.merit_factor = a.merit_factor;
        h.extrema_beta = a.extrema_beta;
        h.predictor = a.predictor;
        h.joint_aperture_field = a.joint;
        h.joint_field_exponent = a.joint_exponent;
        h.joint_step = a.joint_step;
        if (a.insertion_improve > 0) {
            h.insertion_must_improve = true;
            h.insertion_attempts = a.insertion_improve;
        }
        h.prune = a.prune;
        h.prune_at_limit = a.element_budget.has_value();
        h.prune_stages = a.prune_stages;
        h.prune_attempts = a.prune_attempts;
        h.prune_merit_factor = a.prune_factor;
        h.prune_tolerance = a.prune_tolerance;
        h.prune_min_elements = a.prune_min_elements;
        h.augmented_lagrangian.rounds = a.al_rounds;
        h.augmented_lagrangian.tolerance = a.al_tolerance;
        h.al_penalty_scale = a.al_scale;
        h.extrema_stages = a.extrema_stages;
        h.extrema_factor = a.extrema_factor;
        h.max_elements = a.max_elements;
        h.max_steps = a.max_steps;
        h.aperture_step.initial = a.step.initial;
        h.aperture_step.min = a.step.min;
        h.aperture_step.max = a.step.max;
        const bool stream = !a.json && !a.quiet;
        if (!a.json) {
            out << "mikroptim synth " << kVersion << ": " << a.file << '\n';
            kv(out, "System", start.system.name);
            kv(out, "Start",
               "NA " + fixed(p0.aperture, 4) + ", Feldanteil " + fixed(p0.field, 3) + ", " +
                   std::to_string(model.elements(model.start_system())) + " Glaselement(e)");
            kv(out, "Ziel",
               "NA " + fixed(a.target_na.value_or(p0.aperture), 4) +
                   ", RMS <= " + fixed(a.rms_target, 5) + " Wellen");
            kv(out, "Schwelle F_max [Wellen²]", sci(homotopy_merit_threshold(h), 6));
            kv(out, "Backend", a.gpu ? "GPU" : "CPU");
            if (a.beam) {
                kv(out, "Strahlsuche",
                   "B = " + std::to_string(*a.beam) + ", Seed " +
                       std::to_string(a.seed.value_or(0)) + ", Varianten " +
                       std::to_string(a.alternatives.value_or(1)) + ", gestörte Starts " +
                       std::to_string(a.starts.value_or(0)) + (a.prefilter ? ", Vorfilter" : ""));
            }
            kv(out, "Jacobi-Matrix", jacobian_text(a.jacobian));
            if (a.joint) {
                kv(out, "Apertur und Feld gemeinsam",
                   "Feld(t) = F0 + t^" + fixed(a.joint_exponent, 3) + " (F1 - F0), Schritt " +
                       sci(a.joint_step.initial, 3) + " / " + sci(a.joint_step.min, 3) + " / " +
                       sci(a.joint_step.max, 3));
            }
            if (a.prune || a.element_budget) {
                kv(out, "Rückbau",
                   std::string(a.prune ? "nach den Phasen und vor dem Abschluss" : "") +
                       (a.prune && a.element_budget ? ", " : "") +
                       (a.element_budget ? "am Elementbudget " + std::to_string(*a.element_budget)
                                         : std::string()) +
                       "; " + std::to_string(a.prune_stages) + " Stufen, " +
                       std::to_string(a.prune_attempts) +
                       " Versuche pro Runde, F_B <= " + fixed(a.prune_factor, 3) + " x F_max" +
                       (a.prune_tolerance >= 0.0
                            ? " und <= (1 + " + fixed(a.prune_tolerance, 4) + ") x F_B vorher"
                            : std::string()) +
                       ", mindestens " + std::to_string(a.prune_min_elements) + " Element(e)");
            }
            if (ghost_return) {
                std::string lagen;
                for (const GhostReturnPosition& p : ghost_return->positions) {
                    lagen += (lagen.empty() ? "" : ", ") + std::string(to_string(p.location)) +
                             " (Spalt " + std::to_string(p.after_surface) + ", r_g " +
                             fixed(p.norm_radius, 3) + " mm)";
                }
                kv(out, "Rückkehr zu Geisterplatten",
                   std::to_string(ghost_return->system.ghost_plates.size()) + " Platten: " + lagen +
                       "; F vorher/nachher " + sci(ghost_return->merit_before, 6) + " / " +
                       sci(ghost_return->merit_after, 6));
            }
            if (a.field_rule) {
                kv(out, "Feldregel",
                   "RMS Rand >= " + fixed(a.field_ratio, 3) +
                       " x Achse, größter RMS >= " + fixed(a.field_min_rms, 4) + " Wellen" +
                       (a.field_glass.empty() ? std::string(", Glas der Linse vor dem Spalt")
                                              : ", Glas " + a.field_glass));
            }
            if (a.tol_alpha > 0.0) {
                kv(out, "Toleranzbewusst",
                   "alpha = " + fixed(a.tol_alpha, 4) + ", " +
                       (a.tol_samples > 0
                            ? std::to_string(a.tol_samples) + " Stichproben, Startwert " +
                                  std::to_string(a.tol_seed)
                            : std::string("+-delta je Zufallsgröße")) +
                       ", Mikroptim-Standardsatz" +
                       (a.tol_tilt == ToleranceAwareTilt::Remove ? ", gestört ohne Kippung" : ""));
            }
            out << '\n';
            if (stream) {
                if (a.beam) {
                    write_beam_step_header(out);
                } else {
                    write_homotopy_step_header(out);
                }
            }
            out.flush();
        }
        // Einzweigig (run_homotopy) oder Strahlsuche (run_beam_search)
        std::optional<HomotopyResult> single;
        std::optional<BeamResult> beam;
        BeamOptions bo;
        if (a.beam) {
            bo.homotopy = h;
            bo.beam_width = *a.beam;
            bo.alternatives = a.alternatives.value_or(1);
            bo.perturbed_starts = a.starts.value_or(0);
            bo.seed = a.seed.value_or(0);
            bo.prefilter = a.prefilter;
            if (a.dedup) {
                bo.dedup_tolerance = *a.dedup;
            }
            std::function<void(const BeamStep&)> on_beam;
            if (stream) {
                on_beam = [&out](const BeamStep& s) {
                    write_beam_step_line(s, out);
                    out.flush();
                };
            }
            beam = run_beam_search(model, model.start_system(), p0, bo, on_beam);
        } else {
            if (stream) {
                h.on_step = [&out](const HomotopyStep& s) {
                    write_homotopy_step_line(s, out);
                    out.flush();
                };
            }
            single = run_homotopy(model, model.start_system(), p0, h);
        }
        const SystemDescription& end_system = beam ? beam->system : single->system;
        const HomotopyPoint& end_point = beam ? beam->point : single->point;
        const double seconds = beam ? beam->seconds : single->seconds;

        if (!a.output.empty()) {
            const fs::path output(a.output);
            std::error_code ec;
            fs::path out_dir = fs::absolute(output, ec).parent_path();
            if (ec) {
                out_dir = output.parent_path();
            }
            SystemDescription file_system = model.realize(end_system, end_point, false);
            file_system.glass_catalogs = catalogs_for(file_system, start.catalog_files, out_dir);
            write_system_file(file_system, output);
        }

        Json options{{"target_na", a.target_na ? Json(*a.target_na) : Json(nullptr)},
                     {"start", point_short(p0)},
                     {"rms_target_waves", a.rms_target},
                     {"merit_factor", a.merit_factor},
                     {"max_elements", a.max_elements},
                     {"max_iterations", a.max_iterations},
                     {"stall", Json{{"window", a.stall_window}, {"tolerance", a.stall_tolerance}}},
                     {"augmented_lagrangian", a.al_rounds > 0 ? Json{{"rounds", a.al_rounds},
                                                                     {"tolerance", a.al_tolerance},
                                                                     {"penalty_scale", a.al_scale}}
                                                              : Json(nullptr)},
                     {"color_rule", a.color_rule ? Json(a.color_ratio) : Json(nullptr)},
                     {"sine_condition", a.sine_condition ? Json(*a.sine_condition) : Json(nullptr)},
                     {"spectrum_stages", a.spectrum_stages},
                     {"spectrum_continuous", a.spectrum_continuous},
                     {"split", Json{{"preselect", a.split_preselect},
                                    {"start_at_target", a.split_start_target},
                                    {"achromat_start", a.split_achromat}}},
                     {"quadrature", Json{{"rings", a.rings}, {"spokes", a.spokes}}},
                     {"jacobian", to_string(a.jacobian)},
                     {"tolerance_aware",
                      a.tol_alpha > 0.0
                          ? Json{{"alpha", a.tol_alpha},
                                 {"scheme", a.tol_samples > 0 ? "samples" : "plus_minus"},
                                 {"samples", a.tol_samples},
                                 {"seed", a.tol_samples > 0 ? Json(a.tol_seed) : Json(nullptr)},
                                 {"set", "default"},
                                 {"tilt", to_string(a.tol_tilt)}}
                          : Json(nullptr)},
                     {"element_thickness", o.element_thickness},
                     {"element_max_thickness", o.element_max_thickness},
                     {"saddle_delta", o.saddle_delta}};
        // M6-T8: nur mit der Option (ohne sie bleibt das Protokoll wie bisher)
        if (a.joint) {
            options["joint_aperture_field"] = Json{{"field_exponent", a.joint_exponent},
                                                   {"step", Json{{"initial", a.joint_step.initial},
                                                                 {"min", a.joint_step.min},
                                                                 {"max", a.joint_step.max}}}};
        }
        if (a.insertion_improve > 0) {
            options["insertion_improve"] = Json{{"attempts", a.insertion_improve}};
        }
        if (a.prune || a.element_budget) {
            // M6-T9: nur mit der Option
            options["prune"] = Json{
                {"after_phases", a.prune},
                {"element_budget", a.element_budget ? Json(*a.element_budget) : Json(nullptr)},
                {"stages", a.prune_stages},
                {"attempts", a.prune_attempts},
                {"merit_factor", a.prune_factor},
                {"tolerance", a.prune_tolerance >= 0.0 ? Json(a.prune_tolerance) : Json(nullptr)},
                {"min_elements", a.prune_min_elements}};
        }
        if (a.field_rule) {
            options["field_rule"] =
                Json{{"ratio", a.field_ratio},
                     {"min_rms_waves", a.field_min_rms},
                     {"glass", a.field_glass.empty() ? Json(nullptr) : Json(a.field_glass)}};
        }
        if (ghost_return) {
            // M6-T10: nur mit der Option
            Json g = Json::parse(ghost_return_json_text(*ghost_return));
            g["glasses"] = a.ghost_glasses;
            g["order"] = a.ghost_order.value_or(GhostReturnOptions{}.order);
            options["start_from_real"] = std::move(g);
        }
        if (beam) {
            options["beam"] = Json{{"width", bo.beam_width},
                                   {"seed", bo.seed},
                                   {"alternatives", bo.alternatives},
                                   {"perturbed_starts", bo.perturbed_starts},
                                   {"start_perturbation", o.start_perturbation},
                                   {"prefilter", bo.prefilter},
                                   {"dedup_tolerance", bo.dedup_tolerance}};
        }
        Json doc{{"format", kSynthFormat},
                 {"version", kVersion},
                 {"input", a.file},
                 {"backend", backend},
                 {"options", options},
                 {"result", Json::parse(beam ? beam_result_json(*beam, -1)
                                             : homotopy_result_json(*single, -1))},
                 {"output", a.output.empty() ? Json(nullptr) : Json(a.output)}};
        if (a.al_rounds > 0) {
            // Verschiebungen θ am Endpunkt als Neustartzustand (--shifts-from), nicht in der
            // Systemdatei (Entscheidung des Maintainers 2026-11-19)
            Json shifts = Json::array();
            for (const ShiftValue& v : shift_values(end_system)) {
                shifts.push_back(Json{{"item", v.item},
                                      {"index", v.index},
                                      {"type", std::string(to_string(v.type))},
                                      {"shift", v.shift}});
            }
            doc["augmented_lagrangian"] =
                Json{{"shifts_from", a.shifts_from.empty() ? Json(nullptr) : Json(a.shifts_from)},
                     {"shifts", std::move(shifts)}};
        }
        if (!a.log.empty()) {
            std::ofstream f{fs::path(a.log), std::ios::binary};
            f << doc.dump(2) << '\n';
            if (!f) {
                err << "mikroptim synth: Fehler beim Schreiben von " << a.log << '\n';
                return kExitError;
            }
        }
        if (!a.dot.empty()) {
            std::ofstream f{fs::path(a.dot), std::ios::binary};
            write_beam_dot(*beam, f);
            if (!f) {
                err << "mikroptim synth: Fehler beim Schreiben von " << a.dot << '\n';
                return kExitError;
            }
        }
        if (a.json) {
            out << doc.dump(2) << '\n';
        } else {
            out << '\n';
            if (beam) {
                write_beam_summary(*beam, out);
                out << '\n';
                write_beam_tree(*beam, out);
                out << '\n';
            } else {
                write_homotopy_summary(*single, out);
            }
            kv(out, "Laufzeit [s]", fixed(seconds, 1));
            if (!a.output.empty()) {
                out << "Ergebnis geschrieben: " << a.output << '\n';
            }
            if (!a.log.empty()) {
                out << "Protokoll geschrieben: " << a.log << '\n';
            }
            if (!a.dot.empty()) {
                out << "Stammbaum geschrieben: " << a.dot << '\n';
            }
        }
        out.flush();
        if (!out) {
            err << "mikroptim synth: Fehler beim Schreiben der Ausgabe\n";
            return kExitError;
        }
        return kExitOk;
    } catch (const std::exception& e) {
        err << "mikroptim synth: Fehler: " << e.what() << '\n';
        return kExitError;
    }
}

} // namespace mko::cli
