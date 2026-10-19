// Unterbefehl `mikroptim snap`: Glas-Einrasten (docs/optimierung.md, Abschnitt 9) mit Bericht
// als Text oder JSON und Ergebnisdatei. Formate: docs/cli.md.

#include "cli/snap_command.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/glass/glass_hull.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/merit/ray_residual.hpp"
#include "mikroptim/merit/wavefront.hpp"
#include "mikroptim/optim/ghost_penalty.hpp"
#include "mikroptim/optim/glass_snap.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/version.hpp"

#ifdef MIKROPTIM_HAS_CUDA
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/levenberg_marquardt.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#endif

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace mko::cli {

namespace {

namespace fs = std::filesystem;
using Json = nlohmann::ordered_json;

constexpr int kDefaultRings = 10;
constexpr int kDefaultSpokes = 8;
constexpr int kDefaultMaxIterations = 100;
constexpr int kDefaultNeighbors = 3;
constexpr int kDefaultMaxCombinations = 64;
constexpr std::string_view kMeritUnit = "waves^2";

struct SnapArgs {
    std::string file;
    std::string output; // leer: keine Ergebnisdatei
    bool json = false;
    bool help = false;
    bool gpu = false;
    bool have_device = false;
    int device = 0;
    int rings = kDefaultRings;
    int spokes = kDefaultSpokes;
    int max_iterations = kDefaultMaxIterations;
    int neighbors = kDefaultNeighbors;
    int max_combinations = kDefaultMaxCombinations;
    bool hard_failure = false;
    bool have_fail_weight = false;
    double fail_weight = kDefaultFailWeight;
};

std::string_view next_value(std::span<const std::string_view> args, std::size_t& i,
                            std::string_view option) {
    if (i + 1 >= args.size()) {
        throw UsageError(std::string(option) + " erwartet einen Wert");
    }
    return args[++i];
}

SnapArgs parse_args(std::span<const std::string_view> args) {
    SnapArgs a;
    bool have_file = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--help" || arg == "-h") {
            a.help = true;
        } else if (arg == "--json") {
            a.json = true;
        } else if (arg == "--gpu") {
            a.gpu = true;
        } else if (arg == "-o" || arg == "--output") {
            a.output = std::string(next_value(args, i, arg));
            if (a.output.empty()) {
                throw UsageError(std::string(arg) + " erwartet einen Dateinamen");
            }
        } else if (arg == "--device") {
            a.device = parse_int(arg, next_value(args, i, arg));
            if (a.device < 0) {
                throw UsageError("--device muss >= 0 sein");
            }
            a.have_device = true;
        } else if (arg == "--neighbors") {
            a.neighbors = parse_int(arg, next_value(args, i, arg));
            if (a.neighbors < 1) {
                throw UsageError("--neighbors muss >= 1 sein");
            }
        } else if (arg == "--max-combinations") {
            a.max_combinations = parse_int(arg, next_value(args, i, arg));
            if (a.max_combinations < 1) {
                throw UsageError("--max-combinations muss >= 1 sein");
            }
        } else if (arg == "--rings") {
            a.rings = parse_int(arg, next_value(args, i, arg));
            if (a.rings < 1 || a.rings > kMaxPupilRings) {
                throw UsageError("--rings muss in [1, " + std::to_string(kMaxPupilRings) +
                                 "] liegen");
            }
        } else if (arg == "--spokes") {
            a.spokes = parse_int(arg, next_value(args, i, arg));
            if (a.spokes < 1) {
                throw UsageError("--spokes muss >= 1 sein");
            }
        } else if (arg == "--max-iterations") {
            a.max_iterations = parse_int(arg, next_value(args, i, arg));
            if (a.max_iterations < 0) {
                throw UsageError("--max-iterations muss >= 0 sein");
            }
        } else if (arg == "--fail-weight") {
            a.fail_weight = parse_double(arg, next_value(args, i, arg));
            if (!(a.fail_weight > 0.0) || !std::isfinite(a.fail_weight)) {
                throw UsageError("--fail-weight muss endlich und > 0 sein");
            }
            a.have_fail_weight = true;
        } else if (arg == "--hard-failure") {
            a.hard_failure = true;
        } else if (arg.size() > 1 && arg.front() == '-') {
            throw UsageError("unbekannte Option " + std::string(arg));
        } else if (have_file) {
            throw UsageError("mehr als eine Systemdatei angegeben");
        } else {
            a.file = std::string(arg);
            have_file = true;
        }
    }
    if (!have_file && !a.help) {
        throw UsageError("keine Systemdatei angegeben");
    }
    if (a.have_device && !a.gpu) {
        throw UsageError("--device verlangt --gpu");
    }
    if (a.have_fail_weight && a.hard_failure) {
        throw UsageError("--fail-weight gilt nur für weiches Versagen (nicht mit --hard-failure)");
    }
    return a;
}

// ------------------------------------------------------------------------------------------
// Bericht
// ------------------------------------------------------------------------------------------

struct BackendInfo {
    bool gpu = false;
    int device = -1;
    std::string device_name;
};

struct SnapReport {
    const SnapArgs* args = nullptr;
    BackendInfo backend;
    std::string system_name;
    std::size_t hull_glasses = 0;
    int hull_dimension = -1;
    GlassSnapResult snap;
    std::optional<WavefrontResult> wavefront_before;
    std::optional<WavefrontResult> wavefront_after;
    std::string wavefront_error;
    std::vector<std::string> warnings;
};

std::string join(const std::vector<std::string>& v) {
    std::string s;
    for (const std::string& x : v) {
        s += (s.empty() ? "" : ", ") + x;
    }
    return s;
}

const char* unit_of(VariableType type) {
    return type == VariableType::Curvature ? "1/mm" : "mm"; // ohne Glas: alle übrigen Längen
}

std::string variable_text(VariableType type, double v) {
    return type == VariableType::Curvature || type == VariableType::GhostCoefficient ? sci(v, 9)
                                                                                     : fixed(v, 6);
}

std::string combinations_text(const GlassSnapResult& s) {
    std::string t = std::to_string(s.runs.size()) + " von " + std::to_string(s.total_combinations);
    t += s.truncated ? " (kleinste Summe d²)" : " (alle)";
    return t;
}

void write_text(const SnapReport& r, std::ostream& out) {
    const SnapArgs& a = *r.args;
    const GlassSnapResult& s = r.snap;
    const GlassSnapRun& best = s.best_run();
    const SystemDescription& sys = s.continuous.system;

    out << "Glas-Einrasten: " << r.system_name << '\n';
    kv(out, "Datei", a.file);
    kv(out, "Ergebnisdatei", a.output.empty() ? std::string("- (ohne -o)") : a.output);
    if (r.backend.gpu) {
        kv(out, "Backend",
           "GPU " + std::to_string(r.backend.device) + " (" + r.backend.device_name + "), FP64");
    } else {
        kv(out, "Backend", "CPU, FP64");
    }
    kv(out, "Quadratur",
       std::to_string(a.rings) + " Ringe x " + std::to_string(a.spokes) + " Speichen");
    kv(out, "Strahlversagen",
       a.hard_failure ? std::string("hart (versagte Strahlen maskiert)")
                      : "weich, w_fail ab " + sci(a.fail_weight, 3) +
                            " Wellen² (Strafverfahren x10 bis 1e14)");
    kv(out, "Glaselemente", std::to_string(s.elements.size()));
    kv(out, "Glas-Hülle",
       std::to_string(r.hull_glasses) + " Gläser, Dimension " + std::to_string(r.hull_dimension));
    kv(out, "Nachbarn je Element", std::to_string(a.neighbors));
    kv(out, "Kombinationen (Läufe)", combinations_text(s));
    kv(out, "Zyklen höchstens", std::to_string(a.max_iterations));

    out << '\n'
        << "Kandidaten (nächste Katalog-Gläser, Abstand im normierten (n0, nu1, nu2)-Raum der "
           "Glas-Hülle)\n";
    for (const GlassSnapElement& e : s.elements) {
        out << "  Fläche " << e.file_surface
            << ": kontinuierlich n0 = " << fixed(e.coordinates[0], 7)
            << ", nu1 = " << sci(e.coordinates[1], 6) << ", nu2 = " << sci(e.coordinates[2], 6)
            << ", Abstand zur Hülle " << sci(e.hull_distance, 3) << '\n';
        for (std::size_t j = 0; j < e.candidates.size(); ++j) {
            const GlassCandidate& c = e.candidates[j];
            out << "    " << pad_left(std::to_string(j), 4) << "  " << pad_right(c.name, 12)
                << pad_left(sci(c.distance, 6), 14) << '\n';
        }
    }

    out << '\n'
        << "Kombinationen (sortiert nach Summe d²; Merit in Wellen² bei "
        << fixed(sys.spectrum.reference_nm, 3) << " nm)\n";
    out << "  " << pad_left("Lauf", 4) << "  " << pad_right("Gläser", 28)
        << pad_left("Summe d²", 12) << pad_left("eingerastet", 17) << pad_left("reoptimiert", 17)
        << pad_left("Zyklen", 8) << "  " << pad_right("Status", 18) << pad_left("versagt", 8)
        << '\n';
    for (std::size_t i = 0; i < s.runs.size(); ++i) {
        const GlassSnapRun& run = s.runs[i];
        out << "  " << pad_left(std::to_string(i), 4) << "  " << pad_right(join(run.glasses), 28)
            << pad_left(sci(run.combination.cost, 3), 12) << pad_left(sci(run.snapped_merit, 9), 17)
            << pad_left(sci(run.run.merit, 9), 17)
            << pad_left(std::to_string(run.run.iterations), 8) << "  "
            << pad_right(to_string(run.run.status), 18)
            << pad_left(std::to_string(run.run.failed_rays), 8)
            << (static_cast<int>(i) == s.best ? "  bester" : "") << '\n';
    }

    out << '\n' << "Ergebnis\n";
    std::string chosen;
    for (std::size_t e = 0; e < s.elements.size(); ++e) {
        chosen += (e == 0 ? "" : ", ") + std::string("Fläche ") +
                  std::to_string(s.elements[e].file_surface) + ": " + best.glasses[e];
    }
    kv(out, "Gewählte Gläser", chosen);
    kv(out, "Bester Lauf", std::to_string(s.best));
    kv(out, "Status", to_string(best.run.status));
    kv(out, "Merit kontinuierlich [Wellen²]", sci(s.continuous_merit, 9));
    kv(out, "Merit eingerastet, ohne Reopt. [Wellen²]", sci(best.snapped_merit, 9));
    kv(out, "Merit eingerastet, reoptimiert [Wellen²]", sci(best.run.merit, 9));
    kv(out, "Verschlechterung [Wellen²]", sci(s.degradation(), 6));
    kv(out, "Verschlechterung relativ", sci(s.relative_degradation(), 6));
    kv(out, "Versagte Strahlen kontinuierlich/Ergebnis",
       std::to_string(s.continuous_failed_rays) + " / " + std::to_string(best.run.failed_rays));

    out << '\n' << "Variablen (ohne Glas)\n";
    out << "  " << pad_right("Name", 8) << pad_right("Einheit", 8) << pad_left("vorher", 17)
        << pad_left("nachher", 17) << '\n';
    for (int k = 0; k < s.set.size(); ++k) {
        const Variable& v = s.set[k];
        const auto kk = static_cast<std::size_t>(k);
        out << "  " << pad_right(v.name, 8) << pad_right(unit_of(v.type), 8)
            << pad_left(variable_text(v.type, s.x0[kk]), 17)
            << pad_left(variable_text(v.type, best.run.x[kk]), 17) << '\n';
    }

    out << '\n';
    if (!r.wavefront_before || !r.wavefront_after) {
        out << "Wellenfront nicht auswertbar\n  " << r.wavefront_error << '\n';
        return;
    }
    const WavefrontResult& wb = *r.wavefront_before;
    const WavefrontResult& wa = *r.wavefront_after;
    out << "Wellenfront (Quadratur " << a.rings << " Ringe x " << a.spokes
        << " Speichen, CPU, FP64, RMS in Wellen bei " << fixed(sys.spectrum.reference_nm, 3)
        << " nm; vorher = kontinuierlich, nachher = eingerastet)\n";
    out << "  " << pad_left("Feld", 4) << pad_left("Winkel [deg]", 14)
        << pad_left("Wellenlänge [nm]", 18) << pad_left("Strahlen", 10)
        << pad_left("RMS vorher", 14) << pad_left("RMS nachher", 14)
        << pad_left("versagt vorher/nachher", 24) << '\n';
    for (std::size_t i = 0; i < wb.maps.size() && i < wa.maps.size(); ++i) {
        const WavefrontMap& mb = wb.maps[i];
        const WavefrontMap& ma = wa.maps[i];
        const std::string angle =
            sys.source.type == SourceType::Collimated
                ? fixed(sys.source.field_angles_deg[static_cast<std::size_t>(mb.field)], 3)
                : std::string("-");
        out << "  " << pad_left(std::to_string(mb.field), 4) << pad_left(angle, 14)
            << pad_left(fixed(mb.wavelength_nm, 3), 18)
            << pad_left(std::to_string(mb.samples()), 10) << pad_left(fixed(mb.rms_waves), 14)
            << pad_left(fixed(ma.rms_waves), 14)
            << pad_left(std::to_string(mb.failed) + " / " + std::to_string(ma.failed), 24) << '\n';
    }
}

// Nicht endliche Zahlen sind in JSON nicht darstellbar und werden null (docs/cli.md).
Json number(double v) {
    return std::isfinite(v) ? Json(v) : Json(nullptr);
}

Json numbers(std::span<const double> v) {
    Json a = Json::array();
    for (const double x : v) {
        a.push_back(number(x));
    }
    return a;
}

Json glass_point(const Point3& x) {
    return Json{{"n0", number(x[0])}, {"nu1", number(x[1])}, {"nu2", number(x[2])}};
}

Json wavefront_json(const SystemDescription& sys, const WavefrontResult& w) {
    Json entries = Json::array();
    for (const WavefrontMap& m : w.maps) {
        Json e;
        e["field"] = m.field;
        e["field_angle_deg"] =
            sys.source.type == SourceType::Collimated
                ? number(sys.source.field_angles_deg[static_cast<std::size_t>(m.field)])
                : Json(nullptr);
        e["wavelength"] = m.wavelength;
        e["wavelength_nm"] = number(m.wavelength_nm);
        e["rays"] = m.samples();
        e["failed_rays"] = m.failed;
        e["rms_waves"] = number(m.rms_waves);
        e["rms_mm"] = number(m.rms);
        entries.push_back(std::move(e));
    }
    return entries;
}

void write_json(const SnapReport& r, std::ostream& out) {
    const SnapArgs& a = *r.args;
    const GlassSnapResult& s = r.snap;
    const GlassSnapRun& best = s.best_run();
    const SystemDescription& sys = s.continuous.system;
    Json j;
    j["format"] = std::string(kSnapFormat);
    j["mikroptim_version"] = kVersion;
    j["system"] = r.system_name;
    j["file"] = a.file;
    j["output"] = a.output.empty() ? Json(nullptr) : Json(a.output);
    j["backend"] = r.backend.gpu ? "gpu" : "cpu";
    j["device"] = r.backend.gpu ? Json{{"index", r.backend.device}, {"name", r.backend.device_name}}
                                : Json(nullptr);
    j["precision"] = "fp64";
    j["merit_unit"] = std::string(kMeritUnit);
    j["quadrature"] = Json{{"rings", a.rings}, {"spokes", a.spokes}};
    Json settings;
    settings["neighbors"] = a.neighbors;
    settings["max_combinations"] = a.max_combinations;
    settings["max_iterations"] = a.max_iterations;
    settings["failure"] = a.hard_failure ? "hard" : "soft";
    settings["fail_weight"] = a.hard_failure ? Json(nullptr) : number(a.fail_weight);
    settings["distance_weights"] = Json::array({1.0, 1.0, 1.0});
    j["settings"] = std::move(settings);
    j["hull"] = Json{{"glasses", r.hull_glasses}, {"dimension", r.hull_dimension}};

    Json elements = Json::array();
    for (const GlassSnapElement& e : s.elements) {
        Json cands = Json::array();
        for (const GlassCandidate& c : e.candidates) {
            cands.push_back(Json{{"name", c.name},
                                 {"distance", number(c.distance)},
                                 {"coordinates", glass_point(c.coordinates)}});
        }
        elements.push_back(Json{{"surface", e.file_surface},
                                {"coordinates", glass_point(e.coordinates)},
                                {"hull_distance", number(e.hull_distance)},
                                {"candidates", std::move(cands)}});
    }
    j["elements"] = std::move(elements);
    j["total_combinations"] = s.total_combinations;
    j["truncated"] = s.truncated;

    Json runs = Json::array();
    for (std::size_t i = 0; i < s.runs.size(); ++i) {
        const GlassSnapRun& run = s.runs[i];
        Json e;
        e["run"] = static_cast<int>(i);
        e["glasses"] = run.glasses;
        e["choice"] = run.combination.choice;
        e["cost"] = number(run.combination.cost);
        e["snapped_merit"] = number(run.snapped_merit);
        e["snapped_failed_rays"] = run.snapped_failed_rays;
        e["merit"] = number(run.run.merit);
        e["failed_rays"] = run.run.failed_rays;
        e["hard_failed_rays"] = run.run.hard_failed_rays;
        e["status"] = to_string(run.run.status);
        e["iterations"] = run.run.iterations;
        e["penalty_rounds"] = run.run.penalty_rounds;
        e["fail_weight"] = a.hard_failure ? Json(nullptr) : number(run.run.fail_weight);
        e["x"] = numbers(run.run.x);
        runs.push_back(std::move(e));
    }
    j["runs"] = std::move(runs);
    j["best_run"] = s.best;

    Json vars = Json::array();
    for (int k = 0; k < s.set.size(); ++k) {
        const Variable& v = s.set[k];
        const auto kk = static_cast<std::size_t>(k);
        const bool ghost = v.type == VariableType::GhostCoefficient;
        vars.push_back(Json{{"name", v.name},
                            {"type", std::string(to_string(v.type))},
                            {"surface", ghost ? Json(nullptr) : Json(v.file_surface)},
                            {"start", number(s.x0[kk])},
                            {"result", number(best.run.x[kk])}});
    }
    j["variables"] = std::move(vars);

    Json chosen = Json::array();
    for (std::size_t e = 0; e < s.elements.size(); ++e) {
        chosen.push_back(Json{{"surface", s.elements[e].file_surface}, {"glass", best.glasses[e]}});
    }
    Json res;
    res["glasses"] = std::move(chosen);
    res["status"] = to_string(best.run.status);
    res["converged"] = lm_converged(best.run.status);
    res["continuous_merit"] = number(s.continuous_merit);
    res["snapped_merit"] = number(best.snapped_merit);
    res["merit"] = number(best.run.merit);
    res["degradation"] = number(s.degradation());
    res["relative_degradation"] = number(s.relative_degradation());
    res["continuous_failed_rays"] = s.continuous_failed_rays;
    res["failed_rays"] = best.run.failed_rays;
    res["hard_failed_rays"] = best.run.hard_failed_rays;
    j["result"] = std::move(res);

    j["reference_nm"] = number(sys.spectrum.reference_nm);
    if (r.wavefront_before && r.wavefront_after) {
        j["wavefront"] = Json{{"start", wavefront_json(sys, *r.wavefront_before)},
                              {"result", wavefront_json(sys, *r.wavefront_after)}};
        j["wavefront_error"] = nullptr;
    } else {
        j["wavefront"] = nullptr;
        j["wavefront_error"] = r.wavefront_error;
    }
    j["warnings"] = r.warnings;
    out << j.dump(2, ' ', false, Json::error_handler_t::replace) << '\n';
}

} // namespace

void print_snap_usage(std::ostream& out) {
    out << "Aufruf: mikroptim snap <system.json> [-o <ergebnis.json>] [--json]\n"
           "                      [--neighbors k] [--max-combinations N] [--max-iterations N]\n"
           "                      [--rings N] [--spokes M] [--fail-weight w | --hard-failure]\n"
           "                      [--gpu [--device N]]\n"
           "  Rastet die Glasvariablen (Typ glass) auf Katalog-Gläser ein: pro Glaselement die k\n"
           "  nächsten Gläser im normierten (n0, nu1, nu2)-Raum der Glas-Hülle, jede Kombination\n"
           "  ein Lauf des gebündelten Levenberg-Marquardt (übrige Variablen reoptimiert, FP64);\n"
           "  die beste Kombination wird das Ergebnis. Beschreibung: docs/cli.md.\n"
           "  -o DATEI             Ergebnis als neue Systemdatei (nie die Eingabedatei)\n"
           "  --json               maschinenlesbare Ausgabe (Format mikroptim.snap/1)\n"
           "  --neighbors k        Katalog-Gläser pro Element, >= 1 (Standard "
        << kDefaultNeighbors
        << ")\n"
           "  --max-combinations N höchstens N Kombinationen (kleinste Summe der quadrierten\n"
           "                       Abstände), >= 1 (Standard "
        << kDefaultMaxCombinations
        << ")\n"
           "  --max-iterations N   Zyklen pro Lauf, >= 0 (Standard "
        << kDefaultMaxIterations
        << "; 0 = nur einrasten)\n"
           "  --rings N            radiale Ringe der Pupillenquadratur, 1 bis "
        << kMaxPupilRings << " (Standard " << kDefaultRings
        << ")\n"
           "  --spokes M           Speichen der Pupillenquadratur, >= 1 (Standard "
        << kDefaultSpokes
        << ")\n"
           "  --fail-weight w      Start-Strafgewicht w_fail des weichen Strahlversagens in\n"
           "                       Wellen², > 0 (Standard "
        << sci(kDefaultFailWeight, 0)
        << "; je Runde x10 bis 1e14)\n"
           "  --hard-failure       hartes Strahlversagen (ohne Strafresiduen)\n"
           "  --gpu                auf der GPU rechnen statt auf der CPU\n"
           "  --device N           CUDA-Gerät für --gpu (Standard 0, Liste: mikroptim --devices)\n"
           "  --help               diese Hilfe\n"
           "Exit-Code: 0 Erfolg, 1 Fehler beim Laden, Einrasten oder Schreiben, 2 ungültige\n"
           "           Kommandozeile, 3 GPU-Backend nicht verfügbar.\n";
}

int run_snap_command(std::span<const std::string_view> args, std::ostream& out, std::ostream& err) {
    SnapArgs a;
    try {
        a = parse_args(args);
        if (!a.output.empty() && same_file(fs::path(a.file), fs::path(a.output))) {
            throw UsageError("Ergebnisdatei " + a.output +
                             " ist die Eingabedatei; die Eingabe wird nie überschrieben");
        }
    } catch (const UsageError& e) {
        err << "mikroptim snap: " << e.what() << '\n';
        print_snap_usage(err);
        return kExitUsage;
    }
    if (a.help) {
        print_snap_usage(out);
        return kExitOk;
    }

#ifdef MIKROPTIM_HAS_CUDA
    std::optional<cuda::TraceEngine> engine;
    if (a.gpu) {
        const int count = cuda::device_count();
        if (count == 0) {
            err << "mikroptim snap: --gpu: kein CUDA-Gerät gefunden (GPU-Backend nicht "
                   "verfügbar)\n";
            return kExitNoGpu;
        }
        if (a.device >= count) {
            err << "mikroptim snap: Fehler: --device " << a.device << ": Gerät existiert nicht ("
                << count << (count == 1 ? " CUDA-Gerät" : " CUDA-Geräte")
                << ", siehe mikroptim --devices)\n";
            return kExitError;
        }
        try {
            engine.emplace(a.device);
        } catch (const std::exception& e) {
            err << "mikroptim snap: Fehler: GPU " << a.device << ": " << e.what() << '\n';
            return kExitError;
        }
    }
#else
    if (a.gpu) {
        err << "mikroptim snap: --gpu: ohne CUDA gebaut (MIKROPTIM_ENABLE_CUDA=OFF), "
               "GPU-Backend nicht verfügbar\n";
        return kExitNoGpu;
    }
#endif

    try {
        SnapReport r;
        r.args = &a;
        const LoadedSystem loaded = load_system(fs::path(a.file));
        for (const std::string& w : loaded.warnings) {
            err << "mikroptim snap: Warnung: " << w << '\n';
            r.warnings.push_back(w);
        }
        const SystemDescription& sys = loaded.system;
        r.system_name = sys.name;
        const GlassHull hull = make_glass_hull(loaded.catalog);
        r.hull_glasses = hull.points().size();
        r.hull_dimension = hull.dimension();

        // Zusatzresiduen wie mikroptim optimize ohne die Glas-Hülle: paraxiale Nebenbedingungen,
        // L1-Strafe der Geisterkoeffizienten
        const BatchTopology topology = topology_of(sys, loaded.index);
        const ParaxialConstraints constraints = make_paraxial_constraints(sys);
        check_paraxial_constraints(constraints, topology);
        const GhostPenalty penalty = make_ghost_penalty(sys);
        check_ghost_penalty(penalty, topology);

        GlassSnapOptions options;
        options.neighbors = a.neighbors;
        options.max_combinations = a.max_combinations;
        options.lm.max_iterations = a.max_iterations;
        options.lm.jacobian.quadrature.rings = a.rings;
        options.lm.jacobian.quadrature.spokes = a.spokes;
        options.lm.jacobian.quadrature.failure =
            a.hard_failure ? FailureMode::Hard : FailureMode::Soft;
        options.lm.jacobian.quadrature.fail_weight = a.fail_weight;
        options.lm.jacobian.extra = concat_extra_residuals(
            {make_paraxial_residuals(constraints), make_ghost_penalty_residuals(penalty)});

        // Feldziele aus der paraxialen Brennweite (ohne target.focal_length) ändern sich beim
        // erneuten Laden des Ergebnisses (wie bei mikroptim optimize)
        if (sys.source.type == SourceType::Collimated &&
            !(sys.target && sys.target->focal_length)) {
            for (const double angle : sys.source.field_angles_deg) {
                if (angle != 0.0) {
                    const std::string w =
                        "Feldziele aus der paraxialen Brennweite (kein target.focal_length); beim "
                        "erneuten Laden des Ergebnisses ändern sie sich";
                    err << "mikroptim snap: Warnung: " << w << '\n';
                    r.warnings.push_back(w);
                    break;
                }
            }
        }

        GlassSnapOptimizer optimize;
#ifdef MIKROPTIM_HAS_CUDA
        if (engine) {
            r.backend.gpu = true;
            r.backend.device = engine->device();
            r.backend.device_name = engine->device_name();
            optimize = [&engine](const VariableSet& set, const SystemBatchHost& base,
                                 std::span<const double> x0, const LmOptions& o) {
                return cuda::optimize_lm_gpu(*engine, set, base, x0, o);
            };
        }
#endif
        r.snap = snap_glasses(loaded, hull, options, optimize);
        const GlassSnapResult& s = r.snap;

        // Wellenfront vorher (kontinuierlich) und nachher (eingerastet), CPU
        WavefrontOptions wo;
        wo.rings = a.rings;
        wo.spokes = a.spokes;
        try {
            r.wavefront_before = evaluate_wavefront(s.continuous, wo);
            r.wavefront_after = evaluate_wavefront(s.result, wo);
        } catch (const std::invalid_argument& e) {
            r.wavefront_before.reset();
            r.wavefront_after.reset();
            r.wavefront_error = e.what();
        }

        // Ergebnisdatei: Glaskataloge so, dass sie vom Verzeichnis der Ergebnisdatei aus
        // gefunden werden
        if (!a.output.empty()) {
            const fs::path output(a.output);
            std::error_code ec;
            fs::path out_dir = fs::absolute(output, ec).parent_path();
            if (ec) {
                out_dir = output.parent_path();
            }
            SystemDescription file_system = s.result.system;
            file_system.glass_catalogs =
                catalogs_for(s.result.system, s.result.catalog_files, out_dir);
            write_system_file(file_system, output);
        }

        if (a.json) {
            write_json(r, out);
        } else {
            write_text(r, out);
            if (!a.output.empty()) {
                out << '\n' << "Ergebnis geschrieben: " << a.output << '\n';
            }
        }
        out.flush();
        if (!out) {
            err << "mikroptim snap: Fehler beim Schreiben der Ausgabe\n";
            return kExitError;
        }
        return kExitOk;
    } catch (const std::exception& e) {
        err << "mikroptim snap: Fehler: " << e.what() << '\n';
        return kExitError;
    }
}

} // namespace mko::cli
