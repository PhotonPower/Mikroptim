// Unterbefehl `mikroptim synth` (M6-T3): Startdatei laden, Homotopie-Controller mit dem
// optischen Modell (CPU oder --gpu über optimize_lm_gpu), Protokoll als Text oder JSON, optional
// als Datei, Ergebnis als Systemdatei. Formate: docs/cli.md.

#include "cli/synth_command.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/synth/homotopy.hpp"
#include "mikroptim/synth/optical_homotopy.hpp"
#include "mikroptim/version.hpp"

#ifdef MIKROPTIM_HAS_CUDA
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/levenberg_marquardt.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#endif

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
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
    int max_elements = HomotopyOptions{}.max_elements;
    int max_steps = HomotopyOptions{}.max_steps;
    int max_iterations = kDefaultMaxIterations;
    int rings = 10;
    int spokes = 8;
    HomotopyStepControl step{};
    OpticalHomotopyOptions optical{};
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
        } else if (arg == "--max-elements") {
            a.max_elements = parse_int(arg, next_value(args, i, arg));
            if (a.max_elements < 0) {
                throw UsageError("--max-elements muss >= 0 sein");
            }
        } else if (arg == "--max-steps") {
            a.max_steps = parse_int(arg, next_value(args, i, arg));
            if (a.max_steps < 1) {
                throw UsageError("--max-steps muss >= 1 sein");
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
            const std::string v(next_value(args, i, arg));
            const std::size_t c1 = v.find(',');
            const std::size_t c2 = c1 == std::string::npos ? c1 : v.find(',', c1 + 1);
            if (c2 == std::string::npos) {
                throw UsageError("--step erwartet D0,DMIN,DMAX");
            }
            a.step.initial = positive(arg, std::string_view(v).substr(0, c1));
            a.step.min = positive(arg, std::string_view(v).substr(c1 + 1, c2 - c1 - 1));
            a.step.max = positive(arg, std::string_view(v).substr(c2 + 1));
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
        } else if (arg == "--curvature-bound") {
            const double c = positive(arg, next_value(args, i, arg));
            a.optical.meniscus.curvature_variables.min = -c;
            a.optical.meniscus.curvature_variables.max = c;
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
    for (const std::string* f : {&a.output, &a.log}) {
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
           "Homotopie-Controller der Systemsynthese (docs/optimierung.md, Abschnitt 7): öffnet\n"
           "die Apertur (und den Feldanteil) schrittweise mit Schrittweitensteuerung, fügt bei\n"
           "Bedarf (Diagnose der Geisterplatten) Elemente ein und prüft am Ende das Beugungsziel\n"
           "am realen System.\n"
           "  --target-na NA            numerische Apertur am Ende (Standard: wie der Start)\n"
           "  --start-field F           Feldanteil am Start (Standard 1)\n"
           "  --target-field F          Feldanteil am Ende (Standard: wie der Start)\n"
           "  --rms-target W            Beugungsziel [Wellen] pro Feld und Wellenlänge (1/14)\n"
           "  --merit-factor K          Schwelle F_max = (K · Ziel)² (Standard 1)\n"
           "  --step D0,DMIN,DMAX       Schrittweite der Apertur (Standard 0.02,0.002,0.1)\n"
           "  --max-elements N          höchstens N Glaselemente (Standard 6)\n"
           "  --max-steps N             höchstens N Optimierungen (Standard 400)\n"
           "  --max-iterations N        LM-Zyklen pro Schritt (Standard 500)\n"
           "  --rings N, --spokes M     Pupillenquadratur (Standard 10 x 8)\n"
           "  --element-thickness T     Mittendicke eines eingefügten Meniskus [mm] (0.5)\n"
           "  --element-max-thickness T danach Variable bis T [mm] (5; <= Dicke: fest)\n"
           "  --element-stages J        Stufen der inneren Homotopie über die Dicke (5)\n"
           "  --min-air-gap G           kleinster Luftspalt hinter dem Meniskus [mm] (0.1)\n"
           "  --saddle-delta D          zwei Starts am Sattelpunkt, c0 +- D [1/mm] (0.002;\n"
           "                            0: ein Zweig)\n"
           "  --curvature-bound C       Schranken +-C [1/mm] der neuen Krümmungsvariablen\n"
           "  -o, --output <datei>      Ergebnis am Endpunkt als Systemdatei\n"
           "  --log <datei>             Protokoll als JSON (mikroptim.synth/1) in eine Datei\n"
           "  --json                    Ausgabe als JSON statt Text\n"
           "  --quiet                   ohne Schrittprotokoll (nur Zusammenfassung)\n"
           "  --gpu [--device N]        LM auf einem CUDA-Gerät (optimize_lm_gpu)\n";
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
        const LoadedSystem start = load_system(fs::path(a.file));
        for (const std::string& w : start.warnings) {
            err << "mikroptim synth: Warnung: " << w << '\n';
        }
        OpticalHomotopyOptions o = a.optical;
        o.lm.max_iterations = a.max_iterations;
        o.lm.jacobian.quadrature.rings = a.rings;
        o.lm.jacobian.quadrature.spokes = a.spokes;
        Json backend{{"gpu", false}};
#ifdef MIKROPTIM_HAS_CUDA
        if (engine) {
            cuda::TraceEngine* e = &*engine;
            o.optimizer = [e](const VariableSet& set, const SystemBatchHost& base,
                              std::span<const double> x0, const LmOptions& lm) {
                return cuda::optimize_lm_gpu(*e, set, base, x0, lm);
            };
            backend = Json{{"gpu", true},
                           {"device", engine->device()},
                           {"device_name", engine->device_name()}};
        }
#endif
        OpticalHomotopyModel model(start, o);
        HomotopyPoint p0 = model.start_point();
        if (a.start_field) {
            p0.field = *a.start_field;
        }
        HomotopyOptions h;
        h.target_aperture = a.target_na.value_or(0.0);
        h.target_field = a.target_field.value_or(-1.0);
        h.target_rms = a.rms_target;
        h.merit_factor = a.merit_factor;
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
            out << '\n';
            if (stream) {
                write_homotopy_step_header(out);
            }
            out.flush();
        }
        if (stream) {
            h.on_step = [&out](const HomotopyStep& s) {
                write_homotopy_step_line(s, out);
                out.flush();
            };
        }
        const HomotopyResult r = run_homotopy(model, model.start_system(), p0, h);

        if (!a.output.empty()) {
            const fs::path output(a.output);
            std::error_code ec;
            fs::path out_dir = fs::absolute(output, ec).parent_path();
            if (ec) {
                out_dir = output.parent_path();
            }
            SystemDescription file_system = model.realize(r.system, r.point, false);
            file_system.glass_catalogs = catalogs_for(file_system, start.catalog_files, out_dir);
            write_system_file(file_system, output);
        }

        Json doc{{"format", kSynthFormat},
                 {"version", kVersion},
                 {"input", a.file},
                 {"backend", backend},
                 {"options", Json{{"target_na", a.target_na ? Json(*a.target_na) : Json(nullptr)},
                                  {"start", point_short(p0)},
                                  {"rms_target_waves", a.rms_target},
                                  {"merit_factor", a.merit_factor},
                                  {"max_elements", a.max_elements},
                                  {"max_iterations", a.max_iterations},
                                  {"quadrature", Json{{"rings", a.rings}, {"spokes", a.spokes}}},
                                  {"element_thickness", o.element_thickness},
                                  {"element_max_thickness", o.element_max_thickness},
                                  {"saddle_delta", o.saddle_delta}}},
                 {"result", Json::parse(homotopy_result_json(r, -1))},
                 {"output", a.output.empty() ? Json(nullptr) : Json(a.output)}};
        if (!a.log.empty()) {
            std::ofstream f{fs::path(a.log), std::ios::binary};
            f << doc.dump(2) << '\n';
            if (!f) {
                err << "mikroptim synth: Fehler beim Schreiben von " << a.log << '\n';
                return kExitError;
            }
        }
        if (a.json) {
            out << doc.dump(2) << '\n';
        } else {
            out << '\n';
            write_homotopy_summary(r, out);
            kv(out, "Laufzeit [s]", fixed(r.seconds, 1));
            if (!a.output.empty()) {
                out << "Ergebnis geschrieben: " << a.output << '\n';
            }
            if (!a.log.empty()) {
                out << "Protokoll geschrieben: " << a.log << '\n';
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
