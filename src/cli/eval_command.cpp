// Unterbefehl `mikroptim eval` (M8-T1): Laden, Batch mit einer Variante, Modus EVAL auf der CPU
// oder mit --gpu über die TraceEngine, Kennzahlen (eval_statistics), Ausgabe als Text oder JSON,
// optional OPD-Karte als CSV. Ausgabeformate: docs/cli.md.

#include "cli/eval_command.hpp"
#include "cli/cli_util.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/eval/eval.hpp"
#include "mikroptim/version.hpp"

#ifdef MIKROPTIM_HAS_CUDA
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#endif

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <fstream>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mko::cli {

namespace {

using Json = nlohmann::ordered_json;

struct EvalArgs {
    std::string file;
    bool json = false;
    bool help = false;
    bool gpu = false;
    bool have_device = false;
    int device = 0;
    EvalSettings settings{};
    std::string map_file;  // leer: keine OPD-Karte
    bool have_map = false; // --map angegeben
    int map_field = 0;     // Tupel der Karte
    int map_wavelength = 0;
    OpdRemoval removal = OpdRemoval::Piston;
};

std::string_view next_value(std::span<const std::string_view> args, std::size_t& i,
                            std::string_view option) {
    if (i + 1 >= args.size()) {
        throw UsageError(std::string(option) + " erwartet einen Wert");
    }
    return args[++i];
}

EvalArgs parse_args(std::span<const std::string_view> args) {
    EvalArgs a;
    bool have_file = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--help" || arg == "-h") {
            a.help = true;
        } else if (arg == "--json") {
            a.json = true;
        } else if (arg == "--gpu") {
            a.gpu = true;
        } else if (arg == "--half") {
            a.settings.half_pupil = true;
        } else if (arg == "--device") {
            a.device = parse_int(arg, next_value(args, i, arg));
            if (a.device < 0) {
                throw UsageError("--device muss >= 0 sein");
            }
            a.have_device = true;
        } else if (arg == "--grid") {
            a.settings.grid = parse_int(arg, next_value(args, i, arg));
            if (a.settings.grid < 1 || a.settings.grid > kMaxEvalGrid) {
                throw UsageError("--grid muss in [1, " + std::to_string(kMaxEvalGrid) + "] liegen");
            }
        } else if (arg == "--reference") {
            const std::string_view v = next_value(args, i, arg);
            if (v == "exit-pupil") {
                a.settings.reference = EvalReference::ExitPupil;
            } else if (v == "vertex") {
                a.settings.reference = EvalReference::Vertex;
            } else {
                throw UsageError("--reference erwartet exit-pupil oder vertex, nicht '" +
                                 std::string(v) + "'");
            }
        } else if (arg == "--map") {
            const std::string_view v = next_value(args, i, arg);
            const std::size_t comma = v.find(',');
            if (comma == std::string_view::npos) {
                throw UsageError("--map erwartet F,L (Feld und Wellenlänge, ab 0)");
            }
            a.map_field = parse_int(arg, v.substr(0, comma));
            a.map_wavelength = parse_int(arg, v.substr(comma + 1));
            a.have_map = true;
        } else if (arg == "--map-file") {
            a.map_file = std::string(next_value(args, i, arg));
        } else if (arg == "--remove") {
            const std::string_view v = next_value(args, i, arg);
            if (v == "piston") {
                a.removal = OpdRemoval::Piston;
            } else if (v == "tilt") {
                a.removal = OpdRemoval::Tilt;
            } else if (v == "defocus") {
                a.removal = OpdRemoval::Defocus;
            } else {
                throw UsageError("--remove erwartet piston, tilt oder defocus, nicht '" +
                                 std::string(v) + "'");
            }
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
    if (a.have_map && a.map_file.empty()) {
        throw UsageError("--map verlangt --map-file");
    }
    return a;
}

// Nicht endliche Zahlen sind in JSON nicht darstellbar und werden null.
Json number(double v) {
    return std::isfinite(v) ? Json(v) : Json(nullptr);
}

// Strehl nach Maréchal exp(−(2π σ/λ)²) mit dem RMS σ [mm] bei der Wellenlänge λ [mm] des
// Tupels (grobe Angabe, gültig für kleine Fehler; PSF und Strehl über FFT folgen in M8-T2).
double marechal_strehl(double rms_mm, double lambda_mm) {
    const double phase = 2.0 * std::numbers::pi * rms_mm / lambda_mm;
    return std::exp(-phase * phase);
}

const char* reference_name(EvalReference r) {
    return r == EvalReference::Vertex ? "vertex" : "exit-pupil";
}

struct EvalReport {
    std::string file;
    LoadedSystem loaded;
    SystemBatchHost batch;
    EvalResult result;
    std::vector<EvalTupleStats> stats;
    bool gpu = false;
    int device = -1;
    std::string device_name;
};

double field_angle_deg(const EvalReport& r, int f) {
    const SystemDescription& sys = r.loaded.system;
    return sys.source.type == SourceType::Collimated
               ? sys.source.field_angles_deg[static_cast<std::size_t>(f)]
               : std::nan("");
}

void write_json(const EvalReport& r, std::ostream& out) {
    const SystemDescription& sys = r.loaded.system;
    const double lambda_ref_mm = r.result.reference_nm * 1e-6;
    Json j;
    j["format"] = std::string(kEvalFormat);
    j["mikroptim_version"] = kVersion;
    j["system"] = sys.name;
    j["file"] = r.file;
    j["reference_nm"] = number(r.result.reference_nm);
    j["backend"] = r.gpu ? "gpu" : "cpu";
    if (r.gpu) {
        j["device"] = Json{{"index", r.device}, {"name", r.device_name}};
    } else {
        j["device"] = nullptr;
    }
    j["grid"] = r.result.settings.grid;
    j["half_pupil"] = r.result.settings.half_pupil;
    j["grid_points"] = r.result.samples();
    j["reference_sphere"] = reference_name(r.result.settings.reference);
    j["exit_pupil_position"] = number(r.result.exit_pupil_position[0]);
    Json tuples = Json::array();
    for (const EvalTupleStats& t : r.stats) {
        const double lambda_mm = r.batch.wavelength_nm(t.wavelength) * 1e-6;
        Json e;
        e["field"] = t.field;
        e["field_angle_deg"] = number(field_angle_deg(r, t.field));
        e["wavelength"] = t.wavelength;
        e["wavelength_nm"] = number(r.batch.wavelength_nm(t.wavelength));
        e["image_height_mm"] = number(t.image_height);
        e["rays"] = t.rays;
        e["failed_rays"] = t.failed;
        e["rms_opd_mm"] = number(t.rms_opd);
        e["rms_opd_waves"] = number(t.rms_opd / lambda_ref_mm);
        e["pv_opd_waves"] = number(t.pv_opd / lambda_ref_mm);
        e["rms_opd_tilt_waves"] = number(t.rms_opd_tilt / lambda_ref_mm);
        e["rms_opd_defocus_waves"] = number(t.rms_opd_defocus / lambda_ref_mm);
        e["strehl_marechal"] = number(marechal_strehl(t.rms_opd, lambda_mm));
        e["tilt_mm"] = Json::array({number(t.tilt_x), number(t.tilt_y)});
        e["defocus_mm"] = number(t.defocus);
        e["centroid_mm"] = Json::array({number(t.centroid_x), number(t.centroid_y)});
        e["rms_spot_mm"] = number(t.rms_spot);
        e["chief_ray_mm"] = Json::array({number(t.chief_x), number(t.chief_y)});
        e["distortion_chief_percent"] = number(t.distortion_chief);
        e["distortion_centroid_percent"] = number(t.distortion_centroid);
        e["focus_tangential_mm"] = number(t.focus_tangential);
        e["focus_sagittal_mm"] = number(t.focus_sagittal);
        e["astigmatism_mm"] = number(t.astigmatism());
        tuples.push_back(std::move(e));
    }
    j["tuples"] = std::move(tuples);
    j["rays"] = r.result.status.size();
    j["failed_rays"] = r.result.failed_count();
    Json warnings = Json::array();
    for (const std::string& w : r.loaded.warnings) {
        warnings.push_back(w);
    }
    j["warnings"] = std::move(warnings);
    out << j.dump(2, ' ', false, Json::error_handler_t::replace) << '\n';
}

void write_text(const EvalReport& r, std::ostream& out) {
    const double lambda_ref_mm = r.result.reference_nm * 1e-6;
    out << "System: " << r.loaded.system.name << '\n';
    out << "  Datei: " << r.file << '\n';
    out << "  Raster: " << r.result.settings.grid << " x " << r.result.settings.grid << ", "
        << r.result.samples() << " Punkte in der Pupille"
        << (r.result.settings.half_pupil ? " (halbe Pupille)" : "") << '\n';
    out << "  Referenzsphäre: "
        << (r.result.settings.reference == EvalReference::Vertex
                ? std::string("durch den Scheitel der letzten Fläche")
                : "durch die Austrittspupille (z = " +
                      format_number("%.6f", r.result.exit_pupil_position[0]) +
                      " mm ab letzter Fläche)")
        << '\n';
    out << "  Backend: "
        << (r.gpu ? "GPU " + std::to_string(r.device) + " (" + r.device_name + "), FP64"
                  : std::string("CPU, FP64"))
        << '\n';
    out << "  OPD in Wellen bei " << format_number("%.3f", r.result.reference_nm)
        << " nm, Strehl nach Maréchal bei der Wellenlänge des Tupels, Längen in mm\n\n";
    out << "  " << pad_left("Feld", 4) << pad_left("Winkel", 8) << pad_left("λ [nm]", 9)
        << pad_left("versagt", 8) << pad_left("RMS-OPD", 11) << pad_left("ohne Kipp.", 11)
        << pad_left("Strehl", 8) << pad_left("RMS-Spot", 12) << pad_left("Schwerp. y", 13)
        << pad_left("Verz. [%]", 11) << pad_left("z_t", 12) << pad_left("z_s", 12) << '\n';
    for (const EvalTupleStats& t : r.stats) {
        const double lambda_nm = r.batch.wavelength_nm(t.wavelength);
        out << "  " << pad_left(std::to_string(t.field), 4)
            << pad_left(format_number("%.3f", field_angle_deg(r, t.field)), 8)
            << pad_left(format_number("%.2f", lambda_nm), 9)
            << pad_left(std::to_string(t.failed), 8)
            << pad_left(format_number("%.5f", t.rms_opd / lambda_ref_mm), 11)
            << pad_left(format_number("%.5f", t.rms_opd_tilt / lambda_ref_mm), 11)
            << pad_left(format_number("%.4f", marechal_strehl(t.rms_opd, lambda_nm * 1e-6)), 8)
            << pad_left(format_number("%.3e", t.rms_spot), 12)
            << pad_left(format_number("%.4e", t.centroid_y), 13)
            << pad_left(format_number("%.5f", t.distortion_chief), 11)
            << pad_left(format_number("%.4e", t.focus_tangential), 12)
            << pad_left(format_number("%.4e", t.focus_sagittal), 12) << '\n';
    }
    out << "  Versagte Strahlen: " << r.result.failed_count() << " von " << r.result.status.size()
        << '\n';
}

// OPD-Karte als CSV: N Zeilen mit N Werten in Wellen bei λ_ref (verlustfrei, 17 Stellen), erste
// Zeile = größtes y (wie ein Bild der Pupille), leere Felder außerhalb der Pupille und bei
// versagten Strahlen.
void write_map(const EvalReport& r, const EvalArgs& a) {
    if (a.map_field < 0 || a.map_field >= r.result.fields || a.map_wavelength < 0 ||
        a.map_wavelength >= r.result.wavelengths) {
        throw std::invalid_argument("--map " + std::to_string(a.map_field) + "," +
                                    std::to_string(a.map_wavelength) +
                                    ": Feld oder Wellenlänge existiert nicht");
    }
    const EvalMap m = eval_opd_map(r.result, 0, a.map_field, a.map_wavelength, a.removal);
    std::ofstream f(a.map_file);
    if (!f) {
        throw std::runtime_error("OPD-Karte: kann " + a.map_file + " nicht schreiben");
    }
    const double lambda_ref_mm = r.result.reference_nm * 1e-6;
    for (int j = m.size - 1; j >= 0; --j) {
        for (int i = 0; i < m.size; ++i) {
            if (i > 0) {
                f << ',';
            }
            const double v = m.at(i, j);
            if (std::isfinite(v)) {
                char buf[32];
                std::snprintf(buf, sizeof buf, "%.17g", v / lambda_ref_mm);
                f << buf;
            }
        }
        f << '\n';
    }
    if (!f) {
        throw std::runtime_error("OPD-Karte: Fehler beim Schreiben von " + a.map_file);
    }
}

} // namespace

void print_eval_usage(std::ostream& out) {
    out << "Aufruf: mikroptim eval <system.json> [--grid N] [--half] [--reference R] [--json]\n"
           "                      [--map F,L --map-file DATEI.csv [--remove piston|tilt|defocus]]\n"
           "                      [--gpu [--device N]]\n"
           "  Auswertung auf dem dichten kartesischen Pupillenraster (Modus EVAL, FP64):\n"
           "  RMS-OPD, Strehl nach Maréchal, RMS-Spot, Schwerpunkt, Verzeichnung und Brennlinien\n"
           "  pro Feld und Wellenlänge (Beschreibung: docs/cli.md).\n"
           "  --grid N        Raster N x N über die Pupille, 1 bis "
        << kMaxEvalGrid
        << " (Standard 64)\n"
           "  --half          nur die Hälfte x >= 0 rechnen (Symmetrie x -> -x)\n"
           "  --reference R   Referenzsphäre durch die Austrittspupille (exit-pupil, Standard)\n"
           "                  oder durch den Scheitel der letzten Fläche (vertex, wie trace)\n"
           "  --json          maschinenlesbare Ausgabe (Format mikroptim.eval/1)\n"
           "  --map F,L       OPD-Karte von Feld F und Wellenlänge L (ab 0) schreiben ...\n"
           "  --map-file D    ... als CSV nach D (Wellen bei der Referenzwellenlänge)\n"
           "  --remove X      vor der Karte abziehen: piston (Standard), tilt, defocus\n"
           "  --gpu           auf der GPU rechnen (Kernel EVAL) statt auf der CPU\n"
           "  --device N      CUDA-Gerät für --gpu (Standard 0)\n"
           "  --help          diese Hilfe\n"
           "Exit-Code: 0 Erfolg, 1 Fehler beim Laden oder Auswerten, 2 ungültige Kommandozeile,\n"
           "           3 GPU-Backend nicht verfügbar (ohne CUDA gebaut oder kein CUDA-Gerät).\n";
}

int run_eval_command(std::span<const std::string_view> args, std::ostream& out, std::ostream& err) {
    EvalArgs a;
    try {
        a = parse_args(args);
    } catch (const UsageError& e) {
        err << "mikroptim eval: " << e.what() << '\n';
        print_eval_usage(err);
        return kExitUsage;
    }
    if (a.help) {
        print_eval_usage(out);
        return kExitOk;
    }

#ifdef MIKROPTIM_HAS_CUDA
    std::optional<cuda::TraceEngine> engine;
    if (a.gpu) {
        const int count = cuda::device_count();
        if (count == 0) {
            err << "mikroptim eval: --gpu: kein CUDA-Gerät gefunden (GPU-Backend nicht "
                   "verfügbar)\n";
            return kExitNoGpu;
        }
        try {
            engine.emplace(a.device);
        } catch (const std::exception& e) {
            err << "mikroptim eval: Fehler: GPU " << a.device << ": " << e.what() << '\n';
            return kExitError;
        }
    }
#else
    if (a.gpu) {
        err << "mikroptim eval: --gpu: ohne CUDA gebaut (MIKROPTIM_ENABLE_CUDA=OFF), "
               "GPU-Backend nicht verfügbar\n";
        return kExitNoGpu;
    }
#endif

    try {
        EvalReport r;
        r.file = a.file;
        r.loaded = load_system(std::filesystem::path(a.file));
        for (const std::string& w : r.loaded.warnings) {
            err << "mikroptim eval: Warnung: " << w << '\n';
        }
        r.batch = make_system_batch(r.loaded, 1);
#ifdef MIKROPTIM_HAS_CUDA
        if (engine) {
            r.gpu = true;
            r.device = engine->device();
            r.device_name = engine->device_name();
            engine->evaluate(r.batch, a.settings, r.result);
        } else {
            r.result = eval_cpu(r.batch, a.settings);
        }
#else
        r.result = eval_cpu(r.batch, a.settings);
#endif
        r.stats = eval_statistics(r.result, 0);
        if (!a.map_file.empty()) {
            write_map(r, a);
        }
        if (a.json) {
            write_json(r, out);
        } else {
            write_text(r, out);
        }
        out.flush();
        if (!out) {
            err << "mikroptim eval: Fehler beim Schreiben der Ausgabe\n";
            return kExitError;
        }
        return kExitOk;
    } catch (const std::exception& e) {
        err << "mikroptim eval: Fehler: " << e.what() << '\n';
        return kExitError;
    }
}

} // namespace mko::cli
