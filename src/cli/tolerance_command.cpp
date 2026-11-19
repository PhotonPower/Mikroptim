// Unterbefehl `mikroptim tolerance` (M8-T4): Laden, Batch mit einer Variante, Monte-Carlo-
// Tolerierung (run_tolerance_analysis) auf der CPU oder mit --gpu über die TraceEngine und
// optimize_lm_gpu, Ausgabe als Text oder JSON, optional Histogramm und Stichproben als CSV.
// Ausgabeformate: docs/cli.md.

#include "cli/tolerance_command.hpp"
#include "cli/cli_util.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/model/tolerances.hpp"
#include "mikroptim/tolerance/tolerance_analysis.hpp"
#include "mikroptim/version.hpp"

#ifdef MIKROPTIM_HAS_CUDA
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/tolerance.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#endif

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mko::cli {

namespace {

using Json = nlohmann::ordered_json;

struct ToleranceArgs {
    std::string file;
    bool json = false;
    bool help = false;
    bool gpu = false;
    bool have_device = false;
    int device = 0;
    ToleranceSettings settings{};
    std::string histogram_file; // leer: kein Histogramm
    int bins = 20;
    std::string samples_file; // leer: keine Werte pro Stichprobe
};

std::string_view next_value(std::span<const std::string_view> args, std::size_t& i,
                            std::string_view option) {
    if (i + 1 >= args.size()) {
        throw UsageError(std::string(option) + " erwartet einen Wert");
    }
    return args[++i];
}

ToleranceArgs parse_args(std::span<const std::string_view> args) {
    ToleranceArgs a;
    bool have_file = false;
    bool have_bins = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--help" || arg == "-h") {
            a.help = true;
        } else if (arg == "--json") {
            a.json = true;
        } else if (arg == "--gpu") {
            a.gpu = true;
        } else if (arg == "--device") {
            a.device = parse_int(arg, next_value(args, i, arg));
            if (a.device < 0) {
                throw UsageError("--device muss >= 0 sein");
            }
            a.have_device = true;
        } else if (arg == "--samples") {
            a.settings.samples = parse_int(arg, next_value(args, i, arg));
            if (a.settings.samples < 1) {
                throw UsageError("--samples muss >= 1 sein");
            }
        } else if (arg == "--seed") {
            a.settings.seed = parse_uint64(arg, next_value(args, i, arg));
        } else if (arg == "--compensator") {
            const std::string_view v = next_value(args, i, arg);
            if (v == "wd" || v == "working_distance") {
                a.settings.compensator = CompensatorType::WorkingDistance;
            } else if (v == "none") {
                a.settings.compensator = CompensatorType::None;
            } else {
                throw UsageError("--compensator erwartet wd oder none, nicht '" + std::string(v) +
                                 "'");
            }
        } else if (arg == "--threshold") {
            a.settings.strehl_threshold = parse_double(arg, next_value(args, i, arg));
            if (!(a.settings.strehl_threshold >= 0.0 && a.settings.strehl_threshold <= 1.0)) {
                throw UsageError("--threshold muss in [0, 1] liegen");
            }
        } else if (arg == "--rings") {
            a.settings.quadrature.rings = parse_int(arg, next_value(args, i, arg));
            if (a.settings.quadrature.rings < 1 || a.settings.quadrature.rings > 16) {
                throw UsageError("--rings muss in [1, 16] liegen");
            }
        } else if (arg == "--spokes") {
            a.settings.quadrature.spokes = parse_int(arg, next_value(args, i, arg));
            if (a.settings.quadrature.spokes < 1) {
                throw UsageError("--spokes muss >= 1 sein");
            }
        } else if (arg == "--chunk") {
            a.settings.chunk = parse_int(arg, next_value(args, i, arg));
            if (a.settings.chunk < 1) {
                throw UsageError("--chunk muss >= 1 sein");
            }
        } else if (arg == "--max-iterations") {
            a.settings.lm.max_iterations = parse_int(arg, next_value(args, i, arg));
            if (a.settings.lm.max_iterations < 1) {
                throw UsageError("--max-iterations muss >= 1 sein");
            }
        } else if (arg == "--histogram") {
            a.histogram_file = std::string(next_value(args, i, arg));
        } else if (arg == "--bins") {
            a.bins = parse_int(arg, next_value(args, i, arg));
            if (a.bins < 1 || a.bins > 10000) {
                throw UsageError("--bins muss in [1, 10000] liegen");
            }
            have_bins = true;
        } else if (arg == "--samples-file") {
            a.samples_file = std::string(next_value(args, i, arg));
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
    if (have_bins && a.histogram_file.empty()) {
        throw UsageError("--bins verlangt --histogram");
    }
    return a;
}

// Nicht endliche Zahlen sind in JSON nicht darstellbar und werden null.
Json number(double v) {
    return std::isfinite(v) ? Json(v) : Json(nullptr);
}

Json stats_json(const DistributionStats& s) {
    Json j;
    j["count"] = s.count;
    j["mean"] = number(s.mean);
    j["std"] = number(s.std);
    j["min"] = number(s.min);
    j["max"] = number(s.max);
    j["p50"] = number(s.p50);
    j["p90"] = number(s.p90);
    j["p95"] = number(s.p95);
    j["p99"] = number(s.p99);
    return j;
}

// Einheit der aufgelösten Grenze (ResolvedTolerance::limit).
const char* limit_unit(const ResolvedTolerance& t) {
    switch (t.type) {
    case ToleranceType::Radius:
        return t.radius_fringes ? "1/mm" : "1";
    case ToleranceType::Thickness:
    case ToleranceType::Decenter:
    case ToleranceType::ElementDecenter:
        return "mm";
    case ToleranceType::Tilt:
    case ToleranceType::ElementTilt:
        return "rad";
    default:
        return "1";
    }
}

const char* compensator_text(CompensatorType c) {
    return c == CompensatorType::WorkingDistance ? "Arbeitsabstand (LM pro Stichprobe)" : "keiner";
}

struct ToleranceReport {
    std::string file;
    LoadedSystem loaded;
    std::vector<ResolvedTolerance> tolerances;
    ToleranceResult result;
    bool gpu = false;
    int device = -1;
    std::string device_name;
    double seconds = 0.0;
    int rings = 0;
    int spokes = 0;
    int chunk = 0;
};

double field_angle_deg(const ToleranceReport& r, int f) {
    const SystemDescription& sys = r.loaded.system;
    return sys.source.type == SourceType::Collimated
               ? sys.source.field_angles_deg[static_cast<std::size_t>(f)]
               : std::nan("");
}

std::size_t yield_count(const ToleranceResult& res) {
    std::size_t good = 0;
    for (const ToleranceSampleResult& s : res.results) {
        if (s.strehl_min >= res.strehl_threshold) {
            ++good;
        }
    }
    return good;
}

void write_json(const ToleranceReport& r, std::ostream& out) {
    const SystemDescription& sys = r.loaded.system;
    const ToleranceResult& res = r.result;
    Json j;
    j["format"] = std::string(kToleranceFormat);
    j["mikroptim_version"] = kVersion;
    j["system"] = sys.name;
    j["file"] = r.file;
    j["backend"] = r.gpu ? "gpu" : "cpu";
    if (r.gpu) {
        j["device"] = Json{{"index", r.device}, {"name", r.device_name}};
    } else {
        j["device"] = nullptr;
    }
    j["settings"] = Json{{"samples", res.samples},
                         {"seed", res.seed},
                         {"compensator", std::string(to_string(res.compensator))},
                         {"strehl_threshold", res.strehl_threshold},
                         {"rings", r.rings},
                         {"spokes", r.spokes},
                         {"chunk", r.chunk}};
    Json tol = Json::array();
    for (const ResolvedTolerance& t : r.tolerances) {
        Json e;
        e["type"] = std::string(to_string(t.type));
        e["item"] = t.item;
        e["surface"] = t.surface >= 0 ? Json(t.surface) : Json(nullptr);
        e["element"] = t.first >= 0 ? Json::array({t.first, t.last}) : Json(nullptr);
        e["limit"] = t.limit;
        e["unit"] = limit_unit(t);
        e["distribution"] = std::string(to_string(t.distribution));
        e["cutoff_sigma"] =
            t.distribution == ToleranceDistribution::Normal ? Json(t.cutoff) : Json(nullptr);
        e["pivot_z_mm"] = t.type == ToleranceType::ElementTilt ? Json(t.pivot_z) : Json(nullptr);
        tol.push_back(std::move(e));
    }
    j["tolerances"] = std::move(tol);
    j["draws"] = res.draws;
    j["reference_nm"] = res.reference_nm;
    j["nominal"] = Json{{"rms_waves", number(res.nominal.rms_composite)},
                        {"strehl_min", number(res.nominal.strehl_min)},
                        {"working_distance_mm", number(res.nominal.working_distance)},
                        {"compensation_mm", number(res.nominal.compensation)},
                        {"failed_rays", res.nominal.failed_rays}};
    j["rms_waves"] = stats_json(res.rms_composite);
    j["strehl_min"] = stats_json(res.strehl_min);
    j["compensation_mm"] = stats_json(res.compensation);
    j["yield"] = number(res.yield);
    j["yield_samples"] = yield_count(res);
    j["failed_samples"] = res.failed_samples;
    Json tuples = Json::array();
    for (int f = 0; f < res.fields; ++f) {
        for (int l = 0; l < res.wavelengths; ++l) {
            const auto t = static_cast<std::size_t>(f * res.wavelengths + l);
            Json e;
            e["field"] = f;
            e["field_angle_deg"] = number(field_angle_deg(r, f));
            e["wavelength"] = l;
            e["wavelength_nm"] = res.wavelength_nm[static_cast<std::size_t>(l)];
            e["weight"] = res.tuple_weight[t];
            e["nominal_rms_waves"] = number(res.nominal.rms[t]);
            e["nominal_strehl"] = number(res.nominal.strehl[t]);
            e["rms_waves"] = stats_json(res.rms[t]);
            e["strehl"] = stats_json(res.strehl[t]);
            tuples.push_back(std::move(e));
        }
    }
    j["tuples"] = std::move(tuples);
    j["chunks"] = res.chunks;
    j["elapsed_seconds"] = r.seconds;
    Json warnings = Json::array();
    for (const std::string& w : r.loaded.warnings) {
        warnings.push_back(w);
    }
    j["warnings"] = std::move(warnings);
    out << j.dump(2, ' ', false, Json::error_handler_t::replace) << '\n';
}

std::string f6(double v) {
    return format_number("%.6f", v);
}

void write_text(const ToleranceReport& r, std::ostream& out) {
    const ToleranceResult& res = r.result;
    std::size_t items = 0;
    if (r.loaded.system.tolerances) {
        items = r.loaded.system.tolerances->items.size();
    }
    out << "System: " << r.loaded.system.name << '\n';
    out << "  Datei: " << r.file << '\n';
    out << "  Toleranzsatz: " << items << " Einträge, " << r.tolerances.size() << " Toleranzen, "
        << res.draws << " Zufallsgrößen pro Stichprobe\n";
    out << "  Stichproben: " << res.samples << " (Startwert " << res.seed << ", " << res.chunks
        << (res.chunks == 1 ? " Tranche" : " Tranchen") << " zu höchstens " << r.chunk << ")\n";
    out << "  Kompensator: " << compensator_text(res.compensator) << '\n';
    out << "  Backend: "
        << (r.gpu ? "GPU " + std::to_string(r.device) + " (" + r.device_name + "), FP64"
                  : std::string("CPU, FP64"))
        << '\n';
    out << "  Quadratur " << r.rings << " Ringe x " << r.spokes
        << " Speichen, volle Pupille; RMS nach Abzug von Kolben und Kippung in Wellen bei "
        << format_number("%.3f", res.reference_nm)
        << " nm, Strehl bei der Wellenlänge des Tupels\n";
    out << "  Laufzeit: " << format_number("%.2f", r.seconds) << " s\n\n";

    out << "  " << pad_right("Kennzahl", 22) << pad_left("Nominal", 11) << pad_left("Mittel", 11)
        << pad_left("Std", 11) << pad_left("50 %", 11) << pad_left("90 %", 11)
        << pad_left("95 %", 11) << pad_left("99 %", 11) << '\n';
    const auto row = [&out](const char* name, double nominal, const DistributionStats& s) {
        out << "  " << pad_right(name, 22) << pad_left(f6(nominal), 11) << pad_left(f6(s.mean), 11)
            << pad_left(f6(s.std), 11) << pad_left(f6(s.p50), 11) << pad_left(f6(s.p90), 11)
            << pad_left(f6(s.p95), 11) << pad_left(f6(s.p99), 11) << '\n';
    };
    row("RMS gesamt [Wellen]", res.nominal.rms_composite, res.rms_composite);
    row("Strehl (kleinster)", res.nominal.strehl_min, res.strehl_min);
    out << "  (RMS: Wert, den p % der Stichproben nicht überschreiten; Strehl: Wert, den "
           "mindestens p % erreichen)\n";
    if (res.compensator == CompensatorType::WorkingDistance) {
        out << "  Arbeitsabstand nominal " << f6(res.nominal.working_distance)
            << " mm, Kompensation [mm]: Mittel " << format_number("%.3e", res.compensation.mean)
            << ", Std " << format_number("%.3e", res.compensation.std) << ", min "
            << format_number("%.3e", res.compensation.min) << ", max "
            << format_number("%.3e", res.compensation.max) << '\n';
    }
    const std::size_t good = yield_count(res);
    out << "  Ausbeute (Strehl >= " << format_number("%.2f", res.strehl_threshold)
        << " in allen Tupeln): " << format_number("%.2f", 100.0 * res.yield) << " % (" << good
        << " von " << res.samples << ")\n";
    out << "  Stichproben mit versagten Strahlen: " << res.failed_samples << '\n';

    out << "\n  Pro Feld und Wellenlänge (RMS in Wellen, Perzentile wie oben)\n";
    out << "  " << pad_left("Feld", 4) << pad_left("Winkel", 8) << pad_left("λ [nm]", 9)
        << pad_left("RMS nom.", 10) << pad_left("Mittel", 10) << pad_left("95 %", 10)
        << pad_left("Strehl nom.", 12) << pad_left("Mittel", 10) << pad_left("95 %", 10) << '\n';
    for (int f = 0; f < res.fields; ++f) {
        for (int l = 0; l < res.wavelengths; ++l) {
            const auto t = static_cast<std::size_t>(f * res.wavelengths + l);
            out << "  " << pad_left(std::to_string(f), 4)
                << pad_left(format_number("%.3f", field_angle_deg(r, f)), 8)
                << pad_left(format_number("%.2f", res.wavelength_nm[static_cast<std::size_t>(l)]),
                            9)
                << pad_left(format_number("%.5f", res.nominal.rms[t]), 10)
                << pad_left(format_number("%.5f", res.rms[t].mean), 10)
                << pad_left(format_number("%.5f", res.rms[t].p95), 10)
                << pad_left(format_number("%.5f", res.nominal.strehl[t]), 12)
                << pad_left(format_number("%.5f", res.strehl[t].mean), 10)
                << pad_left(format_number("%.5f", res.strehl[t].p95), 10) << '\n';
        }
    }
}

std::ofstream open_output(const std::string& file, const char* what) {
    std::ofstream f(file);
    if (!f) {
        throw std::runtime_error(std::string(what) + ": kann " + file + " nicht schreiben");
    }
    return f;
}

std::string csv_number(double v) {
    if (!std::isfinite(v)) {
        return {};
    }
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
}

// Histogramm des kleinsten Strehl über [0, 1] und des RMS gesamt über [0, größter Wert] mit je
// `bins` gleich breiten Klassen [unten, oben) (die letzte schließt den oberen Rand ein); NaN
// zählt nicht.
void write_histogram(const ToleranceResult& res, const std::string& file, int bins) {
    std::ofstream f = open_output(file, "Histogramm");
    f << "quantity,bin,lower,upper,count,fraction\n";
    const auto emit = [&](const char* name, double upper, auto get) {
        std::vector<std::size_t> count(static_cast<std::size_t>(bins), 0);
        const double width = upper > 0.0 ? upper / bins : 1.0 / bins;
        for (const ToleranceSampleResult& s : res.results) {
            const double v = get(s);
            if (!std::isfinite(v)) {
                continue;
            }
            int b = static_cast<int>(std::floor(v / width));
            b = std::clamp(b, 0, bins - 1);
            ++count[static_cast<std::size_t>(b)];
        }
        for (int b = 0; b < bins; ++b) {
            const std::size_t c = count[static_cast<std::size_t>(b)];
            f << name << ',' << b << ',' << csv_number(b * width) << ','
              << csv_number(b + 1 == bins ? bins * width : (b + 1) * width) << ',' << c << ','
              << csv_number(static_cast<double>(c) / static_cast<double>(res.results.size()))
              << '\n';
        }
    };
    emit("strehl_min", 1.0, [](const ToleranceSampleResult& s) { return s.strehl_min; });
    double max_rms = 0.0;
    for (const ToleranceSampleResult& s : res.results) {
        if (std::isfinite(s.rms_composite)) {
            max_rms = std::max(max_rms, s.rms_composite);
        }
    }
    emit("rms_waves", max_rms, [](const ToleranceSampleResult& s) { return s.rms_composite; });
    if (!f) {
        throw std::runtime_error("Histogramm: Fehler beim Schreiben von " + file);
    }
}

// Werte pro Stichprobe: Index, RMS gesamt, kleinster Strehl, Arbeitsabstand, Kompensation,
// versagte Strahlen, Status des Kompensators, dann RMS und Strehl pro Tupel.
void write_samples(const ToleranceResult& res, const std::string& file) {
    std::ofstream f = open_output(file, "Stichproben");
    f << "index,rms_waves,strehl_min,working_distance_mm,compensation_mm,failed_rays,"
         "compensator_status";
    for (int fi = 0; fi < res.fields; ++fi) {
        for (int l = 0; l < res.wavelengths; ++l) {
            f << ",rms_f" << fi << "_l" << l << ",strehl_f" << fi << "_l" << l;
        }
    }
    f << '\n';
    for (std::size_t i = 0; i < res.results.size(); ++i) {
        const ToleranceSampleResult& s = res.results[i];
        f << i << ',' << csv_number(s.rms_composite) << ',' << csv_number(s.strehl_min) << ','
          << csv_number(s.working_distance) << ',' << csv_number(s.compensation) << ','
          << s.failed_rays << ','
          << (res.compensator == CompensatorType::WorkingDistance ? to_string(s.compensator_status)
                                                                  : "none");
        for (std::size_t t = 0; t < s.rms.size(); ++t) {
            f << ',' << csv_number(s.rms[t]) << ',' << csv_number(s.strehl[t]);
        }
        f << '\n';
    }
    if (!f) {
        throw std::runtime_error("Stichproben: Fehler beim Schreiben von " + file);
    }
}

} // namespace

void print_tolerance_usage(std::ostream& out) {
    out << "Aufruf: mikroptim tolerance <system.json> [--samples N] [--seed n]\n"
           "                           [--compensator wd|none] [--threshold S] [--json]\n"
           "                           [--histogram DATEI.csv [--bins K]] [--samples-file "
           "DATEI.csv]\n"
           "                           [--rings N] [--spokes M] [--chunk K] [--max-iterations N]\n"
           "                           [--gpu [--device N]]\n"
           "  Monte-Carlo-Tolerierung mit dem Toleranzsatz der Systemdatei (Abschnitt "
           "tolerances):\n"
           "  Stichproben als Varianten, Kompensator Arbeitsabstand, Verteilung von "
           "RMS-Wellenfront\n"
           "  und Strehl pro Feld und Wellenlänge (Beschreibung: docs/cli.md).\n"
           "  --samples N       Zahl der Stichproben (Standard 1000)\n"
           "  --seed n          Startwert; Stichprobe i hängt nur von n und i ab (Standard 0)\n"
           "  --compensator X   wd (Arbeitsabstand pro Stichprobe neu optimieren) oder none\n"
           "                    (Standard: Datei, sonst wd)\n"
           "  --threshold S     Strehl-Schwelle der Ausbeute (Standard 0.8)\n"
           "  --json            maschinenlesbare Ausgabe (Format mikroptim.tolerance/1)\n"
           "  --histogram D     Histogramm von kleinstem Strehl und RMS gesamt als CSV nach D\n"
           "  --bins K          Klassen des Histogramms (Standard 20)\n"
           "  --samples-file D  Werte pro Stichprobe als CSV nach D\n"
           "  --rings N         radiale Ringe der Pupillenquadratur (Standard 10)\n"
           "  --spokes M        Speichen der Pupillenquadratur (Standard 8)\n"
           "  --chunk K         Stichproben pro Tranche (Standard 4096; ändert keine Ergebnisse)\n"
           "  --max-iterations N  Zyklen des LM pro Stichprobe beim Kompensator (Standard 100)\n"
           "  --gpu             auf der GPU rechnen (RESIDUALS und LM) statt auf der CPU\n"
           "  --device N        CUDA-Gerät für --gpu (Standard 0)\n"
           "  --help            diese Hilfe\n"
           "Exit-Code: 0 Erfolg, 1 Fehler beim Laden oder Rechnen (auch: kein Toleranzsatz),\n"
           "           2 ungültige Kommandozeile, 3 GPU-Backend nicht verfügbar.\n";
}

int run_tolerance_command(std::span<const std::string_view> args, std::ostream& out,
                          std::ostream& err) {
    ToleranceArgs a;
    try {
        a = parse_args(args);
    } catch (const UsageError& e) {
        err << "mikroptim tolerance: " << e.what() << '\n';
        print_tolerance_usage(err);
        return kExitUsage;
    }
    if (a.help) {
        print_tolerance_usage(out);
        return kExitOk;
    }

#ifdef MIKROPTIM_HAS_CUDA
    std::optional<cuda::TraceEngine> engine;
    if (a.gpu) {
        const int count = cuda::device_count();
        if (count == 0) {
            err << "mikroptim tolerance: --gpu: kein CUDA-Gerät gefunden (GPU-Backend nicht "
                   "verfügbar)\n";
            return kExitNoGpu;
        }
        if (a.device >= count) {
            err << "mikroptim tolerance: Fehler: --device " << a.device
                << ": Gerät existiert nicht (siehe mikroptim --devices)\n";
            return kExitError;
        }
        try {
            engine.emplace(a.device);
        } catch (const std::exception& e) {
            err << "mikroptim tolerance: Fehler: GPU " << a.device << ": " << e.what() << '\n';
            return kExitError;
        }
    }
#else
    if (a.gpu) {
        err << "mikroptim tolerance: --gpu: ohne CUDA gebaut (MIKROPTIM_ENABLE_CUDA=OFF), "
               "GPU-Backend nicht verfügbar\n";
        return kExitNoGpu;
    }
#endif

    try {
        ToleranceReport r;
        r.file = a.file;
        r.loaded = load_system(std::filesystem::path(a.file));
        for (const std::string& w : r.loaded.warnings) {
            err << "mikroptim tolerance: Warnung: " << w << '\n';
        }
        if (!r.loaded.system.tolerances) {
            throw std::invalid_argument("keine Toleranzen: die Systemdatei hat keinen Abschnitt "
                                        "\"tolerances\" (docs/dateiformat.md)");
        }
        r.tolerances = resolve_tolerances(r.loaded.system);
        r.rings = a.settings.quadrature.rings;
        r.spokes = a.settings.quadrature.spokes;
        r.chunk = a.settings.chunk;
        const SystemBatchHost batch = make_system_batch(r.loaded, 1);
#ifdef MIKROPTIM_HAS_CUDA
        if (engine) {
            r.gpu = true;
            r.device = engine->device();
            r.device_name = engine->device_name();
            cuda::use_gpu(a.settings, *engine);
        }
#endif
        const auto start = std::chrono::steady_clock::now();
        r.result = run_tolerance_analysis(r.loaded.system, batch, a.settings);
        r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (!a.histogram_file.empty()) {
            write_histogram(r.result, a.histogram_file, a.bins);
        }
        if (!a.samples_file.empty()) {
            write_samples(r.result, a.samples_file);
        }
        if (a.json) {
            write_json(r, out);
        } else {
            write_text(r, out);
        }
        out.flush();
        if (!out) {
            err << "mikroptim tolerance: Fehler beim Schreiben der Ausgabe\n";
            return kExitError;
        }
        return kExitOk;
    } catch (const std::exception& e) {
        err << "mikroptim tolerance: Fehler: " << e.what() << '\n';
        return kExitError;
    }
}

} // namespace mko::cli
