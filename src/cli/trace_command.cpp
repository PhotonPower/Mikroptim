// Unterbefehl `mikroptim trace`: Laden, Paraxialdaten, Wellenfront (CPU oder mit --gpu über
// die TraceEngine), Ausgabe als Text oder JSON. Ausgabeformate und JSON-Schema: docs/cli.md.

#include "cli/trace_command.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/merit/wavefront.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/version.hpp"

#ifdef MIKROPTIM_HAS_CUDA
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#endif

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mko::cli {

namespace {

using Json = nlohmann::ordered_json;

constexpr int kDefaultRings = 10;
constexpr int kDefaultSpokes = 8;

struct TraceArgs {
    std::string file;
    bool json = false;
    bool help = false;
    bool gpu = false;
    bool have_device = false; // --device angegeben
    int device = 0;
    bool df64 = false; // --precision df64 (nur mit --gpu)
    int rings = kDefaultRings;
    int spokes = kDefaultSpokes;
};

TraceArgs parse_args(std::span<const std::string_view> args) {
    TraceArgs a;
    bool have_file = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--help" || arg == "-h") {
            a.help = true;
        } else if (arg == "--json") {
            a.json = true;
        } else if (arg == "--gpu") {
            a.gpu = true;
        } else if (arg == "--device") {
            if (i + 1 >= args.size()) {
                throw UsageError("--device erwartet einen Wert");
            }
            a.device = parse_int(arg, args[++i]);
            if (a.device < 0) {
                throw UsageError("--device muss >= 0 sein");
            }
            a.have_device = true;
        } else if (arg == "--precision") {
            if (i + 1 >= args.size()) {
                throw UsageError("--precision erwartet einen Wert (fp64 oder df64)");
            }
            const std::string_view value = args[++i];
            if (value == "fp64") {
                a.df64 = false;
            } else if (value == "df64") {
                a.df64 = true;
            } else {
                throw UsageError("--precision erwartet fp64 oder df64, nicht '" +
                                 std::string(value) + "'");
            }
        } else if (arg == "--rings" || arg == "--spokes") {
            if (i + 1 >= args.size()) {
                throw UsageError(std::string(arg) + " erwartet einen Wert");
            }
            const int value = parse_int(arg, args[++i]);
            if (arg == "--rings") {
                if (value < 1 || value > kMaxPupilRings) {
                    throw UsageError("--rings muss in [1, " + std::to_string(kMaxPupilRings) +
                                     "] liegen");
                }
                a.rings = value;
            } else {
                if (value < 1) {
                    throw UsageError("--spokes muss >= 1 sein");
                }
                a.spokes = value;
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
    if (a.df64 && !a.gpu) {
        throw UsageError("--precision df64 verlangt --gpu (die CPU rechnet immer in FP64)");
    }
    return a;
}

// Wellenfront eines Tupels (Feld f, Wellenlänge l), unabhängig vom Backend.
struct TupleReport {
    int field = 0;
    int wavelength = 0;
    double wavelength_nm = 0.0;
    Vec3<double> image_point{0.0, 0.0, 0.0};
    PupilMode pupil_mode = PupilMode::Axis;
    int rays = 0;
    int failed = 0;
    double rms_waves = WavefrontMap::kNaN;
    double rms = WavefrontMap::kNaN;
    double piston = WavefrontMap::kNaN; // nur CPU; die Engine liefert keinen Kolben
};

// Wellenfront aller Tupel, Ablage [f · L + l] wie WavefrontResult::maps.
struct WavefrontReport {
    double focal_length = WavefrontMap::kNaN; // f_soll; NaN bei der Punktquelle
    std::vector<TupleReport> tuples;
    std::size_t rays = 0;
    std::size_t failed = 0;
};

// Rechenwerk der Wellenfront (die Paraxialdaten entstehen immer auf dem Host).
struct BackendInfo {
    bool gpu = false;
    int device = -1;         // nur GPU
    std::string device_name; // nur GPU
    bool df64 = false;       // Rechentyp der Wellenfront: df64 (nur GPU) statt FP64
};

// Alles, was ausgegeben wird.
struct TraceReport {
    std::string file; // wie auf der Kommandozeile angegeben
    LoadedSystem loaded;
    ParaxialData paraxial;
    WavefrontOptions options;
    BackendInfo backend;
    std::optional<WavefrontReport> wavefront; // leer: nicht auswertbar (wavefront_error)
    std::string wavefront_error;
};

// CPU-Ergebnis (evaluate_wavefront) in die Ausgabestruktur.
WavefrontReport report_from(const WavefrontResult& w) {
    WavefrontReport r;
    r.focal_length = w.focal_length;
    r.tuples.reserve(w.maps.size());
    for (const WavefrontMap& m : w.maps) {
        TupleReport t;
        t.field = m.field;
        t.wavelength = m.wavelength;
        t.wavelength_nm = m.wavelength_nm;
        t.image_point = m.image_point;
        t.pupil_mode = m.quadrature.mode;
        t.rays = m.samples();
        t.failed = m.failed;
        t.rms_waves = m.rms_waves;
        t.rms = m.rms;
        t.piston = m.piston;
        r.tuples.push_back(t);
        r.rays += static_cast<std::size_t>(m.samples());
    }
    r.failed = w.failed_count();
    return r;
}

#ifdef MIKROPTIM_HAS_CUDA

// Wellenfront über die TraceEngine (Kernel RESIDUALS mit OPD und Status pro Strahl) mit
// denselben idealen Bildpunkten wie evaluate_wavefront(LoadedSystem): f_soll nach
// target_focal_length (nur kollimiert; afokal ohne target.focal_length wirft wie im CPU-Pfad
// std::invalid_argument), h_f = f_soll · tan θ_f, w = Dicke der letzten Fläche. Die Engine
// liefert keinen Kolben (piston bleibt NaN).
WavefrontReport evaluate_wavefront_gpu(const LoadedSystem& loaded, const WavefrontOptions& options,
                                       cuda::TraceEngine& engine, bool df64) {
    const SystemDescription& sys = loaded.system;
    std::optional<double> focal_length;
    if (sys.source.type == SourceType::Collimated) {
        focal_length = target_focal_length(sys, loaded.catalog);
    }
    const SystemBatchHost batch = make_system_batch(sys, loaded.index, 1, focal_length);

    cuda::TraceSettings settings;
    settings.quadrature.rings = options.rings;
    settings.quadrature.spokes = options.spokes;
    // Auswertung wie der CPU-Pfad (evaluate_wavefront): hartes Versagen, ohne Strafresiduen
    settings.quadrature.failure = FailureMode::Hard;
    settings.opd = true;
    settings.status = true;
    settings.precision = df64 ? cuda::Precision::Df64 : cuda::Precision::Fp64;
    const cuda::TraceOutput out = engine.run(batch, cuda::TraceMode::Residuals, settings);
    const std::vector<TupleWavefront> tuples = tuple_wavefronts(out.residuals, out.layout, 0);

    WavefrontReport r;
    r.focal_length = focal_length ? *focal_length : WavefrontMap::kNaN;
    const double lambda_ref_mm = sys.spectrum.reference_nm * 1e-6; // wie evaluate_wavefront
    const double w = batch.image_distance(0);
    r.tuples.reserve(tuples.size());
    for (const TupleWavefront& tw : tuples) {
        TupleReport t;
        t.field = tw.field;
        t.wavelength = tw.wavelength;
        t.wavelength_nm = batch.wavelength_nm(tw.wavelength);
        t.image_point = {0.0, batch.field_target(0, tw.field), w};
        t.pupil_mode = out.layout.pupil_mode(tw.field);
        t.rays = tw.samples;
        t.failed = tw.failed;
        t.rms = tw.rms;
        t.rms_waves = tw.valid() ? tw.rms / lambda_ref_mm : WavefrontMap::kNaN;
        r.tuples.push_back(t);
        r.rays += static_cast<std::size_t>(tw.samples);
        r.failed += static_cast<std::size_t>(tw.failed);
    }
    return r;
}

#endif

const char* pupil_mode_name(PupilMode mode) {
    switch (mode) {
    case PupilMode::Half:
        return "half";
    case PupilMode::Full:
        return "full";
    case PupilMode::Axis:
        return "axis";
    }
    return "unknown";
}

// ------------------------------------------------------------------------------------------
// JSON
// ------------------------------------------------------------------------------------------

// Nicht endliche Zahlen sind in JSON nicht darstellbar und werden null (docs/cli.md).
Json number(double v) {
    if (!std::isfinite(v)) {
        return Json(nullptr);
    }
    return Json(v);
}

Json vec3(const Vec3<double>& v) {
    return Json::array({number(v.x), number(v.y), number(v.z)});
}

Json paraxial_json(const ParaxialData& p, double wavelength_nm) {
    Json j;
    j["wavelength_nm"] = number(wavelength_nm);
    j["afocal"] = p.afocal;
    j["power"] = number(p.power);
    j["focal_length"] = number(p.focal_length);
    j["back_focal_distance"] = number(p.back_focal_distance);
    j["front_focal_distance"] = number(p.front_focal_distance);
    j["front_principal_plane"] = number(p.front_principal_plane);
    j["rear_principal_plane"] = number(p.rear_principal_plane);
    j["image_defocus"] = number(p.image_defocus);
    j["petzval_sum"] = number(p.petzval_sum);
    j["petzval_curvature"] = number(p.petzval_curvature);
    j["petzval_radius"] = number(p.petzval_radius);
    Json ep;
    ep["at_infinity"] = p.exit_pupil_at_infinity;
    ep["position"] = number(p.exit_pupil_position);
    ep["magnification"] = number(p.exit_pupil_magnification);
    ep["semi_diameter"] = number(p.exit_pupil_semi_diameter);
    j["exit_pupil"] = std::move(ep);
    j["abcd"] =
        Json::array({number(p.abcd[0]), number(p.abcd[1]), number(p.abcd[2]), number(p.abcd[3])});
    return j;
}

Json source_json(const Source& s) {
    Json j;
    if (s.type == SourceType::Collimated) {
        j["type"] = "collimated";
        Json angles = Json::array();
        for (const double a : s.field_angles_deg) {
            angles.push_back(number(a));
        }
        j["field_angles_deg"] = std::move(angles);
    } else {
        j["type"] = "point";
        j["point"] = Json::array({number(s.point[0]), number(s.point[1]), number(s.point[2])});
        j["numerical_aperture"] = number(s.numerical_aperture);
    }
    return j;
}

void write_json(const TraceReport& r, std::ostream& out) {
    const SystemDescription& sys = r.loaded.system;
    Json j;
    j["format"] = std::string(kTraceFormat);
    j["mikroptim_version"] = kVersion;
    j["system"] = sys.name;
    j["file"] = r.file;
    j["reference_nm"] = number(sys.spectrum.reference_nm);
    Json wl = Json::array();
    for (const double w : sys.spectrum.wavelengths_nm) {
        wl.push_back(number(w));
    }
    j["wavelengths_nm"] = std::move(wl);
    j["source"] = source_json(sys.source);
    j["paraxial"] = paraxial_json(r.paraxial, sys.spectrum.reference_nm);
    j["quadrature"] = Json{{"rings", r.options.rings}, {"spokes", r.options.spokes}};
    j["backend"] = r.backend.gpu ? "gpu" : "cpu";
    if (r.backend.gpu) {
        j["device"] = Json{{"index", r.backend.device}, {"name", r.backend.device_name}};
    } else {
        j["device"] = nullptr;
    }
    j["precision"] = r.backend.df64 ? "df64" : "fp64";
    if (r.wavefront) {
        const WavefrontReport& w = *r.wavefront;
        j["target_focal_length"] = number(w.focal_length);
        Json entries = Json::array();
        for (const TupleReport& m : w.tuples) {
            Json e;
            e["field"] = m.field;
            if (sys.source.type == SourceType::Collimated) {
                e["field_angle_deg"] =
                    number(sys.source.field_angles_deg[static_cast<std::size_t>(m.field)]);
            } else {
                e["field_angle_deg"] = nullptr;
            }
            e["wavelength"] = m.wavelength;
            e["wavelength_nm"] = number(m.wavelength_nm);
            e["image_point"] = vec3(m.image_point);
            e["pupil_mode"] = pupil_mode_name(m.pupil_mode);
            e["rays"] = m.rays;
            e["failed_rays"] = m.failed;
            e["rms_waves"] = number(m.rms_waves);
            e["rms_mm"] = number(m.rms);
            e["piston_mm"] = number(m.piston);
            entries.push_back(std::move(e));
        }
        j["wavefront"] = std::move(entries);
        j["wavefront_error"] = nullptr;
        j["rays"] = w.rays;
        j["failed_rays"] = w.failed;
    } else {
        j["target_focal_length"] = nullptr;
        j["wavefront"] = nullptr;
        j["wavefront_error"] = r.wavefront_error;
        j["rays"] = nullptr;
        j["failed_rays"] = nullptr;
    }
    Json warnings = Json::array();
    for (const std::string& w : r.loaded.warnings) {
        warnings.push_back(w);
    }
    j["warnings"] = std::move(warnings);
    // Ungültiges UTF-8 (z. B. ein Dateiname in der ANSI-Codepage) wird ersetzt statt zu werfen.
    out << j.dump(2, ' ', false, Json::error_handler_t::replace) << '\n';
}

// ------------------------------------------------------------------------------------------
// Text
// ------------------------------------------------------------------------------------------

void write_text(const TraceReport& r, std::ostream& out) {
    const SystemDescription& sys = r.loaded.system;
    const ParaxialData& p = r.paraxial;
    const double ref = sys.spectrum.reference_nm;

    out << "System: " << sys.name << '\n';
    kv(out, "Datei", r.file);
    if (sys.source.type == SourceType::Collimated) {
        kv(out, "Quelle",
           "kollimiert, Feldwinkel " + join_fixed(sys.source.field_angles_deg, 3) + " deg");
    } else {
        const auto& q = sys.source.point;
        kv(out, "Quelle",
           "Punktquelle bei (" + fixed(q[0], 6) + ", " + fixed(q[1], 6) + ", " + fixed(q[2], 6) +
               ") mm, NA " + fixed(sys.source.numerical_aperture, 6));
    }
    kv(out, "Wellenlängen", join_fixed(sys.spectrum.wavelengths_nm, 3) + " nm");
    kv(out, "Referenzwellenlänge", fixed(ref, 3) + " nm");
    kv(out, "Flächen", std::to_string(sys.surfaces.size()));

    out << '\n' << "Paraxialdaten bei " << fixed(ref, 3) << " nm\n";
    std::string f = with_unit(fixed(p.focal_length), p.focal_length, "mm");
    if (p.afocal) {
        f += " (afokal)";
    }
    kv(out, "Brennweite f'", f);
    kv(out, "Brechkraft", with_unit(sci(p.power), p.power, "1/mm"));
    kv(out, "Hintere Schnittweite (ab letzter Fläche)",
       with_unit(fixed(p.back_focal_distance), p.back_focal_distance, "mm"));
    kv(out, "Vordere Schnittweite (ab Fläche 0)",
       with_unit(fixed(p.front_focal_distance), p.front_focal_distance, "mm"));
    kv(out, "Hauptebene H (ab Fläche 0)",
       with_unit(fixed(p.front_principal_plane), p.front_principal_plane, "mm"));
    kv(out, "Hauptebene H' (ab letzter Fläche)",
       with_unit(fixed(p.rear_principal_plane), p.rear_principal_plane, "mm"));
    kv(out, "Defokus der Bildebene", with_unit(fixed(p.image_defocus), p.image_defocus, "mm"));
    kv(out, "Petzvalsumme", with_unit(sci(p.petzval_sum), p.petzval_sum, "1/mm"));
    kv(out, "Petzvalkrümmung", with_unit(sci(p.petzval_curvature), p.petzval_curvature, "1/mm"));
    kv(out, "Petzvalradius", with_unit(fixed(p.petzval_radius), p.petzval_radius, "mm"));
    if (p.exit_pupil_at_infinity) {
        kv(out, "Austrittspupille", "im Unendlichen (bildseitig telezentrisch)");
    } else {
        kv(out, "Austrittspupille Lage (ab letzter Fläche)",
           with_unit(fixed(p.exit_pupil_position), p.exit_pupil_position, "mm"));
        kv(out, "Austrittspupille Radius",
           with_unit(fixed(p.exit_pupil_semi_diameter), p.exit_pupil_semi_diameter, "mm"));
        kv(out, "Pupillenvergrößerung", fixed(p.exit_pupil_magnification));
    }

    out << '\n';
    if (!r.wavefront) {
        out << "Wellenfront nicht auswertbar\n  " << r.wavefront_error << '\n';
        return;
    }
    const WavefrontReport& w = *r.wavefront;
    out << "Wellenfront (Quadratur " << r.options.rings << " Ringe x " << r.options.spokes
        << " Speichen, RMS in Wellen bei " << fixed(ref, 3) << " nm)\n";
    if (r.backend.gpu) {
        kv(out, "Backend",
           "GPU " + std::to_string(r.backend.device) + " (" + r.backend.device_name + "), " +
               (r.backend.df64 ? "df64" : "FP64"));
    } else {
        kv(out, "Backend", "CPU, FP64");
    }
    if (std::isfinite(w.focal_length)) {
        kv(out, "Sollbrennweite für die Bildpunkte", fixed(w.focal_length) + " mm");
    }
    out << "  " << pad_left("Feld", 4) << pad_left("Winkel [deg]", 14)
        << pad_left("Wellenlänge [nm]", 18) << pad_left("Strahlen", 10) << pad_left("versagt", 9)
        << pad_left("RMS [Wellen]", 14) << pad_left("RMS [mm]", 15) << '\n';
    for (const TupleReport& m : w.tuples) {
        const std::string angle =
            sys.source.type == SourceType::Collimated
                ? fixed(sys.source.field_angles_deg[static_cast<std::size_t>(m.field)], 3)
                : std::string("-");
        out << "  " << pad_left(std::to_string(m.field), 4) << pad_left(angle, 14)
            << pad_left(fixed(m.wavelength_nm, 3), 18) << pad_left(std::to_string(m.rays), 10)
            << pad_left(std::to_string(m.failed), 9) << pad_left(fixed(m.rms_waves), 14)
            << pad_left(sci(m.rms, 6), 15) << '\n';
    }
    out << "  Versagte Strahlen: " << w.failed << " von " << w.rays << '\n';
}

} // namespace

void print_trace_usage(std::ostream& out) {
    out << "Aufruf: mikroptim trace <system.json> [--json] [--rings N] [--spokes M]\n"
           "                       [--gpu [--device N] [--precision fp64|df64]]\n"
           "  Paraxialdaten, RMS-Wellenfront pro Feld und Wellenlänge und versagte Strahlen\n"
           "  (FP64; Beschreibung: docs/cli.md).\n"
           "  --json       maschinenlesbare Ausgabe (Format mikroptim.trace/1)\n"
           "  --rings N    radiale Ringe der Pupillenquadratur, 1 bis "
        << kMaxPupilRings << " (Standard " << kDefaultRings
        << ")\n"
           "  --spokes M   Speichen der Pupillenquadratur, >= 1 (Standard "
        << kDefaultSpokes
        << ")\n"
           "  --gpu        Wellenfront auf der GPU rechnen (Kernel RESIDUALS) statt auf der CPU\n"
           "  --device N   CUDA-Gerät für --gpu (Standard 0, Liste: mikroptim --devices)\n"
           "  --precision P  Rechentyp der Wellenfront mit --gpu: fp64 (Standard) oder df64\n"
           "               (Paar aus zwei float, auf GeForce-Karten schneller, ca. 1e-9 mm)\n"
           "  --help       diese Hilfe\n"
           "Exit-Code: 0 Erfolg, 1 Fehler beim Laden oder Auswerten, 2 ungültige "
           "Kommandozeile,\n"
           "           3 GPU-Backend nicht verfügbar (ohne CUDA gebaut oder kein CUDA-Gerät).\n";
}

int run_trace_command(std::span<const std::string_view> args, std::ostream& out,
                      std::ostream& err) {
    TraceArgs a;
    try {
        a = parse_args(args);
    } catch (const UsageError& e) {
        err << "mikroptim trace: " << e.what() << '\n';
        print_trace_usage(err);
        return kExitUsage;
    }
    if (a.help) {
        print_trace_usage(out);
        return kExitOk;
    }

    // GPU-Backend vor dem Laden prüfen, damit "nicht verfügbar" (kExitNoGpu) nicht von
    // Fehlern der Systemdatei abhängt.
#ifdef MIKROPTIM_HAS_CUDA
    std::optional<cuda::TraceEngine> engine;
    if (a.gpu) {
        const int count = cuda::device_count();
        if (count == 0) {
            err << "mikroptim trace: --gpu: kein CUDA-Gerät gefunden (GPU-Backend nicht "
                   "verfügbar)\n";
            return kExitNoGpu;
        }
        if (a.device >= count) {
            err << "mikroptim trace: Fehler: --device " << a.device << ": Gerät existiert nicht ("
                << count << (count == 1 ? " CUDA-Gerät" : " CUDA-Geräte")
                << ", siehe mikroptim --devices)\n";
            return kExitError;
        }
        try {
            engine.emplace(a.device);
        } catch (const std::exception& e) {
            err << "mikroptim trace: Fehler: GPU " << a.device << ": " << e.what() << '\n';
            return kExitError;
        }
    }
#else
    if (a.gpu) {
        err << "mikroptim trace: --gpu: ohne CUDA gebaut (MIKROPTIM_ENABLE_CUDA=OFF), "
               "GPU-Backend nicht verfügbar\n";
        return kExitNoGpu;
    }
#endif

    try {
        TraceReport r;
        r.file = a.file;
        r.loaded = load_system(std::filesystem::path(a.file));
        for (const std::string& w : r.loaded.warnings) {
            err << "mikroptim trace: Warnung: " << w << '\n';
        }
        r.paraxial = compute_paraxial(r.loaded.system, r.loaded.catalog);
        r.options.rings = a.rings;
        r.options.spokes = a.spokes;
        try {
#ifdef MIKROPTIM_HAS_CUDA
            if (engine) {
                r.backend.gpu = true;
                r.backend.device = engine->device();
                r.backend.device_name = engine->device_name();
                r.backend.df64 = a.df64;
                r.wavefront = evaluate_wavefront_gpu(r.loaded, r.options, *engine, a.df64);
            } else {
                r.wavefront = report_from(evaluate_wavefront(r.loaded, r.options));
            }
#else
            r.wavefront = report_from(evaluate_wavefront(r.loaded, r.options));
#endif
        } catch (const std::invalid_argument& e) {
            // Quadratur und Tabelle sind hier gültig; übrig bleibt die fehlende Sollbrennweite
            // eines afokalen Systems ohne target.focal_length (docs/cli.md). Der GPU-Pfad
            // meldet sie ebenso (target_focal_length vor dem Lauf).
            r.wavefront_error = e.what();
        }
        if (a.json) {
            write_json(r, out);
        } else {
            write_text(r, out);
        }
        out.flush();
        if (!out) {
            err << "mikroptim trace: Fehler beim Schreiben der Ausgabe\n";
            return kExitError;
        }
        return kExitOk;
    } catch (const std::exception& e) {
        err << "mikroptim trace: Fehler: " << e.what() << '\n';
        return kExitError;
    }
}

} // namespace mko::cli
