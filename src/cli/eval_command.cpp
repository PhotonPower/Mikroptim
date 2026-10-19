// Unterbefehl `mikroptim eval` (M8-T1): Laden, Batch mit einer Variante, Modus EVAL auf der CPU
// oder mit --gpu über die TraceEngine, Kennzahlen (eval_statistics), Ausgabe als Text oder JSON,
// optional OPD-Karte als CSV; mit --psf skalare PSF, MTF und Strehl (M8-T2, psf_cpu bzw.
// PsfEngine), mit --vector die vektorielle PSF nach Richards-Wolf (M8-T5, vector_psf_cpu bzw.
// VectorPsfEngine), optional PSF, Komponenten und MTF eines Tupels als CSV. Ausgabeformate:
// docs/cli.md.

#include "cli/eval_command.hpp"
#include "cli/cli_util.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/eval/eval.hpp"
#include "mikroptim/eval/psf.hpp"
#include "mikroptim/eval/vector_psf.hpp"
#include "mikroptim/version.hpp"

#ifdef MIKROPTIM_HAS_CUDA
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/psf.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/cuda/vector_psf.hpp"
#endif

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <fstream>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
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
    // PSF, MTF, Strehl (M8-T2)
    bool psf = false;
    PsfSettings psf_settings{};
    int psf_field = 0; // Tupel der PSF- und MTF-Dateien
    int psf_wavelength = 0;
    std::string psf_file; // leer: keine PSF-Datei
    std::string mtf_file; // leer: keine MTF-Datei
    // Vektorielle PSF (M8-T5)
    bool vector = false;
    VectorPsfSettings vector_settings{};
    std::string component_prefix; // leer: keine Komponentendateien
};

// Liest "F,L" (Feld und Wellenlänge, ab 0).
void parse_tuple(std::string_view option, std::string_view v, int& field, int& wavelength) {
    const std::size_t comma = v.find(',');
    if (comma == std::string_view::npos) {
        throw UsageError(std::string(option) + " erwartet F,L (Feld und Wellenlänge, ab 0)");
    }
    field = parse_int(option, v.substr(0, comma));
    wavelength = parse_int(option, v.substr(comma + 1));
}

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
            parse_tuple(arg, next_value(args, i, arg), a.map_field, a.map_wavelength);
            a.have_map = true;
        } else if (arg == "--psf") {
            a.psf = true;
        } else if (arg == "--padding") {
            a.psf_settings.padding = parse_double(arg, next_value(args, i, arg));
            if (!(a.psf_settings.padding >= 2.0)) {
                throw UsageError("--padding muss >= 2 sein");
            }
            a.psf = true;
        } else if (arg == "--na") {
            a.psf_settings.numerical_aperture = parse_double(arg, next_value(args, i, arg));
            if (!(a.psf_settings.numerical_aperture > 0.0)) {
                throw UsageError("--na muss > 0 sein");
            }
            a.psf = true;
        } else if (arg == "--psf-tuple") {
            parse_tuple(arg, next_value(args, i, arg), a.psf_field, a.psf_wavelength);
            a.psf = true;
        } else if (arg == "--psf-file") {
            a.psf_file = std::string(next_value(args, i, arg));
            a.psf = true;
        } else if (arg == "--mtf-file") {
            a.mtf_file = std::string(next_value(args, i, arg));
            a.psf = true;
        } else if (arg == "--vector") {
            a.vector = true;
            a.psf = true;
        } else if (arg == "--polarization") {
            const std::string_view v = next_value(args, i, arg);
            try {
                a.vector_settings.polarization = parse_polarization(v);
            } catch (const std::invalid_argument&) {
                throw UsageError(
                    "--polarization erwartet x, y, circular oder unpolarized, nicht '" +
                    std::string(v) + "'");
            }
            a.vector = true;
            a.psf = true;
        } else if (arg == "--defocus") {
            a.vector_settings.defocus = parse_double(arg, next_value(args, i, arg));
            if (!std::isfinite(a.vector_settings.defocus)) {
                throw UsageError("--defocus muss endlich sein");
            }
            a.vector = true;
            a.psf = true;
        } else if (arg == "--component-files") {
            a.component_prefix = std::string(next_value(args, i, arg));
            a.vector = true;
            a.psf = true;
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
// Tupels (grobe Angabe, gültig für kleine Fehler; Strehl über die FFT mit --psf, M8-T2).
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
    // PSF (nur mit --psf): Ergebnisse pro Tupel [f·L + l]; die PSF selbst bleibt nur für das
    // Tupel der Dateien erhalten (bis M² = 6,7·10⁷ Werte pro Tupel). Skalar (M8-T2) stehen nur
    // die Felder von PsfResult, vektoriell (--vector, M8-T5) auch die übrigen.
    bool psf = false;
    bool vector = false;
    PsfSettings psf_settings{};
    VectorPsfSettings vector_settings{};
    int psf_fft = 0;
    std::vector<VectorPsfResult> psfs;
};

// Rechnet die skalare PSF aller Tupel der Variante 0 nacheinander (CPU oder GPU) und behält die
// PSF nur für das Tupel (keep_f, keep_l).
template <class Compute>
void compute_psfs(EvalReport& r, int keep_f, int keep_l, Compute compute) {
    const std::vector<PsfPupil> pupils = make_psf_pupils(r.batch, r.result, 0, r.psf_settings);
    r.psf_fft = psf_fft_size(r.result.grid.size, r.psf_settings.padding);
    r.psfs.clear();
    for (const PsfPupil& p : pupils) {
        VectorPsfResult res;
        static_cast<PsfResult&>(res) = compute(p, r.psf_fft);
        if (p.field != keep_f || p.wavelength != keep_l) {
            std::vector<double>().swap(res.psf);
        }
        r.psfs.push_back(std::move(res));
    }
}

// Dasselbe für die vektorielle PSF (Komponenten ebenfalls nur für das Tupel der Dateien).
template <class Compute>
void compute_vector_psfs(EvalReport& r, int keep_f, int keep_l, Compute compute) {
    const std::vector<VectorPsfPupil> pupils =
        make_vector_psf_pupils(r.batch, r.result, 0, r.psf_settings);
    r.psf_fft = psf_fft_size(r.result.grid.size, r.psf_settings.padding);
    r.psfs.clear();
    for (const VectorPsfPupil& p : pupils) {
        VectorPsfResult res = compute(p, r.psf_fft);
        if (p.pupil.field != keep_f || p.pupil.wavelength != keep_l) {
            std::vector<double>().swap(res.psf);
            for (std::vector<double>& c : res.component) {
                std::vector<double>().swap(c);
            }
        }
        r.psfs.push_back(std::move(res));
    }
}

// Warnungen für Tupel, deren geometrischer Spot (Schwerpunkt ± 3 RMS-Radien) über das halbe
// Bildfenster M·Δx = λN/(2 NA) hinausreicht: Die periodische FFT faltet die PSF dann zurück
// (Abhilfe: größeres Raster N).
std::vector<std::string> psf_window_warnings(const EvalReport& r) {
    std::vector<std::string> out;
    for (std::size_t t = 0; t < r.psfs.size() && t < r.stats.size(); ++t) {
        const PsfResult& p = r.psfs[t];
        const EvalTupleStats& s = r.stats[t];
        const double half_x = 0.5 * p.fft_size * p.dx;
        const double half_y = 0.5 * p.fft_size * p.dy;
        const double reach_x = std::abs(s.centroid_x) + 3.0 * s.rms_spot;
        const double reach_y = std::abs(s.centroid_y) + 3.0 * s.rms_spot;
        if (reach_x > half_x || reach_y > half_y) {
            out.push_back("PSF Feld " + std::to_string(p.field) + ", Wellenlänge " +
                          std::to_string(p.wavelength) + ": geometrischer Spot (" +
                          format_number("%.3e", std::max(reach_x, reach_y)) +
                          " mm) größer als das halbe Bildfenster (" +
                          format_number("%.3e", std::min(half_x, half_y)) +
                          " mm), PSF zurückgefaltet; Raster vergrößern");
        }
    }
    return out;
}

// Kennzahlen eines Tupels als JSON (Längen in mm, Frequenzen in Linienpaaren/mm).
Json psf_json(const PsfResult& p) {
    Json e;
    e["na"] = Json::array({number(p.na_x), number(p.na_y)});
    e["pixel_mm"] = Json::array({number(p.dx), number(p.dy)});
    e["airy_radius_mm"] = number(p.airy_radius);
    e["strehl"] = number(p.strehl);
    e["strehl_reference_mm"] = Json::array({number(p.reference_x), number(p.reference_y)});
    e["strehl_origin"] = number(p.strehl_origin);
    e["strehl_fft"] = number(p.strehl_fft);
    e["peak"] = number(p.peak);
    e["peak_mm"] = Json::array({number(p.peak_x), number(p.peak_y)});
    e["energy"] = number(p.energy);
    e["cutoff_lpmm"] = Json::array({number(p.cutoff_x()), number(p.cutoff_y())});
    e["mtf_half_cutoff"] =
        Json::array({number(p.mtf_sagittal_at(0.5)), number(p.mtf_tangential_at(0.5))});
    return e;
}

// Zusätzliche Kennzahlen der vektoriellen PSF (M8-T5).
Json vector_psf_json(const VectorPsfResult& p) {
    Json e;
    e["fwhm_mm"] = Json::array({number(p.fwhm_x), number(p.fwhm_y)});
    e["energy_fraction"] = Json::array(
        {number(p.energy_fraction[0]), number(p.energy_fraction[1]), number(p.energy_fraction[2])});
    e["component_peak"] = Json::array(
        {number(p.component_peak[0]), number(p.component_peak[1]), number(p.component_peak[2])});
    e["reference_intensity"] = number(p.reference_intensity);
    e["evanescent_points"] = p.evanescent;
    return e;
}

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
    if (r.psf) {
        j["psf"] =
            Json{{"fft_size", r.psf_fft},
                 {"padding", r.psf_settings.padding},
                 {"numerical_aperture", r.psf_settings.numerical_aperture > 0.0
                                            ? Json(r.psf_settings.numerical_aperture)
                                            : Json(nullptr)},
                 {"model", r.vector ? "vector" : "scalar"},
                 {"polarization", r.vector ? Json(polarization_name(r.vector_settings.polarization))
                                           : Json(nullptr)},
                 {"defocus_mm", r.vector ? number(r.vector_settings.defocus) : Json(nullptr)}};
    } else {
        j["psf"] = nullptr;
    }
    Json tuples = Json::array();
    std::size_t index = 0;
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
        if (r.psf) {
            Json p = psf_json(r.psfs[index]);
            p["vector"] = r.vector ? vector_psf_json(r.psfs[index]) : Json(nullptr);
            e["psf"] = std::move(p);
        } else {
            e["psf"] = nullptr;
        }
        ++index;
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
    if (!r.psf) {
        return;
    }
    if (r.vector) {
        out << "\n  PSF vektoriell (Richards-Wolf, Polarisation "
            << polarization_name(r.vector_settings.polarization)
            << ", Ebene z = " << format_number("%.4e", r.vector_settings.defocus) << " mm; FFT "
            << r.psf_fft << " x " << r.psf_fft << ", Zero-Padding "
            << format_number("%.2f", static_cast<double>(r.psf_fft) / r.result.grid.size)
            << "), NA "
            << (r.psf_settings.numerical_aperture > 0.0 ? "vorgegeben"
                                                        : std::string("aus den Feldstrahlen"))
            << "; Strehl auf die perfekte vektorielle PSF normiert, Halbwertsbreiten durch das "
               "Maximum, Anteil von |E_z|² an der Energie der Ebene\n";
        out << "  " << pad_left("Feld", 4) << pad_left("λ [nm]", 9) << pad_left("NA", 9)
            << pad_left("Strehl", 9) << pad_left("im Bildp.", 11) << pad_left("Maximum", 9)
            << pad_left("FWHM x [mm]", 13) << pad_left("FWHM y [mm]", 13)
            << pad_left("Anteil z", 10) << pad_left("max |E_z|²", 12) << pad_left("MTF_s", 8)
            << pad_left("MTF_t", 8) << '\n';
        for (const VectorPsfResult& p : r.psfs) {
            out << "  " << pad_left(std::to_string(p.field), 4)
                << pad_left(format_number("%.2f", p.wavelength_nm), 9)
                << pad_left(format_number("%.5f", p.na_y), 9)
                << pad_left(format_number("%.5f", p.strehl), 9)
                << pad_left(format_number("%.5f", p.strehl_fft), 11)
                << pad_left(format_number("%.5f", p.peak), 9)
                << pad_left(format_number("%.4e", p.fwhm_x), 13)
                << pad_left(format_number("%.4e", p.fwhm_y), 13)
                << pad_left(format_number("%.5f", p.energy_fraction[2]), 10)
                << pad_left(format_number("%.5f", p.component_peak[2]), 12)
                << pad_left(format_number("%.4f", p.mtf_sagittal_at(0.5)), 8)
                << pad_left(format_number("%.4f", p.mtf_tangential_at(0.5)), 8) << '\n';
        }
        return;
    }
    out << "\n  PSF skalar (FFT " << r.psf_fft << " x " << r.psf_fft << ", Zero-Padding "
        << format_number("%.2f", static_cast<double>(r.psf_fft) / r.result.grid.size) << "), NA "
        << (r.psf_settings.numerical_aperture > 0.0 ? "vorgegeben"
                                                    : std::string("aus den Feldstrahlen"))
        << "; Strehl auf die ungestörte Pupille normiert, MTF bei f_c/2 mit f_c = 2 NA/λ\n";
    out << "  " << pad_left("Feld", 4) << pad_left("λ [nm]", 9) << pad_left("NA", 9)
        << pad_left("Airy [mm]", 12) << pad_left("Strehl", 9) << pad_left("im Bildp.", 11)
        << pad_left("Maximum", 9) << pad_left("Max. y [mm]", 13) << pad_left("MTF_s", 8)
        << pad_left("MTF_t", 8) << '\n';
    for (const PsfResult& p : r.psfs) {
        out << "  " << pad_left(std::to_string(p.field), 4)
            << pad_left(format_number("%.2f", p.wavelength_nm), 9)
            << pad_left(format_number("%.5f", p.na_y), 9)
            << pad_left(format_number("%.4e", p.airy_radius), 12)
            << pad_left(format_number("%.5f", p.strehl), 9)
            << pad_left(format_number("%.5f", p.strehl_fft), 11)
            << pad_left(format_number("%.5f", p.peak), 9)
            << pad_left(format_number("%.4e", p.peak_y), 13)
            << pad_left(format_number("%.4f", p.mtf_sagittal_at(0.5)), 8)
            << pad_left(format_number("%.4f", p.mtf_tangential_at(0.5)), 8) << '\n';
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

// M×M-Bild (zeilenweise abgelegt, values[j·M + i]) als CSV mit M Zeilen zu je M Werten, erste
// Zeile = größtes y.
void write_image_csv(const std::string& file, const char* what, int m,
                     const std::vector<double>& values) {
    std::ofstream f = open_output(file, what);
    const auto mu = static_cast<std::size_t>(m);
    for (int j = m - 1; j >= 0; --j) {
        for (int i = 0; i < m; ++i) {
            if (i > 0) {
                f << ',';
            }
            f << csv_number(values[static_cast<std::size_t>(j) * mu + static_cast<std::size_t>(i)]);
        }
        f << '\n';
    }
    if (!f) {
        throw std::runtime_error(std::string(what) + ": Fehler beim Schreiben von " + file);
    }
}

// PSF und MTF des gewählten Tupels als CSV: PSF mit M Zeilen zu je M Werten (normiert auf das
// Maximum der ungestörten PSF), erste Zeile = größtes y; MTF mit Kopfzeile und N + 1 Zeilen;
// vektoriell die Komponenten |E_x|², |E_y|², |E_z|² wie die PSF in PRÄFIX_ex.csv, _ey.csv,
// _ez.csv.
void write_psf_files(const EvalReport& r, const EvalArgs& a) {
    if (a.psf_file.empty() && a.mtf_file.empty() && a.component_prefix.empty()) {
        return;
    }
    const VectorPsfResult& p = r.psfs[static_cast<std::size_t>(a.psf_field) *
                                          static_cast<std::size_t>(r.result.wavelengths) +
                                      static_cast<std::size_t>(a.psf_wavelength)];
    if (!a.psf_file.empty()) {
        write_image_csv(a.psf_file, "PSF", p.fft_size, p.psf);
    }
    if (!a.component_prefix.empty() && r.vector) {
        const char* suffix[3] = {"_ex.csv", "_ey.csv", "_ez.csv"};
        for (std::size_t c = 0; c < 3; ++c) {
            write_image_csv(a.component_prefix + suffix[c], "PSF-Komponente", p.fft_size,
                            p.component[c]);
        }
    }
    if (!a.mtf_file.empty()) {
        std::ofstream f = open_output(a.mtf_file, "MTF");
        f << "k,nu,f_sagittal_lpmm,mtf_sagittal,f_tangential_lpmm,mtf_tangential\n";
        const auto n = static_cast<double>(p.grid);
        for (int k = 0; k <= p.grid; ++k) {
            const auto ku = static_cast<std::size_t>(k);
            const double nu = static_cast<double>(k) / n;
            f << k << ',' << csv_number(nu) << ',' << csv_number(nu * p.cutoff_x()) << ','
              << csv_number(p.mtf_sagittal[ku]) << ',' << csv_number(nu * p.cutoff_y()) << ','
              << csv_number(p.mtf_tangential[ku]) << '\n';
        }
        if (!f) {
            throw std::runtime_error("MTF: Fehler beim Schreiben von " + a.mtf_file);
        }
    }
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
           "                      [--psf [--padding Q] [--na X] [--psf-tuple F,L]\n"
           "                       [--psf-file DATEI.csv] [--mtf-file DATEI.csv]\n"
           "                       [--vector [--polarization P] [--defocus Z]\n"
           "                        [--component-files PRÄFIX]]]\n"
           "                      [--gpu [--device N]]\n"
           "  Auswertung auf dem dichten kartesischen Pupillenraster (Modus EVAL, FP64):\n"
           "  RMS-OPD, Strehl nach Maréchal, RMS-Spot, Schwerpunkt, Verzeichnung und Brennlinien\n"
           "  pro Feld und Wellenlänge, mit --psf zusätzlich skalare PSF, MTF und Strehl, mit\n"
           "  --vector die vektorielle PSF für hohe NA (Beschreibung: docs/cli.md).\n"
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
           "  --psf           skalare PSF, MTF und Strehl über die FFT (für alle Tupel)\n"
           "  --padding Q     Zero-Padding der FFT, M = Zweierpotenz >= Q N (Standard 4, >= 2)\n"
           "  --na X          NA der Pupillenabbildung vorgeben (Standard: aus den Feldstrahlen)\n"
           "  --psf-tuple F,L Tupel für --psf-file und --mtf-file (Standard 0,0)\n"
           "  --psf-file D    PSF als CSV nach D (M x M, normiert auf die ungestörte PSF)\n"
           "  --mtf-file D    MTF-Schnitte (sagittal, tangential) als CSV nach D\n"
           "  --vector        vektorielle PSF nach Richards-Wolf statt der skalaren (setzt --psf)\n"
           "  --polarization P  Eingangsfeld x (Standard), y, circular oder unpolarized\n"
           "  --defocus Z     Auswerteebene Z mm hinter der Bildebene (in +z, Standard 0)\n"
           "  --component-files P  |E_x|², |E_y|², |E_z|² des Tupels als CSV nach P_ex.csv,\n"
           "                  P_ey.csv, P_ez.csv\n"
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
    std::optional<cuda::PsfEngine> psf_engine;
    std::optional<cuda::VectorPsfEngine> vector_engine;
    if (a.gpu) {
        const int count = cuda::device_count();
        if (count == 0) {
            err << "mikroptim eval: --gpu: kein CUDA-Gerät gefunden (GPU-Backend nicht "
                   "verfügbar)\n";
            return kExitNoGpu;
        }
        try {
            engine.emplace(a.device);
            if (a.psf && a.vector) {
                vector_engine.emplace(a.device);
            } else if (a.psf) {
                psf_engine.emplace(a.device);
            }
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
        if (a.psf) {
            if (a.psf_field < 0 || a.psf_field >= r.result.fields || a.psf_wavelength < 0 ||
                a.psf_wavelength >= r.result.wavelengths) {
                throw std::invalid_argument("--psf-tuple " + std::to_string(a.psf_field) + "," +
                                            std::to_string(a.psf_wavelength) +
                                            ": Feld oder Wellenlänge existiert nicht");
            }
            r.psf = true;
            r.vector = a.vector;
            r.psf_settings = a.psf_settings;
            r.vector_settings = a.vector_settings;
            const auto vector_cpu = [&a](const VectorPsfPupil& p, int m) {
                return vector_psf_cpu(p, m, a.vector_settings);
            };
#ifdef MIKROPTIM_HAS_CUDA
            if (vector_engine) {
                compute_vector_psfs(
                    r, a.psf_field, a.psf_wavelength, [&](const VectorPsfPupil& p, int m) {
                        return std::move(vector_engine->compute(
                            std::span<const VectorPsfPupil>(&p, 1), m, a.vector_settings)[0]);
                    });
            } else if (a.vector) {
                compute_vector_psfs(r, a.psf_field, a.psf_wavelength, vector_cpu);
            } else if (psf_engine) {
                compute_psfs(r, a.psf_field, a.psf_wavelength, [&](const PsfPupil& p, int m) {
                    return std::move(psf_engine->compute(std::span<const PsfPupil>(&p, 1), m)[0]);
                });
            } else {
                compute_psfs(r, a.psf_field, a.psf_wavelength,
                             [](const PsfPupil& p, int m) { return psf_cpu(p, m); });
            }
#else
            if (a.vector) {
                compute_vector_psfs(r, a.psf_field, a.psf_wavelength, vector_cpu);
            } else {
                compute_psfs(r, a.psf_field, a.psf_wavelength,
                             [](const PsfPupil& p, int m) { return psf_cpu(p, m); });
            }
#endif
            for (std::string& w : psf_window_warnings(r)) {
                err << "mikroptim eval: Warnung: " << w << '\n';
                r.loaded.warnings.push_back(std::move(w));
            }
            write_psf_files(r, a);
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
