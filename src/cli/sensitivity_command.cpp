// Unterbefehl `mikroptim sensitivity` (M10-T3): Laden, Toleranzsatz (Datei oder Standard),
// Empfindlichkeiten (run_sensitivity_analysis) auf der CPU oder mit --gpu, Herstellbarkeit
// (check_manufacturability), Ausgabe als Text oder JSON. Ausgabeformate: docs/cli.md.

#include "cli/sensitivity_command.hpp"
#include "cli/cli_util.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/model/tolerances.hpp"
#include "mikroptim/tolerance/manufacturability.hpp"
#include "mikroptim/tolerance/sensitivity.hpp"
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
#include <exception>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mko::cli {

namespace {

using Json = nlohmann::ordered_json;

enum class SetChoice {
    Auto,    // Datei, falls vorhanden, sonst Standard
    File,    // Toleranzsatz der Datei (Pflicht)
    Default, // Mikroptim-Standardsatz
};

struct SensitivityArgs {
    std::string file;
    bool json = false;
    bool help = false;
    bool no_reference_warning = false; // --no-reference-warning (ADR-0017)
    bool gpu = false;
    bool have_device = false;
    int device = 0;
    SetChoice set = SetChoice::Auto;
    bool defaults_given = false; // eine Option des Standardsatzes angegeben
    SensitivityDefaults defaults{};
    SensitivitySettings settings{};
    ManufacturabilityLimits limits{};
    bool manufacturability_only = false;
    int top = 10;
};

std::string_view next_value(std::span<const std::string_view> args, std::size_t& i,
                            std::string_view option) {
    if (i + 1 >= args.size()) {
        throw UsageError(std::string(option) + " erwartet einen Wert");
    }
    return args[++i];
}

double non_negative(std::string_view option, std::string_view text) {
    const double v = parse_double(option, text);
    if (!(v >= 0.0)) {
        throw UsageError(std::string(option) + " muss >= 0 sein");
    }
    return v;
}

SensitivityArgs parse_args(std::span<const std::string_view> args) {
    SensitivityArgs a;
    bool have_file = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--help" || arg == "-h") {
            a.help = true;
        } else if (arg == "--json") {
            a.json = true;
        } else if (arg == kNoReferenceWarningOption) {
            a.no_reference_warning = true;
        } else if (arg == "--gpu") {
            a.gpu = true;
        } else if (arg == "--device") {
            a.device = parse_int(arg, next_value(args, i, arg));
            if (a.device < 0) {
                throw UsageError("--device muss >= 0 sein");
            }
            a.have_device = true;
        } else if (arg == "--set") {
            const std::string_view v = next_value(args, i, arg);
            if (v == "file") {
                a.set = SetChoice::File;
            } else if (v == "default") {
                a.set = SetChoice::Default;
            } else {
                throw UsageError("--set erwartet file oder default, nicht '" + std::string(v) +
                                 "'");
            }
        } else if (arg == "--decenter") {
            a.defaults.element_decenter_mm = non_negative(arg, next_value(args, i, arg));
            a.defaults_given = true;
        } else if (arg == "--tilt-arcmin") {
            a.defaults.element_tilt_arcmin = non_negative(arg, next_value(args, i, arg));
            if (!(a.defaults.element_tilt_arcmin < 90.0 * 60.0)) {
                throw UsageError("--tilt-arcmin muss kleiner als 5400 (90°) sein");
            }
            a.defaults_given = true;
        } else if (arg == "--pivot") {
            const std::string_view v = next_value(args, i, arg);
            const std::optional<ElementPivot> p = element_pivot_from_string(v);
            if (!p) {
                throw UsageError("--pivot erwartet first, center oder last, nicht '" +
                                 std::string(v) + "'");
            }
            a.defaults.element_pivot = *p;
            a.defaults_given = true;
        } else if (arg == "--radius") {
            a.defaults.radius_relative = non_negative(arg, next_value(args, i, arg));
            if (!(a.defaults.radius_relative < 1.0)) {
                throw UsageError("--radius muss kleiner als 1 sein");
            }
            a.defaults_given = true;
        } else if (arg == "--thickness") {
            a.defaults.thickness_mm = non_negative(arg, next_value(args, i, arg));
            a.defaults_given = true;
        } else if (arg == "--index") {
            a.defaults.index = non_negative(arg, next_value(args, i, arg));
            a.defaults_given = true;
        } else if (arg == "--step") {
            a.settings.step = parse_double(arg, next_value(args, i, arg));
            if (!(a.settings.step > 0.0 && a.settings.step <= 1.0)) {
                throw UsageError("--step muss in (0, 1] liegen");
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
        } else if (arg == "--top") {
            a.top = parse_int(arg, next_value(args, i, arg));
            if (a.top < 0) {
                throw UsageError("--top muss >= 0 sein");
            }
        } else if (arg == "--manufacturability-only") {
            a.manufacturability_only = true;
        } else if (arg == "--max-hemisphere") {
            a.limits.max_hemisphere = non_negative(arg, next_value(args, i, arg));
        } else if (arg == "--min-center") {
            a.limits.min_center_thickness = non_negative(arg, next_value(args, i, arg));
        } else if (arg == "--min-edge") {
            a.limits.min_edge_thickness = non_negative(arg, next_value(args, i, arg));
        } else if (arg == "--min-edge-relative") {
            a.limits.min_edge_thickness_relative = non_negative(arg, next_value(args, i, arg));
        } else if (arg == "--min-air-edge") {
            a.limits.min_air_edge = non_negative(arg, next_value(args, i, arg));
        } else if (arg == "--max-aspect") {
            a.limits.max_aspect_ratio = non_negative(arg, next_value(args, i, arg));
        } else if (arg == "--min-centering") {
            a.limits.min_centering = non_negative(arg, next_value(args, i, arg));
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
    if (a.defaults_given && a.set == SetChoice::File) {
        throw UsageError("--decenter, --tilt-arcmin, --pivot, --radius, --thickness und --index "
                         "gelten nur für den Standardsatz (--set default)");
    }
    if (a.manufacturability_only && a.gpu) {
        throw UsageError("--manufacturability-only rechnet keinen Trace (ohne --gpu)");
    }
    return a;
}

// Nicht endliche Zahlen sind in JSON nicht darstellbar und werden null.
Json number(double v) {
    return std::isfinite(v) ? Json(v) : Json(nullptr);
}

constexpr double kRadToArcmin = 180.0 * 60.0 / std::numbers::pi;

// Grenze in der Einheit der Datei und Einheit.
std::pair<double, const char*> limit_display(const ResolvedTolerance& t) {
    switch (t.type) {
    case ToleranceType::Radius:
        return t.radius_fringes ? std::pair{t.limit, "1/mm"} : std::pair{t.limit, "relativ"};
    case ToleranceType::Thickness:
    case ToleranceType::Decenter:
    case ToleranceType::ElementDecenter:
        return {t.limit, "mm"};
    case ToleranceType::Tilt:
    case ToleranceType::ElementTilt:
        return {t.limit * kRadToArcmin, "arcmin"};
    case ToleranceType::Abbe:
        return {t.limit, "relativ"};
    case ToleranceType::Index:
        return {t.limit, ""};
    }
    return {t.limit, ""};
}

const char* component_name(const ResolvedTolerance& t, int c) {
    switch (t.type) {
    case ToleranceType::Decenter:
    case ToleranceType::ElementDecenter:
        return c == 0 ? "x" : "y";
    case ToleranceType::Tilt:
    case ToleranceType::ElementTilt:
        return c == 0 ? "alpha" : "beta";
    default:
        return nullptr;
    }
}

std::string where(const ResolvedTolerance& t) {
    if (t.surface >= 0) {
        return "Fläche " + std::to_string(t.surface);
    }
    return "Element " + std::to_string(t.first) + "–" + std::to_string(t.last);
}

const char* type_text(ToleranceType type) {
    switch (type) {
    case ToleranceType::Radius:
        return "Radius";
    case ToleranceType::Thickness:
        return "Dicke";
    case ToleranceType::Index:
        return "Brechzahl";
    case ToleranceType::Abbe:
        return "Abbe-Zahl";
    case ToleranceType::Decenter:
        return "Dezentrierung";
    case ToleranceType::Tilt:
        return "Verkippung";
    case ToleranceType::ElementDecenter:
        return "Element-Dezentr.";
    case ToleranceType::ElementTilt:
        return "Element-Verkipp.";
    }
    return "?";
}

const char* check_text(ManufacturabilityCheckType type) {
    switch (type) {
    case ManufacturabilityCheckType::Hemisphere:
        return "Halbkugelgrad h/|R|";
    case ManufacturabilityCheckType::CenterThickness:
        return "Mittendicke [mm]";
    case ManufacturabilityCheckType::EdgeThickness:
        return "Randdicke [mm]";
    case ManufacturabilityCheckType::AirEdge:
        return "Luftrand [mm]";
    case ManufacturabilityCheckType::AspectRatio:
        return "Durchmesser/Mittendicke";
    case ManufacturabilityCheckType::Centering:
        return "Zentrierbarkeit Z";
    }
    return "?";
}

struct SensitivityReport {
    std::string file;
    LoadedSystem loaded;
    bool default_set = false;
    std::optional<SensitivityResult> result;
    std::vector<SensitivitySummary> summary;
    std::vector<int> ranking; // Toleranzen nach worst_delta_q absteigend (NaN zuerst)
    ManufacturabilityReport manufacturability;
    bool gpu = false;
    int device = -1;
    std::string device_name;
    double seconds = 0.0;
    int rings = 0;
    int spokes = 0;
};

std::vector<int> rank(const std::vector<SensitivitySummary>& summary) {
    std::vector<int> order(summary.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = static_cast<int>(i);
    }
    const auto key = [&](int i) {
        const double v = summary[static_cast<std::size_t>(i)].worst_delta_q;
        return std::isnan(v) ? std::numeric_limits<double>::infinity() : v;
    };
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return key(a) > key(b); });
    return order;
}

Json manufacturability_json(const ManufacturabilityReport& m) {
    Json j;
    j["limits"] = Json{{"max_hemisphere", m.limits.max_hemisphere},
                       {"min_center_thickness_mm", m.limits.min_center_thickness},
                       {"min_edge_thickness_mm", m.limits.min_edge_thickness},
                       {"min_edge_thickness_relative", m.limits.min_edge_thickness_relative},
                       {"min_air_edge_mm", m.limits.min_air_edge},
                       {"max_aspect_ratio", m.limits.max_aspect_ratio},
                       {"min_centering", m.limits.min_centering}};
    Json checks = Json::array();
    for (const ManufacturabilityCheck& c : m.checks) {
        checks.push_back(Json{{"check", std::string(to_string(c.type))},
                              {"surface", c.surface},
                              {"value", number(c.value)},
                              {"limit", c.limit},
                              {"severity", std::string(to_string(c.severity))},
                              {"message", finding_message(c)}});
    }
    j["checks"] = std::move(checks);
    j["warnings"] = m.warnings;
    j["errors"] = m.errors;
    return j;
}

void write_json(const SensitivityReport& r, std::ostream& out) {
    const SystemDescription& sys = r.loaded.system;
    Json j;
    j["format"] = std::string(kSensitivityFormat);
    j["mikroptim_version"] = kVersion;
    j["system"] = sys.name;
    j["file"] = r.file;
    if (r.result) {
        const SensitivityResult& res = *r.result;
        j["backend"] = r.gpu ? "gpu" : "cpu";
        if (r.gpu) {
            j["device"] = Json{{"index", r.device}, {"name", r.device_name}};
        } else {
            j["device"] = nullptr;
        }
        j["settings"] = Json{{"set", r.default_set ? "default" : "file"},
                             {"rings", r.rings},
                             {"spokes", r.spokes},
                             {"step", res.step}};
        j["reference_nm"] = res.reference_nm;
        j["variants"] = res.variants;
        j["nominal"] = Json{{"q_waves2", number(res.q0)},
                            {"rms_waves", number(std::sqrt(res.q0))},
                            {"failed_rays", res.failed_rays}};
        Json tol = Json::array();
        for (std::size_t k = 0; k < res.tolerances.size(); ++k) {
            const ResolvedTolerance& t = res.tolerances[k];
            const SensitivitySummary& s = r.summary[k];
            const auto [value, unit] = limit_display(t);
            Json e;
            e["index"] = static_cast<int>(k);
            e["type"] = std::string(to_string(t.type));
            e["item"] = t.item;
            e["surface"] = t.surface >= 0 ? Json(t.surface) : Json(nullptr);
            e["element"] = t.first >= 0 ? Json::array({t.first, t.last}) : Json(nullptr);
            e["limit"] = value;
            e["unit"] = unit;
            e["pivot_z_mm"] =
                t.type == ToleranceType::ElementTilt ? Json(t.pivot_z) : Json(nullptr);
            e["worst_delta_q_waves2"] = number(s.worst_delta_q);
            e["worst_rms_waves"] = number(s.worst_rms);
            e["mean_quadratic_waves2"] = number(s.mean_quadratic);
            e["mean_jacobian_quadratic_waves2"] = number(s.mean_jacobian_quadratic);
            e["failed_rays"] = s.failed_rays;
            Json comps = Json::array();
            for (const SensitivityComponent& c : res.components) {
                if (c.tolerance != static_cast<int>(k)) {
                    continue;
                }
                const char* name = component_name(t, c.component);
                comps.push_back(Json{{"component", name ? Json(name) : Json(nullptr)},
                                     {"q_plus", number(c.q_plus)},
                                     {"q_minus", number(c.q_minus)},
                                     {"linear", number(c.linear)},
                                     {"quadratic", number(c.quadratic)},
                                     {"jacobian_linear", number(c.jacobian_linear)},
                                     {"jacobian_quadratic", number(c.jacobian_quadratic)},
                                     {"failed_rays", c.failed_rays}});
            }
            e["components"] = std::move(comps);
            tol.push_back(std::move(e));
        }
        j["tolerances"] = std::move(tol);
        j["ranking"] = r.ranking;
        j["elapsed_seconds"] = r.seconds;
    } else {
        j["tolerances"] = nullptr;
    }
    j["manufacturability"] = manufacturability_json(r.manufacturability);
    Json warnings = Json::array();
    for (const std::string& w : r.loaded.warnings) {
        warnings.push_back(w);
    }
    j["warnings"] = std::move(warnings);
    out << j.dump(2, ' ', false, Json::error_handler_t::replace) << '\n';
}

std::string e3(double v) {
    return format_number("%.3e", v);
}

void write_text(const SensitivityReport& r, int top, std::ostream& out) {
    out << "System: " << r.loaded.system.name << '\n';
    out << "  Datei: " << r.file << '\n';
    if (r.result) {
        const SensitivityResult& res = *r.result;
        out << "  Toleranzsatz: "
            << (r.default_set ? "Mikroptim-Standard (--set default)" : "Datei (tolerances)") << ", "
            << res.tolerances.size() << " Toleranzen, " << res.components.size() << " Komponenten, "
            << res.variants << " Varianten\n";
        out << "  Backend: "
            << (r.gpu ? "GPU " + std::to_string(r.device) + " (" + r.device_name + "), FP64"
                      : std::string("CPU, FP64"))
            << "; Quadratur " << r.rings << " Ringe x " << r.spokes
            << " Speichen, volle Pupille, ohne Kompensator\n";
        out << "  Q = Σ ŵ RMS² nach Abzug von Kolben und Kippung [Wellen² bei "
            << format_number("%.3f", res.reference_nm)
            << " nm]; Jacobi-Schritt h = " << format_number("%.3g", res.step) << " δ\n";
        out << "  Nominal: Q0 = " << e3(res.q0) << ", RMS "
            << format_number("%.5f", std::sqrt(res.q0)) << " Wellen, versagte Strahlen "
            << res.failed_rays << '\n';
        out << "  Laufzeit: " << format_number("%.2f", r.seconds) << " s\n\n";
        const std::size_t rows = std::min(r.ranking.size(), static_cast<std::size_t>(top));
        out << "  Empfindlichste Toleranzen (" << rows << " von " << r.ranking.size()
            << ", nach größtem ΔQ bei ±δ)\n";
        out << "  " << pad_left("#", 3) << "  " << pad_right("Art", 17) << pad_right("Ort", 14)
            << pad_left("δ", 12) << "  " << pad_right("Einheit", 8) << pad_left("ΔQ max", 11)
            << pad_left("RMS bei δ", 11) << pad_left("b", 11) << pad_left("‖j‖²", 11)
            << pad_left("Versagt", 9) << '\n';
        for (std::size_t i = 0; i < rows; ++i) {
            const int k = r.ranking[i];
            const ResolvedTolerance& t = res.tolerances[static_cast<std::size_t>(k)];
            const SensitivitySummary& s = r.summary[static_cast<std::size_t>(k)];
            const auto [value, unit] = limit_display(t);
            out << "  " << pad_left(std::to_string(i + 1), 3) << "  "
                << pad_right(type_text(t.type), 17) << pad_right(where(t), 14)
                << pad_left(format_number("%.4g", value), 12) << "  " << pad_right(unit, 8)
                << pad_left(e3(s.worst_delta_q), 11)
                << pad_left(format_number("%.5f", s.worst_rms), 11)
                << pad_left(e3(s.mean_quadratic), 11) << pad_left(e3(s.mean_jacobian_quadratic), 11)
                << pad_left(std::to_string(s.failed_rays), 9) << '\n';
        }
        out << "  (ΔQ max: größte Zunahme von Q über Komponenten und ±δ [Wellen²]; b = (Q+ + Q- - "
               "2 Q0)/2 direkt,\n   ‖j‖² Gauß-Newton-Term der Jacobi-Spalte, beide gemittelt über "
               "x/y bzw. α/β)\n\n";
    }
    const ManufacturabilityReport& m = r.manufacturability;
    out << "  Herstellbarkeit (nur Hinweise, keine Nebenbedingungen): " << m.checks.size()
        << " Prüfungen, " << m.warnings << " Warnungen, " << m.errors << " Fehler\n";
    out << "  Grenzen: h/|R| <= " << format_number("%.3g", m.limits.max_hemisphere)
        << ", Mittendicke >= " << format_number("%.3g", m.limits.min_center_thickness)
        << " mm, Randdicke >= max(" << format_number("%.3g", m.limits.min_edge_thickness) << " mm; "
        << format_number("%.3g", m.limits.min_edge_thickness_relative)
        << " D), Luftrand >= " << format_number("%.3g", m.limits.min_air_edge)
        << " mm, 2h/t <= " << format_number("%.3g", m.limits.max_aspect_ratio)
        << ", Z >= " << format_number("%.3g", m.limits.min_centering) << '\n';
    for (const ManufacturabilityCheck& c : m.checks) {
        if (c.severity == CheckSeverity::Ok) {
            continue;
        }
        out << "  " << pad_right(c.severity == CheckSeverity::Error ? "FEHLER" : "Warnung", 9)
            << pad_right("Fläche " + std::to_string(c.surface), 11)
            << pad_right(check_text(c.type), 25) << pad_right(format_number("%.4g", c.value), 11)
            << finding_message(c) << '\n';
    }
}

} // namespace

void print_sensitivity_usage(std::ostream& out) {
    out << "Aufruf: mikroptim sensitivity <system.json> [--set file|default] [--json] [--top N]\n"
           "                             [--decenter mm] [--tilt-arcmin a] [--pivot P]\n"
           "                             [--radius rel] [--thickness mm] [--index dn]\n"
           "                             [--step h] [--rings N] [--spokes M]\n"
           "                             [--max-hemisphere v] [--min-center mm] [--min-edge mm]\n"
           "                             [--min-edge-relative v]\n"
           "                             [--min-air-edge mm] [--max-aspect v] [--min-centering "
           "Z]\n"
           "                             [--manufacturability-only] [--gpu [--device N]]\n"
           "  Empfindlichkeit der Bildgüte für jede einzelne Toleranz (direkt bei ±δ und aus der\n"
           "  Jacobi-Matrix der Wellenfrontreste) und einfache Herstellbarkeitsprüfungen\n"
           "  (Beschreibung: docs/cli.md).\n"
           "  --set S           file: Toleranzsatz der Datei; default: Mikroptim-Standardsatz\n"
           "                    (Standard: Datei, falls vorhanden, sonst default)\n"
           "  --decenter mm     Standardsatz: Dezentrierung pro Element (Standard 0.01)\n"
           "  --tilt-arcmin a   Standardsatz: Verkippung pro Element (Standard 1)\n"
           "  --pivot P         Standardsatz: Drehpunkt first, center oder last (Standard center)\n"
           "  --radius rel      Standardsatz: Radius relativ (Standard 0.001)\n"
           "  --thickness mm    Standardsatz: Dicken und Luftabstände (Standard 0.02)\n"
           "  --index dn        Standardsatz: Brechzahl (Standard 0.0005); 0 schaltet eine Art ab\n"
           "  --step h          Schritt der Jacobi-Matrix relativ zu δ (Standard 0.01)\n"
           "  --rings N         radiale Ringe der Pupillenquadratur (Standard 10)\n"
           "  --spokes M        Speichen der Pupillenquadratur (Standard 8)\n"
           "  --top N           Zeilen der Rangliste im Text (Standard 10)\n"
           "  --max-hemisphere v  Warnung bei h/|R| > v (Standard 0.95)\n"
           "  --min-center mm   Warnung bei Mittendicke des Glases < mm (Standard 0.5)\n"
           "  --min-edge mm     Warnung bei Randdicke des Glases < max(mm, v D) (Standard 0.5)\n"
           "  --min-edge-relative v  Anteil v des Elementdurchmessers D (Standard 0.05)\n"
           "  --min-air-edge mm Warnung bei Luftrand < mm (Standard 0; < 0 ist immer ein Fehler)\n"
           "  --max-aspect v    Warnung bei Durchmesser/Mittendicke > v (Standard 10)\n"
           "  --min-centering Z Warnung bei Zentrierkoeffizient < Z: Glockenzentrierung\n"
           "                    ungeeignet (Standard 0.1)\n"
           "  --manufacturability-only  nur die Herstellbarkeit, ohne Trace\n"
           "  --json            maschinenlesbare Ausgabe (Format mikroptim.sensitivity/1)\n"
           "  --no-reference-warning  keine Warnung auf stderr, wenn die Datei kein Feld\n"
           "                    reference_sphere hat (dann gilt last_surface)\n"
           "  --gpu             Trace auf der GPU (RESIDUALS) statt auf der CPU\n"
           "  --device N        CUDA-Gerät für --gpu (Standard 0)\n"
           "  --help            diese Hilfe\n"
           "Exit-Code: 0 Erfolg (auch mit Befunden der Herstellbarkeit), 1 Fehler beim Laden oder\n"
           "           Rechnen, 2 ungültige Kommandozeile, 3 GPU-Backend nicht verfügbar.\n";
}

int run_sensitivity_command(std::span<const std::string_view> args, std::ostream& out,
                            std::ostream& err) {
    SensitivityArgs a;
    try {
        a = parse_args(args);
    } catch (const UsageError& e) {
        err << "mikroptim sensitivity: " << e.what() << '\n';
        print_sensitivity_usage(err);
        return kExitUsage;
    }
    if (a.help) {
        print_sensitivity_usage(out);
        return kExitOk;
    }

#ifdef MIKROPTIM_HAS_CUDA
    std::optional<cuda::TraceEngine> engine;
    if (a.gpu) {
        const int count = cuda::device_count();
        if (count == 0) {
            err << "mikroptim sensitivity: --gpu: kein CUDA-Gerät gefunden (GPU-Backend nicht "
                   "verfügbar)\n";
            return kExitNoGpu;
        }
        if (a.device >= count) {
            err << "mikroptim sensitivity: Fehler: --device " << a.device
                << ": Gerät existiert nicht (siehe mikroptim --devices)\n";
            return kExitError;
        }
        try {
            engine.emplace(a.device);
        } catch (const std::exception& e) {
            err << "mikroptim sensitivity: Fehler: GPU " << a.device << ": " << e.what() << '\n';
            return kExitError;
        }
    }
#else
    if (a.gpu) {
        err << "mikroptim sensitivity: --gpu: ohne CUDA gebaut (MIKROPTIM_ENABLE_CUDA=OFF), "
               "GPU-Backend nicht verfügbar\n";
        return kExitNoGpu;
    }
#endif

    try {
        SensitivityReport r;
        r.file = a.file;
        r.loaded = load_system(std::filesystem::path(a.file));
        for (const std::string& w : r.loaded.warnings) {
            err << "mikroptim sensitivity: Warnung: " << w << '\n';
        }
        // ohne Trace (nur Herstellbarkeit) hängt nichts an der Referenzsphäre
        warn_missing_reference_sphere(err, "sensitivity", r.loaded.system,
                                      a.no_reference_warning || a.manufacturability_only);
        r.manufacturability = check_manufacturability(r.loaded.system, a.limits);
        if (!a.manufacturability_only) {
            SystemDescription system = r.loaded.system;
            if (a.set == SetChoice::File && !system.tolerances) {
                throw std::invalid_argument("keine Toleranzen: die Systemdatei hat keinen "
                                            "Abschnitt \"tolerances\" (--set default verwenden)");
            }
            if (a.set == SetChoice::Default || (a.set == SetChoice::Auto && !system.tolerances)) {
                system.tolerances = default_sensitivity_tolerances(system, a.defaults);
                r.default_set = true;
            } else if (a.defaults_given) {
                throw std::invalid_argument("--decenter, --tilt-arcmin, --pivot, --radius, "
                                            "--thickness und --index gelten nur für den "
                                            "Standardsatz; die Datei hat einen Toleranzsatz "
                                            "(--set default verwenden)");
            }
            r.rings = a.settings.quadrature.rings;
            r.spokes = a.settings.quadrature.spokes;
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
            r.result = run_sensitivity_analysis(system, batch, a.settings);
            r.seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            r.summary = summarize_sensitivity(*r.result);
            r.ranking = rank(r.summary);
        }
        if (a.json) {
            write_json(r, out);
        } else {
            write_text(r, a.top, out);
        }
        out.flush();
        if (!out) {
            err << "mikroptim sensitivity: Fehler beim Schreiben der Ausgabe\n";
            return kExitError;
        }
        return kExitOk;
    } catch (const std::exception& e) {
        err << "mikroptim sensitivity: Fehler: " << e.what() << '\n';
        return kExitError;
    }
}

} // namespace mko::cli
