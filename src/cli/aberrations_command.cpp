#include "cli/aberrations_command.hpp"

#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/paraxial/aberrations.hpp"
#include "mikroptim/version.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <exception>
#include <optional>
#include <string>

namespace mko::cli {
namespace {

using Json = nlohmann::ordered_json;

struct AberrationArgs {
    std::string file;
    bool help = false;
    bool json = false;
    AberrationOptions options;
};

std::string_view next_value(std::span<const std::string_view> args, std::size_t& i,
                            std::string_view option) {
    if (i + 1 >= args.size()) {
        throw UsageError(std::string(option) + " erwartet einen Wert");
    }
    return args[++i];
}

AberrationArgs parse_args(std::span<const std::string_view> args) {
    AberrationArgs a;
    bool have_file = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--help" || arg == "-h") {
            a.help = true;
        } else if (arg == "--json") {
            a.json = true;
        } else if (arg == "--field-deg") {
            a.options.field_angle_deg = parse_double(arg, next_value(args, i, arg));
        } else if (arg == "--reference-nm") {
            a.options.reference_nm = parse_double(arg, next_value(args, i, arg));
        } else if (arg == "--short-nm") {
            a.options.short_nm = parse_double(arg, next_value(args, i, arg));
        } else if (arg == "--long-nm") {
            a.options.long_nm = parse_double(arg, next_value(args, i, arg));
        } else if (!arg.empty() && arg.front() == '-') {
            throw UsageError("unbekannte Option " + std::string(arg));
        } else if (!have_file) {
            a.file = std::string(arg);
            have_file = true;
        } else {
            throw UsageError("mehr als eine Systemdatei angegeben");
        }
    }
    if (!a.help && !have_file) {
        throw UsageError("Systemdatei fehlt");
    }
    return a;
}

Json number_or_null(double v) {
    return std::isfinite(v) ? Json(v) : Json(nullptr);
}

Json waves_json(const SeidelWaves& w) {
    return Json{{"W040", w.W040},
                {"W131", w.W131},
                {"W222", w.W222},
                {"W220", w.W220},
                {"W311", number_or_null(w.W311)},
                {"dW020", w.dW020},
                {"dW111", w.dW111},
                {"dW040", w.dW040}};
}

Json report_json(const AberrationReport& rep, const std::string& file) {
    Json j;
    j["format"] = "mikroptim.aberrations/1";
    j["version"] = std::string(kVersion);
    j["system"] = file;
    j["reference_nm"] = rep.reference_nm;
    j["short_nm"] = rep.short_nm;
    j["long_nm"] = rep.long_nm;
    j["field_angle_deg"] = rep.field_angle_deg;
    j["stop_semi_diameter"] = rep.stop_semi_diameter;
    j["lagrange"] = rep.lagrange;
    j["ghost_plates_ignored"] = rep.ghost_plates_ignored;
    j["sign_convention"] = "Welford: W = -OPD von Mikroptim";
    Json totals;
    totals["S"] = Json::array({rep.S[0], rep.S[1], rep.S[2], rep.S[3], number_or_null(rep.S[4])});
    totals["C"] = Json::array({rep.C[0], rep.C[1]});
    totals["spherochromatism"] = rep.spherochromatism;
    totals["waves"] = waves_json(rep.waves());
    j["totals"] = totals;
    Json surfaces = Json::array();
    for (const SurfaceAberration& s : rep.surfaces) {
        Json e;
        e["surface"] = s.surface;
        e["curvature"] = s.curvature;
        e["n_before"] = s.n_before;
        e["n_after"] = s.n_after;
        e["y"] = s.y;
        e["u_before"] = s.u_before;
        e["u_after"] = s.u_after;
        e["ybar"] = s.ybar;
        e["ubar_before"] = s.ubar_before;
        e["ubar_after"] = s.ubar_after;
        e["A"] = s.A;
        e["Abar"] = s.Abar;
        e["S"] = Json::array({s.S[0], s.S[1], s.S[2], s.S[3], number_or_null(s.S[4])});
        e["C"] = Json::array({s.C[0], s.C[1]});
        e["spherochromatism"] = s.spherochromatism;
        e["waves"] = waves_json(rep.waves(s));
        surfaces.push_back(e);
    }
    j["surfaces"] = surfaces;
    return j;
}

void print_text(std::ostream& out, const AberrationReport& rep, const std::string& file) {
    out << "Aberrationsbericht: " << file << '\n';
    kv(out, "Wellenlängen",
       fixed(rep.reference_nm, 2) + " nm (Referenz), " + fixed(rep.short_nm, 2) + " / " +
           fixed(rep.long_nm, 2) + " nm");
    kv(out, "Feldwinkel", fixed(rep.field_angle_deg, 4) + "°");
    kv(out, "Lagrange-Invariante", sci(rep.lagrange, 6) + " mm");
    if (rep.ghost_plates_ignored > 0) {
        kv(out, "Hinweis",
           std::to_string(rep.ghost_plates_ignored) +
               " Geisterplatte(n) mit Koeffizienten ≠ 0 nicht berücksichtigt");
    }
    out << "Beiträge in Wellen bei λ_ref (Welford, W = −OPD):\n";
    out << pad_left("Fl.", 4) << pad_left("y", 12) << pad_left("W040", 12) << pad_left("W131", 12)
        << pad_left("W222", 12) << pad_left("W220", 12) << pad_left("W311", 12)
        << pad_left("ΔW020", 12) << pad_left("ΔW111", 12) << pad_left("ΔW040", 12) << '\n';
    auto row = [&](const std::string& label, double y, const SeidelWaves& w) {
        out << pad_left(label, 4) << pad_left(std::isfinite(y) ? fixed(y, 4) : "", 12)
            << pad_left(sci(w.W040, 3), 12) << pad_left(sci(w.W131, 3), 12)
            << pad_left(sci(w.W222, 3), 12) << pad_left(sci(w.W220, 3), 12)
            << pad_left(std::isfinite(w.W311) ? sci(w.W311, 3) : "–", 12)
            << pad_left(sci(w.dW020, 3), 12) << pad_left(sci(w.dW111, 3), 12)
            << pad_left(sci(w.dW040, 3), 12) << '\n';
    };
    for (const SurfaceAberration& s : rep.surfaces) {
        row(std::to_string(s.surface), s.y, rep.waves(s));
    }
    row("Σ", std::nan(""), rep.waves());
}

} // namespace

void print_aberrations_usage(std::ostream& out) {
    out << "Aufruf: mikroptim aberrations <system.json> [--json] [--field-deg θ]\n"
           "                              [--reference-nm λ] [--short-nm λ] [--long-nm λ]\n"
           "Aberrationsbericht pro Fläche: paraxiale Rand- und Hauptstrahlen, Seidel-Beiträge\n"
           "S_I … S_V, chromatische Beiträge C_I, C_II und Sphärochromasie (docs/numerik.md,\n"
           "Abschnitt 14). Nur kollimierte Quelle; Geisterplatten werden nicht berücksichtigt.\n"
           "Optionen:\n"
           "  --json               maschinenlesbare Ausgabe (Format mikroptim.aberrations/1)\n"
           "  --field-deg θ        Feldwinkel des Hauptstrahls (Standard: größter der Datei)\n"
           "  --reference-nm λ     Referenzwellenlänge (Standard: spectrum.reference_nm)\n"
           "  --short-nm λ         kurze Wellenlänge für C_I, C_II, Sphärochromasie\n"
           "  --long-nm λ          lange Wellenlänge (Standard: kürzeste/längste der Datei)\n"
           "  --help               diese Hilfe\n"
           "Exit-Code: 0 Erfolg, 1 Fehler beim Laden oder Rechnen, 2 ungültige Kommandozeile.\n";
}

int run_aberrations_command(std::span<const std::string_view> args, std::ostream& out,
                            std::ostream& err) {
    AberrationArgs a;
    try {
        a = parse_args(args);
    } catch (const UsageError& e) {
        err << "mikroptim aberrations: " << e.what() << '\n';
        print_aberrations_usage(err);
        return kExitUsage;
    }
    if (a.help) {
        print_aberrations_usage(out);
        return kExitOk;
    }
    try {
        const LoadedSystem l = load_system(a.file);
        const AberrationReport rep = compute_aberrations(l.system, l.catalog, a.options);
        if (a.json) {
            out << report_json(rep, a.file).dump(2) << '\n';
        } else {
            print_text(out, rep, a.file);
        }
    } catch (const std::exception& e) {
        err << "mikroptim aberrations: Fehler: " << e.what() << '\n';
        return kExitError;
    }
    return kExitOk;
}

} // namespace mko::cli
