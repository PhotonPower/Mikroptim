#include "cli/glasses_command.hpp"

#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/glass/glass_preselection.hpp"
#include "mikroptim/glass/glass_properties.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/version.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

namespace mko::cli {
namespace {

using Json = nlohmann::ordered_json;
namespace fs = std::filesystem;

struct GlassesArgs {
    std::vector<std::string> catalogs;
    std::string system;
    std::vector<std::string> fixed;
    bool help = false;
    bool json = false;
    bool list = false;     // Kennzahlen und Bereichsfilter (M10-T5) statt Vorauswahl
    bool radii = false;    // Startkrümmungen des Glieds im Text (M11-T5g)
    bool spectrum = false; // Spektrum der Systemdatei für den spektralen Rest (M11-T5g)
    MemberBending bending = MemberBending::MinSpherical;
    GlassPreselectionOptions options;
    // Bereiche wie --list; dazu Namenslisten für die Vorauswahl (M11-T5g)
    GlassCandidateFilter filter;
};

std::string_view next_value(std::span<const std::string_view> args, std::size_t& i,
                            std::string_view option) {
    if (i + 1 >= args.size()) {
        throw UsageError(std::string(option) + " erwartet einen Wert");
    }
    return args[++i];
}

std::vector<double> parse_list(std::string_view option, std::string_view text) {
    std::vector<double> out;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string_view part = text.substr(
            start, comma == std::string_view::npos ? std::string_view::npos : comma - start);
        out.push_back(parse_double(option, part));
        if (comma == std::string_view::npos) {
            break;
        }
        start = comma + 1;
    }
    return out;
}

std::vector<std::string> parse_names(std::string_view option, std::string_view text) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string_view part = text.substr(
            start, comma == std::string_view::npos ? std::string_view::npos : comma - start);
        if (part.empty()) {
            throw UsageError(std::string(option) + ": leerer Glasname");
        }
        out.emplace_back(part);
        if (comma == std::string_view::npos) {
            break;
        }
        start = comma + 1;
    }
    return out;
}

// "min,max"; eine leere Seite ist offen (z. B. ",35" oder "1.6,")
ValueRange parse_range(std::string_view option, std::string_view text) {
    const std::size_t comma = text.find(',');
    if (comma == std::string_view::npos || text.find(',', comma + 1) != std::string_view::npos) {
        throw UsageError(std::string(option) + " erwartet min,max (eine Seite darf leer sein)");
    }
    ValueRange r;
    const std::string_view lo = text.substr(0, comma);
    const std::string_view hi = text.substr(comma + 1);
    if (!lo.empty()) {
        r.min = parse_double(option, lo);
    }
    if (!hi.empty()) {
        r.max = parse_double(option, hi);
    }
    return r;
}

GlassesArgs parse_args(std::span<const std::string_view> args) {
    GlassesArgs a;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--help" || arg == "-h") {
            a.help = true;
        } else if (arg == "--json") {
            a.json = true;
        } else if (arg == "--list") {
            a.list = true;
        } else if (arg == "--nd") {
            a.filter.properties.n_d = parse_range(arg, next_value(args, i, arg));
        } else if (arg == "--vd") {
            a.filter.properties.v_d = parse_range(arg, next_value(args, i, arg));
        } else if (arg == "--pgf") {
            a.filter.properties.p_gF = parse_range(arg, next_value(args, i, arg));
        } else if (arg == "--dpgf") {
            a.filter.properties.dp_gF = parse_range(arg, next_value(args, i, arg));
        } else if (arg == "--only") {
            for (std::string& n : parse_names(arg, next_value(args, i, arg))) {
                a.filter.only.push_back(std::move(n));
            }
        } else if (arg == "--exclude") {
            for (std::string& n : parse_names(arg, next_value(args, i, arg))) {
                a.filter.exclude.push_back(std::move(n));
            }
        } else if (arg == "--wavelengths") {
            a.options.wavelengths_nm = parse_list(arg, next_value(args, i, arg));
            if (a.options.wavelengths_nm.size() < 2) {
                throw UsageError("--wavelengths braucht mindestens zwei Wellenlängen");
            }
        } else if (arg == "--spectrum") {
            a.spectrum = true;
        } else if (arg == "--radii") {
            a.radii = true;
        } else if (arg == "--bending") {
            const std::string_view v = next_value(args, i, arg);
            if (v == "spherical") {
                a.bending = MemberBending::MinSpherical;
            } else if (v == "coma") {
                a.bending = MemberBending::ZeroComa;
            } else {
                throw UsageError("--bending erwartet spherical oder coma");
            }
        } else if (arg == "--system") {
            a.system = std::string(next_value(args, i, arg));
        } else if (arg == "--order") {
            a.options.order = parse_int(arg, next_value(args, i, arg));
            if (a.options.order != 1 && a.options.order != 2) {
                throw UsageError("--order muss 1 oder 2 sein");
            }
        } else if (arg == "--fixed") {
            a.fixed.emplace_back(next_value(args, i, arg));
        } else if (arg == "--weights") {
            a.options.weights = parse_list(arg, next_value(args, i, arg));
        } else if (arg == "--power") {
            a.options.power = parse_double(arg, next_value(args, i, arg));
        } else if (arg == "--max-power-sum") {
            a.options.max_power_sum = parse_double(arg, next_value(args, i, arg));
        } else if (arg == "--top") {
            a.options.max_results = parse_int(arg, next_value(args, i, arg));
            if (a.options.max_results < 1) {
                throw UsageError("--top muss >= 1 sein");
            }
        } else if (!arg.empty() && arg.front() == '-') {
            throw UsageError("unbekannte Option " + std::string(arg));
        } else {
            a.catalogs.emplace_back(arg);
        }
    }
    if (!a.help && a.catalogs.empty() && a.system.empty()) {
        throw UsageError("Glaskatalog oder --system fehlt");
    }
    if (a.list && (!a.filter.only.empty() || !a.filter.exclude.empty() || a.radii || a.spectrum ||
                   !a.options.wavelengths_nm.empty())) {
        throw UsageError(
            "--only, --exclude, --radii, --spectrum und --wavelengths nur ohne --list");
    }
    if (a.spectrum && (a.system.empty() || !a.options.wavelengths_nm.empty())) {
        throw UsageError("--spectrum nur mit --system und ohne --wavelengths");
    }
    return a;
}

void print_list(const GlassCatalog& cat, const GlassesArgs& a, std::ostream& out) {
    std::vector<std::string> skipped;
    const std::vector<GlassProperties> list =
        list_glass_properties(cat, a.filter.properties, &skipped);
    if (a.json) {
        Json j;
        j["format"] = "mikroptim.glasses/1";
        j["version"] = std::string(kVersion);
        j["mode"] = "list";
        j["count"] = list.size();
        j["skipped"] = skipped;
        Json arr = Json::array();
        for (const GlassProperties& p : list) {
            arr.push_back(Json{{"name", p.name},
                               {"n_d", p.n_d},
                               {"n_F", p.n_F},
                               {"n_C", p.n_C},
                               {"n_g", p.n_g},
                               {"v_d", p.v_d},
                               {"p_gF", p.p_gF},
                               {"dp_gF", p.dp_gF},
                               {"range_nm", Json::array({p.range_nm[0], p.range_nm[1]})}});
        }
        j["list"] = arr;
        out << j.dump(2) << '\n';
        return;
    }
    out << "Glaskennzahlen: " << list.size() << " Gläser";
    if (!skipped.empty()) {
        out << ", " << skipped.size() << " ohne die Linien g bis C übersprungen";
    }
    // ν und Δ sind in UTF-8 zwei Bytes, pad_left zählt Bytes
    out << '\n'
        << pad_right("Glas", 16) << pad_left("n_d", 10) << pad_left("ν_d", 10)
        << pad_left("P_g,F", 9) << pad_left("ΔP_g,F", 11) << '\n';
    for (const GlassProperties& p : list) {
        out << pad_right(p.name, 16) << pad_left(fixed(p.n_d, 5), 10)
            << pad_left(fixed(p.v_d, 2), 9) << pad_left(fixed(p.p_gF, 4), 9)
            << pad_left(fixed(p.dp_gF, 4), 10) << '\n';
    }
}

} // namespace

void print_glasses_usage(std::ostream& out) {
    out << "Aufruf: mikroptim glasses <katalog.json>... [--system <system.json>] [--order 1|2]\n"
           "                          [--fixed GLAS]... [--weights w1,w2[,w3]] [--power P]\n"
           "                          [--max-power-sum S] [--top N] [--json]\n"
           "                          [--nd|--vd|--pgf|--dpgf MIN,MAX] [--only G1,G2,...]\n"
           "                          [--exclude G1,...] [--radii] [--bending spherical|coma]\n"
           "                          [--wavelengths L1,L2,... | --spectrum]\n"
           "       mikroptim glasses <katalog.json>... [--system <system.json>] --list\n"
           "                          [--nd MIN,MAX] [--vd MIN,MAX] [--pgf MIN,MAX]\n"
           "                          [--dpgf MIN,MAX] [--json]\n"
           "Algebraische Glasvorauswahl (M11-T5): Paare (Achromat, --order 1) bzw. Tripel\n"
           "(Apochromat, --order 2) dünner Elemente, deren gewichtete Brechkraft in den\n"
           "Buchdahl-Termen bis zur Ordnung k farbfrei ist; Rangfolge nach dem Rest der Ordnung\n"
           "k + 1 (sekundäres bzw. tertiäres Spektrum; mit --wavelengths bzw. --spectrum nach\n"
           "dem spektralen Rest), dann nach Σ|w φ|/|φ| (docs/numerik.md, Abschnitt 15).\n"
           "  --system DATEI       Kataloge aus glass_catalogs der Systemdatei\n"
           "  --order K            1 Achromat (Paare, Standard), 2 Apochromat (Tripel)\n"
           "  --fixed GLAS         Glas, das jede Kombination enthält (mehrfach möglich)\n"
           "  --weights W          Gewichte w_i = h_i² (Randstrahlhöhen, normiert; Standard 1)\n"
           "  --power P            Gesamtbrechkraft φ [1/mm] (Standard 1: Anteile)\n"
           "  --max-power-sum S    verwerfen bei Σ|w φ|/|φ| > S (Standard 10)\n"
           "  --top N              höchstens N Ergebnisse (Standard 20)\n"
           "  --list               Kennzahlen n_d, ν_d, P_g,F, ΔP_g,F aller Gläser (M10-T5,\n"
           "                       docs/numerik.md, Abschnitt 17) statt der Vorauswahl\n"
           "  --nd, --vd, --pgf, --dpgf MIN,MAX\n"
           "                       geschlossener Bereich, eine Seite darf leer sein (z. B.\n"
           "                       --vd ,35); mit --list Auswahl der Liste, sonst Filter der\n"
           "                       Kandidaten der Vorauswahl (feste Gläser ausgenommen)\n"
           "  --only G1,G2,...     nur diese Gläser als Kandidaten (Vorauswahl)\n"
           "  --exclude G1,...     diese Gläser nie (Vorauswahl)\n"
           "  --wavelengths L1,L2,...\n"
           "                       Rang nach dem spektralen Rest: Streuung des Brechkraftrests\n"
           "                       aller nicht korrigierten Buchdahl-Terme über diese\n"
           "                       Wellenlängen [nm] (Farblängsfehler nach bester Fokussierung)\n"
           "  --spectrum           wie --wavelengths mit dem Spektrum der --system-Datei\n"
           "  --radii              Startkrümmungen des dünnen Glieds im Text (JSON: immer, wenn\n"
           "                       alle Gewichte 1 sind; docs/numerik.md, Abschnitt 15)\n"
           "  --bending B          spherical: S_I = 0 bzw. kleinstes |S_I| (Standard),\n"
           "                       coma: S_II = 0\n"
           "  --json             Ausgabe im Format mikroptim.glasses/1\n"
           "Exit-Code: 0 Erfolg, 1 Fehler beim Laden, 2 ungültige Kommandozeile.\n";
}

int run_glasses_command(std::span<const std::string_view> args, std::ostream& out,
                        std::ostream& err) {
    GlassesArgs a;
    try {
        a = parse_args(args);
    } catch (const UsageError& e) {
        err << "mikroptim glasses: " << e.what() << '\n';
        print_glasses_usage(err);
        return kExitUsage;
    }
    if (a.help) {
        print_glasses_usage(out);
        return kExitOk;
    }
    try {
        std::vector<fs::path> files(a.catalogs.begin(), a.catalogs.end());
        if (!a.system.empty()) {
            const SystemDescription sys = read_system_file(a.system);
            if (a.spectrum) {
                a.options.wavelengths_nm = sys.spectrum.wavelengths_nm;
                if (a.options.wavelengths_nm.size() < 2) {
                    a.options.wavelengths_nm.clear(); // monochromatisch: Koeffizient wie bisher
                }
            }
            for (const fs::path& p :
                 resolve_glass_catalogs(sys, fs::path(a.system).parent_path())) {
                files.push_back(p);
            }
        }
        GlassCatalog cat;
        for (const fs::path& f : files) {
            cat.merge(GlassCatalog::from_file(f));
        }
        if (a.list) {
            print_list(cat, a, out);
            return kExitOk;
        }
        std::vector<std::string> skipped;
        const std::vector<GlassDispersion> all = catalog_dispersions(cat, {}, &skipped);
        for (const std::string& name : a.fixed) {
            if (std::none_of(all.begin(), all.end(),
                             [&](const GlassDispersion& d) { return d.name == name; })) {
                throw std::invalid_argument("festes Glas " + name +
                                            " nicht im Katalog oder ohne Buchdahl-Fit");
            }
        }
        // Kandidaten nach dem Filter; feste Gläser bleiben (M11-T5g)
        const std::vector<GlassDispersion> g = filter_dispersions(all, cat, a.filter, a.fixed);
        for (const std::string& name : a.fixed) {
            const auto it = std::find_if(g.begin(), g.end(),
                                         [&](const GlassDispersion& d) { return d.name == name; });
            if (it == g.end()) {
                throw std::invalid_argument("festes Glas " + name +
                                            " nicht im Katalog oder ohne Buchdahl-Fit");
            }
            a.options.fixed.push_back(static_cast<int>(it - g.begin()));
        }
        const std::vector<PreselectedGlasses> r = preselect_glasses(g, a.options);
        const bool spectral = !a.options.wavelengths_nm.empty();
        // Startkrümmungen nur für Elemente in Kontakt (alle Gewichte 1)
        const bool contact = std::all_of(a.options.weights.begin(), a.options.weights.end(),
                                         [](double w) { return w == 1.0; });
        const auto shape = [&](const PreselectedGlasses& c) {
            std::vector<double> n;
            for (int i : c.glasses) {
                n.push_back(g[static_cast<std::size_t>(i)].n0);
            }
            return thin_member_shape(n, c.powers, a.bending);
        };
        if (a.json) {
            Json j;
            j["format"] = "mikroptim.glasses/1";
            j["version"] = std::string(kVersion);
            j["order"] = a.options.order;
            j["power"] = a.options.power;
            j["glasses"] = g.size();
            j["catalog_glasses"] = all.size();
            j["skipped"] = skipped;
            j["bending"] = a.bending == MemberBending::ZeroComa ? "coma" : "spherical";
            j["wavelengths_nm"] = a.options.wavelengths_nm;
            Json list = Json::array();
            for (const PreselectedGlasses& c : r) {
                Json e;
                Json names = Json::array();
                Json eta = Json::array();
                for (int i : c.glasses) {
                    const GlassDispersion& d = g[static_cast<std::size_t>(i)];
                    names.push_back(d.name);
                    eta.push_back(Json{{"n0", d.n0}, {"eta", d.eta}});
                }
                e["glasses"] = names;
                e["powers"] = c.powers;
                e["power_sum"] = c.power_sum;
                e["residual"] = c.residual;
                if (spectral) {
                    e["spectral_residual"] = c.spectral_residual;
                }
                e["dispersion"] = eta;
                if (contact) {
                    const ThinMemberShape m = shape(c);
                    e["curvatures"] = m.curvatures;
                    e["seidel"] = Json{{"S_I", m.spherical},
                                       {"S_II", m.coma},
                                       {"spherical_zero", m.spherical_zero}};
                }
                list.push_back(e);
            }
            j["results"] = list;
            out << j.dump(2) << '\n';
        } else {
            out << "Glasvorauswahl "
                << (a.options.order == 1 ? "Achromat (Paare)" : "Apochromat (Tripel)") << ": "
                << g.size() << " Gläser";
            if (g.size() != all.size()) {
                out << " (von " << all.size() << " nach dem Filter)";
            }
            if (!skipped.empty()) {
                out << ", " << skipped.size() << " ohne Buchdahl-Fit übersprungen";
            }
            out << '\n';
            out << pad_left("Rang", 5) << "  " << pad_right("Gläser", 36)
                << pad_left(spectral ? "Rest (Spektrum)" : "Rest", spectral ? 16 : 12)
                << pad_left("Σ|wφ|/|φ|", 11) << "  Brechkräfte\n";
            int rank = 1;
            for (const PreselectedGlasses& c : r) {
                std::string names;
                std::string powers;
                for (std::size_t i = 0; i < c.glasses.size(); ++i) {
                    names += (i ? " + " : "") + g[static_cast<std::size_t>(c.glasses[i])].name;
                    powers += (i ? ", " : "") + sci(c.powers[i], 3);
                }
                out << pad_left(std::to_string(rank++), 5) << "  " << pad_right(names, 36)
                    << pad_left(sci(spectral ? c.spectral_residual : c.residual, 3),
                                spectral ? 16 : 12)
                    << pad_left(fixed(c.power_sum, 3), 11) << "  " << powers << '\n';
                if (a.radii && contact) {
                    const ThinMemberShape m = shape(c);
                    std::string curv;
                    for (std::size_t i = 0; i < m.curvatures.size(); ++i) {
                        curv += (i ? ", " : "") + sci(m.curvatures[i], 4);
                    }
                    out << "       Krümmungen [1/mm]: " << curv << "  (S_I " << sci(m.spherical, 3)
                        << ", S_II " << sci(m.coma, 3) << ")\n";
                }
            }
        }
    } catch (const std::exception& e) {
        err << "mikroptim glasses: Fehler: " << e.what() << '\n';
        return kExitError;
    }
    return kExitOk;
}

} // namespace mko::cli
