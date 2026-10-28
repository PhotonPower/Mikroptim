#include "cli/glasses_command.hpp"

#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/glass/glass_preselection.hpp"
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
    GlassPreselectionOptions options;
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

GlassesArgs parse_args(std::span<const std::string_view> args) {
    GlassesArgs a;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--help" || arg == "-h") {
            a.help = true;
        } else if (arg == "--json") {
            a.json = true;
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
    return a;
}

} // namespace

void print_glasses_usage(std::ostream& out) {
    out << "Aufruf: mikroptim glasses <katalog.json>... [--system <system.json>] [--order 1|2]\n"
           "                          [--fixed GLAS]... [--weights w1,w2[,w3]] [--power P]\n"
           "                          [--max-power-sum S] [--top N] [--json]\n"
           "Algebraische Glasvorauswahl (M11-T5): Paare (Achromat, --order 1) bzw. Tripel\n"
           "(Apochromat, --order 2) dünner Elemente, deren gewichtete Brechkraft in den\n"
           "Buchdahl-Termen bis zur Ordnung k farbfrei ist; Rangfolge nach dem Rest der Ordnung\n"
           "k + 1 (sekundäres bzw. tertiäres Spektrum), dann nach Σ|w φ|/|φ| (docs/numerik.md,\n"
           "Abschnitt 15).\n"
           "  --system DATEI       Kataloge aus glass_catalogs der Systemdatei\n"
           "  --order K            1 Achromat (Paare, Standard), 2 Apochromat (Tripel)\n"
           "  --fixed GLAS         Glas, das jede Kombination enthält (mehrfach möglich)\n"
           "  --weights W          Gewichte w_i = h_i² (Randstrahlhöhen, normiert; Standard 1)\n"
           "  --power P            Gesamtbrechkraft φ [1/mm] (Standard 1: Anteile)\n"
           "  --max-power-sum S    verwerfen bei Σ|w φ|/|φ| > S (Standard 10)\n"
           "  --top N              höchstens N Ergebnisse (Standard 20)\n"
           "  --json               Ausgabe im Format mikroptim.glasses/1\n"
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
            for (const fs::path& p :
                 resolve_glass_catalogs(sys, fs::path(a.system).parent_path())) {
                files.push_back(p);
            }
        }
        GlassCatalog cat;
        for (const fs::path& f : files) {
            cat.merge(GlassCatalog::from_file(f));
        }
        std::vector<std::string> skipped;
        const std::vector<GlassDispersion> g = catalog_dispersions(cat, {}, &skipped);
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
        if (a.json) {
            Json j;
            j["format"] = "mikroptim.glasses/1";
            j["version"] = std::string(kVersion);
            j["order"] = a.options.order;
            j["power"] = a.options.power;
            j["glasses"] = g.size();
            j["skipped"] = skipped;
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
                e["dispersion"] = eta;
                list.push_back(e);
            }
            j["results"] = list;
            out << j.dump(2) << '\n';
        } else {
            out << "Glasvorauswahl "
                << (a.options.order == 1 ? "Achromat (Paare)" : "Apochromat (Tripel)") << ": "
                << g.size() << " Gläser";
            if (!skipped.empty()) {
                out << ", " << skipped.size() << " ohne Buchdahl-Fit übersprungen";
            }
            out << '\n';
            out << pad_left("Rang", 5) << "  " << pad_right("Gläser", 36) << pad_left("Rest", 12)
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
                    << pad_left(sci(c.residual, 3), 12) << pad_left(fixed(c.power_sum, 3), 11)
                    << "  " << powers << '\n';
            }
        }
    } catch (const std::exception& e) {
        err << "mikroptim glasses: Fehler: " << e.what() << '\n';
        return kExitError;
    }
    return kExitOk;
}

} // namespace mko::cli
