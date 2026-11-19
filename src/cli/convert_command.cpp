#include "cli/convert_command.hpp"

#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/io/raytatouille.hpp"
#include "mikroptim/io/system_json.hpp"

#include <nlohmann/json.hpp>

#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace mko::cli {

namespace {

namespace fs = std::filesystem;

struct ConvertArgs {
    std::string input;
    std::string output;
    std::vector<std::string> glass_catalogs;
    std::string rtt_catalog;
    std::optional<double> semi_diameter;
    bool help = false;
};

std::string_view next_value(std::span<const std::string_view> args, std::size_t& i,
                            std::string_view option) {
    if (i + 1 >= args.size()) {
        throw UsageError(std::string(option) + " erwartet einen Wert");
    }
    return args[++i];
}

ConvertArgs parse_args(std::span<const std::string_view> args) {
    ConvertArgs a;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--help" || arg == "-h") {
            a.help = true;
        } else if (arg == "-o" || arg == "--output") {
            a.output = std::string(next_value(args, i, arg));
        } else if (arg == "--glass-catalog") {
            a.glass_catalogs.emplace_back(next_value(args, i, arg));
        } else if (arg == "--rtt-catalog") {
            a.rtt_catalog = std::string(next_value(args, i, arg));
            if (a.rtt_catalog.empty() || a.rtt_catalog.find(':') != std::string::npos) {
                throw UsageError("--rtt-catalog: Katalogname ohne ':' erwartet");
            }
        } else if (arg == "--semi-diameter") {
            a.semi_diameter = parse_double(arg, next_value(args, i, arg));
            if (!(*a.semi_diameter > 0.0)) {
                throw UsageError("--semi-diameter muss > 0 sein");
            }
        } else if (!arg.empty() && arg.front() == '-') {
            throw UsageError("unbekannte Option " + std::string(arg));
        } else if (a.input.empty()) {
            a.input = std::string(arg);
        } else {
            throw UsageError("mehr als eine Eingabedatei");
        }
    }
    if (a.help) {
        return a;
    }
    if (a.input.empty()) {
        throw UsageError("Eingabedatei fehlt");
    }
    if (a.output.empty()) {
        throw UsageError("Ausgabedatei fehlt (-o)");
    }
    if (same_file(a.input, a.output)) {
        throw UsageError("Ausgabedatei ist die Eingabedatei");
    }
    return a;
}

std::string read_text(const fs::path& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Datei kann nicht geöffnet werden: " + utf8(file));
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

void write_text(const fs::path& file, const std::string& text) {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Datei kann nicht geschrieben werden: " + utf8(file));
    }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.close();
    if (!out) {
        throw std::runtime_error("Fehler beim Schreiben: " + utf8(file));
    }
}

// Katalogname im Sinn von Raytatouille: Dateistamm in Großbuchstaben (`schott.json` → SCHOTT).
std::string catalog_name(const fs::path& file) {
    std::string s = utf8(file.stem());
    for (char& ch : s) {
        if (ch >= 'a' && ch <= 'z') {
            ch = static_cast<char>(ch - 'a' + 'A');
        }
    }
    return s;
}

struct Catalogs {
    std::vector<std::pair<std::string, GlassCatalog>> files; // Katalogname, Inhalt

    std::string name_of(std::string_view glass) const {
        for (const auto& [name, cat] : files) {
            if (cat.contains(glass)) {
                return name;
            }
        }
        return {};
    }
};

Catalogs load_catalogs(const std::vector<fs::path>& files) {
    Catalogs c;
    for (const fs::path& f : files) {
        c.files.emplace_back(catalog_name(f), GlassCatalog::from_file(f));
    }
    return c;
}

void print_warnings(std::ostream& err, const std::vector<std::string>& warnings) {
    for (const std::string& w : warnings) {
        err << "mikroptim convert: Warnung: " << w << '\n';
    }
}

int rtt_to_mikroptim(const ConvertArgs& a, const std::string& text, std::ostream& out,
                     std::ostream& err) {
    std::vector<fs::path> files;
    SystemDescription probe;
    for (const std::string& c : a.glass_catalogs) {
        files.emplace_back(c);
        probe.glass_catalogs.push_back(c);
    }
    const Catalogs catalogs = load_catalogs(files);
    RttImportOptions options;
    std::error_code ec;
    const fs::path out_dir = fs::absolute(fs::path(a.output), ec).parent_path();
    std::vector<fs::path> resolved;
    for (const fs::path& f : files) {
        resolved.push_back(fs::absolute(f, ec));
    }
    options.glass_catalogs = catalogs_for(probe, resolved, out_dir);
    if (!catalogs.files.empty()) {
        options.glass_exists = [&catalogs](std::string_view g) {
            return !catalogs.name_of(g).empty();
        };
        options.glass_catalog_name = [&catalogs](std::string_view g) {
            return catalogs.name_of(g);
        };
    }
    options.default_semi_diameter = a.semi_diameter;
    options.fallback_name = utf8(fs::path(a.input).stem().stem());
    std::vector<std::string> warnings;
    const SystemDescription sys = system_from_rtt_json(text, options, &warnings);
    print_warnings(err, warnings);
    write_system_file(sys, a.output);
    out << "Raytatouille → Mikroptim: " << a.output << " (" << sys.surfaces.size() << " Flächen, "
        << warnings.size() << " Warnungen)\n";
    return kExitOk;
}

int mikroptim_to_rtt(const ConvertArgs& a, std::ostream& out, std::ostream& err) {
    if (!a.glass_catalogs.empty()) {
        throw UsageError("--glass-catalog nur für Raytatouille → Mikroptim (die Kataloge "
                         "stehen in der Systemdatei)");
    }
    if (a.semi_diameter) {
        throw UsageError("--semi-diameter nur für Raytatouille → Mikroptim");
    }
    std::vector<std::string> warnings;
    const SystemDescription sys = read_system_file(a.input, &warnings);
    const std::vector<fs::path> files =
        resolve_glass_catalogs(sys, fs::path(a.input).parent_path());
    const Catalogs catalogs = load_catalogs(files);
    RttExportOptions options;
    const std::string fixed = a.rtt_catalog;
    options.glass_catalog_name = [&catalogs, fixed](std::string_view g) {
        const std::string own = catalogs.name_of(g);
        return own.empty() || fixed.empty() ? own : fixed;
    };
    const std::string text = system_to_rtt_json(sys, options, &warnings);
    print_warnings(err, warnings);
    write_text(a.output, text);
    out << "Mikroptim → Raytatouille: " << a.output << " (" << sys.surfaces.size() << " Flächen, "
        << warnings.size() << " Warnungen)\n";
    return kExitOk;
}

} // namespace

void print_convert_usage(std::ostream& out) {
    out << "Aufruf: mikroptim convert <eingabe> -o <ausgabe> [--glass-catalog <katalog.json>]...\n"
           "                          [--semi-diameter MM] [--rtt-catalog NAME]\n"
           "Umwandlung zwischen Raytatouille (*.rtt.json, Schema 0.11) und mikroptim.system/1\n"
           "für sequenzielle, koaxiale, sphärische Systeme (M9-T2, docs/dateiformat.md,\n"
           "Abschnitt „Raytatouille-Adapter“). Die Richtung ergibt sich aus der Eingabe\n"
           "(schema_version: Raytatouille, format: Mikroptim). Nicht abbildbare Stellen sind\n"
           "Fehler (alle werden gemeldet); entfallende Angaben ohne optische Wirkung Warnungen.\n"
           "  -o DATEI             Ausgabedatei (Pflicht, nicht die Eingabedatei)\n"
           "  --glass-catalog K    nur Raytatouille → Mikroptim: Mikroptim-Glaskatalog für\n"
           "                       KATALOG:NAME (mehrfach möglich; wird in glass_catalogs\n"
           "                       eingetragen)\n"
           "  --semi-diameter MM   nur Raytatouille → Mikroptim: Halbdurchmesser für Flächen\n"
           "                       ohne Apertur (sonst Fehler)\n"
           "  --rtt-catalog NAME   nur Mikroptim → Raytatouille: Katalogname aller Gläser\n"
           "                       (Standard: Dateistamm des Glaskatalogs in Großbuchstaben)\n"
           "Exit-Code: 0 Erfolg, 1 Fehler (nicht abbildbar, Datei, Format), 2 ungültige\n"
           "Kommandozeile.\n";
}

int run_convert_command(std::span<const std::string_view> args, std::ostream& out,
                        std::ostream& err) {
    ConvertArgs a;
    try {
        a = parse_args(args);
    } catch (const UsageError& e) {
        err << "mikroptim convert: " << e.what() << '\n';
        print_convert_usage(err);
        return kExitUsage;
    }
    if (a.help) {
        print_convert_usage(out);
        return kExitOk;
    }
    try {
        const std::string text = read_text(a.input);
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(text);
        } catch (const nlohmann::json::exception& e) {
            throw FormatError("", std::string("JSON-Syntaxfehler: ") + e.what(), a.input);
        }
        if (j.is_object() && j.contains("schema_version")) {
            return rtt_to_mikroptim(a, text, out, err);
        }
        if (j.is_object() && j.contains("format") && j["format"].is_string() &&
            j["format"].get<std::string>().rfind("mikroptim.system/", 0) == 0) {
            return mikroptim_to_rtt(a, out, err);
        }
        throw FormatError("",
                          "weder Raytatouille (schema_version) noch mikroptim.system "
                          "(format)",
                          a.input);
    } catch (const UsageError& e) {
        err << "mikroptim convert: " << e.what() << '\n';
        print_convert_usage(err);
        return kExitUsage;
    } catch (const RttConversionError& e) {
        err << "mikroptim convert: " << a.input << ": " << e.what() << '\n';
        return kExitError;
    } catch (const std::exception& e) {
        err << "mikroptim convert: " << e.what() << '\n';
        return kExitError;
    }
}

} // namespace mko::cli
