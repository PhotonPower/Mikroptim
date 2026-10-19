// Unterbefehl `mikroptim insert` (M6-T2): Systemdatei laden, Operator A (insert_meniscus) oder B
// (split_lens, optional free_glass) anwenden, Merit vorher und nachher auf der CPU, neues System
// schreiben (Glaskataloge relativ zum Verzeichnis der Ergebnisdatei). docs/cli.md.

#include "cli/insert_command.hpp"
#include "cli/cli_util.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/optim/extra_residuals.hpp"
#include "mikroptim/optim/ghost_diagnosis.hpp"
#include "mikroptim/optim/ghost_penalty.hpp"
#include "mikroptim/optim/insertion.hpp"
#include "mikroptim/optim/mechanical_constraints.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "mikroptim/paraxial/paraxial.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace mko::cli {

namespace {

namespace fs = std::filesystem;

struct InsertArgs {
    std::string file;
    std::string output;
    bool help = false;
    std::optional<MeniscusInsertion> meniscus;
    std::optional<LensSplit> split;
    bool free_glass = false;
    bool variables = true;
    double thickness = 0.0;
};

std::string_view next_value(std::span<const std::string_view> args, std::size_t& i,
                            std::string_view option) {
    if (i + 1 >= args.size()) {
        throw UsageError(std::string(option) + " erwartet einen Wert");
    }
    return args[++i];
}

std::vector<std::string_view> split_commas(std::string_view v) {
    std::vector<std::string_view> out;
    std::size_t start = 0;
    while (true) {
        const std::size_t comma = v.find(',', start);
        out.push_back(v.substr(start, comma == std::string_view::npos ? v.npos : comma - start));
        if (comma == std::string_view::npos) {
            return out;
        }
        start = comma + 1;
    }
}

// Glas: Katalogname oder "n=<Zahl>" (konstante Brechzahl).
MaterialSpec parse_glass(std::string_view option, std::string_view v) {
    MaterialSpec m;
    if (v.substr(0, 2) == "n=") {
        m.kind = MaterialKind::Constant;
        m.n = parse_double(option, v.substr(2));
        return m;
    }
    if (v.empty() || v == "AIR") {
        throw UsageError(std::string(option) + ": Glas fehlt (Katalogname oder n=<Brechzahl>)");
    }
    m.kind = MaterialKind::Catalog;
    m.name = std::string(v);
    return m;
}

InsertArgs parse_args(std::span<const std::string_view> args) {
    InsertArgs a;
    bool have_file = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--help" || arg == "-h") {
            a.help = true;
        } else if (arg == "-o" || arg == "--output") {
            a.output = std::string(next_value(args, i, arg));
        } else if (arg == "--meniscus") {
            const std::vector<std::string_view> p = split_commas(next_value(args, i, arg));
            if (p.size() != 4) {
                throw UsageError("--meniscus erwartet I,POS,C0,GLAS");
            }
            MeniscusInsertion m;
            m.after_surface = parse_int(arg, p[0]);
            m.position = parse_double(arg, p[1]);
            m.curvature = parse_double(arg, p[2]);
            m.glass = parse_glass(arg, p[3]);
            a.meniscus = m;
        } else if (arg == "--split") {
            const std::vector<std::string_view> p = split_commas(next_value(args, i, arg));
            if (p.size() != 2 && p.size() != 3) {
                throw UsageError("--split erwartet I,C_S oder I,C_S,ANTEIL");
            }
            LensSplit s;
            s.lens_surface = parse_int(arg, p[0]);
            s.curvature = parse_double(arg, p[1]);
            if (p.size() == 3) {
                s.fraction = parse_double(arg, p[2]);
            }
            a.split = s;
        } else if (arg == "--thickness") {
            a.thickness = parse_double(arg, next_value(args, i, arg));
        } else if (arg == "--free-glass") {
            a.free_glass = true;
        } else if (arg == "--no-variables") {
            a.variables = false;
        } else if (!arg.empty() && arg.front() == '-') {
            throw UsageError("unbekannte Option '" + std::string(arg) + "'");
        } else if (!have_file) {
            a.file = std::string(arg);
            have_file = true;
        } else {
            throw UsageError("mehr als eine Systemdatei angegeben");
        }
    }
    if (a.help) {
        return a;
    }
    if (!have_file) {
        throw UsageError("Systemdatei fehlt");
    }
    if (a.meniscus.has_value() == a.split.has_value()) {
        throw UsageError("genau einer von --meniscus und --split ist nötig");
    }
    if (a.output.empty()) {
        throw UsageError("Ergebnisdatei fehlt (-o <datei>)");
    }
    if (a.free_glass && !a.split) {
        throw UsageError("--free-glass nur mit --split");
    }
    if (a.thickness != 0.0 && !a.meniscus) {
        throw UsageError("--thickness nur mit --meniscus");
    }
    if (a.meniscus) {
        a.meniscus->thickness = a.thickness;
        a.meniscus->curvature_variables.enabled = a.variables;
    } else {
        a.split->curvature_variable.enabled = a.variables;
    }
    return a;
}

// F wie im LM ohne Glas-Hülle: Strahl- und Strafresiduen, paraxiale Nebenbedingungen, L1-Strafe,
// mechanische Bedingungen (M4-T3).
double cpu_merit(const LoadedSystem& l) {
    const SystemBatchHost b = make_system_batch(l, 1);
    MechanicalResiduals mechanical =
        make_mechanical_residuals(make_mechanical_constraints(l.system));
    const ExtraResiduals extra =
        concat_extra_residuals({make_paraxial_residuals(make_paraxial_constraints(l.system)),
                                make_ghost_penalty_residuals(make_ghost_penalty(l.system)),
                                std::move(mechanical.geometric)});
    return make_cpu_merit({}, extra, 0, std::move(mechanical.traced))(b).at(0);
}

} // namespace

void print_insert_usage(std::ostream& out) {
    out << "Aufruf: mikroptim insert <system.json> -o <neu.json>\n"
           "                         (--meniscus I,POS,C0,GLAS [--thickness T] |\n"
           "                          --split I,C_S[,ANTEIL] [--free-glass]) [--no-variables]\n"
           "Einfüge-Operatoren der Systemsynthese (docs/optimierung.md, Abschnitt 6).\n"
           "  --meniscus I,POS,C0,GLAS  Operator A: zwei Flächen der Krümmung C0 [1/mm] im\n"
           "                            Luftspalt nach Fläche I, Abstand POS [mm] vom Scheitel\n"
           "                            von I; GLAS ist ein Katalogname oder n=<Brechzahl>\n"
           "  --thickness T             Mittendicke des Meniskus [mm] (Standard 0: Merit\n"
           "                            unverändert)\n"
           "  --split I,C_S[,ANTEIL]    Operator B: Linse aus Fläche I und I+1 an einer\n"
           "                            Kittfläche der Krümmung C_S teilen; ANTEIL der\n"
           "                            Mittendicke vor der Kittfläche (Standard 0,5)\n"
           "  --free-glass              Glas des hinteren Teils als Variable freigeben\n"
           "  --no-variables            keine Krümmungsvariablen für die neuen Flächen\n"
           "  -o, --output <datei>      neues System (Pflicht)\n";
}

int run_insert_command(std::span<const std::string_view> args, std::ostream& out,
                       std::ostream& err) {
    InsertArgs a;
    try {
        a = parse_args(args);
    } catch (const UsageError& e) {
        err << "mikroptim insert: " << e.what() << '\n';
        print_insert_usage(err);
        return kExitUsage;
    }
    if (a.help) {
        print_insert_usage(out);
        return kExitOk;
    }
    {
        // Die Eingabe wird nie überschrieben (wie bei `mikroptim optimize`)
        std::error_code ec;
        if (fs::exists(a.output, ec) && fs::equivalent(a.output, a.file, ec) && !ec) {
            err << "mikroptim insert: Ergebnisdatei ist die Eingabedatei\n";
            return kExitUsage;
        }
    }
    try {
        const LoadedSystem before = load_system(a.file);
        InsertionResult r;
        if (a.meniscus) {
            r = insert_meniscus(before.system, *a.meniscus);
        } else {
            r = split_lens(before.system, *a.split);
            if (a.free_glass) {
                free_glass(r.system, r.first_new_surface);
            }
        }
        const fs::path input_dir = fs::path(a.file).parent_path();
        const LoadedSystem after = prepare_system(r.system, input_dir);
        const double f_before = cpu_merit(before);
        const double f_after = cpu_merit(after);
        const double rel = std::abs(f_after - f_before) / std::abs(f_before);

        const fs::path output(a.output);
        std::error_code ec;
        fs::path out_dir = fs::absolute(output, ec).parent_path();
        if (ec) {
            out_dir = output.parent_path();
        }
        SystemDescription file_system = r.system;
        file_system.glass_catalogs = catalogs_for(r.system, after.catalog_files, out_dir);
        write_system_file(file_system, output);

        out << "mikroptim insert: "
            << (a.meniscus ? "Operator A (Meniskus)" : "Operator B (Kittglied)") << '\n';
        kv(out, "Eingefügte Flächen",
           std::to_string(r.first_new_surface) +
               (r.new_surfaces > 1
                    ? " bis " + std::to_string(r.first_new_surface + r.new_surfaces - 1)
                    : std::string()));
        kv(out, "Flächen vorher / nachher",
           std::to_string(before.system.surfaces.size()) + " / " +
               std::to_string(r.system.surfaces.size()));
        kv(out, "Merit vorher [Wellen²]", sci(f_before, 15));
        kv(out, "Merit nachher [Wellen²]", sci(f_after, 15));
        kv(out, "relative Änderung", sci(rel, 2));
        const double fl_before = compute_paraxial(before.system, before.catalog).focal_length;
        const double fl_after = compute_paraxial(after.system, after.catalog).focal_length;
        kv(out, "Brennweite vorher / nachher [mm]",
           fixed(fl_before, 9) + " / " + fixed(fl_after, 9));
        out << "Ergebnis geschrieben: " << a.output << '\n';
        out.flush();
        if (!out) {
            err << "mikroptim insert: Fehler beim Schreiben der Ausgabe\n";
            return kExitError;
        }
        return kExitOk;
    } catch (const std::exception& e) {
        err << "mikroptim insert: Fehler: " << e.what() << '\n';
        return kExitError;
    }
}

} // namespace mko::cli
