// Unterbefehl `mikroptim insert` (M6-T2, M10-T4): Systemdatei laden, Operator A (insert_meniscus)
// oder B (split_lens, optional free_glass) oder einen Operator der Entwurfssteuerung
// (remove_element, collapse_element, replace_glass, place_ghost_dictionary,
// insert_aplanatic_shell, insert_null_doublet) anwenden, Merit vorher und nachher auf der CPU,
// neues System schreiben (Glaskataloge relativ zum Verzeichnis der Ergebnisdatei). docs/cli.md.

#include "cli/insert_command.hpp"
#include "cli/cli_util.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/optim/design_operators.hpp"
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

#include <nlohmann/json.hpp>

namespace mko::cli {

namespace {

namespace fs = std::filesystem;
using Json = nlohmann::ordered_json;

// Operator eines Aufrufs (genau einer).
enum class Op {
    Meniscus,
    Split,
    Remove,
    Collapse,
    ReplaceGlass,
    GhostDictionary,
    AplanaticShell,
    NullDoublet
};

struct InsertArgs {
    std::string file;
    std::string output;
    bool help = false;
    bool no_reference_warning = false; // --no-reference-warning (ADR-0017)
    std::optional<MeniscusInsertion> meniscus;
    std::optional<LensSplit> split;
    bool free_glass = false;
    std::optional<MaterialSpec> start_glass; // --start-glass (M11-T5)
    bool variables = true;
    double thickness = 0.0;
    // M10-T4
    std::optional<int> remove;                                 // --remove I
    std::optional<std::pair<int, double>> collapse;            // --collapse I,F
    std::optional<std::pair<int, MaterialSpec>> replace_glass; // --replace-glass I,GLAS
    std::optional<GhostDictionaryPlacement> ghost;             // --ghost-dictionary I,DIST
    std::optional<AplanaticShell> shell;                       // --aplanatic-shell I,POS,T,GLAS
    std::optional<NullDoublet> doublet;                        // --null-doublet I,POS,C0,G1,G2
    std::vector<MaterialSpec> ghost_glasses;
    std::optional<int> ghost_order;
    std::optional<double> ghost_weight;
    std::optional<double> norm_radius;
    bool refocus = true;
    bool json = false;
    Op op = Op::Meniscus;
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
        } else if (arg == "--remove") {
            a.remove = parse_int(arg, next_value(args, i, arg));
        } else if (arg == "--collapse") {
            const std::vector<std::string_view> p = split_commas(next_value(args, i, arg));
            if (p.size() != 2) {
                throw UsageError("--collapse erwartet I,ANTEIL");
            }
            a.collapse = std::make_pair(parse_int(arg, p[0]), parse_double(arg, p[1]));
        } else if (arg == "--replace-glass") {
            const std::vector<std::string_view> p = split_commas(next_value(args, i, arg));
            if (p.size() != 2) {
                throw UsageError("--replace-glass erwartet I,GLAS");
            }
            a.replace_glass = std::make_pair(parse_int(arg, p[0]), parse_glass(arg, p[1]));
        } else if (arg == "--ghost-dictionary") {
            const std::vector<std::string_view> p = split_commas(next_value(args, i, arg));
            if (p.size() != 2) {
                throw UsageError("--ghost-dictionary erwartet I,DIST");
            }
            GhostDictionaryPlacement g;
            g.after_surface = parse_int(arg, p[0]);
            g.distance = parse_double(arg, p[1]);
            a.ghost = g;
        } else if (arg == "--ghost-glass") {
            a.ghost_glasses.push_back(parse_glass(arg, next_value(args, i, arg)));
        } else if (arg == "--ghost-order") {
            a.ghost_order = parse_int(arg, next_value(args, i, arg));
        } else if (arg == "--ghost-weight") {
            a.ghost_weight = parse_double(arg, next_value(args, i, arg));
        } else if (arg == "--norm-radius") {
            a.norm_radius = parse_double(arg, next_value(args, i, arg));
        } else if (arg == "--aplanatic-shell") {
            const std::vector<std::string_view> p = split_commas(next_value(args, i, arg));
            if (p.size() != 4) {
                throw UsageError("--aplanatic-shell erwartet I,POS,T,GLAS");
            }
            AplanaticShell sh;
            sh.after_surface = parse_int(arg, p[0]);
            sh.position = parse_double(arg, p[1]);
            sh.thickness = parse_double(arg, p[2]);
            sh.glass = parse_glass(arg, p[3]);
            a.shell = sh;
        } else if (arg == "--no-refocus") {
            a.refocus = false;
        } else if (arg == "--null-doublet") {
            const std::vector<std::string_view> p = split_commas(next_value(args, i, arg));
            if (p.size() != 5) {
                throw UsageError("--null-doublet erwartet I,POS,C0,GLAS1,GLAS2");
            }
            NullDoublet d;
            d.after_surface = parse_int(arg, p[0]);
            d.position = parse_double(arg, p[1]);
            d.curvature = parse_double(arg, p[2]);
            d.front_glass = parse_glass(arg, p[3]);
            d.back_glass = parse_glass(arg, p[4]);
            a.doublet = d;
        } else if (arg == "--json") {
            a.json = true;
        } else if (arg == kNoReferenceWarningOption) {
            a.no_reference_warning = true;
        } else if (arg == "--thickness") {
            a.thickness = parse_double(arg, next_value(args, i, arg));
        } else if (arg == "--free-glass") {
            a.free_glass = true;
        } else if (arg == "--start-glass") {
            a.start_glass = parse_glass(arg, next_value(args, i, arg));
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
    const int ops = int(a.meniscus.has_value()) + int(a.split.has_value()) +
                    int(a.remove.has_value()) + int(a.collapse.has_value()) +
                    int(a.replace_glass.has_value()) + int(a.ghost.has_value()) +
                    int(a.shell.has_value()) + int(a.doublet.has_value());
    if (ops != 1) {
        throw UsageError("genau einer von --meniscus, --split, --remove, --collapse, "
                         "--replace-glass, --ghost-dictionary, --aplanatic-shell und "
                         "--null-doublet ist nötig");
    }
    a.op = a.meniscus        ? Op::Meniscus
           : a.split         ? Op::Split
           : a.remove        ? Op::Remove
           : a.collapse      ? Op::Collapse
           : a.replace_glass ? Op::ReplaceGlass
           : a.ghost         ? Op::GhostDictionary
           : a.shell         ? Op::AplanaticShell
                             : Op::NullDoublet;
    if (a.output.empty()) {
        throw UsageError("Ergebnisdatei fehlt (-o <datei>)");
    }
    if (a.free_glass && !a.split) {
        throw UsageError("--free-glass nur mit --split");
    }
    if (a.start_glass && !a.free_glass) {
        throw UsageError("--start-glass nur mit --free-glass");
    }
    if (a.thickness != 0.0 && !a.meniscus) {
        throw UsageError("--thickness nur mit --meniscus");
    }
    if ((!a.ghost_glasses.empty() || a.ghost_order || a.ghost_weight || a.norm_radius) &&
        !a.ghost) {
        throw UsageError("--ghost-glass, --ghost-order, --ghost-weight und --norm-radius nur mit "
                         "--ghost-dictionary");
    }
    if (!a.refocus && !a.shell) {
        throw UsageError("--no-refocus nur mit --aplanatic-shell");
    }
    if (a.meniscus) {
        a.meniscus->thickness = a.thickness;
        a.meniscus->curvature_variables.enabled = a.variables;
    } else if (a.split) {
        a.split->curvature_variable.enabled = a.variables;
    } else if (a.ghost) {
        if (a.ghost_glasses.empty()) {
            throw UsageError("--ghost-dictionary braucht mindestens ein --ghost-glass");
        }
        a.ghost->glasses = a.ghost_glasses;
        a.ghost->order = a.ghost_order.value_or(6);
        a.ghost->penalty_weight = a.ghost_weight;
        a.ghost->norm_radius = a.norm_radius.value_or(0.0);
    } else if (a.shell) {
        a.shell->refocus = a.refocus;
        a.shell->curvature_variables.enabled = a.variables;
    } else if (a.doublet) {
        a.doublet->curvature_variables.enabled = a.variables;
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
    out << "Aufruf: mikroptim insert <system.json> -o <neu.json> <Operator> [--no-variables] "
           "[--json]\n"
           "Einfüge-Operatoren der Systemsynthese (docs/optimierung.md, Abschnitt 6) und\n"
           "Operatoren der Entwurfssteuerung (Abschnitt 13). Genau ein Operator:\n"
           "  --meniscus I,POS,C0,GLAS  Operator A: zwei Flächen der Krümmung C0 [1/mm] im\n"
           "                            Luftspalt nach Fläche I, Abstand POS [mm] vom Scheitel\n"
           "                            von I; GLAS ist ein Katalogname oder n=<Brechzahl>\n"
           "  --thickness T             Mittendicke des Meniskus [mm] (Standard 0: Merit\n"
           "                            unverändert)\n"
           "  --split I,C_S[,ANTEIL]    Operator B: Linse aus Fläche I und I+1 an einer\n"
           "                            Kittfläche der Krümmung C_S teilen; ANTEIL der\n"
           "                            Mittendicke vor der Kittfläche (Standard 0,5)\n"
           "  --free-glass              Glas des hinteren Teils als Variable freigeben\n"
           "  --start-glass GLAS        das freigegebene Glas startet bei GLAS statt beim Glas\n"
           "                            der Linse (Katalogname oder n=<Brechzahl>; M11-T5)\n"
           "  --remove I                Glasblock (Linse oder Kittglied) ab Fläche I entfernen,\n"
           "                            Spalte vereinigen (Merit gleich bei einem Null-Element)\n"
           "  --collapse I,ANTEIL       Stufe des Rückbaus: Glasdicken und Krümmungsunterschiede\n"
           "                            des Blocks ab I um ANTEIL (0 < ANTEIL <= 1) gegen 0,\n"
           "                            seine Variablen werden festgesetzt\n"
           "  --replace-glass I,GLAS    Glas nach Fläche I tauschen (Glasvariable festsetzen)\n"
           "  --ghost-dictionary I,DIST Geisterplatten mit Koeffizienten 0 im Spalt nach I im\n"
           "                            Abstand DIST einfügen (Merit unverändert), dazu:\n"
           "    --ghost-glass GLAS      Glas einer Platte (mehrfach, mindestens einmal)\n"
           "    --ghost-order K         Ordnungen (Standard 6)\n"
           "    --ghost-weight MU       eigener Eintrag ghost_penalty mit Gewicht MU\n"
           "    --norm-radius R         Normierungsradius [mm] (Standard: größerer\n"
           "                            Halbdurchmesser der Spaltflächen)\n"
           "  --aplanatic-shell I,POS,T,GLAS\n"
           "                            aplanatisch-konzentrische Schale der Dicke T im Spalt\n"
           "                            nach I (auch nach der letzten Fläche) aus dem realen\n"
           "                            Randstrahl\n"
           "    --no-refocus            Arbeitsabstand nicht nachfokussieren\n"
           "  --null-doublet I,POS,C0,GLAS1,GLAS2\n"
           "                            Kittglied ohne Wirkung (drei Flächen C0, Dicke 0)\n"
           "  --no-variables            keine Krümmungsvariablen für die neuen Flächen\n"
           "  --json                    Bericht als JSON (mikroptim.insert/1)\n"
           "  --no-reference-warning    keine Warnung auf stderr, wenn die Datei kein Feld\n"
           "                            reference_sphere hat (dann gilt last_surface)\n"
           "  -o, --output <datei>      neues System (Pflicht)\n";
}

namespace {

const char* op_name(Op op) {
    switch (op) {
    case Op::Meniscus:
        return "meniscus";
    case Op::Split:
        return "split";
    case Op::Remove:
        return "remove";
    case Op::Collapse:
        return "collapse";
    case Op::ReplaceGlass:
        return "replace_glass";
    case Op::GhostDictionary:
        return "ghost_dictionary";
    case Op::AplanaticShell:
        return "aplanatic_shell";
    case Op::NullDoublet:
        return "null_doublet";
    }
    return "?";
}

const char* op_title(Op op) {
    switch (op) {
    case Op::Meniscus:
        return "Operator A (Meniskus)";
    case Op::Split:
        return "Operator B (Kittglied)";
    case Op::Remove:
        return "Element entfernen";
    case Op::Collapse:
        return "Rückbau-Stufe";
    case Op::ReplaceGlass:
        return "Glas tauschen";
    case Op::GhostDictionary:
        return "Geisterplatten-Wörterbuch";
    case Op::AplanaticShell:
        return "aplanatische Schale";
    case Op::NullDoublet:
        return "Null-Kittglied";
    }
    return "?";
}

Json json_number(double v) {
    return std::isfinite(v) ? Json(v) : Json(nullptr);
}

Json int_list(const std::vector<int>& v) {
    Json a = Json::array();
    for (int x : v) {
        a.push_back(x);
    }
    return a;
}

} // namespace

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
        warn_missing_reference_sphere(err, "insert", before.system, a.no_reference_warning);
        SystemDescription result;
        // Eingefügte bzw. entfernte Flächen für den Bericht
        int first_new = -1;
        int new_count = 0;
        Json details = Json::object();
        // Warnungen der Operatoren (angehobene Schranken beim Rückbau)
        std::vector<std::string> warnings;
        switch (a.op) {
        case Op::Meniscus:
        case Op::Split: {
            InsertionResult r;
            if (a.meniscus) {
                r = insert_meniscus(before.system, *a.meniscus);
            } else {
                r = split_lens(before.system, *a.split);
                if (a.free_glass) {
                    free_glass(r.system, r.first_new_surface, a.start_glass);
                }
            }
            first_new = r.first_new_surface;
            new_count = r.new_surfaces;
            result = std::move(r.system);
            break;
        }
        case Op::Remove: {
            const bool null_element = is_null_element(before.system, *a.remove);
            RemovalResult r = remove_element(before.system, *a.remove);
            details["removed_first"] = r.first_removed;
            details["removed_surfaces"] = r.removed_surfaces;
            details["merged_gap"] = r.merged_gap;
            details["null_element"] = null_element;
            for (const BoundWidening& w : r.widened) {
                warnings.push_back(bound_widening_warning(w));
            }
            result = std::move(r.system);
            break;
        }
        case Op::Collapse: {
            std::vector<BoundWidening> widened;
            result =
                collapse_element(before.system, a.collapse->first, a.collapse->second, &widened);
            for (const BoundWidening& w : widened) {
                warnings.push_back(bound_widening_warning(w));
            }
            details["first"] = a.collapse->first;
            details["last"] = glass_block_end(before.system, a.collapse->first);
            details["fraction"] = a.collapse->second;
            details["null_element"] = is_null_element(result, a.collapse->first);
            break;
        }
        case Op::ReplaceGlass: {
            result = before.system;
            replace_glass(result, a.replace_glass->first, a.replace_glass->second);
            details["surface"] = a.replace_glass->first;
            break;
        }
        case Op::GhostDictionary: {
            GhostDictionaryResult r = place_ghost_dictionary(before.system, *a.ghost);
            details["new_plates"] = int_list(r.new_plates);
            details["plate_new_of_old"] = int_list(r.plate_new_of_old);
            result = std::move(r.system);
            break;
        }
        case Op::AplanaticShell: {
            AplanaticShellResult r =
                insert_aplanatic_shell(before.system, before.catalog, *a.shell);
            first_new = r.insertion.first_new_surface;
            new_count = r.insertion.new_surfaces;
            details["n_glass"] = json_number(r.n_glass);
            details["intercept"] = json_number(r.intercept);
            details["image_intercept"] = json_number(r.image_intercept);
            details["c_aplanatic"] = json_number(r.c_aplanatic);
            details["c_concentric"] = json_number(r.c_concentric);
            details["working_distance"] = json_number(r.working_distance);
            result = std::move(r.insertion.system);
            break;
        }
        case Op::NullDoublet: {
            InsertionResult r = insert_null_doublet(before.system, *a.doublet);
            first_new = r.first_new_surface;
            new_count = r.new_surfaces;
            result = std::move(r.system);
            break;
        }
        }
        const fs::path input_dir = fs::path(a.file).parent_path();
        const LoadedSystem after = prepare_system(result, input_dir);
        const double f_before = cpu_merit(before);
        const double f_after = cpu_merit(after);
        const double rel = std::abs(f_after - f_before) / std::abs(f_before);
        const double fl_before = compute_paraxial(before.system, before.catalog).focal_length;
        const double fl_after = compute_paraxial(after.system, after.catalog).focal_length;

        const fs::path output(a.output);
        std::error_code ec;
        fs::path out_dir = fs::absolute(output, ec).parent_path();
        if (ec) {
            out_dir = output.parent_path();
        }
        SystemDescription file_system = result;
        file_system.glass_catalogs = catalogs_for(result, after.catalog_files, out_dir);
        write_system_file(file_system, output);

        if (a.json) {
            Json j;
            j["format"] = "mikroptim.insert/1";
            j["operator"] = op_name(a.op);
            j["input"] = a.file;
            j["output"] = a.output;
            j["surfaces_before"] = before.system.surfaces.size();
            j["surfaces_after"] = result.surfaces.size();
            j["first_new_surface"] = first_new >= 0 ? Json(first_new) : Json(nullptr);
            j["new_surfaces"] = new_count;
            j["ghost_plates_before"] = before.system.ghost_plates.size();
            j["ghost_plates_after"] = result.ghost_plates.size();
            j["merit_before"] = json_number(f_before);
            j["merit_after"] = json_number(f_after);
            j["relative_change"] = json_number(rel);
            j["focal_length_before"] = json_number(fl_before);
            j["focal_length_after"] = json_number(fl_after);
            j["details"] = details;
            j["warnings"] = warnings;
            out << j.dump(2) << '\n';
        } else {
            out << "mikroptim insert: " << op_title(a.op) << '\n';
            if (first_new >= 0) {
                kv(out, "Eingefügte Flächen",
                   std::to_string(first_new) +
                       (new_count > 1 ? " bis " + std::to_string(first_new + new_count - 1)
                                      : std::string()));
            }
            kv(out, "Flächen vorher / nachher",
               std::to_string(before.system.surfaces.size()) + " / " +
                   std::to_string(result.surfaces.size()));
            if (a.op == Op::GhostDictionary) {
                kv(out, "Geisterplatten vorher / nachher",
                   std::to_string(before.system.ghost_plates.size()) + " / " +
                       std::to_string(result.ghost_plates.size()));
            }
            kv(out, "Merit vorher [Wellen²]", sci(f_before, 15));
            kv(out, "Merit nachher [Wellen²]", sci(f_after, 15));
            kv(out, "relative Änderung", sci(rel, 2));
            kv(out, "Brennweite vorher / nachher [mm]",
               fixed(fl_before, 9) + " / " + fixed(fl_after, 9));
            for (const std::string& w : warnings) {
                out << "Warnung: " << w << '\n';
            }
            out << "Ergebnis geschrieben: " << a.output << '\n';
        }
        for (const std::string& w : warnings) {
            err << "mikroptim insert: Warnung: " << w << '\n';
        }
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
