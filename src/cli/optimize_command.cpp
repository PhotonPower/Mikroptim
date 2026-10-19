// Unterbefehl `mikroptim optimize`: Laden, Variablen und Nebenbedingungen, gebündelter
// Levenberg-Marquardt (CPU oder mit --gpu über die TraceEngine), Multistart, Ergebnisdatei,
// Iterationsprotokoll und Zusammenfassung als Text oder JSON. Formate: docs/cli.md.

#include "cli/optimize_command.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/merit/wavefront.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/model/format_error.hpp"
#include "mikroptim/optim/ghost_diagnosis.hpp"
#include "mikroptim/optim/ghost_penalty.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "mikroptim/optim/variant_generator.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/version.hpp"

#ifdef MIKROPTIM_HAS_CUDA
#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/levenberg_marquardt.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#endif

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace mko::cli {

namespace {

namespace fs = std::filesystem;
using Json = nlohmann::ordered_json;

constexpr int kDefaultRings = 10;
constexpr int kDefaultSpokes = 8;
constexpr int kDefaultMaxIterations = 100; // LmOptions::max_iterations
constexpr double kDefaultPerturbation = 0.05;
// Einheit der Merit (docs/optimierung.md, Abschnitt 2: Strahlresiduen in Wellen bei λ_ref,
// Zusatzresiduen über ihre Gewichte dimensionslos).
constexpr std::string_view kMeritUnit = "waves^2";

struct OptimizeArgs {
    std::string file;
    std::string output; // leer: keine Ergebnisdatei
    bool json = false;
    bool quiet = false;
    bool help = false;
    bool gpu = false;
    bool have_device = false;
    int device = 0;
    int rings = kDefaultRings;
    int spokes = kDefaultSpokes;
    int max_iterations = kDefaultMaxIterations;
    int runs = 1;
    bool have_perturbation = false;
    double perturbation = kDefaultPerturbation;
    bool have_seed = false;
    std::uint64_t seed = 0;
};

std::string_view next_value(std::span<const std::string_view> args, std::size_t& i,
                            std::string_view option) {
    if (i + 1 >= args.size()) {
        throw UsageError(std::string(option) + " erwartet einen Wert");
    }
    return args[++i];
}

OptimizeArgs parse_args(std::span<const std::string_view> args) {
    OptimizeArgs a;
    bool have_file = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--help" || arg == "-h") {
            a.help = true;
        } else if (arg == "--json") {
            a.json = true;
        } else if (arg == "--quiet" || arg == "-q") {
            a.quiet = true;
        } else if (arg == "--gpu") {
            a.gpu = true;
        } else if (arg == "-o" || arg == "--output") {
            a.output = std::string(next_value(args, i, arg));
            if (a.output.empty()) {
                throw UsageError(std::string(arg) + " erwartet einen Dateinamen");
            }
        } else if (arg == "--device") {
            a.device = parse_int(arg, next_value(args, i, arg));
            if (a.device < 0) {
                throw UsageError("--device muss >= 0 sein");
            }
            a.have_device = true;
        } else if (arg == "--rings") {
            a.rings = parse_int(arg, next_value(args, i, arg));
            if (a.rings < 1 || a.rings > kMaxPupilRings) {
                throw UsageError("--rings muss in [1, " + std::to_string(kMaxPupilRings) +
                                 "] liegen");
            }
        } else if (arg == "--spokes") {
            a.spokes = parse_int(arg, next_value(args, i, arg));
            if (a.spokes < 1) {
                throw UsageError("--spokes muss >= 1 sein");
            }
        } else if (arg == "--max-iterations") {
            a.max_iterations = parse_int(arg, next_value(args, i, arg));
            if (a.max_iterations < 0) {
                throw UsageError("--max-iterations muss >= 0 sein");
            }
        } else if (arg == "--runs") {
            a.runs = parse_int(arg, next_value(args, i, arg));
            if (a.runs < 1) {
                throw UsageError("--runs muss >= 1 sein");
            }
        } else if (arg == "--perturbation") {
            a.perturbation = parse_double(arg, next_value(args, i, arg));
            if (a.perturbation < 0.0) {
                throw UsageError("--perturbation muss >= 0 sein");
            }
            a.have_perturbation = true;
        } else if (arg == "--seed") {
            a.seed = parse_uint64(arg, next_value(args, i, arg));
            a.have_seed = true;
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
    if ((a.have_perturbation || a.have_seed) && a.runs < 2) {
        throw UsageError("--perturbation und --seed verlangen --runs K mit K >= 2");
    }
    return a;
}

// Dieselbe Datei? Existieren beide, entscheidet das Dateisystem (auch Verknüpfungen), sonst
// der Vergleich der normalisierten absoluten Pfade.
bool same_file(const fs::path& a, const fs::path& b) {
    std::error_code ec;
    if (fs::exists(a, ec) && fs::exists(b, ec)) {
        const bool same = fs::equivalent(a, b, ec);
        if (!ec) {
            return same;
        }
    }
    const fs::path ca = fs::weakly_canonical(a, ec);
    const fs::path cb = fs::weakly_canonical(b, ec);
    return ca == cb;
}

// UTF-8-Text eines Pfads (Dateiformat speichert Pfade als UTF-8).
std::string utf8(const fs::path& p) {
    const std::u8string s = p.generic_u8string();
    return std::string(s.begin(), s.end());
}

// Einträge von glass_catalogs für eine Systemdatei im Verzeichnis `out_dir`: Ein Eintrag bleibt
// unverändert, wenn er von dort aus (resolve_glass_catalogs, mit der Suche in den
// Elternverzeichnissen) auf dieselbe Datei führt wie von der Eingabedatei aus; sonst wird er
// durch den relativen Pfad von `out_dir` zur aufgelösten Datei ersetzt, und wenn es keinen gibt
// (anderes Laufwerk), durch den absoluten Pfad.
std::vector<std::string> catalogs_for(const SystemDescription& system,
                                      const std::vector<fs::path>& resolved,
                                      const fs::path& out_dir) {
    std::vector<std::string> out;
    for (std::size_t i = 0; i < system.glass_catalogs.size(); ++i) {
        const std::string& entry = system.glass_catalogs[i];
        const fs::path& target = resolved.at(i);
        SystemDescription probe;
        probe.glass_catalogs = {entry};
        bool keep = false;
        try {
            const std::vector<fs::path> r = resolve_glass_catalogs(probe, out_dir);
            std::error_code ec;
            keep = fs::equivalent(r.at(0), target, ec) && !ec;
        } catch (const FormatError&) {
            keep = false;
        }
        if (keep) {
            out.push_back(entry);
            continue;
        }
        std::error_code ec;
        const fs::path abs_target = fs::absolute(target, ec).lexically_normal();
        const fs::path rel = fs::relative(abs_target, out_dir, ec);
        out.push_back(utf8(!ec && !rel.empty() ? rel : abs_target));
    }
    return out;
}

// ------------------------------------------------------------------------------------------
// Bericht
// ------------------------------------------------------------------------------------------

// Ein Zyklus eines Laufs (aus LmIterationInfo, ohne x).
struct IterationRecord {
    int run = 0;
    int iteration = 0;
    LmStatus status = LmStatus::Running;
    bool evaluated = false;
    bool accepted = false;
    double merit = 0.0;
    double trial_merit = 0.0;
    double gain_ratio = 0.0;
    double damping = 0.0;
    double step_norm = 0.0;
    std::int64_t failed_rays = 0;
};

// Eine paraxiale Nebenbedingung vorher und nachher. `value` ist die Istgröße (Brennweite
// 1/φ, Arbeitsabstand, Abgleichlänge, bei paraxial_focus die Randstrahlhöhe q in der
// Bildebene), `residual` das Residuum √w·q̃ aus paraxial_residual.
struct ConstraintReport {
    ParaxialConstraint constraint;
    double value_before = 0.0;
    double value_after = 0.0;
    double residual_before = 0.0;
    double residual_after = 0.0;
};

struct WavefrontReport {
    std::optional<WavefrontResult> before;
    std::optional<WavefrontResult> after;
    std::string error; // Grund, wenn before oder after fehlt
};

struct BackendInfo {
    bool gpu = false;
    int device = -1;
    std::string device_name;
};

struct OptimizeReport {
    const OptimizeArgs* args = nullptr;
    LoadedSystem loaded;
    BackendInfo backend;
    VariableSet set;
    ParaxialConstraints constraints;
    GhostPenalty penalty; // L1-Strafe der Geisterkoeffizienten (ghost_penalty)
    std::vector<double> x0;
    std::vector<double> starts; // Startpunkte aller Läufe hintereinander
    LmResult result;
    std::int64_t start_failed = 0; // versagte Strahlen bei x₀
    int best = 0;
    std::vector<IterationRecord> iterations;
    std::vector<ConstraintReport> constraint_reports;
    WavefrontReport wavefront;
    std::optional<GhostDiagnosis> ghost; // nur mit Geisterplatten
    std::vector<std::string> warnings;   // Lader und Befehl
    bool focal_from_start = false;       // Feldziele aus der paraxialen Brennweite des Starts

    const LmRunResult& best_run() const { return result.runs[static_cast<std::size_t>(best)]; }
    // F am Startpunkt der Datei (Lauf 0 startet immer dort).
    double start_merit() const { return result.runs.front().history.front(); }
};

// Istgröße einer Bedingung für Variante 0 von `batch` (p = batch_paraxial(batch, 0)).
double constraint_value(const ParaxialConstraint& c, const SystemBatchHost& batch,
                        const ParaxialData& p) {
    switch (c.type) {
    case ConstraintType::FocalLength:
        return p.focal_length;
    case ConstraintType::WorkingDistance:
        return batch.image_distance(0);
    case ConstraintType::ParfocalLength: {
        double s = c.offset; // Reihenfolge wie paraxial_residual: offset, dann j aufsteigend
        for (int j = c.reference_surface; j < batch.surfaces(); ++j) {
            s += batch.thickness(0, j);
        }
        return s;
    }
    case ConstraintType::ParaxialFocus: {
        ParaxialConstraint raw = c;
        raw.weight = 1.0;
        raw.mode = ConstraintMode::Equal;
        return paraxial_residual(raw, batch, 0, p);
    }
    default:
        break;
    }
    return std::nan("");
}

std::vector<ConstraintReport> constraint_reports(const ParaxialConstraints& constraints,
                                                 const SystemBatchHost& before,
                                                 const SystemBatchHost& after) {
    std::vector<ConstraintReport> out;
    if (constraints.items.empty()) {
        return out;
    }
    const ParaxialData pb = batch_paraxial(before, 0);
    const ParaxialData pa = batch_paraxial(after, 0);
    for (const ParaxialConstraint& c : constraints.items) {
        ConstraintReport r;
        r.constraint = c;
        r.value_before = constraint_value(c, before, pb);
        r.value_after = constraint_value(c, after, pa);
        r.residual_before = paraxial_residual(c, before, 0, pb);
        r.residual_after = paraxial_residual(c, after, 0, pa);
        out.push_back(r);
    }
    return out;
}

// Bester Lauf: endliche Merit, dann die wenigsten versagten Strahlen (hart versagte Strahlen
// haben r = 0, eine Merit mit mehr versagten Strahlen ist nicht vergleichbar), dann die
// kleinste Merit, bei Gleichstand der kleinste Index. −1, wenn kein Lauf eine endliche Merit hat.
int select_best(const LmResult& result) {
    int best = -1;
    for (std::size_t i = 0; i < result.runs.size(); ++i) {
        const LmRunResult& r = result.runs[i];
        if (!std::isfinite(r.merit)) {
            continue;
        }
        if (best < 0) {
            best = static_cast<int>(i);
            continue;
        }
        const LmRunResult& b = result.runs[static_cast<std::size_t>(best)];
        if (r.failed_rays < b.failed_rays ||
            (r.failed_rays == b.failed_rays && r.merit < b.merit)) {
            best = static_cast<int>(i);
        }
    }
    return best;
}

const char* step_name(const IterationRecord& r) {
    if (!r.evaluated) {
        return "gescheitert";
    }
    return r.accepted ? "angenommen" : "verworfen";
}

const char* unit_of(VariableType type) {
    return type == VariableType::Curvature ? "1/mm" : "mm";
}

// Variablenwert für die Textausgabe: Krümmungen und Geisterkoeffizienten mit 9 Stellen in
// Exponentialschreibweise, Längen mit 6 Nachkommastellen.
std::string variable_text(VariableType type, double v) {
    return type == VariableType::Curvature || type == VariableType::GhostCoefficient ? sci(v, 9)
                                                                                     : fixed(v, 6);
}

// ------------------------------------------------------------------------------------------
// Text
// ------------------------------------------------------------------------------------------

void write_text_header(const OptimizeReport& r, std::ostream& out) {
    const OptimizeArgs& a = *r.args;
    const SystemDescription& sys = r.loaded.system;
    out << "Optimierung: " << sys.name << '\n';
    kv(out, "Datei", a.file);
    kv(out, "Ergebnisdatei", a.output.empty() ? std::string("- (ohne -o)") : a.output);
    if (r.backend.gpu) {
        kv(out, "Backend",
           "GPU " + std::to_string(r.backend.device) + " (" + r.backend.device_name + "), FP64");
    } else {
        kv(out, "Backend", "CPU, FP64");
    }
    kv(out, "Quadratur",
       std::to_string(a.rings) + " Ringe x " + std::to_string(a.spokes) + " Speichen");
    kv(out, "Variablen", std::to_string(r.set.size()));
    kv(out, "Nebenbedingungen (paraxial)", std::to_string(r.constraints.size()));
    if (!sys.ghost_plates.empty()) {
        kv(out, "Geisterplatten",
           std::to_string(sys.ghost_plates.size()) +
               " (L1-Strafe: " + std::to_string(r.penalty.size()) + " Koeffizienten)");
    }
    std::string runs = std::to_string(a.runs);
    if (a.runs > 1) {
        runs += " (Lauf 0 ungestört, Störung " + fixed(a.perturbation, 6) + " relativ, Startwert " +
                std::to_string(a.seed) + ")";
    }
    kv(out, "Läufe", runs);
    kv(out, "Zyklen höchstens", std::to_string(a.max_iterations));
}

void write_iteration_header(std::ostream& out, double reference_nm) {
    out << '\n'
        << "Iterationsprotokoll (Merit F = Summe r² in Wellen² bei " << fixed(reference_nm, 3)
        << " nm, docs/optimierung.md)\n";
    out << "  " << pad_left("Lauf", 4) << pad_left("Zyklus", 8) << pad_left("Schritt", 12)
        << pad_left("Merit", 17) << pad_left("Testmerit", 17) << pad_left("Gewinn", 11)
        << pad_left("Dämpfung", 11) << pad_left("Schrittnorm", 12) << pad_left("versagt", 8)
        << "  Status\n";
}

void write_iteration_line(const IterationRecord& r, std::ostream& out) {
    out << "  " << pad_left(std::to_string(r.run), 4) << pad_left(std::to_string(r.iteration), 8)
        << pad_left(step_name(r), 12) << pad_left(sci(r.merit, 9), 17)
        << pad_left(sci(r.trial_merit, 9), 17) << pad_left(sci(r.gain_ratio, 3), 11)
        << pad_left(sci(r.damping, 3), 11) << pad_left(sci(r.step_norm, 3), 12)
        << pad_left(std::to_string(r.failed_rays), 8);
    if (r.status != LmStatus::Running) {
        out << "  " << to_string(r.status);
    }
    out << '\n';
}

void write_text_summary(const OptimizeReport& r, std::ostream& out) {
    const OptimizeArgs& a = *r.args;
    const SystemDescription& sys = r.loaded.system;
    const LmRunResult& b = r.best_run();
    const double f0 = r.start_merit();

    out << '\n' << "Ergebnis\n";
    kv(out, "Status", to_string(b.status));
    if (a.runs > 1) {
        kv(out, "Bester Lauf", std::to_string(r.best));
    }
    kv(out, "Zyklen", std::to_string(b.iterations));
    kv(out, "Angenommene Schritte", std::to_string(b.accepted));
    kv(out, "Jacobi-Matrizen", std::to_string(b.jacobians));
    kv(out, "Merit vorher [Wellen²]", sci(f0, 9));
    kv(out, "Merit nachher [Wellen²]", sci(b.merit, 9));
    kv(out, "Merit nachher/vorher", sci(f0 > 0.0 ? b.merit / f0 : std::nan(""), 6));
    kv(out, "Versagte Strahlen vorher/nachher",
       std::to_string(r.start_failed) + " / " + std::to_string(b.failed_rays));

    if (a.runs > 1) {
        out << '\n' << "Läufe\n";
        out << "  " << pad_left("Lauf", 4) << "  " << pad_right("Status", 20)
            << pad_left("Zyklen", 7) << pad_left("Merit Start", 17) << pad_left("Merit Ende", 17)
            << pad_left("versagt", 9) << '\n';
        for (std::size_t i = 0; i < r.result.runs.size(); ++i) {
            const LmRunResult& run = r.result.runs[i];
            out << "  " << pad_left(std::to_string(i), 4) << "  "
                << pad_right(to_string(run.status), 20)
                << pad_left(std::to_string(run.iterations), 7)
                << pad_left(sci(run.history.front(), 9), 17) << pad_left(sci(run.merit, 9), 17)
                << pad_left(std::to_string(run.failed_rays), 9)
                << (static_cast<int>(i) == r.best ? "  bester" : "") << '\n';
        }
    }

    out << '\n' << "Variablen\n";
    out << "  " << pad_right("Name", 8) << pad_right("Einheit", 8)
        << pad_left("untere Schranke", 17) << pad_left("vorher", 17) << pad_left("nachher", 17)
        << pad_left("obere Schranke", 17) << '\n';
    for (int k = 0; k < r.set.size(); ++k) {
        const Variable& v = r.set[k];
        const auto kk = static_cast<std::size_t>(k);
        const double x = b.x[kk];
        std::string flag;
        if (x == v.lower) {
            flag = "  auf unterer Schranke";
        } else if (x == v.upper) {
            flag = "  auf oberer Schranke";
        }
        out << "  " << pad_right(v.name, 8) << pad_right(unit_of(v.type), 8)
            << pad_left(variable_text(v.type, v.lower), 17)
            << pad_left(variable_text(v.type, r.x0[kk]), 17)
            << pad_left(variable_text(v.type, x), 17)
            << pad_left(variable_text(v.type, v.upper), 17) << flag << '\n';
    }

    if (!r.constraint_reports.empty()) {
        out << '\n' << "Nebenbedingungen (Istwerte in mm, Residuum r = Wurzel(w) q)\n";
        out << "  " << pad_left("Nr", 3) << "  " << pad_right("Typ", 17) << pad_right("Modus", 6)
            << pad_left("Gewicht", 11) << pad_left("Soll", 13) << pad_left("Ist vorher", 13)
            << pad_left("Ist nachher", 13) << pad_left("r vorher", 14) << pad_left("r nachher", 14)
            << '\n';
        for (std::size_t i = 0; i < r.constraint_reports.size(); ++i) {
            const ConstraintReport& c = r.constraint_reports[i];
            const int nr = c.constraint.item >= 0 ? c.constraint.item : static_cast<int>(i);
            out << "  " << pad_left(std::to_string(nr), 3) << "  "
                << pad_right(to_string(c.constraint.type), 17)
                << pad_right(to_string(c.constraint.mode), 6)
                << pad_left(sci(c.constraint.weight, 3), 11)
                << pad_left(fixed(c.constraint.target, 6), 13)
                << pad_left(fixed(c.value_before, 6), 13) << pad_left(fixed(c.value_after, 6), 13)
                << pad_left(sci(c.residual_before, 6), 14) << pad_left(sci(c.residual_after, 6), 14)
                << '\n';
        }
    }

    if (r.ghost) {
        out << '\n';
        write_ghost_diagnosis_text(*r.ghost, out);
    }

    out << '\n';
    if (!r.wavefront.before || !r.wavefront.after) {
        out << "Wellenfront nicht auswertbar\n  " << r.wavefront.error << '\n';
        return;
    }
    const WavefrontResult& wb = *r.wavefront.before;
    const WavefrontResult& wa = *r.wavefront.after;
    out << "Wellenfront (Quadratur " << a.rings << " Ringe x " << a.spokes
        << " Speichen, CPU, FP64, RMS in Wellen bei " << fixed(sys.spectrum.reference_nm, 3)
        << " nm)\n";
    out << "  " << pad_left("Feld", 4) << pad_left("Winkel [deg]", 14)
        << pad_left("Wellenlänge [nm]", 18) << pad_left("Strahlen", 10)
        << pad_left("RMS vorher", 14) << pad_left("RMS nachher", 14)
        << pad_left("versagt vorher/nachher", 24) << '\n';
    for (std::size_t i = 0; i < wb.maps.size() && i < wa.maps.size(); ++i) {
        const WavefrontMap& mb = wb.maps[i];
        const WavefrontMap& ma = wa.maps[i];
        const std::string angle =
            sys.source.type == SourceType::Collimated
                ? fixed(sys.source.field_angles_deg[static_cast<std::size_t>(mb.field)], 3)
                : std::string("-");
        out << "  " << pad_left(std::to_string(mb.field), 4) << pad_left(angle, 14)
            << pad_left(fixed(mb.wavelength_nm, 3), 18)
            << pad_left(std::to_string(mb.samples()), 10) << pad_left(fixed(mb.rms_waves), 14)
            << pad_left(fixed(ma.rms_waves), 14)
            << pad_left(std::to_string(mb.failed) + " / " + std::to_string(ma.failed), 24) << '\n';
    }
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

Json numbers(std::span<const double> v) {
    Json a = Json::array();
    for (const double x : v) {
        a.push_back(number(x));
    }
    return a;
}

Json wavefront_json(const SystemDescription& sys, const WavefrontResult& w) {
    Json entries = Json::array();
    for (const WavefrontMap& m : w.maps) {
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
        e["rays"] = m.samples();
        e["failed_rays"] = m.failed;
        e["rms_waves"] = number(m.rms_waves);
        e["rms_mm"] = number(m.rms);
        entries.push_back(std::move(e));
    }
    return entries;
}

void write_json(const OptimizeReport& r, std::ostream& out) {
    const OptimizeArgs& a = *r.args;
    const SystemDescription& sys = r.loaded.system;
    const LmRunResult& b = r.best_run();
    Json j;
    j["format"] = std::string(kOptimizeFormat);
    j["mikroptim_version"] = kVersion;
    j["system"] = sys.name;
    j["file"] = a.file;
    j["output"] = a.output.empty() ? Json(nullptr) : Json(a.output);
    j["backend"] = r.backend.gpu ? "gpu" : "cpu";
    if (r.backend.gpu) {
        j["device"] = Json{{"index", r.backend.device}, {"name", r.backend.device_name}};
    } else {
        j["device"] = nullptr;
    }
    j["precision"] = "fp64";
    j["merit_unit"] = std::string(kMeritUnit);
    j["quadrature"] = Json{{"rings", a.rings}, {"spokes", a.spokes}};
    Json settings;
    settings["max_iterations"] = a.max_iterations;
    settings["runs"] = a.runs;
    settings["perturbation"] = a.runs > 1 ? number(a.perturbation) : Json(nullptr);
    settings["seed"] = a.runs > 1 ? Json(a.seed) : Json(nullptr);
    j["settings"] = std::move(settings);

    Json vars = Json::array();
    for (int k = 0; k < r.set.size(); ++k) {
        const Variable& v = r.set[k];
        const auto kk = static_cast<std::size_t>(k);
        Json e;
        e["name"] = v.name;
        e["type"] = std::string(to_string(v.type));
        const bool ghost = v.type == VariableType::GhostCoefficient;
        e["surface"] = ghost ? Json(nullptr) : Json(v.file_surface);
        e["plate"] = ghost ? Json(v.ghost_plate) : Json(nullptr);
        e["order"] = ghost ? Json(v.ghost_order) : Json(nullptr);
        e["lower"] = number(v.lower);
        e["upper"] = number(v.upper);
        e["typical"] = number(v.typical);
        e["start"] = number(r.x0[kk]);
        e["result"] = number(b.x[kk]);
        vars.push_back(std::move(e));
    }
    j["variables"] = std::move(vars);

    Json cons = Json::array();
    for (const ConstraintReport& c : r.constraint_reports) {
        Json e;
        e["item"] = c.constraint.item;
        e["type"] = std::string(to_string(c.constraint.type));
        e["mode"] = std::string(to_string(c.constraint.mode));
        e["weight"] = number(c.constraint.weight);
        e["target"] = number(c.constraint.target);
        e["value_start"] = number(c.value_before);
        e["value_result"] = number(c.value_after);
        e["residual_start"] = number(c.residual_before);
        e["residual_result"] = number(c.residual_after);
        cons.push_back(std::move(e));
    }
    j["constraints"] = std::move(cons);
    j["ghost_diagnosis"] = r.ghost ? Json::parse(ghost_diagnosis_json(*r.ghost)) : Json(nullptr);

    if (a.quiet) {
        j["iterations"] = nullptr;
    } else {
        Json its = Json::array();
        for (const IterationRecord& it : r.iterations) {
            Json e;
            e["run"] = it.run;
            e["iteration"] = it.iteration;
            e["status"] = to_string(it.status);
            e["evaluated"] = it.evaluated;
            e["accepted"] = it.accepted;
            e["merit"] = number(it.merit);
            e["trial_merit"] = number(it.trial_merit);
            e["gain_ratio"] = number(it.gain_ratio);
            e["damping"] = number(it.damping);
            e["step_norm"] = number(it.step_norm);
            e["failed_rays"] = it.failed_rays;
            its.push_back(std::move(e));
        }
        j["iterations"] = std::move(its);
    }

    Json runs = Json::array();
    for (std::size_t i = 0; i < r.result.runs.size(); ++i) {
        const LmRunResult& run = r.result.runs[i];
        Json e;
        e["run"] = static_cast<int>(i);
        e["status"] = to_string(run.status);
        e["iterations"] = run.iterations;
        e["accepted"] = run.accepted;
        e["jacobians"] = run.jacobians;
        e["start_merit"] = number(run.history.front());
        e["merit"] = number(run.merit);
        e["failed_rays"] = run.failed_rays;
        e["start"] = numbers(std::span<const double>(r.starts).subspan(
            i * static_cast<std::size_t>(r.set.size()), static_cast<std::size_t>(r.set.size())));
        e["x"] = numbers(run.x);
        runs.push_back(std::move(e));
    }
    j["runs"] = std::move(runs);
    j["best_run"] = r.best;

    Json res;
    res["status"] = to_string(b.status);
    res["converged"] = lm_converged(b.status);
    res["iterations"] = b.iterations;
    res["accepted"] = b.accepted;
    res["jacobians"] = b.jacobians;
    res["start_merit"] = number(r.start_merit());
    res["merit"] = number(b.merit);
    res["start_failed_rays"] = r.start_failed;
    res["failed_rays"] = b.failed_rays;
    res["damping"] = number(b.damping);
    res["gradient_norm"] = number(b.gradient_norm);
    res["model_decrease"] = number(b.model_decrease);
    j["result"] = std::move(res);

    j["reference_nm"] = number(sys.spectrum.reference_nm);
    if (r.wavefront.before && r.wavefront.after) {
        j["wavefront"] = Json{{"start", wavefront_json(sys, *r.wavefront.before)},
                              {"result", wavefront_json(sys, *r.wavefront.after)}};
        j["wavefront_error"] = nullptr;
    } else {
        j["wavefront"] = nullptr;
        j["wavefront_error"] = r.wavefront.error;
    }
    Json warnings = Json::array();
    for (const std::string& w : r.warnings) {
        warnings.push_back(w);
    }
    j["warnings"] = std::move(warnings);
    out << j.dump(2, ' ', false, Json::error_handler_t::replace) << '\n';
}

} // namespace

void print_optimize_usage(std::ostream& out) {
    out << "Aufruf: mikroptim optimize <system.json> [-o <ergebnis.json>] [--json] [--quiet]\n"
           "                          [--max-iterations N] [--rings N] [--spokes M]\n"
           "                          [--runs K [--perturbation s] [--seed n]]\n"
           "                          [--gpu [--device N]]\n"
           "  Optimiert die Variablen (Abschnitt variables) mit Levenberg-Marquardt (FP64),\n"
           "  Merit = Summe der Quadrate aus Wellenfront- und Zusatzresiduen (constraints,\n"
           "  L1-Strafe der Geisterkoeffizienten) in Wellen²; mit Geisterplatten folgt ein\n"
           "  Diagnosebericht. Beschreibung: docs/cli.md.\n"
           "  -o DATEI           Ergebnis als neue Systemdatei (nie die Eingabedatei)\n"
           "  --json             maschinenlesbare Ausgabe (Format mikroptim.optimize/1)\n"
           "  --quiet, -q        ohne Iterationsprotokoll\n"
           "  --max-iterations N Zyklen pro Lauf, >= 0 (Standard "
        << kDefaultMaxIterations
        << "; 0 = nur auswerten)\n"
           "  --rings N          radiale Ringe der Pupillenquadratur, 1 bis "
        << kMaxPupilRings << " (Standard " << kDefaultRings
        << ")\n"
           "  --spokes M         Speichen der Pupillenquadratur, >= 1 (Standard "
        << kDefaultSpokes
        << ")\n"
           "  --runs K           Multistart mit K Läufen, Lauf 0 ungestört (Standard 1)\n"
           "  --perturbation s   relative Störung der Startpunkte (Standard "
        << fixed(kDefaultPerturbation, 2)
        << ")\n"
           "  --seed n           Startwert der Störungen (Standard 0)\n"
           "  --gpu              auf der GPU rechnen statt auf der CPU\n"
           "  --device N         CUDA-Gerät für --gpu (Standard 0, Liste: mikroptim --devices)\n"
           "  --help             diese Hilfe\n"
           "Exit-Code: 0 Erfolg, 1 Fehler beim Laden, Optimieren oder Schreiben, 2 ungültige\n"
           "           Kommandozeile, 3 GPU-Backend nicht verfügbar.\n";
}

int run_optimize_command(std::span<const std::string_view> args, std::ostream& out,
                         std::ostream& err) {
    OptimizeArgs a;
    try {
        a = parse_args(args);
        if (!a.output.empty() && same_file(fs::path(a.file), fs::path(a.output))) {
            throw UsageError("Ergebnisdatei " + a.output +
                             " ist die Eingabedatei; die Eingabe wird nie überschrieben");
        }
    } catch (const UsageError& e) {
        err << "mikroptim optimize: " << e.what() << '\n';
        print_optimize_usage(err);
        return kExitUsage;
    }
    if (a.help) {
        print_optimize_usage(out);
        return kExitOk;
    }

    // GPU-Backend vor dem Laden prüfen (wie mikroptim trace).
#ifdef MIKROPTIM_HAS_CUDA
    std::optional<cuda::TraceEngine> engine;
    if (a.gpu) {
        const int count = cuda::device_count();
        if (count == 0) {
            err << "mikroptim optimize: --gpu: kein CUDA-Gerät gefunden (GPU-Backend nicht "
                   "verfügbar)\n";
            return kExitNoGpu;
        }
        if (a.device >= count) {
            err << "mikroptim optimize: Fehler: --device " << a.device
                << ": Gerät existiert nicht (" << count
                << (count == 1 ? " CUDA-Gerät" : " CUDA-Geräte")
                << ", siehe mikroptim --devices)\n";
            return kExitError;
        }
        try {
            engine.emplace(a.device);
        } catch (const std::exception& e) {
            err << "mikroptim optimize: Fehler: GPU " << a.device << ": " << e.what() << '\n';
            return kExitError;
        }
    }
#else
    if (a.gpu) {
        err << "mikroptim optimize: --gpu: ohne CUDA gebaut (MIKROPTIM_ENABLE_CUDA=OFF), "
               "GPU-Backend nicht verfügbar\n";
        return kExitNoGpu;
    }
#endif

    try {
        OptimizeReport r;
        r.args = &a;
        r.loaded = load_system(fs::path(a.file));
        for (const std::string& w : r.loaded.warnings) {
            err << "mikroptim optimize: Warnung: " << w << '\n';
            r.warnings.push_back(w);
        }
        const SystemDescription& sys = r.loaded.system;
        if (!sys.variables || sys.variables->items.empty()) {
            err << "mikroptim optimize: Fehler: " << a.file
                << ": keine Variablen (Abschnitt variables fehlt oder ist leer)\n";
            return kExitError;
        }
        const SystemBatchHost base = make_system_batch(r.loaded, 1);
        r.set = VariableSet(sys, base.topology());
        if (r.set.empty()) {
            err << "mikroptim optimize: Fehler: " << a.file << ": keine Variablen\n";
            return kExitError;
        }
        r.constraints = make_paraxial_constraints(sys);
        check_paraxial_constraints(r.constraints, base.topology());
        r.penalty = make_ghost_penalty(sys);
        check_ghost_penalty(r.penalty, base.topology());
        r.x0 = r.set.extract(base, 0);

        // Feldziele h_f = f_soll · tan θ_f bleiben während der Optimierung fest. Ohne
        // target.focal_length stammt f_soll aus der paraxialen Brennweite des Startsystems; das
        // geladene Ergebnis hätte dann andere Feldziele (docs/cli.md).
        if (sys.source.type == SourceType::Collimated &&
            !(sys.target && sys.target->focal_length)) {
            for (const double angle : sys.source.field_angles_deg) {
                if (angle != 0.0) {
                    r.focal_from_start = true;
                }
            }
        }
        if (r.focal_from_start) {
            const std::string w =
                "Feldziele aus der paraxialen Brennweite des Startsystems (kein "
                "target.focal_length); beim erneuten Laden des Ergebnisses ändern sie sich";
            err << "mikroptim optimize: Warnung: " << w << '\n';
            r.warnings.push_back(w);
        }

        LmOptions options;
        options.max_iterations = a.max_iterations;
        options.jacobian.quadrature.rings = a.rings;
        options.jacobian.quadrature.spokes = a.spokes;
        // Zusatzresiduen: paraxiale Nebenbedingungen, dahinter die L1-Strafe
        options.jacobian.extra = concat_extra_residuals(
            {make_paraxial_residuals(r.constraints), make_ghost_penalty_residuals(r.penalty)});

        // Startpunkte: Lauf 0 ist x₀, Lauf j ≥ 1 gestört (perturbed_vector mit first_is_base).
        std::vector<double>& starts = r.starts;
        if (a.runs == 1) {
            starts = r.x0;
        } else {
            PerturbationOptions p;
            p.curvature = {a.perturbation, 0.0};
            p.thickness = {a.perturbation, 0.0};
            p.working_distance = {a.perturbation, 0.0};
            p.ghost_coefficient = {a.perturbation, 0.0};
            p.seed = a.seed;
            p.first_is_base = true;
            for (int j = 0; j < a.runs; ++j) {
                const std::vector<double> y =
                    perturbed_vector(r.set, r.x0, p, static_cast<std::uint64_t>(j));
                starts.insert(starts.end(), y.begin(), y.end());
            }
        }

#ifdef MIKROPTIM_HAS_CUDA
        if (engine) {
            r.backend.gpu = true;
            r.backend.device = engine->device();
            r.backend.device_name = engine->device_name();
        }
#endif
        const bool stream_text = !a.json && !a.quiet;
        if (!a.json) {
            write_text_header(r, out);
            if (stream_text) {
                write_iteration_header(out, sys.spectrum.reference_nm);
            }
            out.flush();
        }
        options.on_iteration = [&](const LmIterationInfo& info) {
            IterationRecord rec;
            rec.run = info.run;
            rec.iteration = info.iteration;
            rec.status = info.status;
            rec.evaluated = info.evaluated;
            rec.accepted = info.accepted;
            rec.merit = info.merit;
            rec.trial_merit = info.trial_merit;
            rec.gain_ratio = info.gain_ratio;
            rec.damping = info.damping;
            rec.step_norm = info.step_norm;
            rec.failed_rays = info.failed_rays;
            if (stream_text) {
                write_iteration_line(rec, out);
            } else if (a.json && !a.quiet) {
                r.iterations.push_back(rec);
            }
        };

#ifdef MIKROPTIM_HAS_CUDA
        if (engine) {
            r.result = cuda::optimize_lm_gpu(*engine, r.set, base, starts, options);
        } else {
            r.result = optimize_lm_cpu(r.set, base, starts, options);
        }
#else
        r.result = optimize_lm_cpu(r.set, base, starts, options);
#endif
        // Versagte Strahlen bei x₀ (das Protokoll kennt sie nur an den Versuchspunkten); F(x₀)
        // ist history[0] von Lauf 0.
        {
            std::vector<double> f(1, 0.0);
            std::vector<std::int64_t> failed(1, 0);
            const int run0 = 0;
#ifdef MIKROPTIM_HAS_CUDA
            if (engine) {
                cuda::GpuLmBackend backend(*engine, r.set, base, options);
                backend.merit(std::span<const int>(&run0, 1), r.x0, f, failed);
            } else {
                CpuLmBackend backend(r.set, base, options);
                backend.merit(std::span<const int>(&run0, 1), r.x0, f, failed);
            }
#else
            CpuLmBackend backend(r.set, base, options);
            backend.merit(std::span<const int>(&run0, 1), r.x0, f, failed);
#endif
            r.start_failed = failed[0];
        }
        r.best = select_best(r.result);
        if (r.best < 0) {
            err << "mikroptim optimize: Fehler: Merit in keinem Lauf endlich (Startpunkt nicht "
                   "auswertbar)\n";
            return kExitError;
        }
        const LmRunResult& best = r.best_run();

        // Nebenbedingungen vorher und nachher
        const int run0 = 0;
        const SystemBatchHost after =
            lm_point_batch(r.set, base, std::span<const int>(&run0, 1), best.x);
        r.constraint_reports = constraint_reports(r.constraints, base, after);

        // Diagnose der Geisterplatten am Ergebnis (Merit auf der CPU wie die Wellenfront)
        if (!sys.ghost_plates.empty()) {
            r.ghost = diagnose_ghost_plates(
                sys, r.loaded.catalog, after, 0, r.penalty,
                make_cpu_merit(options.jacobian.quadrature, options.jacobian.extra));
        }

        // Ergebnis als Systembeschreibung (Glaskataloge noch relativ zur Eingabedatei)
        SystemDescription result_system = sys;
        r.set.apply_to_description(result_system, best.x);
        std::error_code ec;
        fs::path input_dir = fs::absolute(fs::path(a.file), ec).parent_path();
        if (ec) {
            input_dir = fs::path(a.file).parent_path();
        }
        const LoadedSystem result_loaded = prepare_system(result_system, input_dir);

        // Wellenfront vorher und nachher (CPU, ideale Bildpunkte wie mikroptim trace)
        WavefrontOptions wo;
        wo.rings = a.rings;
        wo.spokes = a.spokes;
        try {
            r.wavefront.before = evaluate_wavefront(r.loaded, wo);
            r.wavefront.after = evaluate_wavefront(result_loaded, wo);
        } catch (const std::invalid_argument& e) {
            r.wavefront.before.reset();
            r.wavefront.after.reset();
            r.wavefront.error = e.what();
        }

        // Ergebnisdatei: Glaskataloge so, dass sie vom Verzeichnis der Ergebnisdatei aus gefunden
        // werden.
        if (!a.output.empty()) {
            const fs::path output(a.output);
            fs::path out_dir = fs::absolute(output, ec).parent_path();
            if (ec) {
                out_dir = output.parent_path();
            }
            SystemDescription file_system = result_system;
            file_system.glass_catalogs =
                catalogs_for(result_system, r.loaded.catalog_files, out_dir);
            write_system_file(file_system, output);
        }

        if (a.json) {
            write_json(r, out);
        } else {
            write_text_summary(r, out);
            if (!a.output.empty()) {
                out << '\n' << "Ergebnis geschrieben: " << a.output << '\n';
            }
        }
        out.flush();
        if (!out) {
            err << "mikroptim optimize: Fehler beim Schreiben der Ausgabe\n";
            return kExitError;
        }
        return kExitOk;
    } catch (const std::exception& e) {
        err << "mikroptim optimize: Fehler: " << e.what() << '\n';
        return kExitError;
    }
}

} // namespace mko::cli
