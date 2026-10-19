// Optisches Modell des Homotopie-Controllers (M6-T3): Punkt realisieren, LM-Schritt, Diagnose,
// Einfüge-Operatoren A und B, Zielprüfung am realen System. docs/optimierung.md, Abschnitt 7.

#include "mikroptim/synth/optical_homotopy.hpp"

#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/glass/glass_hull.hpp"
#include "mikroptim/optim/ghost_penalty.hpp"
#include "mikroptim/optim/glass_snap.hpp"
#include "mikroptim/optim/glass_variables.hpp"
#include "mikroptim/optim/mechanical_constraints.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "mikroptim/optim/variable_set.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace mko {

namespace {

[[noreturn]] void invalid(const std::string& what) {
    throw std::invalid_argument("Homotopie (optisch): " + what);
}

std::string number(double v, const char* fmt = "%.6g") {
    char buf[64];
    std::snprintf(buf, sizeof buf, fmt, v);
    return buf;
}

bool has_glass_variables(const SystemDescription& d) {
    if (!d.variables) {
        return false;
    }
    for (const VariableSpec& v : d.variables->items) {
        if (v.type == VariableType::Glass) {
            return true;
        }
    }
    return false;
}

bool has_constraint(const SystemDescription& d, ConstraintType type) {
    if (!d.constraints) {
        return false;
    }
    for (const ConstraintSpec& c : d.constraints->items) {
        if (c.type == type) {
            return true;
        }
    }
    return false;
}

// Zusatzresiduen wie `mikroptim optimize`: paraxial, L1-Strafe, Glas-Hülle, Mechanik
// (geometrisch); trace-abhängige Mechanik getrennt.
struct Extras {
    ExtraResiduals extra;
    TraceExtraResiduals traced;
};

Extras make_extras(const LoadedSystem& l, const SystemBatchHost& base, const VariableSet& set,
                   const GlassHull* hull) {
    const SystemDescription& d = l.system;
    const ParaxialConstraints paraxial = make_paraxial_constraints(d);
    check_paraxial_constraints(paraxial, base.topology());
    GhostPenalty ghost = make_ghost_penalty(d);
    check_ghost_penalty(ghost, base.topology());
    GlassHullPenalty glass;
    if (hull != nullptr && has_glass_variables(d) && has_constraint(d, ConstraintType::GlassHull)) {
        glass = make_glass_hull_penalty(d, set, *hull);
        check_glass_hull_penalty(glass, base.topology());
    }
    const MechanicalConstraints mech = make_mechanical_constraints(d);
    check_mechanical_constraints(mech, base.topology());
    MechanicalResiduals m = make_mechanical_residuals(mech);
    Extras e;
    e.extra = concat_extra_residuals(
        {make_paraxial_residuals(paraxial), make_ghost_penalty_residuals(std::move(ghost)),
         make_glass_hull_residuals(std::move(glass)), std::move(m.geometric)});
    e.traced = std::move(m.traced);
    return e;
}

double ghost_abs_sum(const SystemDescription& d) {
    double s = 0.0;
    for (const GhostPlate& g : d.ghost_plates) {
        for (const double a : g.coefficients) {
            s += std::abs(a);
        }
    }
    return s;
}

// Größter RMS [Wellen bei λ des Tupels] der Tupel mit Gewicht > 0 (NaN, wenn eines keinen
// gültigen Strahl hat).
std::vector<HomotopyTupleRms> tuple_rms(const LoadedSystem& l, const SystemBatchHost& batch,
                                        const ResidualSettings& quadrature) {
    const ResidualLayout layout(batch.topology(), quadrature);
    const ResidualResult res = residuals_cpu(batch, 0, layout);
    const std::vector<TupleWavefront> w = tuple_wavefronts(res, layout, 0);
    const SystemDescription& d = l.system;
    std::vector<HomotopyTupleRms> out;
    for (const TupleWavefront& t : w) {
        const auto f = static_cast<std::size_t>(t.field);
        const auto k = static_cast<std::size_t>(t.wavelength);
        const double wf =
            d.source.type == SourceType::Collimated ? d.source.field_weights.at(f) : 1.0;
        if (!(wf > 0.0) || !(d.spectrum.weights.at(k) > 0.0)) {
            continue;
        }
        HomotopyTupleRms r;
        r.field = t.field;
        r.wavelength = t.wavelength;
        r.field_angle =
            d.source.type == SourceType::Collimated ? d.source.field_angles_deg.at(f) : 0.0;
        r.wavelength_nm = d.spectrum.wavelengths_nm[k];
        r.rms =
            t.valid() ? t.rms / (r.wavelength_nm * 1e-6) : std::numeric_limits<double>::quiet_NaN();
        r.failed = t.failed;
        out.push_back(r);
    }
    return out;
}

double max_rms(const std::vector<HomotopyTupleRms>& t) {
    double m = 0.0;
    for (const HomotopyTupleRms& r : t) {
        if (!std::isfinite(r.rms)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        m = std::max(m, r.rms);
    }
    return m;
}

} // namespace

bool is_one_sided(const ConstraintSpec& spec) {
    if (is_mechanical(spec.type)) {
        return true;
    }
    switch (spec.type) {
    case ConstraintType::Petzval:
    case ConstraintType::GlassHull:
        return true;
    case ConstraintType::GhostPenalty:
        return false;
    default:
        return spec.mode_or_default() != ConstraintMode::Equal;
    }
}

int glass_elements(const SystemDescription& system) {
    int n = 0;
    for (std::size_t i = 0; i + 1 < system.surfaces.size(); ++i) {
        if (system.surfaces[i].material.kind != MaterialKind::Air) {
            ++n;
        }
    }
    return n;
}

OpticalHomotopyModel::OpticalHomotopyModel(const LoadedSystem& start,
                                           OpticalHomotopyOptions options)
    : like_(start), start_(start.system), options_(std::move(options)) {
    if (start_.source.type == SourceType::Collimated) {
        if (!start_.target || !start_.target->focal_length ||
            !(*start_.target->focal_length > 0.0)) {
            invalid("kollimierte Quelle verlangt target.focal_length > 0 (Pupillenradius = "
                    "f_soll · NA)");
        }
        focal_ = *start_.target->focal_length;
    } else if (!(start_.source.numerical_aperture > 0.0)) {
        invalid("Punktquelle ohne numerische Apertur");
    }
    if (!(options_.element_thickness > 0.0) || options_.element_stages < 1 ||
        !(options_.min_air_gap >= 0.0) || !(options_.penalty_raise >= 1.0) ||
        !std::isfinite(options_.penalty_raise) || !(options_.saddle_delta >= 0.0) ||
        !(options_.mechanical_margin >= 0.0) || !(options_.angle_margin >= 0.0) ||
        !(options_.petzval_margin >= 0.0) || !(options_.petzval_margin < 1.0) ||
        !(options_.glass_hull_margin >= 0.0)) {
        invalid("ungültige Optionen (Dicke, Stufen, Spalt, Faktoren, Vorhalt)");
    }
    // Glasvariablen: Startgläser einmal in Buchdahl-Materialien umwandeln (wie optimize)
    (void)convert_glass_variables(start_, like_.catalog);
    like_ = rebuild_loaded_system(like_, start_);
}

HomotopyPoint OpticalHomotopyModel::start_point() const {
    HomotopyPoint p;
    p.aperture = start_.source.type == SourceType::Collimated
                     ? start_.surfaces.front().semi_diameter / focal_
                     : start_.source.numerical_aperture;
    return p;
}

SystemDescription OpticalHomotopyModel::realize(const SystemDescription& system,
                                                const HomotopyPoint& p, bool margins) const {
    if (!(p.aperture > 0.0) || !std::isfinite(p.aperture) || !(p.field >= 0.0) ||
        !std::isfinite(p.field) || !(p.ghost_weight > 0.0) || !(p.ghost_smoothing > 0.0) ||
        !(p.penalty_scale > 0.0)) {
        invalid("ungültiger Homotopie-Punkt");
    }
    SystemDescription d = system;
    double ratio = 1.0;
    if (d.source.type == SourceType::Collimated) {
        const double a = focal_ * p.aperture;
        ratio = a / system.surfaces.front().semi_diameter;
        d.surfaces.front().semi_diameter = a;
        for (double& angle : d.source.field_angles_deg) {
            angle *= p.field;
        }
    } else {
        ratio = p.aperture / system.source.numerical_aperture;
        d.source.numerical_aperture = p.aperture;
    }
    for (GhostPlate& g : d.ghost_plates) {
        g.norm_radius *= ratio;
    }
    if (options_.spectra.empty()) {
        if (p.spectrum != 0) {
            invalid("Spektrumstufe " + std::to_string(p.spectrum) + " ohne Stufenliste");
        }
    } else {
        if (p.spectrum < 0 || p.spectrum >= static_cast<int>(options_.spectra.size())) {
            invalid("Spektrumstufe " + std::to_string(p.spectrum) + " außerhalb der Liste");
        }
        d.spectrum = options_.spectra[static_cast<std::size_t>(p.spectrum)];
    }
    if (d.constraints) {
        for (ConstraintSpec& c : d.constraints->items) {
            if (c.type == ConstraintType::GhostPenalty) {
                c.weight = c.weight_or_default() * p.ghost_weight;
                c.smoothing = c.smoothing_or_default() * p.ghost_smoothing;
                continue;
            }
            if (!is_one_sided(c)) {
                continue;
            }
            c.weight = c.weight_or_default() * p.penalty_scale;
            if (!margins) {
                continue;
            }
            const double m = options_.mechanical_margin;
            switch (c.type) {
            case ConstraintType::MinCenterThickness:
            case ConstraintType::MinEdgeThickness:
                c.thickness = *c.thickness + m;
                break;
            case ConstraintType::MaxCenterThickness:
                c.thickness = *c.thickness - m;
                break;
            case ConstraintType::MaxSemiDiameter:
                c.clearance = c.clearance_or_default() + m;
                break;
            case ConstraintType::MaxIncidenceAngle:
                c.angle = *c.angle - options_.angle_margin;
                break;
            case ConstraintType::Petzval:
                c.max_curvature = *c.max_curvature * (1.0 - options_.petzval_margin);
                break;
            case ConstraintType::GlassHull:
                if (options_.glass_hull_margin > 0.0) {
                    c.margin = c.margin_or_default() + options_.glass_hull_margin;
                }
                break;
            default:
                break;
            }
        }
    }
    return d;
}

LoadedSystem OpticalHomotopyModel::load(const SystemDescription& realized) const {
    return rebuild_loaded_system(like_, realized);
}

HomotopySolve OpticalHomotopyModel::solve(SystemDescription& system, const HomotopyPoint& p) {
    const LoadedSystem l = load(realize(system, p));
    const SystemBatchHost base = make_system_batch(l, 1);
    const VariableSet set(l.system, base.topology());
    const std::vector<double> x0 = set.extract(base, 0);
    GlassHull hull;
    const bool need_hull =
        has_glass_variables(l.system) && has_constraint(l.system, ConstraintType::GlassHull);
    if (need_hull) {
        hull = make_glass_hull(like_.catalog);
    }
    LmOptions o = options_.lm;
    Extras e = make_extras(l, base, set, need_hull ? &hull : nullptr);
    o.jacobian.extra = std::move(e.extra);
    o.jacobian.trace_extra = std::move(e.traced);
    const LmResult res = options_.optimizer ? options_.optimizer(set, base, x0, o)
                                            : optimize_lm_cpu(set, base, x0, o);
    const LmRunResult& run = res.runs.at(0);
    set.apply_to_description(system, run.x);

    HomotopySolve s;
    s.status = run.status;
    s.iterations = run.iterations;
    s.start_merit =
        run.history.empty() ? std::numeric_limits<double>::quiet_NaN() : run.history.front();
    s.merit = run.merit;
    s.failed_rays = run.failed_rays;
    s.ghost_sum = ghost_abs_sum(system);
    const int run0 = 0;
    const SystemBatchHost at = lm_point_batch(set, base, std::span<const int>(&run0, 1), run.x);
    s.rms = max_rms(tuple_rms(l, at, o.jacobian.quadrature));
    s.wavefront = make_cpu_merit(o.jacobian.quadrature, {}, options_.lm.threads)(at).at(0);
    s.ghost_penalty = ghost_penalty_merit(make_ghost_penalty(l.system), at, 0);
    return s;
}

HomotopyNeed OpticalHomotopyModel::diagnose(const SystemDescription& system,
                                            const HomotopyPoint& p) {
    HomotopyNeed n;
    if (system.ghost_plates.empty()) {
        n.text = "keine Geisterplatten";
        return n;
    }
    const LoadedSystem l = load(realize(system, p));
    const SystemBatchHost base = make_system_batch(l, 1);
    const VariableSet set(l.system, base.topology());
    GlassHull hull;
    const bool need_hull =
        has_glass_variables(l.system) && has_constraint(l.system, ConstraintType::GlassHull);
    if (need_hull) {
        hull = make_glass_hull(like_.catalog);
    }
    Extras e = make_extras(l, base, set, need_hull ? &hull : nullptr);
    n.diagnosis =
        diagnose_ghost_plates(l.system, l.catalog, base, 0, make_ghost_penalty(l.system),
                              make_cpu_merit(options_.lm.jacobian.quadrature, std::move(e.extra),
                                             options_.lm.threads, std::move(e.traced)),
                              options_.diagnosis);
    n.recommendation = n.diagnosis.recommendation;
    n.text = n.diagnosis.recommendation_text;
    return n;
}

HomotopyInsertion OpticalHomotopyModel::insert(const SystemDescription& system,
                                               const HomotopyPoint& p, const HomotopyNeed& need) {
    const SystemDescription realized = realize(system, p);
    HomotopyInsertion ins;
    ins.op = need.recommendation;
    // Mechanische Einträge mit Standardauswahl (renumber_surfaces macht daraus Listen): ihre
    // Ordnungszahl unter den mechanischen Einträgen (scale_ghost_penalty kann Einträge der Strafe
    // einfügen, die Reihenfolge der mechanischen bleibt)
    if (system.constraints) {
        int ordinal = 0;
        for (const ConstraintSpec& c : system.constraints->items) {
            if (!is_mechanical(c.type)) {
                continue;
            }
            if (c.surface_selection == SurfaceSelection::None ||
                c.surface_selection == SurfaceSelection::All) {
                ins.default_constraints.push_back(ordinal);
            }
            ++ordinal;
        }
    }
    if (need.recommendation == GhostOperator::A) {
        const MeniscusPlan plan =
            meniscus_from_diagnosis(realized, need.diagnosis, options_.meniscus);
        InsertionResult r = insert_meniscus(system, plan.insertion);
        scale_ghost_penalty(r.system, plan.plate, options_.penalty_raise);
        ins.first_surface = r.first_new_surface;
        ins.new_surfaces = r.new_surfaces;
        ins.plate = plan.plate;
        ins.gap = r.system.surfaces[static_cast<std::size_t>(r.first_new_surface + 1)].thickness;
        ins.min_gap = options_.min_air_gap;
        // untere Schranke einer Dickenvariable des Spalts hinter dem Meniskus
        {
            const LoadedSystem l = load(realize(r.system, p));
            const SystemBatchHost b = make_system_batch(l, 1);
            const VariableSet set(l.system, b.topology());
            for (int k = 0; k < set.size(); ++k) {
                if (set[k].type == VariableType::Thickness &&
                    set[k].file_surface == r.first_new_surface + 1 &&
                    std::isfinite(set.lower()[static_cast<std::size_t>(k)])) {
                    ins.min_gap = std::max(ins.min_gap, set.lower()[static_cast<std::size_t>(k)]);
                }
            }
        }
        for (int j = 0; j < options_.element_stages; ++j) {
            double t = options_.element_thickness;
            for (int k = j; k < options_.element_stages - 1; ++k) {
                t *= 0.5;
            }
            ins.stages.push_back(t);
        }
        if (options_.saddle_delta > 0.0) {
            const auto starts = meniscus_saddle_starts(
                r.system, r.first_new_surface, options_.saddle_delta, -options_.saddle_delta);
            ins.branches.assign(starts.begin(), starts.end());
        } else {
            ins.branches.push_back(r.system);
        }
        // Glasname aus dem Bericht (Katalogname oder Modell mit Koordinaten)
        std::string glass_name = "Modell";
        for (const GhostPlateReport& pr : need.diagnosis.plates) {
            if (pr.plate == plan.plate) {
                glass_name = pr.glass.name;
            }
        }
        ins.text =
            "Operator A: Meniskus nach Fläche " + std::to_string(plan.insertion.after_surface) +
            " bei " + number(plan.insertion.position) +
            " mm, c0 = " + number(plan.insertion.curvature) + "/mm, Glas " + glass_name +
            " (Platte " + std::to_string(plan.plate) + ", mu x" + number(options_.penalty_raise) +
            "), Dicke bis " + number(options_.element_thickness) + " mm, " +
            std::to_string(ins.branches.size()) + (ins.branches.size() == 1 ? " Zweig" : " Zweige");
        return ins;
    }
    if (need.recommendation == GhostOperator::B ||
        need.recommendation == GhostOperator::BAnomalous) {
        const SplitPlan plan =
            split_from_diagnosis(realized, like_.catalog, need.diagnosis, options_.split);
        InsertionResult r = split_lens(system, plan.split);
        free_glass(r.system, plan.free_surface);
        if (!has_constraint(r.system, ConstraintType::GlassHull)) {
            ConstraintSpec c;
            c.type = ConstraintType::GlassHull;
            if (!r.system.constraints) {
                r.system.constraints = Constraints{};
            }
            r.system.constraints->items.push_back(c);
        }
        (void)convert_glass_variables(r.system, like_.catalog);
        scale_ghost_penalty(r.system, plan.plate, options_.penalty_raise);
        ins.first_surface = r.first_new_surface;
        ins.new_surfaces = r.new_surfaces;
        ins.plate = plan.plate;
        ins.branches.push_back(std::move(r.system));
        ins.text = "Operator " + std::string(to_string(need.recommendation)) +
                   ": Linse an Fläche " + std::to_string(plan.split.lens_surface) +
                   " geteilt, c_s = " + number(plan.split.curvature) + "/mm, Glas nach Fläche " +
                   std::to_string(plan.free_surface) + " frei (Platte " +
                   std::to_string(plan.plate) + ", mu x" + number(options_.penalty_raise) + ")";
        return ins;
    }
    invalid("kein Operator empfohlen");
}

void OpticalHomotopyModel::set_stage(SystemDescription& system, const HomotopyInsertion& ins,
                                     double t) {
    if (ins.op != GhostOperator::A) {
        return;
    }
    const auto f = static_cast<std::size_t>(ins.first_surface);
    system.surfaces.at(f).thickness = t;
    const double gap = std::max(ins.gap - t, ins.min_gap);
    system.surfaces.at(f + 1).thickness = gap;
    // Geisterplatten im Spalt hinter dem Meniskus bleiben im Spalt
    for (GhostPlate& g : system.ghost_plates) {
        if (g.after_surface == ins.first_surface + 1 && g.distance > gap) {
            g.distance = gap;
        }
    }
}

void OpticalHomotopyModel::finish_insertion(SystemDescription& system,
                                            const HomotopyInsertion& ins) {
    if (ins.op == GhostOperator::A && options_.element_max_thickness > options_.element_thickness) {
        VariableSpec v;
        v.type = VariableType::Thickness;
        v.selection = SurfaceSelection::One;
        v.surfaces = {ins.first_surface};
        v.min = options_.element_thickness;
        v.max = options_.element_max_thickness;
        v.comment = "Mittendicke des Meniskus (Homotopie)";
        if (!system.variables) {
            system.variables = Variables{};
        }
        system.variables->items.push_back(std::move(v));
    }
    if (options_.constrain_new_elements && system.constraints) {
        int ordinal = 0;
        for (ConstraintSpec& c : system.constraints->items) {
            if (!is_mechanical(c.type)) {
                continue;
            }
            if (std::find(ins.default_constraints.begin(), ins.default_constraints.end(),
                          ordinal) != ins.default_constraints.end()) {
                c.surface_selection = SurfaceSelection::None;
                c.surfaces.clear();
            }
            ++ordinal;
        }
    }
    validate_system(system);
}

HomotopyGoal OpticalHomotopyModel::goal(const SystemDescription& system, const HomotopyPoint& p,
                                        double target) {
    SystemDescription d = realize(system, p, false);
    for (GhostPlate& g : d.ghost_plates) {
        std::fill(g.coefficients.begin(), g.coefficients.end(), 0.0);
    }
    const LoadedSystem l = load(d);
    const SystemBatchHost b = make_system_batch(l, 1);
    ResidualSettings q = options_.lm.jacobian.quadrature;
    q.failure = FailureMode::Hard;
    HomotopyGoal g;
    g.target = target;
    g.tuples = tuple_rms(l, b, q);
    g.max_rms = max_rms(g.tuples);
    g.reached = !g.tuples.empty() && std::isfinite(g.max_rms) && g.max_rms <= target;
    for (const HomotopyTupleRms& t : g.tuples) {
        if (t.failed != 0) {
            g.reached = false;
        }
    }
    return g;
}

int OpticalHomotopyModel::elements(const SystemDescription& system) const {
    return glass_elements(system);
}

bool OpticalHomotopyModel::one_sided_constraints(const SystemDescription& system) const {
    if (!system.constraints) {
        return false;
    }
    for (const ConstraintSpec& c : system.constraints->items) {
        if (is_one_sided(c)) {
            return true;
        }
    }
    return false;
}

} // namespace mko
