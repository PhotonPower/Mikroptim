// Optisches Modell des Homotopie-Controllers (M6-T3): Punkt realisieren, LM-Schritt, Diagnose,
// Einfüge-Operatoren A und B, Zielprüfung am realen System. docs/optimierung.md, Abschnitt 7.

#include "mikroptim/synth/optical_homotopy.hpp"

#include "mikroptim/merit/wavefront.hpp"
#include "mikroptim/paraxial/aberrations.hpp"
#include "mikroptim/trace/surface_table.hpp"

#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/glass/glass_hull.hpp"
#include "mikroptim/optim/ghost_penalty.hpp"
#include "mikroptim/optim/glass_snap.hpp"
#include "mikroptim/optim/glass_variables.hpp"
#include "mikroptim/optim/mechanical_constraints.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "mikroptim/optim/variant_generator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <random>
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

// Zustand am Punkt, vorbereitet für den LM: geladenes System, Vorlage mit einer Variante,
// Variablen und Startwerte.
struct Prepared {
    LoadedSystem l;
    SystemBatchHost base;
    VariableSet set;
    std::vector<double> x0;
};

Prepared prepare(const OpticalHomotopyModel& m, const SystemDescription& system,
                 const HomotopyPoint& p) {
    Prepared r;
    r.l = m.load(m.realize(system, p));
    r.base = make_system_batch(r.l, 1);
    r.set = VariableSet(r.l.system, r.base.topology());
    r.x0 = r.set.extract(r.base, 0);
    return r;
}

// Gleiche Topologie (gemeinsamer LM-Aufruf, Vergleich der Variablen): gleiche Batch-Topologie
// und Variablen (Flächen, Schranken, typische Größen, Versätze), gleiche realisierte Beschreibung,
// wenn in beide die Variablenwerte von b geschrieben werden (Nebenbedingungen, Gewichte,
// Gläser, Dicken außerhalb der Variablen), und die Vorlage von a mit den Werten von b bitgleich
// zur Vorlage von b. Dann sieht der LM für b mit der Vorlage von a dieselben Batches und
// Zusatzresiduen wie allein.
bool same_topology(const Prepared& a, const Prepared& b) {
    if (a.base.topology() != b.base.topology() ||
        a.l.system.surfaces.size() != b.l.system.surfaces.size() ||
        a.l.system.ghost_plates.size() != b.l.system.ghost_plates.size() ||
        !std::ranges::equal(a.set.variables(), b.set.variables())) {
        return false;
    }
    for (std::size_t i = 0; i < a.l.system.surfaces.size(); ++i) {
        if (a.l.system.surfaces[i].type != b.l.system.surfaces[i].type) {
            return false;
        }
    }
    for (std::size_t g = 0; g < a.l.system.ghost_plates.size(); ++g) {
        if (a.l.system.ghost_plates[g].coefficients.size() !=
            b.l.system.ghost_plates[g].coefficients.size()) {
            return false;
        }
    }
    SystemDescription da = a.l.system;
    a.set.apply_to_description(da, b.x0);
    SystemDescription db = b.l.system;
    b.set.apply_to_description(db, b.x0);
    if (!(da == db)) {
        return false;
    }
    const int run0 = 0;
    const std::span<const int> one(&run0, 1);
    return lm_point_batch(a.set, a.base, one, b.x0) == lm_point_batch(b.set, b.base, one, b.x0);
}

// Ergebnis eines Laufs in die Beschreibung und Bericht (Status, F, Anteile, RMS, Σ|a|).
HomotopySolve finish_solve(const Prepared& pr, SystemDescription& system, const LmRunResult& run,
                           const ResidualSettings& quadrature, int threads) {
    pr.set.apply_to_description(system, run.x);
    HomotopySolve s;
    s.status = run.status;
    s.iterations = run.iterations;
    s.start_merit =
        run.history.empty() ? std::numeric_limits<double>::quiet_NaN() : run.history.front();
    s.merit = run.merit;
    s.failed_rays = run.failed_rays;
    s.ghost_sum = ghost_abs_sum(system);
    const int run0 = 0;
    const SystemBatchHost at =
        lm_point_batch(pr.set, pr.base, std::span<const int>(&run0, 1), run.x);
    s.rms = max_rms(tuple_rms(pr.l, at, quadrature));
    s.wavefront = make_cpu_merit(quadrature, {}, threads)(at).at(0);
    s.ghost_penalty = ghost_penalty_merit(make_ghost_penalty(pr.l.system), at, 0);
    return s;
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
    if (options_.spectrum_continuous && !options_.spectra.empty()) {
        invalid("Spektrumstufen und stetige Spektrumbreite schließen sich aus");
    }
    HomotopyPoint p;
    p.aperture = start_.source.type == SourceType::Collimated
                     ? start_.surfaces.front().semi_diameter / focal_
                     : start_.source.numerical_aperture;
    p.spectrum_width = options_.spectrum_continuous ? 0.0 : 1.0;
    return p;
}

SystemDescription OpticalHomotopyModel::realize(const SystemDescription& system,
                                                const HomotopyPoint& p, bool margins) const {
    if (!(p.aperture > 0.0) || !std::isfinite(p.aperture) || !(p.field >= 0.0) ||
        !std::isfinite(p.field) || !(p.ghost_weight > 0.0) || !(p.ghost_smoothing > 0.0) ||
        !(p.penalty_scale > 0.0) || !(p.extrema_beta >= 0.0) || !std::isfinite(p.extrema_beta) ||
        !(p.spectrum_width >= 0.0 && p.spectrum_width <= 1.0)) {
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
    if (options_.spectrum_continuous && p.spectrum_width < 1.0) {
        // stetige Spektrumbreite (M11-T5f): λ_ref + w·(λ_i − λ_ref); w = 0 monochromatisch
        if (p.spectrum_width == 0.0) {
            d.spectrum.wavelengths_nm = {d.spectrum.reference_nm};
            d.spectrum.weights = {1.0};
        } else {
            for (double& l : d.spectrum.wavelengths_nm) {
                l = d.spectrum.reference_nm + p.spectrum_width * (l - d.spectrum.reference_nm);
            }
        }
    }
    if (d.constraints && p.extrema_beta > 0.0) {
        // β-Fortsetzung der Strahl-Extrema (M11-T1b); bei β = 0 gilt die Angabe der Datei
        d.constraints->extrema_smoothing = ExtremaSmoothingSpec{p.extrema_beta, p.extrema_beta};
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
    SystemDescription* one = &system;
    return solve_all(std::span<SystemDescription* const>(&one, 1), p).at(0);
}

std::vector<HomotopySolve>
OpticalHomotopyModel::solve_all(std::span<SystemDescription* const> systems, const HomotopyPoint& p,
                                std::vector<int>* groups) {
    const std::size_t n = systems.size();
    std::vector<Prepared> prep;
    prep.reserve(n);
    for (SystemDescription* const s : systems) {
        prep.push_back(prepare(*this, *s, p));
    }
    // Gruppen: jedes System gehört zur ersten Gruppe, deren Leiter dieselbe Topologie hat
    std::vector<int> group(n, -1);
    std::vector<std::size_t> leaders;
    for (std::size_t i = 0; i < n; ++i) {
        if (options_.group_runs) {
            for (std::size_t k = 0; k < leaders.size(); ++k) {
                if (same_topology(prep[leaders[k]], prep[i])) {
                    group[i] = static_cast<int>(k);
                    break;
                }
            }
        }
        if (group[i] < 0) {
            group[i] = static_cast<int>(leaders.size());
            leaders.push_back(i);
        }
    }
    std::vector<HomotopySolve> out(n);
    for (std::size_t k = 0; k < leaders.size(); ++k) {
        const Prepared& lead = prep[leaders[k]];
        std::vector<std::size_t> members;
        std::vector<double> x0;
        for (std::size_t i = 0; i < n; ++i) {
            if (group[i] == static_cast<int>(k)) {
                members.push_back(i);
                x0.insert(x0.end(), prep[i].x0.begin(), prep[i].x0.end());
            }
        }
        GlassHull hull;
        const bool need_hull = has_glass_variables(lead.l.system) &&
                               has_constraint(lead.l.system, ConstraintType::GlassHull);
        if (need_hull) {
            hull = make_glass_hull(like_.catalog);
        }
        LmOptions o = options_.lm;
        Extras e = make_extras(lead.l, lead.base, lead.set, need_hull ? &hull : nullptr);
        o.jacobian.extra = std::move(e.extra);
        o.jacobian.trace_extra = std::move(e.traced);
        const LmResult res = options_.optimizer ? options_.optimizer(lead.set, lead.base, x0, o)
                                                : optimize_lm_cpu(lead.set, lead.base, x0, o);
        for (std::size_t j = 0; j < members.size(); ++j) {
            const std::size_t i = members[j];
            out[i] = finish_solve(prep[i], *systems[i], res.runs.at(j), o.jacobian.quadrature,
                                  options_.lm.threads);
        }
    }
    if (groups != nullptr) {
        *groups = std::move(group);
    }
    return out;
}

// Farbregel (M11-T5c): Operator B, wenn der Farblängsfehler dominiert; leer sonst.
std::optional<HomotopyNeed> OpticalHomotopyModel::color_need(const SystemDescription& system,
                                                             const HomotopyPoint& p) const {
    const SystemDescription realized = realize(system, p);
    if (realized.spectrum.wavelengths_nm.size() < 2 ||
        realized.source.type != SourceType::Collimated) {
        return std::nullopt;
    }
    // Farbe dominiert? RMS pro Feld und Wellenlänge der getracten Wellenfront
    double worst_ratio = 0.0;
    double worst_rms = 0.0;
    try {
        const LoadedSystem l = load(realized);
        const WavefrontResult wf = evaluate_wavefront(l);
        // Bezug: RMS bei λ_ref pro Feld. Liegt λ_ref nicht im Spektrum (z. B. Farbstufe der
        // Zielspezifikation mit 480/644 nm, λ_ref = 546 nm), wird monochromatisch bei λ_ref
        // nachgerechnet; der Vergleich mit der nächstgelegenen Wellenlänge sähe sonst keine Farbe.
        std::vector<double> ref_rms(static_cast<std::size_t>(wf.fields), 0.0);
        int lref = -1;
        for (int k = 0; k < wf.wavelengths; ++k) {
            if (std::abs(wf.at(0, k).wavelength_nm - realized.spectrum.reference_nm) < 1e-9) {
                lref = k;
            }
        }
        if (lref >= 0) {
            for (int f = 0; f < wf.fields; ++f) {
                ref_rms[static_cast<std::size_t>(f)] = wf.at(f, lref).rms_waves;
            }
        } else {
            SystemDescription mono = realized;
            mono.spectrum.wavelengths_nm = {realized.spectrum.reference_nm};
            mono.spectrum.weights = {1.0};
            const WavefrontResult wm = evaluate_wavefront(load(mono));
            for (int f = 0; f < wf.fields && f < wm.fields; ++f) {
                ref_rms[static_cast<std::size_t>(f)] = wm.at(f, 0).rms_waves;
            }
        }
        for (int f = 0; f < wf.fields; ++f) {
            const double ref = ref_rms[static_cast<std::size_t>(f)];
            for (int k = 0; k < wf.wavelengths; ++k) {
                const double r = wf.at(f, k).rms_waves;
                if (!std::isfinite(r) || !std::isfinite(ref)) {
                    continue;
                }
                worst_rms = std::max(worst_rms, r);
                worst_ratio = std::max(worst_ratio, ref > 0.0 ? r / ref : 0.0);
            }
        }
    } catch (const std::exception&) {
        return std::nullopt;
    }
    if (!(worst_rms >= options_.color_min_rms)) {
        return std::nullopt;
    }
    AberrationOptions ao;
    ao.real_rays = false;
    ao.focal_shift = false;
    AberrationReport rep;
    try {
        rep = compute_aberrations(realized, like_.catalog, ao);
    } catch (const std::exception&) {
        return std::nullopt;
    }
    const SeidelWaves w = rep.waves();
    const bool by_ratio = worst_ratio >= options_.color_ratio;
    const bool by_seidel =
        std::abs(w.dW020) >= options_.color_seidel && std::abs(w.dW020) >= std::abs(w.W040);
    // (c) Schritt der Spektrum-Homotopie (Stufe > 0): die Farbe ist per Konstruktion der Grund
    const bool by_stage = (p.spectrum > 0 && !options_.spectra.empty()) ||
                          (options_.spectrum_continuous && p.spectrum_width > 0.0);
    if (!by_ratio && !by_seidel && !by_stage) {
        return std::nullopt;
    }
    // Linsen nach ihrem Beitrag gleichen Vorzeichens zum Gesamtwert; die erste, die sich teilen
    // lässt (Katalogglas, Partner in der Vorauswahl, dick genug für zwei Teile)
    const double sign = w.dW020 >= 0.0 ? 1.0 : -1.0;
    std::vector<std::pair<double, int>> lenses;
    const auto& surf = realized.surfaces;
    for (std::size_t s = 1; s + 1 < surf.size(); ++s) {
        if (surf[s].material.kind != MaterialKind::Catalog &&
            surf[s].material.kind != MaterialKind::Buchdahl) {
            continue; // Katalogglas oder Glasvariable (Vorauswahl braucht die Dispersion)
        }
        const double v =
            sign * (rep.waves(rep.surfaces[s]).dW020 + rep.waves(rep.surfaces[s + 1]).dW020);
        if (v > 0.0) {
            lenses.emplace_back(v, static_cast<int>(s));
        }
    }
    // Einzellinsen (Luft vor und nach dem Glas) vor Teilen von Kittgliedern: sonst teilt die Regel
    // dasselbe Glied immer weiter (Zielspezifikation, M6-T6 v4: 14 → 15 → 16)
    const auto singlet = [&](int s) {
        return (s == 0 ||
                surf[static_cast<std::size_t>(s - 1)].material.kind == MaterialKind::Air) &&
               surf[static_cast<std::size_t>(s + 1)].material.kind == MaterialKind::Air;
    };
    std::stable_sort(lenses.begin(), lenses.end(), [&](const auto& a, const auto& b) {
        if (singlet(a.second) != singlet(b.second)) {
            return singlet(a.second);
        }
        return a.first > b.first;
    });
    int best = -1;
    std::vector<int> candidates;
    for (const auto& [value, lens] : lenses) {
        try {
            (void)split_for_color(realized, like_.catalog, lens, options_.split);
            if (best < 0) {
                best = lens;
            }
            candidates.push_back(lens);
        } catch (const std::exception&) {
            candidates.push_back(lens); // am guten Zustand vielleicht teilbar (andere Dicken)
        }
    }
    if (best < 0) {
        return std::nullopt;
    }
    HomotopyNeed n;
    n.recommendation = GhostOperator::B;
    n.split_lens = best;
    n.split_candidates = candidates;
    n.text = std::string("Farbregel (") +
             (by_stage   ? "Spektrumstufe "
              : by_ratio ? "RMS-Verhältnis "
                         : "Seidel ") +
             number(worst_ratio) + ", größter RMS " + number(worst_rms) +
             " Wellen, ΔW020 = " + number(w.dW020) + ", W040 = " + number(w.W040) +
             "): Operator B an der Linse ab Fläche " + std::to_string(best);
    return n;
}

HomotopyNeed OpticalHomotopyModel::diagnose(const SystemDescription& system,
                                            const HomotopyPoint& p) {
    if (options_.color_rule) {
        if (std::optional<HomotopyNeed> c = color_need(system, p)) {
            return *c;
        }
    }
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
    // Platz in der Flächentabelle (Dateiflächen + Geisterplatten ≤ kMaxSurfaces): Operator A und B
    // fügen je zwei Flächen hinzu (vorher Absturz von synth bei 59 Flächen + 6 Platten, M6-T6)
    if (static_cast<int>(system.surfaces.size() + system.ghost_plates.size()) + 2 > kMaxSurfaces) {
        invalid("Flächentabelle voll (" + std::to_string(system.surfaces.size()) + " Flächen und " +
                std::to_string(system.ghost_plates.size()) + " Geisterplatten, höchstens " +
                std::to_string(kMaxSurfaces) + " Tabellenflächen)");
    }
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
        MeniscusPlan plan = meniscus_from_diagnosis(realized, need.diagnosis, options_.meniscus);
        // Startkrümmungen innerhalb der Schranken des neuen Krümmungseintrags halten: c₀ kommt
        // aus einer vorhandenen Fläche (deren Schranken weiter sein können), die Sattelstarts
        // liegen bei c₀ ± δ. Ohne Begrenzung lehnt der Lader einen Start außerhalb der Schranken
        // ab (FormatError) und der Lauf bricht ab. Bei Dicke 0 ändert c₀ die Merit nicht.
        const NewCurvatureVariables& cv = options_.meniscus.curvature_variables;
        const auto clamp_new = [&cv](double c) {
            if (!cv.enabled) {
                return c;
            }
            if (cv.min) {
                c = std::max(c, *cv.min);
            }
            if (cv.max) {
                c = std::min(c, *cv.max);
            }
            return c;
        };
        {
            const double d = options_.saddle_delta;
            double c0 = clamp_new(plan.insertion.curvature);
            if (cv.enabled && d > 0.0 && cv.min && cv.max && *cv.max - *cv.min >= 2.0 * d) {
                c0 = std::clamp(c0, *cv.min + d, *cv.max - d);
            }
            plan.insertion.curvature = c0;
        }
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
            // Schranken enger als 2δ: die Sattelstarts selbst begrenzen
            for (SystemDescription& br : ins.branches) {
                for (int j = r.first_new_surface; j <= r.first_new_surface + 1; ++j) {
                    double& c = br.surfaces[static_cast<std::size_t>(j)].curvature;
                    c = clamp_new(c);
                }
            }
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
        SplitPlan plan;
        if (need.split_lens >= 0) {
            // Farbregel: empfohlene Linse, sonst die nächste Kandidatin, die sich an diesem
            // Zustand teilen lässt (die Empfehlung kann am verworfenen Versuch entstanden sein,
            // dessen Dicken sich unterscheiden)
            std::vector<int> order{need.split_lens};
            for (int c : need.split_candidates) {
                if (c != need.split_lens) {
                    order.push_back(c);
                }
            }
            std::string why;
            bool found = false;
            for (int lens : order) {
                try {
                    plan = split_for_color(realized, like_.catalog, lens, options_.split);
                    found = true;
                    break;
                } catch (const std::invalid_argument& e) {
                    if (why.empty()) {
                        why = e.what();
                    }
                }
            }
            if (!found) {
                invalid(why);
            }
        } else {
            plan = split_from_diagnosis(realized, like_.catalog, need.diagnosis, options_.split);
        }
        InsertionResult r = split_lens(system, plan.split);
        // Zweig „wie bisher“ und, mit achromat_start, ein zweiter Zweig mit Achromat-Start; der
        // Controller wählt den besseren (am Zielspezifikationslauf v5 brach der Achromat-Start
        // allein bei dicken Linsen und NA 0,6 die Strahlen, F bis 1e121)
        std::vector<std::pair<SystemDescription, bool>> variants;
        variants.emplace_back(r.system, false);
        if (options_.split.achromat_start && plan.preselected) {
            SystemDescription a = r.system;
            if (apply_achromat_start(a, plan, like_.catalog, realized.spectrum.reference_nm)) {
                variants.emplace_back(std::move(a), true);
            }
        }
        bool achromat = false;
        for (auto& [sys, with_achromat] : variants) {
            if ((options_.split.start_at_target || with_achromat) && plan.preselected) {
                free_glass(sys, plan.free_surface, plan.target_glass);
            } else {
                free_glass(sys, plan.free_surface);
            }
            if (!has_constraint(sys, ConstraintType::GlassHull)) {
                ConstraintSpec c;
                c.type = ConstraintType::GlassHull;
                if (!sys.constraints) {
                    sys.constraints = Constraints{};
                }
                sys.constraints->items.push_back(c);
            }
            (void)convert_glass_variables(sys, like_.catalog);
            if (plan.plate >= 0) {
                scale_ghost_penalty(sys, plan.plate, options_.penalty_raise);
            }
            achromat = achromat || with_achromat;
        }
        ins.first_surface = r.first_new_surface;
        ins.new_surfaces = r.new_surfaces;
        ins.plate = plan.plate;
        for (auto& v : variants) {
            ins.branches.push_back(std::move(v.first));
        }
        ins.text = "Operator " + std::string(to_string(need.recommendation)) +
                   ": Linse an Fläche " + std::to_string(plan.split.lens_surface) +
                   " geteilt, c_s = " + number(plan.split.curvature) + "/mm, Glas nach Fläche " +
                   std::to_string(plan.free_surface) + " frei" +
                   (plan.preselected ? ", Zielglas " + plan.target_glass.name +
                                           (achromat ? " (2 Zweige: wie bisher, Achromat-Start)"
                                            : options_.split.start_at_target ? " (Start)"
                                                                             : "")
                                     : std::string()) +
                   (plan.plate >= 0 ? " (Platte " + std::to_string(plan.plate) + ", mu x" +
                                          number(options_.penalty_raise) + ")"
                                    : std::string(" (Farbregel)"));
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

// ---------------------------------------------------------------------------------------------
// Strahlsuche (M6-T4)
// ---------------------------------------------------------------------------------------------

std::vector<HomotopyNeed> OpticalHomotopyModel::insertion_options(const SystemDescription&,
                                                                  const HomotopyPoint&,
                                                                  const HomotopyNeed& need,
                                                                  int max) {
    std::vector<HomotopyNeed> out{need};
    const GhostDiagnosis& d = need.diagnosis;
    const int rec = d.recommended_position;
    if (max <= 1 || need.recommendation == GhostOperator::None || rec < 0 ||
        rec >= static_cast<int>(d.positions.size())) {
        return out;
    }
    // Übrige Lagen mit Empfehlung nach Merit-Gewinn (ohne Merit-Werte: Σ|a|) absteigend
    std::vector<int> order;
    bool gains = true;
    for (int i = 0; i < static_cast<int>(d.positions.size()); ++i) {
        const GhostPositionReport& pr = d.positions[static_cast<std::size_t>(i)];
        if (i != rec && pr.recommendation != GhostOperator::None) {
            order.push_back(i);
            gains = gains && std::isfinite(pr.merit_gain);
        }
    }
    const auto score = [&](int i) {
        const GhostPositionReport& pr = d.positions[static_cast<std::size_t>(i)];
        return gains ? pr.merit_gain : pr.sum_abs;
    };
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return score(a) > score(b); });
    const auto option = [&](int pos, int plate) {
        HomotopyNeed n = need;
        GhostPositionReport& pr = n.diagnosis.positions[static_cast<std::size_t>(pos)];
        n.diagnosis.recommended_position = pos;
        n.diagnosis.recommendation = pr.recommendation;
        n.recommendation = pr.recommendation;
        if (plate >= 0) {
            pr.active_plates = {plate};
        }
        n.text = "Variante: Operator " + std::string(to_string(pr.recommendation)) +
                 " nach Fläche " + std::to_string(pr.after_surface) + " bei " +
                 number(pr.distance) + " mm" +
                 (plate >= 0 ? ", Glas der Platte " + std::to_string(plate) : std::string()) +
                 " (" + pr.finding + ")";
        return n;
    };
    // 1. Standardwahl der übrigen Lagen
    for (const int pos : order) {
        if (static_cast<int>(out.size()) >= max) {
            return out;
        }
        out.push_back(option(pos, -1));
    }
    // 2. Andere Gläser derselben Lage (nur Operator A: Glas der aktivsten Platte): die übrigen
    //    aktiven Platten nach Σ|a| absteigend
    std::vector<int> all{rec};
    all.insert(all.end(), order.begin(), order.end());
    for (const int pos : all) {
        const GhostPositionReport& pr = d.positions[static_cast<std::size_t>(pos)];
        if (pr.recommendation != GhostOperator::A || pr.active_plates.size() < 2) {
            continue;
        }
        const auto sum = [&](int g) { return d.plates[static_cast<std::size_t>(g)].sum_abs; };
        int best = pr.active_plates.front(); // wie meniscus_from_diagnosis
        for (const int g : pr.active_plates) {
            if (sum(g) > sum(best)) {
                best = g;
            }
        }
        std::vector<int> others;
        for (const int g : pr.active_plates) {
            if (g != best) {
                others.push_back(g);
            }
        }
        std::stable_sort(others.begin(), others.end(),
                         [&](int a, int b) { return sum(a) > sum(b); });
        for (const int g : others) {
            if (static_cast<int>(out.size()) >= max) {
                return out;
            }
            out.push_back(option(pos, g));
        }
    }
    return out;
}

double OpticalHomotopyModel::distance(const SystemDescription& a, const SystemDescription& b,
                                      const HomotopyPoint& p) {
    if (a.surfaces.size() != b.surfaces.size() || a.ghost_plates.size() != b.ghost_plates.size()) {
        return std::numeric_limits<double>::infinity();
    }
    const Prepared pa = prepare(*this, a, p);
    const Prepared pb = prepare(*this, b, p);
    if (!same_topology(pa, pb)) {
        return std::numeric_limits<double>::infinity();
    }
    double d = 0.0;
    const std::span<const double> typ = pa.set.typical();
    for (std::size_t k = 0; k < pa.x0.size(); ++k) {
        const double scale = std::max({std::abs(pa.x0[k]), std::abs(pb.x0[k]), typ[k]});
        d = std::max(d, std::abs(pa.x0[k] - pb.x0[k]) / scale);
    }
    return d;
}

std::optional<SystemDescription> OpticalHomotopyModel::predict(const SystemDescription& previous,
                                                               const SystemDescription& current,
                                                               const HomotopyPoint& p,
                                                               double ratio) {
    if (!std::isfinite(ratio) || previous.surfaces.size() != current.surfaces.size() ||
        previous.ghost_plates.size() != current.ghost_plates.size()) {
        return std::nullopt;
    }
    try {
        const Prepared pa = prepare(*this, previous, p);
        const Prepared pc = prepare(*this, current, p);
        if (!same_topology(pa, pc)) {
            return std::nullopt;
        }
        std::vector<double> x(pc.x0.size());
        for (std::size_t k = 0; k < x.size(); ++k) {
            x[k] = pc.x0[k] + ratio * (pc.x0[k] - pa.x0[k]);
        }
        pc.set.clamp_to_bounds(x);
        SystemDescription out = current;
        pc.set.apply_to_description(out, x);
        return out;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::vector<SystemDescription>
OpticalHomotopyModel::perturbed_starts(const SystemDescription& start, const HomotopyInsertion& ins,
                                       const HomotopyPoint& p, int count, std::uint64_t seed) {
    std::vector<SystemDescription> out;
    if (count <= 0 || ins.new_surfaces <= 0 || !(options_.start_perturbation > 0.0)) {
        return out;
    }
    // Schranken der Krümmungsvariablen der neuen Flächen (falls vorhanden)
    const Prepared pr = prepare(*this, start, p);
    const auto first = static_cast<std::size_t>(ins.first_surface);
    const auto last = first + static_cast<std::size_t>(ins.new_surfaces);
    std::vector<double> lower(last - first, -std::numeric_limits<double>::infinity());
    std::vector<double> upper(last - first, std::numeric_limits<double>::infinity());
    for (int k = 0; k < pr.set.size(); ++k) {
        const Variable& v = pr.set[k];
        const auto f = static_cast<std::size_t>(v.file_surface);
        if (v.type == VariableType::Curvature && f >= first && f < last) {
            lower[f - first] = v.lower;
            upper[f - first] = v.upper;
        }
    }
    for (int j = 1; j <= count; ++j) {
        std::mt19937_64 rng(variant_seed(seed, static_cast<std::uint64_t>(j)));
        SystemDescription d = start;
        for (std::size_t f = first; f < last; ++f) {
            const double xi = 2.0 * unit_uniform(rng()) - 1.0;
            double& c = d.surfaces.at(f).curvature;
            c = std::clamp(std::fma(options_.start_perturbation, xi, c), lower[f - first],
                           upper[f - first]);
        }
        out.push_back(std::move(d));
    }
    return out;
}

std::vector<bool> OpticalHomotopyModel::prefilter(std::span<const SystemDescription* const> starts,
                                                  const HomotopyPoint& p) {
    const std::size_t n = starts.size();
    std::vector<bool> keep(n, true);
    std::vector<Prepared> prep;
    prep.reserve(n);
    for (const SystemDescription* s : starts) {
        prep.push_back(prepare(*this, *s, p));
    }
    std::vector<int> group(n, -1);
    std::vector<std::size_t> leaders;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < leaders.size(); ++k) {
            if (same_topology(prep[leaders[k]], prep[i])) {
                group[i] = static_cast<int>(k);
                break;
            }
        }
        if (group[i] < 0) {
            group[i] = static_cast<int>(leaders.size());
            leaders.push_back(i);
        }
    }
    const ResidualSettings& quadrature = options_.lm.jacobian.quadrature;
    for (std::size_t k = 0; k < leaders.size(); ++k) {
        const Prepared& lead = prep[leaders[k]];
        std::vector<std::size_t> members;
        std::vector<int> runs;
        std::vector<double> xs;
        for (std::size_t i = 0; i < n; ++i) {
            if (group[i] == static_cast<int>(k)) {
                runs.push_back(static_cast<int>(members.size()));
                members.push_back(i);
                xs.insert(xs.end(), prep[i].x0.begin(), prep[i].x0.end());
            }
        }
        const SystemBatchHost batch = lm_point_batch(lead.set, lead.base, runs, xs);
        // Zusatzresiduen in FP64 (ohne die trace-abhängigen: L_v bleibt eine untere Schranke)
        GlassHull hull;
        const bool need_hull = has_glass_variables(lead.l.system) &&
                               has_constraint(lead.l.system, ConstraintType::GlassHull);
        if (need_hull) {
            hull = make_glass_hull(like_.catalog);
        }
        const Extras e = make_extras(lead.l, lead.base, lead.set, need_hull ? &hull : nullptr);
        const std::vector<double> extra =
            extra_merit_values(e.extra, batch, options_.lm.jacobian.extra_threads);
        const PrefilterResult res =
            options_.prefilter_fn
                ? options_.prefilter_fn(batch, quadrature, options_.prefilter, extra)
                : prefilter_cpu(batch, ResidualLayout(batch.topology(), quadrature),
                                options_.prefilter, extra);
        for (std::size_t j = 0; j < members.size(); ++j) {
            keep[members[j]] = res.keep(static_cast<int>(j));
        }
    }
    return keep;
}

std::vector<Spectrum> spectrum_stages(const Spectrum& target, int n) {
    if (n < 1) {
        invalid("spectrum_stages: mindestens eine Stufe");
    }
    std::vector<Spectrum> out;
    Spectrum mono;
    mono.wavelengths_nm = {target.reference_nm};
    mono.weights = {1.0};
    mono.reference_nm = target.reference_nm;
    out.push_back(mono);
    for (int k = 1; k <= n; ++k) {
        Spectrum st = target;
        const double t = static_cast<double>(k) / n;
        for (double& l : st.wavelengths_nm) {
            l = target.reference_nm + t * (l - target.reference_nm);
        }
        out.push_back(st);
    }
    return out;
}

} // namespace mko
