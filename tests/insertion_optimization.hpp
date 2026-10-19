#pragma once

// Optimierung nach den Einfüge-Operatoren (M6-T2, CPU und GPU):
//   B  Einzellinse bei zwei Wellenlängen → Kittglied, Glas des hinteren Teils frei (Hülle als
//      Strafterm); Abnahme: Farblängsfehler um mehr als den Faktor 10 kleiner.
//   A  Einzellinse NA 0,5 mit Geisterplatten → Diagnose → Meniskus aus dem Bericht, Strafgewicht
//      der Platte erhöht, Dicke des Meniskus in Stufen geöffnet (einfache Homotopie im Test).
// docs/optimierung.md, Abschnitt 6.

#include "ghost_test_systems.hpp"
#include "insertion_test_systems.hpp"
#include "mikroptim/glass/glass_hull.hpp"
#include "mikroptim/optim/glass_variables.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mko::test {

// Optimierer: LmResult optimize(const VariableSet&, const SystemBatchHost& base,
//                               std::span<const double> x0, const LmOptions&) mit einem Lauf.

// ---------------------------------------------------------------------------------------------
// Operator B: Achromat aus der Einzellinse

// Einzellinse: Zyklen bis zur Konvergenz. Kittglied: Penalty-Fortsetzung über das Gewicht der
// Glas-Hülle wie in M6-T1, w_s = w_Datei · 10^(s − kAchromatStages + 1), jede Stufe ab dem
// Ergebnis der vorigen. N-BK7 liegt auf dem Rand der Hülle; mit dem vollen Gewicht von Anfang an
// bleibt das Glas an dieser Ecke hängen. Die Zyklen pro Stufe sind begrenzt: Nach dem
// Achromatisieren kriecht LM im flachen Tal der Achromate (Farbe ≈ 0, nur noch sphärische
// Aberration bei NA 0,05) und konvergiert erst nach vielen tausend Zyklen.
inline constexpr int kAchromatIterations = 1000;
inline constexpr int kAchromatStages = 7;
inline constexpr int kAchromatSplitIterations = 300;
// Abnahme: Farblängsfehler nach dem Operator B kleiner als dieser Anteil des Ausgangswerts
// (Plan: mehr als Faktor 10).
inline constexpr double kAchromatColorFactor = 0.1;

// Farblängsfehler: paraxiale hintere Schnittweite bei der kürzesten minus der längsten
// Wellenlänge des Spektrums [mm].
inline double color_focal_shift(const SystemDescription& d, const GlassCatalog& catalog) {
    const double s = d.spectrum.wavelengths_nm.front();
    const double l = d.spectrum.wavelengths_nm.back();
    return compute_paraxial(d, catalog, s).back_focal_distance -
           compute_paraxial(d, catalog, l).back_focal_distance;
}

// Index der ersten Glasvariable (n₀) in der Menge.
inline int glass_variable(const VariableSet& set) {
    for (int k = 0; k < set.size(); ++k) {
        if (set[k].type == VariableType::Glass && set[k].glass_component == 0) {
            return k;
        }
    }
    throw std::logic_error("keine Glasvariable");
}

struct AchromatRun {
    LmRunResult lens;             // Einzellinse allein (c₁, c₂, w)
    SystemDescription lens_opt;   // ihr Optimum
    double color_lens = 0.0;      // Farblängsfehler der Einzellinse [mm]
    double focal_lens = 0.0;      // Brennweite bei λ_ref
    double merit_lens = 0.0;      // F am Optimum der Einzellinse
    double merit_split = 0.0;     // F nach dem Operator B (beide Teile N-BK7)
    double merit_converted = 0.0; // F nach der Umwandlung des freien Glases in Buchdahl
    std::string target;           // Zielglas (kleinste Abbezahl der Hülle)
    double abbe_lens = 0.0;
    double abbe_target = 0.0;
    double cs = 0.0; // Kittfläche
    // Kittglied mit freiem Glas: Stufen der Fortsetzung (Gewicht, Lauf, Farblängsfehler am Ende)
    std::vector<double> weights;
    std::vector<LmRunResult> stages;
    std::vector<double> stage_color;
    // Farblängsfehler in der ersten Stufe alle 50 Zyklen (Zyklus, Wert [mm])
    std::vector<std::pair<int, double>> color_history;
    SystemDescription final;
    double color_final = 0.0;
    double focal_final = 0.0;
    Point3 glass{};
    double abbe_final = 0.0;
    double hull_distance = 0.0;
    std::string nearest; // nächstes Glas der Hülle (normiert)
};

template <class Optimize>
AchromatRun achromat_run(Optimize&& optimize) {
    AchromatRun a;
    const LoadedSystem raw = load_residual_system("einzellinse_achromat_start.json");
    const FdProblem p = make_fd_problem(raw);
    LmOptions o;
    o.max_iterations = kAchromatIterations;
    o.jacobian.extra = make_paraxial_residuals(make_paraxial_constraints(p.loaded.system));
    a.lens = optimize(p.set, p.base, std::span<const double>(p.x0), o).runs.at(0);
    a.lens_opt = p.loaded.system;
    p.set.apply_to_description(a.lens_opt, a.lens.x);
    a.color_lens = color_focal_shift(a.lens_opt, raw.catalog);
    a.focal_lens = compute_paraxial(a.lens_opt, raw.catalog).focal_length;
    a.merit_lens = system_merit(a.lens_opt);

    // Zielglas für die Kittfläche: kleinste Abbezahl unter den Gläsern der Hülle
    const GlassHull hull = make_glass_hull(raw.catalog);
    a.abbe_target = std::numeric_limits<double>::infinity();
    for (const GlassHullPoint& g : hull.points()) {
        const double abbe = abbe_number(catalog_glass(g.name.c_str()), raw.catalog);
        if (abbe < a.abbe_target) {
            a.abbe_target = abbe;
            a.target = g.name;
        }
    }
    a.abbe_lens = abbe_number(a.lens_opt.surfaces[1].material, raw.catalog);
    a.cs = achromat_split_curvature(a.lens_opt.surfaces[1].curvature,
                                    a.lens_opt.surfaces[2].curvature, a.abbe_lens, a.abbe_target);
    LensSplit split = lens_split(1, a.cs, 0.5);
    split.curvature_variable.min = -0.2;
    split.curvature_variable.max = 0.2;
    InsertionResult r = split_lens(a.lens_opt, split);
    a.merit_split = system_merit(r.system);

    // Glas des hinteren Teils freigeben (Start N-BK7, Buchdahl-Fit), Hülle als Strafterm
    free_glass(r.system, 2);
    SystemDescription d = r.system;
    (void)convert_glass_variables(d, raw.catalog);
    const FdProblem q = make_fd_problem(prepare_test_system(d));
    a.merit_converted = system_merit(q.loaded.system);
    const auto color_at = [&](std::span<const double> xi) {
        SystemDescription di = q.loaded.system;
        q.set.apply_to_description(di, xi);
        return color_focal_shift(di, raw.catalog);
    };
    std::size_t hull_item = 0;
    while (q.loaded.system.constraints->items[hull_item].type != ConstraintType::GlassHull) {
        ++hull_item;
    }
    const double w_file = q.loaded.system.constraints->items[hull_item].weight_or_default();
    std::vector<double> x = q.x0;
    for (int s = 0; s < kAchromatStages; ++s) {
        const double w = w_file * std::pow(10.0, s - kAchromatStages + 1);
        SystemDescription dw = q.loaded.system;
        dw.constraints->items[hull_item].weight = w;
        LmOptions so;
        so.max_iterations = kAchromatSplitIterations;
        so.jacobian.extra = concat_extra_residuals(
            {make_paraxial_residuals(make_paraxial_constraints(q.loaded.system)),
             make_glass_hull_residuals(make_glass_hull_penalty(dw, q.set, hull))});
        if (s == 0) {
            so.on_iteration = [&](const LmIterationInfo& info) {
                if (info.iteration % 50 == 0) {
                    a.color_history.emplace_back(info.iteration, color_at(info.x));
                }
            };
        }
        a.weights.push_back(w);
        a.stages.push_back(optimize(q.set, q.base, std::span<const double>(x), so).runs.at(0));
        x = a.stages.back().x;
        a.stage_color.push_back(color_at(x));
    }
    a.final = q.loaded.system;
    q.set.apply_to_description(a.final, x);
    a.color_final = color_focal_shift(a.final, raw.catalog);
    a.focal_final = compute_paraxial(a.final, raw.catalog).focal_length;
    a.abbe_final = abbe_number(a.final.surfaces[2].material, raw.catalog);
    SystemBatchHost b = q.base;
    q.set.apply(b, 0, x);
    a.glass = batch_glass_coordinates(b, 0, q.set[glass_variable(q.set)].batch_surface);
    a.hull_distance = hull.distance(a.glass);
    double best = std::numeric_limits<double>::infinity();
    const Point3 u = hull.normalize(a.glass);
    for (const GlassHullPoint& g : hull.points()) {
        const Point3 v = hull.normalize(g.coordinates);
        const double dist = std::hypot(u[0] - v[0], u[1] - v[1], u[2] - v[2]);
        if (dist < best) {
            best = dist;
            a.nearest = g.name;
        }
    }
    return a;
}

// ---------------------------------------------------------------------------------------------
// Operator A: Meniskus aus der Diagnose

// Dicken des Meniskus in der Homotopie [mm] (Spalt hinter der Platte 0,2 mm) und Faktor, um den
// das Strafgewicht der Quellplatte erhöht wird.
inline const std::vector<double> kMeniscusThicknesses{0.01, 0.02, 0.05, 0.1, 0.15};
inline constexpr double kMeniscusPenaltyFactor = 10.0;
inline constexpr int kMeniscusIterations = 1000;
// Erwartung nach der Homotopie: Σ|a| der Quellplatte unter diesem Anteil des Werts vor dem
// Einfügen (die Korrektur ist in das reale Element gewandert; gemessen ~3e-5).
inline constexpr double kMeniscusPlateShare = 0.01;

struct MeniscusStep {
    double thickness = 0.0;
    LmRunResult run;
    double rms = 0.0;        // Wellenfront-RMS [Wellen]
    double plate_sum = 0.0;  // Σ|a| der Quellplatte [mm]
    double others_sum = 0.0; // Σ|a| aller übrigen Platten [mm]
    double c_front = 0.0;    // Krümmungen des Meniskus
    double c_back = 0.0;
};

struct MeniscusRun {
    LmRunResult ghosts; // Lauf mit Geisterplatten (wie M4-T6, Abnahme Bedarf)
    double rms_ghosts = 0.0;
    double plate_sum_ghosts = 0.0;
    GhostDiagnosis diagnosis;
    MeniscusPlan plan;
    double merit_before = 0.0;   // F am Optimum mit Geisterplatten (inkl. Strafe)
    double merit_inserted = 0.0; // nach dem Operator A
    std::vector<MeniscusStep> steps;
};

inline double plate_abs_sum(const SystemDescription& d, int plate) {
    double s = 0.0;
    for (const double a : d.ghost_plates[static_cast<std::size_t>(plate)].coefficients) {
        s += std::abs(a);
    }
    return s;
}

// sign = +1: c₀ aus dem Plan, −1: Gegenrichtung (zweiter Zweig des Sattels).
template <class Optimize>
MeniscusRun meniscus_run(Optimize&& optimize, double sign = 1.0) {
    MeniscusRun m;
    const FdProblem p = ghost_need_problem();
    const GhostPenalty pen = ghost_penalty_of(p.loaded.system);
    LmOptions o;
    o.max_iterations = kMeniscusIterations;
    o.jacobian.extra = ghost_extra(p.loaded.system, pen);
    m.ghosts = optimize(p.set, p.base, std::span<const double>(p.x0), o).runs.at(0);
    m.rms_ghosts = wavefront_rms(p, m.ghosts.x);
    const int run0 = 0;
    const SystemBatchHost b =
        lm_point_batch(p.set, p.base, std::span<const int>(&run0, 1), m.ghosts.x);
    m.diagnosis = diagnose_ghost_plates(p.loaded.system, p.loaded.catalog, b, 0, pen,
                                        make_cpu_merit({}, o.jacobian.extra));
    SystemDescription d = p.loaded.system;
    p.set.apply_to_description(d, m.ghosts.x);
    MeniscusInitOptions io;
    io.curvature_variables.min = -2.0;
    io.curvature_variables.max = 2.0;
    m.plan = meniscus_from_diagnosis(d, m.diagnosis, io);
    m.plan.insertion.curvature *= sign;
    m.plate_sum_ghosts = plate_abs_sum(d, m.plan.plate);
    m.merit_before = system_merit(d);
    const InsertionResult r = insert_meniscus(d, m.plan.insertion);
    m.merit_inserted = system_merit(r.system);

    // Strafgewicht der Quellplatte erhöhen, Dicke des Meniskus in Stufen öffnen; der Luftspalt
    // dahinter wird um dieselbe Dicke kürzer (Lage der folgenden Flächen fest)
    SystemDescription cur = r.system;
    scale_ghost_penalty(cur, m.plan.plate, kMeniscusPenaltyFactor);
    const int f = r.first_new_surface;
    const double gap = cur.surfaces[static_cast<std::size_t>(f + 1)].thickness;
    for (const double t : kMeniscusThicknesses) {
        MeniscusStep step;
        step.thickness = t;
        cur.surfaces[static_cast<std::size_t>(f)].thickness = t;
        cur.surfaces[static_cast<std::size_t>(f + 1)].thickness = gap - t;
        const FdProblem q = make_fd_problem(prepare_test_system(cur));
        LmOptions so;
        so.max_iterations = kMeniscusIterations;
        so.jacobian.extra = insertion_extra(q.loaded.system);
        so.jacobian.trace_extra = insertion_trace_extra(q.loaded.system);
        step.run = optimize(q.set, q.base, std::span<const double>(q.x0), so).runs.at(0);
        q.set.apply_to_description(cur, step.run.x);
        step.rms = wavefront_rms(q, step.run.x);
        step.plate_sum = plate_abs_sum(cur, m.plan.plate);
        for (int g = 0; g < static_cast<int>(cur.ghost_plates.size()); ++g) {
            if (g != m.plan.plate) {
                step.others_sum += plate_abs_sum(cur, g);
            }
        }
        step.c_front = cur.surfaces[static_cast<std::size_t>(f)].curvature;
        step.c_back = cur.surfaces[static_cast<std::size_t>(f + 1)].curvature;
        m.steps.push_back(step);
    }
    return m;
}

// ---------------------------------------------------------------------------------------------
// Ausgabe und Prüfungen (CPU und GPU)

inline void print_achromat(const char* tag, const AchromatRun& a) {
    std::printf("[%s] Einzellinse: F %.6e -> %.10e, %d Zyklen, %s; f = %.8f mm, "
                "Farblängsfehler %.6e mm\n",
                tag, a.lens.history.front(), a.lens.merit, a.lens.iterations,
                to_string(a.lens.status), a.focal_lens, a.color_lens);
    std::printf("[%s] Operator B: Zielglas %s (nu_d %.2f, Linse %.2f), c_s = %.6f; F %.15e -> "
                "%.15e (|dF|/F = %.2e), nach Buchdahl-Umwandlung %.15e\n",
                tag, a.target.c_str(), a.abbe_target, a.abbe_lens, a.cs, a.merit_lens,
                a.merit_split, relative_change(a.merit_lens, a.merit_split), a.merit_converted);
    for (const auto& [cycle, color] : a.color_history) {
        std::printf("[%s]   Stufe 1, Zyklus %4d: Farblängsfehler %+.6e mm\n", tag, cycle, color);
    }
    for (std::size_t s = 0; s < a.stages.size(); ++s) {
        const LmRunResult& r = a.stages[s];
        std::printf("[%s]   w = %8.1e: F %.6e -> %.10e, %4d Zyklen, %-16s Farblängsfehler "
                    "%+.6e mm\n",
                    tag, a.weights[s], r.history.front(), r.merit, r.iterations,
                    to_string(r.status), a.stage_color[s]);
    }
    std::printf("[%s] Kittglied: c = %.6f / %.6f / %.6f, w = %.4f; Glas n0 = %.6f nu1 = %.5e nu2 = "
                "%.5e (nu_d %.2f, Abstand zur Hülle %.2e, nächstes Glas %s); f = %.8f mm, "
                "Farblängsfehler %.6e mm (Faktor %.1f kleiner)\n",
                tag, a.final.surfaces[1].curvature, a.final.surfaces[2].curvature,
                a.final.surfaces[3].curvature, a.final.surfaces[3].thickness, a.glass[0],
                a.glass[1], a.glass[2], a.abbe_final, a.hull_distance, a.nearest.c_str(),
                a.focal_final, a.color_final, std::abs(a.color_lens / a.color_final));
}

inline void check_achromat(const AchromatRun& a) {
    EXPECT_TRUE(lm_converged(a.lens.status)) << to_string(a.lens.status);
    EXPECT_EQ(a.lens.failed_rays, 0);
    EXPECT_GT(std::abs(a.color_lens), 0.5); // Einzellinse: f/ν ≈ 0,8 mm
    EXPECT_LE(relative_change(a.merit_lens, a.merit_split), kInsertionMeritRelative);
    for (const LmRunResult& r : a.stages) {
        EXPECT_EQ(r.failed_rays, 0);
        EXPECT_NE(r.status, LmStatus::NumericalError);
    }
    EXPECT_LT(a.stages.back().merit, 1e-2 * a.merit_lens);
    // Abnahme: Farblängsfehler um mehr als den Faktor 10 kleiner, Brennweite gehalten, Glas in
    // der Hülle (Restverletzung der Strafe)
    EXPECT_LT(std::abs(a.color_final), kAchromatColorFactor * std::abs(a.color_lens));
    EXPECT_LE(std::abs(a.focal_final - a.focal_lens), kAchromatFocalLength);
    EXPECT_LE(a.hull_distance, kGlassHullPenaltyDistance);
}

inline void print_meniscus(const char* tag, const MeniscusRun& m) {
    std::printf("[%s] mit Geisterplatten: F -> %.10e, %d Zyklen, %s, RMS %.4e Wellen, Platte %d "
                "Sum|a| = %.4e mm\n",
                tag, m.ghosts.merit, m.ghosts.iterations, to_string(m.ghosts.status), m.rms_ghosts,
                m.plan.plate, m.plate_sum_ghosts);
    std::printf("[%s] %s\n", tag, m.plan.reason.c_str());
    std::printf("[%s] Operator A: F %.15e -> %.15e (|dF|/F = %.2e)\n", tag, m.merit_before,
                m.merit_inserted, relative_change(m.merit_before, m.merit_inserted));
    for (const MeniscusStep& s : m.steps) {
        std::printf("[%s]   Dicke %.3f mm: F -> %.8e, %4d Zyklen, %-16s RMS %.4e Wellen, Platte "
                    "Sum|a| %.3e mm, übrige %.3e mm, c = %+.5f / %+.5f\n",
                    tag, s.thickness, s.run.merit, s.run.iterations, to_string(s.run.status), s.rms,
                    s.plate_sum, s.others_sum, s.c_front, s.c_back);
    }
}

// Ablauf A: Bericht empfiehlt A an der aktivsten Platte (Flint hinter der Linse, M4-T6), das
// Einfügen ändert die Merit nicht, nach dem Öffnen der Dicke trägt das reale Element die
// Korrektur: Koeffizienten der Quellplatte praktisch 0, Wellenfront besser als mit der Platte.
inline void check_meniscus(const MeniscusRun& m) {
    EXPECT_TRUE(lm_converged(m.ghosts.status)) << to_string(m.ghosts.status);
    EXPECT_EQ(m.diagnosis.recommendation, GhostOperator::A);
    EXPECT_EQ(m.plan.plate, 8);
    EXPECT_EQ(m.plan.insertion.after_surface, 2);
    EXPECT_LE(relative_change(m.merit_before, m.merit_inserted), kInsertionMeritRelative);
    ASSERT_EQ(m.steps.size(), kMeniscusThicknesses.size());
    for (const MeniscusStep& s : m.steps) {
        EXPECT_TRUE(lm_converged(s.run.status)) << s.thickness << ": " << to_string(s.run.status);
        EXPECT_EQ(s.run.failed_rays, 0);
    }
    const MeniscusStep& last = m.steps.back();
    EXPECT_LT(last.plate_sum, kMeniscusPlateShare * m.plate_sum_ghosts);
    EXPECT_LT(last.rms, m.rms_ghosts);
}

} // namespace mko::test
