// Augmented Lagrangian für einseitige Bedingungen (M11-T2): Fortschreibung und Schranken der
// Verschiebung, Residuen mit Verschiebung (ohne Verschiebung bitgleich), Fortschreibung an
// paraxialen und mechanischen Bedingungen und die Abnahme: Petzval (M4-T4) und Mechanik (M4-T3)
// mit festem, moderatem Gewicht unter den heutigen Toleranzen, mit nicht mehr LM-Zyklen als die
// Penalty-Fortsetzung (docs/optimierung.md, Abschnitt 2.6).

#include "augmented_lagrangian_test_systems.hpp"
#include "tolerances.hpp"

#include "mikroptim/model/format_error.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace mko::test {
namespace {

TEST(AugmentedLagrangian, UpdateShiftFormulaAndBounds) {
    // θ' = min(max(0, θ + g), θ_max/√w)
    EXPECT_EQ(update_shift(0.0, 0.25, 4.0, 1e3), 0.25);
    EXPECT_EQ(update_shift(0.5, -0.25, 4.0, 1e3), 0.25);
    EXPECT_EQ(update_shift(0.5, -1.0, 4.0, 1e3), 0.0); // inaktiv: θ fällt auf 0
    EXPECT_EQ(update_shift(0.0, 10.0, 4.0, 3.0), 1.5); // Schranke √w θ ≤ 3
    EXPECT_EQ(update_shift(1.0, 1.0, 0.0, 1e3), 0.0);  // w = 0: keine Verschiebung
    EXPECT_EQ(update_shift(0.0, 0.0, 4.0, 1e3), 0.0);
}

TEST(AugmentedLagrangian, ShiftsAreValidatedAndOnlyForOneSidedConstraints) {
    const FdProblem p = mechanical_problem();
    SystemDescription d = p.loaded.system;
    std::size_t focal = 0;
    std::size_t mech = 0;
    for (std::size_t i = 0; i < d.constraints->items.size(); ++i) {
        if (d.constraints->items[i].type == ConstraintType::FocalLength) {
            focal = i;
        } else if (is_mechanical(d.constraints->items[i].type)) {
            mech = i;
        }
    }
    EXPECT_FALSE(supports_shift(d.constraints->items[focal]));
    EXPECT_TRUE(supports_shift(d.constraints->items[mech]));
    EXPECT_TRUE(has_shiftable_constraints(d));

    SystemDescription bad = d;
    bad.constraints->items[focal].shift = {0.1};
    EXPECT_THROW(check_constraints(bad), FormatError);
    bad = d;
    bad.constraints->items[mech].shift = {-1.0};
    EXPECT_THROW(check_constraints(bad), FormatError);
    bad = d;
    bad.constraints->items[mech].shift = {std::nan("")};
    EXPECT_THROW(check_constraints(bad), FormatError);

    // Länge passt nicht zur Zahl der Flächen: alle Verschiebungen des Eintrags 0
    bad = d;
    bad.constraints->items[mech].shift = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    for (const MechanicalConstraint& c : make_mechanical_constraints(bad).items) {
        EXPECT_EQ(c.shift, 0.0);
    }
}

// Residuen mit Verschiebung: √w · max(0, g + θ) (paraxial mit dem Vorzeichen des Modus); ohne
// Verschiebung bitgleich zur reinen Strafe.
TEST(AugmentedLagrangian, ResidualsWithShift) {
    const FdProblem p = mechanical_problem();
    const MechanicalConstraints mech = make_mechanical_constraints(p.loaded.system);
    const SurfaceExtremaResult e = surface_extrema_cpu(
        p.base, 0, ResidualLayout(p.base.topology(), LmOptions{}.jacobian.quadrature));
    const VariantExtrema ve = variant_extrema(e, 0);
    for (const MechanicalConstraint& c0 : mech.items) {
        MechanicalConstraint c = c0;
        const double g = mechanical_violation(c, mechanical_value(c, p.base, 0, &ve));
        const double r0 = mechanical_residual(c, p.base, 0, &ve);
        EXPECT_EQ(std::bit_cast<std::uint64_t>(r0),
                  std::bit_cast<std::uint64_t>(std::sqrt(c.weight) * std::max(g, 0.0)));
        c.shift = 0.125;
        EXPECT_EQ(mechanical_residual(c, p.base, 0, &ve),
                  std::sqrt(c.weight) * std::max(g + 0.125, 0.0));
    }

    const FdProblem pz = petzval_problem();
    ParaxialConstraints pc = make_paraxial_constraints(pz.loaded.system);
    const ParaxialData pd = batch_paraxial(pz.base, 0);
    for (ParaxialConstraint& c : pc.items) {
        const double q = paraxial_deviation(c, pz.base, 0, pd);
        const double r0 = paraxial_residual(c, pz.base, 0, pd);
        if (c.mode == ConstraintMode::Equal) {
            EXPECT_EQ(r0, std::sqrt(c.weight) * q);
            c.shift = 0.5;
            EXPECT_THROW(check_paraxial_constraints(pc, pz.base.topology()), std::invalid_argument);
            c.shift = 0.0;
            continue;
        }
        // Petzval: Modus max, g = q = |κ_P| − κ_max
        EXPECT_EQ(paraxial_violation(c, q), q);
        c.shift = 1e-3;
        EXPECT_EQ(paraxial_residual(c, pz.base, 0, pd),
                  std::sqrt(c.weight) * std::max(q + 1e-3, 0.0));
        // Untergrenze: Vorzeichen des Modus bleibt (r ≤ 0)
        ParaxialConstraint m = c;
        m.mode = ConstraintMode::Min;
        m.shift = 0.0;
        EXPECT_EQ(paraxial_violation(m, q), -q);
        m.shift = 2.0 * std::abs(q) + 1e-3;
        EXPECT_EQ(paraxial_residual(m, pz.base, 0, pd),
                  std::sqrt(c.weight) * std::min(q - m.shift, 0.0));
    }
}

// Fortschreibung an allen Bedingungen des Mechanik-Systems: Verletzungen wie die Auswertung,
// θ' = max(0, θ + g), Einträge zurück in die Beschreibung, dort wirksam.
TEST(AugmentedLagrangian, ShiftUpdateMatchesConstraintValues) {
    const FdProblem p = mechanical_problem();
    const MechanicalState s = mechanical_state(p, p.x0);
    const ShiftUpdate u = shift_update(p.loaded.system, p.base, 0, ResidualSettings{});
    ASSERT_EQ(u.entries.size(), s.violation.size()); // Brennweite ist eine Gleichung
    double max_change = 0.0;
    for (std::size_t k = 0; k < u.entries.size(); ++k) {
        const ShiftEntry& e = u.entries[k];
        EXPECT_EQ(e.item, s.constraints.items[k].item);
        EXPECT_EQ(e.file_surface, s.constraints.items[k].file_surface);
        EXPECT_EQ(e.violation, s.violation[k]);
        EXPECT_EQ(e.shift, 0.0);
        EXPECT_EQ(e.new_shift, std::max(0.0, s.violation[k]));
        max_change = std::max(max_change, std::sqrt(e.weight) * e.new_shift);
    }
    EXPECT_EQ(u.max_change, max_change);
    EXPECT_GT(u.max_change, 0.0);

    SystemDescription d = p.loaded.system;
    apply_shift_update(d, u);
    EXPECT_NO_THROW(check_constraints(d));
    const MechanicalConstraints mech = make_mechanical_constraints(d);
    ASSERT_EQ(mech.items.size(), u.entries.size());
    for (std::size_t k = 0; k < u.entries.size(); ++k) {
        EXPECT_EQ(mech.items[k].shift, u.entries[k].new_shift);
    }
    // Zweite Fortschreibung am selben Punkt: θ'' = min(max(0, θ' + g), θ_max/√w); die Schranke
    // greift hier (w = 10⁶/mm², θ_max/√w = 1 mm bei Verletzungen bis 0,9 mm)
    const ShiftUpdate u2 = shift_update(d, p.base, 0, ResidualSettings{});
    int capped = 0;
    for (std::size_t k = 0; k < u2.entries.size(); ++k) {
        const double cap = kDefaultMaxShift / std::sqrt(u2.entries[k].weight);
        const double expected =
            std::min(std::max(0.0, u.entries[k].new_shift + s.violation[k]), cap);
        EXPECT_EQ(u2.entries[k].new_shift, expected);
        capped += expected == cap ? 1 : 0;
    }
    EXPECT_GT(capped, 0);
    clear_shifts(d);
    EXPECT_EQ(d, p.loaded.system);
}

// Abnahme Petzval (M4-T4) mit festem Gewicht w = w_Datei · 10⁻⁴.
TEST(AugmentedLagrangian, PetzvalWithFixedWeight) {
    const FdProblem p = petzval_problem();
    const double kappa_max =
        p.loaded.system.constraints->items[petzval_item(p.loaded.system)].max_curvature.value();
    const double f_target = p.loaded.system.target->focal_length.value();
    const auto cpu = [&](const LmOptions& o, std::span<const double> x0) {
        return optimize_lm_cpu(p.set, p.base, x0, o);
    };
    const SystemDescription d = petzval_al_system(p);
    const double weight = d.constraints->items[petzval_item(d)].weight_or_default();
    const AlResult al = al_loop(p, d, petzval_al_options, cpu,
                                al_tolerance(weight, kPetzvalPenaltyRelative * kappa_max));
    for (std::size_t k = 0; k < al.rounds.size(); ++k) {
        const AlRound& r = al.rounds[k];
        const ShiftEntry& e = r.update.entries.at(0);
        std::printf("[CPU] AL-Lauf %zu: F %.6e -> %.10e, %3d Zyklen, %-16s g = %+.3e 1/mm "
                    "(relativ %+.3e), θ = %.6e, √w|Δθ| = %.3e\n",
                    k, r.run.history.front(), r.run.merit, r.run.iterations,
                    to_string(r.run.status), e.violation, e.violation / kappa_max, e.new_shift,
                    r.update.max_change);
        EXPECT_TRUE(lm_converged(r.run.status)) << k << ": " << to_string(r.run.status);
    }
    EXPECT_LT(al.rounds.size(), static_cast<std::size_t>(kAlRounds)); // Abbruch über T
    const ParaxialData end = paraxial_at(p, al.x);
    const double kappa = std::abs(end.petzval_curvature);
    std::printf("[CPU] AL: %zu Läufe, %d Zyklen; |κ_P| − κ_max = %+.3e (relativ %+.3e), f − f_s = "
                "%+.3e mm\n",
                al.rounds.size(), al.iterations, kappa - kappa_max, (kappa - kappa_max) / kappa_max,
                end.focal_length - f_target);
    EXPECT_LE(kappa, kappa_max * (1.0 + kPetzvalPenaltyRelative));
    EXPECT_GE(kappa, kappa_max * (1.0 - kPetzvalPenaltyRelative));
    EXPECT_LT(std::abs(end.focal_length - f_target), kPetzvalFocalLength);
    EXPECT_GT(al.rounds.back().update.entries.at(0).new_shift, 0.0); // Bedingung aktiv

    // Vergleich: Penalty-Fortsetzung (heutige Abnahme)
    const std::vector<PetzvalStage> stages = petzval_continuation(p, cpu);
    const int continuation = stage_iterations(stages);
    std::printf("[CPU] Penalty-Fortsetzung: %zu Stufen, %d Zyklen\n", stages.size(), continuation);
    EXPECT_LE(al.iterations, continuation);
}

// Abnahme Mechanik (M4-T3) mit festem Gewicht w = w_Datei · 10⁻¹.
TEST(AugmentedLagrangian, MechanicalWithFixedWeight) {
    const FdProblem p = mechanical_problem();
    const auto cpu = [&](const LmOptions& o, std::span<const double> x0) {
        return optimize_lm_cpu(p.set, p.base, x0, o);
    };
    const SystemDescription d = mechanical_al_system(p);
    double weight = 0.0;
    for (const ConstraintSpec& c : d.constraints->items) {
        weight = is_mechanical(c.type) ? c.weight_or_default() : weight;
    }
    const AlResult al =
        al_loop(p, d, mechanical_al_options, cpu,
                al_tolerance(weight, std::min(kMechanicalPenaltyLength, kMechanicalPenaltyCos)));
    for (std::size_t k = 0; k < al.rounds.size(); ++k) {
        const AlRound& r = al.rounds[k];
        std::printf("[CPU] AL-Lauf %zu: F %.6e -> %.10e, %3d Zyklen, %-16s max √w·g+ = %.3e, "
                    "√w|Δθ| = %.3e\n",
                    k, r.run.history.front(), r.run.merit, r.run.iterations,
                    to_string(r.run.status), r.update.max_violation, r.update.max_change);
        EXPECT_TRUE(lm_converged(r.run.status)) << k << ": " << to_string(r.run.status);
        EXPECT_EQ(r.run.failed_rays, 0);
    }
    EXPECT_LT(al.rounds.size(), static_cast<std::size_t>(kAlRounds)); // Abbruch über T
    const MechanicalState s = mechanical_state(p, al.x);
    for (std::size_t k = 0; k < s.violation.size(); ++k) {
        const MechanicalConstraint& c = s.constraints.items[k];
        std::printf("[CPU] Ende %-21s Fläche %d  g = %+.3e  θ = %.3e\n",
                    std::string(to_string(c.type)).c_str(), c.file_surface, s.violation[k],
                    al.rounds.back().update.entries.at(k).new_shift);
        EXPECT_LE(s.violation[k],
                  mechanical_tolerance(c.type, kMechanicalPenaltyLength, kMechanicalPenaltyCos))
            << to_string(c.type) << " Fläche " << c.file_surface;
    }
    // Bedingungen binden (Luftspalt, Randdicken). Der Lauf endet in einem anderen lokalen
    // Minimum als die Fortsetzung: Einfallswinkel und Strahlhöhe sind dort nicht aktiv.
    for (const ConstraintType t :
         {ConstraintType::MaxCenterThickness, ConstraintType::MinEdgeThickness}) {
        EXPECT_GT(s.worst(t), -kMechanicalPenaltyLength) << to_string(t);
    }
    const double f = focal_length_at(p, al.x);
    EXPECT_LT(std::abs(f - 10.0), kMechanicalFocalLength);

    const std::vector<MechanicalStage> stages = mechanical_continuation(p, cpu);
    const int continuation = stage_iterations(stages);
    std::printf("[CPU] AL: %zu Läufe, %d Zyklen, f = %.9f mm; Penalty-Fortsetzung: %zu Stufen, "
                "%d Zyklen\n",
                al.rounds.size(), al.iterations, f, stages.size(), continuation);
    EXPECT_LE(al.iterations, continuation);
}

} // namespace
} // namespace mko::test
