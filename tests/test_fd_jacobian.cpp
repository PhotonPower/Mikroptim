// Finite-Differenzen-Jacobi (M3-T2), CPU-Pfad: Abnahme gegen Richardson-Extrapolation,
// Ablage, Schablonen an Schranken, Zusatzresiduen, mehrere Läufe in einem Batch,
// Status-Wechsel an der Totalreflexion, Fehlerfälle.

#include "fd_test_systems.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/optim/extra_residuals.hpp"
#include "mikroptim/optim/fd_formula.hpp"
#include "mikroptim/optim/fd_jacobian.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "mikroptim/optim/variant_generator.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using mko::ExtraResiduals;
using mko::FdJacobian;
using mko::FdJacobianOptions;
using mko::FdScheme;
using mko::FdStencil;
using mko::ResidualLayout;
using mko::SystemBatchHost;
using mko::test::FdProblem;
using mko::test::kFdJacobianFirstOrder;
using mko::test::kFdJacobianRichardson;
using mko::test::kFdRichardsonReference;

namespace {

template <class T>
bool same_bytes(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

bool same_bits(std::span<const double> a, std::span<const double> b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

const char* stencil_name(FdStencil s) {
    switch (s) {
    case FdStencil::Central:
        return "zentral";
    case FdStencil::OneSided:
        return "einseitig";
    case FdStencil::Truncated:
        return "gekürzt";
    case FdStencil::Forward:
        return "vorwärts";
    }
    return "?";
}

// Vergleich einer FD-Jacobi (ein Lauf) mit der Richardson-Referenz. Abnahme in der Norm
// ‖(J − D)S‖_F/‖D S‖_F mit S = diag(max(|x₀|, x_typ)) (unabhängig von den Einheiten der
// Variablen); dazu pro Spalte ‖J_p − D_p‖₂/‖D_p‖₂ < Schranke_p + Rundungsschätzung_p (schwach
// empfindliche Spalten sind rundungsbegrenzt, column_rounding_estimate).
struct RichardsonCheck {
    mko::test::JacobianDeviation plain;  // relativ pro Spalte und Frobenius ohne Skalierung
    mko::test::JacobianDeviation scaled; // Frobenius mit S
    std::vector<double> rounding;        // Rundungsschätzung pro Spalte
};

RichardsonCheck compare_with_richardson(const FdProblem& p, const FdJacobian& j, const char* name,
                                        const ExtraResiduals& extra = {}) {
    const ResidualLayout layout(p.base.topology());
    const mko::test::RichardsonReference ref =
        mko::test::richardson_reference(p, p.x0, layout, extra);
    EXPECT_EQ(ref.failed, 0u);
    const int P = p.set.size();
    const auto conv = mko::test::jacobian_deviation(ref.r1, ref.r2, P);
    EXPECT_LT(conv.max_column, kFdRichardsonReference) << "Referenz nicht konvergiert";
    const std::span<const double> jr = mko::test::run_jacobian(j, 0);
    RichardsonCheck c;
    c.plain = mko::test::jacobian_deviation(jr, ref.r2, P);
    c.scaled = mko::test::jacobian_deviation(mko::test::scale_columns(jr, p, p.x0),
                                             mko::test::scale_columns(ref.r2, p, p.x0), P);
    std::printf("[%s] P = %d, R = %lld: skaliert (Frobenius) %.3e, unskaliert %.3e, "
                "Referenz |R1 - R2| %.3e\n",
                name, P, static_cast<long long>(j.shape.rows()), c.scaled.frobenius,
                c.plain.frobenius, conv.max_column);
    for (int k = 0; k < P; ++k) {
        const auto kk = static_cast<std::size_t>(k);
        const double rho = mko::test::column_rounding_estimate(
            p, layout, j.patterns[0], k, mko::test::column_norm(ref.r2, P, k));
        c.rounding.push_back(rho);
        std::printf("    %-5s %-9s Spalte %.3e  Rundungsschätzung %.3e\n", p.set[k].name.c_str(),
                    stencil_name(j.patterns[0].stencil[kk]), c.plain.column[kk], rho);
    }
    return c;
}

// Standard-Jacobi im Inneren (alle Schablonen zentral) gegen Richardson.
void check_against_richardson(const FdProblem& p, const char* name,
                              const ExtraResiduals& extra = {}) {
    const ResidualLayout layout(p.base.topology());
    FdJacobianOptions opt;
    opt.extra = extra;
    const FdJacobian j = mko::fd_jacobian_cpu(p.set, p.base, p.x0, opt);
    EXPECT_EQ(j.shape.runs, 1);
    EXPECT_EQ(j.shape.extra_rows, extra.count);
    EXPECT_EQ(j.shape.variables, p.set.size());
    EXPECT_EQ(j.shape.ray_rows, layout.trace_rows());
    for (int k = 0; k < p.set.size(); ++k) {
        EXPECT_EQ(j.status_change(0, k), 0) << p.set[k].name;
        EXPECT_EQ(j.patterns[0].stencil[static_cast<std::size_t>(k)], FdStencil::Central);
    }
    const RichardsonCheck c = compare_with_richardson(p, j, name, extra);
    EXPECT_LT(c.scaled.frobenius, kFdJacobianRichardson); // Abnahmekriterium
    EXPECT_LT(c.plain.frobenius, kFdJacobianRichardson);
    for (int k = 0; k < p.set.size(); ++k) {
        const auto kk = static_cast<std::size_t>(k);
        EXPECT_LT(c.plain.column[kk], kFdJacobianRichardson + c.rounding[kk]) << p.set[k].name;
    }
}

} // namespace

// Die Formeln sind für Polynome bis zum Grad 2 exakt (zentral und einseitig mit ungleichen
// Schritten), vorwärts und gekürzt für Grad 1; Zahlen so gewählt, dass alles exakt
// darstellbar ist.
TEST(FdFormula, ExactForLowDegreePolynomials) {
    const auto f = [](double x) { return 3.0 + 2.0 * x - 0.5 * x * x; }; // f'(1) = 1
    const double x0 = 1.0;
    EXPECT_EQ(mko::fd_derivative(FdStencil::Central, f(x0), f(1.25), f(0.75), 0.25, -0.25), 1.0);
    // Einseitig mit d_a = −0,25, d_b = −0,5 und ungleich mit d_a = 0,25, d_b = 0,75
    EXPECT_EQ(mko::fd_derivative(FdStencil::OneSided, f(x0), f(0.75), f(0.5), -0.25, -0.5), 1.0);
    EXPECT_EQ(mko::fd_derivative(FdStencil::OneSided, f(x0), f(1.25), f(1.75), 0.25, 0.75), 1.0);
    const auto g = [](double x) { return 4.0 - 3.0 * x; };
    EXPECT_EQ(mko::fd_derivative(FdStencil::Forward, g(x0), g(1.5), 0.0, 0.5, 0.0), -3.0);
    EXPECT_EQ(mko::fd_derivative(FdStencil::Truncated, g(x0), g(1.25), g(0.5), 0.25, -0.5), -3.0);
    // Gekürzt mit einem Schritt 0 = einseitige Differenz
    EXPECT_EQ(mko::fd_derivative(FdStencil::Truncated, g(x0), g(x0), g(0.5), 0.0, -0.5), -3.0);
}

// Abnahme M3-T2: Standard-Jacobi (zentral, h = ε^{1/3}·max(|x|, x_typ)) gegen
// Richardson-Extrapolation (H = 10⁻³·max(|x|, x_typ), H/2, H/4, Fehler O(H⁶)): relativ in der
// spaltenskalierten Frobeniusnorm < 1·10⁻⁶ (auch unskaliert), pro Spalte bis auf den
// Rundungsanteil. Mehrlinser: 13 Variablen (Krümmungen, Dicken, Arbeitsabstand), 680
// Strahlresiduen über 3 Felder und 4 Wellenlängen.
TEST(FdJacobian, MatchesRichardsonMehrlinser) {
    const FdProblem p = mko::test::mehrlinser_problem();
    ASSERT_EQ(p.set.size(), 13);
    check_against_richardson(p, "Mehrlinser");
}

// Dasselbe für die Einzellinse mit Variablen aus der Beispieldatei (mit Schranken, alle
// Variablen im Inneren, kleine Apertur).
TEST(FdJacobian, MatchesRichardsonEinzellinse) {
    const FdProblem p = mko::test::einzellinse_problem();
    ASSERT_EQ(p.set.size(), 4);
    check_against_richardson(p, "Einzellinse");
}

// Abnahme mit echten paraxialen Zusatzresiduen (M3-T3, make_paraxial_residuals): Die Zeilen
// nach den Strahlresiduen gehören zur Jacobi und zur Referenz. Einzellinse aus
// einzellinse_bk7_nebenbedingungen.json (Brennweite mit Gewicht 100, Abgleichlänge,
// Arbeitsabstand als inaktive Untergrenze) und Mehrlinser mit Brennweite, Arbeitsabstand und
// Bildebene im paraxialen Fokus.
TEST(FdJacobian, MatchesRichardsonWithParaxialResiduals) {
    const FdProblem lens = mko::test::einzellinse_constraints_problem();
    const ExtraResiduals lens_extra = mko::test::paraxial_extra(lens);
    ASSERT_EQ(lens_extra.count, 3);
    check_against_richardson(lens, "Einzellinse + paraxial", lens_extra);

    const FdProblem multi = mko::test::mehrlinser_constraints_problem();
    const ExtraResiduals multi_extra = mko::test::paraxial_extra(multi);
    ASSERT_EQ(multi_extra.count, 3);
    check_against_richardson(multi, "Mehrlinser + paraxial", multi_extra);

    // Die paraxialen Zeilen allein (ohne Strahlzeilen) gegen die Referenz
    FdJacobianOptions opt;
    opt.extra = multi_extra;
    const FdJacobian j = mko::fd_jacobian_cpu(multi.set, multi.base, multi.x0, opt);
    const ResidualLayout layout(multi.base.topology());
    const auto ref = mko::test::richardson_reference(multi, multi.x0, layout, multi_extra);
    const std::int64_t R = j.shape.rows();
    const std::int64_t rv = j.shape.ray_rows;
    double worst = 0.0;
    for (int i = 0; i < multi_extra.count; ++i) {
        double diff = 0.0;
        double norm = 0.0;
        for (int k = 0; k < multi.set.size(); ++k) {
            const auto idx = static_cast<std::size_t>(k * R + rv + i);
            const double s =
                std::max(std::abs(multi.x0[static_cast<std::size_t>(k)]), multi.set[k].typical);
            diff += std::pow((j.jacobian[idx] - ref.r2[idx]) * s, 2);
            norm += std::pow(ref.r2[idx] * s, 2);
        }
        worst = std::max(worst, std::sqrt(diff / norm));
    }
    std::printf("[Mehrlinser + paraxial] paraxiale Zeilen (skaliert, je Zeile) max. %.3e\n", worst);
    EXPECT_LT(worst, kFdJacobianRichardson);
}

// Ablage: r₀ = Residuen der Basisvariante (bitgleich zu residuals_cpu), Spalte p von Lauf 0
// zusammenhängend und elementweise gleich der Formel aus den Residuen der FD-Varianten.
TEST(FdJacobian, LayoutAndBaseResiduals) {
    const FdProblem p = mko::test::mehrlinser_problem();
    const ResidualLayout layout(p.base.topology());
    const FdJacobian j = mko::fd_jacobian_cpu(p.set, p.base, p.x0);
    const std::int64_t R = j.shape.rows();
    ASSERT_EQ(R, layout.trace_rows());
    ASSERT_EQ(j.jacobian.size(), static_cast<std::size_t>(p.set.size() * R));
    ASSERT_EQ(j.residuals.size(), static_cast<std::size_t>(R));
    const mko::ResidualResult base = mko::residuals_cpu(p.base, layout);
    EXPECT_TRUE(same_bits(j.residual(0), base.residuals));

    const mko::FdBatch fd = mko::make_fd_batch(p.set, p.base, 0, p.x0);
    const mko::ResidualResult all = mko::residuals_cpu(fd.batch, layout);
    for (int k = 0; k < p.set.size(); ++k) {
        const auto kk = static_cast<std::size_t>(k);
        const std::span<const double> col = j.column(0, k);
        ASSERT_EQ(col.data(), j.jacobian.data() + kk * static_cast<std::size_t>(R));
        for (std::int64_t r = 0; r < R; r += 37) {
            const double expected = (all.residual(2 * k + 1, r) - all.residual(2 * k + 2, r)) /
                                    (fd.pattern.step_plus[kk] - fd.pattern.step_minus[kk]);
            EXPECT_EQ(col[static_cast<std::size_t>(r)], expected);
            EXPECT_EQ(j.at(0, r, k), expected);
        }
    }
}

// Schranken: x₀ auf der oberen Schranke (c[1]) und knapp über der unteren (t[1]) → einseitig
// zweiter Ordnung (x₀ + σh, x₀ + 2σh), sehr enger Bereich (c[2]) → gekürzt erster Ordnung,
// w im Inneren → zentral. Gegen Richardson (ohne Schranken, r(x) ist auch außerhalb definiert):
// einseitig wie zentral < 1·10⁻⁶ (pro Spalte bis auf den Rundungsanteil), gekürzt
// < kFdJacobianFirstOrder.
TEST(FdJacobian, BoundStencils) {
    const FdProblem p = mko::test::einzellinse_bound_problem();
    const double h0 = mko::fd_step(p.x0[0], p.set[0].typical);
    const double h2 = mko::fd_step(p.x0[2], p.set[2].typical);
    const FdJacobian j = mko::fd_jacobian_cpu(p.set, p.base, p.x0);
    const mko::FdPattern& pat = j.patterns[0];
    EXPECT_EQ(pat.stencil[0], FdStencil::OneSided);
    EXPECT_EQ(pat.point_plus[0], p.x0[0] - h0);
    EXPECT_EQ(pat.point_minus[0], p.x0[0] - 2.0 * h0);
    EXPECT_EQ(pat.stencil[1], FdStencil::Truncated);
    EXPECT_EQ(pat.point_plus[1], p.set[1].upper);
    EXPECT_EQ(pat.point_minus[1], p.set[1].lower);
    EXPECT_EQ(pat.stencil[2], FdStencil::OneSided);
    EXPECT_EQ(pat.point_plus[2], p.x0[2] + h2);
    EXPECT_EQ(pat.point_minus[2], p.x0[2] + 2.0 * h2);
    EXPECT_EQ(pat.stencil[3], FdStencil::Central);

    const RichardsonCheck c = compare_with_richardson(p, j, "Schranken");
    EXPECT_LT(c.scaled.frobenius, kFdJacobianRichardson);
    EXPECT_LT(c.plain.column[0], kFdJacobianRichardson + c.rounding[0]); // einseitig
    EXPECT_LT(c.plain.column[1], kFdJacobianFirstOrder + c.rounding[1]); // gekürzt
    EXPECT_LT(c.plain.column[2], kFdJacobianRichardson + c.rounding[2]); // einseitig
    EXPECT_LT(c.plain.column[3], kFdJacobianRichardson + c.rounding[3]); // zentral
}

// Zusatzresiduen: Zeilen nach den Strahlresiduen, r₀ = e(x₀) exakt, Ableitungen gegen die
// analytischen Werte; die Strahlzeilen bleiben bitgleich zum Lauf ohne Zusatzresiduen.
TEST(FdJacobian, ExtraResiduals) {
    const FdProblem p = mko::test::einzellinse_problem();
    FdJacobianOptions opt;
    opt.extra = mko::test::einzellinse_extra(p.set);
    const FdJacobian j = mko::fd_jacobian_cpu(p.set, p.base, p.x0, opt);
    const FdJacobian plain = mko::fd_jacobian_cpu(p.set, p.base, p.x0);
    const std::int64_t rv = plain.shape.rows();
    ASSERT_EQ(j.shape.ray_rows, rv);
    ASSERT_EQ(j.shape.extra_rows, 3);
    ASSERT_EQ(j.shape.rows(), rv + 3);
    std::vector<double> e0(3);
    opt.extra.evaluate(p.base, 0, e0);
    double worst = 0.0;
    for (int k = 0; k < p.set.size(); ++k) {
        const std::span<const double> col = j.column(0, k);
        EXPECT_TRUE(same_bits(col.first(static_cast<std::size_t>(rv)), plain.column(0, k)));
        for (int i = 0; i < 3; ++i) {
            const double exact = mko::test::analytic_extra_derivative(i, k, p.x0);
            const double fd = col[static_cast<std::size_t>(rv + i)];
            EXPECT_LE(std::abs(fd - exact), mko::test::kFdExtraAnalytic * std::abs(exact))
                << "e" << i << " nach " << p.set[k].name;
            if (exact != 0.0) {
                worst = std::max(worst, std::abs(fd - exact) / std::abs(exact));
            }
        }
    }
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(j.residual(0)[static_cast<std::size_t>(rv + i)], e0[static_cast<std::size_t>(i)]);
    }
    std::printf("[Zusatzresiduen] max. relative Abweichung von der analytischen Ableitung %.3e\n",
                worst);
}

// Mehrere Läufe in einem Batch (Lauf i belegt Varianten i·(2P + 1) …) liefern bitgleich
// dieselben Matrizen, Residuen, Status-Wechsel und Muster wie die Läufe einzeln; auch mit einer
// Vorlage pro Lauf (verschiedene Pupillenradien).
TEST(FdJacobian, MultipleRunsMatchSingleRuns) {
    const FdProblem p = mko::test::mehrlinser_problem();
    const int P = p.set.size();
    mko::PerturbationOptions po;
    po.seed = 11;
    std::vector<double> x0 = p.x0;
    for (int i = 1; i < 3; ++i) {
        const std::vector<double> xi =
            mko::perturbed_vector(p.set, p.x0, po, static_cast<std::uint64_t>(i));
        x0.insert(x0.end(), xi.begin(), xi.end());
    }
    FdJacobianOptions opt;
    opt.extra = mko::test::mehrlinser_extra(p.set);
    SystemBatchHost templates = p.base;
    templates.resize(3);
    for (int i = 0; i < 3; ++i) {
        templates.copy_variant(0, i);
        templates.pupil_radius(i) *= 1.0 - 0.1 * i;
    }
    const std::array<const SystemBatchHost*, 2> bases = {&p.base, &templates};
    for (const SystemBatchHost* base : bases) {
        const FdJacobian all = mko::fd_jacobian_cpu(p.set, *base, x0, opt);
        ASSERT_EQ(all.shape.runs, 3);
        for (int i = 0; i < 3; ++i) {
            SystemBatchHost single_base = *base;
            if (base->variants() > 1) { // Vorlage des Laufs i als einzige Variante
                single_base.copy_variant(i, 0);
                single_base.resize(1);
            }
            const std::span<const double> xi(x0.data() + static_cast<std::size_t>(i * P),
                                             static_cast<std::size_t>(P));
            const FdJacobian one = mko::fd_jacobian_cpu(p.set, single_base, xi, opt);
            EXPECT_TRUE(same_bits(mko::test::run_jacobian(all, i), one.jacobian)) << i;
            EXPECT_TRUE(same_bits(all.residual(i), one.residuals)) << i;
            for (int k = 0; k < P; ++k) {
                EXPECT_EQ(all.status_change(i, k), one.status_changes[static_cast<std::size_t>(k)]);
            }
            EXPECT_EQ(all.patterns[static_cast<std::size_t>(i)], one.patterns[0]);
        }
    }
}

// Versagensgrenze: Der FD-Punkt c[2] − h totalreflektiert genau den äußersten Ring; die
// Spalte c[2] zählt einen Status-Wechsel, w keinen. Im weichen Modus (Standard, M4-T2) ist der
// Strahl dort weich versagt (SoftTIR) und trägt ein Strafresiduum, im harten Modus TIR.
TEST(FdJacobian, StatusChangesAtTotalReflection) {
    const FdProblem p = mko::test::tir_problem();
    for (const mko::FailureMode mode : {mko::FailureMode::Soft, mko::FailureMode::Hard}) {
        SCOPED_TRACE(mko::failure_mode_name(mode));
        mko::ResidualSettings settings;
        settings.failure = mode;
        const ResidualLayout layout(p.base.topology(), settings);
        ASSERT_EQ(mko::residuals_cpu(p.base, layout).failed_count(), 0u);
        FdJacobianOptions opt;
        opt.quadrature = settings;
        const FdJacobian j = mko::fd_jacobian_cpu(p.set, p.base, p.x0, opt);
        ASSERT_EQ(p.set[0].name, "c[2]");
        EXPECT_EQ(j.status_change(0, 0), 1);
        EXPECT_EQ(j.status_change(0, 1), 0);
        // Der versagte Strahl ist der äußerste Ring in der Variante x₀ − h (Variante 2).
        const mko::FdBatch fd = mko::make_fd_batch(p.set, p.base, 0, p.x0);
        const mko::ResidualResult r = mko::residuals_cpu(fd.batch, layout);
        const std::int64_t last = layout.rays_per_variant() - 1;
        EXPECT_EQ(r.status_at(2, last), mode == mko::FailureMode::Soft ? mko::TraceStatus::SoftTIR
                                                                       : mko::TraceStatus::TIR);
        EXPECT_EQ(r.failed_count(), 1u);
        if (mode == mko::FailureMode::Soft) {
            EXPECT_GT(r.penalty(2, last), 0.0);
            EXPECT_NE(r.residual(2, last), 0.0); // bleibt im Kolben
        } else {
            EXPECT_EQ(r.residual(2, last), 0.0);
        }
    }
}

// Vorwärtsschema (P + 1 Varianten, erste Ordnung) gegen Richardson.
TEST(FdJacobian, ForwardScheme) {
    const FdProblem p = mko::test::mehrlinser_problem();
    FdJacobianOptions opt;
    opt.scheme = FdScheme::Forward;
    const FdJacobian j = mko::fd_jacobian_cpu(p.set, p.base, p.x0, opt);
    EXPECT_EQ(j.scheme, FdScheme::Forward);
    EXPECT_EQ(j.patterns[0].stencil[0], FdStencil::Forward);
    const ResidualLayout layout(p.base.topology());
    const auto ref = mko::test::richardson_reference(p, p.x0, layout);
    const auto dev =
        mko::test::jacobian_deviation(mko::test::run_jacobian(j, 0), ref.r2, p.set.size());
    std::printf("[Vorwärts] max. Spalte %.3e (%s)\n", dev.max_column,
                p.set[dev.worst_column].name.c_str());
    EXPECT_LT(dev.max_column, kFdJacobianFirstOrder);
}

TEST(FdJacobian, InvalidArguments) {
    const FdProblem p = mko::test::einzellinse_problem();
    std::vector<double> x = p.x0;
    x.push_back(1.0); // kein Vielfaches von P
    EXPECT_THROW((void)mko::fd_jacobian_cpu(p.set, p.base, x), std::invalid_argument);
    EXPECT_THROW((void)mko::fd_jacobian_cpu(p.set, p.base, std::span<const double>{}),
                 std::invalid_argument);
    SystemBatchHost two = p.base;
    two.resize(2);
    two.copy_variant(0, 1);
    EXPECT_THROW((void)mko::fd_jacobian_cpu(p.set, two, p.x0), std::invalid_argument);
    FdJacobianOptions opt;
    opt.extra.count = 2; // ohne Auswertungsfunktion
    EXPECT_THROW((void)mko::fd_jacobian_cpu(p.set, p.base, p.x0, opt), std::invalid_argument);
    opt.extra.count = -1;
    EXPECT_THROW((void)mko::fd_jacobian_cpu(p.set, p.base, p.x0, opt), std::invalid_argument);
    std::vector<double> outside = p.x0;
    outside[0] = 1.0; // außerhalb der Schranken
    EXPECT_THROW((void)mko::fd_jacobian_cpu(p.set, p.base, outside), std::invalid_argument);
    const mko::VariableSet empty;
    EXPECT_THROW((void)mko::make_fd_jacobian_batch(empty, p.base, p.x0), std::invalid_argument);
}
