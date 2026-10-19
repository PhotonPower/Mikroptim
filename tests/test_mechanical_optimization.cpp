// Abnahme der mechanischen Nebenbedingungen (M4-T3) auf der CPU: Zweilinser mit Durchmesser,
// Mitten- und Randdicke und Einfallswinkel; LM mit Penalty-Fortsetzung über die Gewichte → alle
// Grenzen eingehalten (bis auf die dokumentierte Restverletzung der Strafe), der Kontrolllauf
// ohne Grenzen verletzt jede Bedingungsart (docs/optimierung.md, Abschnitt 2.5). Dazu die
// Ausgabe von `mikroptim optimize` (Tabelle der mechanischen Bedingungen in Text und JSON).

#include "cli/optimize_command.hpp"
#include "mechanical_test_systems.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace mko::test {
namespace {

void print_state(const char* tag, const MechanicalState& s) {
    for (std::size_t k = 0; k < s.violation.size(); ++k) {
        const MechanicalConstraint& c = s.constraints.items[k];
        const bool angle = c.type == ConstraintType::MaxIncidenceAngle;
        std::printf("%s   %-21s Fläche %d  Grenze %.6f  Ist %.9f  g = %+.3e\n", tag,
                    std::string(to_string(c.type)).c_str(), c.file_surface,
                    angle ? c.angle_deg : c.limit,
                    angle ? incidence_angle_deg(s.value[k]) : s.value[k], s.violation[k]);
    }
}

TEST(MechanicalOptimization, CpuKeepsAllLimits) {
    const FdProblem p = mechanical_problem();

    // Kontrolle: das freie Optimum (alle mechanischen Gewichte 0) verletzt jede Bedingungsart
    const LmOptions free_options = mechanical_options(with_mechanical_weight(p.loaded.system, 0.0),
                                                      kMechanicalStageIterations);
    const LmRunResult free_run = optimize_lm_cpu(p.set, p.base, p.x0, free_options).runs.at(0);
    const MechanicalState free_state = mechanical_state(p, free_run.x);
    std::printf("[CPU] ohne Grenzen: F = %.10e, %d Zyklen, %s\n", free_run.merit,
                free_run.iterations, to_string(free_run.status));
    print_state("[CPU]", free_state);
    EXPECT_TRUE(lm_converged(free_run.status)) << to_string(free_run.status);
    for (const ConstraintType t : mechanical_types()) {
        EXPECT_GT(free_state.worst(t), 1e-3) << to_string(t) << " im freien Optimum nicht verletzt";
    }

    // Fortsetzung über die Gewichte 10² … 10⁶
    const std::vector<MechanicalStage> stages =
        mechanical_continuation(p, [&](const LmOptions& o, std::span<const double> x0) {
            return optimize_lm_cpu(p.set, p.base, x0, o);
        });
    ASSERT_EQ(stages.size(), static_cast<std::size_t>(kMechanicalStages));
    for (const MechanicalStage& s : stages) {
        std::printf("[CPU]   w = %8.1e: F %.6e -> %.10e, %3d Zyklen, %-16s", s.weight,
                    s.run.history.front(), s.run.merit, s.run.iterations, to_string(s.run.status));
        for (const ConstraintType t : mechanical_types()) {
            std::printf("  %s %+.2e", std::string(to_string(t)).c_str(), s.state.worst(t));
        }
        std::printf("\n");
        EXPECT_TRUE(lm_converged(s.run.status)) << s.weight << ": " << to_string(s.run.status);
        EXPECT_EQ(s.run.failed_rays, 0);
        EXPECT_TRUE(p.set.feasible(s.run.x));
    }

    // Ende: alle Grenzen eingehalten bis auf die Restverletzung der Strafe
    const MechanicalStage& last = stages.back();
    print_state("[CPU] Ende", last.state);
    for (std::size_t k = 0; k < last.state.violation.size(); ++k) {
        const ConstraintType t = last.state.constraints.items[k].type;
        EXPECT_LE(last.state.violation[k],
                  mechanical_tolerance(t, kMechanicalPenaltyLength, kMechanicalPenaltyCos))
            << to_string(t) << " Fläche " << last.state.constraints.items[k].file_surface;
    }
    // Die Bedingungen binden: Rand, Luftspalt und Einfallswinkel liegen auf ihrer Grenze
    for (const ConstraintType t :
         {ConstraintType::MaxCenterThickness, ConstraintType::MinEdgeThickness,
          ConstraintType::MaxIncidenceAngle}) {
        EXPECT_GT(last.state.worst(t), -kMechanicalPenaltyLength) << to_string(t);
    }
    // Brennweite bleibt nahe am Soll (Gleichgewicht der Brennweitenstrafe)
    const double f = focal_length_at(p, last.run.x);
    std::printf("[CPU] Ende: f = %.9f mm\n", f);
    EXPECT_LT(std::abs(f - 10.0), kMechanicalFocalLength);
}

// `mikroptim optimize` gibt die Tabelle der mechanischen Bedingungen aus (Text und JSON).
TEST(MechanicalOptimization, CliReportsMechanicalConstraints) {
    const std::string file = std::string(MKO_SOURCE_DIR) + "/data/systems/zweilinser_mechanik.json";
    const auto run = [&](const std::vector<std::string>& args, std::string& out) {
        const std::vector<std::string_view> v(args.begin(), args.end());
        std::ostringstream os;
        std::ostringstream es;
        const int code = mko::cli::run_optimize_command(v, os, es);
        out = os.str();
        return code;
    };
    std::string text;
    ASSERT_EQ(run({file, "--max-iterations", "3", "-q"}, text), 0);
    EXPECT_NE(text.find("Nebenbedingungen (mechanisch)"), std::string::npos);
    EXPECT_NE(text.find("Mechanische Nebenbedingungen"), std::string::npos);
    EXPECT_NE(text.find("min_edge_thickness"), std::string::npos);
    EXPECT_NE(text.find("max_incidence_angle"), std::string::npos);

    std::string js;
    ASSERT_EQ(run({file, "--max-iterations", "0", "--json"}, js), 0);
    const nlohmann::json j = nlohmann::json::parse(js);
    const nlohmann::json& m = j.at("mechanical_constraints");
    // Einträge: Fläche 1, Flächen 3 und 4, Luftspalte 0-1 und 2-3 (max), Glas 1-2 und 3-4,
    // Luft 0-1 und 2-3 (min Rand), Einfallswinkel 1 … 4
    ASSERT_EQ(m.size(), 13U);
    const FdProblem p = mechanical_problem();
    const MechanicalState s = mechanical_state(p, p.x0);
    for (std::size_t k = 0; k < m.size(); ++k) {
        const MechanicalConstraint& c = s.constraints.items[k];
        EXPECT_EQ(m[k]["type"], std::string(to_string(c.type)));
        EXPECT_EQ(m[k]["item"], c.item);
        EXPECT_EQ(m[k]["surface"], c.file_surface);
        const bool angle = c.type == ConstraintType::MaxIncidenceAngle;
        EXPECT_EQ(m[k]["unit"], angle ? "deg" : "mm");
        const double value = angle ? incidence_angle_deg(s.value[k]) : s.value[k];
        // ohne Zyklen ist Start = Ergebnis = x₀
        EXPECT_DOUBLE_EQ(m[k]["value_start"].get<double>(), value);
        EXPECT_DOUBLE_EQ(m[k]["value_result"].get<double>(), value);
        EXPECT_DOUBLE_EQ(m[k]["violation_start"].get<double>(), s.violation[k]);
        EXPECT_DOUBLE_EQ(m[k]["residual_start"].get<double>(),
                         std::sqrt(c.weight) * std::max(s.violation[k], 0.0));
        EXPECT_EQ(m[k]["satisfied"], !(s.violation[k] > 0.0));
        if (c.type == ConstraintType::MinEdgeThickness) {
            EXPECT_TRUE(m[k]["height_start"].is_number());
        } else {
            EXPECT_TRUE(m[k]["height_start"].is_null());
        }
    }
}

} // namespace
} // namespace mko::test
