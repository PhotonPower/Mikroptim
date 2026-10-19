#pragma once

// Strahlsuche (M6-T4): Vergleich mit dem einzweigigen Controller und Protokoll ohne Laufzeiten
// (docs/optimierung.md, Abschnitt 8).

#include "mikroptim/synth/beam_search.hpp"
#include "mikroptim/synth/homotopy.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

namespace mko::test {

// Protokoll als JSON ohne Laufzeiten (Vergleich auf Bitgleichheit; Zahlen verlustfrei).
inline std::string beam_json_without_seconds(const BeamResult& r) {
    nlohmann::json j = nlohmann::json::parse(beam_result_json(r));
    j.erase("seconds");
    for (nlohmann::json& s : j["steps"]) {
        s.erase("seconds");
    }
    return j.dump();
}

// Dasselbe ohne Gruppennummern (Vergleich gebündelt gegen einzeln).
inline std::string beam_json_without_groups(const BeamResult& r) {
    nlohmann::json j = nlohmann::json::parse(beam_json_without_seconds(r));
    j["beam_search"].erase("groups");
    for (nlohmann::json& s : j["steps"]) {
        s.erase("group");
    }
    return j.dump();
}

// B = 1 ohne Varianten: dieselben Entscheidungen wie run_homotopy. Gleich sind Status, Meldung,
// Endsystem (bitgleich), Punkt, Zähler und alle Schritte außer der inneren Homotopie in
// derselben Reihenfolge; die Stufen der inneren Homotopie rechnet die Strahlsuche stufenweise für
// alle Zweige gemeinsam (andere Reihenfolge), verglichen als Menge (Zweig, Stufe, F).
inline void expect_beam_same_as_homotopy(const HomotopyResult& h, const BeamResult& b) {
    EXPECT_EQ(b.status, h.status);
    EXPECT_EQ(b.message, h.message);
    EXPECT_TRUE(b.system == h.system);
    EXPECT_EQ(b.point, h.point);
    EXPECT_EQ(b.solves, h.solves);
    EXPECT_EQ(b.insertions, h.insertions);
    EXPECT_EQ(b.elements, h.elements);
    EXPECT_EQ(b.start_elements, h.start_elements);
    EXPECT_EQ(b.goal.reached, h.goal.reached);
    EXPECT_EQ(b.goal.max_rms, h.goal.max_rms);
    std::vector<const HomotopyStep*> hs;
    std::vector<const HomotopyStep*> bs;
    using Stage = std::tuple<int, double, double, int>;
    std::vector<Stage> hstages;
    std::vector<Stage> bstages;
    for (const HomotopyStep& s : h.steps) {
        if (s.kind == HomotopyStepKind::InsertionStage) {
            hstages.emplace_back(s.branch, s.stage, s.solve.merit, s.solve.iterations);
        } else {
            hs.push_back(&s);
        }
    }
    for (const BeamStep& s : b.steps) {
        EXPECT_GE(s.candidate, 0);
        if (s.step.kind == HomotopyStepKind::InsertionStage) {
            bstages.emplace_back(s.step.branch, s.step.stage, s.step.solve.merit,
                                 s.step.solve.iterations);
        } else {
            bs.push_back(&s.step);
        }
    }
    std::sort(hstages.begin(), hstages.end());
    std::sort(bstages.begin(), bstages.end());
    EXPECT_EQ(hstages, bstages);
    ASSERT_EQ(hs.size(), bs.size());
    for (std::size_t i = 0; i < hs.size(); ++i) {
        const HomotopyStep& x = *hs[i];
        const HomotopyStep& y = *bs[i];
        EXPECT_EQ(x.kind, y.kind) << i;
        EXPECT_EQ(x.point, y.point) << i;
        EXPECT_EQ(x.delta, y.delta) << i;
        EXPECT_EQ(x.accepted, y.accepted) << i;
        EXPECT_EQ(x.action, y.action) << i;
        EXPECT_EQ(x.next_delta, y.next_delta) << i;
        EXPECT_EQ(x.recommendation, y.recommendation) << i;
        EXPECT_EQ(x.operation, y.operation) << i;
        EXPECT_EQ(x.elements, y.elements) << i;
        EXPECT_EQ(x.solved, y.solved) << i;
        EXPECT_EQ(x.solve.merit, y.solve.merit) << i;
        EXPECT_EQ(x.solve.iterations, y.solve.iterations) << i;
        EXPECT_EQ(x.solve.rms, y.solve.rms) << i;
        EXPECT_EQ(x.goal.reached, y.goal.reached) << i;
        EXPECT_EQ(x.goal.max_rms, y.goal.max_rms) << i;
    }
}

} // namespace mko::test
