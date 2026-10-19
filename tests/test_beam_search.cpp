// Kandidatenverwaltung (M6-T4): Strahlsuche mit dem synthetischen Modell – Gleichheit mit dem
// einzweigigen Controller bei B = 1, Verzweigung, Auswahl der besten B, Deduplizierung,
// Varianten und gestörte Starts, Vorfilter, Gruppen, Gleichschritt, Stammbaum, Determinismus
// (docs/optimierung.md, Abschnitt 8).

#include "beam_test_tools.hpp"
#include "homotopy_mock.hpp"
#include "mikroptim/synth/beam_search.hpp"
#include "mikroptim/synth/homotopy.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mko::test {
namespace {

BeamOptions beam_options(double target, int width) {
    BeamOptions o;
    o.homotopy = mock_options(target);
    o.beam_width = width;
    return o;
}

const BeamNode& node(const BeamResult& r, int id) {
    return r.nodes.at(static_cast<std::size_t>(id));
}

std::vector<const BeamStep*> beam_steps_of(const BeamResult& r, HomotopyStepKind kind) {
    std::vector<const BeamStep*> out;
    for (const BeamStep& s : r.steps) {
        if (s.step.kind == kind) {
            out.push_back(&s);
        }
    }
    return out;
}

struct Scenario {
    const char* name;
    MockSpec spec;
    HomotopyOptions options;
    HomotopyPoint start;
};

std::vector<Scenario> scenarios() {
    std::vector<Scenario> v;
    v.push_back({"Vergrößern", {}, mock_options(0.6), mock_point(0.3)});
    {
        MockSpec s;
        s.max_jump = 0.03;
        v.push_back({"Halbieren", s, mock_options(0.4), mock_point(0.3)});
    }
    {
        MockSpec s;
        s.capacity0 = 0.4;
        s.capacity_step = 0.15;
        s.one_sided = true;
        v.push_back({"Einfügen", s, mock_options(0.6), mock_point(0.3)});
    }
    {
        MockSpec s;
        s.capacity0 = 0.4;
        s.need = false;
        v.push_back({"OhneBedarf", s, mock_options(0.6), mock_point(0.3)});
    }
    {
        MockSpec s;
        s.capacity0 = 0.4;
        s.capacity_step = 0.1;
        HomotopyOptions o = mock_options(0.6);
        o.max_elements = 2;
        v.push_back({"MaxElemente", s, o, mock_point(0.3)});
    }
    {
        MockSpec s;
        s.one_sided = true;
        s.goal_ghost_weight = 100.0;
        HomotopyOptions o = mock_options(0.32);
        o.smoothing_stages = 2;
        v.push_back({"Fortsetzungen", s, o, mock_point(0.3)});
    }
    {
        MockSpec s;
        s.goal_never = true;
        HomotopyOptions o = mock_options(0.32);
        o.max_elements = 2;
        v.push_back({"ZielVerfehlt", s, o, mock_point(0.3)});
    }
    {
        MockSpec s;
        s.start_merit = 5.0;
        s.need = false;
        v.push_back({"StartAbgelehnt", s, mock_options(0.6), mock_point(0.3)});
    }
    {
        MockSpec s;
        s.start_merit = 5.0;
        s.capacity0 = 0.2;
        s.capacity_step = 0.5;
        v.push_back({"StartEinfügen", s, mock_options(0.6), mock_point(0.3)});
    }
    {
        HomotopyOptions o = mock_options(0.32);
        o.target_field = 1.0;
        o.target_spectrum = 2;
        HomotopyPoint p = mock_point(0.3);
        p.field = 0.0;
        v.push_back({"FeldSpektrum", {}, o, p});
    }
    {
        HomotopyOptions o = mock_options(0.6);
        o.max_steps = 3;
        v.push_back({"MaxSchritte", {}, o, mock_point(0.3)});
    }
    return v;
}

TEST(BeamSearch, SingleBeamMatchesHomotopyController) {
    for (const Scenario& sc : scenarios()) {
        SCOPED_TRACE(sc.name);
        MockModel mh(sc.spec);
        const HomotopyResult h =
            run_homotopy(mh, MockModel::start(sc.start.aperture), sc.start, sc.options);
        MockModel mb(sc.spec);
        BeamOptions o;
        o.homotopy = sc.options;
        o.beam_width = 1;
        const BeamResult b = run_beam_search(mb, MockModel::start(sc.start.aperture), sc.start, o);
        expect_beam_same_as_homotopy(h, b);
        EXPECT_EQ(mh.finishes, mb.finishes);
        // genau ein Kandidat pro Zeitpunkt im Strahl
        EXPECT_LE(b.beam.size(), 1U);
    }
}

// B = 2: Nach jedem Einfügen bleiben die besten zwei Kandidaten; der Stammbaum zeigt Eltern,
// Kinder und Status.
TEST(BeamSearch, BranchesAndKeepsBestB) {
    MockSpec spec;
    spec.capacity0 = 0.4;
    spec.capacity_step = 0.15; // 1 Element: 0,4; 2: 0,55; 3: 0,7
    MockModel m(spec);
    const BeamResult r =
        run_beam_search(m, MockModel::start(0.3), mock_point(0.3), beam_options(0.6, 2));
    ASSERT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    // Wurzel → 2 Kinder → beide fügen ein → 4 Enkel, davon 2 behalten
    ASSERT_EQ(r.nodes.size(), 7U);
    EXPECT_EQ(node(r, 0).status, BeamNodeStatus::Expanded);
    EXPECT_EQ(node(r, 0).children, (std::vector<int>{1, 2}));
    EXPECT_EQ(node(r, 1).parent, 0);
    EXPECT_EQ(node(r, 1).branch, 0);
    EXPECT_EQ(node(r, 2).branch, 1);
    EXPECT_EQ(node(r, 1).status, BeamNodeStatus::Expanded);
    EXPECT_EQ(node(r, 2).status, BeamNodeStatus::Expanded);
    int finished = 0;
    int discarded = 0;
    for (int id = 3; id < 7; ++id) {
        const BeamNode& n = node(r, id);
        EXPECT_TRUE(n.parent == 1 || n.parent == 2);
        EXPECT_EQ(n.elements, 3);
        if (n.status == BeamNodeStatus::Finished) {
            ++finished;
        } else {
            EXPECT_EQ(n.status, BeamNodeStatus::Discarded);
            EXPECT_NE(n.reason.find("Rang"), std::string::npos) << n.reason;
            ++discarded;
        }
    }
    EXPECT_EQ(finished, 2);
    EXPECT_EQ(discarded, 2);
    EXPECT_EQ(r.beam.size(), 2U);
    EXPECT_EQ(r.insertions, 3);
    EXPECT_EQ(r.elements, 3);
    // Ergebnis: Zweig 1 (besser) mit kleinster Nummer unter Gleichen
    EXPECT_EQ(r.system.surfaces.back().curvature, 1.0);
    EXPECT_EQ(node(r, r.best).status, BeamNodeStatus::Finished);
    EXPECT_EQ(r.beam.front(), r.best);
    EXPECT_EQ(r.point.aperture, 0.6);
}

// Rangfolge: annehmbar vor nicht annehmbar, dann weniger Elemente, dann kleineres F, dann Nummer.
TEST(BeamSearch, RankOrder) {
    BeamNode a;
    BeamNode b;
    a.id = 1;
    b.id = 2;
    a.merit = b.merit = 0.5;
    a.acceptable = b.acceptable = true;
    a.elements = 3;
    b.elements = 2;
    EXPECT_TRUE(beam_rank_before(b, a)); // weniger Elemente
    b.elements = 3;
    b.merit = 0.4;
    EXPECT_TRUE(beam_rank_before(b, a)); // kleineres F
    b.merit = 0.5;
    EXPECT_TRUE(beam_rank_before(a, b)); // kleinere Nummer
    b.acceptable = false;
    b.elements = 1;
    b.merit = 0.1;
    EXPECT_TRUE(beam_rank_before(a, b)); // annehmbar zuerst
    a.acceptable = false;
    a.failed_rays = 3;
    b.failed_rays = 1;
    EXPECT_TRUE(beam_rank_before(b, a)); // nicht annehmbar: weniger versagte Strahlen
    a.failed_rays = 1;
    a.merit = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(beam_rank_before(b, a)); // endliche Merit zuerst
}

// Deduplizierung: Zwei Zweige im selben Minimum (Abstand 1e-9) – der schlechtere wird verworfen.
TEST(BeamSearch, Deduplication) {
    MockSpec spec;
    spec.capacity0 = 0.4;
    spec.capacity_step = 0.5;
    spec.branch_spread = 1e-9;
    MockModel m(spec);
    const BeamResult r =
        run_beam_search(m, MockModel::start(0.3), mock_point(0.3), beam_options(0.6, 2));
    ASSERT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    ASSERT_EQ(r.nodes.size(), 3U);
    EXPECT_EQ(node(r, 1).status, BeamNodeStatus::Discarded);
    EXPECT_NE(node(r, 1).reason.find("Duplikat von #2"), std::string::npos) << node(r, 1).reason;
    EXPECT_EQ(r.beam, std::vector<int>{2});
    // abgeschaltet: beide bleiben
    BeamOptions o = beam_options(0.6, 2);
    o.dedup_tolerance = -1.0;
    MockModel m2(spec);
    const BeamResult r2 = run_beam_search(m2, MockModel::start(0.3), mock_point(0.3), o);
    EXPECT_EQ(r2.beam.size(), 2U);
}

// Varianten × Zweige × Starts: Jedes Einfügen erzeugt 3 · 2 · 3 Kinder; die beste Variante
// gewinnt.
TEST(BeamSearch, AlternativesAndPerturbedStarts) {
    MockSpec spec;
    spec.capacity0 = 0.4;
    spec.capacity_step = 0.5;
    spec.alternatives = 3;
    spec.option_bonus = 0.1;
    spec.perturbation = 0.01;
    MockModel m(spec);
    BeamOptions o = beam_options(0.6, 3);
    o.alternatives = 3;
    o.perturbed_starts = 2;
    o.seed = 11;
    const BeamResult r = run_beam_search(m, MockModel::start(0.3), mock_point(0.3), o);
    ASSERT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    ASSERT_EQ(r.nodes.size(), 1U + 18U);
    EXPECT_EQ(r.insertions, 3);
    EXPECT_EQ(beam_steps_of(r, HomotopyStepKind::Insertion).size(), 3U);
    int id = 1;
    for (int k = 0; k < 3; ++k) {
        for (int b = 0; b < 2; ++b) {
            for (int j = 0; j < 3; ++j, ++id) {
                const BeamNode& n = node(r, id);
                EXPECT_EQ(n.option, k);
                EXPECT_EQ(n.branch, b);
                EXPECT_EQ(n.start, j);
                EXPECT_EQ(n.parent, 0);
            }
        }
    }
    // Startwerte: ein Strom pro Zweig, fortlaufend
    ASSERT_EQ(m.seeds.size(), 6U);
    for (std::size_t n = 0; n < m.seeds.size(); ++n) {
        EXPECT_EQ(m.seeds[n], variant_seed(11, n));
    }
    // beste Variante (2) im Ergebnis, drei Kandidaten im Strahl
    EXPECT_EQ(r.system.surfaces.back().semi_diameter, 2.0);
    EXPECT_EQ(r.beam.size(), 3U);
    for (const int b : r.beam) {
        EXPECT_EQ(node(r, b).option, 2);
    }
}

// Fester Startwert: zwei Läufe bitgleich (Protokoll ohne Laufzeiten, Systeme, Stammbaum); ein
// anderer Startwert ändert die gestörten Starts.
TEST(BeamSearch, ReproducibleWithSeed) {
    MockSpec spec;
    spec.capacity0 = 0.4;
    spec.capacity_step = 0.15;
    spec.perturbation = 0.3;
    BeamOptions o = beam_options(0.6, 3);
    o.perturbed_starts = 2;
    o.seed = 42;
    MockModel m1(spec);
    MockModel m2(spec);
    const BeamResult a = run_beam_search(m1, MockModel::start(0.3), mock_point(0.3), o);
    const BeamResult b = run_beam_search(m2, MockModel::start(0.3), mock_point(0.3), o);
    ASSERT_EQ(a.status, HomotopyStatus::GoalReached) << a.message;
    EXPECT_EQ(beam_json_without_seconds(a), beam_json_without_seconds(b));
    EXPECT_EQ(a.system, b.system);
    o.seed = 43;
    MockModel m3(spec);
    const BeamResult c = run_beam_search(m3, MockModel::start(0.3), mock_point(0.3), o);
    EXPECT_NE(beam_json_without_seconds(a), beam_json_without_seconds(c));
    EXPECT_NE(a.system, c.system);
}

// Gemeinsame Aufrufe: Kandidaten gleicher Topologie (hier: Flächenzahl) laufen zusammen; das
// Ergebnis hängt nicht von der Gruppierung ab.
TEST(BeamSearch, GroupsDoNotChangeResults) {
    MockSpec spec;
    spec.capacity0 = 0.4;
    spec.capacity_step = 0.15;
    spec.branch0_capacity = 0.05;
    MockSpec grouped = spec;
    grouped.group_by_elements = true;
    MockModel m1(spec);
    MockModel m2(grouped);
    const BeamResult a =
        run_beam_search(m1, MockModel::start(0.3), mock_point(0.3), beam_options(0.6, 2));
    const BeamResult b =
        run_beam_search(m2, MockModel::start(0.3), mock_point(0.3), beam_options(0.6, 2));
    ASSERT_EQ(a.status, HomotopyStatus::GoalReached) << a.message;
    EXPECT_EQ(a.system, b.system);
    ASSERT_EQ(a.steps.size(), b.steps.size());
    for (std::size_t i = 0; i < a.steps.size(); ++i) {
        EXPECT_EQ(a.steps[i].candidate, b.steps[i].candidate) << i;
        EXPECT_EQ(homotopy_step_json(a.steps[i].step).substr(0, 40),
                  homotopy_step_json(b.steps[i].step).substr(0, 40))
            << i;
        EXPECT_EQ(a.steps[i].step.solve.merit, b.steps[i].step.solve.merit) << i;
        EXPECT_EQ(a.steps[i].step.action, b.steps[i].step.action) << i;
    }
    EXPECT_EQ(a.solves, b.solves);
    EXPECT_LT(b.groups, a.groups);
    EXPECT_EQ(a.groups, a.solves); // ohne Bündelung: ein Aufruf pro Optimierung
    // Stufen der inneren Homotopie: Geschwister im selben Aufruf
    for (const BeamStep* s : beam_steps_of(b, HomotopyStepKind::InsertionStage)) {
        for (const BeamStep* t : beam_steps_of(b, HomotopyStepKind::InsertionStage)) {
            if (s->step.stage == t->step.stage && s->step.point == t->step.point &&
                node(b, s->candidate).parent == node(b, t->candidate).parent) {
                EXPECT_EQ(s->group, t->group);
            }
        }
    }
    EXPECT_GT(*std::max_element(m2.batches.begin(), m2.batches.end()), 1);
}

// Gleichschritt: Ein Kandidat (Zweig 0, geringere Kapazität) scheitert, wo der andere
// weiterkommt – beide halbieren; unter Δ_min fügt nur der gescheiterte ein, der andere bleibt
// am guten Zustand. Danach Auswahl aus dem Verbliebenen und den Kindern.
TEST(BeamSearch, LockstepMixedRejection) {
    MockSpec spec;
    spec.capacity0 = 0.4;
    spec.capacity_step = 0.3; // 2 Elemente: 0,7 (Zweig 0: 0,6); 3: 1,0
    spec.branch0_capacity = 0.1;
    spec.need = true;
    MockModel m(spec);
    const BeamResult r =
        run_beam_search(m, MockModel::start(0.3), mock_point(0.3), beam_options(0.7, 2));
    ASSERT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    // Kandidat #1 (Zweig 0) fügt bei ~0,6 ein, #2 bleibt
    EXPECT_EQ(node(r, 1).status, BeamNodeStatus::Expanded);
    EXPECT_NE(node(r, 2).status, BeamNodeStatus::Expanded);
    bool waited = false;
    bool shrunk_acceptable = false;
    for (const BeamStep& s : r.steps) {
        if (s.candidate == 2 &&
            s.step.operation.find("bleibt am guten Zustand") != std::string::npos) {
            waited = true;
            EXPECT_FALSE(s.step.accepted);
        }
        if (s.candidate == 2 && s.step.action == HomotopyAction::Shrink) {
            shrunk_acceptable = true;
        }
    }
    EXPECT_TRUE(waited);
    EXPECT_TRUE(shrunk_acceptable);
    // Rang: #2 (2 Elemente) vor den Kindern von #1 (3 Elemente)
    EXPECT_EQ(r.beam.size(), 2U);
    EXPECT_EQ(r.beam.front(), 2);
    EXPECT_EQ(r.best, 2);
    EXPECT_EQ(r.elements, 2);
    EXPECT_EQ(node(r, r.beam.back()).parent, 1);
    // alle Schritte eines Aufrufs am selben Punkt
    for (const BeamStep& s : r.steps) {
        if (s.step.kind == HomotopyStepKind::Aperture) {
            for (const BeamStep& t : r.steps) {
                if (t.group == s.group) {
                    EXPECT_EQ(t.step.point, s.step.point);
                }
            }
        }
    }
}

// Vorfilter: Starts mit großer Marke werden vor der inneren Homotopie verworfen.
TEST(BeamSearch, PrefilterDropsStarts) {
    MockSpec spec;
    spec.capacity0 = 0.4;
    spec.capacity_step = 0.5;
    spec.perturbation = 0.5;
    spec.prefilter_max = 1.0; // Zweig 1 + positive Störung fällt heraus
    MockModel m(spec);
    BeamOptions o = beam_options(0.6, 4);
    o.perturbed_starts = 3;
    o.prefilter = true;
    o.seed = 3;
    const BeamResult r = run_beam_search(m, MockModel::start(0.3), mock_point(0.3), o);
    ASSERT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    int filtered = 0;
    for (const BeamNode& n : r.nodes) {
        if (n.reason.find("Vorfilter") != std::string::npos) {
            ++filtered;
            EXPECT_EQ(n.solves, 0);
            EXPECT_GT(n.branch, 0);
        }
    }
    EXPECT_GT(filtered, 0);
    for (const BeamStep* s : beam_steps_of(r, HomotopyStepKind::InsertionStage)) {
        EXPECT_EQ(node(r, s->candidate).reason.find("Vorfilter"), std::string::npos);
    }
}

// Kein Kandidat hat einen Ausweg: Abbruch mit dem Grund des besten.
TEST(BeamSearch, StopsWhenBeamEmpty) {
    MockSpec spec;
    spec.capacity0 = 0.4;
    spec.capacity_step = 0.1;
    MockModel m(spec);
    BeamOptions o = beam_options(0.6, 2);
    o.homotopy.max_elements = 2;
    const BeamResult r = run_beam_search(m, MockModel::start(0.3), mock_point(0.3), o);
    EXPECT_EQ(r.status, HomotopyStatus::MaxElements) << r.message;
    EXPECT_NE(r.message.find("höchste Elementzahl 2"), std::string::npos);
    EXPECT_EQ(r.beam.size(), 2U);
    EXPECT_EQ(r.elements, 2);
    EXPECT_LE(r.point.aperture, 0.5);
}

TEST(BeamSearch, InvalidOptionsThrow) {
    MockModel m({});
    BeamOptions o = beam_options(0.6, 0);
    EXPECT_THROW(run_beam_search(m, MockModel::start(0.3), mock_point(0.3), o),
                 std::invalid_argument);
    o = beam_options(0.6, 2);
    o.alternatives = 0;
    EXPECT_THROW(run_beam_search(m, MockModel::start(0.3), mock_point(0.3), o),
                 std::invalid_argument);
    o = beam_options(0.6, 2);
    o.perturbed_starts = -1;
    EXPECT_THROW(run_beam_search(m, MockModel::start(0.3), mock_point(0.3), o),
                 std::invalid_argument);
    o = beam_options(0.6, 2);
    o.dedup_tolerance = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(run_beam_search(m, MockModel::start(0.3), mock_point(0.3), o),
                 std::invalid_argument);
    o = beam_options(0.6, 2);
    o.homotopy.penalty_factor = 1.0;
    EXPECT_THROW(run_beam_search(m, MockModel::start(0.3), mock_point(0.3), o),
                 std::invalid_argument);
}

// Protokoll: Stammbaum als Text, JSON (Kandidaten, Kandidat und Gruppe pro Schritt) und DOT.
TEST(BeamSearch, ProtocolTreeJsonDot) {
    MockSpec spec;
    spec.capacity0 = 0.4;
    spec.capacity_step = 0.15;
    MockModel m(spec);
    std::vector<int> streamed;
    const BeamResult r =
        run_beam_search(m, MockModel::start(0.3), mock_point(0.3), beam_options(0.6, 2),
                        [&](const BeamStep& s) { streamed.push_back(s.step.index); });
    ASSERT_EQ(streamed.size(), r.steps.size());
    for (std::size_t i = 0; i < streamed.size(); ++i) {
        EXPECT_EQ(streamed[i], static_cast<int>(i));
    }
    const nlohmann::json j = nlohmann::json::parse(beam_result_json(r));
    EXPECT_EQ(j["status"], "goal_reached");
    const nlohmann::json& bs = j["beam_search"];
    EXPECT_EQ(bs["beam_width"], 2);
    EXPECT_EQ(bs["best"], r.best);
    ASSERT_EQ(bs["candidates"].size(), r.nodes.size());
    for (std::size_t i = 0; i < r.nodes.size(); ++i) {
        const nlohmann::json& c = bs["candidates"][i];
        EXPECT_EQ(c["id"], static_cast<int>(i));
        EXPECT_EQ(c["status"], to_string(r.nodes[i].status));
        if (r.nodes[i].parent < 0) {
            EXPECT_TRUE(c["parent"].is_null());
        } else {
            EXPECT_EQ(c["parent"], r.nodes[i].parent);
        }
        EXPECT_EQ(c["elements"], r.nodes[i].elements);
    }
    ASSERT_EQ(j["steps"].size(), r.steps.size());
    for (std::size_t i = 0; i < r.steps.size(); ++i) {
        EXPECT_EQ(j["steps"][i]["candidate"], r.steps[i].candidate);
        EXPECT_EQ(j["steps"][i]["index"], static_cast<int>(i));
        EXPECT_EQ(j["steps"][i]["group"].is_null(), r.steps[i].group < 0);
    }
    std::ostringstream text;
    write_beam_step_header(text);
    for (const BeamStep& s : r.steps) {
        write_beam_step_line(s, text);
    }
    write_beam_summary(r, text);
    write_beam_tree(r, text);
    const std::string t = text.str();
    EXPECT_NE(t.find("Kand  Gr    Schritt  Art"), std::string::npos);
    EXPECT_NE(t.find("Stammbaum"), std::string::npos);
    EXPECT_NE(t.find("#0 start"), std::string::npos);
    EXPECT_NE(t.find("|-- #1 Operator A, Zweig 0"), std::string::npos) << t;
    EXPECT_NE(t.find("`-- #2 Operator A, Zweig 1"), std::string::npos) << t;
    EXPECT_NE(t.find("[expandiert"), std::string::npos);
    EXPECT_NE(t.find("[fertig: Ziel erreicht; Ergebnis]"), std::string::npos) << t;
    EXPECT_NE(t.find("[verworfen: Rang"), std::string::npos);
    std::ostringstream dot;
    write_beam_dot(r, dot);
    EXPECT_EQ(dot.str().rfind("digraph stammbaum {", 0), 0U);
    EXPECT_NE(dot.str().find("n0 -> n1;"), std::string::npos);
    EXPECT_NE(dot.str().find("penwidth=3"), std::string::npos);
}

} // namespace
} // namespace mko::test
