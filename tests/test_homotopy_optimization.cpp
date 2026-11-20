// Abnahme des Homotopie-Controllers (M6-T3, CPU): Einzellinse NA 0,3 → beugungsbegrenztes
// System bei NA 0,6 auf der Achse (docs/optimierung.md, Abschnitt 7).

#include "homotopy_test_systems.hpp"

#include "mikroptim/io/system_json.hpp"
#include "mikroptim/optim/design_operators.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <iostream>

namespace mko::test {
namespace {

// Kurzlauf (CI): alle Phasen und ein Einfügen, Abbruch bei der höchsten Elementzahl.
TEST(HomotopySmoke, ShortRunCpu) {
    const HomotopyResult r = homotopy_smoke_run();
    write_homotopy_step_header(std::cout);
    for (const HomotopyStep& s : r.steps) {
        write_homotopy_step_line(s, std::cout);
    }
    write_homotopy_summary(r, std::cout);
    check_homotopy_smoke(r);
}

// Determinismus: gleiche Eingabe → gleiches Protokoll (bitgleich bis auf die Laufzeiten).
TEST(HomotopySmoke, DeterministicCpu) {
    const HomotopyResult a = homotopy_smoke_run();
    const HomotopyResult b = homotopy_smoke_run();
    expect_same_decisions(a, b, 0.0);
    EXPECT_EQ(a.system, b.system);
}

// Rückbau am optischen Modell (M6-T9): Einzellinse mit einem Null-Meniskus hinter der Linse (frei
// erfundenes Testsystem). Vor dem Abschluss wird der Meniskus (Seidel-Beitrag 0) entfernt, F_B
// bleibt; die Linse selbst ist notwendig: Ohne sie verfehlt F_B die Schwelle, der Versuch wird
// verworfen und der Zustand zurückgesetzt.
TEST(HomotopySmoke, PruneRemovesNullElementKeepsLens) {
    const LoadedSystem start = load_residual_system(kHomotopyStartFile);
    OpticalHomotopyOptions o = homotopy_optical_options();
    o.lm.max_iterations = 40;
    OpticalHomotopyModel model(start, o);
    MeniscusInsertion mi;
    mi.after_surface = 2;
    mi.position = 1.5;
    mi.curvature = -0.05;
    mi.glass.kind = MaterialKind::Catalog;
    mi.glass.name = "N-BK7";
    SystemDescription system = insert_meniscus(model.start_system(), mi).system;
    system.constraints = model.start_system().constraints; // Standardauswahl auch für den Meniskus
    const int frozen[] = {3, 4};                           // der Meniskus bleibt ein Null-Element
    freeze_variables(system, frozen, {}, {});
    ASSERT_EQ(glass_elements(system), 2);
    HomotopyOptions h;
    // F_max wie der Kurzlauf (die Einzellinse mit Geisterplatten genügt), Ziel 2 Wellen: das
    // Ziel ist am Ende erreicht, der Abschluss fügt nichts ein
    h.merit_threshold = (10.0 / 14.0) * (10.0 / 14.0);
    h.target_rms = 2.0;
    h.prune = true;
    h.prune_stages = 2;
    h.prune_min_elements = 0; // auch die Linse versuchen
    const HomotopyResult r = run_homotopy(model, system, model.start_point(), h);
    write_homotopy_step_header(std::cout);
    for (const HomotopyStep& s : r.steps) {
        write_homotopy_step_line(s, std::cout);
    }
    write_homotopy_summary(r, std::cout);
    EXPECT_EQ(r.status, HomotopyStatus::GoalReached) << r.message;
    EXPECT_EQ(r.start_elements, 2);
    EXPECT_EQ(r.elements, 1);
    EXPECT_EQ(r.removals, 1);
    EXPECT_EQ(r.removal_attempts, 2);
    std::vector<const HomotopyStep*> notes;
    const HomotopyStep* before = nullptr; // letzte Optimierung vor dem Rückbau
    for (const HomotopyStep& s : r.steps) {
        if (s.kind == HomotopyStepKind::Removal) {
            notes.push_back(&s);
        } else if (notes.empty() && s.solved && s.kind != HomotopyStepKind::RemovalStage) {
            before = &s;
        }
    }
    ASSERT_EQ(notes.size(), 2U);
    ASSERT_NE(before, nullptr);
    EXPECT_TRUE(notes[0]->accepted) << notes[0]->operation;
    EXPECT_NE(notes[0]->operation.find("Element ab Fläche 3"), std::string::npos);
    EXPECT_FALSE(notes[1]->accepted) << notes[1]->operation;
    EXPECT_NE(notes[1]->operation.find("Element ab Fläche 1"), std::string::npos);
    // Null-Element: F_B nach der Rückbau-Homotopie höchstens so groß wie davor (Nachoptimierung)
    const HomotopyStep& last_stage = r.steps[static_cast<std::size_t>(notes[0]->index) - 1];
    EXPECT_LE(last_stage.solve.wavefront + last_stage.solve.ghost_penalty,
              (before->solve.wavefront + before->solve.ghost_penalty) * (1.0 + 1e-9));
    // Ergebnis: die Linse des Starts (gleiche Flächenzahl, Glas der Linse)
    ASSERT_EQ(r.system.surfaces.size(), model.start_system().surfaces.size());
    EXPECT_EQ(r.system.surfaces[1].material.kind, MaterialKind::Buchdahl);
    EXPECT_EQ(r.system.surfaces[1].material.n0, model.start_system().surfaces[1].material.n0);
}

// Abnahme (Laufzeit in Release ~20 s, im Debug-Build Minuten; lokal mit
// --gtest_also_run_disabled_tests).
TEST(HomotopyOptimization, DISABLED_ApertureToNa06Cpu) {
    const HomotopyRun run = homotopy_acceptance_run({}, true);
    print_homotopy(run.result, std::cout);
    std::cout << write_system_json(run.realized) << '\n';
    check_homotopy_acceptance(run.result);
}

} // namespace
} // namespace mko::test
