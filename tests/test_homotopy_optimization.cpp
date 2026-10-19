// Abnahme des Homotopie-Controllers (M6-T3, CPU): Einzellinse NA 0,3 → beugungsbegrenztes
// System bei NA 0,6 auf der Achse (docs/optimierung.md, Abschnitt 7).

#include "homotopy_test_systems.hpp"

#include "mikroptim/io/system_json.hpp"

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
