// Optimierung nach den Einfüge-Operatoren (M6-T2) auf der CPU: Abnahme B (Einzellinse bei zwei
// Wellenlängen → Kittglied, Farblängsfehler um mehr als den Faktor 10 kleiner) und der Ablauf
// des Operators A aus der Diagnose (docs/optimierung.md, Abschnitt 6).

#include "insertion_optimization.hpp"

#include <gtest/gtest.h>

#include <span>

namespace mko::test {
namespace {

LmResult cpu_optimize(const VariableSet& set, const SystemBatchHost& base,
                      std::span<const double> x0, const LmOptions& o) {
    return optimize_lm_cpu(set, base, x0, o);
}

TEST(InsertionOptimization, OperatorBMakesAchromatCpu) {
    const AchromatRun a = achromat_run(cpu_optimize);
    print_achromat("CPU", a);
    check_achromat(a);
}

TEST(InsertionOptimization, OperatorAFromDiagnosisCpu) {
    const MeniscusRun m = meniscus_run(cpu_optimize);
    print_meniscus("CPU", m);
    check_meniscus(m);
    // Zweiter Zweig (−c₀): Hier enden beide Richtungen im selben Minimum (der Meniskus wird zur
    // sammelnden Linse, das Vorzeichen von c₀ verliert sich in den ersten Zyklen)
    const MeniscusRun alt = meniscus_run(cpu_optimize, -1.0);
    print_meniscus("CPU, -c0", alt);
    check_meniscus(alt);
}

} // namespace
} // namespace mko::test
