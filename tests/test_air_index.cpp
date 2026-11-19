// Brechzahl der Luft nach Ciddor 1996 (Raytatouille-Adapter, Entscheidung Maintainer
// 2026-11-19): gegen veröffentlichte Werte des NIST-Ciddor-Rechners (Orakel) und die
// Grenzfälle der Formel.

#include "mikroptim/glass/air_index.hpp"
#include "tolerances.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include <cmath>
#include <cstdio>

namespace {

// NIST Engineering Metrology Toolbox, Ciddor-Rechner (emtoolbox.nist.gov/Wavelength/Ciddor.asp)
// bei 0 % relativer Feuchte und 450 µmol/mol CO₂, 9 Nachkommastellen: Vakuumwellenlänge [nm],
// Temperatur [°C], Druck [kPa], Brechzahl. Dieselben Werte nennt Raytatouille als Referenz
// (docs/quellen.md dort, abgefragt am 04.10.2026).
struct NistValue {
    double nm, celsius, kpa, n;
};
constexpr NistValue kNist[] = {
    {633, 20, 101.325, 1.0002718},    {400, 20, 101.325, 1.000277925},
    {1550, 20, 101.325, 1.000268586}, {300, 20, 101.325, 1.000286581},
    {1700, 20, 101.325, 1.000268479}, {633, 0, 101.325, 1.000291767},
    {633, 40, 101.325, 1.000254398},  {633, -20, 101.325, 1.000314912},
    {633, 20, 50, 1.000134099},       {633, 20, 120, 1.000321916},
    {500, 35, 80, 1.000205918},       {1000, 10, 110, 1.000302923},
};

// Zahl für RecordProperty (wissenschaftlich, 3 Stellen).
std::string sci(double v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.3e", v);
    return buf;
}

} // namespace

TEST(AirIndex, MatchesPublishedCiddorValues) {
    double worst = 0.0;
    for (const NistValue& v : kNist) {
        const double n = mko::ciddor_air_index(v.nm / 1000.0, v.celsius, v.kpa / 101.325);
        EXPECT_NEAR(n, v.n, mko::test::kCiddorPublished)
            << v.nm << " nm, " << v.celsius << " °C, " << v.kpa << " kPa";
        worst = std::max(worst, std::abs(n - v.n));
    }
    RecordProperty("max_abweichung", sci(worst));
}

TEST(AirIndex, LimitsAndMonotonicity) {
    // ohne Druck keine Luft
    EXPECT_EQ(mko::ciddor_air_index(0.5876, 20.0, 0.0), 1.0);
    // normale Dispersion: n fällt mit der Wellenlänge; n − 1 wächst mit dem Druck und fällt mit
    // der Temperatur (ideales Gas: etwa ∝ p/T)
    EXPECT_GT(mko::ciddor_air_index(0.4, 20.0, 1.0), mko::ciddor_air_index(0.7, 20.0, 1.0));
    const double r1 = mko::ciddor_air_index(0.55, 20.0, 1.0) - 1.0;
    const double r2 = mko::ciddor_air_index(0.55, 20.0, 2.0) - 1.0;
    EXPECT_NEAR(r2 / r1, 2.0, 2e-3); // Kompressibilität: kleine Abweichung vom idealen Gas
    const double r_warm = mko::ciddor_air_index(0.55, 40.0, 1.0) - 1.0;
    EXPECT_NEAR(r_warm / r1, 293.15 / 313.15, 2e-3);
}
