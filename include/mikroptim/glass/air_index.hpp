#pragma once

// Brechzahl der Luft nach Ciddor (1996) für den Raytatouille-Adapter (Folge zu M9-T2,
// Entscheidung des Maintainers 2026-11-19; docs/numerik.md, Abschnitt „Brechzahl der Luft“).
// Mikroptim rechnet relativ zu Luft (n_Luft = 1); Raytatouille rechnet absolut mit Ciddor-Luft.
// Eigene Umsetzung der Gleichungen (A21)–(A41) der NIST-Dokumentation (Engineering Metrology
// Toolbox, „Refractive Index of Air – Documentation“, Appendix A-III) mit denselben Festlegungen
// wie Raytatouille: trockene Luft (Wasserdampf-Molenbruch x_v = 0, alle Wasserdampfterme
// entfallen), CO₂-Gehalt 450 µmol/mol, Vakuumwellenlänge in µm, Temperatur in °C, Druck in atm
// (1 atm = 101 325 Pa). Ergebnis: Phasenbrechzahl der Luft, absolut. Gültig laut NIST-Rechner
// für 0,3–1,7 µm, −40 … 100 °C, 10–140 kPa; außerhalb wird ohne Warnung extrapoliert.

namespace mko {

// Eine physikalische Atmosphäre in Pa (NIST (A27), p_R1).
inline constexpr double kStandardAtmospherePa = 101325.0;
// CO₂-Gehalt [µmol/mol] wie Raytatouille (Material AIR).
inline constexpr double kAirCo2Ppm = 450.0;
// Standardumgebung von Raytatouille und Bezug der Katalogdaten: 20 °C, 1 atm.
inline constexpr double kAirReferenceTemperatureC = 20.0;
inline constexpr double kAirReferencePressureAtm = 1.0;

// Phasenbrechzahl trockener Luft (x_v = 0) mit kAirCo2Ppm CO₂ nach Ciddor 1996 in der Form der
// NIST-Gleichungen (A21)–(A41). wavelength_um: Vakuumwellenlänge [µm], temperature_c [°C],
// pressure_atm [atm]. Druck 0 ergibt genau 1.
inline double ciddor_air_index(double wavelength_um, double temperature_c, double pressure_atm) {
    const double p = pressure_atm * kStandardAtmospherePa; // [Pa]
    // (A22) Dispersionskonstanten der Standardluft [µm⁻²]
    const double k0 = 238.0185;
    const double k1 = 5792105.0;
    const double k2 = 57.362;
    const double k3 = 167917.0;
    // (A23) Kompressibilität (die Terme mit x_v entfallen), (A26) zweiter Virialterm
    const double a0 = 1.58123e-6;
    const double a1 = -2.9331e-8;
    const double a2 = 1.1043e-10;
    const double d = 1.83e-11;
    // (A27)–(A30) Bezugsbedingungen der Standardluft, deren Kompressibilität, Gaskonstante
    const double p_r1 = kStandardAtmospherePa;
    const double t_r1 = 288.15;
    const double z_a = 0.9995922115;
    const double r_gas = 8.314472;
    // (A31) S = 1/λ², (A32) Refraktivität der Standardluft
    const double s = 1.0 / (wavelength_um * wavelength_um);
    const double r_as = 1e-8 * (k1 / (k0 - s) + k3 / (k2 - s));
    // (A34) Molmasse der trockenen Luft, (A35) Refraktivität für den CO₂-Gehalt
    const double m_a = 0.0289635 + 1.2011e-8 * (kAirCo2Ppm - 400.0);
    const double r_axs = r_as * (1.0 + 5.34e-7 * (kAirCo2Ppm - 450.0));
    // (A36) Temperatur in K, (A37) Kompressibilität bei x_v = 0
    const double t = temperature_c;
    const double t_k = t + 273.15;
    const double p_over_t = p / t_k;
    const double z_m = 1.0 - p_over_t * (a0 + a1 * t + a2 * t * t) + p_over_t * p_over_t * d;
    // (A38) Dichte der Standardluft, (A40) Dichte der trockenen Luft
    const double rho_axs = p_r1 * m_a / (z_a * r_gas * t_r1);
    const double rho_a = p * m_a / (z_m * r_gas * t_k);
    // (A41) ohne Wasserdampfterm
    return 1.0 + (rho_a / rho_axs) * r_axs;
}

} // namespace mko
