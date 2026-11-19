#pragma once

// Einfache Herstellbarkeitsprüfungen (M10-T3, Host): Halbkugelgrad, Mitten- und Randdicken,
// Luftränder, Verhältnis Durchmesser zu Mittendicke und Zentrierbarkeit (Meniskusform) pro
// Fläche bzw. Spalt der Systemdatei, nur aus der Geometrie der Datei (kein Trace). Grenzwerte
// konfigurierbar; Standardwerte und Herkunft: docs/optimierung.md, Abschnitt „Herstellbarkeit“.
// Die Befunde sind nur Diagnose (Warnungen bzw. Fehler im Bericht), nie Nebenbedingungen der
// Optimierung (Entscheidung des Maintainers 2026-11-19).
// Pfeilhöhe und Randdicke wie die mechanischen Nebenbedingungen (mikroptim/optim/
// mechanical_constraints.hpp).

#include "mikroptim/model/system.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace mko {

// Grenzwerte. Eine Grenze ≤ 0 (bzw. bei Maxima ≤ 0) schaltet die Warnung ab; Fehler
// (geometrisch unmöglich) bleiben. „Mikroptim-Annahme“: Grenzwert ohne Quelle.
struct ManufacturabilityLimits {
    double max_hemisphere = 0.95;      // h/|R| (Zhang 2019, §3.2: Hyperhemisphäre nicht fassbar)
    double min_center_thickness = 0.5; // Glas [mm] (Mikroptim-Annahme)
    // Randdicke des Glases ≥ max(min_edge_thickness, min_edge_thickness_relative · D) mit dem
    // Durchmesser D = 2h des Elements (Mikroptim-Annahme; beide 0 schalten die Warnung ab)
    double min_edge_thickness = 0.5;           // [mm]
    double min_edge_thickness_relative = 0.05; // Anteil des Durchmessers
    double min_air_edge = 0.0;      // Luftrand [mm]; < 0 ist immer ein Fehler (Berührung)
    double max_aspect_ratio = 10.0; // 2h / t des Glases (Mikroptim-Annahme, Faustregel)
    // Z = (h/2) |c₁ − c₂|: darunter ist die Glockenzentrierung ungeeignet (Mikroptim-Annahme,
    // Richtwert), andere Zentrierverfahren sind nötig; kein Befund „nicht herstellbar“
    double min_centering = 0.1;
};

enum class ManufacturabilityCheckType {
    Hemisphere,      // Fläche s (sphere): h_s |c_s|, h = Halbdurchmesser der Fläche
    CenterThickness, // Glas nach Fläche s: t_s
    EdgeThickness,   // Glas nach Fläche s: t_s − sag(c_s, h) + sag(c_{s+1}, h)
    AirEdge,         // Luft nach Fläche s (nicht die letzte): Randabstand wie EdgeThickness
    AspectRatio,     // Glas nach Fläche s: 2h / t_s
    Centering,       // Glas nach Fläche s: Z = (h/2) |c_s − c_{s+1}|
};

enum class CheckSeverity {
    Ok,
    Warning, // Grenzwert überschritten
    Error,   // geometrisch unmöglich: h ≥ |R|, Randdicke bzw. Luftrand < 0
};

// Eine Prüfung. Spaltprüfungen gelten für das Stück von Fläche `surface` bis `surface + 1` mit
// h = max(h_s, h_{s+1}) aus den Halbdurchmessern der Datei.
struct ManufacturabilityCheck {
    ManufacturabilityCheckType type = ManufacturabilityCheckType::Hemisphere;
    int surface = 0;
    double value = 0.0;
    double limit = 0.0; // verwendete Grenze (Warnung; Randdicke: durchmesserabhängig)
    CheckSeverity severity = CheckSeverity::Ok;
};

struct ManufacturabilityReport {
    ManufacturabilityLimits limits;
    std::vector<ManufacturabilityCheck> checks; // nach Fläche, je Fläche in Reihenfolge der Typen
    int warnings = 0;
    int errors = 0;
};

// Prüft alle Flächen und Spalte von `system` (Geisterplatten zählen nicht; Dicken der Datei).
// Halbkugelgrad: Fehler bei h|c| ≥ 1 (Halbdurchmesser ≥ Radius), Warnung über max_hemisphere.
// Glas (Material nach Fläche s nicht Luft, s < letzte Fläche): Mittendicke, Randdicke (Fehler
// < 0 oder wenn eine der beiden Flächen h|c| ≥ 1 hat), Durchmesser/Dicke, Zentrierbarkeit.
// Luft zwischen zwei Flächen (nicht der Arbeitsabstand nach der letzten): Luftrand.
// Wirft std::invalid_argument bei nicht endlichen Grenzwerten.
ManufacturabilityReport check_manufacturability(const SystemDescription& system,
                                                const ManufacturabilityLimits& limits = {});

// Grenze der Randdicke für ein Element mit Halbdurchmesser h:
// max(min_edge_thickness, min_edge_thickness_relative · 2h).
double edge_thickness_limit(const ManufacturabilityLimits& limits, double semi_diameter);

// Kurzer deutscher Text zu einem Befund (Warnung oder Fehler) für Berichte, z. B. bei
// Zentrierbarkeit „Glockenzentrierung ungeeignet (Z < 0,1), andere Zentrierverfahren nötig“;
// leer bei Ok.
std::string finding_message(const ManufacturabilityCheck& check);

// "hemisphere", "center_thickness", "edge_thickness", "air_edge", "aspect_ratio", "centering";
// "ok", "warning", "error".
std::string_view to_string(ManufacturabilityCheckType type);
std::string_view to_string(CheckSeverity severity);

} // namespace mko
