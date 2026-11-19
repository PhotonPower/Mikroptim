#pragma once

// Toleranzsatz der Systembeschreibung (Abschnitt `tolerances` von `mikroptim.system/1`,
// docs/dateiformat.md, ab M8-T4). Das Modell hält die Einträge so, wie sie in der Datei stehen
// (fehlende Felder bleiben leer, damit das Schreiben die Form erhält); resolve_tolerances löst
// sie in einzelne Zufallsgrößen pro Fläche bzw. Element auf und prüft dabei die Regeln.
// Stichproben und ihre Anwendung auf einen SystemBatch: mikroptim/tolerance/tolerance_model.hpp,
// Verfahren: docs/optimierung.md, Abschnitt „Monte-Carlo-Tolerierung“.

#include "mikroptim/model/variables.hpp"

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mko {

struct SystemDescription;

// Art einer Toleranz. Unbekannte Typen lehnt der Leser mit FormatError ab.
enum class ToleranceType {
    Radius,          // Krümmungsradius einer Fläche vom Typ sphere: relativ (ΔR/R) oder in Passen
    Thickness,       // Dicke bzw. Luftabstand nach einer Fläche (nicht der letzten) [mm]
    Index,           // Brechzahl des Mediums nach einer Fläche (Glas), konstant Δn
    Abbe,            // Abbe-Zahl des Mediums nach einer Fläche (Glas), relativ Δν/ν
    Decenter,        // Dezentrierung einer Fläche (Δx, Δy) [mm], ADR-0011
    Tilt,            // Verkippung einer Fläche (α, β) um ihren Scheitel
    ElementDecenter, // Dezentrierung eines Elements (Flächen first … last als starrer Körper)
    ElementTilt,     // Verkippung eines Elements um einen Drehpunkt auf der Achse
};

// Verteilung einer Toleranz: gleichverteilt in [−δ, δ] oder normalverteilt mit σ = δ/c,
// abgeschnitten bei ±δ (c = cutoff_sigma).
enum class ToleranceDistribution {
    Uniform,
    Normal,
};

// Kompensator der Monte-Carlo-Tolerierung: keiner oder der Arbeitsabstand (pro Stichprobe neu
// optimiert).
enum class CompensatorType {
    None,
    WorkingDistance,
};

// Drehpunkt einer Elementverkippung auf der Achse (nominelle Scheitellagen): Scheitel der ersten
// bzw. letzten Fläche, Mitte dazwischen oder ein Abstand vom Scheitel der ersten Fläche.
enum class ElementPivot {
    First,
    Center,
    Last,
    Offset,
};

// Ein Eintrag von `tolerances.items`, wie er in der Datei steht.
struct ToleranceSpec {
    ToleranceType type = ToleranceType::Radius;
    // Flächen (Index in SystemDescription::surfaces) bei Radius, Thickness, Index, Abbe,
    // Decenter, Tilt: "surface" (One), "surfaces" (List oder All). Bei den Elementtypen None.
    SurfaceSelection selection = SurfaceSelection::None;
    std::vector<int> surfaces;
    // Elemente [erste, letzte Fläche] bei ElementDecenter und ElementTilt: "element" (One),
    // "elements" (List oder All = alle Linsen und Kittglieder). Sonst None.
    SurfaceSelection element_selection = SurfaceSelection::None;
    std::vector<std::array<int, 2>> elements;
    // Grenzen ±δ (Feldname in der Datei): delta [mm] bei Thickness, Decenter, ElementDecenter,
    // [−] bei Index; relative bei Radius (ΔR/R) und Abbe (Δν/ν); fringes bei Radius (Passen,
    // Newtonringe bei test_wavelength_nm); arcmin oder deg bei Tilt und ElementTilt.
    std::optional<double> delta;
    std::optional<double> relative;
    std::optional<double> fringes;
    std::optional<double> test_wavelength_nm;
    std::optional<double> arcmin;
    std::optional<double> deg;
    // Nur ElementTilt: Drehpunkt ("first", "center", "last" oder Zahl = Abstand vom Scheitel der
    // ersten Fläche in +z [mm], dann pivot = Offset und pivot_offset).
    std::optional<ElementPivot> pivot;
    double pivot_offset = 0.0;
    // Verteilung und Abschneiden dieses Eintrags; fehlt: Werte des Abschnitts.
    std::optional<ToleranceDistribution> distribution;
    std::optional<double> cutoff_sigma;
    std::optional<std::string> comment;

    friend bool operator==(const ToleranceSpec&, const ToleranceSpec&) = default;
};

// Abschnitt `tolerances` der Systemdatei.
struct Tolerances {
    std::optional<ToleranceDistribution> distribution; // Standard für alle Einträge (Normal)
    std::optional<double> cutoff_sigma;                // Standard für alle Einträge (2)
    std::optional<CompensatorType> compensator;        // Standard WorkingDistance
    std::vector<ToleranceSpec> items;

    ToleranceDistribution distribution_or_default() const;
    double cutoff_sigma_or_default() const;
    CompensatorType compensator_or_default() const;

    friend bool operator==(const Tolerances&, const Tolerances&) = default;
};

// Standardwerte und Grenzen
inline constexpr ToleranceDistribution kDefaultToleranceDistribution =
    ToleranceDistribution::Normal;
inline constexpr double kDefaultToleranceCutoff = 2.0; // ±δ = ±2σ
// Abschneiden c: 0,5 ≤ c ≤ 6 (die Irwin-Hall-Näherung der Normalverteilung ist auf ±6 begrenzt;
// bei c = 0,5 werden noch 38 % der Ziehungen angenommen).
inline constexpr double kMinToleranceCutoff = 0.5;
inline constexpr double kMaxToleranceCutoff = 6.0;
// Prüfwellenlänge der Passen (ISO 10110: e-Linie, sofern nicht anders angegeben) [nm]
inline constexpr double kDefaultTestWavelengthNm = 546.07;
inline constexpr CompensatorType kDefaultCompensator = CompensatorType::WorkingDistance;

// Eine aufgelöste Toleranz: eine (Radius, Dicke, Brechzahl, Abbe) oder zwei (Dezentrierung,
// Verkippung; x und y bzw. α und β) unabhängige Zufallsgrößen ξ ∈ [−1, 1], Wert = limit · ξ.
struct ResolvedTolerance {
    ToleranceType type = ToleranceType::Radius;
    int item = 0;     // Eintrag in tolerances.items (für Meldungen)
    int surface = -1; // Dateifläche (Flächentypen), sonst −1
    int first = -1;   // Element: erste und letzte Dateifläche, sonst −1
    int last = -1;
    // Grenze in internen Einheiten: Radius relativ ε (c' = c/(1 + ε)) bzw. in Passen die
    // Krümmungsänderung Δc [1/mm] (c' = c + Δc, radius_fringes); Dicke und Dezentrierung [mm];
    // Brechzahl Δn; Abbe relativ e (Dispersion n(λ) − n(λ_ref) geteilt durch 1 + e);
    // Verkippung [rad].
    double limit = 0.0;
    bool radius_fringes = false;
    ToleranceDistribution distribution = kDefaultToleranceDistribution;
    double cutoff = kDefaultToleranceCutoff; // c (nur Normal)
    // Nur ElementTilt: Drehpunkt z_P [mm] auf der Achse im System der Blende (nominelle Lage,
    // Summe der Dicken der Datei ab Fläche 0).
    double pivot_z = 0.0;

    // Zahl der Zufallsgrößen: 2 bei Dezentrierung und Verkippung, sonst 1.
    int draws() const;

    friend bool operator==(const ResolvedTolerance&, const ResolvedTolerance&) = default;
};

// Löst `system.tolerances` auf und prüft die Regeln aus docs/dateiformat.md. Reihenfolge:
// Einträge in Dateireihenfolge, innerhalb eines Eintrags die Flächen bzw. Elemente wie angegeben
// (bei "all" aufsteigend). Diese Reihenfolge legt die Ziehungen pro Stichprobe fest. Ohne
// Abschnitt `tolerances` ist das Ergebnis leer. Wirft FormatError mit JSON-Pfad, z. B.
// `tolerances.items[2].surfaces[1]`. Wird von validate_system aufgerufen.
std::vector<ResolvedTolerance> resolve_tolerances(const SystemDescription& system);

// Nominelle Lage z_s des Scheitels der Dateifläche s auf der Achse im System der Blende:
// Summe der Dicken der Dateiflächen 0 … s − 1 (aufsteigend summiert).
double nominal_vertex_z(const SystemDescription& system, int surface);

// Elemente für "elements": "all": jede größte Folge von Flächen first … last mit Glas nach
// first … last − 1 und Luft nach last (Linsen und Kittglieder), first ≥ 1.
std::vector<std::array<int, 2>> all_elements(const SystemDescription& system);

// Namen für Meldungen und Dateiformat: "radius", "thickness", "index", "abbe", "decenter",
// "tilt", "element_decenter", "element_tilt"; "uniform", "normal"; "none", "working_distance";
// "first", "center", "last" (Offset: "offset").
std::string_view to_string(ToleranceType type);
std::string_view to_string(ToleranceDistribution distribution);
std::string_view to_string(CompensatorType compensator);
std::string_view to_string(ElementPivot pivot);
// Umkehrungen von to_string; leer bei unbekanntem Namen (ElementPivot ohne "offset").
std::optional<ToleranceType> tolerance_type_from_string(std::string_view name);
std::optional<ToleranceDistribution> tolerance_distribution_from_string(std::string_view name);
std::optional<CompensatorType> compensator_from_string(std::string_view name);
std::optional<ElementPivot> element_pivot_from_string(std::string_view name);

} // namespace mko
