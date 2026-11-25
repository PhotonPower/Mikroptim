#pragma once

// Nebenbedingungen der Systembeschreibung (Abschnitt `constraints` von `mikroptim.system/1`,
// docs/dateiformat.md, ab M3-T3). Das Modell hält die Einträge so, wie sie in der Datei stehen
// (fehlende Felder bleiben leer, damit das Schreiben die Form erhält). Die Sollwerte stehen
// nicht hier, sondern in `target` (eine Quelle der Wahrheit); nur reine Grenzwerte ohne
// Gegenstück in `target` (κ_max der Petzval-Bedingung) stehen im Eintrag. Residuen der paraxialen
// Bedingungen: mikroptim/optim/paraxial_residuals.hpp, Definitionen in docs/optimierung.md,
// Abschnitte 2.2 und 2.3. Die L1-Strafe der Geisterkoeffizienten (ghost_penalty, M4-T6) steht
// als weiterer Eintrag hier; ihre Residuen: mikroptim/optim/ghost_penalty.hpp. Die Strafe
// außerhalb der Glas-Hülle (glass_hull, M6-T1): mikroptim/optim/glass_variables.hpp. Mechanische
// Bedingungen (M4-T3, Grenzen im Eintrag): mikroptim/optim/mechanical_constraints.hpp,
// Definitionen in docs/optimierung.md, Abschnitt 2.5.

#include "mikroptim/model/variables.hpp"

#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mko {

struct SystemDescription;

// Art einer Nebenbedingung. Unbekannte Typen lehnt der Leser mit FormatError ab.
enum class ConstraintType {
    FocalLength,     // Brennweite bei λ_ref, Soll target.focal_length (nur Gleichung)
    ParfocalLength,  // Abgleichlänge, Soll target.parfocal_length
    WorkingDistance, // Arbeitsabstand = Dicke der letzten Fläche, Soll target.working_distance
    ParaxialFocus,   // Bildebene im paraxialen Brennpunkt bei λ_ref, Soll 0 (nur Gleichung)
    Petzval,         // |κ_P| ≤ κ_max bei λ_ref (M4-T4), Grenze im Eintrag: max_curvature (nur
                     // Obergrenze)
    GhostPenalty,    // geglättete L1-Strafe der Geisterkoeffizienten (M4-T6): Gewicht μ,
                     // Ordnungsgewichte w_k = w₀ κ^k, Glättung ε; keine Sollgröße, kein Modus
    GlassHull,       // Strafe √w · Abstand außerhalb der Hülle der Katalog-Gläser für alle
                     // Glasvariablen (M6-T1): Gewicht w, Sicherheitsabstand nach innen; keine
                     // Sollgröße, kein Modus
    // Mechanische Bedingungen (M4-T3), einseitig √w · max(0, g), Grenzen im Eintrag, kein Modus;
    // mehrere Einträge eines Typs erlaubt (docs/optimierung.md, Abschnitt 2.5):
    MaxSemiDiameter,    // größte Strahlhöhe + clearance ≤ Halbdurchmesser (Extrema des Traces)
    MinCenterThickness, // Mittendicke t ≥ thickness (Glas und/oder Luft, medium)
    MaxCenterThickness, // Mittendicke t ≤ thickness
    MinEdgeThickness,   // Randdicke t − sag₁(h) + sag₂(h) ≥ thickness, h aus Halbdurchmessern
                        // oder Strahlhöhen (height) plus clearance
    MaxIncidenceAngle,  // größter Einfallswinkel ≤ angle [deg] (Extrema des Traces)
    // Verletzung der Sinusbedingung (OSC, Conrady) aus realen meridionalen Zonenstrahlen des
    // Achsbündels bei λ_ref (M11-T4): ein Residuum √w · OSC(ρ_k) pro Zone, Soll 0, nur
    // Gleichung, nur kollimierte Quelle, höchstens ein Eintrag (docs/optimierung.md, 2.7)
    SineCondition,
    // Verzeichnung (ADR-0019): relative Abweichung D_f = (ȳ_f − h_f)/h_f der Höhe des realen
    // Hauptstrahls bei λ_ref in der Bildebene (batch_chief_ray_height) von der idealen Höhe
    // h_f = f_soll tan θ_f; ein Residuum √w · max(0, |D_f| − D_max) pro Feld (0 auf der Achse),
    // Grenze im Eintrag (max_distortion, Pflicht), nur Obergrenze, nur kollimierte Quelle,
    // höchstens ein Eintrag (docs/optimierung.md, 2.8)
    Distortion,
};

// Medium nach der Fläche, für das eine Dicken-Bedingung gilt (M4-T3): Glas (jedes Material
// außer "AIR") oder Luft. Fehlt das Feld, gilt die Bedingung für beide.
enum class ConstraintMedium {
    Glass,
    Air,
};

// Halbdurchmesser h der Randdicke (M4-T3): mechanisch aus `semi_diameter` der Flächen (Standard,
// unabhängig vom Trace) oder größte Strahlhöhe auf den Flächen (Extrema des Traces).
enum class EdgeHeight {
    SemiDiameter,
    Rays,
};

// Art der Bedingung: Gleichung Ist = Soll oder einseitig Ist ≥ Soll (Untergrenze) bzw.
// Ist ≤ Soll (Obergrenze).
enum class ConstraintMode {
    Equal,
    Min,
    Max,
};

// Standardwerte fehlender Felder
inline constexpr double kDefaultConstraintWeight = 1.0;
inline constexpr int kDefaultParfocalReferenceSurface = 0; // Blende
inline constexpr double kDefaultParfocalOffset = 0.0;      // mm
// L1-Strafe der Geisterkoeffizienten (docs/optimierung.md, Abschnitte 2 und 3)
inline constexpr double kDefaultGhostOrderWeight = 1.0; // w₀
inline constexpr double kDefaultGhostOrderFactor = 2.0; // κ
inline constexpr double kDefaultGhostSmoothing = 1e-6;  // ε [mm]
// Glas-Hülle (docs/optimierung.md, Abschnitt 2.4): Gewicht w [Wellen²] für den Abstand d in
// normierten Glaskoordinaten (eine Einheit = Spannweite des Katalogs): d = 0,01 kostet bei
// w = 10⁴ eine Welle². Sicherheitsabstand nach innen (normiert), Standard 0.
inline constexpr double kDefaultGlassHullWeight = 1e4;
inline constexpr double kDefaultGlassHullMargin = 0.0;
// Mechanische Bedingungen (M4-T3): Fassungszugabe zur Strahlhöhe bzw. zum Halbdurchmesser [mm]
inline constexpr double kDefaultMechanicalClearance = 0.0;
// Sinusbedingung (M11-T4): Pupillenzonen ρ_k (Anteil des Pupillenradius) ohne Angabe, wie im
// Aberrationsbericht Zone 1/√2 und Rand.
inline constexpr double kDefaultSineConditionZones[] = {0.70710678118654752440, 1.0};

// Ein Eintrag von `constraints.items`, wie er in der Datei steht.
struct ConstraintSpec {
    ConstraintType type = ConstraintType::FocalLength;
    std::optional<ConstraintMode> mode; // fehlt = Equal
    std::optional<double> weight;       // Gewicht w >= 0, fehlt = 1
    // Nur ParfocalLength: Dateifläche, ab deren Scheitel die Abgleichlänge zählt (fehlt = 0),
    // und Abstand der Anlagefläche vor diesem Scheitel [mm] (fehlt = 0).
    std::optional<int> reference_surface;
    std::optional<double> offset;
    std::optional<std::string> comment;
    // Nur Petzval (Pflicht): größter zulässiger Betrag der Petzvalkrümmung κ_max [1/mm], ≥ 0.
    // Einzige Quelle der Grenze (nicht in `target`); ein Mindest-Petzvalradius R_min entspricht
    // κ_max = 1/R_min.
    std::optional<double> max_curvature;
    // Nur GhostPenalty: Geisterplatten, für die der Eintrag gilt ("plate"/"plates", Index in
    // ghost_plates; fehlt = alle), Ordnungsgewichte w_k = order_weight · order_factor^k (w₀,
    // κ, beide > 0) und Glättung ε > 0 [mm, Einheit des Dickenprofils]. Das Strafgewicht μ ist
    // `weight`.
    SurfaceSelection plate_selection = SurfaceSelection::None;
    std::vector<int> plates{};
    std::optional<double> order_weight{};
    std::optional<double> order_factor{};
    std::optional<double> smoothing{};
    // Nur GlassHull: Sicherheitsabstand δ ≥ 0 nach innen in normierten Glaskoordinaten (die
    // Strafe beginnt δ vor dem Rand; nur bei einer dreidimensionalen Hülle). Das Gewicht ist
    // `weight` (Standard kDefaultGlassHullWeight).
    std::optional<double> margin{};
    // Nur mechanische Typen (M4-T3): Dateiflächen ("surface"/"surfaces"; fehlt = Standardauswahl
    // des Typs, docs/dateiformat.md), Medium der Dicken-Bedingungen (fehlt = Glas und Luft),
    // Grenzen: semi_diameter [mm] (max_semi_diameter; fehlt = semi_diameter jeder Fläche),
    // thickness [mm] (Mitten- und Randdicke, Pflicht), angle [deg] (max_incidence_angle,
    // Pflicht); clearance ≥ 0 [mm] (Fassungszugabe, Standard 0) bei max_semi_diameter und
    // min_edge_thickness; height (nur min_edge_thickness, Standard SemiDiameter).
    SurfaceSelection surface_selection = SurfaceSelection::None;
    std::vector<int> surfaces{};
    std::optional<ConstraintMedium> medium{};
    std::optional<double> semi_diameter{};
    std::optional<double> thickness{};
    std::optional<double> angle{};
    std::optional<double> clearance{};
    std::optional<EdgeHeight> height{};
    // Verschiebungen θ ≥ 0 des Augmented Lagrangian (M11-T2, docs/optimierung.md, Abschnitt 2.6)
    // pro Residuum des Eintrags, in der Einheit von g: das Residuum ist √w · max(0, g + θ). Nur
    // bei einseitigen Bedingungen (supports_shift), Länge 1 bei paraxialen, sonst die Zahl der
    // Flächen aus constraint_surfaces; leer = alle 0. Passt die Länge nicht zur Zahl der
    // Residuen (z. B. nach einem Einfügen), gelten alle als 0. Zustand des Optimierers, nicht
    // Teil des Dateiformats (wird weder gelesen noch geschrieben).
    std::vector<double> shift{};
    // Nur SineCondition (M11-T4): Pupillenzonen ρ_k ∈ (0, 1], keine doppelt; leer = Standard
    // (kDefaultSineConditionZones).
    std::vector<double> zones{};
    // Nur Distortion (Pflicht, ADR-0019): größter zulässiger Betrag der relativen Verzeichnung
    // D_max ≥ 0 (Anteil, 0,01 = 1 %). Einzige Quelle der Grenze (nicht in `target`).
    std::optional<double> max_distortion{};

    friend bool operator==(const ConstraintSpec&, const ConstraintSpec&) = default;

    // Fehlender Modus: Equal, bei Petzval und Distortion Max (einzig erlaubter Modus).
    ConstraintMode mode_or_default() const {
        return mode.value_or(type == ConstraintType::Petzval || type == ConstraintType::Distortion
                                 ? ConstraintMode::Max
                                 : ConstraintMode::Equal);
    }
    // Fehlendes Gewicht: kDefaultGlassHullWeight bei GlassHull, sonst kDefaultConstraintWeight.
    double weight_or_default() const {
        return weight.value_or(type == ConstraintType::GlassHull ? kDefaultGlassHullWeight
                                                                 : kDefaultConstraintWeight);
    }
    double margin_or_default() const { return margin.value_or(kDefaultGlassHullMargin); }
    int reference_surface_or_default() const {
        return reference_surface.value_or(kDefaultParfocalReferenceSurface);
    }
    double offset_or_default() const { return offset.value_or(kDefaultParfocalOffset); }
    double order_weight_or_default() const {
        return order_weight.value_or(kDefaultGhostOrderWeight);
    }
    double order_factor_or_default() const {
        return order_factor.value_or(kDefaultGhostOrderFactor);
    }
    double smoothing_or_default() const { return smoothing.value_or(kDefaultGhostSmoothing); }
    double clearance_or_default() const { return clearance.value_or(kDefaultMechanicalClearance); }
    EdgeHeight height_or_default() const { return height.value_or(EdgeHeight::SemiDiameter); }
    std::vector<double> zones_or_default() const {
        return zones.empty() ? std::vector<double>(std::begin(kDefaultSineConditionZones),
                                                   std::end(kDefaultSineConditionZones))
                             : zones;
    }
};

// Glättung der Max-Reduktionen über Strahlen (M11-T1, ADR-0013): β > 0 ersetzt das harte
// Maximum der Strahlhöhe [1/mm] bzw. das harte Minimum von cos I [dimensionslos] durch
// Log-Sum-Exp; 0 = hart (Standard).
struct ExtremaSmoothingSpec {
    double height_beta = 0.0;
    double cos_beta = 0.0;

    friend bool operator==(const ExtremaSmoothingSpec&, const ExtremaSmoothingSpec&) = default;
};

// Abschnitt `constraints` der Systemdatei.
struct Constraints {
    std::vector<ConstraintSpec> items;
    std::optional<ExtremaSmoothingSpec> extrema_smoothing{}; // ab M11-T1

    friend bool operator==(const Constraints&, const Constraints&) = default;
};

// Prüft `system.constraints` nach den Regeln aus docs/dateiformat.md (Sollwert in `target`
// vorhanden, Modus und Felder zum Typ passend, Gewicht endlich und >= 0, Referenzfläche
// existiert, max_curvature bei petzval vorhanden, endlich und >= 0, sine_condition nur bei
// kollimierter Quelle mit Zonen in (0, 1], jeder Typ höchstens einmal
// außer ghost_penalty (dort jede Geisterplatte höchstens einmal) und den mechanischen Typen,
// paraxial_focus nur bei kollimierter Quelle, glass_hull ohne Modus mit margin endlich und ≥ 0,
// mechanische Typen mit Grenzen im Eintrag und gültiger Flächenauswahl). Wirft
// FormatError mit JSON-Pfad, z. B. `constraints.items[1].mode`. Wird von validate_system
// aufgerufen; ohne Abschnitt `constraints` passiert nichts.
void check_constraints(const SystemDescription& system);

// Sollwert eines Typs aus `system.target` (Soll 0 bei ParaxialFocus); leer, wenn er fehlt, und
// immer leer bei Petzval (die Grenze steht im Eintrag, siehe constraint_bound), GhostPenalty und
// GlassHull und den mechanischen Typen.
std::optional<double> constraint_target(const SystemDescription& system, ConstraintType type);

// Soll- bzw. Grenzwert eines Eintrags: max_curvature bei Petzval, thickness bei den Dicken,
// angle [deg] bei MaxIncidenceAngle, semi_diameter bei MaxSemiDiameter (leer: Halbdurchmesser
// jeder Fläche), sonst constraint_target(system, spec.type); leer, wenn er fehlt (immer bei
// GhostPenalty und GlassHull).
std::optional<double> constraint_bound(const SystemDescription& system, const ConstraintSpec& spec);

// Dateiflächen eines mechanischen Eintrags (M4-T3): die angegebenen in ihrer Reihenfolge, ohne
// Angabe (oder "all") die Standardauswahl des Typs aufsteigend: Flächen 1 … S − 1 bei
// max_semi_diameter und max_incidence_angle, bei den Dicken alle Spalte i = 0 … S − 2 (Fläche i
// bis i + 1), deren Medium nach Fläche i zu `medium` passt. Leer für andere Typen. Wirft
// FormatError wie check_constraints (ungültige oder leere Auswahl).
std::vector<int> constraint_surfaces(const SystemDescription& system, const ConstraintSpec& spec);

// Wahr für die paraxialen Typen (auch Petzval) und die Sinusbedingung (Paraxialdaten plus ein
// realer Zonenstrahl je Zone, ausgewertet mit den paraxialen Residuen), falsch für GhostPenalty,
// GlassHull und die mechanischen.
bool is_paraxial(ConstraintType type);
// Wahr für die mechanischen Typen (M4-T3).
bool is_mechanical(ConstraintType type);
// Wahr, wenn der Eintrag eine Verschiebung θ des Augmented Lagrangian tragen kann (M11-T2):
// paraxiale Bedingungen mit Modus min oder max (auch Petzval) und alle mechanischen. Nicht:
// Gleichungen, ghost_penalty, glass_hull (vektorielles Residuum, docs/optimierung.md, 2.6).
bool supports_shift(const ConstraintSpec& spec);

// Namen für Meldungen und Dateiformat: "focal_length", "parfocal_length", "working_distance",
// "paraxial_focus", "petzval", "ghost_penalty", "glass_hull", "max_semi_diameter",
// "min_center_thickness", "max_center_thickness", "min_edge_thickness", "max_incidence_angle",
// "sine_condition", "distortion" bzw. "equal", "min", "max", bzw. "glass", "air" bzw. "semi_diameter", "rays".
std::string_view to_string(ConstraintType type);
std::string_view to_string(ConstraintMode mode);
std::string_view to_string(ConstraintMedium medium);
std::string_view to_string(EdgeHeight height);
// Umkehrungen von to_string; leer bei unbekanntem Namen.
std::optional<ConstraintType> constraint_type_from_string(std::string_view name);
std::optional<ConstraintMode> constraint_mode_from_string(std::string_view name);
std::optional<ConstraintMedium> constraint_medium_from_string(std::string_view name);
std::optional<EdgeHeight> edge_height_from_string(std::string_view name);

} // namespace mko
