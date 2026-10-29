#pragma once

// Einfüge-Operatoren der Systemsynthese (M6-T2, Host; docs/optimierung.md, Abschnitt 6):
//
//   A  insert_meniscus: Meniskus ohne Brechkraft (nach Bociort) – zwei Flächen gleicher
//      Krümmung c₀ im Abstand 0 mit einem Glas, in einem Luftspalt eingefügt.
//   B  split_lens: Aufspalten einer Linse an einer inneren Kittfläche in zwei Teile aus
//      demselben Glas; danach optional das Glas eines Teils freigeben (free_glass, M6-T1).
//
// Beide ändern die Topologie (neuer Batch nötig) und lassen die Merit bis auf Rundung
// unverändert. Alle Flächenbezüge der Systembeschreibung (Variablen, Geisterplatten,
// Nebenbedingungen) werden über eine zentrale Umnummerierung angepasst
// (renumber_surfaces). Initialisierung aus dem Diagnosebericht der Geisterplatten
// (meniscus_from_diagnosis, split_from_diagnosis), Erhöhen des Strafgewichts einer Platte
// (scale_ghost_penalty) und die beiden Startvarianten an einem Sattelpunkt
// (meniscus_saddle_starts). Die Verwaltung der Zweige ist M6-T4, die Homotopie M6-T3.

#include "mikroptim/model/material.hpp"
#include "mikroptim/model/system.hpp"

#include <array>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace mko {

class GlassCatalog;
struct GhostDiagnosis;

// ---------------------------------------------------------------------------------------------
// Umnummerierung der Dateiflächen
// ---------------------------------------------------------------------------------------------

// Ersetzt die Flächen von `system` durch `new_surfaces` und bildet alle Flächenbezüge über
// `new_of_old` ab (alte Dateifläche i → neue Dateifläche new_of_old[i]). Verlangt:
// new_of_old.size() == system.surfaces.size(), streng monoton steigend, new_of_old[0] = 0 und
// new_of_old.back() = new_surfaces.size() − 1 (Blende bleibt erste, Frontfläche letzte Fläche).
// Angepasst werden
//   - Variablen (`variables.items`): Flächenlisten der Krümmungen, Dicken und Gläser über die
//     Abbildung; Einträge mit "all" werden in die explizite Liste der alten Flächen (neu
//     nummeriert) umgewandelt, damit die Variablenmenge erhalten bleibt und eingefügte Flächen
//     nicht unbemerkt Variablen werden. Arbeitsabstand und Geisterkoeffizienten bleiben (die
//     Geisterplatten behalten ihre Indizes).
//   - Geisterplatten: after_surface über die Abbildung; distance bleibt (Platten in einem
//     geteilten Spalt verschiebt der Operator selbst).
//   - Nebenbedingungen: reference_surface (parfocal_length) und die Flächen der mechanischen
//     Bedingungen (M4-T3; Standardauswahl bzw. "all" → explizite Liste der alten Flächen aus
//     constraint_surfaces, bei Dicken-Bedingungen Fläche i = Spalt i → i + 1). Weitere Typen
//     mit Flächenbezug gehören hierher.
// Die Reihenfolge der Variablen bleibt erhalten. Wirft std::invalid_argument bei ungültiger
// Abbildung, FormatError bei ungültigem Abschnitt `variables` (vor der Umnummerierung).
SystemDescription renumber_surfaces(const SystemDescription& system,
                                    std::span<const int> new_of_old,
                                    std::vector<Surface> new_surfaces);

// Ergebnis eines Einfüge-Operators.
struct InsertionResult {
    SystemDescription system;
    std::vector<int> new_of_old; // alte Dateifläche → neue Dateifläche
    int first_new_surface = 0;   // erste eingefügte Dateifläche
    int new_surfaces = 0;        // Zahl der eingefügten Flächen (A: 2, B: 1)
};

// Neue Krümmungsvariablen für eingefügte Flächen (ein eigener Eintrag `curvature`).
struct NewCurvatureVariables {
    bool enabled = true;
    std::optional<double> min; // Schranken des Eintrags (fehlt = offen)
    std::optional<double> max;
};

// ---------------------------------------------------------------------------------------------
// Operator A: Meniskus ohne Brechkraft
// ---------------------------------------------------------------------------------------------

struct MeniscusInsertion {
    int after_surface = 0;      // Dateifläche, nach der der Luftspalt liegt (Medium Luft, nicht
                                // die letzte Fläche)
    double position = 0.0;      // Abstand der Vorderfläche vom Scheitel von after_surface [mm],
                                // 0 ≤ position ≤ Spaltdicke − thickness
    double curvature = 0.0;     // c₀ beider Flächen [1/mm]
    MaterialSpec glass;         // Glas des Meniskus (nicht Luft)
    double thickness = 0.0;     // Mittendicke [mm]; 0: Merit unverändert (die Homotopie, M6-T3,
                                // bringt sie danach auf die Mindestdicke)
    double semi_diameter = 0.0; // ≤ 0: größerer Halbdurchmesser der beiden Spaltflächen
    NewCurvatureVariables curvature_variables{}; // beide Meniskusflächen
};

// Fügt den Meniskus ein: Dateifläche i = after_surface behält Medium Luft und hat danach die
// Dicke `position`; neue Flächen i + 1 (Typ sphere, c₀, Glas, Dicke `thickness`) und i + 2
// (Typ sphere, c₀, Luft, Dicke T − position − thickness); alle späteren Flächen rücken um 2.
// Geisterplatten im Spalt mit distance ≤ position bleiben vor dem Meniskus, die übrigen
// (distance ≥ position + thickness) liegen danach nach Fläche i + 2 mit distance − position −
// thickness; eine Platte im Glas ist ein Fehler. Eine Dickenvariable des Spalts gilt danach für
// beide Teilstücke (i und i + 2, gleicher Eintrag und gleiche Schranken). Mit Dicke 0 sind beide
// Brechungen zueinander invers: die Merit bleibt bis auf Rundung gleich (Test: |ΔF| ≤ 10⁻¹²
// relativ). Wirft std::invalid_argument bei ungültigen Parametern, FormatError, wenn das
// Ergebnis die Regeln der Systemdatei verletzt (z. B. ein Teilstück unter der Schranke der
// Dickenvariable).
InsertionResult insert_meniscus(const SystemDescription& system, const MeniscusInsertion& m);

// Zwei Startvarianten an dem durch den Meniskus erzeugten Sattelpunkt (docs/optimierung.md,
// Abschnitt 6): Krümmungen der Meniskusflächen first, first + 1 gestört um ±(δ_a, δ_b), also
// c_a = c₀ ± δ_a, c_b = c₀ ± δ_b. Standard (δ, −δ): der Meniskus erhält die dünne Brechkraft
// ±2 (n − 1) δ, die beiden Abstiegsrichtungen. Wirft std::invalid_argument, wenn first und
// first + 1 keine Sphären sind.
std::array<SystemDescription, 2> meniscus_saddle_starts(const SystemDescription& system,
                                                        int first_surface, double delta_front,
                                                        double delta_back);

// ---------------------------------------------------------------------------------------------
// Operator B: Aufspalten in ein Kittglied
// ---------------------------------------------------------------------------------------------

struct LensSplit {
    int lens_surface = 1;       // Vorderfläche i der Linse (Medium nach i ist ein Glas), i + 1 die
                                // Rückfläche
    double curvature = 0.0;     // c_s der Kittfläche [1/mm]
    double fraction = 0.5;      // Anteil der Mittendicke vor der Kittfläche, 0 ≤ fraction ≤ 1
    double semi_diameter = 0.0; // ≤ 0: kleinerer Halbdurchmesser der beiden Linsenflächen
    NewCurvatureVariables curvature_variable{}; // Kittfläche
};

// Teilt die Linse: Fläche i hat danach die Dicke t₁ = fl(fraction · t), neue Fläche i + 1
// (Typ sphere, c_s, dasselbe Material wie i, Dicke t₂ = fl(t − t₁)); alle späteren Flächen
// rücken um 1. Eine Dicken- oder Glasvariable der Linse gilt danach für beide Teile (gleicher
// Eintrag). An der Kittfläche ist μ = 1: Richtung und Merit bleiben bis auf Rundung gleich.
// Wirft std::invalid_argument bei ungültigen Parametern, FormatError wie insert_meniscus.
InsertionResult split_lens(const SystemDescription& system, const LensSplit& s);

// Gibt das Glas des Mediums nach Dateifläche `surface` als Variable frei (Eintrag `glass`,
// M6-T1; nichts, wenn es schon Variable ist). Optional vorher ein anderes Startglas. Danach
// convert_glass_variables und die Glas-Hülle (Nebenbedingung glass_hull) wie in M6-T1. Wirft
// std::invalid_argument bei einer Fläche außerhalb oder Medium Luft.
void free_glass(SystemDescription& system, int surface,
                const std::optional<MaterialSpec>& start_glass = std::nullopt);

// Krümmung c_s der Kittfläche für ein dünnes Achromat aus der Linse c₁, c₂ (Glas mit Brechzahl
// n und Abbezahl ν₁, vorderer Teil) und einem hinteren Teil mit Abbezahl ν₂: Teilbrechkräfte
// φ₁ = φ ν₁/(ν₁ − ν₂), φ₂ = −φ ν₂/(ν₁ − ν₂) mit φ = (n − 1)(c₁ − c₂), also
// c_s = c₁ − φ₁/(n − 1) = c₁ − (c₁ − c₂) ν₁/(ν₁ − ν₂) (unabhängig von n). Wirft
// std::invalid_argument bei ν₁ = ν₂ oder nicht endlichen Werten.
double achromat_split_curvature(double c_front, double c_back, double abbe_front, double abbe_back);

// Abbezahl ν_d = (n_d − 1)/(n_F − n_C) eines Materials (Katalog, Buchdahl, konstant: +∞).
double abbe_number(const MaterialSpec& material, const GlassCatalog& catalog);

// ---------------------------------------------------------------------------------------------
// Initialisierung aus dem Diagnosebericht
// ---------------------------------------------------------------------------------------------

// Strafterme einer Geisterplatte: Gewicht μ der Platte mit `factor` multiplizieren. Deckt der
// Eintrag `ghost_penalty` weitere Platten ab, wird die Platte daraus entfernt ("all" → Liste
// der übrigen) und erhält einen eigenen Eintrag mit μ · factor und denselben Ordnungsgewichten
// und derselben Glättung. Wirft std::invalid_argument, wenn keine Strafe die Platte abdeckt oder
// factor nicht endlich und > 0 ist.
void scale_ghost_penalty(SystemDescription& system, int plate, double factor);

// Gewicht μ der Strafe einer Platte (std::nullopt ohne Strafe).
std::optional<double> ghost_penalty_weight(const SystemDescription& system, int plate);

// Koeffizient von r⁴ im Dickenprofil einer Geisterplatte: t₄ = Σ_k a_k q_k / r_g⁴ mit dem
// Koeffizienten q_k = (−1)^k C(k, 2) C(k + 2, 2) von u² in P_k(2u − 1) (q₂ = 6, q₃ = −30,
// q₄ = 90). Maß der sphärischen Korrektur der Platte (t₂ beschreibt die Brechkraft).
double ghost_profile_t4(const GhostPlate& plate);

struct MeniscusInitOptions {
    // Lage: Index in diagnosis.positions (< 0: die empfohlene Lage).
    int position = -1;
    // Betrag von c₀ (≤ 0: automatisch, Krümmung der dem Meniskus näheren Spaltfläche, sonst
    // der anderen, sonst 0 – eine Planplatte).
    double curvature_magnitude = 0.0;
    double thickness = 0.0;
    NewCurvatureVariables curvature_variables{};
};

struct MeniscusPlan {
    MeniscusInsertion insertion;
    int plate = -1;    // Quelle: aktivste Platte der Lage (größte Σ|a|)
    int position = -1; // Index in diagnosis.positions
    double t4 = 0.0;   // ghost_profile_t4 der Platte
    std::string reason;
};

// Operator A aus dem Bericht: Lage = die empfohlene (oder options.position) mit after_surface
// und distance; Glas der aktivsten Platte dort; |c₀| nach options; Vorzeichen von c₀ =
// Vorzeichen von (n_G − n_m)·t₄ der Platte (Ordnung 2 dominiert: Vorzeichen von a₂), bei t₄ = 0
// das von φ_G. Die Festlegung ist eine Heuristik; die Gegenrichtung liefert
// meniscus_saddle_starts bzw. ein Lauf mit −c₀ (docs/optimierung.md, Abschnitt 6). Wirft
// std::invalid_argument ohne aktive Platte an der Lage oder wenn Bericht und System nicht
// zusammenpassen.
MeniscusPlan meniscus_from_diagnosis(const SystemDescription& system,
                                     const GhostDiagnosis& diagnosis,
                                     const MeniscusInitOptions& options = {});

struct SplitInitOptions {
    int position = -1;     // wie MeniscusInitOptions::position
    double fraction = 0.5; // Dickenaufteilung
    int free_part = 1;     // Teil, dessen Glas freigegeben wird: 0 vorn, 1 hinten
    // Krümmung der Kittfläche: true – dünnes Achromat mit dem Zielglas (achromat_split_
    // curvature), false – Mittel der Außenkrümmungen.
    bool achromat_curvature = true;
    // Mindest-Randdicke beider Teile [mm] bei der Höhe h = kleinerer Halbdurchmesser der Linse
    // (höchstens 0,99/|c| jeder beteiligten Fläche; hat schon die ungeteilte Linse dort eine
    // kleinere Randdicke, die größte Höhe, bei der sie passt): Verletzt c_s sie, wird c_s auf der
    // Strecke zum Mittel der Außenkrümmungen so weit zurückgenommen, bis beide Teile sie
    // einhalten (Bisektion, 60 Schritte; im ungünstigsten Fall bis zum Mittel).
    double min_edge_thickness = 0.0;
    NewCurvatureVariables curvature_variable{};
    // Algebraische Glasvorauswahl (M11-T5): Zielglas = bester Partner des Linsenglases aus dem
    // Katalog (preselect_glasses, Paare, kleinstes sekundäres Spektrum, Σ|φ|/φ ≤
    // preselect_max_power_sum) statt des Glases der Geisterplatte; c_s aus den Anteilen des
    // dünnen Achromats in η. Bei Katalogglas oder Glasvariable (Buchdahl, gleiche Koordinate wie
    // der Katalog-Fit) der Linse; sonst wie ohne.
    bool preselect = false;
    double preselect_max_power_sum = 6.0;
    // Das freigegebene Glas startet beim Zielglas statt beim Linsenglas (Merit ändert sich; der
    // Start liegt dann in der Nähe eines Achromats).
    bool start_at_target = false;
};

struct SplitPlan {
    LensSplit split;
    int free_surface = -1;     // Dateifläche des freigegebenen Teils im neuen System
    MaterialSpec target_glass; // Glas der Platte, deren Dispersion am weitesten von der Linse
                               // abweicht (bei B anomal: größtes ΔP_g,F)
    int plate = -1;            // Platte des Zielglases
    int position = -1;
    double abbe_lens = 0.0;
    double abbe_target = 0.0;
    double achromat_curvature = 0.0; // c_s des dünnen Achromats vor der Rücknahme (NaN ohne)
    bool preselected = false;        // Zielglas aus der Glasvorauswahl (M11-T5)
    double secondary_spectrum = 0.0; // Rest der Vorauswahl |Σ φ η₂|/|φ| (NaN ohne)
    std::string reason;
};

// Operator B aus dem Bericht: Linse = die dem Spalt der Lage benachbarte Glaslinse (vor oder
// hinter dem Spalt, die nähere zur Plattenlage); Zielglas = Glas der aktiven Platte mit der
// größten Abweichung |1/ν_Platte − 1/ν_Linse| (bei BAnomalous die mit dem größten ΔP_g,F);
// c_s nach options (mit Rücknahme bei zu kleiner Randdicke). Das freigegebene Glas startet beim
// Linsenglas (Merit unverändert), das Zielglas bestimmt nur c_s. Wirft std::invalid_argument ohne
// aktive Platte oder ohne Linse am Spalt.
SplitPlan split_from_diagnosis(const SystemDescription& system, const GlassCatalog& catalog,
                               const GhostDiagnosis& diagnosis,
                               const SplitInitOptions& options = {});

// Operator B ohne Geisterdiagnose (Farbregel, M11-T5c): Linse ab Fläche `lens`, Zielglas und
// Kittkrümmung aus der Glasvorauswahl (options.preselect wird erzwungen), Platte −1. Wirft
// std::invalid_argument, wenn an `lens` keine Linse beginnt oder die Vorauswahl keinen Partner
// findet (Linsenglas kein Katalogglas, alle Paare über preselect_max_power_sum).
SplitPlan split_for_color(const SystemDescription& system, const GlassCatalog& catalog, int lens,
                          const SplitInitOptions& options = {});

} // namespace mko
