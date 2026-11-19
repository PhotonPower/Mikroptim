#pragma once

// Operatoren der Entwurfssteuerung (M10-T4, Host; docs/entwurfssteuerung.md, Abschnitt 4, und
// docs/optimierung.md, Abschnitt 13):
//
//   remove_element          Glasblock (Einzellinse oder Kittglied) entfernen, Spalte vereinigen
//   collapse_element        Stufe der Rückbau-Homotopie: Dicken und Brechkraft gegen 0
//   replace_glass           Glas eines Mediums tauschen (Glasvariable festsetzen)
//   place_ghost_dictionary  Geisterplatten-Wörterbuch mit Koeffizienten 0 einfügen
//   insert_aplanatic_shell  aplanatisch-konzentrische Schalenlinse aus dem realen Randstrahl
//   insert_null_doublet     Kittglied ohne Wirkung (drei Flächen gleicher Krümmung, Dicke 0)
//
// Wie die Einfüge-Operatoren (insertion.hpp) arbeiten alle auf der SystemDescription, geben ein
// neues, mit validate_system geprüftes System zurück und passen sämtliche Flächen- und
// Plattenbezüge an (Variablen, Geisterplatten, Nebenbedingungen). Merit unverändert bis auf
// Rundung: place_ghost_dictionary, insert_null_doublet und remove_element eines Null-Elements
// (is_null_element). Die übrigen ändern die Merit; die Nachoptimierung steuert der Aufrufer
// (mikroptim.design, CLI mikroptim insert).

#include "mikroptim/model/material.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/optim/insertion.hpp"

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace mko {

class GlassCatalog;

// ---------------------------------------------------------------------------------------------
// Allgemeine Umnummerierung (auch mit entfernten Flächen)
// ---------------------------------------------------------------------------------------------

// Abbildung alter auf neue Dateiflächen. surface[i]: neue Fläche der alten Fläche i (Krümmungs-
// und Glasvariablen, Bezugsfläche der Abgleichlänge, mechanische Bedingungen pro Fläche), −1:
// entfernt. gap[i]: neuer Spalt des alten Spalts nach Fläche i (Dickenvariablen, mechanische
// Dicken-Bedingungen, Geisterplatten), −1: entfernt; mehrere alte Spalte dürfen auf denselben
// neuen fallen (vereinigte Spalte).
struct SurfaceRemap {
    std::vector<int> surface;
    std::vector<int> gap;
};

// Wie renumber_surfaces, aber ohne die Forderung, dass die letzte Fläche die letzte bleibt, und
// mit entfernten Flächen: Einträge mit "all" bzw. Standardauswahl werden zuerst in die explizite
// Liste der alten Flächen umgewandelt; Bezüge auf entfernte Flächen entfallen, doppelte Bezüge
// (vereinigte Spalte, auch über mehrere Einträge eines Typs) bleiben nur beim ersten Vorkommen;
// Einträge ohne Flächen entfallen. Geisterplatten: after_surface über gap (distance passt der
// Aufrufer an; eine Platte in einem entfernten Spalt ist ein Fehler). Verschiebungen θ des
// Augmented Lagrangian werden gelöscht. Wirft std::invalid_argument bei ungültiger Abbildung oder
// einer entfernten Bezugsfläche der Abgleichlänge. Das Ergebnis ist nicht validiert.
SystemDescription remap_surfaces(const SystemDescription& system, const SurfaceRemap& remap,
                                 std::vector<Surface> new_surfaces);

// Variablen festsetzen: Bezüge der Krümmungen an `curvature_surfaces`, der Dicken der Spalte
// `thickness_gaps` und der Gläser nach `glass_surfaces` entfallen ("all" wird vorher zur
// expliziten Liste); Einträge ohne Flächen entfallen. Die Werte im System bleiben.
void freeze_variables(SystemDescription& system, std::span<const int> curvature_surfaces,
                      std::span<const int> thickness_gaps, std::span<const int> glass_surfaces);

// ---------------------------------------------------------------------------------------------
// Rückbau: remove_element, collapse_element
// ---------------------------------------------------------------------------------------------

// Letzte Fläche des Glasblocks, der an Dateifläche `first` beginnt: Medium vor `first` Luft,
// nach first … last − 1 Glas, nach `last` Luft (Einzellinse: last = first + 1, Kittglied mit k
// Gläsern: first + k). Wirft std::invalid_argument, wenn dort kein solcher Block beginnt
// (first < 1, Glas vor first, letzte Fläche ohne Luft dahinter).
int glass_block_end(const SystemDescription& system, int first);

// Null-Element: alle Glasdicken des Blocks 0 und alle Krümmungen gleich (Ebene = Krümmung 0).
// Dann sind alle Brechungen zueinander invers, und remove_element lässt die Merit bis auf
// Rundung gleich.
bool is_null_element(const SystemDescription& system, int first);

// Angehobene obere Schranke einer Dickenvariable bzw. des Arbeitsabstands (Rückbau und
// Rückbau-Homotopie verlängern Spalte). Die Operatoren heben die Schranke weiterhin an, melden es
// aber als Warnung (Entscheidung des Maintainers 2026-11-19).
struct BoundWidening {
    int gap = 0;                   // Dateifläche vor dem Spalt (Ergebnissystem)
    bool working_distance = false; // Schranke des Arbeitsabstands statt einer Dickenvariable
    double old_max = 0.0;          // bisherige Schranke [mm]
    double new_max = 0.0;          // neue Schranke = Dicke des Spalts [mm]
};

// Text der Warnung, z. B. „obere Schranke der Dicke nach Fläche 2 von 3 auf 4,5 mm angehoben“.
std::string bound_widening_warning(const BoundWidening& w);

struct RemovalResult {
    SystemDescription system;
    std::vector<int> new_of_old; // alte Dateifläche → neue, −1: entfernt
    int first_removed = 0;       // erste entfernte Dateifläche
    int removed_surfaces = 0;
    int merged_gap = 0;                 // Fläche vor dem vereinigten Luftspalt (first_removed − 1)
    std::vector<BoundWidening> widened; // angehobene Schranken (Warnungen)
};

// Entfernt den Glasblock first … glass_block_end(first): Spalt nach first − 1 wird
// t_{first−1} + Σ t_first … t_last (fortlaufend summiert; ist der Block die Frontgruppe, ist das
// der neue Arbeitsabstand). Variablen und Bedingungen über remap_surfaces (Dickenbezüge des
// Spalts nach `last` gehen auf den vereinigten Spalt über; liegt seine Dicke über der oberen
// Schranke der Dickenvariable bzw. des Arbeitsabstands, wird die Schranke angehoben und in
// `widened` als Warnung gemeldet);
// Geisterplatten nach `last` liegen danach nach first − 1 mit distance + t_{first−1} + … +
// t_{last−1}. Umkehrung von Operator A (insert_meniscus mit Dicke 0) und insert_null_doublet: Das
// Ergebnis ist das Ausgangssystem (Spaltdicke bis auf die Rundung von p + fl(T − p)). Wirft
// std::invalid_argument wie glass_block_end, FormatError, wenn das Ergebnis die Regeln der
// Systemdatei verletzt.
RemovalResult remove_element(const SystemDescription& system, int first);

// Stufe der Rückbau-Homotopie mit Anteil `fraction` ∈ (0, 1] des verbleibenden Wegs zum
// Null-Element: Glasdicken t ← (1 − f) t, Krümmungen der Flächen first + 1 … last
// c ← (1 − f) c + f c_first (bei f = 1 genau 0 bzw. c_first); die abgezogene Dicke kommt zum Spalt
// nach `last` (Lage der folgenden Scheitel bleibt; die obere Schranke seiner Dickenvariable wird
// bei Bedarf angehoben). Krümmungen, Glasdicken und Gläser des Blocks werden festgesetzt
// (freeze_variables). Aufeinanderfolgende Stufen mit f = 1/K, 1/(K − 1), …,
// 1 erreichen das Null-Element genau. Wirft std::invalid_argument wie glass_block_end oder bei
// f außerhalb (0, 1], FormatError wie remove_element. Mit `widened` ≠ nullptr werden angehobene
// Schranken dort angehängt (Warnungen).
SystemDescription collapse_element(const SystemDescription& system, int first, double fraction,
                                   std::vector<BoundWidening>* widened = nullptr);

// ---------------------------------------------------------------------------------------------
// Glas tauschen
// ---------------------------------------------------------------------------------------------

// Medium nach `surface` (ein Glas) wird `glass` (nicht Luft); eine Glasvariable dieses Mediums
// wird festgesetzt. Ob das Glas in den Katalogen existiert, prüft das Laden (prepare_system).
// Wirft std::invalid_argument bei Luft oder einer Fläche außerhalb.
void replace_glass(SystemDescription& system, int surface, const MaterialSpec& glass);

// ---------------------------------------------------------------------------------------------
// Geisterplatten-Wörterbuch
// ---------------------------------------------------------------------------------------------

struct GhostDictionaryPlacement {
    int after_surface = 0;                // Luftspalt nach dieser Dateifläche (nicht der letzten)
    double distance = 0.0;                // Abstand vom Scheitel, 0 ≤ distance ≤ Spaltdicke
    std::vector<MaterialSpec> glasses;    // eine Platte pro Glas (nicht Luft)
    int order = 6;                        // K ≥ 1, Koeffizienten a_1 … a_K = 0
    double norm_radius = 0.0;             // r_g; ≤ 0: größerer Halbdurchmesser der Spaltflächen
    std::optional<double> penalty_weight; // eigener Eintrag ghost_penalty mit diesem μ
};

struct GhostDictionaryResult {
    SystemDescription system;
    std::vector<int> new_plates;       // Indizes der neuen Platten
    std::vector<int> plate_new_of_old; // alte Platte → neue Platte
};

// Fügt die Platten mit Koeffizienten 0 nach allen Platten mit (after_surface, distance) ≤ der
// Lage ein (Sortierung bleibt) und nummeriert die Plattenbezüge um (Variablen
// ghost_coefficient, Nebenbedingung ghost_penalty). Variablen: Deckt kein Eintrag mit "all"
// bzw. ohne Plattenauswahl die neuen Platten ab, kommt ein Eintrag ghost_coefficient (alle
// Ordnungen) dazu. Strafe: mit penalty_weight ein eigener Eintrag (w₀, κ, ε wie der erste
// vorhandene Eintrag, sonst Standard; Einträge mit "all" werden zur Liste der alten Platten),
// sonst decken vorhandene Einträge mit "all" die Platten ab bzw. sie kommen in den ersten Eintrag
// mit Liste. Merit unverändert: Platten mit a = 0 haben keine Dicke, ihre Strafe ist 0. Wirft
// std::invalid_argument bei ungültiger Lage, ohne Glas, mit Luft, bei K < 1 oder ohne Strafe
// und ohne penalty_weight.
GhostDictionaryResult place_ghost_dictionary(const SystemDescription& system,
                                             const GhostDictionaryPlacement& placement);

// ---------------------------------------------------------------------------------------------
// Vorlagen der Linsenmodule (docs/entwurfssteuerung.md, Abschnitt 5.2)
// ---------------------------------------------------------------------------------------------

struct AplanaticShell {
    int after_surface = 0;      // Luftspalt nach dieser Dateifläche; auch die letzte Fläche
                                // (Schale vor dem Objekt, neue Frontlinse)
    double position = 0.0;      // Abstand der ersten Schalenfläche vom Scheitel von after_surface
    double thickness = 0.0;     // Mittendicke > 0
    MaterialSpec glass;         // Glas der Schale (nicht Luft)
    double semi_diameter = 0.0; // ≤ 0: größerer Halbdurchmesser der Spaltflächen
    // Arbeitsabstand so setzen, dass der Abstand der Objektebene zum paraxialen Fokus bleibt.
    bool refocus = true;
    NewCurvatureVariables curvature_variables{}; // beide Schalenflächen
};

struct AplanaticShellResult {
    InsertionResult insertion;
    double n_glass = 0.0;          // n der Schale bei λ_ref
    double intercept = 0.0;        // Achsenschnittpunkt Q des realen Randstrahls ab der ersten
                                   // Schalenfläche [mm] (vor der Schale)
    double image_intercept = 0.0;  // Q′ = Q n_m/n_G ab der ersten Schalenfläche (in der Schale)
    double c_aplanatic = 0.0;      // Krümmung der ersten Schalenfläche (in Tracerichtung)
    double c_concentric = 0.0;     // Krümmung der zweiten Schalenfläche
    double working_distance = 0.0; // Arbeitsabstand danach [mm]
};

// Aplanatisch-konzentrische Schalenlinse (Zhang 2019, Abschnitt 3.2; Welford, Abschnitt
// „aplanatische Flächen“). Der reale meridionale Randstrahl (ρ = 1, Achse, λ_ref, ohne
// Geisterplatten) schneidet im Spalt die Achse im Abstand Q von der ersten Schalenfläche. In
// Tracerichtung ist die erste Fläche aplanatisch für das Medium n_m davor und das Glas n_G:
// R₁ = Q n_m/(n_m + n_G); der Strahl zielt danach auf Q′ = Q n_m/n_G. Die zweite Fläche (Abstand
// `thickness`) ist konzentrisch zu Q′: R₂ = Q′ − t. Beide Flächen tragen für diesen Strahl
// mathematisch nichts zur sphärischen Aberration aller Ordnungen bei (S_FSPH, numerik.md,
// Abschnitt 14); nachfolgende Flächen sehen ein um n_G/n_m steileres Bündel. Geisterplatten im
// Spalt wie bei insert_meniscus, Dickenvariablen und mechanische Dicken-Bedingungen des Spalts
// gelten für beide Teilstücke. Wirft std::invalid_argument bei ungültigen Parametern, wenn der
// Randstrahl versagt oder achsparallel ist, Q′ = t, die Schale nicht in den Spalt passt oder der
// Arbeitsabstand nach dem Nachfokussieren nicht positiv ist; GlassError bei unbekanntem Glas.
AplanaticShellResult insert_aplanatic_shell(const SystemDescription& system,
                                            const GlassCatalog& catalog,
                                            const AplanaticShell& shell);

struct NullDoublet {
    int after_surface = 0;      // Luftspalt nach dieser Dateifläche (nicht der letzten)
    double position = 0.0;      // Abstand vom Scheitel von after_surface
    double curvature = 0.0;     // c₀ aller drei Flächen
    MaterialSpec front_glass;   // Glas nach der ersten Fläche (in Tracerichtung)
    MaterialSpec back_glass;    // Glas nach der Kittfläche
    double semi_diameter = 0.0; // ≤ 0: größerer Halbdurchmesser der Spaltflächen
    NewCurvatureVariables curvature_variables{}; // alle drei Flächen
};

// Kittglied ohne Wirkung: Operator A (insert_meniscus, Dicke 0) mit einer Kittfläche gleicher
// Krümmung (split_lens) und dem zweiten Glas dahinter. Drei Brechungen an derselben Stelle mit
// derselben Normale heben sich auf: Merit unverändert bis auf Rundung. Danach Dicken-Homotopie
// wie bei A. Wirft wie insert_meniscus.
InsertionResult insert_null_doublet(const SystemDescription& system, const NullDoublet& doublet);

} // namespace mko
