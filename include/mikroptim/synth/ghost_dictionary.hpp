#pragma once

// Erweiterung des Glas-Wörterbuchs der Geisterplatten um Flint- und anomale Gläser (M6-T12,
// docs/optimierung.md, Abschnitt 5 „Glas-Wörterbuch“ und „Erweitertes Wörterbuch“). Host-Code,
// Bibliothek Mikroptim::synth.
//
// Rangfolge: Jedes Katalogglas wird mit jedem anderen Katalogglas als dünner Achromat gelöst
// (preselect_glasses, Paare, Σ|w φ|/|φ| ≤ max_power_sum) und nach dem kleinsten spektralen Rest
// über `wavelengths_nm` bewertet (Rangfolge der Glasvorauswahl, M11-T5g). Die Rangfolge hängt
// nicht vom System ab: Jeder Aufruf wählt dieselben Gläser, ein wiederholter Aufruf auf dem
// erweiterten System ändert nichts. Klassen (Kennzahlen glass_properties):
//   - anomal:  |ΔP_g,F| ≥ anomalous_dpgf (Glasregel anomalous_pgf der Wissensbasis, 0,01),
//   - Flint:   sonst, wenn ν_d < 55 bei n_d < 1,60 bzw. ν_d < 50 bei n_d ≥ 1,60 (Kron-Flint-Grenze
//              im n_d-ν_d-Diagramm, Mikroptim-Annahme nach der üblichen Einteilung der Glaskarten).
// Gewählt werden ausdrücklich genannte Gläser (`names`) und danach `count` Gläser abwechselnd aus
// der Rangfolge der Flint- und der anomalen Gläser (Flint zuerst; ist eine Klasse erschöpft, die
// andere). Ein gewähltes Glas, das schon im Wörterbuch steht (Katalogname einer Platte oder
// Duplikat eines Modells, z. B. des Buchdahl-Fits im Startsystem), zählt mit und bekommt keine
// neue Platte: Das Wörterbuch enthält danach die `count` besten Gläser der beiden Klassen.

#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/model/system.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace mko {

enum class GhostGlassClass {
    Flint,     // Flintglas mit normaler Teildispersion
    Anomalous, // |ΔP_g,F| ≥ anomalous_dpgf
    Other,     // ausdrücklich genanntes Kronglas mit normaler Teildispersion
};

// "flint", "anomalous", "other"
std::string_view to_string(GhostGlassClass c);

struct GhostDictionaryGlassOptions {
    int count = 0;                  // automatisch gewählte Zusatzgläser (≥ 0)
    std::vector<std::string> names; // ausdrücklich genannte Zusatzgläser (Katalognamen)
    // Spektrum der Rangfolge: Linien g, F, e, d, C (436–656 nm, Zielspektrum der Spezifikation),
    // unabhängig vom Spektrum der Datei, damit alle Stufen dasselbe Wörterbuch erhalten.
    std::vector<double> wavelengths_nm{435.8343, 486.1327, 546.074, 587.5618, 656.2725};
    double max_power_sum = 6.0;   // wie SplitInitOptions::preselect_max_power_sum
    double anomalous_dpgf = 0.01; // Glasregel anomalous_pgf
    // Gleiche Gläser: |Δn_d| < duplicate_nd und |Δν_d| < duplicate_vd. Ein Kandidat, der einem
    // Modell im Wörterbuch gleicht (z. B. dem Buchdahl-Fit eines Katalogglases im Startsystem),
    // gilt als vorhanden; einer, der einem schon gewählten Glas gleicht, wird übersprungen.
    double duplicate_nd = 0.005;
    double duplicate_vd = 1.0;
};

struct GhostDictionaryGlass {
    std::string name;
    GhostGlassClass glass_class = GhostGlassClass::Other;
    bool automatic = false; // aus der Rangfolge (sonst ausdrücklich genannt)
    bool present = false;   // stand schon im Wörterbuch (keine neue Platte)
    double n_d = 0.0;
    double v_d = 0.0;
    double dp_gF = 0.0;
    std::string partner;            // Katalogglas des Achromats mit dem kleinsten Rest
    double spectral_residual = 0.0; // Rest des Paares (NaN ohne zulässiges Paar)
    double power_sum = 0.0;         // Σ|w φ|/|φ| des Paares (NaN ohne)
};

// Gläser des vorhandenen Wörterbuchs (Platten aller Lagen, ohne Wiederholung), Bezeichnung wie
// im Diagnosebericht (Katalogname bzw. Modell).
std::vector<MaterialSpec> ghost_dictionary_materials(const SystemDescription& system);

// Auswahl nach der Rangfolge oben (`system` nur für das Merkmal present). Wirft
// std::invalid_argument bei count < 0, einem unbekannten oder doppelt genannten Namen in `names`.
std::vector<GhostDictionaryGlass> select_ghost_dictionary_glasses(
    const SystemDescription& system, const GlassCatalog& catalog,
    const GhostDictionaryGlassOptions& options);

struct GhostDictionaryExtension {
    SystemDescription system;
    std::vector<GhostDictionaryGlass> glasses; // gewählte Gläser (Reihenfolge der Platten)
    int positions = 0;                         // erweiterte Lagen
    int added_plates = 0;
};

// Setzt an jeder Lage (Platten mit gleichem after_surface und distance) eine Platte je gewähltem
// Glas dazu (place_ghost_dictionary: Koeffizienten 0, Ordnung und r_g der ersten Platte der Lage,
// Variablen und Strafe wie die vorhandenen Platten). Gläser, die an einer Lage schon als
// Katalogglas liegen, werden dort übersprungen (wiederholter Aufruf ändert nichts). Die Merit
// bleibt bis auf Rundung gleich. Ohne Geisterplatten oder ohne gewählte Gläser unverändert.
GhostDictionaryExtension extend_ghost_dictionaries(const SystemDescription& system,
                                                   const GlassCatalog& catalog,
                                                   const GhostDictionaryGlassOptions& options);

// Textzeile der Auswahl, z. B. "SYN-21 (Flint, n_d 1.6571, v_d 34.47, dP -0.0020, Partner …)".
std::string ghost_dictionary_text(const std::vector<GhostDictionaryGlass>& glasses);

} // namespace mko
