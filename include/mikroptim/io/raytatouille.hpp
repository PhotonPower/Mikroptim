#pragma once

// Adapter zwischen Raytatouille (`*.rtt.json`, Schema 0.11) und `mikroptim.system/1` (M9-T2,
// ADR-0008). Abgebildet wird der Teilumfang, den beide Formate gleich beschreiben: sequenziell,
// koaxial, sphärisch, Objekt im Unendlichen, Blende als erste Fläche in Luft. Was sich nicht
// abbilden lässt, wird nie still weggelassen: Elemente, die die Optik ändern würden (Asphären,
// Dezentrierung, Spiegel, endliches Objekt, unbekannte Gläser, ...), sind Fehler
// (RttConversionError mit allen gefundenen Stellen); Angaben ohne Gegenstück, die die Optik
// nicht ändern (Variablen, Merit-Funktion, Toleranzen, Namen, ...), entfallen mit Warnung.
// Regeln und Konventionen: docs/dateiformat.md, Abschnitt „Raytatouille-Adapter“.

#include "mikroptim/model/system.hpp"

#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mko {

// Schema-Version, die system_to_rtt_json schreibt. Gelesen werden 0.4 bis 0.11.
inline constexpr std::string_view kRttSchemaVersion = "0.11.0";

// Eine nicht abbildbare Stelle: `location` ist bei Raytatouille-Eingaben ein JSON-Pointer
// (`/root/children/1/surfaces/0/shape`), bei Mikroptim-Eingaben ein Pfad wie in FormatError
// (`surfaces[2].decenter`).
struct RttIssue {
    std::string location;
    std::string message;
};

// Die Eingabe enthält nicht abbildbare Elemente. what() listet alle Stellen zeilenweise.
class RttConversionError : public std::runtime_error {
public:
    explicit RttConversionError(std::vector<RttIssue> issues);
    const std::vector<RttIssue>& issues() const noexcept { return issues_; }

private:
    std::vector<RttIssue> issues_;
};

struct RttImportOptions {
    // Einträge von `glass_catalogs` der erzeugten Systembeschreibung (unverändert übernommen).
    std::vector<std::string> glass_catalogs;
    // Gibt es das Glas NAME (aus `KATALOG:NAME`) in den Mikroptim-Katalogen? Ohne Abfrage ist
    // jedes Katalogglas ein Fehler.
    std::function<bool(std::string_view)> glass_exists;
    // Optional: Katalogname (Großbuchstaben), unter dem Mikroptim das Glas führt. Weicht er von
    // KATALOG ab, gibt es eine Warnung (das Glas wird trotzdem übernommen).
    std::function<std::string(std::string_view)> glass_catalog_name;
    // Halbdurchmesser für Flächen ohne Apertur [mm] (mit Warnung); fehlt er, ist eine Fläche
    // ohne kreisförmige Apertur ein Fehler.
    std::optional<double> default_semi_diameter;
    // Name des Systems, wenn die Datei keinen `name` hat.
    std::string fallback_name = "Raytatouille-System";
};

struct RttExportOptions {
    // Katalogname KATALOG für das Mikroptim-Glas NAME (Referenz `KATALOG:NAME`); leer oder ohne
    // Abfrage ist das Glas ein Fehler.
    std::function<std::string(std::string_view)> glass_catalog_name;
};

// Raytatouille → Mikroptim. Die Beschreibung wird validiert (validate_system), Glasnamen über
// `options.glass_exists` geprüft. Warnungen werden als "<pointer>: <meldung>" angehängt.
// Fehler: FormatError bei JSON-Syntaxfehlern, RttConversionError bei nicht abbildbaren Stellen.
SystemDescription system_from_rtt_json(std::string_view rtt_text,
                                       const RttImportOptions& options = {},
                                       std::vector<std::string>* warnings = nullptr);

// Mikroptim → Raytatouille: kanonische Form (flacher Baum, alle Lagen relativ zur
// vorangehenden Fläche, ein automatischer Pfad), Zahlen verlustfrei. Warnungen wie oben mit
// Mikroptim-Pfaden. Fehler: FormatError (ungültige Beschreibung), RttConversionError.
std::string system_to_rtt_json(const SystemDescription& system,
                               const RttExportOptions& options = {},
                               std::vector<std::string>* warnings = nullptr);

// Hilfsfunktionen der verlustfreien Umrechnung (auch für Tests):
// Krümmung aus einem Radius, c = 1/R.
double rtt_curvature_from_radius(double radius);
// Radius R zu einer Krümmung c ≠ 0 mit 1/R == c (bitgleich, soweit es ein solches R in wenigen
// ulp um 1/c gibt; unter mehreren das mit der kürzesten Dezimaldarstellung).
double rtt_radius_from_curvature(double curvature);
// Wellenlänge µm → nm bzw. nm → µm durch Verschieben des Dezimalkommas der kürzesten
// Darstellung (0,5876 µm ↔ 587,6 nm exakt; Hin- und Rückweg bitgleich).
double rtt_um_to_nm(double um);
double rtt_nm_to_um(double nm);

} // namespace mko
