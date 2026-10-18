#pragma once

// Einlesen und Schreiben von `mikroptim.system/1` (docs/dateiformat.md).
// Fehler werfen mko::FormatError mit JSON-Pfad; Dateizugriffsfehler std::runtime_error.

#include "mikroptim/model/format_error.hpp"
#include "mikroptim/model/system.hpp"

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace mko {

// Liest eine Systembeschreibung aus JSON-Text und validiert sie (validate_system).
// Unbekannte Felder werden ignoriert; ist `warnings` gesetzt, wird je Feld eine Warnung
// "<pfad>: unbekanntes Feld wird ignoriert" angehängt.
SystemDescription read_system_json(std::string_view text,
                                   std::vector<std::string>* warnings = nullptr);

// Wie read_system_json, liest aus einer Datei. FormatError trägt den Dateinamen.
SystemDescription read_system_file(const std::filesystem::path& file,
                                   std::vector<std::string>* warnings = nullptr);

// Schreibt das Modell als eingerücktes JSON (Feldreihenfolge wie in docs/dateiformat.md,
// Zahlen verlustfrei, abschließender Zeilenumbruch). Validiert vorher (validate_system).
std::string write_system_json(const SystemDescription& system);

// Wie write_system_json, schreibt in eine Datei.
void write_system_file(const SystemDescription& system, const std::filesystem::path& file);

// Löst die Einträge von `glass_catalogs` zu existierenden Dateien auf. Relative Einträge werden
// zuerst relativ zu `base_dir` (Verzeichnis der Systemdatei) gesucht, danach in dessen
// Elternverzeichnissen aufwärts (deckt „relativ zum Repository-Wurzelverzeichnis“ ab).
// Wirft FormatError mit Pfad `glass_catalogs[i]`, wenn ein Eintrag nicht gefunden wird.
std::vector<std::filesystem::path> resolve_glass_catalogs(const SystemDescription& system,
                                                          const std::filesystem::path& base_dir);

} // namespace mko
