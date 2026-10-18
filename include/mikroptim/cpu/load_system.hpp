#pragma once

// Laden eines Systems zum Tracen: Systemdatei, Glaskataloge, Brechzahl- und Flächentabelle
// (docs/architektur.md, Abschnitt "Sequenzieller Trace").

#include "mikroptim/cpu/host_surface_table.hpp"
#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/glass/index_table.hpp"
#include "mikroptim/model/system.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace mko {

struct LoadedSystem {
    SystemDescription system;
    std::vector<std::filesystem::path> catalog_files; // aufgelöste Einträge von glass_catalogs
    GlassCatalog catalog;                             // alle Kataloge zusammengeführt
    IndexTable index;                                 // n[s][l] für system.spectrum.wavelengths_nm
    HostSurfaceTable table;                           // Eingabe für trace_ray
    std::vector<std::string> warnings; // ignorierte Felder in System- und Glasdateien
};

// Liest die Systemdatei (read_system_file), löst die Glaskataloge relativ zu ihrem Verzeichnis
// auf (resolve_glass_catalogs), lädt und vereinigt sie (GlassCatalog::from_file, merge), prüft
// die Glasnamen (validate_materials) und berechnet Brechzahl- und Flächentabelle.
// Fehler: FormatError (Inhalt der Systemdatei, nicht gefundene Kataloge, unbekannte Gläser),
// GlassError (Glasdateien, Brechzahlen außerhalb des Gültigkeitsbereichs),
// std::runtime_error (Datei nicht lesbar), std::invalid_argument (zu viele Flächen).
LoadedSystem load_system(const std::filesystem::path& file);

// Wie load_system für eine bereits eingelesene Beschreibung; relative Katalogpfade werden ab
// `base_dir` gesucht (leer: aktuelles Verzeichnis). Die Beschreibung wird validiert.
LoadedSystem prepare_system(SystemDescription system, const std::filesystem::path& base_dir);

} // namespace mko
