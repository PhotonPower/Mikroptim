#pragma once

// Datenmodell der Systembeschreibung `mikroptim.system/1` (Host, docs/dateiformat.md).
// Das Modell ist frei von JSON-Typen; Einlesen und Schreiben: mikroptim/io/system_json.hpp.

#include "mikroptim/model/material.hpp"
#include "mikroptim/model/variables.hpp"

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mko {

// Formatkennung, die geschrieben und beim Einlesen verlangt wird.
inline constexpr std::string_view kSystemFormat = "mikroptim.system/1";

enum class SurfaceType { Stop, Plane, Sphere };
enum class SourceType { Collimated, Point };

struct Spectrum {
    std::vector<double> wavelengths_nm;
    std::vector<double> weights; // gleiche Länge wie wavelengths_nm
    double reference_nm = 0.0;

    friend bool operator==(const Spectrum&, const Spectrum&) = default;
};

struct Source {
    SourceType type = SourceType::Collimated;
    // nur bei Collimated
    std::vector<double> field_angles_deg; // Feldwinkel θ in der y-z-Ebene
    std::vector<double> field_weights;    // gleiche Länge wie field_angles_deg
    // nur bei Point
    std::array<double, 3> point{0.0, 0.0, 0.0}; // lokale Koordinaten der ersten Fläche
    double numerical_aperture = 0.0;            // NA im Medium der Quelle

    friend bool operator==(const Source&, const Source&) = default;
};

// Sollwerte (ab M3 ausgewertet); fehlende Felder bleiben leer.
struct Target {
    std::optional<double> focal_length;
    std::optional<double> parfocal_length;
    std::optional<double> working_distance;

    friend bool operator==(const Target&, const Target&) = default;
};

struct Surface {
    SurfaceType type = SurfaceType::Plane;
    double curvature = 0.0; // 1/mm, nur bei Sphere von 0 verschieden
    double thickness = 0.0; // Abstand zum nächsten Scheitel bzw. zur Bildebene
    MaterialSpec material;  // Medium nach der Fläche
    double semi_diameter = 0.0;
    std::optional<std::string> comment;

    // Vergleicht alle Felder, auch die des Materials (MaterialSpec selbst hat kein operator==).
    friend bool operator==(const Surface& a, const Surface& b) {
        return a.type == b.type && a.curvature == b.curvature && a.thickness == b.thickness &&
               a.material.kind == b.material.kind && a.material.name == b.material.name &&
               a.material.n == b.material.n && a.material.n0 == b.material.n0 &&
               a.material.nu == b.material.nu && a.semi_diameter == b.semi_diameter &&
               a.comment == b.comment;
    }
};

// Geisterplatte (docs/dateiformat.md, „Geisterplatten“; Physik: docs/numerik.md, Abschnitt 6):
// virtuelle dünne Glasplatte im Luftspalt nach Fläche `after_surface`, im Abstand `distance`
// von deren Scheitel. Dickenprofil t(r) = Σ_{k=1}^{K} a_k P_k(2 (r/r_g)² − 1) mit
// K = coefficients.size() (in der Datei `order`).
struct GhostPlate {
    int after_surface = 0;            // Dateifläche, nach der der Luftspalt liegt
    double distance = 0.0;            // Abstand vom Scheitel von after_surface [mm]
    MaterialSpec glass;               // virtuelles Glas (Katalogname oder Modell, nicht Luft)
    double norm_radius = 0.0;         // Normierungsradius r_g [mm]
    std::vector<double> coefficients; // a_1 … a_K [mm]

    // Vergleicht alle Felder, auch die des Glases (MaterialSpec selbst hat kein operator==).
    friend bool operator==(const GhostPlate& a, const GhostPlate& b) {
        return a.after_surface == b.after_surface && a.distance == b.distance &&
               a.glass.kind == b.glass.kind && a.glass.name == b.glass.name &&
               a.glass.n == b.glass.n && a.glass.n0 == b.glass.n0 && a.glass.nu == b.glass.nu &&
               a.norm_radius == b.norm_radius && a.coefficients == b.coefficients;
    }
};

struct SystemDescription {
    std::string format{kSystemFormat};
    std::string name;
    std::optional<std::string> description;
    std::string units = "mm";
    std::vector<std::string> glass_catalogs; // wie in der Datei, Auflösung: resolve_glass_catalogs
    Spectrum spectrum;
    Source source;
    std::optional<Target> target;
    std::vector<Surface> surfaces; // in Tracerichtung, surfaces[0] ist die Blende
    // Geisterplatten, sortiert nach (after_surface, distance); leer: keine
    std::vector<GhostPlate> ghost_plates;

    // Abschnitt späterer Meilensteine (M3-T3/M4-T3), dessen Schema noch nicht festgelegt ist:
    // kompakter JSON-Text, wird beim Schreiben unverändert wieder ausgegeben.
    std::optional<std::string> constraints_json;

    // Optimierungsvariablen (ab M3-T1, mikroptim/model/variables.hpp)
    std::optional<Variables> variables;

    friend bool operator==(const SystemDescription&, const SystemDescription&) = default;
};

// Prüft die inhaltlichen Regeln (docs/dateiformat.md, „Einlesen und Validierung“) und wirft
// FormatError mit dem JSON-Pfad der ersten verletzten Regel. Glasnamen werden nicht geprüft.
// Der Abschnitt `variables` wird mit resolve_variables geprüft.
void validate_system(const SystemDescription& system);

// Prüft für jede Fläche und Geisterplatte mit Katalogmaterial, ob das Glas existiert; wirft
// sonst FormatError mit Pfad `surfaces[i].material` bzw. `ghost_plates[g].glass`.
void validate_materials(const SystemDescription& system,
                        const std::function<bool(std::string_view)>& glass_exists);

// Abbildung zwischen den Flächen der Datei (`surfaces`) und den Flächen der Flächentabelle
// (Tracer, Batch), in die die Geisterplatten als zusätzliche Flächen vom Typ GhostPlate
// eingefügt werden (docs/dateiformat.md, „Abbildung Dateiflächen ↔ Tabellenflächen“):
// Tabellenfläche j folgt in Tracerichtung; nach Dateifläche i stehen die Geisterplatten mit
// after_surface = i in der Reihenfolge der Datei (nach distance sortiert). Geisterplatte g
// ist die g-te Tabellenfläche vom Typ GhostPlate (wie `ghost_surface[g]` im Batch).
struct SurfaceMap {
    std::vector<int> table_of_file;  // [S_Datei] → Tabellenfläche
    std::vector<int> file_of_table;  // [S_Tabelle] → Dateifläche, −1 bei Geisterplatten
    std::vector<int> ghost_of_table; // [S_Tabelle] → Geisterplatte g, −1 bei Dateiflächen
    std::vector<int> table_of_ghost; // [G] → Tabellenfläche

    int file_surfaces() const { return static_cast<int>(table_of_file.size()); }
    int table_surfaces() const { return static_cast<int>(file_of_table.size()); }

    // Dateifläche einer Tabellenfläche j (z. B. versagende Fläche eines Strahls): −1 bei einer
    // Geisterplatte; j = S_Tabelle (Referenzsphäre nach der letzten Fläche) → S_Datei.
    // Wirft std::out_of_range außerhalb von [0, S_Tabelle].
    int file_surface(int table_surface) const;
    // Tabellenfläche einer Dateifläche i; i = S_Datei → S_Tabelle. Wirft std::out_of_range.
    int table_surface(int file_surface) const;
};

// Baut die Abbildung. Verlangt after_surface in [0, S_Datei − 2] und die Sortierung nach
// (after_surface, distance) wie validate_system; wirft sonst std::invalid_argument.
SurfaceMap make_surface_map(const SystemDescription& system);

// Zahl der Tabellenflächen: surfaces.size() + ghost_plates.size().
int table_surface_count(const SystemDescription& system);

// Namen für Meldungen und Dateiformat
std::string_view to_string(SurfaceType type);
std::string_view to_string(SourceType type);

} // namespace mko
