#pragma once

// Datenmodell der Systembeschreibung `mikroptim.system/1` (Host, docs/dateiformat.md).
// Das Modell ist frei von JSON-Typen; Einlesen und Schreiben: mikroptim/io/system_json.hpp.

#include "mikroptim/model/material.hpp"

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

    // Abschnitte späterer Meilensteine (M3/M4), deren Schema noch nicht festgelegt ist:
    // kompakter JSON-Text, wird beim Schreiben unverändert wieder ausgegeben.
    std::optional<std::string> ghost_plates_json;
    std::optional<std::string> variables_json;
    std::optional<std::string> constraints_json;

    friend bool operator==(const SystemDescription&, const SystemDescription&) = default;
};

// Prüft die inhaltlichen Regeln (docs/dateiformat.md, „Einlesen und Validierung“) und wirft
// FormatError mit dem JSON-Pfad der ersten verletzten Regel. Glasnamen werden nicht geprüft.
void validate_system(const SystemDescription& system);

// Prüft für jede Fläche mit Katalogmaterial, ob das Glas existiert; wirft sonst FormatError
// mit Pfad `surfaces[i].material`.
void validate_materials(const SystemDescription& system,
                        const std::function<bool(std::string_view)>& glass_exists);

// Namen für Meldungen und Dateiformat
std::string_view to_string(SurfaceType type);
std::string_view to_string(SourceType type);

} // namespace mko
