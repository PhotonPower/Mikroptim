#pragma once

// Wellenfrontauswertung eines Systems auf der CPU: OPD aller Pupillenstützstellen und
// RMS-Wellenfront pro Feld und Wellenlänge (docs/numerik.md, Abschnitt 7; API:
// docs/architektur.md, Abschnitt "Wellenfront"). Bibliothek Mikroptim::merit.

#include "mikroptim/cpu/host_surface_table.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/math/vec3.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/surface_ops.hpp"

#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace mko {

struct WavefrontOptions {
    int rings = 10; // N (radiale Ringe), numerik.md, Abschnitt 8
    int spokes = 8; // M (Speichen); im Achsfall 1
    // Leer: Regel aus wavefront_pupil_mode. Gesetzt: dieser Modus für alle Felder.
    std::optional<PupilMode> pupil_mode;
};

// Wellenfront eines Tupels (Feld f, Wellenlänge l).
struct WavefrontMap {
    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

    int field = 0;
    int wavelength = 0;
    double wavelength_nm = 0.0;
    Vec3<double> image_point{0.0, 0.0, 0.0}; // Mittelpunkt M_f der Referenzsphäre
    PupilQuadrature<double> quadrature{};
    std::vector<PupilSample<double>> pupil; // Stützstellen k (Index der Quadratur)
    std::vector<double> opl;                // OPL bis zur Referenzsphäre [mm]; NaN bei Versagen
    std::vector<double> opd;                // OPD nach Kolbenabzug [mm]; NaN bei Versagen
    std::vector<double> opd_waves;          // OPD in Wellen bei λ_ref; NaN bei Versagen
    std::vector<TraceStatus> status;
    // Fläche des Versagens: −1 ohne Versagen, S (= Flächenzahl) für die Referenzsphäre.
    // Nummern der Flächentabelle (mit Geisterplatten); Dateifläche: SurfaceMap::file_surface.
    std::vector<int> failed_surface;
    double piston = kNaN;    // Σ W OPL / Σ W über gültige Strahlen [mm]
    double rms = kNaN;       // RMS der OPD [mm]; NaN ohne gültigen Strahl
    double rms_waves = kNaN; // RMS in Wellen bei λ_ref
    double valid_weight = 0; // Σ W der gültigen Strahlen (1 ohne Versagen)
    int failed = 0;          // Anzahl versagter Strahlen

    int samples() const { return static_cast<int>(pupil.size()); }
    // Mindestens ein gültiger Strahl (sonst sind piston und rms NaN).
    bool valid() const { return valid_weight > 0.0; }
};

struct WavefrontResult {
    int fields = 0;
    int wavelengths = 0;
    double reference_nm = 0.0; // λ_ref für die Angaben in Wellen
    // Verwendete Sollbrennweite f_soll [mm]; NaN, wenn die Bildpunkte vorgegeben wurden oder
    // keine gebraucht wird (Punktquelle).
    double focal_length = WavefrontMap::kNaN;
    std::vector<WavefrontMap> maps; // Ablage maps[f · L + l]

    const WavefrontMap& at(int field, int wavelength) const {
        return maps.at(static_cast<std::size_t>(field) * static_cast<std::size_t>(wavelengths) +
                       static_cast<std::size_t>(wavelength));
    }
    // Summe der versagten Strahlen über alle Tupel.
    std::size_t failed_count() const;
};

// Sollbrennweite f_soll: target.focal_length, falls gesetzt; sonst die paraxiale Brennweite
// (compute_paraxial) bei der Referenzwellenlänge spectrum.reference_nm.
// Wirft std::invalid_argument, wenn das Ergebnis nicht endlich oder 0 ist (afokales System),
// GlassError, wenn eine Brechzahl bei reference_nm nicht bestimmbar ist.
double target_focal_length(const SystemDescription& system, const GlassCatalog& catalog);

// Ideale Bildpunkte M_f = (0, h_f, w) in lokalen Koordinaten der letzten Fläche, einer pro
// Feld: w = table.thickness[S−1] (Arbeitsabstand), h_f = focal_length · tan θ_f bei der
// kollimierten Quelle; bei der Punktquelle ein Feld mit h = 0 (focal_length wird dann nicht
// verwendet). Wirft std::invalid_argument bei nicht endlicher Brennweite (nur kollimiert) und
// wenn Tabelle und System nicht zusammenpassen.
std::vector<Vec3<double>> ideal_image_points(const SystemDescription& system,
                                             const HostSurfaceTable& table, double focal_length);

// Pupillenmodus für ein Feld: kollimiert θ_f = 0 → Axis, sonst Half; Punktquelle auf der
// Achse (x = y = 0) → Axis, mit x = 0 → Half, sonst Full. options.pupil_mode überschreibt.
PupilMode wavefront_pupil_mode(const SystemDescription& system, int field,
                               const WavefrontOptions& options = {});

// Wellenfront mit vorgegebenen Mittelpunkten der Referenzsphäre (image_points[f], lokale
// Koordinaten der letzten Fläche, Radius |M_f|). Für jedes Feld f, jede Wellenlänge l und jede
// Stützstelle k: Startstrahl (source_ray), trace_to_reference_sphere, danach opd_statistics.
// Wirft std::invalid_argument (Größen passen nicht, ungültige Quadratur) und die Ausnahmen von
// source_ray.
WavefrontResult evaluate_wavefront(const SystemDescription& system, const HostSurfaceTable& table,
                                   std::span<const Vec3<double>> image_points,
                                   const WavefrontOptions& options = {});

// Wellenfront eines geladenen Systems mit den idealen Bildpunkten: f_soll nach
// target_focal_length (nur bei kollimierter Quelle), M_f nach ideal_image_points.
WavefrontResult evaluate_wavefront(const LoadedSystem& system,
                                   const WavefrontOptions& options = {});

} // namespace mko
