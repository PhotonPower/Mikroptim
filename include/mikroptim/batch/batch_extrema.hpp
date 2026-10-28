#pragma once

// Max-Reduktionen pro Variante und Fläche (größte Strahlhöhe, größter Einfallswinkel) eines
// SystemBatch: Ergebnis (SurfaceExtremaResult) und CPU-Referenz (surface_extrema_cpu) für die
// Kernel MERIT und RESIDUALS (M2-T3). Reines C++ ohne CUDA-Typen. Definition:
// docs/architektur.md, Abschnitt "Max-Reduktionen"; Pro-Strahl-Teil in
// include/mikroptim/trace/surface_extrema.hpp.

#include "mikroptim/batch/system_batch.hpp"

#include <cstddef>
#include <vector>

namespace mko {

class ResidualLayout; // batch_residuals.hpp (bindet diesen Header ein)

// Pro Variante v und Tabellenfläche s (Dateiflächen plus Geisterplatten in Tracerichtung, wie
// failed_surface; Zuordnung zu Dateiflächen: SurfaceMap), Ablage values[(v · S + s) · 2 + {0, 1}]:
//   {0} größte Strahlhöhe √(x² + y²) [mm] im Schnittpunkt mit Fläche s (lokale Koordinaten),
//   {1} kleinster cos I = d · N (d vor der Brechung, N Flächennormale), also der größte
//       Einfallswinkel I = arccos(cos I),
// jeweils über alle gültigen Strahlen (Status Ok, inklusive Referenzsphäre) aller Felder,
// Wellenlängen und Stützstellen der Variante. Fläche 0 (Blende, keine Brechung): Startpunkt in
// ihrer Ebene und cos I = d_z des Startstrahls. Versagte Strahlen tragen nichts bei, auch nicht
// zu den Flächen vor der versagenden. Ohne gültigen Strahl bleiben die neutralen Elemente
// Höhe 0 und cos I = 1 (erkennbar an der Zahl versagter Strahlen).
struct SurfaceExtremaResult {
    int variants = 0;
    int surfaces = 0;           // S
    std::vector<double> values; // [V · S · 2]; leer, wenn nicht angefordert

    bool empty() const { return values.empty(); }
    double max_height(int v, int s) const { return values[index(v, s)]; }
    double min_cos_incidence(int v, int s) const { return values[index(v, s) + 1]; }
    // Größter Einfallswinkel [rad] = arccos(min cos I), cos I auf [−1, 1] begrenzt.
    double max_incidence_angle(int v, int s) const;
    std::size_t index(int v, int s) const {
        return (static_cast<std::size_t>(v) * static_cast<std::size_t>(surfaces) +
                static_cast<std::size_t>(s)) *
               2;
    }
};

// Glättung der Reduktionen (M11-T1, ADR-0013; aus constraints.extrema_smoothing): β > 0
// ersetzt das harte Maximum der Strahlhöhe [1/mm] bzw. das harte Minimum von cos I
// [dimensionslos] durch Log-Sum-Exp über die gültigen Strahlen (smooth_max_* in
// surface_extrema.hpp). 0 = hart. Die Ablage von SurfaceExtremaResult bleibt gleich.
struct ExtremaSmoothing {
    double height = 0.0;
    double cos_incidence = 0.0;
    bool enabled() const { return height > 0.0 || cos_incidence > 0.0; }
    friend bool operator==(const ExtremaSmoothing&, const ExtremaSmoothing&) = default;
};

// CPU-Referenz der Max-Reduktionen für Variante v: für jedes Tupel (f, l) und jede Stützstelle
// k batch_ray_opl mit einem RaySurfaceRecord als Beobachter, bei Status Ok
// SurfaceExtrema::add_ray; zuletzt √ der größten Höhenquadrate. Ergebnis mit variants = 1.
// Wirft wie residuals_cpu (std::out_of_range, std::invalid_argument).
SurfaceExtremaResult surface_extrema_cpu(const SystemBatchHost& batch, int v,
                                         const ResidualLayout& layout,
                                         const ExtremaSmoothing& smoothing = {});

// Wie oben für alle Varianten.
SurfaceExtremaResult surface_extrema_cpu(const SystemBatchHost& batch, const ResidualLayout& layout,
                                         const ExtremaSmoothing& smoothing = {});

} // namespace mko
