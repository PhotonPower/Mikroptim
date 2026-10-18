#pragma once

// CPU-Referenztracer: Schleife über Felder × Wellenlängen × Pupillenstützstellen, die für jeden
// Strahl trace_ray aufruft (docs/architektur.md, Abschnitt "Sequenzieller Trace").

#include "mikroptim/cpu/host_surface_table.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/ray.hpp"
#include "mikroptim/trace/surface_ops.hpp"

#include <cstddef>
#include <vector>

namespace mko {

// Ergebnis für alle Strahlen eines Systems. Lineare Ablage (f · L + l) · N + k für Feld f,
// Wellenlänge l und Pupillenstützstelle k, also dieselbe Reihenfolge wie die Blöcke
// (Variante, Feld, Wellenlänge) und Threads (Stützstelle) im Kernel.
struct CpuTraceResult {
    int fields = 0;      // Feldwinkel der kollimierten Quelle, 1 bei der Punktquelle
    int wavelengths = 0; // L
    int samples = 0;     // N · M der Quadratur
    std::vector<PupilSample<double>> pupil; // Stützstellen k
    // Status Ok: Zustand im Schnittpunkt der letzten Fläche (Ende von trace_ray).
    // Sonst: teilweise fortgeschriebener Zustand, nicht weiter verwenden.
    std::vector<Ray<double>> rays;
    std::vector<TraceStatus> status;
    std::vector<int> failed_surface; // Fläche des Versagens, −1 bei Ok

    std::size_t at(int field, int wavelength, int sample) const {
        return (static_cast<std::size_t>(field) * static_cast<std::size_t>(wavelengths) +
                static_cast<std::size_t>(wavelength)) *
                   static_cast<std::size_t>(samples) +
               static_cast<std::size_t>(sample);
    }
    // Anzahl der Strahlen mit Status ≠ Ok.
    std::size_t failed_count() const;
};

// Anzahl der Felder der Quelle: Feldwinkel bei der kollimierten Quelle, 1 bei der Punktquelle.
int source_field_count(const SystemDescription& system);

// Startstrahl in lokalen Koordinaten von Fläche 0 (docs/numerik.md, Abschnitt 4).
// Kollimiert: collimated_ray mit a = table.semi_diameter[0] und θ = field_angles_deg[field].
// Punktquelle: point_source_ray im Punkt source.point mit sin_max = NA / n, n = Brechzahl des
// Mediums nach Fläche 0 bei Wellenlänge `wavelength`.
// Wirft std::out_of_range bei ungültigem Feld/Wellenlänge, std::invalid_argument bei NA >= n.
Ray<double> source_ray(const SystemDescription& system, const HostSurfaceTable& table, int field,
                       int wavelength, const PupilSample<double>& sample);

// Traced alle Strahlen auf der CPU. Die Quadratur gilt für alle Felder (der Aufrufer wählt
// den Modus, z. B. PupilMode::Axis für ein System nur mit Feld 0).
// Wirft std::invalid_argument, wenn Tabelle und System nicht zusammenpassen.
CpuTraceResult trace_system_cpu(const SystemDescription& system, const HostSurfaceTable& table,
                                const PupilQuadrature<double>& pupil);

} // namespace mko
