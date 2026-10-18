#pragma once

// Sicht auf einen SystemBatch (Structure of Arrays), gemeinsam für CPU und GPU.
// Header-only, ohne Ausnahmen, Allokation und std-Container. Layout, Indexformeln und
// Begründung: docs/architektur.md, Abschnitt "SystemBatch und Datenlayout".

#include "mikroptim/config.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/ray.hpp"
#include "mikroptim/trace/surface_table.hpp"
#include "mikroptim/trace/trace.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace mko {

// Art der Quelle eines Batches (1 Byte, Werte stabil wie SurfaceKind).
enum class SourceKind : std::uint8_t {
    Collimated = 0, // kollimierte Quelle: Pupillenradius pro Variante, Feldwinkel pro Variante
    Point = 1,      // Punktquelle (Tests): Quellpunkt und NA pro Batch, genau ein Feld
};

// Größen eines Batches und die Indexformeln aller Felder. Alle Indizes sind lineare
// Positionen im jeweiligen Feld (Einheit: Elemente, nicht Bytes), Variante v ist immer der
// äußerste (langsamste) Index. Vorbedingungen (0 <= v < variants usw.) prüft der Aufrufer.
struct BatchShape {
    int variants;     // V
    int surfaces;     // S, inklusive Blende, 2 … kMaxSurfaces
    int wavelengths;  // L >= 1
    int fields;       // F >= 1 (Punktquelle: 1)
    int ghost_plates; // G >= 0
    int ghost_terms;  // K >= 0 (>= 1, wenn G > 0)

    // curvature[v][s], thickness[v][s]: v·S + s
    MKO_HD std::size_t surface_at(int v, int s) const {
        return static_cast<std::size_t>(v) * static_cast<std::size_t>(surfaces) +
               static_cast<std::size_t>(s);
    }
    // index[v][s][l]: (v·S + s)·L + l
    MKO_HD std::size_t index_at(int v, int s, int l) const {
        return surface_at(v, s) * static_cast<std::size_t>(wavelengths) +
               static_cast<std::size_t>(l);
    }
    // field_angle[v][f], field_target[v][f]: v·F + f
    MKO_HD std::size_t field_at(int v, int f) const {
        return static_cast<std::size_t>(v) * static_cast<std::size_t>(fields) +
               static_cast<std::size_t>(f);
    }
    // ghost_coeff[v][g][k]: (v·G + g)·K + k
    MKO_HD std::size_t ghost_coeff_at(int v, int g, int k) const {
        return (static_cast<std::size_t>(v) * static_cast<std::size_t>(ghost_plates) +
                static_cast<std::size_t>(g)) *
                   static_cast<std::size_t>(ghost_terms) +
               static_cast<std::size_t>(k);
    }
    // ghost_index[v][g][l]: (v·G + g)·L + l
    MKO_HD std::size_t ghost_index_at(int v, int g, int l) const {
        return (static_cast<std::size_t>(v) * static_cast<std::size_t>(ghost_plates) +
                static_cast<std::size_t>(g)) *
                   static_cast<std::size_t>(wavelengths) +
               static_cast<std::size_t>(l);
    }

    // Elementzahlen pro Variante
    MKO_HD std::size_t surface_stride() const { return static_cast<std::size_t>(surfaces); }
    MKO_HD std::size_t index_stride() const {
        return static_cast<std::size_t>(surfaces) * static_cast<std::size_t>(wavelengths);
    }
    MKO_HD std::size_t field_stride() const { return static_cast<std::size_t>(fields); }
    MKO_HD std::size_t ghost_coeff_stride() const {
        return static_cast<std::size_t>(ghost_plates) * static_cast<std::size_t>(ghost_terms);
    }
    MKO_HD std::size_t ghost_index_stride() const {
        return static_cast<std::size_t>(ghost_plates) * static_cast<std::size_t>(wavelengths);
    }
};

static_assert(std::is_trivially_copyable_v<BatchShape>);
static_assert(std::is_standard_layout_v<BatchShape>);

// Sicht auf alle Felder eines Batches: nur Zeiger und Größen, trivial kopierbar (wird per Wert
// an Kernel übergeben). Zeigt auf Host-Vektoren (SystemBatchHost::view) oder auf den
// Gerätespeicher (cuda::DeviceSystemBatch::view).
//
// Gemeinsam für den Batch: type[S], ghost_surface[G], ghost_norm_radius[G], wavelength_nm[L],
// wavelength_weight[L], field_weight[F] und die Skalare. Pro Variante: alle übrigen Felder,
// Index über die Formeln von BatchShape.
template <class R>
struct SystemBatchView {
    BatchShape shape;

    // Quelle und Skalare (gemeinsam)
    SourceKind source;
    R source_point[3];         // Punktquelle: Quellpunkt in lokalen Koordinaten von Fläche 0
    R source_na;               // Punktquelle: NA im Medium nach Fläche 0
    R reference_wavelength_nm; // Referenzwellenlänge λ_ref (OPD in Wellen) [nm]

    // Gemeinsam für den Batch
    const SurfaceKind* type;    // [S]
    const int* ghost_surface;   // [G], Fläche der Geisterplatte g (type == GhostPlate)
    const R* ghost_norm_radius; // [G], Normierungsradius r_g [mm]
    const R* wavelength_nm;     // [L], Wellenlängen [nm], für die index berechnet wurde
    const R* wavelength_weight; // [L], Gewichte w_λ
    const R* field_weight;      // [F], Gewichte w_f

    // Pro Variante
    const R* curvature;    // [V][S], 1/mm (0 bei Stop und Plane)
    const R* thickness;    // [V][S], mm; thickness[v][S−1] = Arbeitsabstand (Bildebene)
    const R* index;        // [V][S][L], Medium nach Fläche s
    const R* pupil_radius; // [V], Pupillenradius a = Halbdurchmesser der Blende [mm]
    const R* field_angle;  // [V][F], Feldwinkel θ_f [rad] (Punktquelle: 0)
    const R* field_target; // [V][F], ideale Bildhöhe h_f [mm] (Referenzsphäre)
    const R* ghost_coeff;  // [V][G][K], Koeffizienten a_{g,k} des Dickenprofils
    const R* ghost_index;  // [V][G][L], Brechzahl des virtuellen Glases

    // Flächentabelle der Variante v für trace_ray: Zeiger um v·S, v·S·L, v·G·K bzw. v·G·L
    // versetzt; type und ghost_norm_radius sind für alle Varianten dieselben.
    MKO_HD SurfaceTableView<R> surface_table(int v) const {
        return SurfaceTableView<R>{
            shape.surfaces,
            shape.wavelengths,
            type,
            curvature + shape.surface_at(v, 0),
            thickness + shape.surface_at(v, 0),
            index + shape.index_at(v, 0, 0),
            shape.ghost_plates,
            shape.ghost_terms,
            shape.ghost_plates > 0 ? ghost_coeff + shape.ghost_coeff_at(v, 0, 0) : nullptr,
            shape.ghost_plates > 0 ? ghost_index + shape.ghost_index_at(v, 0, 0) : nullptr,
            ghost_norm_radius};
    }

    // Arbeitsabstand (Abstand letzte Fläche – Bildebene) der Variante v. Einzige Quelle ist
    // thickness[v][S−1]; ein eigenes Feld image_distance gibt es nicht.
    MKO_HD R image_distance(int v) const {
        return thickness[shape.surface_at(v, shape.surfaces - 1)];
    }
};

static_assert(std::is_trivially_copyable_v<SystemBatchView<double>>);
static_assert(std::is_standard_layout_v<SystemBatchView<double>>);

// Startstrahl der Variante v für Feld f, Wellenlänge l und Pupillenstützstelle `s` in lokalen
// Koordinaten von Fläche 0 (docs/numerik.md, Abschnitt 4), wie source_ray in cpu_trace.hpp:
// - kollimiert: collimated_ray(pupil_radius[v], field_angle[v][f], ρ, φ),
// - Punktquelle: point_source_ray(source_point, source_na / index[v][0][l], ρ, φ).
// Prüft nichts (Vorbedingung bei der Punktquelle: 0 < source_na < index[v][0][l]).
template <class R>
MKO_HD inline Ray<R> batch_source_ray(const SystemBatchView<R>& b, int v, int f, int l,
                                      const PupilSample<R>& s) {
    if (b.source == SourceKind::Point) {
        const R sin_max = b.source_na / b.index[b.shape.index_at(v, 0, l)];
        return point_source_ray(Vec3<R>{b.source_point[0], b.source_point[1], b.source_point[2]},
                                sin_max, s.rho, s.phi);
    }
    return collimated_ray(b.pupil_radius[v], b.field_angle[b.shape.field_at(v, f)], s.rho, s.phi);
}

} // namespace mko
