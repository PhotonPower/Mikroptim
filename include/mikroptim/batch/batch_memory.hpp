#pragma once

// Ablage eines SystemBatch im Gerätespeicher und Speicherabschätzung für Batch plus Ausgaben
// (docs/architektur.md, Abschnitte "SystemBatch und Datenlayout" und "Speicherbedarf").
// Reines C++; die Gerätegrößen (freier Speicher) liefert mko::cuda::device_memory_info.

#include "mikroptim/batch/system_batch.hpp"

#include <cstddef>
#include <cstdint>

namespace mko {

// Ausrichtung jedes Feldes im Gerätepuffer [Byte]. cudaMalloc liefert mindestens 256 Byte;
// mit derselben Ausrichtung beginnt jedes Feld auf einer eigenen Speichertransaktion.
inline constexpr std::size_t kBatchAlignment = 256;

// Rundet `bytes` auf ein Vielfaches von kBatchAlignment auf.
constexpr std::size_t align_bytes(std::size_t bytes) {
    return (bytes + kBatchAlignment - 1) / kBatchAlignment * kBatchAlignment;
}

// Bereich eines Feldes im Puffer: Anfang (Vielfaches von kBatchAlignment) und Nutzbytes.
struct BufferRegion {
    std::size_t offset = 0;
    std::size_t bytes = 0;
};

// Ablage aller Felder eines Batches in einem zusammenhängenden Puffer, in dieser Reihenfolge:
// erst die gemeinsamen, dann die Felder pro Variante. Jeder Bereich beginnt ausgerichtet;
// total_bytes = Summe der aufgerundeten Bereichsgrößen.
struct SystemBatchLayout {
    // gemeinsam
    BufferRegion type;              // S · 1 Byte
    BufferRegion ghost_surface;     // G · 4 Byte (int)
    BufferRegion ghost_norm_radius; // G · 8
    BufferRegion wavelength_nm;     // L · 8
    BufferRegion wavelength_weight; // L · 8
    BufferRegion field_weight;      // F · 8
    // pro Variante
    BufferRegion curvature;    // V · S · 8
    BufferRegion thickness;    // V · S · 8
    BufferRegion index;        // V · S · L · 8
    BufferRegion pupil_radius; // V · 8
    BufferRegion field_angle;  // V · F · 8
    BufferRegion field_target; // V · F · 8
    BufferRegion ghost_coeff;  // V · G · K · 8
    BufferRegion ghost_index;  // V · G · L · 8
    std::size_t total_bytes = 0;
};

// Wirft std::invalid_argument bei ungültiger Topologie oder variants < 0.
SystemBatchLayout make_batch_layout(const BatchTopology& topology, int variants);

// Ausgaben eines Traces, für die Speicher reserviert werden muss (Kernel-Modi in
// docs/architektur.md). Die Zahl der Strahlen pro Variante gibt der Aufrufer vor, z. B. mit
// rays_per_variant().
struct OutputRequest {
    std::int64_t rays_per_variant = 0; // Σ_f L · n_f (n_f = Stützstellen von Feld f)
    std::int64_t extra_residuals = 0;  // weitere Residuen pro Variante (Paraxial, Strafen)
    bool residuals = true;             // Residuenvektor: rays + extra Werte (FP64)
    bool ray_status = true;            // Status pro Strahl (1 Byte, TraceStatus)
    bool opd = false;                  // rohe OPD pro Strahl nach Kolbenabzug (FP64), z. B.
                                       // für Tests und Auswertung (Kernel RESIDUALS, M2-T2)
    bool merit = true;                 // Merit (FP64) + Teilsumme pro Block (v, f, l)
    bool surface_max = true;           // max. Strahlhöhe und Einfallswinkel pro Fläche (2 · S FP64)
                                       // + dasselbe pro Block (v, f, l)
};

// Strahlen pro Variante: L · (axis_fields · axis_samples + (fields − axis_fields) · samples).
// `samples` = sample_count der Quadratur für Feldpunkte (N · M), `axis_samples` = sample_count
// der Quadratur im Achsfall (N) für axis_fields der Felder (an beliebiger Stelle; die Ablage
// pro Feld legt ResidualLayout fest, batch_residuals.hpp). Wirft std::invalid_argument bei
// negativen Werten oder axis_fields > fields.
std::int64_t rays_per_variant(int fields, int wavelengths, int samples, int axis_fields = 0,
                              int axis_samples = 0);

// Ablage der Ausgaben, gleiche Regeln wie SystemBatchLayout (Bereiche ausgerichtet, nicht
// angeforderte Ausgaben haben 0 Byte). `extra` = V · extra_bytes_per_variant (z. B.
// Arbeitsspeicher eines Aufrufers wie des LM in M3).
struct OutputLayout {
    BufferRegion residuals;           // V · (rays + extra_residuals) · 8
    BufferRegion ray_status;          // V · rays · 1
    BufferRegion opd;                 // V · rays · 8
    BufferRegion merit;               // V · 8
    BufferRegion merit_partial;       // V · F · L · 8
    BufferRegion surface_max;         // V · 2 · S · 8
    BufferRegion surface_max_partial; // V · F · L · 2 · S · 8
    BufferRegion extra;               // V · extra_bytes_per_variant
    std::size_t total_bytes = 0;
};

// Wirft std::invalid_argument bei ungültiger Topologie, variants < 0 oder negativen Größen.
OutputLayout make_output_layout(const BatchTopology& topology, int variants,
                                const OutputRequest& outputs,
                                std::size_t extra_bytes_per_variant = 0);

// Geschätzter Gerätespeicher für Batch und Ausgaben:
// make_batch_layout(...).total_bytes + make_output_layout(...).total_bytes.
std::size_t estimate_device_bytes(const BatchTopology& topology, int variants,
                                  const OutputRequest& outputs,
                                  std::size_t extra_bytes_per_variant = 0);

// Sicherheitsreserve vom freien Speicher: max(min_bytes, fraction · free) bleibt frei
// (Kontext, Kernel-Stacks, Local Memory für TraceRecord, Fragmentierung).
struct MemoryReserve {
    std::size_t min_bytes = std::size_t{256} << 20; // 256 MiB
    double fraction = 0.05;
};

// Nutzbarer Speicher: free − max(min_bytes, fraction · free), mindestens 0.
std::size_t usable_bytes(std::size_t free_bytes, const MemoryReserve& reserve = {});

// Größtes Grid eines Kernels (gridDim.x ab sm_30): V · F · L Blöcke dürfen das nicht
// überschreiten.
inline constexpr std::int64_t kMaxGridBlocks = 2147483647;

// Größte Variantenzahl V mit estimate_device_bytes(V) <= usable_bytes(free_bytes), begrenzt
// durch V · F · L <= kMaxGridBlocks. 0, wenn nicht einmal eine Variante passt.
int max_variants(std::size_t free_bytes, const BatchTopology& topology,
                 const OutputRequest& outputs, const MemoryReserve& reserve = {},
                 std::size_t extra_bytes_per_variant = 0);

} // namespace mko
