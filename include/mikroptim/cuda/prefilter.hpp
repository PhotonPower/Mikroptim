#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code.
// FP32-Vorfilterung (M7-T2) auf der GPU: Kernel wie MERIT, aber in float, mit Grenzabständen
// pro Strahl und den Rohdaten der konservativen Entscheidung (PrefilterStats). Programme
// benutzen den Pfad über mko::cuda::TraceEngine::prefilter_variants; Ablauf, Shared Memory und
// Reduktionen: docs/architektur.md, Abschnitt "FP32-Vorfilterung".
//
// VORFILTERUNG: Dies ist der einzige Kernel mit R = float (AGENTS.md §6.1, ADR-0003). Seine
// Ergebnisse dürfen nie in Merit, Residuen, Jacobi-Matrix oder Optimierung einfließen; sie
// entscheiden nur, welche Multistart-Varianten danach in FP64 (oder df64) bewertet werden.

#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/batch/batch_prefilter.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/cuda/cuda_error.hpp"
#include "mikroptim/cuda/device_system_batch.hpp"
#include "mikroptim/cuda/residuals.hpp"

#include <cstddef>
#include <cstdint>

namespace mko::cuda {

// Threads pro Block (wie RESIDUALS und MERIT).
inline constexpr int kPrefilterThreads = kResidualThreads;

// Shared Memory pro Block, das der Kernel höchstens belegt (ohne Opt-in auf beiden Karten).
inline constexpr std::size_t kPrefilterMaxSharedBytes = 48 * 1024;

// Ablage der Ausgaben im Gerätepuffer (jeder Bereich auf kBatchAlignment ausgerichtet),
// B = F · L Blöcke pro Variante.
struct PrefilterOutputLayout {
    BufferRegion merit;            // [V] FP64: F32_v (PrefilterStats::merit)
    BufferRegion optical_length;   // [V] FP64: Λ_v
    BufferRegion failed;           // [V] int64: sicher versagte Strahlen
    BufferRegion uncertain_ok;     // [V] int64
    BufferRegion uncertain_failed; // [V] int64
    BufferRegion merit_partial;    // [V · B] FP64: Σ r² des Blocks
    BufferRegion length_partial;   // [V · B] FP64: größte optische Länge des Blocks
    BufferRegion counts_partial;   // [V · B][3] int32: versagt, unsicher gültig, unsicher versagt
    BufferRegion plan;             // [F] ResidualFieldPlan
    std::size_t total_bytes = 0;
};

// Wirft std::invalid_argument bei ungültiger Topologie oder variants < 0.
PrefilterOutputLayout make_prefilter_output_layout(const BatchTopology& topology, int variants);

// Gerätespeicher für Batch und Ausgaben der Vorfilterung mit `variants` Varianten.
std::size_t prefilter_device_bytes(const BatchTopology& topology, int variants);

// Ausgabepuffer der Vorfilterung auf dem Gerät (RAII, nur verschiebbar), wiederverwendet, wenn
// groß genug. CUDA-Fehler werfen CudaError, falsche Argumente std::invalid_argument; alle
// Aufrufe stellen das aktuelle Gerät des Aufrufers wieder her.
class DevicePrefilter {
public:
    DevicePrefilter() = default;
    ~DevicePrefilter();

    DevicePrefilter(const DevicePrefilter&) = delete;
    DevicePrefilter& operator=(const DevicePrefilter&) = delete;
    DevicePrefilter(DevicePrefilter&& other) noexcept;
    DevicePrefilter& operator=(DevicePrefilter&& other) noexcept;

    // Startet den Kernel (ein Block pro (v, f, l), Rechentyp float) und die zweite Stufe für
    // alle Varianten von `batch` und wartet auf das Ende. `boundary_margin` = η
    // (PrefilterMargins::boundary). Wirft std::invalid_argument, wenn `layout` nicht zur
    // Topologie passt, V · F · L > kMaxGridBlocks, η < 0 oder nicht endlich ist oder die
    // Stützstellen eines Feldes nicht in das Shared Memory passen (prefilter_max_samples);
    // std::logic_error bei leerem Batch.
    void run(const DeviceSystemBatch& batch, const ResidualLayout& layout, double boundary_margin);

    // Lädt die Rohdaten des letzten Laufs herunter: neu (variants = V) oder an die Stellen
    // first … first + V − 1 eines vorhandenen Ergebnisses (Tranchen; dessen Vektoren müssen
    // groß genug sein, sonst std::invalid_argument). Wirft std::logic_error ohne Lauf.
    PrefilterStats download() const;
    void download(PrefilterStats& stats, int first) const;

    void release() noexcept;

    int device() const { return device_; }
    int variants() const { return variants_; }
    std::size_t capacity_bytes() const { return capacity_; }

private:
    int device_ = -1;
    void* buffer_ = nullptr;
    std::size_t capacity_ = 0;
    bool ran_ = false;
    int variants_ = 0;
    std::int64_t rays_ = 0;
    PrefilterOutputLayout layout_{};
};

// Größte Stützstellenzahl n_f eines Feldes, deren OPL (float) und Klasse (1 Byte) der Kernel im
// Shared Memory ablegen kann, mit G Geisterplatten der Ordnung K (Kernelattribute des aktuellen
// Geräts; CudaError). std::invalid_argument bei negativen Zahlen.
int prefilter_max_samples(int ghost_plates = 0, int ghost_terms = 0);

} // namespace mko::cuda
