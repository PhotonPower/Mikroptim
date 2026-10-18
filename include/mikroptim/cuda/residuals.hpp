#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code.
// Kernel RESIDUALS (M2-T2): Residuenvektor pro Variante auf der GPU. Kernel-Abbildung,
// Shared-Memory-Belegung, Reduktionsschema und Ablage: docs/architektur.md, Abschnitt
// "Kernel RESIDUALS". Die TraceEngine (M2-T4) kapselt diese Klasse.

#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/cuda/cuda_error.hpp"
#include "mikroptim/cuda/device_system_batch.hpp"

#include <cstddef>
#include <cstdint>

namespace mko::cuda {

// Threads pro Block des Kernels RESIDUALS (Vielfaches von 32; in M7 tunbar). Die Summation
// des Kolbens hängt von diesem Wert ab: Ergebnisse sind für einen festen Wert bitgleich
// reproduzierbar.
inline constexpr int kResidualThreads = 128;

// Optionen eines Laufs.
struct ResidualRunOptions {
    bool opd = false;                 // rohe OPD pro Strahl zusätzlich ablegen
    std::int64_t extra_residuals = 0; // weitere Residuen pro Variante nach den Strahlresiduen
                                      // (Paraxial, Strafen ab M3); run setzt sie auf 0
    bool extrema = false;             // Max-Reduktionen pro Variante und Fläche (M2-T3,
                                      // ResidualResult::extrema; Definition batch_extrema.hpp)
};

// Ausgabepuffer des Kernels RESIDUALS auf dem Gerät (RAII, nur verschiebbar): ein
// cudaMalloc im Layout make_output_layout (Residuen, Status, optional OPD) plus die
// Feldablage (ResidualFieldPlan[F]). Der Puffer wird wiederverwendet, wenn er groß genug ist.
// CUDA-Fehler werfen CudaError, falsche Argumente std::invalid_argument; alle Aufrufe stellen
// das aktuelle Gerät des Aufrufers wieder her.
class DeviceResiduals {
public:
    DeviceResiduals() = default;
    ~DeviceResiduals();

    DeviceResiduals(const DeviceResiduals&) = delete;
    DeviceResiduals& operator=(const DeviceResiduals&) = delete;
    DeviceResiduals(DeviceResiduals&& other) noexcept;
    DeviceResiduals& operator=(DeviceResiduals&& other) noexcept;

    // Startet den Kernel für alle Varianten von `batch` auf dessen Gerät und wartet auf das
    // Ende. Ein Block pro (v, f, l), V · F · L <= kMaxGridBlocks (sonst std::invalid_argument;
    // Tranchen sind Sache der TraceEngine, M2-T4). `layout` muss zur Topologie passen
    // (check_residual_layout). Die Werte der Varianten prüft der Aufrufer auf dem Host
    // (SystemBatchHost::validate_values; check_pupil_modes läuft schon beim Upload).
    // Wirft std::logic_error bei leerem Batch.
    void run(const DeviceSystemBatch& batch, const ResidualLayout& layout,
             const ResidualRunOptions& options = {});

    // Lädt alle Ergebnisse des letzten Laufs herunter (OPD und Max-Reduktionen nur, wenn
    // angefordert; Status immer). Wirft std::logic_error ohne vorherigen Lauf.
    ResidualResult download() const;
    void download(ResidualResult& result) const;

    // Gibt den Gerätepuffer frei.
    void release() noexcept;

    int device() const { return device_; }
    int variants() const { return variants_; }
    std::int64_t rays_per_variant() const { return rays_; }
    std::int64_t stride() const { return stride_; }
    bool has_opd() const { return opd_; }
    bool has_extrema() const { return extrema_; }
    const OutputLayout& layout() const { return layout_; }
    std::size_t capacity_bytes() const { return capacity_; }

    // Gerätezeiger auf die Ausgaben des letzten Laufs (für Folgekernel, z. B. JᵀJ in M3);
    // gültig bis zum nächsten run oder release. nullptr ohne Lauf (OPD, Max-Reduktionen: ohne
    // Anforderung). device_extrema: [V][S][2] wie SurfaceExtremaResult::values.
    const double* device_residuals() const;
    const std::uint8_t* device_status() const;
    const double* device_opd() const;
    const double* device_extrema() const;

private:
    int device_ = -1;
    void* buffer_ = nullptr;
    std::size_t capacity_ = 0;
    bool ran_ = false;
    bool opd_ = false;
    bool extrema_ = false;
    int variants_ = 0;
    int surfaces_ = 0;
    std::int64_t rays_ = 0;
    std::int64_t stride_ = 0;
    OutputLayout layout_{};
};

// Bequemlichkeit: ein Lauf mit eigenem Puffer und Download aller Ergebnisse.
ResidualResult run_residuals(const DeviceSystemBatch& batch, const ResidualLayout& layout,
                             const ResidualRunOptions& options = {});

} // namespace mko::cuda
