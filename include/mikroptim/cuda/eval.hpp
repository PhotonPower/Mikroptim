#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code.
// Kernel EVAL (M8-T1): Auswertung auf dem dichten kartesischen Pupillenraster (OPD-Karten,
// Spotdiagramme, Feldstrahlen für Verzeichnung und Feldkrümmung), jedes Versagen hart.
// Programme benutzen den Modus über mko::cuda::TraceEngine::evaluate; Ablauf, Ablage und
// Speicher: docs/architektur.md, Abschnitt "Modus EVAL". Ergebnis, CPU-Referenz und Kennzahlen:
// mikroptim/eval/eval.hpp.

#include "mikroptim/cuda/cuda_error.hpp"
#include "mikroptim/cuda/device_system_batch.hpp"
#include "mikroptim/cuda/residuals.hpp"
#include "mikroptim/eval/eval.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace mko::cuda {

// Threads pro Block (wie RESIDUALS und MERIT); Thread t bearbeitet die Rasterpunkte
// k = t, t + 128, …
inline constexpr int kEvalThreads = kResidualThreads;

// Ausgabepuffer des Kernels EVAL auf dem Gerät (RAII, nur verschiebbar), wiederverwendet, wenn
// groß genug. CUDA-Fehler werfen CudaError, falsche Argumente std::invalid_argument; alle
// Aufrufe stellen das aktuelle Gerät des Aufrufers wieder her.
class DeviceEval {
public:
    DeviceEval() = default;
    ~DeviceEval();

    DeviceEval(const DeviceEval&) = delete;
    DeviceEval& operator=(const DeviceEval&) = delete;
    DeviceEval(DeviceEval&& other) noexcept;
    DeviceEval& operator=(DeviceEval&& other) noexcept;

    // Startet den Kernel (ein Block pro (v, f, l)) für alle Varianten von `batch` mit dem Raster
    // `grid` und wartet auf das Ende. `reference_curvature` sind die Krümmungen κ der
    // Referenzsphären der Varianten des Batches auf dem Gerät ([V·F], eval_reference_geometry).
    // Wirft std::invalid_argument bei ungültigen Einstellungen, leerem Raster, falscher Länge
    // von `reference_curvature` und V · F · L > kMaxGridBlocks; std::logic_error bei leerem
    // Batch.
    void run(const DeviceSystemBatch& batch, const EvalGrid& grid, const EvalSettings& settings,
             std::span<const double> reference_curvature);

    // Kopiert die Ausgaben des letzten Laufs an die Stellen der Varianten first … first + V − 1
    // von `out` (prepare_eval_result; Vektoren groß genug, gleiches Raster, sonst
    // std::invalid_argument). Wirft std::logic_error ohne Lauf.
    void download(EvalResult& out, int first) const;

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
    int fields_ = 0;
    int wavelengths_ = 0;
    std::int64_t samples_ = 0;
    EvalOutputLayout layout_{};
};

} // namespace mko::cuda
