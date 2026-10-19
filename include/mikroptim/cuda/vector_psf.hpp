#pragma once

// Reiner C++-Header (keine CUDA- oder cuFFT-Typen), nutzbar aus Host-Code.
// Vektorielle PSF auf der GPU (M8-T5, Richards-Wolf): pro Pupille und Eingangsfeld drei
// Feldkomponenten in der Pupille (Kernel), zweidimensionale FFT mit cuFFT (Z2Z, FP64, ein Plan
// für alle Komponenten einer Tranche), Intensität, Komponenten und zentrierte PSF (Kernel),
// FFT⁻¹ der Intensität für die MTF. Dieselben Ergebnisse und Kennzahlen wie die CPU-Referenz
// vector_psf_cpu (mikroptim/eval/vector_psf.hpp). Ablauf und Speicher: docs/architektur.md,
// Abschnitt "Vektorielle PSF"; Formeln: docs/numerik.md, Abschnitt "Vektorielle PSF (M8-T5)".

#include "mikroptim/cuda/cuda_error.hpp"
#include "mikroptim/cuda/psf.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/eval/vector_psf.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace mko::cuda {

// Vektorieller PSF-Rechner auf einem Gerät (RAII, nur verschiebbar). Hält Puffer und die beiden
// cuFFT-Pläne zwischen den Aufrufen und verwendet sie wieder, solange FFT-Größe, Zahl der
// Eingangsfelder und Tranche gleich bleiben. CUDA- und cuFFT-Fehler werfen CudaError, falsche
// Argumente std::invalid_argument; alle Aufrufe stellen das aktuelle Gerät des Aufrufers wieder
// her.
class VectorPsfEngine {
public:
    // Wirft CudaError ohne Gerät, std::invalid_argument bei ungültigem Gerät oder Reserve.
    explicit VectorPsfEngine(int device, const EngineOptions& options = {});
    ~VectorPsfEngine();

    VectorPsfEngine(const VectorPsfEngine&) = delete;
    VectorPsfEngine& operator=(const VectorPsfEngine&) = delete;
    VectorPsfEngine(VectorPsfEngine&&) noexcept;
    VectorPsfEngine& operator=(VectorPsfEngine&&) noexcept;

    // Vektorielle PSF, Komponenten, MTF und Kennzahlen aller Pupillen (gleiche Rasterkante N)
    // mit FFT-Größe M (Zweierpotenz, 2N <= M <= kMaxPsfFft). Tranchen von T Pupillen nach dem
    // freien Gerätespeicher (cudaMemGetInfo, EngineOptions); Kennzahlen mit finish_vector_psf
    // auf dem Host. Wirft std::invalid_argument (Pupillen, M, Einstellungen) und CudaError (auch
    // wenn nicht eine Pupille in den Speicher passt).
    std::vector<VectorPsfResult> compute(std::span<const VectorPsfPupil> pupils, int fft_size,
                                         const VectorPsfSettings& settings = {});

    // Alle Tupel der Variante v (make_vector_psf_pupils, M = psf_fft_size(N, padding)).
    std::vector<VectorPsfResult> compute(const SystemBatchHost& batch, const EvalResult& result,
                                         int v, const PsfSettings& psf_settings = {},
                                         const VectorPsfSettings& settings = {});

    const PsfRunInfo& last_run() const;
    int device() const;
    const EngineOptions& options() const;
    // Zurzeit belegter Gerätespeicher (Puffer und cuFFT-Arbeitsspeicher) [Byte].
    std::size_t device_bytes() const;
    // Gibt Puffer und Pläne frei.
    void release() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mko::cuda
