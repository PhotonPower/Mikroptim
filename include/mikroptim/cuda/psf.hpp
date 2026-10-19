#pragma once

// Reiner C++-Header (keine CUDA- oder cuFFT-Typen), nutzbar aus Host-Code.
// Skalare PSF, MTF und Strehl auf der GPU (M8-T2): Pupillenfunktion, zweidimensionale FFT mit
// cuFFT (Z2Z, FP64, ein Plan für alle Pupillen einer Tranche, cufftPlanMany), Intensität und
// zentrierte PSF, FFT⁻¹ für die MTF-Schnitte. Dieselben Ergebnisse und Kennzahlen wie die
// CPU-Referenz psf_cpu (mikroptim/eval/psf.hpp). Ablauf und Speicher: docs/architektur.md,
// Abschnitt "PSF, MTF, Strehl"; Formeln: docs/numerik.md, Abschnitt "Beugung: PSF, MTF,
// Strehl (M8-T2)".

#include "mikroptim/cuda/cuda_error.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/eval/psf.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace mko::cuda {

// Angaben zum letzten Lauf von PsfEngine::compute.
struct PsfRunInfo {
    int pupils = 0;               // gerechnete Pupillen
    int tranche = 0;              // Pupillen pro Tranche (Batch des cuFFT-Plans)
    int tranches = 0;             // Zahl der Tranchen
    std::size_t work_bytes = 0;   // Arbeitsspeicher von cuFFT [Byte]
    std::size_t device_bytes = 0; // belegter Gerätespeicher nach dem Lauf [Byte]
    std::size_t budget_bytes = 0; // nutzbarer Speicher nach Reserve [Byte]
};

// PSF-Rechner auf einem Gerät (RAII, nur verschiebbar). Hält Puffer und cuFFT-Plan zwischen
// den Aufrufen und verwendet sie wieder, solange FFT-Größe und Tranche gleich bleiben.
// CUDA- und cuFFT-Fehler werfen CudaError, falsche Argumente std::invalid_argument; alle
// Aufrufe stellen das aktuelle Gerät des Aufrufers wieder her.
class PsfEngine {
public:
    // Wirft CudaError ohne Gerät, std::invalid_argument bei ungültigem Gerät oder Reserve.
    explicit PsfEngine(int device, const EngineOptions& options = {});
    ~PsfEngine();

    PsfEngine(const PsfEngine&) = delete;
    PsfEngine& operator=(const PsfEngine&) = delete;
    PsfEngine(PsfEngine&&) noexcept;
    PsfEngine& operator=(PsfEngine&&) noexcept;

    // PSF, MTF und Kennzahlen aller Pupillen (gleiche Rasterkante N) mit FFT-Größe M
    // (Zweierpotenz, 2N <= M <= kMaxPsfFft). Tranchen von T Pupillen nach dem freien
    // Gerätespeicher (cudaMemGetInfo, EngineOptions), pro Tranche Upload der OPD-Karten,
    // Kernel, zwei FFTs, Download von PSF und MTF; Kennzahlen mit finish_psf auf dem Host.
    // Wirft std::invalid_argument (Pupillen, M) und CudaError (auch wenn nicht eine Pupille in
    // den Speicher passt).
    std::vector<PsfResult> compute(std::span<const PsfPupil> pupils, int fft_size);

    // Alle Tupel der Variante v (make_psf_pupils, M = psf_fft_size(N, settings.padding)).
    std::vector<PsfResult> compute(const SystemBatchHost& batch, const EvalResult& result, int v,
                                   const PsfSettings& settings = {});

    const PsfRunInfo& last_run() const;
    int device() const;
    const EngineOptions& options() const;
    // Zurzeit belegter Gerätespeicher (Puffer und cuFFT-Arbeitsspeicher) [Byte].
    std::size_t device_bytes() const;
    // Gibt Puffer und Plan frei.
    void release() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mko::cuda
