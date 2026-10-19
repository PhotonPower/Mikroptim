#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code.
// Kernel MERIT (M2-T3): Merit F_v = Σ r_k² pro Variante, versagte Strahlen und optional die
// Max-Reduktionen pro Fläche auf der GPU, ohne den Residuenvektor im globalen Speicher abzulegen
// (Liniensuche, Multistart-Screening). Reduktionsschema, Shared-Memory-Belegung und
// Summationsreihenfolge: docs/architektur.md, Abschnitt "Kernel MERIT". Die TraceEngine (M2-T4)
// kapselt diese Klasse.

#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/batch/batch_merit.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/cuda/cuda_error.hpp"
#include "mikroptim/cuda/device_system_batch.hpp"
#include "mikroptim/cuda/precision.hpp"
#include "mikroptim/cuda/residuals.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace mko::cuda {

// Threads pro Block des Kernels MERIT (gleich kResidualThreads). Beide Kernel bilden den
// Kolben mit derselben Baumreduktion (128 virtuelle Lanes pro Tupel, unabhängig von Blockgröße
// und Paketen, M7-T4): OPD und Residuen sind in beiden Modi bitgleich, und die Merit
// unterscheidet sich von Σ RESIDUALS² nur durch die Reihenfolge der Quadratsumme.
inline constexpr int kMeritThreads = kResidualThreads;

// Shared Memory pro Block, das der Kernel höchstens belegt (ohne Opt-in auf beiden Karten).
inline constexpr std::size_t kMeritMaxSharedBytes = 48 * 1024;

// Optionen eines Laufs.
struct MeritRunOptions {
    bool extrema = false; // Max-Reduktionen pro Variante und Fläche (MeritResult::extrema)
    // Erweiterungspunkt für weitere Residuen (Paraxial, Strafen ab M3/M4): leer oder ein Wert
    // pro Variante, den der Kernel nach der Summe der Strahlresiduen addiert,
    // F_v = (Σ_(f, l) Teilsumme) + extra_merit[v]. Der Aufrufer berechnet ihn (z. B. Σ e² der
    // weiteren Residuen); die Daten werden bei run kopiert.
    std::span<const double> extra_merit{};
    Precision precision = Precision::Fp64; // Rechentyp im Kernel (M7-T1); Ergebnisse immer FP64
    // Größte Paketgröße in Strahlen (>= 1, M7-T4); nur Laufzeit, die Ergebnisse sind für jeden
    // Wert bitgleich.
    int pack_rays = kDefaultPackRays;
};

// Ausgabepuffer des Kernels MERIT auf dem Gerät (RAII, nur verschiebbar): ein cudaMalloc im
// Layout make_output_layout (Merit, Teilsummen, versagte Strahlen, optional Max-Reduktionen)
// plus Feldablage (ResidualFieldPlan[F]) und optional extra_merit[V]. Der Puffer wird
// wiederverwendet, wenn er groß genug ist. CUDA-Fehler werfen CudaError, falsche Argumente
// std::invalid_argument; alle Aufrufe stellen das aktuelle Gerät des Aufrufers wieder her.
class DeviceMerit {
public:
    DeviceMerit() = default;
    ~DeviceMerit();

    DeviceMerit(const DeviceMerit&) = delete;
    DeviceMerit& operator=(const DeviceMerit&) = delete;
    DeviceMerit(DeviceMerit&& other) noexcept;
    DeviceMerit& operator=(DeviceMerit&& other) noexcept;

    // Startet den Kernel (ein Block pro Tupelpaket einer Variante, M7-T4) und die zweite
    // Reduktionsstufe für alle
    // Varianten von `batch` auf dessen Gerät und wartet auf das Ende. Wirft
    // std::invalid_argument, wenn `layout` nicht zur Topologie passt, V · F · L >
    // kMaxGridBlocks, extra_merit weder leer noch V Werte lang ist, options.precision unbekannt
    // ist, pack_rays < 1 ist oder OPL, Status und Strafresiduen der größten Stützstellenzahl
    // eines Feldes nicht in das Shared Memory passen (merit_max_samples);
    // std::logic_error bei leerem Batch. Die Werte der Varianten prüft der Aufrufer auf dem Host
    // (SystemBatchHost::validate_values).
    void run(const DeviceSystemBatch& batch, const ResidualLayout& layout,
             const MeritRunOptions& options = {});

    // Lädt Merit, versagte Strahlen und (wenn angefordert) Max-Reduktionen des letzten Laufs
    // herunter. Wirft std::logic_error ohne vorherigen Lauf.
    MeritResult download() const;
    void download(MeritResult& result) const;

    // Gibt den Gerätepuffer frei.
    void release() noexcept;

    int device() const { return device_; }
    int variants() const { return variants_; }
    bool has_extrema() const { return extrema_; }
    const OutputLayout& layout() const { return layout_; }
    std::size_t capacity_bytes() const { return capacity_; }

    // Gerätezeiger auf die Ergebnisse des letzten Laufs (für Folgekernel, z. B. die Auswahl in
    // einer Liniensuche); gültig bis zum nächsten run oder release, nullptr ohne Lauf.
    // device_merit: [V] (FP64), device_failed und device_hard_failed: [V] (int64),
    // device_extrema: [V][S][2] oder nullptr ohne Anforderung.
    const double* device_merit() const;
    const std::int64_t* device_failed() const;
    const std::int64_t* device_hard_failed() const;
    const double* device_extrema() const;

private:
    int device_ = -1;
    void* buffer_ = nullptr;
    std::size_t capacity_ = 0;
    bool ran_ = false;
    bool extrema_ = false;
    int variants_ = 0;
    int surfaces_ = 0;
    std::int64_t rays_ = 0;
    OutputLayout layout_{};
};

// Bequemlichkeit: ein Lauf mit eigenem Puffer und Download.
MeritResult run_merit(const DeviceSystemBatch& batch, const ResidualLayout& layout,
                      const MeritRunOptions& options = {});

// Größte Stützstellenzahl n_f eines Feldes, deren OPL, Status und (weiches Versagen, M4-T2)
// Strafresiduen der Kernel MERIT im Shared Memory ablegen kann: n_f · 9 Byte (hart) bzw.
// n_f · 17 Byte (weich) zusätzlich zum statischen Shared Memory, den Geisterdaten
// (G · K + 2 G) · 8 Byte und den Brechzahlen einer Wellenlänge für kMaxSurfaces Flächen,
// zusammen <= kMeritMaxSharedBytes (Tupelpakete, M7-T4); mit Max-Reduktionen und
// Geisterplatten kleiner. Fragt die Kernelattribute der Kernel-Variante für `precision` auf dem
// aktuellen Gerät ab (CudaError); std::invalid_argument bei negativen Zahlen oder unbekanntem
// Zahlentyp.
int merit_max_samples(bool extrema, int ghost_plates = 0, int ghost_terms = 0,
                      Precision precision = Precision::Fp64,
                      FailureMode failure = FailureMode::Soft);

} // namespace mko::cuda
