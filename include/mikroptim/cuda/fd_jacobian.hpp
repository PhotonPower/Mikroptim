#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code ohne nvcc.
// Finite-Differenzen-Jacobi auf der GPU (M3-T2): Die Varianten des FD-Batches laufen über die
// TraceEngine (Modus Residuals, run_on_device), die Differenzen bildet ein elementweiser
// Kernel direkt aus dem Ausgabepuffer der Engine. J und r₀ bleiben im Gerätespeicher (für JᵀJ
// und Jᵀr per cuBLAS, M3-T4) und lassen sich bei Bedarf herunterladen. Ablage und Formeln wie
// der CPU-Pfad (mikroptim/optim/fd_jacobian.hpp); docs/optimierung.md, Abschnitt
// „Finite-Differenzen-Jacobi“. Nur in Builds mit MIKROPTIM_ENABLE_CUDA.

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/optim/fd_jacobian.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "mikroptim/optim/variant_generator.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mko::cuda {

// Threads pro Block des Differenzkernels (ein Block pro Lauf und Spalte, Threads über die
// Zeilen).
inline constexpr int kFdJacobianThreads = 128;

// Jacobi-Matrizen mehrerer Läufe im Gerätespeicher (RAII, nur verschiebbar). Ein cudaMalloc
// für J [runs][P][R], r₀ [runs][R], Status-Wechsel [runs][P] (int64) und die Hilfsdaten
// (Zusatzresiduen aller Varianten, Schritte und Schablonen); wiederverwendet, wenn groß genug.
// CUDA-Fehler werfen CudaError, falsche Argumente std::invalid_argument; alle Aufrufe stellen
// das aktuelle Gerät des Aufrufers wieder her.
class DeviceFdJacobian {
public:
    DeviceFdJacobian() = default;
    ~DeviceFdJacobian();

    DeviceFdJacobian(const DeviceFdJacobian&) = delete;
    DeviceFdJacobian& operator=(const DeviceFdJacobian&) = delete;
    DeviceFdJacobian(DeviceFdJacobian&& other) noexcept;
    DeviceFdJacobian& operator=(DeviceFdJacobian&& other) noexcept;

    // Rechnet die Jacobi-Matrizen für runs = x0.size()/P Läufe auf dem Gerät von `engine`:
    // make_fd_jacobian_batch, Zusatzresiduen aller Varianten auf dem Host
    // (evaluate_extra_residuals aus paraxial_residuals.hpp), Puffer anlegen und Hilfsdaten
    // hochladen, dann engine.run_on_device mit Tranchen aus ganzen Läufen; pro Tranche bildet der
    // Differenzkernel die Spalten ihrer Läufe. Synchron: Bei der Rückkehr sind J, r₀ und die
    // Status-Wechsel fertig. Die Zusatzresiduen gehen nicht durch den Ausgabepuffer der
    // Engine (extra_residuals = 0), sondern liegen in einem eigenen Gerätepuffer [V · E] mit
    // E = options.extra_count(). Trace-abhängige Zusatzresiduen (options.trace_extra, M4-T3):
    // die Engine liefert die Max-Reduktionen (extrema), pro Tranche werden sie heruntergeladen,
    // auf dem Host ausgewertet (evaluate_trace_extra_residuals) und die Zeilen der Varianten der
    // Tranche im Gerätepuffer ersetzt, bevor der Differenzkernel läuft.
    // Wirft wie make_fd_jacobian_batch, evaluate_extra_residuals und TraceEngine::run_on_device;
    // CudaError, wenn J nicht in den Gerätespeicher passt. Nach einer Ausnahme ist der Inhalt
    // unbestimmt (ready() == false). `precision`: Rechentyp der Trace-Kernel (M7-T1; ab M5-T3
    // für den Vergleich mit der adjungierten Jacobi, Benchmark jacobian_throughput), Ausgaben FP64.
    void compute(TraceEngine& engine, const VariableSet& set, const SystemBatchHost& base,
                 std::span<const double> x0, const FdJacobianOptions& options = {},
                 Precision precision = Precision::Fp64);

    // Lädt J, r₀ und Status-Wechsel herunter (Ablage wie FdJacobian, Muster vom Host).
    // Wirft std::logic_error ohne erfolgreiches compute.
    FdJacobian download() const;
    void download(FdJacobian& out) const;

    // Gibt den Gerätepuffer frei.
    void release() noexcept;

    bool ready() const { return ready_; }
    int device() const { return device_; }
    const JacobianShape& shape() const { return shape_; }
    FdScheme scheme() const { return scheme_; }
    std::span<const FdPattern> patterns() const { return patterns_; }
    // Statistik des letzten Laufs der Engine (Tranchen usw.).
    const TraceRunInfo& run_info() const { return info_; }
    std::size_t capacity_bytes() const { return capacity_; }

    // Gerätezeiger (gültig bis zum nächsten compute oder release; nullptr ohne compute).
    // J_i(r, p) bei (i·P + p)·R + r, r₀_i(r) bei i·R + r, Wechsel bei i·P + p.
    const double* device_jacobian() const;
    const double* device_residuals() const;
    const std::int64_t* device_status_changes() const;

    // Gerätespeicher des Puffers für `shape` und `variants` FD-Varianten [Byte] (ohne die
    // Puffer der Engine), z. B. zur Wahl der Zahl der Läufe pro Batch.
    static std::size_t buffer_bytes(const JacobianShape& shape, int variants);

private:
    int device_ = -1;
    void* buffer_ = nullptr;
    std::size_t capacity_ = 0;
    bool ready_ = false;
    JacobianShape shape_{};
    FdScheme scheme_ = FdScheme::Central;
    std::vector<FdPattern> patterns_;
    TraceRunInfo info_{};
    std::size_t residual_offset_ = 0; // Byte-Offsets der Regionen im Puffer
    std::size_t changes_offset_ = 0;
};

// Bequemlichkeit: compute mit eigenem Puffer und Download.
FdJacobian fd_jacobian_gpu(TraceEngine& engine, const VariableSet& set, const SystemBatchHost& base,
                           std::span<const double> x0, const FdJacobianOptions& options = {});

} // namespace mko::cuda
