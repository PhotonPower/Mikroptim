#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code ohne nvcc.
// Modus ADJOINT, adjungierte Jacobi-Matrix auf der GPU (M5-T2): ein Block pro (Lauf, Feld,
// Wellenlänge), pro Strahl Vorwärtsdurchlauf mit Aufzeichnung (AdjointRecord im Local Memory),
// Rückwärtsdurchlauf (ray_gradient_rows, derselbe Code wie die CPU-Referenz) und danach der
// Kolbenabzug als Projektion mit deterministischen Blockreduktionen. Zusatzzeilen (finite
// Differenzen auf dem Host, hybrider Ansatz) werden hochgeladen. J, r₀ und Status bleiben im
// Gerätespeicher in der Ablage der FD-Jacobi (J_i(r, p) bei (i·P + p)·R + r), damit der LM
// (M5-T3) sie austauschen kann. docs/architektur.md, Abschnitt "Modus ADJOINT: Jacobi-Matrix
// (M5-T2)". Nur in Builds mit MIKROPTIM_ENABLE_CUDA.

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cuda/adjoint.hpp"
#include "mikroptim/cuda/device_system_batch.hpp"
#include "mikroptim/cuda/precision.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/optim/adjoint_jacobian.hpp"
#include "mikroptim/optim/fd_jacobian.hpp"
#include "mikroptim/optim/variable_set.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mko::cuda {

// Threads pro Block (wie RESIDUALS und das Grundgerüst ADJOINT: gleiche Kolbensumme).
inline constexpr int kAdjointJacobianThreads = kAdjointThreads;

// Ressourcen der Kernel-Variante (Kapazitätsklasse 16, 32, 64 der Aufzeichnung, Rechentyp) auf
// `device` (< 0: aktuelles Gerät). Wirft std::invalid_argument, CudaError.
AdjointKernelInfo adjoint_jacobian_kernel_info(int capacity, Precision precision = Precision::Fp64,
                                               int device = -1);

// Adjungierte Jacobi-Matrizen mehrerer Läufe im Gerätespeicher (RAII, nur verschiebbar). Ein
// cudaMalloc für J [runs][P][R], r₀ [runs][R], Status [runs][rays] und die Hilfsdaten
// (Feldablage, Spaltenzuordnung); wiederverwendet, wenn groß genug. Der Batch der Läufe (eine
// Variante pro Lauf) liegt in einem eigenen DeviceSystemBatch. CUDA-Fehler werfen CudaError,
// falsche Argumente std::invalid_argument; alle Aufrufe stellen das aktuelle Gerät des
// Aufrufers wieder her.
class DeviceAdjointJacobian {
public:
    DeviceAdjointJacobian() = default;
    ~DeviceAdjointJacobian();

    DeviceAdjointJacobian(const DeviceAdjointJacobian&) = delete;
    DeviceAdjointJacobian& operator=(const DeviceAdjointJacobian&) = delete;
    DeviceAdjointJacobian(DeviceAdjointJacobian&& other) noexcept;
    DeviceAdjointJacobian& operator=(DeviceAdjointJacobian&& other) noexcept;

    // Rechnet die Jacobi-Matrizen für runs = x0.size()/P Läufe auf dem Gerät von `engine`:
    // make_adjoint_batch, Spaltenzuordnung (AdjointParameters), Zusatzzeilen auf dem Host
    // (extra_rows_from_fd; trace-abhängige Zusatzresiduen aus den Extrema eines MERIT-Laufs der
    // Engine über den FD-Batch), Upload, Kernel, Zusatzzeilen in J und r₀ kopieren. Synchron.
    // `precision`: Rechentyp im Kernel (Ausgaben FP64). Wirft wie adjoint_jacobian_cpu und
    // TraceEngine::run, std::invalid_argument bei zu großem Grid, CudaError bei Speichermangel.
    // Nach einer Ausnahme ist ready() == false.
    void compute(TraceEngine& engine, const VariableSet& set, const SystemBatchHost& base,
                 std::span<const double> x0, const AdjointJacobianOptions& options = {},
                 Precision precision = Precision::Fp64);

    // Lädt J, r₀ und Status herunter (Ablage wie AdjointJacobian). Wirft std::logic_error ohne
    // erfolgreiches compute.
    AdjointJacobian download() const;
    void download(AdjointJacobian& out) const;

    void release() noexcept;

    bool ready() const { return ready_; }
    int device() const { return device_; }
    const JacobianShape& shape() const { return shape_; }
    int capacity() const { return capacity_class_; } // Kapazitätsklasse der Aufzeichnung
    std::size_t capacity_bytes() const { return bytes_; }

    // Gerätezeiger (gültig bis zum nächsten compute oder release; nullptr ohne compute).
    // J_i(r, p) bei (i·P + p)·R + r, r₀_i(r) bei i·R + r, Status bei i·rays + k.
    const double* device_jacobian() const;
    const double* device_residuals() const;
    const std::uint8_t* device_status() const;

private:
    int device_ = -1;
    void* buffer_ = nullptr;
    std::size_t bytes_ = 0;
    bool ready_ = false;
    JacobianShape shape_{};
    std::int64_t rays_ = 0;
    int capacity_class_ = 0;
    std::size_t residual_offset_ = 0; // Byte-Offsets im Puffer
    std::size_t status_offset_ = 0;
    DeviceSystemBatch batch_;
};

// Bequemlichkeit: compute mit eigenem Puffer und Download.
AdjointJacobian adjoint_jacobian_gpu(TraceEngine& engine, const VariableSet& set,
                                     const SystemBatchHost& base, std::span<const double> x0,
                                     const AdjointJacobianOptions& options = {},
                                     Precision precision = Precision::Fp64);

} // namespace mko::cuda
