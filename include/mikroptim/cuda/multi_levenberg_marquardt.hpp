#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code ohne nvcc.
// Gebündelter Levenberg-Marquardt auf mehreren GPUs (M7-T3): Die Läufe eines Aufrufs werden
// in zusammenhängende Bereiche geteilt (Anteile der MultiTraceEngine, Arten Jacobian und Merit
// in FP64), jeder Bereich läuft mit einem GpuLmBackend auf seinem Gerät. Da die Ergebnisse
// eines Laufs nur von seinem Index und seinem Punkt abhängen (LmBackend), sind sie bitgleich
// zu optimize_lm_gpu auf einem Gerät gleicher Architektur. Nur in Builds mit
// MIKROPTIM_ENABLE_CUDA.

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cuda/levenberg_marquardt.hpp"
#include "mikroptim/cuda/multi_trace_engine.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/variable_set.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace mko::cuda {

// Backend auf allen Slots von `engines`. normal_equations und merit teilen `runs` nach
// engines.plan(·, Jacobian bzw. Merit, Fp64), rechnen die Teile parallel (run_parallel) und
// melden die Zeiten an engines.record (gleitender Mittelwert der Anteile). Die Zusatzresiduen
// (options.jacobian.extra.evaluate) werden dabei aus mehreren Threads gleichzeitig aufgerufen
// und müssen thread-sicher sein (make_paraxial_residuals ist es). `engines`, `set`, `base` und
// `options` müssen leben, solange das Backend benutzt wird; nicht gleichzeitig aus mehreren
// Threads benutzen.
class MultiGpuLmBackend final : public LmBackend {
public:
    MultiGpuLmBackend(MultiTraceEngine& engines, const VariableSet& set,
                      const SystemBatchHost& base, const LmOptions& options);
    ~MultiGpuLmBackend() override;

    MultiGpuLmBackend(const MultiGpuLmBackend&) = delete;
    MultiGpuLmBackend& operator=(const MultiGpuLmBackend&) = delete;

    void normal_equations(std::span<const int> runs, std::span<const double> x, std::span<double> a,
                          std::span<double> g) override;
    void merit(std::span<const int> runs, std::span<const double> x, std::span<double> merit,
               std::span<std::int64_t> failed) override;

    // Statistik pro Slot (Summen über alle Aufrufe) und über alle Slots summiert
    // (max_runs_per_chunk als Maximum).
    std::vector<LmGpuStats> slot_stats() const;
    LmGpuStats stats() const;
    // Läufe pro Slot über alle Aufrufe von normal_equations bzw. merit.
    const std::vector<std::int64_t>& jacobian_runs_per_slot() const { return jacobian_runs_; }
    const std::vector<std::int64_t>& merit_points_per_slot() const { return merit_points_; }

private:
    MultiTraceEngine& engines_;
    const VariableSet& set_;
    std::vector<std::unique_ptr<GpuLmBackend>> backends_;
    std::vector<std::int64_t> jacobian_runs_;
    std::vector<std::int64_t> merit_points_;
};

// Optimiert runs = x0.size()/P Läufe auf allen Geräten von `engines` (Rechentyp FP64); sonst
// wie optimize_lm_gpu. `stats` (optional) erhält die Statistik pro Slot.
LmResult optimize_lm_multi_gpu(MultiTraceEngine& engines, const VariableSet& set,
                               const SystemBatchHost& base, std::span<const double> x0,
                               const LmOptions& options = {},
                               std::vector<LmGpuStats>* stats = nullptr);

} // namespace mko::cuda
