#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code ohne nvcc.
// GPU-Backend des gebündelten Levenberg-Marquardt (M3-T4): Jacobi-Matrizen per
// DeviceFdJacobian, Normalgleichungen per DeviceNormalEquations (deterministischer Kernel),
// Merit der Versuchspunkte im Modus MERIT der TraceEngine (Zusatzresiduen als extra_merit).
// Die Steuerlogik ist dieselbe wie auf der CPU (levenberg_marquardt in
// mikroptim/optim/levenberg_marquardt.hpp); docs/optimierung.md, Abschnitt 4. Nur in Builds mit
// MIKROPTIM_ENABLE_CUDA.

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/cuda/fd_jacobian.hpp"
#include "mikroptim/cuda/normal_equations.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/variable_set.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace mko::cuda {

// Statistik des GPU-Backends (Summen über alle Aufrufe).
struct LmGpuStats {
    int jacobian_calls = 0;  // Aufrufe normal_equations
    int jacobian_chunks = 0; // Teilstücke (Läufe, deren J zusammen in den Speicher passt)
    std::int64_t jacobian_runs = 0;
    int merit_calls = 0;
    std::int64_t merit_points = 0;
    int max_runs_per_chunk = 0;         // größte Zahl von Läufen in einem Teilstück
    std::int64_t jacobian_tranches = 0; // Tranchen der Engine in den Jacobi-Rechnungen
    std::int64_t merit_tranches = 0;    // Tranchen der Engine in den Merit-Rechnungen
};

// Backend auf dem Gerät von `engine`. normal_equations teilt die Läufe in Teilstücke, deren
// Jacobi-Matrizen zusammen höchstens die Hälfte des verfügbaren Speichers belegen
// (cudaMemGetInfo zuzüglich des schon belegten J-Puffers, höchstens
// EngineOptions::memory_limit_bytes, abzüglich der Reserve der Engine); die andere Hälfte bleibt
// für die Engine. Teilstücke bestehen aus ganzen Läufen, die Ergebnisse eines Laufs hängen
// nicht von der Aufteilung ab. `engine`, `set`, `base` und `options` müssen leben, solange das
// Backend benutzt wird; nicht gleichzeitig aus mehreren Threads benutzen.
class GpuLmBackend final : public LmBackend {
public:
    GpuLmBackend(TraceEngine& engine, const VariableSet& set, const SystemBatchHost& base,
                 const LmOptions& options);

    void normal_equations(std::span<const int> runs, std::span<const double> x, std::span<double> a,
                          std::span<double> g) override;
    void merit(std::span<const int> runs, std::span<const double> x, std::span<double> merit,
               std::span<std::int64_t> failed) override;

    // Größte Zahl von Läufen, deren Jacobi-Matrizen gemeinsam gerechnet werden (aus dem
    // Speicher, siehe oben; mindestens 1).
    int max_jacobian_runs() const;
    const LmGpuStats& stats() const { return stats_; }

private:
    TraceEngine& engine_;
    const VariableSet& set_;
    const SystemBatchHost& base_;
    const LmOptions& options_;
    DeviceFdJacobian jacobian_;
    DeviceNormalEquations normal_;
    TraceOutput output_;
    LmGpuStats stats_{};
};

// Optimiert runs = x0.size()/P Läufe auf dem Gerät von `engine` (Rechentyp FP64). `base` hat
// eine Variante (für alle Läufe) oder eine pro Lauf; die Zusatzresiduen kommen aus
// options.jacobian.extra. Wirft wie levenberg_marquardt, DeviceFdJacobian und TraceEngine.
LmResult optimize_lm_gpu(TraceEngine& engine, const VariableSet& set, const SystemBatchHost& base,
                         std::span<const double> x0, const LmOptions& options = {},
                         LmGpuStats* stats = nullptr);

} // namespace mko::cuda
