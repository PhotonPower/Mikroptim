#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code ohne nvcc.
// GPU-Backend der Monte-Carlo-Tolerierung (M8-T4): setzt in ToleranceSettings den Trace auf den
// Modus RESIDUALS der TraceEngine (OPD und Status, hartes Versagen, FP64) und den Optimierer des
// Kompensators auf optimize_lm_gpu. Die Ergebnisse jeder Stichprobe hängen nicht von der
// Aufteilung in Tranchen der Engine ab (ADR-0007). Nur in Builds mit MIKROPTIM_ENABLE_CUDA
// (Bibliotheken Mikroptim::cuda und Mikroptim::tolerance).

#include "mikroptim/cuda/levenberg_marquardt.hpp"
#include "mikroptim/cuda/trace_engine.hpp"
#include "mikroptim/tolerance/tolerance_analysis.hpp"

#include <span>
#include <utility>

namespace mko::cuda {

// Setzt settings.trace und settings.optimizer auf das Gerät von `engine`; `engine` muss leben,
// solange settings benutzt wird.
inline void use_gpu(ToleranceSettings& settings, TraceEngine& engine) {
    TraceEngine* e = &engine;
    settings.trace = [e](const SystemBatchHost& batch, const ResidualLayout& layout,
                         ResidualResult& out) {
        TraceSettings ts;
        ts.quadrature = layout.settings();
        ts.opd = true;
        ts.status = true;
        TraceOutput o;
        e->run(batch, TraceMode::Residuals, ts, o);
        out = std::move(o.residuals);
    };
    settings.optimizer = [e](const VariableSet& set, const SystemBatchHost& base,
                             std::span<const double> x0, const LmOptions& options) {
        return optimize_lm_gpu(*e, set, base, x0, options);
    };
}

} // namespace mko::cuda
