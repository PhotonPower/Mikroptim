#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code.

#include "mikroptim/cuda/device_system_batch.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/ray.hpp"

#include <vector>

namespace mko::cuda {

// Prüfkernel für das Batch-Layout (M2-T1, keine Kernel-Architektur aus M2-T2): Jeder Thread
// traced einen Strahl (v, f, l, k) mit batch_source_ray und trace_ray über
// view.surface_table(v) direkt aus dem globalen Speicher. Ablage wie trace_variant_cpu, mit
// Variante als äußerstem Index: ((v · F + f) · L + l) · N + k, N = sample_count(pupil).
// Status als unsigned (TraceStatus), versagende Fläche (−1 bei Ok).
// Wirft CudaError bei CUDA-Fehlern, std::logic_error bei leerem Batch.
void trace_batch_gpu(const DeviceSystemBatch& batch, const PupilQuadrature<double>& pupil,
                     std::vector<Ray<double>>& rays, std::vector<unsigned>& status,
                     std::vector<int>& failed_surface);

} // namespace mko::cuda
