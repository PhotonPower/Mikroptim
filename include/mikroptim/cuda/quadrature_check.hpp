#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code.

#include "mikroptim/pupil/quadrature.hpp"

#include <vector>

namespace mko::cuda {

// Prüfkernel für den CPU/GPU-Abgleich der Pupillenquadratur: berechnet alle Stützstellen
// sample(q, k), k = 0 … sample_count(q) − 1, auf der GPU. Die Quadratur wird per Wert an
// den Kernel übergeben, so wie später im Trace-Kernel.
// Rückgabe false bei CUDA-Fehlern.
bool pupil_samples_gpu(const PupilQuadrature<double>& q, std::vector<double>& rho,
                       std::vector<double>& phi, std::vector<double>& weight);

} // namespace mko::cuda
