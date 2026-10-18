#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code.

#include <cstddef>

namespace mko::cuda {

// Anzahl verfügbarer CUDA-Geräte, 0 bei Fehlern oder ohne Treiber.
int device_count();

// Gibt Name, Architektur, Speicher und FP32/FP64-Verhältnis aller Geräte aus.
// Rückgabe: Anzahl der Geräte.
int print_devices();

// Freier und gesamter Gerätespeicher in Byte (cudaMemGetInfo).
struct DeviceMemoryInfo {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
};

// Speicher von Gerät `device` (< 0: aktuelles Gerät); das aktuelle Gerät des Aufrufers bleibt
// unverändert. Wirft CudaError (mikroptim/cuda/cuda_error.hpp) bei CUDA-Fehlern.
DeviceMemoryInfo device_memory_info(int device = -1);

} // namespace mko::cuda
