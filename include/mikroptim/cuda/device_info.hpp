#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code.

#include <cstddef>
#include <vector>

namespace mko::cuda {

// Kleinste unterstützte Rechenfähigkeit (major·10 + minor): sm_86 (ADR-0015).
inline constexpr int kMinComputeCapability = 86;

// true, wenn die Kernel des Programms auf sm_<major><minor> laufen.
bool supported_architecture(int major, int minor);

// Anzahl verfügbarer CUDA-Geräte, 0 bei Fehlern oder ohne Treiber.
int device_count();

// Alle sichtbaren Geräte ab sm_86 (für die die Kernel übersetzt sind, ADR-0015), Indizes
// aufsteigend; leer ohne Gerät (Standardliste der Mehrgeräte-Engine, M7-T3).
std::vector<int> usable_devices();

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

// Kopiert `bytes` Byte aus Gerätespeicher (z. B. Zeiger aus DeviceTranche) in Host-Speicher
// (synchron, cudaMemcpy). Für Tests und Rückrufe ohne CUDA-Header. Wirft CudaError.
void copy_device_to_host(void* host, const void* device, std::size_t bytes);

} // namespace mko::cuda
