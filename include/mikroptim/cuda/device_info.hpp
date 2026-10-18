#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code.

namespace mko::cuda {

// Anzahl verfügbarer CUDA-Geräte, 0 bei Fehlern oder ohne Treiber.
int device_count();

// Gibt Name, Architektur, Speicher und FP32/FP64-Verhältnis aller Geräte aus.
// Rückgabe: Anzahl der Geräte.
int print_devices();

} // namespace mko::cuda
