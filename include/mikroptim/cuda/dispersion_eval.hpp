#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code.

#include <array>
#include <vector>

namespace mko::cuda {

// Wertet die Dispersionsformeln aus include/mikroptim/glass/dispersion.hpp auf der GPU aus
// (Prüfung, dass der gemeinsame Kern unter nvcc dieselben Werte liefert wie auf der CPU).
// Pro Wellenlänge lambda_um[i] [µm]: Sellmeier mit (B, C) und Buchdahl mit
// (n0, nu[0], nu[1], nu[2]). Rückgabe false bei CUDA-Fehlern.
bool evaluate_dispersion(const std::array<double, 3>& B, const std::array<double, 3>& C, double n0,
                         const std::array<double, 3>& nu, const std::vector<double>& lambda_um,
                         std::vector<double>& n_sellmeier, std::vector<double>& n_buchdahl);

} // namespace mko::cuda
