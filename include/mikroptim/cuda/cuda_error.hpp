#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code.

#include <stdexcept>

namespace mko::cuda {

// Fehler einer CUDA-API beim Konfigurieren (Speicher anlegen, Upload, Download, Geräteabfrage).
// what() enthält Datei, Zeile und die Meldung der CUDA-Laufzeit.
class CudaError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

} // namespace mko::cuda
