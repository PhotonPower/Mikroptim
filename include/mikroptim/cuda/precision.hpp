#pragma once

// Reiner C++-Header (keine CUDA-Typen), nutzbar aus Host-Code.
// Rechentyp der Trace-Kernel (M7-T1). Batch und Ausgaben sind in beiden Fällen FP64; der
// Zahlentyp bestimmt nur, womit der Kernel pro Strahl rechnet (docs/architektur.md, Abschnitt
// "Zahlentypen der Kernel", docs/numerik.md, Abschnitt 12).

#include <stdexcept>

namespace mko::cuda {

enum class Precision {
    Fp64 = 0, // double (Standard; Referenz, Tesla P100)
    Df64 = 1, // df64, Paar aus zwei float, ca. 48 Bit (GeForce-Karten, z. B. RTX 3090)
};

// Name für Meldungen ("FP64", "DF64").
inline const char* to_string(Precision precision) {
    switch (precision) {
    case Precision::Fp64:
        return "FP64";
    case Precision::Df64:
        return "DF64";
    }
    return "?";
}

// Wirft std::invalid_argument bei einem unbekannten Wert.
inline void check_precision(Precision precision) {
    switch (precision) {
    case Precision::Fp64:
    case Precision::Df64:
        return;
    }
    throw std::invalid_argument("unbekannter Zahlentyp (Precision)");
}

} // namespace mko::cuda
