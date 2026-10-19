#pragma once

// Elementweise Schritte der PSF (M8-T2), gemeinsam für die CPU-Referenz (psf_cpu) und die
// Kernel der PsfEngine (ADR-0005). Header-only, MKO_HD, templatisiert über R; ohne Ausnahmen,
// Allokation und std-Container. Festlegungen: docs/numerik.md, Abschnitt "Beugung: PSF, MTF,
// Strehl (M8-T2)".

#include "mikroptim/config.hpp"
#include "mikroptim/math/scalar.hpp"

#include <cstdint>

namespace mko {

// 2π in FP64 (gerundet); die Phase ist kPsfTwoPi · (OPD/λ), auf CPU und GPU dieselbe Rundung.
inline constexpr double kPsfTwoPi = 6.283185307179586;

// Pupillenfunktion P = exp(2πi·OPD/λ) eines Rasterpunkts mit Amplitude 1; 0 bei OPD = NaN
// (außerhalb der Pupille oder versagter Strahl). λ als Vakuumwellenlänge in mm.
template <class R>
MKO_HD inline void psf_pupil_value(R opd, R lambda_mm, R& re, R& im) {
    if (!(opd == opd)) {
        re = R(0);
        im = R(0);
        return;
    }
    const R phase = R(kPsfTwoPi) * (opd / lambda_mm);
    re = mcos(phase);
    im = msin(phase);
}

// FFT-Index q (0 … M − 1) der Ausgabezeile oder -spalte i der zentrierten PSF:
// q ≡ s · (i − M/2) (mod M) mit s = +1 (konvergent) oder −1 (divergent, gespiegelt). Umgekehrt
// gehört der FFT-Index q zur Ausgabe i ≡ M/2 + s · q (mod M) (psf_output_index).
MKO_HD inline std::int64_t psf_fft_index(std::int64_t i, std::int64_t m, int sign) {
    const std::int64_t q = (static_cast<std::int64_t>(sign) * (i - m / 2)) % m;
    return q < 0 ? q + m : q;
}

MKO_HD inline std::int64_t psf_output_index(std::int64_t q, std::int64_t m, int sign) {
    const std::int64_t i = (m / 2 + static_cast<std::int64_t>(sign) * q) % m;
    return i < 0 ? i + m : i;
}

// Parameter einer Pupille für die Kernel der PsfEngine (24 Byte, trivial kopierbar).
struct PsfPupilParams {
    double lambda_mm; // Vakuumwellenlänge [mm]
    double scale;     // 1 / n_voll² (Normierung der Intensität)
    int sign_x;       // +1 konvergent, −1 divergent (gespiegelt)
    int sign_y;
};

// Betragsquadrat re² + im².
template <class R>
MKO_HD inline R psf_norm2(R re, R im) {
    return re * re + im * im;
}

} // namespace mko
