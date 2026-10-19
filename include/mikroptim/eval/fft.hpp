#pragma once

// Schnelle Fourier-Transformation in FP64 für die CPU-Referenz der PSF (M8-T2), ohne externe
// Bibliothek (AGENTS.md, Abschnitt 9): iterativer Radix-2-Algorithmus (Cooley-Tukey,
// Bitumkehr, Twiddle-Faktoren aus std::cos und std::sin), nur Zweierpotenzen. Dieselbe
// Konvention wie cuFFT (Z2Z): Vorwärts mit exp(−2πi jk/n), rückwärts mit exp(+2πi jk/n), beide
// ohne Normierung (rückwärts nach vorwärts ergibt das n-fache). Festlegungen: docs/numerik.md,
// Abschnitt "Beugung: PSF, MTF, Strehl (M8-T2)".

#include <complex>
#include <span>

namespace mko {

// Vorzeichen des Exponenten.
enum class FftDirection { Forward, Inverse };

// true für n = 1, 2, 4, …
bool is_power_of_two(long long n);

// Eindimensionale FFT an Ort und Stelle; data.size() muss eine Zweierpotenz sein, sonst
// std::invalid_argument.
void fft(std::span<std::complex<double>> data, FftDirection direction);

// Zweidimensionale FFT an Ort und Stelle über ein zeilenweise abgelegtes Feld
// data[r · cols + c] (erst alle Zeilen, dann alle Spalten). rows und cols Zweierpotenzen,
// data.size() = rows · cols, sonst std::invalid_argument.
void fft2d(std::span<std::complex<double>> data, int rows, int cols, FftDirection direction);

} // namespace mko
