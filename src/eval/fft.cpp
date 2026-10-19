// Radix-2-FFT in FP64 (M8-T2), CPU-Referenz der PSF. Iterativ: Bitumkehr, dann log2(n) Stufen
// von Schmetterlingen. Die zweidimensionale Transformation läuft über die Zeilen (Elemente
// zusammenhängend) und danach über die Spalten, wobei dieselbe Stufenfolge auf ganze Zeilen
// angewandt wird (Zeile j ist dann das Element j; zusammenhängender Speicherzugriff).

#include "mikroptim/eval/fft.hpp"

#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mko {

namespace {

using Complex = std::complex<double>;

// Produkt ohne die Sonderbehandlung von NaN/∞ aus C99 Anhang G (std::complex::operator* ruft
// dafür in GCC eine Bibliotheksfunktion auf); die Eingaben sind endlich.
inline Complex mul(Complex a, Complex b) {
    return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
}

// Bitumkehr-Permutation und Twiddle-Faktoren w_k = exp(∓2πi k/n), k < n/2, einer Länge n.
struct Plan {
    std::vector<std::size_t> reverse;
    std::vector<Complex> twiddle;

    Plan(std::size_t n, FftDirection direction) : reverse(n), twiddle(n / 2) {
        int bits = 0;
        while ((std::size_t{1} << bits) < n) {
            ++bits;
        }
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t r = 0;
            for (int b = 0; b < bits; ++b) {
                r |= ((i >> b) & 1u) << (bits - 1 - b);
            }
            reverse[i] = r;
        }
        const double sign = direction == FftDirection::Forward ? -1.0 : 1.0;
        for (std::size_t k = 0; k < n / 2; ++k) {
            const double angle =
                2.0 * std::numbers::pi * static_cast<double>(k) / static_cast<double>(n);
            twiddle[k] = {std::cos(angle), sign * std::sin(angle)};
        }
    }
};

// Transformation von n Elementen der Breite `width` (Element j liegt bei data[j·stride …
// j·stride + width − 1]; width = 1, stride = 1 für eine Zeile, width = stride = cols für die
// Spalten).
void transform(Complex* data, std::size_t n, std::size_t width, std::size_t stride,
               const Plan& plan) {
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t r = plan.reverse[i];
        if (r > i) {
            for (std::size_t c = 0; c < width; ++c) {
                std::swap(data[i * stride + c], data[r * stride + c]);
            }
        }
    }
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const std::size_t half = len / 2;
        const std::size_t step = n / len;
        for (std::size_t start = 0; start < n; start += len) {
            for (std::size_t k = 0; k < half; ++k) {
                const Complex w = plan.twiddle[k * step];
                Complex* a = data + (start + k) * stride;
                Complex* b = data + (start + k + half) * stride;
                for (std::size_t c = 0; c < width; ++c) {
                    const Complex t = mul(w, b[c]);
                    b[c] = a[c] - t;
                    a[c] += t;
                }
            }
        }
    }
}

void require_power_of_two(long long n, const char* what) {
    if (!is_power_of_two(n)) {
        throw std::invalid_argument(std::string("FFT: ") + what + " = " + std::to_string(n) +
                                    " ist keine Zweierpotenz");
    }
}

} // namespace

bool is_power_of_two(long long n) {
    return n > 0 && (n & (n - 1)) == 0;
}

void fft(std::span<std::complex<double>> data, FftDirection direction) {
    require_power_of_two(static_cast<long long>(data.size()), "Länge");
    const Plan plan(data.size(), direction);
    transform(data.data(), data.size(), 1, 1, plan);
}

void fft2d(std::span<std::complex<double>> data, int rows, int cols, FftDirection direction) {
    require_power_of_two(rows, "Zeilenzahl");
    require_power_of_two(cols, "Spaltenzahl");
    const auto r = static_cast<std::size_t>(rows);
    const auto c = static_cast<std::size_t>(cols);
    if (data.size() != r * c) {
        throw std::invalid_argument("FFT: Feldgröße passt nicht zu rows · cols");
    }
    const Plan row_plan(c, direction);
    for (std::size_t j = 0; j < r; ++j) {
        transform(data.data() + j * c, c, 1, 1, row_plan);
    }
    const Plan column_plan(r, direction);
    transform(data.data(), r, c, c, column_plan);
}

} // namespace mko
