#pragma once

// Normalgleichungen A = JᵀJ und g = Jᵀr eines Laufs auf dem Host (M3-T4). Feste
// Summationsreihenfolge: jedes Element ist eine Kette von FMA über die Zeilen in aufsteigender
// Reihenfolge, acc ← fma(u_r, v_r, acc) ab acc = 0. Der Kernel auf dem Gerät
// (mikroptim/cuda/normal_equations.hpp) rechnet jedes Element mit genau derselben Kette, die
// Ergebnisse sind daher bitgleich (docs/optimierung.md, Abschnitt 4).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace mko {

// Σ_r u_r·v_r als FMA-Kette in aufsteigender Reihenfolge r = 0 … n − 1 (std::fma ist exakt
// gerundet und damit plattformunabhängig, unabhängig von der FMA-Kontraktion des Compilers).
inline double gram_dot(const double* u, const double* v, std::int64_t n) {
    double acc = 0.0;
    for (std::int64_t r = 0; r < n; ++r) {
        acc = std::fma(u[r], v[r], acc);
    }
    return acc;
}

// A = JᵀJ (P × P spaltenweise, symmetrisch: A(p, q) = a[q·P + p]) und g = Jᵀr (Länge P) für
// J mit R Zeilen spaltenweise (Spalte p ab jacobian[p·R], Ablage eines Laufs von FdJacobian)
// und r (Länge R). Es wird nur p ≤ q gerechnet und gespiegelt (fma ist symmetrisch in den
// Faktoren). Wirft std::invalid_argument bei unpassenden Längen.
inline void normal_equations_host(std::span<const double> jacobian,
                                  std::span<const double> residual, int variables,
                                  std::span<double> a, std::span<double> g) {
    const auto P = static_cast<std::size_t>(variables);
    const std::size_t R = residual.size();
    if (variables < 1 || jacobian.size() != P * R || a.size() != P * P || g.size() != P) {
        throw std::invalid_argument("normal_equations_host: Längen passen nicht");
    }
    const auto n = static_cast<std::int64_t>(R);
    for (std::size_t q = 0; q < P; ++q) {
        const double* cq = jacobian.data() + q * R;
        for (std::size_t p = 0; p <= q; ++p) {
            const double v = gram_dot(jacobian.data() + p * R, cq, n);
            a[q * P + p] = v;
            a[p * P + q] = v;
        }
        g[q] = gram_dot(cq, residual.data(), n);
    }
}

} // namespace mko
