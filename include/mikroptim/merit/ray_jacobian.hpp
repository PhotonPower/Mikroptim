#pragma once

// Gradientenzeilen eines Strahls für die adjungierte Jacobi-Matrix (M5-T2), gemeinsam für die
// CPU-Referenz (adjoint_jacobian_cpu) und den Kernel ADJOINT (ADR-0005). Header-only, MKO_HD,
// templatisiert über R; ohne Ausnahmen, Allokation und std-Container. Formeln und Ablage:
// docs/numerik.md, Abschnitt "Rückwärtsdurchlauf (M5-T2)"; docs/architektur.md, Abschnitt
// "Modus ADJOINT: Jacobi-Matrix (M5-T2)".
//
// Die Spalten der Jacobi-Matrix sind die Variablen des Optimierers (VariableSet). Eine Variable
// ist genau ein Parameter der Flächentabelle (curvature[s], thickness[s], ghost_coeff[g][k]) oder
// – bei Glasvariablen – eine Buchdahl-Koordinate (n₀, ν₁, ν₂) des Mediums nach Fläche s, die über
// n(λ_l) = n₀ + ν₁ ω_l + ν₂ ω_l² + ν₃ ω_l³ in index[s][l] eingeht (∂n/∂n₀ = 1, ∂n/∂ν₁ = ω_l,
// ∂n/∂ν₂ = ω_l²). AdjointParameterMap ordnet jedem Parameter seine Spalte zu.

#include "mikroptim/config.hpp"
#include "mikroptim/math/vec3.hpp"
#include "mikroptim/trace/adjoint_backward.hpp"
#include "mikroptim/trace/surface_ops.hpp"

#include <cstdint>
#include <type_traits>

namespace mko {

// Spalte (Variable) jedes Parameters der Flächentabelle, −1 ohne Variable. Nur Zeiger und
// Größen, trivial kopierbar (zeigt auf Host-Vektoren, mko::AdjointParameters, oder in den
// Gerätespeicher).
struct AdjointParameterMap {
    int variables = 0;                   // P
    int surfaces = 0;                    // S
    int wavelengths = 0;                 // L
    int ghost_plates = 0;                // G
    int ghost_terms = 0;                 // K
    const int* curvature = nullptr;      // [S]: curvature[s]
    const int* thickness = nullptr;      // [S]: thickness[s]; s = S − 1 ist der Arbeitsabstand
    const int* glass = nullptr;          // [S · 3]: (n₀, ν₁, ν₂) des Mediums nach Fläche s
    const double* glass_slope = nullptr; // [L · 3]: ∂n(λ_l)/∂(n₀, ν₁, ν₂) = (1, ω_l, ω_l²)
    const int* ghost = nullptr;          // [G · K]: ghost_coeff[g][k]
};

static_assert(std::is_trivially_copyable_v<AdjointParameterMap>);

// Senke des Rückwärtsdurchlaufs (Schnittstelle in adjoint_backward.hpp), die die Ableitungen in
// eine Zeile der Jacobi-Matrix schreibt: J[p · rows + row] = factor · ∂Φ/∂x_p für jeden
// gemeldeten Parameter mit Variable p (spaltenweise Ablage, `column0` = Spalte 0 der Variante).
// Jede Variable wird pro Spur höchstens einmal geschrieben (ein Parameter pro Variable, jeder
// Parameter höchstens einmal gemeldet); nicht geschriebene Einträge muss der Aufrufer vorher
// auf 0 setzen. `wavelength` wählt die Steigungen der Glasvariablen.
template <class R>
struct JacobianRowWriter {
    AdjointParameterMap map;
    int wavelength;
    double* column0;
    std::int64_t rows;
    std::int64_t row;
    R factor;

    MKO_HD void write(int p, R value) const {
        if (p >= 0) {
            column0[static_cast<std::int64_t>(p) * rows + row] =
                static_cast<double>(factor * value);
        }
    }
    MKO_HD void curvature(int s, R value) const { write(map.curvature[s], value); }
    MKO_HD void thickness(int s, R value) const { write(map.thickness[s], value); }
    MKO_HD void index(int s, R value) const {
        for (int c = 0; c < 3; ++c) {
            const int p = map.glass[s * 3 + c];
            if (p >= 0) {
                write(p, value * R(map.glass_slope[wavelength * 3 + c]));
            }
        }
    }
    MKO_HD void ghost(int g, int k, R value) const {
        write(map.ghost[g * map.ghost_terms + k], value);
    }
};

// Hat der Strahl einen Gradienten seiner Strafgröße? Nur im weichen Modus und nur, wenn er
// mindestens ein weiches Ereignis hatte (soft_first ≠ Ok, auch wenn er danach hart versagt): Der
// Festanteil des harten Versagens ist konstant.
MKO_HD inline bool has_penalty_gradient(bool soft_mode, TraceStatus soft_first) {
    return soft_mode && soft_first != TraceStatus::Ok;
}

// Gradientenzeilen eines aufgezeichneten Strahls (Vorwärtsdurchlauf mit Aufzeichnung `rec` über
// batch_ray_opl, Status `status`, erstes weiches Ereignis `soft_first`):
// - Strahl im Kolben (in_piston): ∂OPL/∂x roh (vor dem Kolbenabzug) nach `opl_row`,
// - weiches Ereignis vorhanden: penalty_factor · ∂π/∂x nach `penalty_row` (penalty_factor =
//   penalty_scale · √(W_i V_j), das fertige Strafresiduum).
// Hart versagte Strahlen ohne weiches Ereignis schreiben nichts (Wellenfront- und Strafzeile 0).
template <class R, class SurfaceTable, class Record>
MKO_HD inline void
ray_gradient_rows(const SurfaceTable& table, int table_l, const Record& rec, TraceStatus status,
                  TraceStatus soft_first, bool soft_mode, const Vec3<R>& center, bool point_source,
                  const JacobianRowWriter<R>& opl_row, const JacobianRowWriter<R>& penalty_row) {
    const bool reached = in_piston(status);
    if (reached) {
        backward_ray(table, table_l, rec, true, center, point_source, AdjointSeed<R>{R(1), R(0)},
                     opl_row);
    }
    if (has_penalty_gradient(soft_mode, soft_first)) {
        backward_ray(table, table_l, rec, reached, center, point_source, AdjointSeed<R>{R(0), R(1)},
                     penalty_row);
    }
}

} // namespace mko
