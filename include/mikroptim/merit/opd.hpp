#pragma once

// Kolbenabzug und RMS-Wellenfront über die Pupillenstützstellen eines Tupels
// (Feld, Wellenlänge), gemeinsam für CPU und GPU (docs/numerik.md, Abschnitt 7).
// Header-only, MKO_HD, templatisiert über R; ohne Ausnahmen und Allokation.

#include "mikroptim/config.hpp"
#include "mikroptim/math/scalar.hpp"
#include "mikroptim/trace/surface_ops.hpp"
#include "mikroptim/trace/trace.hpp"

namespace mko {

template <class R>
struct OpdStatistics {
    R valid_weight; // Σ W_k über die Strahlen im Kolben (in_piston: Ok oder weich versagt)
    R piston;       // Kolben Σ W_k OPL_k / valid_weight [mm]; 0 ohne gültigen Strahl
    R rms;          // √(Σ W_k OPD_k² / valid_weight) [mm]; 0 ohne gültigen Strahl
    int failed;     // Anzahl der Strahlen mit Status ≠ Ok (weich und hart)
};

// Kolbenabzug und gewichteter RMS der OPD über `count` Stützstellen (Arrays opl, weight,
// status der Länge count; Gewichte W_k aus PupilQuadrature, Summe 1).
//
// Hart versagte Strahlen (is_hard_failure) sind ausgeschlossen: Sie gehen weder in den Kolben
// noch in den RMS ein, die Gewichte der übrigen werden renormiert (Division durch
// valid_weight). Ihre OPD wird 0 gesetzt (maskiert, wie ein Residuum im Kernel). Weich
// versagte Strahlen (M4-T2, nur aus dem weichen Modus) zählen wie gültige ("ok" unten heißt
// in_piston); ohne sie ist das die Rechnung von M1-T7.
//
//   Pass 1: valid_weight = Σ_ok W_k,  S = Σ_ok W_k OPL_k,  piston = S / valid_weight
//   Pass 2: OPD_k = OPL_k − piston (ok), 0 (versagt);  rms = √(Σ_ok W_k OPD_k² / valid_weight)
//
// Zwei Durchläufe statt √(⟨OPL²⟩ − ⟨OPL⟩²), weil OPL ≈ 100 mm und OPD ≈ 1e-6 mm sonst
// vollständig ausgelöscht würden. Summiert wird in aufsteigender Reihenfolge k = 0 … count−1
// (Index der Quadratur, ringweise). Das Ergebnis ist damit deterministisch; ein Kernel mit
// Baumreduktion (andere, aber feste Reihenfolge, ADR-0007) weicht nur im Rundungsfehler ab.
//
// `opd` darf nullptr sein (dann nur die Statistik; R folgt aus opl und weight).
// Ohne gültigen Strahl ist valid_weight = 0, piston = rms = 0 und jede OPD 0; der Aufrufer
// prüft valid_weight > 0.
template <class R>
MKO_HD inline OpdStatistics<R> opd_statistics(const R* opl, const R* weight,
                                              const TraceStatus* status, int count,
                                              typename detail::NonDeduced<R>::type* opd) {
    OpdStatistics<R> st{R(0), R(0), R(0), 0};
    R sum = R(0);
    for (int k = 0; k < count; ++k) {
        if (in_piston(status[k])) {
            st.valid_weight += weight[k];
            sum += weight[k] * opl[k];
        }
        if (status[k] != TraceStatus::Ok) {
            ++st.failed;
        }
    }
    if (!(st.valid_weight > R(0))) {
        st.valid_weight = R(0);
        if (opd != nullptr) {
            for (int k = 0; k < count; ++k) {
                opd[k] = R(0);
            }
        }
        return st;
    }
    st.piston = sum / st.valid_weight;
    R sum_sq = R(0);
    for (int k = 0; k < count; ++k) {
        R e = R(0);
        if (in_piston(status[k])) {
            e = opl[k] - st.piston;
            sum_sq += weight[k] * e * e;
        }
        if (opd != nullptr) {
            opd[k] = e;
        }
    }
    st.rms = msqrt(sum_sq / st.valid_weight);
    return st;
}

} // namespace mko
