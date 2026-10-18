#pragma once

// Dispersionsformeln, gemeinsam für CPU und GPU (docs/numerik.md, Abschnitt 10).
// Alle Formeln erwarten die Wellenlänge in µm. Dateien und APIs verwenden nm;
// umgerechnet wird ausschließlich mit nm_to_um() (docs/numerik.md, Abschnitt 1).

#include "mikroptim/config.hpp"
#include "mikroptim/math/scalar.hpp"

namespace mko {

// Referenzwellenlänge der Buchdahl-Koordinate (Hg e-Linie) [µm].
inline constexpr double kBuchdahlLambda0Um = 0.546074;
// Konstante alpha in omega = dl / (1 + alpha dl) [1/µm].
inline constexpr double kBuchdahlAlpha = 2.5;

// Einzige Umrechnungsstelle nm -> µm für die Dispersionsformeln.
template <class R> MKO_HD inline R nm_to_um(R lambda_nm) {
    return lambda_nm / R(1000);
}

// Sellmeier: n^2 = 1 + sum_i B_i lambda^2 / (lambda^2 - C_i), lambda in µm, C_i in µm^2.
// B und C haben je drei Einträge. Keine Bereichsprüfung: Der Aufrufer stellt sicher,
// dass lambda im Gültigkeitsbereich liegt (dort ist n^2 > 0 und kein Pol).
template <class R> MKO_HD inline R sellmeier_index(const R B[3], const R C[3], R lambda_um) {
    const R l2 = lambda_um * lambda_um;
    R n2 = R(1);
    for (int i = 0; i < 3; ++i) {
        n2 += B[i] * l2 / (l2 - C[i]);
    }
    return msqrt(n2);
}

// Nenner der Buchdahl-Koordinate, 1 + alpha (lambda - lambda0). Die Koordinate ist nur für
// positive Werte definiert, also für lambda > lambda0 - 1/alpha = 0,146074 µm.
template <class R> MKO_HD inline R buchdahl_denominator(R lambda_um) {
    return R(1) + R(kBuchdahlAlpha) * (lambda_um - R(kBuchdahlLambda0Um));
}

// Buchdahl-Koordinate omega = dl / (1 + 2,5 dl) mit dl = lambda - lambda0 [µm].
template <class R> MKO_HD inline R buchdahl_omega(R lambda_um) {
    return (lambda_um - R(kBuchdahlLambda0Um)) / buchdahl_denominator(lambda_um);
}

// Buchdahl: n = n0 + nu1 omega + nu2 omega^2 + nu3 omega^3 (Horner-Schema).
template <class R> MKO_HD inline R buchdahl_index(R n0, R nu1, R nu2, R nu3, R lambda_um) {
    const R w = buchdahl_omega(lambda_um);
    return n0 + w * (nu1 + w * (nu2 + w * nu3));
}

} // namespace mko
