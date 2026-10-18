#pragma once

// Paraxialer y-nu-Trace, gemeinsam für CPU und GPU (docs/numerik.md, Abschnitt 9).
// Header-only, MKO_HD, templatisiert über den Zahlentyp R; ohne Ausnahmen und Allokation.
//
// Flächendaten als nackte Arrays der Länge `count` (Fläche 0 = Blende):
//   c[s]  Krümmung von Fläche s [1/mm]
//   t[s]  Dicke nach Fläche s (Abstand zum Scheitel von Fläche s+1) [mm]
//   n[s]  Brechzahl des Mediums nach Fläche s; vor Fläche 0 ist Luft (n = 1)

#include "mikroptim/config.hpp"

namespace mko {

// Paraxialer Strahl an einer Fläche: Höhe y [mm] und paraxialer Winkel u (Steigung dy/dz)
// im Medium, in dem der Strahl gerade läuft.
template <class R> struct ParaxialRay {
    R y;
    R u;
};

// Brechkraft einer Fläche: phi = c (n' - n).
template <class R> MKO_HD inline R surface_power(R c, R n_before, R n_after) {
    return c * (n_after - n_before);
}

// Brechung an einer Fläche der Krümmung c von n nach n': n' u' = n u - y phi.
template <class R>
MKO_HD inline void paraxial_refract(ParaxialRay<R>& r, R c, R n_before, R n_after) {
    r.u = (n_before * r.u - r.y * surface_power(c, n_before, n_after)) / n_after;
}

// Übergang zur nächsten Fläche im Abstand t: y <- y + t u.
template <class R> MKO_HD inline void paraxial_transfer(ParaxialRay<R>& r, R t) {
    r.y += t * r.u;
}

// Verfolgt einen paraxialen Strahl von Fläche 0 bis zur letzten Fläche.
// Eingabe: Höhe y an Fläche 0 und Winkel u vor Fläche 0 (in Luft).
// Rückgabe: Höhe an der letzten Fläche und Winkel u' nach ihr (im Bildraum).
// Ist `at_surface` gesetzt (Länge count), erhält at_surface[s] die Höhe an Fläche s und den
// Winkel nach der Brechung an Fläche s. Die Dicke der letzten Fläche wird nicht verwendet.
template <class R>
MKO_HD inline ParaxialRay<R> paraxial_trace(const R* c, const R* t, const R* n, int count,
                                            ParaxialRay<R> r,
                                            ParaxialRay<R>* at_surface = nullptr) {
    R n_before = R(1);
    for (int s = 0; s < count; ++s) {
        if (s > 0) {
            paraxial_transfer(r, t[s - 1]);
        }
        paraxial_refract(r, c[s], n_before, n[s]);
        if (at_surface != nullptr) {
            at_surface[s] = r;
        }
        n_before = n[s];
    }
    return r;
}

// Petzvalsumme P = sum_s c_s (n'_s - n_s) / (n_s n'_s) über alle Flächen (ohne Geisterplatten,
// deren Beiträge phi_G / n_G ergänzt der Aufrufer, docs/numerik.md, Abschnitt 6).
template <class R> MKO_HD inline R petzval_sum(const R* c, const R* n, int count) {
    R p = R(0);
    R n_before = R(1);
    for (int s = 0; s < count; ++s) {
        p += surface_power(c[s], n_before, n[s]) / (n_before * n[s]);
        n_before = n[s];
    }
    return p;
}

} // namespace mko
