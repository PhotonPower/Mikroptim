#pragma once

// Differenzenformeln der finiten-Differenzen-Jacobi (M3-T2), header-only für Host und Gerät
// (MKO_HD, templatisiert über den Zahlentyp R). Die Schablone pro Variable legt fd_pattern
// fest (variant_generator.hpp); Formeln, Fehlerordnung und Rundung: docs/optimierung.md,
// Abschnitt „Finite-Differenzen-Jacobi“.

#include "mikroptim/config.hpp"

#include <cstdint>

namespace mko {

// Schablone der Differenz für eine Variable. Die zwei gestörten Punkte heißen a und b
// (Varianten 2k + 1 und 2k + 2 des zentralen Musters, a = „plus“, b = „minus“), d_a und d_b
// sind die tatsächlichen Schritte fl(x − x₀).
enum class FdStencil : std::uint8_t {
    // x₀ ± h liegen in den Schranken: d_a ≈ h, d_b ≈ −h. J = (r_a − r_b)/(d_a − d_b),
    // Abbruchfehler h²/6·r‴ (zweite Ordnung).
    Central = 0,
    // Eine Schranke liegt näher als h, auf der anderen Seite σ ist Platz für 2h:
    // d_a ≈ σh, d_b ≈ 2σh (beide auf derselben Seite). Dreipunktformel für ungleiche Schritte
    // J = (d_b²(r_a − r₀) − d_a²(r_b − r₀)) / (d_a·d_b·(d_b − d_a)), Abbruchfehler
    // −d_a·d_b/6·r‴ = −h²/3·r‴ (zweite Ordnung).
    OneSided = 1,
    // Bereich der Variable kürzer als etwa 3h: d_a = min(x₀ + h, upper) − x₀ ≥ 0,
    // d_b = max(x₀ − h, lower) − x₀ ≤ 0. J = (r_a − r_b)/(d_a − d_b), Fehler
    // (d_a + d_b)/2·r″ (erste Ordnung; liegt x₀ auf einer Schranke, ist ein Schritt 0 und
    // der Punkt gleich x₀, also eine einseitige Differenz).
    Truncated = 2,
    // Vorwärtsschema (P + 1 Varianten): J = (r_a − r₀)/d_a, Fehler d_a/2·r″ (erste Ordnung).
    Forward = 3,
};

// Ableitungsschätzung aus r₀ = r(x₀), r_a = r(x₀ + d_a), r_b = r(x₀ + d_b) (r_b wird bei
// Forward nicht gelesen). Dieselbe Funktion rechnet auf dem Host (fd_jacobian_cpu) und im
// Differenzkernel auf dem Gerät.
template <class R>
MKO_HD inline R fd_derivative(FdStencil stencil, R r0, R ra, R rb, R da, R db) {
    switch (stencil) {
    case FdStencil::OneSided: {
        const R ga = ra - r0;
        const R gb = rb - r0;
        return (db * db * ga - da * da * gb) / (da * db * (db - da));
    }
    case FdStencil::Forward:
        return (ra - r0) / da;
    case FdStencil::Central:
    case FdStencil::Truncated:
    default:
        return (ra - rb) / (da - db);
    }
}

} // namespace mko
