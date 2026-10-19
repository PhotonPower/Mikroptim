#pragma once

// Geisterplatte (docs/numerik.md, Abschnitt 6): virtuelle dünne Glasplatte in einem Luftspalt,
// modelliert als Ebene mit Phasenfunktion. Gemeinsam für CPU-Tracer und CUDA-Kernel
// (ADR-0005): header-only, MKO_HD, templatisiert über R, ohne Ausnahmen, Allokation und
// std-Container.
//
// Getrennt sind
// - der allgemeine Ablenkungsschritt an einer ebenen Phasenfläche mit gegebenem Φ und ∇Φ
//   (phase_surface_step), z. B. für eine lineare Phase Φ = a·y,
// - das rotationssymmetrische Dickenprofil als Summe von Legendre-Polynomen in u = (r/r_g)²
//   (ghost_profile) und
// - der Schritt an der Geisterplatte, der beides verbindet (ghost_plate_step).

#include "mikroptim/config.hpp"
#include "mikroptim/math/scalar.hpp"
#include "mikroptim/math/vec3.hpp"
#include "mikroptim/trace/ray.hpp"
#include "mikroptim/trace/surface_ops.hpp"

namespace mko {

// Daten einer Geisterplatte bei einer Wellenlänge, wie sie der Tracer an propagate_surface
// übergibt (trivial kopierbar, nur ein Zeiger auf die Koeffizienten).
template <class R>
struct GhostPlateParams {
    const R* coeff; // a_1 … a_K des Dickenprofils [mm], coeff[k − 1] = a_k
    int terms;      // K >= 0
    R norm_radius;  // Normierungsradius r_g [mm], > 0
    R n_glass;      // Brechzahl n_G(λ) des virtuellen Glases
};

// Legendre-Polynom P_k(ξ) und seine Ableitung P'_k(ξ) über die Rekursionen
// (k + 1) P_{k+1} = (2k + 1) ξ P_k − k P_{k−1} und P'_{k+1} = P'_{k−1} + (2k + 1) P_k
// mit P_0 = 1, P_1 = ξ, P'_0 = 0, P'_1 = 1 (docs/numerik.md, Abschnitt 6). k >= 0.
template <class R>
MKO_HD inline void legendre(int k, R xi, R& p, R& dp) {
    R p_prev = R(1); // P_{j−1}
    R p_cur = xi;    // P_j
    R dp_prev = R(0);
    R dp_cur = R(1);
    if (k == 0) {
        p = R(1);
        dp = R(0);
        return;
    }
    for (int j = 1; j < k; ++j) {
        const R p_next = (R(2 * j + 1) * xi * p_cur - R(j) * p_prev) / R(j + 1);
        const R dp_next = dp_prev + R(2 * j + 1) * p_cur;
        p_prev = p_cur;
        p_cur = p_next;
        dp_prev = dp_cur;
        dp_cur = dp_next;
    }
    p = p_cur;
    dp = dp_cur;
}

// Dickenprofil an der Stelle u = (r/r_g)² und seine Ableitung nach u.
template <class R>
struct GhostProfile {
    R t;     // t(u) = Σ_{k=1}^{K} a_k P_k(2u − 1) [mm]
    R dt_du; // dt/du = 2 Σ_k a_k P'_k(2u − 1) [mm]
};

// Wertet t und dt/du in einem Durchlauf der Rekursionen aus (Summation in aufsteigender
// Ordnung k = 1 … K). u darf größer als 1 sein (Strahlen außerhalb von r_g); die Polynome
// wachsen dann, bleiben aber endlich.
template <class R>
MKO_HD inline GhostProfile<R> ghost_profile(const R* coeff, int terms, R u) {
    const R xi = R(2) * u - R(1);
    R p_prev = R(1); // P_{k−1}
    R p_cur = xi;    // P_k
    R dp_prev = R(0);
    R dp_cur = R(1);
    R t = R(0);
    R dt = R(0); // Σ a_k P'_k
    for (int k = 1; k <= terms; ++k) {
        const R a = coeff[k - 1];
        t += a * p_cur;
        dt += a * dp_cur;
        const R p_next = (R(2 * k + 1) * xi * p_cur - R(k) * p_prev) / R(k + 1);
        const R dp_next = dp_prev + R(2 * k + 1) * p_cur;
        p_prev = p_cur;
        p_cur = p_next;
        dp_prev = dp_cur;
        dp_cur = dp_next;
    }
    return GhostProfile<R>{t, R(2) * dt};
}

// Paraxialer Koeffizient t₂ des Dickenprofils, t(r) = t(0) + t₂ r² + O(r⁴):
// t₂ = (1/r_g²) Σ_k a_k (−1)^{k+1} k (k + 1), aus P'_k(−1) = (−1)^{k+1} k (k + 1)/2.
template <class R>
MKO_HD inline R ghost_profile_t2(const R* coeff, int terms, R norm_radius) {
    R sum = R(0);
    for (int k = 1; k <= terms; ++k) {
        const R term = coeff[k - 1] * R(k * (k + 1));
        sum += (k % 2 == 1) ? term : -term;
    }
    return sum / (norm_radius * norm_radius);
}

// Paraxiale Brechkraft der Geisterplatte im Medium n_m (Luft: n_m = 1):
// φ_G = −2 (n_G − n_m) t₂ (docs/numerik.md, Abschnitt 6).
template <class R>
MKO_HD inline R ghost_plate_power(const R* coeff, int terms, R norm_radius, R n_glass, R n_medium) {
    return R(-2) * (n_glass - n_medium) * ghost_profile_t2(coeff, terms, norm_radius);
}

// Beitrag der Geisterplatte zur Petzvalsumme: φ_G / (n_m n_G), in Luft φ_G / n_G (dünne Linse
// aus dem Glas n_G im Medium n_m, docs/numerik.md, Abschnitte 6 und 9).
template <class R>
MKO_HD inline R ghost_plate_petzval(const R* coeff, int terms, R norm_radius, R n_glass,
                                    R n_medium) {
    return ghost_plate_power(coeff, terms, norm_radius, n_glass, n_medium) / (n_medium * n_glass);
}

// Allgemeiner Schritt an einer ebenen Phasenfläche im Strahlschnittpunkt (z = 0): OPL += Φ,
// dann n' d'_⊥ = n d_⊥ + ∇Φ und d'_z = √(1 − |d'_⊥|²) (Vorwärtsrichtung). `grad_x`, `grad_y`
// sind die Komponenten von ∇Φ im Schnittpunkt (dimensionslos, Φ in mm). Bei Geisterplatten
// ist n = n' = n_m (Luftspalt); n ≠ n' ergibt eine brechende Ebene mit Phase.
// Status TIR, wenn der Radikand 1 − |d'_⊥|² negativ ist (evaneszente Welle wie bei der
// Totalreflexion); der Strahl ist dann teilweise fortgeschrieben (OPL) und nicht zu verwenden.
// Weiches Versagen (`soft` gesetzt, numerik.md §11): −Radikand wird addiert und mit
// Radikand := 0 fortgesetzt, d' = (d'_x, d'_y, 0)/|d'_⊥| (Status SoftTIR: fortgesetzt, Strahl
// gültig; seit M5-T1, damit der Aufrufer den Zweig kennt). Wegen d'_z = 0 versagt der
// Strahl an der nächsten Prüfung von d_z hart (Backward); das Strafresiduum enthält dann
// −Radikand und den festen Anteil des harten Versagens.
template <class R>
MKO_HD inline TraceStatus
phase_surface_step(Ray<R>& ray, R phase, R grad_x, R grad_y, R n_before, R n_after,
                   typename detail::NonDeduced<SoftFailure<R>>::type* soft = nullptr) {
    ray.opl += phase;
    const R dx = (n_before * ray.d.x + grad_x) / n_after;
    const R dy = (n_before * ray.d.y + grad_y) / n_after;
    const R radicand = R(1) - dx * dx - dy * dy;
    if (radicand < R(0)) {
        if (soft == nullptr) {
            return TraceStatus::TIR;
        }
        soft->add(-radicand, TraceStatus::SoftTIR);
        const R inv = R(1) / msqrt(dx * dx + dy * dy); // |d'_⊥|² = 1 − Radikand > 1
        ray.d = Vec3<R>{dx * inv, dy * inv, R(0)};
        return TraceStatus::SoftTIR;
    }
    ray.d = Vec3<R>{dx, dy, msqrt(radicand)};
    return TraceStatus::Ok;
}

// Schritt an einer Geisterplatte im Strahlschnittpunkt (Ebene z = 0, Medium n_m = n_before):
// u = (x² + y²)/r_g², Φ = (n_G − n_m) t(u),
// ∇Φ = (n_G − n_m) (dt/du) (2/r_g²) (x, y) = (n_G − n_m) (4/r_g²) Σ_k a_k P'_k(2u − 1) (x, y),
// danach phase_surface_step. Status und weiches Versagen wie dort.
template <class R>
MKO_HD inline TraceStatus
ghost_plate_step(Ray<R>& ray, const GhostPlateParams<R>& g, R n_before, R n_after,
                 typename detail::NonDeduced<SoftFailure<R>>::type* soft = nullptr) {
    const R inv_r2 = R(1) / (g.norm_radius * g.norm_radius);
    const R u = (ray.p.x * ray.p.x + ray.p.y * ray.p.y) * inv_r2;
    const GhostProfile<R> prof = ghost_profile(g.coeff, g.terms, u);
    const R dn = g.n_glass - n_before;
    const R radial = dn * prof.dt_du * R(2) * inv_r2; // ∇Φ = radial · (x, y)
    return phase_surface_step(ray, dn * prof.t, radial * ray.p.x, radial * ray.p.y, n_before,
                              n_after, soft);
}

} // namespace mko
