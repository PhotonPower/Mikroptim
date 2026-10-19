#pragma once

// Rückwärtsdurchlauf des adjungierten Modus (M5-T2), gemeinsam für CPU-Referenz und CUDA-Kernel
// (ADR-0005): header-only, MKO_HD, templatisiert über R, ohne Ausnahmen, Allokation und
// std-Container. Herleitung und Formeln: docs/numerik.md, Abschnitt "Rückwärtsdurchlauf (M5-T2)".
//
// Pro Strahl ist die Ausgabe einer Spur ("Lane") der Skalar Φ = ō·OPL + π̄·π mit konstanten
// Gewichten ō, π̄ (AdjointSeed): (1, 0) für die optische Weglänge bis zur Referenzsphäre,
// (0, 1) für die Strafgröße π des weichen Versagens (numerik.md §11). Beide gehen additiv ein
// (OPL += …, π += …), ihre Adjungierten sind deshalb auf jeder Fläche dieselben Konstanten.
// Der Rückwärtsdurchlauf läuft von der Referenzsphäre über die Flächen S − 1 … 1 zum
// Startstrahl und trägt die Adjungierte des Strahlzustands (p̄, d̄) (RayAdjoint); jede
// Flächenoperation liefert dabei die Adjungierten ihrer Systemparameter (Dicke des Transfers,
// Krümmung, Brechzahlen vor und nach der Fläche, Koeffizienten der Geisterplatte, Mittelpunkt
// der Referenzsphäre).
//
// Die lokalen Zwischengrößen jeder Operation werden aus dem Eingangszustand (aufgezeichnet im
// Vorwärtsdurchlauf, adjoint_record.hpp) mit denselben Formeln wie im Vorwärtsdurchlauf neu
// berechnet, und zwar für den aufgezeichneten Zweig des weichen Versagens (branch): Der
// Rückwärtsdurchlauf differenziert genau die Fortsetzung, die der Vorwärtsdurchlauf genommen hat.
// Abgeleitet werden die berechneten Formeln selbst (nicht ihre geometrische Bedeutung, z. B.
// |d| = 1 oder |N| = 1 auf der Sphäre); damit stimmen die Ableitungen mit finiten Differenzen
// der Implementierung überein.

#include "mikroptim/config.hpp"
#include "mikroptim/math/scalar.hpp"
#include "mikroptim/math/vec3.hpp"
#include "mikroptim/trace/adjoint_record.hpp"
#include "mikroptim/trace/ghost_plate.hpp"
#include "mikroptim/trace/surface_ops.hpp"
#include "mikroptim/trace/surface_table.hpp"

namespace mko {

// Gewichte einer Spur: Φ = opl · OPL + penalty · π.
template <class R>
struct AdjointSeed {
    R opl;
    R penalty;
};

// Adjungierte eines Strahlzustands (Ort und Richtung), Φ nach p bzw. d.
template <class R>
struct RayAdjoint {
    Vec3<R> p;
    Vec3<R> d;
};

// Adjungierte der Parameter eines Flächenschritts (werden addiert; der Aufrufer setzt 0).
template <class R>
struct SurfaceAdjoint {
    R thickness; // Abstand t der Scheitel (thickness[s − 1]) des Transfers
    R curvature; // Krümmung c der Fläche (wirksam; an Ebenen ohne Bedeutung)
    R n_before;  // Brechzahl vor der Fläche (index[s − 1])
    R n_after;   // Brechzahl nach der Fläche (index[s])
};

namespace detail {

template <class R>
MKO_HD inline Vec3<R> vec_zero() {
    return Vec3<R>{R(0), R(0), R(0)};
}

// Transfer in die Tangentialebene (transfer_to_tangent_plane): s₀ = (t − z)/d_z,
// (x, y) = (p_x + s₀ d_x, p_y + s₀ d_y), z = 0.
template <class R>
struct TransferLocal {
    R s0;
    R x;
    R y;
};

template <class R>
MKO_HD inline TransferLocal<R> transfer_local(const Vec3<R>& p, const Vec3<R>& d, R t) {
    TransferLocal<R> tl;
    const R z = p.z - t;
    tl.s0 = -z / d.z;
    tl.x = p.x + tl.s0 * d.x;
    tl.y = p.y + tl.s0 * d.y;
    return tl;
}

// Rückwärts durch den Transfer: aus den Adjungierten (x̄, ȳ) des Punkts in der Tangentialebene
// und s̄₀ die Adjungierten des Eingangszustands (addiert auf in.p, in.d) und von t.
template <class R>
MKO_HD inline void transfer_backward(const Vec3<R>& p, const Vec3<R>& d, const TransferLocal<R>& tl,
                                     R xb, R yb, R s0b, RayAdjoint<R>& in, R& tb) {
    (void)p;
    in.p.x += xb;
    in.p.y += yb;
    s0b += xb * d.x + yb * d.y;
    in.d.x += xb * tl.s0;
    in.d.y += yb * tl.s0;
    // s₀ = (t − p_z)/d_z
    const R q = s0b / d.z;
    tb += q;
    in.p.z += -q;
    in.d.z += -q * tl.s0;
}

} // namespace detail

// Rückwärts durch einen Schritt an einer Fläche vom Typ Stop, Plane oder Sphere
// (propagate_surface: Transfer um t, Schnitt mit der Sphäre der wirksamen Krümmung c, OPL +=
// n_vor (s₀ + s), Brechung mit μ = n_vor/n_nach; Zweige kBranchSoftMissed: s = B/c mit normierter
// Normale und Strafe −D, kBranchSoftTIR: d′ = (d − cos I N)/|…| mit Strafe −k).
// Eingang: Zustand (p, d) vor der Fläche (nach Fläche s − 1, in deren lokalen Koordinaten).
// `adj` enthält beim Aufruf die Adjungierte des Ausgangszustands (Schnittpunkt, Richtung nach der
// Brechung) und bei der Rückkehr die des Eingangszustands; die Parameteradjungierten werden auf
// `out` addiert.
template <class R>
MKO_HD inline void refracting_step_backward(const Vec3<R>& p, const Vec3<R>& d, R t, R c,
                                            R n_before, R n_after, unsigned char branch,
                                            AdjointSeed<R> seed, RayAdjoint<R>& adj,
                                            SurfaceAdjoint<R>& out) {
    // ---- Vorwärts (lokale Größen, Formeln wie in surface_ops.hpp) ----
    const detail::TransferLocal<R> tl = detail::transfer_local(p, d, t);
    const R x = tl.x;
    const R y = tl.y;
    const R rr = x * x + y * y;
    const R H = c * rr;
    const R xd = x * d.x + y * d.y;
    const R B = d.z - c * xd;
    const R D = B * B - c * H;
    const bool missed = (branch & kBranchSoftMissed) != 0;
    const bool tir = (branch & kBranchSoftTIR) != 0;
    R root_d = R(0);
    R s;
    if (missed) {
        s = B / c;
    } else {
        root_d = msqrt(D);
        s = H / (B + root_d);
    }
    const Vec3<R> q{x + s * d.x, y + s * d.y, s * d.z};
    const Vec3<R> n_raw{-c * q.x, -c * q.y, R(1) - c * q.z};
    R inv_norm = R(1);
    Vec3<R> N = n_raw;
    if (missed) {
        inv_norm = R(1) / norm(n_raw);
        N = inv_norm * n_raw;
    }
    const R mu = n_before / n_after;
    const R cos_i = dot(d, N);
    const R sin2 = R(1) - cos_i * cos_i;
    const R k = R(1) - mu * mu * sin2;

    // ---- Rückwärts ----
    const Vec3<R> qb_out = adj.p; // Adjungierte des Schnittpunkts
    const Vec3<R> d1b = adj.d;    // Adjungierte der Richtung nach der Brechung
    RayAdjoint<R> in{detail::vec_zero<R>(), detail::vec_zero<R>()};
    Vec3<R> nb = detail::vec_zero<R>(); // Adjungierte von N
    R cosb = R(0);
    R kb = R(0);
    R mub = R(0);
    if (tir) {
        // d′ = τ/|τ|, τ = d − cos I N
        const Vec3<R> tau = d - cos_i * N;
        const R inv_tau = R(1) / norm(tau);
        const Vec3<R> d1 = inv_tau * tau;
        const Vec3<R> taub = inv_tau * (d1b - dot(d1, d1b) * d1);
        in.d += taub;
        cosb += -dot(taub, N);
        nb += (-cos_i) * taub;
        kb += -seed.penalty; // Strafe −k
    } else {
        // d′ = μ d + (√k − μ cos I) N
        const R root_k = msqrt(k);
        const R a = root_k - mu * cos_i;
        in.d += mu * d1b;
        nb += a * d1b;
        const R ab = dot(d1b, N);
        mub += dot(d1b, d);
        kb += ab / (R(2) * root_k);
        mub += -ab * cos_i;
        cosb += -ab * mu;
    }
    // k = 1 − μ² (1 − cos² I)
    mub += -kb * R(2) * mu * sin2;
    cosb += kb * R(2) * mu * mu * cos_i;
    // cos I = d·N
    in.d += cosb * N;
    nb += cosb * d;
    // μ = n_vor / n_nach
    out.n_before += mub / n_after;
    out.n_after += -mub * mu / n_after;
    // N = N_roh (regulär) bzw. N_roh/|N_roh| (Zweig SoftMissed)
    Vec3<R> nrb = nb;
    if (missed) {
        nrb = inv_norm * (nb - dot(N, nb) * N);
    }
    // N_roh = (−c q_x, −c q_y, 1 − c q_z)
    const Vec3<R> qb = qb_out + (-c) * nrb;
    R cb = -dot(q, nrb);
    // q = (x + s d_x, y + s d_y, s d_z)
    R xb = qb.x;
    R yb = qb.y;
    R sb = dot(qb, d);
    in.d += s * qb;
    // OPL += n_vor (s₀ + s)
    sb += seed.opl * n_before;
    R s0b = seed.opl * n_before;
    out.n_before += seed.opl * (tl.s0 + s);
    R hb = R(0);
    R bb = R(0);
    if (missed) {
        // s = B/c
        bb += sb / c;
        cb += -sb * s / c;
        // Strafe −D, D = B² − c H
        const R db = -seed.penalty;
        bb += db * R(2) * B;
        cb += -db * H;
        hb += -db * c;
    } else {
        // s = H/(B + √D) ist die Nullstelle von c s² − 2 B s + H: ds = (s² dc − 2 s dB + dH)/(2√D)
        const R inv = R(1) / (R(2) * root_d);
        hb += sb * inv;
        bb += -sb * s * R(2) * inv;
        cb += sb * s * s * inv;
    }
    // H = c (x² + y²)
    cb += hb * rr;
    xb += hb * R(2) * c * x;
    yb += hb * R(2) * c * y;
    // B = d_z − c (x d_x + y d_y)
    in.d.z += bb;
    cb += -bb * xd;
    xb += -bb * c * d.x;
    yb += -bb * c * d.y;
    in.d.x += -bb * c * x;
    in.d.y += -bb * c * y;
    // Transfer
    detail::transfer_backward(p, d, tl, xb, yb, s0b, in, out.thickness);
    out.curvature += cb;
    adj = in;
}

// Rückwärts durch einen Schritt an einer Geisterplatte (propagate_surface mit kind = GhostPlate:
// Transfer in die Ebene, OPL += n_vor s₀, ghost_plate_step; Zweig kBranchSoftTIR: d′ = (d′_x,
// d′_y, 0)/|d′_⊥| mit Strafe −Radikand). Parameter wie refracting_step_backward (out.curvature
// bleibt unberührt), dazu die Koeffizienten: coeff_sink(k, ā) mit k = 0 … K − 1 für a_{k+1}
// (Φ nach dem Koeffizienten). Brechzahl n_G und Normierungsradius sind feste Daten.
// Profil: t(u) = Σ a_k P_k(ξ), dt/du = 2 Σ a_k P′_k(ξ), d²t/du² = 4 Σ a_k P″_k(ξ), ξ = 2u − 1,
// mit P″_{k+1} = P″_{k−1} + (2k + 1) P′_k (Ableitung der Rekursion für P′).
template <class R, class CoeffSink>
MKO_HD inline void
ghost_step_backward(const Vec3<R>& p, const Vec3<R>& d, R t, const GhostPlateParams<R>& g,
                    R n_before, R n_after, unsigned char branch, AdjointSeed<R> seed,
                    RayAdjoint<R>& adj, SurfaceAdjoint<R>& out, CoeffSink&& coeff_sink) {
    // ---- Vorwärts ----
    const detail::TransferLocal<R> tl = detail::transfer_local(p, d, t);
    const R x = tl.x;
    const R y = tl.y;
    const R inv_r2 = R(1) / (g.norm_radius * g.norm_radius);
    const R u = (x * x + y * y) * inv_r2;
    const R xi = R(2) * u - R(1);
    R prof_t = R(0);
    R sum_dp = R(0);  // Σ a_k P′_k
    R sum_ddp = R(0); // Σ a_k P″_k
    {
        R p_prev = R(1);
        R p_cur = xi;
        R dp_prev = R(0);
        R dp_cur = R(1);
        R ddp_prev = R(0);
        R ddp_cur = R(0);
        for (int k = 1; k <= g.terms; ++k) {
            const R a = g.coeff[k - 1];
            prof_t += a * p_cur;
            sum_dp += a * dp_cur;
            sum_ddp += a * ddp_cur;
            const R p_next = (R(2 * k + 1) * xi * p_cur - R(k) * p_prev) / R(k + 1);
            const R dp_next = dp_prev + R(2 * k + 1) * p_cur;
            const R ddp_next = ddp_prev + R(2 * k + 1) * dp_cur;
            p_prev = p_cur;
            p_cur = p_next;
            dp_prev = dp_cur;
            dp_cur = dp_next;
            ddp_prev = ddp_cur;
            ddp_cur = ddp_next;
        }
    }
    const R dt_du = R(2) * sum_dp;
    const R d2t_du2 = R(4) * sum_ddp;
    const R dn = g.n_glass - n_before;
    const R radial = dn * dt_du * R(2) * inv_r2;
    const R dx = (n_before * d.x + radial * x) / n_after;
    const R dy = (n_before * d.y + radial * y) / n_after;
    const R radicand = R(1) - dx * dx - dy * dy;
    const bool tir = (branch & kBranchSoftTIR) != 0;

    // ---- Rückwärts ----
    const Vec3<R> qb = adj.p; // Schnittpunkt (x, y, 0): z ist konstant
    const Vec3<R> d1b = adj.d;
    RayAdjoint<R> in{detail::vec_zero<R>(), detail::vec_zero<R>()};
    R dxb;
    R dyb;
    R radb;
    if (tir) {
        // d′ = (d′_x, d′_y, 0)/m, m = √(d′_x² + d′_y²)
        const R inv_m = R(1) / msqrt(dx * dx + dy * dy);
        const R ex = dx * inv_m;
        const R ey = dy * inv_m;
        const R proj = ex * d1b.x + ey * d1b.y;
        dxb = (d1b.x - proj * ex) * inv_m;
        dyb = (d1b.y - proj * ey) * inv_m;
        radb = -seed.penalty; // Strafe −Radikand (Radikand selbst geht nicht in d′ ein)
    } else {
        dxb = d1b.x;
        dyb = d1b.y;
        radb = d1b.z / (R(2) * msqrt(radicand));
    }
    // Radikand = 1 − d′_x² − d′_y²
    dxb += -R(2) * dx * radb;
    dyb += -R(2) * dy * radb;
    // d′_x = (n d_x + radial·x)/n′
    in.d.x += dxb * n_before / n_after;
    in.d.y += dyb * n_before / n_after;
    out.n_before += (dxb * d.x + dyb * d.y) / n_after;
    out.n_after += -(dxb * dx + dyb * dy) / n_after;
    const R gxb = dxb / n_after;
    const R gyb = dyb / n_after;
    const R radialb = gxb * x + gyb * y;
    R xb = qb.x + gxb * radial;
    R yb = qb.y + gyb * radial;
    // OPL += n_vor s₀ + Φ, Φ = Δn t(u)
    const R s0b = seed.opl * n_before;
    out.n_before += seed.opl * tl.s0;
    const R phaseb = seed.opl;
    // radial = Δn (dt/du) 2/r_g²
    const R dnb = phaseb * prof_t + radialb * dt_du * R(2) * inv_r2;
    const R tb = phaseb * dn;
    const R dtdub = radialb * dn * R(2) * inv_r2;
    out.n_before += -dnb; // Δn = n_G − n_vor
    // Profil nach u und nach den Koeffizienten
    const R ub = tb * dt_du + dtdub * d2t_du2;
    {
        R p_prev = R(1);
        R p_cur = xi;
        R dp_prev = R(0);
        R dp_cur = R(1);
        for (int k = 1; k <= g.terms; ++k) {
            coeff_sink(k - 1, tb * p_cur + dtdub * R(2) * dp_cur);
            const R p_next = (R(2 * k + 1) * xi * p_cur - R(k) * p_prev) / R(k + 1);
            const R dp_next = dp_prev + R(2 * k + 1) * p_cur;
            p_prev = p_cur;
            p_cur = p_next;
            dp_prev = dp_cur;
            dp_cur = dp_next;
        }
    }
    // u = (x² + y²)/r_g²
    xb += ub * R(2) * x * inv_r2;
    yb += ub * R(2) * y * inv_r2;
    detail::transfer_backward(p, d, tl, xb, yb, s0b, in, out.thickness);
    adj = in;
}

// Rückwärts durch den Schritt auf die Referenzsphäre (reference_sphere_step) um M = `center` im
// Medium n_B: q = p − M, b = −q·d, g = p·(p − 2M), s_ref = g/(b + σ√(b² − g)), OPL += n_B s_ref;
// Zweig kBranchSoftMissed: s_ref = b mit Strafe −(b² − g)/|M|². s_ref ist die Nullstelle von
// s² − 2 b s + g, also ds = σ (dg − 2 s db)/(2√(b² − g)).
// Eingang: Zustand (p, d) nach der letzten Fläche. Der Zustand auf der Sphäre geht nicht weiter
// ein: `adj` erhält die Adjungierte des Eingangszustands; center_adj und n_image_adj werden
// addiert.
template <class R>
MKO_HD inline void reference_step_backward(const Vec3<R>& p, const Vec3<R>& d,
                                           const Vec3<R>& center, R n_image, unsigned char branch,
                                           AdjointSeed<R> seed, RayAdjoint<R>& adj,
                                           Vec3<R>& center_adj, R& n_image_adj) {
    const Vec3<R> q = p - center;
    const R b = -dot(q, d);
    const R g = dot(p, p - R(2) * center);
    const R disc = b * b - g;
    const bool missed = (branch & kBranchSoftMissed) != 0;
    const R sigma = center.z >= R(0) ? R(1) : R(-1);
    R s;
    R root = R(0);
    if (missed) {
        s = b;
    } else {
        root = msqrt(disc);
        s = g / (b + sigma * root);
    }
    const R sb = seed.opl * n_image;
    n_image_adj += seed.opl * s;
    R bb;
    R gb;
    if (missed) {
        bb = sb;
        gb = R(0);
        // Strafe −disc/|M|²
        const R radius_sq = dot(center, center);
        const R discb = -seed.penalty / radius_sq;
        bb += discb * R(2) * b;
        gb += -discb;
        center_adj += (seed.penalty * R(2) * disc / (radius_sq * radius_sq)) * center;
    } else {
        gb = sb * sigma / (R(2) * root);
        bb = -sb * sigma * s / root;
    }
    RayAdjoint<R> in{detail::vec_zero<R>(), detail::vec_zero<R>()};
    // b = −(p − M)·d
    in.p += (-bb) * d;
    in.d += (-bb) * q;
    center_adj += bb * d;
    // g = p·p − 2 p·M
    in.p += (R(2) * gb) * (p - center);
    center_adj += (R(-2) * gb) * p;
    adj = in;
}

// Rückwärts durch den Startstrahl der Punktquelle (point_source_ray): d₀ hängt über
// sin_max = NA/n₀ von der Brechzahl n₀ = index[0] des Startmediums ab,
// ∂d₀/∂sin_max = (d_x, d_y, −(d_x² + d_y²)/d_z)/sin_max, also
// ∂Φ/∂n₀ = −d̄·(d_x, d_y, −(d_x² + d_y²)/d_z)/n₀. Der Quellpunkt ist fest, OPL₀ = 0.
// Die kollimierte Quelle hängt von keinem Systemparameter außer Pupillenradius und Feldwinkel ab
// (beide keine Variablen).
template <class R>
MKO_HD inline R point_source_backward(const Vec3<R>& d0, R n0, const RayAdjoint<R>& adj) {
    const Vec3<R> ds{d0.x, d0.y, -(d0.x * d0.x + d0.y * d0.y) / d0.z};
    return -dot(adj.d, ds) / n0;
}

// ---- Rückwärtsdurchlauf eines Strahls ----------------------------------------------------------
//
// Senke für die Adjungierten der Systemparameter (Schnittstelle, jeder Typ mit):
//
//     MKO_HD void curvature(int s, R value);   // curvature[s] (nur Flächen vom Typ Sphere)
//     MKO_HD void thickness(int s, R value);   // thickness[s] (s = S − 1: Arbeitsabstand)
//     MKO_HD void index(int s, R value);       // index[s][l] (Brechzahl nach Fläche s)
//     MKO_HD void ghost(int g, int k, R value); // ghost_coeff[g][k] (a_{k+1} der Platte g)
//
// Jeder Parameter wird pro Spur höchstens einmal gemeldet (die Brechzahl nach Fläche s, die an
// Fläche s und s + 1 eingeht, wird dafür über einen Übertrag zusammengefasst). Nicht gemeldete
// Parameter haben die Ableitung 0 (z. B. Flächen hinter einer hart versagenden Fläche).
// Das Feldziel h_f (M_y) ist keine Variable und wird nicht gemeldet.

// Rückwärtsdurchlauf einer Spur für einen aufgezeichneten Strahl (AdjointRecord oder
// CheckpointRecord). `reached_image`: Der Schritt auf die Referenzsphäre ist gelaufen (Status
// in_piston); dann ist rec.count = S und der Zweig rec.image_branch. `center` = M_f,
// `point_source`: Startstrahl der Punktquelle (sonst kollimiert). Die Tabelle ist dieselbe wie im
// Vorwärtsdurchlauf (Wellenlängenindex `wavelength_index`).
template <class R, class SurfaceTable, class Record, class Sink>
MKO_HD inline void backward_ray(const SurfaceTable& sys, int wavelength_index, const Record& rec,
                                bool reached_image, const Vec3<R>& center, bool point_source,
                                AdjointSeed<R> seed, Sink& sink) {
    const int S = sys.surfaces;
    const int L = sys.wavelengths;
    const int count = rec.count;
    RayAdjoint<R> adj{detail::vec_zero<R>(), detail::vec_zero<R>()};
    R n_carry = R(0); // Adjungierte von index[j] aus dem Schritt j + 1 (bzw. der Referenzsphäre)
    int ghost = 0;    // Geisterplatten unter den Flächen 1 … count − 1
    for (int j = 1; j < count; ++j) {
        if (sys.type[j] == SurfaceKind::GhostPlate) {
            ++ghost;
        }
    }
    unsigned char next_branch = kBranchRegular; // Zweig der Fläche s + 1
    Vec3<R> d0 = detail::vec_zero<R>();
    for_each_state_reverse(
        sys, wavelength_index, rec,
        [&](int s, const Vec3<R>& p, const Vec3<R>& d, unsigned char branch) {
            if (s == count - 1) {
                if (reached_image && count == S) {
                    Vec3<R> center_adj = detail::vec_zero<R>();
                    R n_image_adj = R(0);
                    reference_step_backward(p, d, center,
                                            R(sys.index[(S - 1) * L + wavelength_index]),
                                            rec.image_branch, seed, adj, center_adj, n_image_adj);
                    sink.thickness(S - 1, center_adj.z); // M_z = Arbeitsabstand
                    n_carry = n_image_adj;
                }
            } else {
                const int j = s + 1; // Fläche, deren Eingang Zustand s ist
                const SurfaceKind kind = sys.type[j];
                const R t = R(sys.thickness[s]);
                const R n_before = R(sys.index[s * L + wavelength_index]);
                const R n_after = R(sys.index[j * L + wavelength_index]);
                SurfaceAdjoint<R> par{R(0), R(0), R(0), R(0)};
                if (kind == SurfaceKind::GhostPlate) {
                    --ghost;
                    const int g = ghost;
                    GhostPlateParams<R> plate{};
                    plate.coeff = sys.ghost_coeff + g * sys.ghost_terms;
                    plate.terms = sys.ghost_terms;
                    plate.norm_radius = R(sys.ghost_norm_radius[g]);
                    plate.n_glass = R(sys.ghost_index[g * L + wavelength_index]);
                    ghost_step_backward(p, d, t, plate, n_before, n_after, next_branch, seed, adj,
                                        par, [&](int k, R value) { sink.ghost(g, k, value); });
                } else {
                    const R c = surface_curvature(kind, R(sys.curvature[j]));
                    refracting_step_backward(p, d, t, c, n_before, n_after, next_branch, seed, adj,
                                             par);
                    if (kind == SurfaceKind::Sphere) {
                        sink.curvature(j, par.curvature);
                    }
                }
                sink.thickness(s, par.thickness);
                sink.index(j, par.n_after + n_carry);
                n_carry = par.n_before;
            }
            next_branch = branch;
            if (s == 0) {
                d0 = d;
            }
        });
    if (point_source && count > 0) {
        n_carry += point_source_backward(d0, R(sys.index[wavelength_index]), adj);
    }
    sink.index(0, n_carry);
}

} // namespace mko
