#pragma once

// Elementweise Schritte der vektoriellen PSF (M8-T5, Debye-Wolf/Richards-Wolf), gemeinsam für die
// CPU-Referenz (vector_psf_cpu) und die Kernel der VectorPsfEngine (ADR-0005). Header-only,
// MKO_HD, templatisiert über R; ohne Ausnahmen, Allokation und std-Container. Festlegungen:
// docs/numerik.md, Abschnitt "Vektorielle PSF (M8-T5)".

#include "mikroptim/config.hpp"
#include "mikroptim/eval/psf_ops.hpp"
#include "mikroptim/math/scalar.hpp"

namespace mko {

// Höchstens zwei Eingangsfelder (Jones-Vektoren) pro Rechnung: unpolarisiert ist die
// inkohärente Summe aus x und y.
inline constexpr int kMaxVectorPsfInputs = 2;

// Jones-Vektor J = (J_x, J_y) des Eingangsfeldes in der Eintrittspupille (Basis x̂_in, ŷ_in
// senkrecht zum Eingangsbündel) mit dem Gewicht w der inkohärenten Summe (40 Byte, trivial
// kopierbar).
struct VectorPsfJones {
    double re[2];
    double im[2];
    double weight;
};

// Parameter einer Pupille für die vektorielle PSF (trivial kopierbar, 88 Byte). Die
// Bildraumrichtung des Rasterpunkts (x, y) ist s_⊥ = (c_x + a_x x/n', c_y + a_y y/n'),
// s_z = √(1 − |s_⊥|²); a_x, a_y sind die vorzeichenbehafteten Steigungen n' ∂d/∂x, n' ∂d/∂y
// (negativ bei konvergenter Welle), c die Richtung des Hauptstrahls.
struct VectorPsfParams {
    double lambda_mm; // Vakuumwellenlänge [mm]
    double scale;     // 1 / I_0 (Intensität der perfekten Pupille im Fokus)
    double a_x;       // n' ∂d_x/∂x
    double a_y;       // n' ∂d_y/∂y
    double index;     // n' (Bildraum)
    double chief_x;   // Richtungskosinus des Hauptstrahls im Bildraum
    double chief_y;
    double input_y; // Richtung des Eingangsbündels u = (0, u_y, u_z) im Raum vor der Blende
    double input_z;
    double defocus; // Abstand z der Auswerteebene von der Bildebene [mm] (> 0: in +z)
    int sign_x;     // +1 konvergent, −1 divergent (gespiegelt), wie PsfPupilParams
    int sign_y;
};

namespace detail {

// out = e + v × e + q · v × (v × e)  (Rotation um v = u × s mit q = 1/(1 + u·s))
template <class R>
MKO_HD inline void vector_psf_rotate(R vx, R vy, R vz, R q, R ex, R ey, R ez, R out[3]) {
    const R wx = vy * ez - vz * ey;
    const R wy = vz * ex - vx * ez;
    const R wz = vx * ey - vy * ex;
    const R w2x = vy * wz - vz * wy;
    const R w2y = vz * wx - vx * wz;
    const R w2z = vx * wy - vy * wx;
    out[0] = ex + wx + q * w2x;
    out[1] = ey + wy + q * w2y;
    out[2] = ez + wz + q * w2z;
}

} // namespace detail

// Geometrie des Rasterpunkts (x, y) (normierte Pupillenkoordinaten): Bilder e_x = R x̂_in und
// e_y = R ŷ_in der Eingangsbasis unter der Rotation R, die die Richtung u des Eingangsbündels in
// die Bildraumrichtung s dreht (ideale aplanatische Linse: Komponenten in der Meridionalebene
// von u und s werden mitgedreht, die dazu senkrechten bleiben), Gewicht 1/√s_z (Aplanatfaktor
// √cos θ auf der Referenzsphäre mal Flächenelement dΩ = ds_x ds_y / s_z) und die optische
// Weglänge des Defokus n' z (s_z − 1) [mm]. x̂_in = (1, 0, 0), ŷ_in = u × x̂_in = (0, u_z, −u_y).
// false, wenn |s_⊥| >= 1 (keine ausbreitungsfähige Welle); dann sind die Ausgaben unbestimmt.
template <class R>
MKO_HD inline bool vector_psf_basis(const VectorPsfParams& p, R x, R y, R ex[3], R ey[3], R& weight,
                                    R& defocus_opd) {
    const R n = R(p.index);
    const R sx = R(p.chief_x) + R(p.a_x) * x / n;
    const R sy = R(p.chief_y) + R(p.a_y) * y / n;
    const R t2 = sx * sx + sy * sy;
    if (!(t2 < R(1))) {
        return false;
    }
    const R sz = msqrt(R(1) - t2);
    weight = R(1) / msqrt(sz);
    // s_z − 1 = −|s_⊥|² / (1 + s_z) ohne Auslöschung; für z = 0 exakt −0
    defocus_opd = -(n * R(p.defocus)) * t2 / (R(1) + sz);
    const R uy = R(p.input_y);
    const R uz = R(p.input_z);
    // v = u × s, c = u · s (u = (0, u_y, u_z))
    const R vx = uy * sz - uz * sy;
    const R vy = uz * sx;
    const R vz = -(uy * sx);
    const R q = R(1) / (R(1) + (uy * sy + uz * sz));
    detail::vector_psf_rotate(vx, vy, vz, q, R(1), R(0), R(0), ex);
    detail::vector_psf_rotate(vx, vy, vz, q, R(0), uz, -uy, ey);
    return true;
}

// Feldkomponenten G_c (c = x, y, z) des Rasterpunkts (x, y) für das Eingangsfeld J:
// G = (1/√s_z) · exp(2πi (OPD + n' z (s_z − 1))/λ) · (e_x J_x + e_y J_y). Null bei OPD = NaN
// (außerhalb der Pupille oder versagter Strahl) und bei |s_⊥| >= 1; dann false.
template <class R>
MKO_HD inline bool vector_psf_pupil_value(const VectorPsfParams& p, const VectorPsfJones& j, R x,
                                          R y, R opd, R re[3], R im[3]) {
    for (int c = 0; c < 3; ++c) {
        re[c] = R(0);
        im[c] = R(0);
    }
    if (!(opd == opd)) {
        return false;
    }
    R ex[3];
    R ey[3];
    R weight = R(0);
    R defocus_opd = R(0);
    if (!vector_psf_basis(p, x, y, ex, ey, weight, defocus_opd)) {
        return false;
    }
    const R phase = R(kPsfTwoPi) * ((opd + defocus_opd) / R(p.lambda_mm));
    const R cr = weight * mcos(phase);
    const R ci = weight * msin(phase);
    for (int c = 0; c < 3; ++c) {
        const R ar = ex[c] * R(j.re[0]) + ey[c] * R(j.re[1]);
        const R ai = ex[c] * R(j.im[0]) + ey[c] * R(j.im[1]);
        re[c] = cr * ar - ci * ai;
        im[c] = cr * ai + ci * ar;
    }
    return true;
}

} // namespace mko
