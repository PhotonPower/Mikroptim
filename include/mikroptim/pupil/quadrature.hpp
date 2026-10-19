#pragma once

// Gauß-Quadratur über die Kreispupille (Forbes), gemeinsam für CPU und GPU.
// Konventionen, Index-Abbildung und Exaktheitsbereich: docs/numerik.md, Abschnitt 8.
//
// Ablauf: Der Host erzeugt einmal eine PupilQuadrature (make_pupil_quadrature, wirft
// bei ungültiger Konfiguration) und übergibt sie per Wert an Kernel oder CPU-Tracer.
// Dort liefert sample(q, index) für 0 <= index < sample_count(q) die Stützstelle
// (rho, phi, weight) ohne Tabellenzugriff außerhalb der Struktur.

#include "mikroptim/config.hpp"
#include "mikroptim/math/scalar.hpp"
#include "mikroptim/pupil/gauss_legendre_table.hpp"

#include <stdexcept>
#include <type_traits>

namespace mko {

// Größte unterstützte Zahl radialer Ringe N.
inline constexpr int kMaxPupilRings = detail::kGaussLegendreMaxOrder;

enum class PupilMode : int {
    // Halbe Pupille φ ∈ (0, π), d. h. x > 0 (Symmetrie x → −x bei Feld entlang y):
    // φ_j = (j + ½)·π/M, j = 0 … M−1. Exakt für cos(mφ) mit 0 ≤ m ≤ 2M−1.
    Half = 0,
    // Volle Pupille φ ∈ (0, 2π): φ_j = (j + ½)·2π/M. Exakt für cos(mφ) und sin(mφ)
    // mit 0 ≤ m ≤ M−1. Die ersten M/2 Speichen einer vollen Regel mit gerader Speichenzahl M
    // stimmen mit der halben Regel mit M/2 Speichen überein.
    Full = 1,
    // Achsfall (Feld 0): ein meridionaler Fächer mit φ = 0 (+y-Richtung), M = 1.
    // Exakt nur für rotationssymmetrische Funktionen (m = 0).
    Axis = 2,
};

// Trivial kopierbarer Parametersatz der Quadratur; wird per Wert an Kernel übergeben.
// Index-Abbildung: index = i·spokes + j mit Ring i = 0 … rings−1 (u aufsteigend, innen
// nach außen) und Speiche j = 0 … spokes−1.
template <class R>
struct PupilQuadrature {
    int rings;           // N, 1 … kMaxPupilRings
    int spokes;          // M (im Achsfall 1)
    PupilMode mode;      // halbe Pupille, volle Pupille oder Achsfall
    R u[kMaxPupilRings]; // radiale Knoten u_i = ρ_i², aufsteigend; gültig für i < rings
    R w[kMaxPupilRings]; // radiale Gewichte W_i, Summe 1; gültig für i < rings
};

static_assert(std::is_trivially_copyable_v<PupilQuadrature<double>>);
static_assert(std::is_standard_layout_v<PupilQuadrature<double>>);

// Eine Stützstelle: normierter Pupillenradius ρ ∈ (0, 1), Azimut φ (gemessen von +y,
// docs/numerik.md, Abschnitt 4) und Gewicht W_i·V_j (Summe über alle Stützstellen 1).
template <class R>
struct PupilSample {
    R rho;
    R phi;
    R weight;
};

// Anzahl der Stützstellen N·M.
template <class R>
MKO_HD inline int sample_count(const PupilQuadrature<R>& q) {
    return q.rings * q.spokes;
}

// Stützstelle zum linearen Index mit Speichenzahl und Pupillenart als Argument (Knoten und
// Gewichte aus q): Kernel, die Stützstellen mehrerer Felder mit verschiedener Pupillenart im
// selben Block rechnen (Tupelpakete, M7-T4). Vorbedingung: 0 <= index < q.rings · spokes
// (wird nicht geprüft, damit der Aufruf im Kernel verzweigungsarm bleibt).
template <class R>
MKO_HD inline PupilSample<R> sample(const PupilQuadrature<R>& q, int index, int spokes,
                                    PupilMode mode) {
    const R pi = R(3.14159265358979323846264338327950288);
    const int i = index / spokes;
    const int j = index - i * spokes;
    PupilSample<R> s;
    s.rho = msqrt(q.u[i]);
    // V_j = 1/M für alle Modi; eine Division statt Multiplikation mit 1/M (eine Rundung).
    s.weight = q.w[i] / R(spokes);
    switch (mode) {
    case PupilMode::Half:
        s.phi = R(2 * j + 1) * pi / R(2 * spokes);
        break;
    case PupilMode::Full:
        s.phi = R(2 * j + 1) * pi / R(spokes);
        break;
    default: // PupilMode::Axis
        s.phi = R(0);
        break;
    }
    return s;
}

// Stützstelle zum linearen Index. Vorbedingung: 0 <= index < sample_count(q)
// (wird nicht geprüft, damit der Aufruf im Kernel verzweigungsarm bleibt).
template <class R>
MKO_HD inline PupilSample<R> sample(const PupilQuadrature<R>& q, int index) {
    return sample(q, index, q.spokes, q.mode);
}

// Gewicht der Stützstelle im Residuum: √(W_i·V_j) (docs/optimierung.md, Abschnitt Merit).
template <class R>
MKO_HD inline R residual_weight(const PupilSample<R>& s) {
    return msqrt(s.weight);
}

// Nur Host: erzeugt die Quadratur mit N = rings radialen Ringen und M = spokes Speichen.
// Knoten und Gewichte stammen aus der korrekt gerundeten Tabelle in
// gauss_legendre_table.hpp. Im Achsfall wird spokes geprüft, aber durch 1 ersetzt.
// Wirft std::invalid_argument bei rings ∉ [1, kMaxPupilRings] oder spokes < 1.
template <class R = double>
inline PupilQuadrature<R> make_pupil_quadrature(int rings, int spokes, PupilMode mode) {
    if (rings < 1 || rings > kMaxPupilRings) {
        throw std::invalid_argument("Pupillenquadratur: Zahl der Ringe muss in [1, 16] liegen");
    }
    if (spokes < 1) {
        throw std::invalid_argument("Pupillenquadratur: Zahl der Speichen muss >= 1 sein");
    }
    if (mode != PupilMode::Half && mode != PupilMode::Full && mode != PupilMode::Axis) {
        throw std::invalid_argument("Pupillenquadratur: unbekannter Modus");
    }
    PupilQuadrature<R> q{};
    q.rings = rings;
    q.spokes = (mode == PupilMode::Axis) ? 1 : spokes;
    q.mode = mode;
    const int offset = detail::gauss_legendre_offset(rings);
    for (int i = 0; i < rings; ++i) {
        q.u[i] = R(detail::kGaussLegendreU[offset + i]);
        q.w[i] = R(detail::kGaussLegendreW[offset + i]);
    }
    return q;
}

} // namespace mko
