#pragma once

// Flächenoperationen für sphärische Flächen, gemeinsam für CPU und GPU.
// Formeln und Konventionen: docs/numerik.md, Abschnitt "Flächenoperationen".

#include "mikroptim/config.hpp"
#include "mikroptim/math/vec3.hpp"
#include "mikroptim/trace/ray.hpp"

namespace mko {

// Status eines Strahls (Statusmaske, docs/numerik.md, Abschnitte 5 und 11). Die Werte sind
// stabil, weil sie als Zahl zwischen Host und Device ausgetauscht werden.
enum class TraceStatus : unsigned {
    Ok = 0,
    Missed = 1,      // Fläche wird nicht getroffen (Diskriminante < 0)
    TIR = 2,         // Totalreflexion; an einer Geisterplatte: Radikand 1 − |d'_⊥|² < 0
    Backward = 3,    // d_z <= 0 vor dem Transfer: Strahl läuft rückwärts (oder ist NaN)
    Unsupported = 4, // Flächentyp wird vom Tracer nicht unterstützt (z. B. Geisterplatte ohne
                     // Daten in der Tabelle)
};

// Kurzname für Meldungen und Tests.
MKO_HD constexpr const char* trace_status_name(TraceStatus s) {
    switch (s) {
    case TraceStatus::Ok:
        return "Ok";
    case TraceStatus::Missed:
        return "Missed";
    case TraceStatus::TIR:
        return "TIR";
    case TraceStatus::Backward:
        return "Backward";
    case TraceStatus::Unsupported:
        return "Unsupported";
    }
    return "?";
}

// Verschiebt den Strahl in die Tangentialebene (z = 0) der nächsten Fläche,
// deren Scheitel im Abstand t (entlang z) vom Scheitel der aktuellen Fläche liegt.
// Gibt die geometrische Weglänge zurück.
template <class R>
MKO_HD inline R transfer_to_tangent_plane(Ray<R>& r, R t) {
    r.p.z -= t;
    const R s = -r.p.z / r.d.z;
    r.p += s * r.d;
    r.p.z = R(0);
    return s;
}

// Schnitt mit der Sphäre der Krümmung c (c = 0: Ebene), Strahl liegt in der Tangentialebene.
// Numerisch stabile Form nach Spencer und Murty / Welford.
template <class R>
MKO_HD inline TraceStatus intersect_sphere(Ray<R>& r, R c, R& s) {
    const R H = c * (r.p.x * r.p.x + r.p.y * r.p.y);
    const R B = r.d.z - c * (r.p.x * r.d.x + r.p.y * r.d.y);
    const R D = B * B - c * H;
    if (D < R(0)) {
        return TraceStatus::Missed;
    }
    const R denom = B + msqrt(D);
    if (!(denom > R(0))) {
        return TraceStatus::Missed;
    }
    s = H / denom;
    r.p += s * r.d;
    return TraceStatus::Ok;
}

// Brechung an der Sphäre der Krümmung c im aktuellen Schnittpunkt, mu = n / n'.
template <class R>
MKO_HD inline TraceStatus refract_sphere(Ray<R>& r, R c, R mu) {
    const Vec3<R> N{-c * r.p.x, -c * r.p.y, R(1) - c * r.p.z};
    const R cosI = dot(r.d, N);
    const R k = R(1) - mu * mu * (R(1) - cosI * cosI);
    if (k < R(0)) {
        return TraceStatus::TIR;
    }
    r.d = mu * r.d + (msqrt(k) - mu * cosI) * N;
    return TraceStatus::Ok;
}

// z-Koordinate, an der ein meridionaler Strahl (x = 0) die Achse schneidet.
template <class R>
MKO_HD inline R meridional_axis_crossing(const Ray<R>& r) {
    return r.p.z - r.p.y * r.d.z / r.d.y;
}

} // namespace mko
