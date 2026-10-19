#pragma once

// Flächenoperationen für sphärische Flächen, gemeinsam für CPU und GPU.
// Formeln und Konventionen: docs/numerik.md, Abschnitt "Flächenoperationen"; weiches Versagen:
// Abschnitt 11.

#include "mikroptim/config.hpp"
#include "mikroptim/math/vec3.hpp"
#include "mikroptim/trace/ray.hpp"

namespace mko {

namespace detail {
// Verhindert die Ableitung von R aus einem Argument (R folgt allein aus dem Strahl), damit
// optionale Zeiger (z. B. `record`, `soft`) als nullptr übergeben werden können.
template <class T>
struct NonDeduced {
    using type = T;
};
} // namespace detail

// Status eines Strahls (Statusmaske, docs/numerik.md, Abschnitte 5 und 11). Die Werte sind
// stabil, weil sie als Zahl zwischen Host und Device ausgetauscht werden.
enum class TraceStatus : unsigned {
    Ok = 0,
    Missed = 1,      // Fläche wird nicht getroffen (Diskriminante < 0)
    TIR = 2,         // Totalreflexion; an einer Geisterplatte: Radikand 1 − |d'_⊥|² < 0
    Backward = 3,    // d_z <= 0 vor dem Transfer: Strahl läuft rückwärts (oder ist NaN)
    Unsupported = 4, // Flächentyp wird vom Tracer nicht unterstützt (z. B. Geisterplatte ohne
                     // Daten in der Tabelle)
    // Weiches Versagen (M4-T2, numerik.md §11): Der Strahl ist über die Fortsetzung bis zur
    // Referenzsphäre verfolgt, seine OPL ist definiert und geht in den Kolben ein, dazu trägt er
    // ein Strafresiduum. Der Wert nennt das erste weiche Ereignis des Strahls.
    SoftMissed = 5, // Fläche oder Referenzsphäre verfehlt (Diskriminante < 0), fortgesetzt
    SoftTIR = 6,    // Totalreflexion oder Radikand der Geisterplatte < 0, fortgesetzt
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
    case TraceStatus::SoftMissed:
        return "SoftMissed";
    case TraceStatus::SoftTIR:
        return "SoftTIR";
    }
    return "?";
}

// Weich versagt: verfolgt, mit Strafresiduum (numerik.md §11).
MKO_HD constexpr bool is_soft_failure(TraceStatus s) {
    return s == TraceStatus::SoftMissed || s == TraceStatus::SoftTIR;
}

// Der Strahl geht in Kolben und Wellenfrontresiduen ein: Ok oder weich versagt. Alle übrigen
// Werte sind hartes Versagen (maskiert, Wellenfrontresiduum 0, im weichen Modus festes
// Strafresiduum).
MKO_HD constexpr bool in_piston(TraceStatus s) {
    return s == TraceStatus::Ok || is_soft_failure(s);
}

// Hart versagt: alles außer Ok und weichem Versagen.
MKO_HD constexpr bool is_hard_failure(TraceStatus s) {
    return !in_piston(s);
}

// Behandlung des Strahlversagens (numerik.md §11): Soft in den Optimierungsmodi RESIDUALS und
// MERIT (Standard), Hard wie in M1–M3 (versagte Strahlen maskiert, ohne Strafresiduum;
// Auswertung, CLI `trace`, Modus EVAL). Die Werte sind stabil (Kernelparameter).
enum class FailureMode : unsigned char {
    Soft = 0,
    Hard = 1,
};

// Name für Meldungen ("soft", "hard").
MKO_HD constexpr const char* failure_mode_name(FailureMode m) {
    return m == FailureMode::Hard ? "hard" : "soft";
}

// Akkumulator des weichen Versagens eines Strahls (numerik.md §11). Ein Strahl im weichen Modus
// erhält einen Zeiger darauf (nullptr: hartes Versagen wie bisher); die Flächenoperationen
// addieren die Strafgröße jedes Ereignisses (−D, −k, −Radikand, −disc/R_ref², alle
// dimensionslos) und setzen die Fortsetzung ein. `first` ist Ok oder das erste weiche
// Ereignis. Trivial kopierbar; Start mit {R(0), TraceStatus::Ok}.
template <class R>
struct SoftFailure {
    R penalty;
    TraceStatus first;

    MKO_HD void add(R amount, TraceStatus kind) {
        penalty += amount;
        if (first == TraceStatus::Ok) {
            first = kind;
        }
    }
};

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
//
// Weiches Versagen (`soft` gesetzt, numerik.md §11): Bei D = B² − cH < 0 wird −D in `soft`
// addiert und mit D := 0 in der Form s = (B − √D)/c = B/c fortgesetzt, dem Punkt der Geraden mit
// dem kleinsten Abstand zum Krümmungsmittelpunkt (Scheitel der Parabel H − 2Bs + cs²). Für
// D → 0 ist das der Berührpunkt H/B der Tangente, die Fortsetzung ist also stetig. Die
// algebraisch gleiche Form H/(B + √D) ergäbe mit D := 0 s = H/B, das für B → 0 unbeschränkt
// wächst; B/c bleibt endlich (|s| < √(x² + y²), weil D < 0 nur bei |B| < |c|·√(x² + y²)
// möglich ist). Rückgabe dann SoftMissed: Der Punkt liegt nicht auf der Sphäre, refract_sphere
// muss die Normale normieren. Der zweite Fall „Fläche verfehlt“ (D >= 0, aber B + √D <= 0, auch
// NaN) bleibt hart; für d_z > 0 ist er außer durch Rundung nicht erreichbar (beide Nullstellen
// lägen bei z <= 0, die Sphäre aber bei z >= 0 für c > 0 bzw. die Strahlen bei z > 0 für c < 0).
template <class R>
MKO_HD inline TraceStatus
intersect_sphere(Ray<R>& r, R c, R& s,
                 typename detail::NonDeduced<SoftFailure<R>>::type* soft = nullptr) {
    const R H = c * (r.p.x * r.p.x + r.p.y * r.p.y);
    const R B = r.d.z - c * (r.p.x * r.d.x + r.p.y * r.d.y);
    const R D = B * B - c * H;
    if (D < R(0)) {
        // D < 0 setzt c ≠ 0 voraus (cH = c²(x² + y²) > B² >= 0); die Prüfung von c hält die
        // Division auch für Übersetzer sichtbar frei von c = 0 (Ebene, konstant eingesetzt)
        if (soft == nullptr || (!(c > R(0)) && !(c < R(0)))) {
            return TraceStatus::Missed;
        }
        soft->add(-D, TraceStatus::SoftMissed);
        s = B / c;
        r.p += s * r.d;
        return TraceStatus::SoftMissed;
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
//
// Weiches Versagen (`soft` gesetzt): Bei k < 0 wird −k addiert und mit k := 0 fortgesetzt,
// d' = μ (d − cos I N), normiert auf |d'| = 1 (mit k := 0 allein wäre |d'|² = 1 − k > 1). Der
// Strahl läuft dann tangential zur Fläche (d'·N = 0); ist dabei d'_z <= 0, versagt er an der
// nächsten Prüfung hart (Backward). `off_surface`: Der Punkt liegt nicht auf der Sphäre
// (Fortsetzung von intersect_sphere); N = (−c x, −c y, 1 − c z) wird dann normiert
// (|N| = |c|·|p − Mittelpunkt| > 1, stetig zu |N| = 1 an der Grenze D = 0). Dort ist
// cos I = d·N = 0.
template <class R>
MKO_HD inline TraceStatus
refract_sphere(Ray<R>& r, R c, R mu,
               typename detail::NonDeduced<SoftFailure<R>>::type* soft = nullptr,
               bool off_surface = false) {
    Vec3<R> N{-c * r.p.x, -c * r.p.y, R(1) - c * r.p.z};
    if (off_surface) {
        N = (R(1) / norm(N)) * N;
    }
    const R cosI = dot(r.d, N);
    const R k = R(1) - mu * mu * (R(1) - cosI * cosI);
    if (k < R(0)) {
        if (soft == nullptr) {
            return TraceStatus::TIR;
        }
        soft->add(-k, TraceStatus::SoftTIR);
        const Vec3<R> t = r.d - cosI * N; // |t|² = 1 − cos² I > 1/μ² > 0
        r.d = (R(1) / norm(t)) * t;
        return TraceStatus::Ok;
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
