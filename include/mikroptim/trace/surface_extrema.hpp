#pragma once

// Strahlhöhe und Einfallswinkel pro Fläche (Max-Reduktionen, M2-T3), gemeinsam für CPU und
// CUDA-Kernel (ADR-0005). Header-only, MKO_HD, templatisiert über R; ohne Ausnahmen,
// Allokation und std-Container. Definitionen: docs/architektur.md, Abschnitt "Kernel MERIT",
// "Max-Reduktionen".
//
// trace_ray ruft einen optionalen Beobachter nach jeder erfolgreich passierten Fläche auf
// (Template-Parameter, Standard NoSurfaceObserver ohne Wirkung und ohne Kosten). Ein
// Beobachter ist jeder Typ mit der Elementfunktion
//
//     MKO_HD void surface(int s, R c, const Vec3<R>& d_before, const Ray<R>& ray);
//
// s = Fläche (0 = Startzustand in der Ebene von Fläche 0), c = wirksame Krümmung der Fläche
// (surface_curvature; 0 für s = 0, an der nicht gebrochen wird), d_before = Richtung vor der
// Brechung, ray = Zustand nach der Brechung im Schnittpunkt (lokale Koordinaten von Fläche s).
// Versagt der Strahl später, hat der Beobachter die früheren Flächen schon gesehen: Wer nur
// gültige Strahlen zählen will, sammelt pro Strahl (RaySurfaceRecord) und übernimmt erst nach
// dem Status Ok (SurfaceExtrema::add_ray).

#include "mikroptim/config.hpp"
#include "mikroptim/math/vec3.hpp"
#include "mikroptim/trace/ray.hpp"
#include "mikroptim/trace/surface_table.hpp"

namespace mko {

// Wirksame Krümmung einer Fläche im Tracer: Stop, Plane und GhostPlate (Ebene mit
// Phasenfunktion, M4-T5) rechnen mit c = 0, Sphere mit curvature[s] (gleiche Regel wie
// propagate_surface). An einer Geisterplatte ist cos I also d_z vor dem Phasenschritt.
template <class R>
MKO_HD inline R surface_curvature(SurfaceKind kind, R c) {
    return kind == SurfaceKind::Sphere ? c : R(0);
}

// Beobachter ohne Wirkung (Standard von trace_ray; der Aufruf entfällt zur Übersetzungszeit).
struct NoSurfaceObserver {
    template <class R>
    MKO_HD void surface(int /*s*/, R /*c*/, const Vec3<R>& /*d_before*/, const Ray<R>& /*ray*/) {}
};

// Neutrale Elemente der Reduktionen: Höhe² 0 (Höhen sind >= 0) und cos I = 1 (I = 0). Eine
// Variante ohne gültigen Strahl behält sie.
template <class R>
MKO_HD inline R neutral_height_sq() {
    return R(0);
}
template <class R>
MKO_HD inline R neutral_cos_incidence() {
    return R(1);
}

// Größerer bzw. kleinerer Wert (NaN im zweiten Argument wird ignoriert). Exakt und
// unabhängig von der Reihenfolge; CPU und Kernel verwenden dieselben Funktionen.
template <class R>
MKO_HD inline R extremum_max(R acc, R x) {
    return x > acc ? x : acc;
}
template <class R>
MKO_HD inline R extremum_min(R acc, R x) {
    return x < acc ? x : acc;
}

// Beobachter, der für einen Strahl pro Fläche s das Quadrat der Strahlhöhe x² + y² im
// Schnittpunkt und cos I = d_before · N mit der Flächennormale N = (−c x, −c y, 1 − c z)
// (Einheitsvektor, docs/numerik.md, Abschnitt 5) ablegt. An Fläche 0 (keine Brechung) ist
// N = (0, 0, 1), also cos I = d_z des Startstrahls. Einträge nach einer versagenden Fläche
// bleiben unbestimmt. Im Kernel Local Memory (2 · kMaxSurfaces · sizeof(R) Byte).
template <class R>
struct RaySurfaceRecord {
    R height_sq[kMaxSurfaces];
    R cos_incidence[kMaxSurfaces];

    MKO_HD void surface(int s, R c, const Vec3<R>& d_before, const Ray<R>& ray) {
        const Vec3<R>& p = ray.p;
        height_sq[s] = p.x * p.x + p.y * p.y;
        const Vec3<R> normal{-c * p.x, -c * p.y, R(1) - c * p.z};
        cos_incidence[s] = dot(d_before, normal);
    }
};

// Akkumulator der Max-Reduktionen über viele Strahlen: pro Fläche das größte Höhenquadrat und
// der kleinste cos I (= größter Einfallswinkel, I = arccos(cos I) fällt monoton). Die Wurzel
// der Höhe zieht erst der letzte Schritt (√ ist monoton und korrekt gerundet, also gleich dem
// Maximum der Wurzeln).
template <class R>
struct SurfaceExtrema {
    R max_height_sq[kMaxSurfaces];
    R min_cos_incidence[kMaxSurfaces];

    // Setzt die ersten `surfaces` Einträge auf die neutralen Elemente.
    MKO_HD void reset(int surfaces) {
        for (int s = 0; s < surfaces; ++s) {
            max_height_sq[s] = neutral_height_sq<R>();
            min_cos_incidence[s] = neutral_cos_incidence<R>();
        }
    }

    MKO_HD void add(int s, R height_sq, R cos_incidence) {
        max_height_sq[s] = extremum_max(max_height_sq[s], height_sq);
        min_cos_incidence[s] = extremum_min(min_cos_incidence[s], cos_incidence);
    }

    // Übernimmt einen vollständig gültigen Strahl (Status Ok) für die Flächen 0 … surfaces − 1.
    MKO_HD void add_ray(const RaySurfaceRecord<R>& ray, int surfaces) {
        for (int s = 0; s < surfaces; ++s) {
            add(s, ray.height_sq[s], ray.cos_incidence[s]);
        }
    }
};

} // namespace mko
