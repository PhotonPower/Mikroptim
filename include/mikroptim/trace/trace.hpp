#pragma once

// Sequenzieller Trace eines Strahls durch ein System, gemeinsam für CPU-Tracer und
// CUDA-Kernel (ADR-0005). Header-only, ohne Ausnahmen, Allokation und std-Container.
// Konventionen: docs/numerik.md, Abschnitte 3–5; API: docs/architektur.md, Abschnitt
// "Sequenzieller Trace".

#include "mikroptim/config.hpp"
#include "mikroptim/math/scalar.hpp"
#include "mikroptim/math/vec3.hpp"
#include "mikroptim/trace/ghost_plate.hpp"
#include "mikroptim/trace/ray.hpp"
#include "mikroptim/trace/surface_extrema.hpp"
#include "mikroptim/trace/surface_ops.hpp"
#include "mikroptim/trace/surface_table.hpp"

namespace mko {

// Optionale Aufzeichnung der Zwischenzustände eines Strahls (für den adjungierten Modus in M5
// und für Strahlhöhen und Einfallswinkel pro Fläche in M4-T3).
//
// Eintrag s (0 <= s < count) ist der Zustand nach Fläche s: Schnittpunkt p[s] in lokalen
// Koordinaten von Fläche s, Richtung d[s] nach der Brechung und OPL bis dorthin. Eintrag 0 ist
// der Startzustand. Die Richtung vor der Brechung an Fläche s ist d[s − 1]. Bei Versagen an
// Fläche s ist count = s (die Einträge davor bleiben gültig).
template <class R>
struct TraceRecord {
    int count;
    Vec3<R> p[kMaxSurfaces];
    Vec3<R> d[kMaxSurfaces];
    R opl[kMaxSurfaces];
};

// Startstrahl der kollimierten Quelle (docs/numerik.md, Abschnitt 4): Pupillenradius a
// (Halbdurchmesser der Blende), Feldwinkel theta [rad] in der y-z-Ebene, Pupillenkoordinaten
// (rho, phi) mit phi gemessen von +y (wie PupilSample). Lokale Koordinaten der Blende;
// OPL₀ = p₀·d₀ (Start auf der ebenen Wellenfront durch den Ursprung der Blendenebene).
template <class R>
MKO_HD inline Ray<R> collimated_ray(R a, R theta, R rho, R phi) {
    const R r = a * rho;
    Ray<R> ray;
    ray.p = Vec3<R>{r * msin(phi), r * mcos(phi), R(0)};
    ray.d = Vec3<R>{R(0), msin(theta), mcos(theta)};
    ray.opl = dot(ray.p, ray.d);
    return ray;
}

// Startstrahl einer Punktquelle (docs/numerik.md, Abschnitt 4) im Punkt q (lokale Koordinaten
// von Fläche 0). Die Richtungen sind gleichmäßig im Raum der Richtungskosinus verteilt:
// (d_x, d_y) = sin_max · rho · (sin phi, cos phi), d_z = √(1 − sin_max² rho²) > 0, mit
// sin_max = NA / n_Quelle (Sinus des halben Öffnungswinkels). OPL₀ = 0 im Quellpunkt.
template <class R>
MKO_HD inline Ray<R> point_source_ray(Vec3<R> q, R sin_max, R rho, R phi) {
    const R s = sin_max * rho;
    Ray<R> ray;
    ray.p = q;
    ray.d = Vec3<R>{s * msin(phi), s * mcos(phi), msqrt((R(1) - s) * (R(1) + s))};
    ray.opl = R(0);
    return ray;
}

// Ein Schritt des sequenziellen Trace (docs/numerik.md, Abschnitt 5): Der Strahl steht in
// lokalen Koordinaten der vorigen Fläche, deren Scheitel den Abstand t zum Scheitel der
// nächsten Fläche hat, im Medium n_before. Transfer in die Tangentialebene, Schnitt,
// OPL += n_before · (s₀ + s), Brechung mit μ = n_before / n_after. Danach steht der Strahl im
// Schnittpunkt, in lokalen Koordinaten der neuen Fläche.
//
// Geisterplatte (kind = GhostPlate, Daten in `ghost`, docs/numerik.md, Abschnitt 6): Transfer
// in die Ebene z = 0 (c wird ignoriert), OPL += n_before · s₀, dann ghost_plate_step (OPL += Φ,
// Ablenkung um ∇Φ). Ohne Daten (ghost = nullptr) ist der Typ nicht unterstützt.
//
// Status: Backward, wenn d_z <= 0 (oder NaN) vor dem Transfer; Missed, TIR aus den
// Flächenoperationen (an der Geisterplatte TIR bei negativem Radikanden); Unsupported für
// unbekannte Flächentypen. Bei Versagen ist der Strahlzustand teilweise fortgeschrieben und
// nicht weiter zu verwenden.
//
// Weiches Versagen (`soft` gesetzt, numerik.md §11): Fläche verfehlt (D < 0), Totalreflexion
// und negativer Radikand der Geisterplatte addieren ihre Strafgröße in `soft` und werden
// fortgesetzt (intersect_sphere, refract_sphere, phase_surface_step); die Rückgabe ist dann Ok.
// Backward, Unsupported und der Fall D >= 0, B + √D <= 0 bleiben hart.
template <class R>
MKO_HD inline TraceStatus
propagate_surface(Ray<R>& ray, R t, SurfaceKind kind, R c, R n_before, R n_after,
                  const typename detail::NonDeduced<GhostPlateParams<R>>::type* ghost = nullptr,
                  typename detail::NonDeduced<SoftFailure<R>>::type* soft = nullptr) {
    R curv;
    switch (kind) {
    case SurfaceKind::Stop:
    case SurfaceKind::Plane:
        curv = R(0);
        break;
    case SurfaceKind::Sphere:
        curv = c;
        break;
    case SurfaceKind::GhostPlate:
        if (ghost == nullptr) {
            return TraceStatus::Unsupported;
        }
        curv = R(0);
        break;
    default:
        return TraceStatus::Unsupported;
    }
    if (!(ray.d.z > R(0))) {
        return TraceStatus::Backward;
    }
    const R s0 = transfer_to_tangent_plane(ray, t);
    if (kind == SurfaceKind::GhostPlate) {
        ray.opl += n_before * s0; // Schnitt mit der Ebene: s = 0
        return ghost_plate_step(ray, *ghost, n_before, n_after, soft);
    }
    R s = R(0);
    const TraceStatus hit = intersect_sphere(ray, curv, s, soft);
    if (hit != TraceStatus::Ok && hit != TraceStatus::SoftMissed) {
        return hit;
    }
    ray.opl += n_before * (s0 + s);
    return refract_sphere(ray, curv, n_before / n_after, soft, hit == TraceStatus::SoftMissed);
}

// Sequenzieller Trace durch die Flächen 1 … S−1 der Tabelle `sys` bei Wellenlänge
// `wavelength_index` (Anforderungen an den Tabellentyp: SurfaceTableView). Geisterplatten
// werden in Tracerichtung durchgezählt; die g-te Fläche vom Typ GhostPlate erhält die Daten
// der Geisterplatte g der Tabelle.
//
// Start: `ray` in lokalen Koordinaten von Fläche 0, im Medium nach Fläche 0
// (index[0·L + l]); an Fläche 0 selbst wird nicht gebrochen. Bei Systemen aus Dateien ist
// Fläche 0 die Blende in Luft (ADR-0006).
// Ende (Status Ok): `ray` im Schnittpunkt mit der letzten Fläche S−1, in deren lokalen
// Koordinaten, Richtung nach der Brechung, OPL bis zu diesem Schnittpunkt. Den Rest bis zur
// Bildebene erledigt transfer_to_image_plane, die Referenzsphäre (M1-T7) setzt hier an.
//
// Versagen: Rückgabe des Status der ersten versagenden Fläche, *failed_surface = deren Nummer
// (sonst −1). Die Funktion kehrt dann sofort zurück und wirft nie; ein Kernel führt den Strahl
// über die Statusmaske maskiert weiter. `record` und `failed_surface` dürfen nullptr sein.
// `observer` (optional, Schnittstelle in surface_extrema.hpp) sieht den Startzustand (s = 0)
// und jede erfolgreich passierte Fläche; ohne Beobachter entfällt der Aufruf.
// Vorbedingung: 2 <= S <= kMaxSurfaces (prüft der Host beim Aufbau der Tabelle).
//
// Weiches Versagen (`soft` gesetzt, Start {0, Ok}; numerik.md §11): Verfehlte Flächen,
// Totalreflexion und negative Radikanden an Geisterplatten werden fortgesetzt und in `soft`
// bestraft; Beobachter und Aufzeichnung sehen die fortgesetzten Flächen wie passierte. Endet
// der Strahl ohne hartes Versagen, ist die Rückgabe soft->first (Ok ohne weiches Ereignis,
// sonst SoftMissed oder SoftTIR) und *failed_surface die Fläche des ersten weichen Ereignisses
// (−1 ohne). Hartes Versagen (Backward, Unsupported, Missed bei D >= 0) kehrt wie oben sofort
// zurück; die bis dahin gesammelte Strafe bleibt in `soft`.
template <class R, class SurfaceTable, class Observer = NoSurfaceObserver>
MKO_HD inline TraceStatus
trace_ray(const SurfaceTable& sys, int wavelength_index, Ray<R>& ray,
          typename detail::NonDeduced<TraceRecord<R>>::type* record, int* failed_surface = nullptr,
          Observer* observer = nullptr,
          typename detail::NonDeduced<SoftFailure<R>>::type* soft = nullptr) {
    const int L = sys.wavelengths;
    const int S = sys.surfaces;
    int soft_surface = -1; // Fläche des ersten weichen Ereignisses
    if (record != nullptr) {
        record->count = 1;
        record->p[0] = ray.p;
        record->d[0] = ray.d;
        record->opl[0] = ray.opl;
    }
    if (observer != nullptr) {
        observer->surface(0, R(0), ray.d, ray);
    }
    R n_before = R(sys.index[wavelength_index]);
    int ghost = 0; // Nummer der nächsten Geisterplatte (g-te Fläche vom Typ GhostPlate)
    for (int s = 1; s < S; ++s) {
        const R n_after = R(sys.index[s * L + wavelength_index]);
        const SurfaceKind kind = sys.type[s];
        GhostPlateParams<R> plate{};
        const GhostPlateParams<R>* plate_ptr = nullptr;
        if (kind == SurfaceKind::GhostPlate) {
            if (ghost < sys.ghost_plates) {
                plate.coeff = sys.ghost_coeff + ghost * sys.ghost_terms;
                plate.terms = sys.ghost_terms;
                plate.norm_radius = R(sys.ghost_norm_radius[ghost]);
                plate.n_glass = R(sys.ghost_index[ghost * L + wavelength_index]);
                plate_ptr = &plate;
            }
            ++ghost;
        }
        const Vec3<R> d_before = ray.d;
        const TraceStatus st =
            propagate_surface(ray, R(sys.thickness[s - 1]), kind, R(sys.curvature[s]), n_before,
                              n_after, plate_ptr, soft);
        if (st != TraceStatus::Ok) {
            if (failed_surface != nullptr) {
                *failed_surface = s;
            }
            return st;
        }
        if (soft != nullptr && soft_surface < 0 && soft->first != TraceStatus::Ok) {
            soft_surface = s;
        }
        if (observer != nullptr) {
            observer->surface(s, surface_curvature(kind, R(sys.curvature[s])), d_before, ray);
        }
        if (record != nullptr && s < kMaxSurfaces) {
            record->count = s + 1;
            record->p[s] = ray.p;
            record->d[s] = ray.d;
            record->opl[s] = ray.opl;
        }
        n_before = n_after;
    }
    if (failed_surface != nullptr) {
        *failed_surface = soft_surface;
    }
    return soft != nullptr ? soft->first : TraceStatus::Ok;
}

// Transfer vom Schnittpunkt der letzten Fläche (Ende von trace_ray) in die Bildebene im
// Abstand thickness[S−1], Medium index[(S−1)·L + l]: OPL += n · s₀. Danach steht der Strahl
// in der Bildebene (z = 0) in deren lokalen Koordinaten (Ursprung auf der Achse).
// Status Backward, wenn d_z <= 0.
template <class R, class SurfaceTable>
MKO_HD inline TraceStatus transfer_to_image_plane(const SurfaceTable& sys, int wavelength_index,
                                                  Ray<R>& ray) {
    if (!(ray.d.z > R(0))) {
        return TraceStatus::Backward;
    }
    const int last = sys.surfaces - 1;
    const R s0 = transfer_to_tangent_plane(ray, R(sys.thickness[last]));
    ray.opl += R(sys.index[last * sys.wavelengths + wavelength_index]) * s0;
    return TraceStatus::Ok;
}

} // namespace mko
