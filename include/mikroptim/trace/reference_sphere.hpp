#pragma once

// Referenzsphäre (docs/numerik.md, Abschnitt 7), gemeinsam für CPU-Tracer und CUDA-Kernel
// (ADR-0005). Header-only, ohne Ausnahmen, Allokation und std-Container.

#include "mikroptim/config.hpp"
#include "mikroptim/math/scalar.hpp"
#include "mikroptim/math/vec3.hpp"
#include "mikroptim/trace/ray.hpp"
#include "mikroptim/trace/surface_ops.hpp"
#include "mikroptim/trace/trace.hpp"

namespace mko {

// Schritt vom Schnittpunkt mit der letzten Fläche (Ende von trace_ray) auf die Referenzsphäre
// um den idealen Bildpunkt `center` = M (lokale Koordinaten der letzten Fläche) mit Radius
// R_ref = |M|, also durch den Scheitel der letzten Fläche. Medium Brechzahl n_image (n_B).
//
// Mit q = p − M, b = −q·d und g = |q|² − R_ref² = p·(p − 2M) ist der Strahlparameter
//
//     s_ref = g / (b + σ √(b² − g)),   σ = +1 für M_z >= 0, σ = −1 für M_z < 0,
//
// die Nullstelle von s² − 2 b s + g = 0 auf der dem Scheitel zugewandten Kappe der Sphäre
// (Herleitung: docs/numerik.md, Abschnitt 7). σ = +1 ist die dort angegebene Formel (reeller
// Bildpunkt hinter der letzten Fläche); σ = −1 deckt einen virtuellen Bildpunkt vor ihr ab.
// g wird als p·(p − 2M) gerechnet: algebraisch gleich |q|² − R_ref², aber ohne die Auslöschung
// zweier Quadrate der Größe R_ref².
//
// Danach steht der Strahl auf der Referenzsphäre: p += s_ref d, OPL += n_image · s_ref.
// s_ref darf negativ sein (Sphäre liegt vor dem Schnittpunkt). Ist `s_out` gesetzt, erhält es
// s_ref.
//
// Status Missed, wenn b² − g < 0 (Gerade verfehlt die Sphäre) oder σ (b + σ √(b² − g)) <= 0
// (die Gerade trifft die zugewandte Kappe nicht; auch bei NaN). Der Strahl bleibt dann
// unverändert.
template <class R>
MKO_HD inline TraceStatus reference_sphere_step(Ray<R>& ray, Vec3<R> center, R n_image,
                                                R* s_out = nullptr) {
    const Vec3<R> q = ray.p - center;
    const R b = -dot(q, ray.d);
    const R g = dot(ray.p, ray.p - R(2) * center);
    const R disc = b * b - g;
    if (disc < R(0)) {
        return TraceStatus::Missed;
    }
    const R root = msqrt(disc);
    const bool real_image = center.z >= R(0);
    const R denom = real_image ? b + root : b - root;
    if (!(real_image ? denom > R(0) : denom < R(0))) {
        return TraceStatus::Missed;
    }
    const R s = g / denom;
    ray.p += s * ray.d;
    ray.opl += n_image * s;
    if (s_out != nullptr) {
        *s_out = s;
    }
    return TraceStatus::Ok;
}

// Sequenzieller Trace (trace_ray) und anschließend der Schritt auf die Referenzsphäre um
// `center` im Medium nach der letzten Fläche (index[(S−1)·L + l]). Pro-Strahl-Teil der
// Wellenfrontauswertung für CPU und Kernel.
//
// Status: der von trace_ray (dann *failed_surface = versagende Fläche); Backward, wenn der
// Strahl nach der Brechung an der letzten Fläche rückwärts läuft (d_z <= 0, auch NaN;
// numerik.md §11), oder Missed von reference_sphere_step. In beiden letzten Fällen ist
// *failed_surface = S: der Schritt in den Bildraum zählt als Fläche nach der letzten. Ohne
// Versagen *failed_surface = −1. Ende (Ok): Strahl auf der Referenzsphäre in lokalen
// Koordinaten der letzten Fläche, OPL bis dorthin. `observer` wie bei trace_ray (die
// Referenzsphäre sieht er nicht).
template <class R, class SurfaceTable, class Observer = NoSurfaceObserver>
MKO_HD inline TraceStatus trace_to_reference_sphere(const SurfaceTable& sys, int wavelength_index,
                                                    Ray<R>& ray, Vec3<R> center,
                                                    int* failed_surface = nullptr,
                                                    Observer* observer = nullptr) {
    const TraceStatus st = trace_ray(sys, wavelength_index, ray, nullptr, failed_surface, observer);
    if (st != TraceStatus::Ok) {
        return st;
    }
    // trace_ray prüft d_z nur vor jedem Transfer; nach der letzten Brechung folgt keiner mehr.
    if (!(ray.d.z > R(0))) {
        if (failed_surface != nullptr) {
            *failed_surface = sys.surfaces;
        }
        return TraceStatus::Backward;
    }
    const int last = sys.surfaces - 1;
    const R n_image = R(sys.index[last * sys.wavelengths + wavelength_index]);
    const TraceStatus ref = reference_sphere_step(ray, center, n_image);
    if (ref != TraceStatus::Ok && failed_surface != nullptr) {
        *failed_surface = sys.surfaces;
    }
    return ref;
}

} // namespace mko
