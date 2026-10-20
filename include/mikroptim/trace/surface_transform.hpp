#pragma once

// Dezentrierung und Verkippung einzelner Flächen (M8-T3, ADR-0011), gemeinsam für CPU und
// CUDA-Kernel (ADR-0005): header-only, MKO_HD, templatisiert über R, ohne Ausnahmen, Allokation
// und std-Container. Konventionen und Formeln: docs/numerik.md, Abschnitt 2, "Dezentrierung und
// Verkippung".
//
// Jede Fläche s darf relativ zur nominellen (koaxialen) Lage ihres Scheitels verschoben
// (Δx, Δy) und um ihre lokale x- und y-Achse gekippt (α, β) sein; Drehpunkt ist der Scheitel.
// Die Drehung ist R = R_y(β) · R_x(α) (aktiv, rechtshändig): Ein Punkt q im gekippten System der
// Fläche liegt im nominellen System bei R q + Δ mit Δ = (Δx, Δy, 0). Der Strahl wird vor dem
// Transfer in die Tangentialebene ins gekippte System gebracht (p ← Rᵀ (p − Δ), d ← Rᵀ d) und
// nach Schnitt und Brechung zurück (p ← R p + Δ, d ← R d) ("dezentrieren und zurück", kein
// Koordinatenbruch für Folgeflächen). Die Abbildung ist eine Bewegung; Weglängen und damit die
// OPL bleiben unverändert.

#include "mikroptim/config.hpp"
#include "mikroptim/math/scalar.hpp"
#include "mikroptim/math/vec3.hpp"
#include "mikroptim/trace/ray.hpp"
#include "mikroptim/trace/surface_table.hpp"

#include <type_traits>

namespace mko {

// Transformation einer Fläche: Dezentrierung und die Sinus und Kosinus der Kippwinkel.
template <class R>
struct SurfaceTransform {
    R dx; // Δx [mm]
    R dy; // Δy [mm]
    R ca; // cos α (α: Drehung um die lokale x-Achse)
    R sa; // sin α
    R cb; // cos β (β: Drehung um die lokale y-Achse)
    R sb; // sin β
};

// Transformation aus Dezentrierung (dx, dy) [mm] und Kippwinkeln alpha, beta [rad].
template <class R>
MKO_HD inline SurfaceTransform<R> make_surface_transform(R dx, R dy, R alpha, R beta) {
    return SurfaceTransform<R>{dx, dy, mcos(alpha), msin(alpha), mcos(beta), msin(beta)};
}

// Vektor aus dem nominellen in das gekippte System: Rᵀ v = R_x(−α) · R_y(−β) · v.
template <class R>
MKO_HD inline Vec3<R> rotate_to_surface(const SurfaceTransform<R>& t, const Vec3<R>& v) {
    const R x = t.cb * v.x - t.sb * v.z; // R_y(−β)
    const R z = t.sb * v.x + t.cb * v.z;
    return Vec3<R>{x, t.ca * v.y + t.sa * z, t.ca * z - t.sa * v.y}; // R_x(−α)
}

// Vektor aus dem gekippten in das nominelle System: R v = R_y(β) · R_x(α) · v.
template <class R>
MKO_HD inline Vec3<R> rotate_from_surface(const SurfaceTransform<R>& t, const Vec3<R>& v) {
    const R y = t.ca * v.y - t.sa * v.z; // R_x(α)
    const R z = t.sa * v.y + t.ca * v.z;
    return Vec3<R>{t.cb * v.x + t.sb * z, y, t.cb * z - t.sb * v.x}; // R_y(β)
}

// Punkt aus dem nominellen in das gekippte System: Rᵀ (p − Δ).
template <class R>
MKO_HD inline Vec3<R> point_to_surface(const SurfaceTransform<R>& t, const Vec3<R>& p) {
    return rotate_to_surface(t, Vec3<R>{p.x - t.dx, p.y - t.dy, p.z});
}

// Punkt aus dem gekippten in das nominelle System: R q + Δ.
template <class R>
MKO_HD inline Vec3<R> point_from_surface(const SurfaceTransform<R>& t, const Vec3<R>& q) {
    const Vec3<R> p = rotate_from_surface(t, q);
    return Vec3<R>{p.x + t.dx, p.y + t.dy, p.z};
}

// Strahl (Ort und Richtung) in das gekippte System bzw. zurück; die OPL bleibt.
template <class R>
MKO_HD inline void ray_to_surface(Ray<R>& ray, const SurfaceTransform<R>& t) {
    ray.p = point_to_surface(t, ray.p);
    ray.d = rotate_to_surface(t, ray.d);
}
template <class R>
MKO_HD inline void ray_from_surface(Ray<R>& ray, const SurfaceTransform<R>& t) {
    ray.p = point_from_surface(t, ray.p);
    ray.d = rotate_from_surface(t, ray.d);
}

namespace detail {

// Tabellentyp, der zur Übersetzungszeit koaxial ist (Element `static constexpr bool kCoaxial`,
// z. B. CoaxialTableView).
template <class T, class = void>
struct IsCoaxialTable : std::false_type {};
template <class T>
struct IsCoaxialTable<T, std::void_t<decltype(T::kCoaxial)>> : std::bool_constant<T::kCoaxial> {};

// Laufzeitteil von surface_transform_of (unten).
template <class R, class SurfaceTable>
MKO_HD inline bool surface_transform_from_table(const SurfaceTable& sys, int s,
                                                SurfaceTransform<R>& t) {
    using T = std::remove_cv_t<std::remove_pointer_t<decltype(sys.decenter_tilt)>>;
    if (sys.decenter_tilt == nullptr) {
        return false;
    }
    const T* v = sys.decenter_tilt + s * kSurfaceTransformValues;
    // 0 heißt weder > 0 noch < 0 (nur diese Vergleiche verlangt der Tracer vom Zahlentyp)
    const auto zero = [](const T& x) { return !(x > T(0)) && !(x < T(0)); };
    if (zero(v[0]) && zero(v[1]) && zero(v[2]) && zero(v[3])) {
        return false;
    }
    t.dx = R(v[0]);
    t.dy = R(v[1]);
    if (sys.tilt_trig != nullptr) {
        const T* c = sys.tilt_trig + s * kSurfaceTransformValues;
        t.ca = R(c[0]);
        t.sa = R(c[1]);
        t.cb = R(c[2]);
        t.sb = R(c[3]);
    } else {
        const R alpha = R(v[2]);
        const R beta = R(v[3]);
        t.ca = mcos(alpha);
        t.sa = msin(alpha);
        t.cb = mcos(beta);
        t.sb = msin(beta);
    }
    return true;
}

} // namespace detail

// Transformation der Fläche s aus der Tabelle (Felder decenter_tilt und tilt_trig der
// SurfaceTableView, Ablage [s · kSurfaceTransformValues + k]). Rückgabe false, wenn die Tabelle
// koaxial ist (decenter_tilt = nullptr, oder zur Übersetzungszeit: CoaxialTableView) oder alle
// vier Werte der Fläche 0 sind: Dann rechnet der Tracer genau wie ohne Transformation
// (bitgleich). Sonst true und `t` mit den Werten im Rechentyp R; Sinus und Kosinus aus
// tilt_trig, wenn vorhanden, sonst mit msin/mcos aus den Winkeln.
template <class R, class SurfaceTable>
MKO_HD inline bool surface_transform_of(const SurfaceTable& sys, int s, SurfaceTransform<R>& t) {
    if constexpr (detail::IsCoaxialTable<SurfaceTable>::value) {
        (void)sys;
        (void)s;
        (void)t;
        return false;
    } else {
        return detail::surface_transform_from_table(sys, s, t);
    }
}

} // namespace mko
