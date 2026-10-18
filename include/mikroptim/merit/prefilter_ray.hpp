#pragma once

// FP32-Vorfilterung (M7-T2): Pro-Strahl-Teil, gemeinsam für die CPU-Referenz
// (prefilter_stats_cpu) und den Kernel der Vorfilterung (ADR-0005). Header-only, MKO_HD,
// templatisiert über R; ohne Ausnahmen, Allokation und std-Container.
//
// VORFILTERUNG: Dieser Pfad ist der einzige, in dem R = float erlaubt ist (AGENTS.md §6.1,
// ADR-0003). Seine Ergebnisse dienen nur dem Aussortieren hoffnungsloser Varianten im
// Multistart und dürfen nie in Merit, Residuen, Jacobi-Matrix oder Optimierung einfließen.
// Begründung, Fehlerabschätzung und Entscheidungsregel: docs/numerik.md, Abschnitt 12
// ("FP32-Vorfilterung").
//
// Zusätzlich zu Status und OPL liefert jeder Strahl den Abstand seiner Prüfgrößen zur
// Versagensgrenze ("Grenzabstand", in Einheiten eines Richtungskosinus). Liegt er unter der
// Schwelle η der Vorfilterung, kann FP64 den Strahl anders entscheiden; er gilt dann als
// unsicher. Kleine Kosinus (streifende Strahlen) verstärken zugleich die Rundungsfehler aller
// folgenden Flächen; auch dagegen schützt die Schwelle.

#include "mikroptim/batch/system_batch_view.hpp"
#include "mikroptim/config.hpp"
#include "mikroptim/math/scalar.hpp"
#include "mikroptim/math/vec3.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/ghost_plate.hpp"
#include "mikroptim/trace/ray.hpp"
#include "mikroptim/trace/reference_sphere.hpp"
#include "mikroptim/trace/surface_extrema.hpp"
#include "mikroptim/trace/surface_ops.hpp"
#include "mikroptim/trace/surface_table.hpp"
#include "mikroptim/trace/trace.hpp"

#include <type_traits>

namespace mko {

// Kleinerer Wert für Grenzabstände; NaN (in x) ergibt 0, also "unsicher".
template <class R>
MKO_HD inline R prefilter_min_margin(R acc, R x) {
    if (x < acc) {
        return x;
    }
    return x >= acc ? acc : R(0);
}

// Betrag eines Versagens einer linearen Prüfgröße x (Versagen bei x <= 0): −x; NaN ergibt 0.
template <class R>
MKO_HD inline R prefilter_failure_linear(R x) {
    return -x > R(0) ? -x : R(0);
}

// Betrag eines Versagens einer quadratischen Prüfgröße q = cos² (Versagen bei q < 0, z. B.
// D = cos² I, k = cos² I', Radikand): √(−q), also dieselbe Einheit wie ein Kosinus; NaN
// ergibt 0.
template <class R>
MKO_HD inline R prefilter_failure_root(R q) {
    return -q > R(0) ? msqrt(-q) : R(0);
}

// Beobachter für trace_ray: kleinster Grenzabstand der erfolgreich passierten Flächen s >= 1.
// Die Prüfgrößen des Tracers (docs/numerik.md, Abschnitte 5 und 6) im Schnittpunkt, jeweils als
// Kosinus:
//   - d_z > 0 vor dem Transfer: d_before.z,
//   - Schnitt mit der Sphäre D = B² − cH >= 0: √D = cos I = d_before · N (für |d| = 1 exakt,
//     Herleitung in numerik.md, Abschnitt 12); an Ebenen ist das d_z,
//   - Brechung k = 1 − μ²(1 − cos² I) >= 0: √k = d' · N = cos I'; an einer Geisterplatte
//     (c = 0, N = e_z) ist d'_z = √(1 − |d'_⊥|²).
template <class R>
struct PrefilterMarginObserver {
    R margin;

    MKO_HD void surface(int s, R c, const Vec3<R>& d_before, const Ray<R>& ray) {
        if (s == 0) {
            return; // Startzustand: an Fläche 0 wird weder geschnitten noch gebrochen
        }
        const Vec3<R>& p = ray.p;
        const Vec3<R> normal{-c * p.x, -c * p.y, R(1) - c * p.z};
        margin = prefilter_min_margin(margin, d_before.z);
        margin = prefilter_min_margin(margin, dot(d_before, normal));
        margin = prefilter_min_margin(margin, dot(ray.d, normal));
    }
};

// Ergebnis eines Strahls der Vorfilterung.
template <class R>
struct PrefilterRay {
    TraceStatus status;
    R opl;    // OPL − opl_offset bis zur Referenzsphäre [mm], nur bei Ok gültig
    R margin; // Grenzabstand (>= 0, Einheit eines Kosinus): bei Ok der kleinste Abstand aller
              // Prüfgrößen zur Grenze, bei Versagen der Betrag der verletzten Prüfgröße; 0 bei
              // NaN
};

namespace detail {

// Betrag des Versagens an der Fläche `fs` (1 … S − 1) aus dem Strahlzustand, in dem trace_ray
// zurückgekehrt ist (docs/numerik.md, Abschnitt 5): Backward vor dem Transfer (Richtung
// unverändert), Missed in der Tangentialebene (vor dem Schnitt), TIR im Schnittpunkt (Richtung
// vor der Brechung) bzw. an der Geisterplatte in deren Ebene. Rechnet die Prüfgröße mit
// denselben Formeln wie surface_ops.hpp und ghost_plate.hpp nach.
template <class R, class SurfaceTable>
MKO_HD inline R prefilter_surface_failure(const SurfaceTable& sys, int l, int fs, TraceStatus st,
                                          const Ray<R>& ray) {
    const int L = sys.wavelengths;
    const SurfaceKind kind = sys.type[fs];
    switch (st) {
    case TraceStatus::Backward:
        return prefilter_failure_linear(ray.d.z);
    case TraceStatus::Missed: {
        // FP64 trifft nur bei D >= 0 und B + √D > 0; wegen cH = c²(x² + y²) >= 0 ist
        // √D <= |B|, also verlangt das B > 0. Sicher verfehlt, wenn √(−D) oder −B groß ist.
        const R c = surface_curvature(kind, R(sys.curvature[fs]));
        const R H = c * (ray.p.x * ray.p.x + ray.p.y * ray.p.y);
        const R B = ray.d.z - c * (ray.p.x * ray.d.x + ray.p.y * ray.d.y);
        const R D = B * B - c * H;
        const R a = prefilter_failure_root(D);
        const R b = prefilter_failure_linear(B);
        return a > b ? a : b;
    }
    case TraceStatus::TIR: {
        const R n_before = R(sys.index[(fs - 1) * L + l]);
        const R n_after = R(sys.index[fs * L + l]);
        if (kind == SurfaceKind::GhostPlate) {
            int g = 0; // Nummer der Geisterplatte (g-te Fläche vom Typ GhostPlate)
            for (int s = 1; s < fs; ++s) {
                g += sys.type[s] == SurfaceKind::GhostPlate ? 1 : 0;
            }
            // wie ghost_plate_step und phase_surface_step
            const R norm_radius = R(sys.ghost_norm_radius[g]);
            const R inv_r2 = R(1) / (norm_radius * norm_radius);
            const R u = (ray.p.x * ray.p.x + ray.p.y * ray.p.y) * inv_r2;
            const GhostProfile<R> prof =
                ghost_profile(sys.ghost_coeff + g * sys.ghost_terms, sys.ghost_terms, u);
            const R dn = R(sys.ghost_index[g * L + l]) - n_before;
            const R radial = dn * prof.dt_du * R(2) * inv_r2;
            const R dx = (n_before * ray.d.x + radial * ray.p.x) / n_after;
            const R dy = (n_before * ray.d.y + radial * ray.p.y) / n_after;
            return prefilter_failure_root(R(1) - dx * dx - dy * dy);
        }
        // wie refract_sphere
        const R c = surface_curvature(kind, R(sys.curvature[fs]));
        const Vec3<R> normal{-c * ray.p.x, -c * ray.p.y, R(1) - c * ray.p.z};
        const R cos_i = dot(ray.d, normal);
        const R mu = n_before / n_after;
        return prefilter_failure_root(R(1) - mu * mu * (R(1) - cos_i * cos_i));
    }
    default:
        // Unsupported hängt nur von der Topologie ab und ist in FP64 dasselbe.
        return R(1);
    }
}

} // namespace detail

// Strahl (v, f, l, Stützstelle `s`) der Vorfilterung: batch_source_ray, trace_ray mit
// PrefilterMarginObserver, Prüfung d_z > 0 nach der letzten Brechung und der Schritt auf die
// Referenzsphäre um `center` (reference_sphere_step, wie batch_ray_opl). Tabelle und
// Wellenlängenindex wie bei batch_ray_opl.
//
// `opl_offset` wird vor dem Trace von der Start-OPL abgezogen (OPL relativ zu einer Konstante
// des Tupels, im Kernel die axiale optische Länge bis zur letzten Fläche): Die OPD ändert das
// nicht, aber Kolben und Quadratsumme werden in FP32 aus kleinen Zahlen gebildet.
//
// Grenzabstand der Referenzsphäre (Prüfgrößen von reference_sphere_step): √(b² − g)/R_ref
// (= Kosinus des Winkels zwischen Strahl und Sphärennormale) und der Nenner
// σ (b + σ√(b² − g))/R_ref.
template <class R, class B, class SurfaceTable>
MKO_HD inline PrefilterRay<R> prefilter_ray(const SystemBatchView<B>& b, const SurfaceTable& table,
                                            int table_l, int v, int f, int l,
                                            const PupilSample<R>& s, Vec3<R> center, R opl_offset) {
    Ray<R> ray = batch_source_ray(b, v, f, l, s);
    ray.opl -= opl_offset;
    PrefilterMarginObserver<R> obs{R(1)};
    int failed_surface = -1;
    PrefilterRay<R> out{TraceStatus::Ok, R(0), R(0)};
    const TraceStatus st = trace_ray(table, table_l, ray, nullptr, &failed_surface, &obs);
    if (st != TraceStatus::Ok) {
        out.status = st;
        out.margin = detail::prefilter_surface_failure(table, table_l, failed_surface, st, ray);
        return out;
    }
    // Wie trace_to_reference_sphere: d_z > 0 nach der letzten Brechung
    if (!(ray.d.z > R(0))) {
        out.status = TraceStatus::Backward;
        out.margin = prefilter_failure_linear(ray.d.z);
        return out;
    }
    R margin = prefilter_min_margin(obs.margin, ray.d.z);
    // Prüfgrößen der Referenzsphäre (dieselben Ausdrücke wie reference_sphere_step)
    const R r2 = dot(center, center);
    const R inv_r = r2 > R(0) ? R(1) / msqrt(r2) : R(1);
    const Vec3<R> q = ray.p - center;
    const R bq = -dot(q, ray.d);
    const R g = dot(ray.p, ray.p - R(2) * center);
    const R disc = bq * bq - g;
    const R sigma = center.z >= R(0) ? R(1) : R(-1);
    const R root = disc > R(0) ? msqrt(disc) : R(0);
    const R denom = sigma * (bq + sigma * root) * inv_r;

    const int last = table.surfaces - 1;
    const R n_image = R(table.index[last * table.wavelengths + table_l]);
    const TraceStatus ref = reference_sphere_step(ray, center, n_image);
    if (ref != TraceStatus::Ok) {
        const R a = prefilter_failure_root(disc * inv_r * inv_r);
        const R c = disc >= R(0) ? prefilter_failure_linear(denom) : R(0);
        out.status = ref;
        out.margin = a > c ? a : c;
        return out;
    }
    margin = prefilter_min_margin(margin, root * inv_r);
    margin = prefilter_min_margin(margin, denom);
    out.opl = ray.opl;
    out.margin = margin;
    return out;
}

// Konstante des Tupels für opl_offset: axiale optische Länge Σ_{s=0}^{S−2} n_s t_s vom
// Scheitel der Blende bis zum Scheitel der letzten Fläche (Medium nach Fläche s, Index der
// Wellenlänge `table_l`). Jede Konstante ergäbe dieselbe OPD; diese liegt nahe an den OPL aller
// Strahlen eines Tupels (die Referenzsphäre geht durch den Scheitel der letzten Fläche).
template <class SurfaceTable>
MKO_HD inline auto prefilter_opl_offset(const SurfaceTable& table, int table_l) {
    using R = std::remove_cv_t<std::remove_reference_t<decltype(table.thickness[0])>>;
    R sum = R(0);
    for (int s = 0; s + 1 < table.surfaces; ++s) {
        sum += R(table.index[s * table.wavelengths + table_l]) * R(table.thickness[s]);
    }
    return sum;
}

} // namespace mko
