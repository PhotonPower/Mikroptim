#pragma once

// Pro-Strahl-Teil der Wellenfrontresiduen einer Batch-Variante, gemeinsam für die
// CPU-Referenz (residuals_cpu) und den Kernel RESIDUALS (ADR-0005). Header-only, MKO_HD,
// templatisiert über R; ohne Ausnahmen, Allokation und std-Container.
// Definition des Residuums: docs/numerik.md, Abschnitt 7 ("Residuen pro Strahl, M2").

#include "mikroptim/batch/system_batch_view.hpp"
#include "mikroptim/config.hpp"
#include "mikroptim/math/vec3.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/ray.hpp"
#include "mikroptim/trace/reference_sphere.hpp"
#include "mikroptim/trace/surface_ops.hpp"

namespace mko {

// Mittelpunkt der Referenzsphäre M_f = (0, h_f, w) der Variante v für Feld f in lokalen
// Koordinaten der letzten Fläche: h_f = field_target[v][f], w = thickness[S−1] aus `table`
// (Flächentabelle derselben Variante; im Kernel die Kopie im Shared Memory).
template <class R, class SurfaceTable>
MKO_HD inline Vec3<R> batch_image_point(const SystemBatchView<R>& b, const SurfaceTable& table,
                                        int v, int f) {
    return Vec3<R>{R(0), b.field_target[b.shape.field_at(v, f)],
                   table.thickness[table.surfaces - 1]};
}

// Optische Weglänge eines Strahls (v, f, l, Stützstelle `s`) bis zur Referenzsphäre um
// `center`: batch_source_ray, dann trace_to_reference_sphere über `table` mit
// Wellenlängenindex `table_l`. `table` ist die Flächentabelle der Variante v: auf der CPU
// view.surface_table(v) mit table_l = l, im Kernel die Kopie im Shared Memory, die nur die
// Brechzahlen der Wellenlänge l enthält (wavelengths = 1, table_l = 0). Beide liefern
// dieselben Zahlen.
// Rückgabe: Status des Strahls; *opl ist nur bei Ok gültig (sonst der teilweise
// fortgeschriebene Wert, der nicht verwendet werden darf).
// `observer` (optional) sieht Startzustand und Flächen wie bei trace_ray, z. B. ein
// RaySurfaceRecord für die Max-Reduktionen (Kernel MERIT, M2-T3); die OPL hängt nicht davon ab.
template <class R, class SurfaceTable, class Observer = NoSurfaceObserver>
MKO_HD inline TraceStatus batch_ray_opl(const SystemBatchView<R>& b, const SurfaceTable& table,
                                        int table_l, int v, int f, int l, const PupilSample<R>& s,
                                        Vec3<R> center, R* opl, Observer* observer = nullptr) {
    Ray<R> ray = batch_source_ray(b, v, f, l, s);
    const TraceStatus st =
        trace_to_reference_sphere(table, table_l, ray, center, nullptr, observer);
    *opl = ray.opl;
    return st;
}

// Erweiterungspunkt für die Gewichte eines Tupels (Feld f, Wellenlänge l). In M2 ist der
// Faktor 1: r_k = √(W_i V_j) · OPD_k [mm]. M4-T1 ersetzt ihn durch
// √(w_f · w_λ) / λ_ref (docs/optimierung.md, Abschnitt 2) aus field_weight, wavelength_weight
// und reference_wavelength_nm der Sicht; CPU-Referenz und Kernel rufen dieselbe Funktion.
template <class R>
MKO_HD inline R residual_scale(const SystemBatchView<R>& /*b*/, int /*f*/, int /*l*/) {
    return R(1);
}

// Residuum eines Strahls: scale · √(W_i V_j) · OPD_k. Versagte Strahlen übergeben OPD = 0
// (hartes Versagen wie opd_statistics) und erhalten das Residuum 0.
template <class R>
MKO_HD inline R ray_residual(R scale, const PupilSample<R>& s, R opd) {
    return scale * residual_weight(s) * opd;
}

} // namespace mko
