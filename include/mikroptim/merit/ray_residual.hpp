#pragma once

// Pro-Strahl-Teil der Wellenfrontresiduen einer Batch-Variante, gemeinsam für die
// CPU-Referenz (residuals_cpu) und den Kernel RESIDUALS (ADR-0005). Header-only, MKO_HD,
// templatisiert über R; ohne Ausnahmen, Allokation und std-Container.
// Definition des Residuums: docs/numerik.md, Abschnitt 7 ("Residuen pro Strahl, M2").

#include "mikroptim/batch/system_batch_view.hpp"
#include "mikroptim/config.hpp"
#include "mikroptim/math/scalar.hpp"
#include "mikroptim/math/vec3.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/ray.hpp"
#include "mikroptim/trace/reference_sphere.hpp"
#include "mikroptim/trace/surface_ops.hpp"

#include <type_traits>

namespace mko {

// Mittelpunkt der Referenzsphäre M_f = (0, h_f, w) der Variante v für Feld f in lokalen
// Koordinaten der letzten Fläche: h_f = field_target[v][f], w = thickness[S−1] aus `table`
// (Flächentabelle derselben Variante; im Kernel die Kopie im Shared Memory). Der Rechentyp ist
// der Zahlentyp der Tabelle; der Batch darf einen anderen Speichertyp haben (M7-T1).
template <class B, class SurfaceTable>
MKO_HD inline auto batch_image_point(const SystemBatchView<B>& b, const SurfaceTable& table, int v,
                                     int f) {
    using R = std::remove_cv_t<std::remove_reference_t<decltype(table.thickness[0])>>;
    return Vec3<R>{R(0), R(b.field_target[b.shape.field_at(v, f)]),
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
// Rechentyp R (Stützstelle, Tabelle, Ergebnis) und Speichertyp B des Batches dürfen verschieden
// sein (Kernel mit R = df64, M7-T1; batch_source_ray).
// `soft` (optional, Start {0, Ok}): weiches Versagen (numerik.md §11). Dann ist *opl auch für
// weich versagte Strahlen (in_piston) gültig, und soft->penalty enthält die Summe der
// Strafgrößen; ohne `soft` hartes Versagen wie in M2.
// `recorder` (optional, M5-T1): Aufzeichnung des Vorwärtsdurchlaufs für den adjungierten Modus
// (trace_to_reference_sphere); die OPL hängt nicht davon ab.
template <class R, class B, class SurfaceTable, class Observer = NoSurfaceObserver,
          class Recorder = NoTraceRecorder>
MKO_HD inline TraceStatus
batch_ray_opl(const SystemBatchView<B>& b, const SurfaceTable& table, int table_l, int v, int f,
              int l, const PupilSample<R>& s, Vec3<R> center, R* opl, Observer* observer = nullptr,
              typename detail::NonDeduced<SoftFailure<R>>::type* soft = nullptr,
              Recorder* recorder = nullptr) {
    Ray<R> ray = batch_source_ray(b, v, f, l, s);
    const TraceStatus st =
        trace_to_reference_sphere(table, table_l, ray, center, nullptr, observer, soft, recorder);
    *opl = ray.opl;
    return st;
}

// Normiertes Gewicht ŵ_f · ŵ_λ = (w_f / Σ w_f') · (w_λ / Σ w_λ') eines Tupels (0, wenn w_f = 0
// oder w_λ = 0, ohne Division). Gemeinsamer Teil von residual_scale und penalty_scale; Summen in
// aufsteigender Reihenfolge, kein a·b + c.
template <class B>
MKO_HD inline B tuple_weight(const SystemBatchView<B>& b, int f, int l) {
    const B wf = b.field_weight[f];
    const B wl = b.wavelength_weight[l];
    if (!(wf > B(0)) || !(wl > B(0))) {
        return B(0);
    }
    B sum_f = B(0);
    for (int i = 0; i < b.shape.fields; ++i) {
        sum_f += b.field_weight[i];
    }
    B sum_l = B(0);
    for (int i = 0; i < b.shape.wavelengths; ++i) {
        sum_l += b.wavelength_weight[i];
    }
    return (wf / sum_f) * (wl / sum_l);
}

// Gewichtsfaktor s_{f,l} eines Tupels (Feld f, Wellenlänge l), M4-T1 (docs/numerik.md,
// Abschnitt 7; docs/optimierung.md, Abschnitt 2):
//
//     s_{f,l} = √(ŵ_f · ŵ_λ) / λ_ref,   ŵ_f = w_f / Σ_f' w_f',   ŵ_λ = w_λ / Σ_l' w_λ',
//
// mit w_f = field_weight, w_λ = wavelength_weight und λ_ref = reference_wavelength_nm · 10⁻⁶
// in mm (wie evaluate_wavefront). Damit ist r_k = s_{f,l} · √(W_i V_j) · OPD_k dimensionslos
// (Wellen bei λ_ref) und die Merit ohne Versagen Σ_(f,l) ŵ_f ŵ_λ (RMS_fλ / λ_ref)², ein
// gewichteter mittlerer quadratischer RMS in Wellen², unabhängig von der Zahl der Felder und
// Wellenlängen und vom Maßstab der Gewichte. Ein Tupel mit w_f = 0 oder w_λ = 0 erhält den
// Faktor 0 (ohne Division; dann können die Summen 0 sein). Die Summen laufen in aufsteigender
// Reihenfolge, die Rechnung enthält kein a·b + c (keine FMA-Kontraktion), und Division und
// Wurzel sind auf dem Gerät korrekt gerundet: CPU und Kernel liefern bitgleiche Faktoren.
// CPU-Referenz und Kernel rufen dieselbe Funktion. Ergebnis im Speichertyp B des Batches; ein
// Kernel mit anderem Rechentyp wandelt es einmal pro Block um (M7-T1).
template <class B>
MKO_HD inline B residual_scale(const SystemBatchView<B>& b, int f, int l) {
    const B w = tuple_weight(b, f, l);
    if (!(w > B(0))) {
        return B(0);
    }
    const B lambda_ref_mm = b.reference_wavelength_nm * B(1e-6);
    return msqrt(w) / lambda_ref_mm;
}

// Residuum eines Strahls: scale · √(W_i V_j) · OPD_k. Versagte Strahlen übergeben OPD = 0
// (hartes Versagen wie opd_statistics) und erhalten das Residuum 0.
template <class R>
MKO_HD inline R ray_residual(R scale, const PupilSample<R>& s, R opd) {
    return scale * residual_weight(s) * opd;
}

// ---- Strafresiduen des weichen Versagens (M4-T2, docs/numerik.md, Abschnitt 11) ----------
//
// Jeder Strahl hat im weichen Modus genau ein Strafresiduum
//
//     p_k = √(w_fail · ŵ_f · ŵ_λ) · √(W_i V_j) · π_k   [Wellen bei λ_ref],
//
// π_k = 0 (Ok), Σ der Strafgrößen seiner weichen Ereignisse (weich versagt) bzw. diese Summe
// plus kHardFailurePenalty (hart versagt). Die Gewichtung wie bei den Wellenfrontresiduen macht
// Σ p_k² = w_fail · Σ_(f,l) ŵ_f ŵ_λ Σ_k W_k π_k² zum gewichteten Mittel von π² über Pupille, Felder
// und Wellenlängen, unabhängig von der Quadratur; w_fail hat die Einheit Wellen².

// Strafgröße eines hart versagten Strahls (d_z <= 0, verfehlte Kappe der Referenzsphäre,
// Unsupported), zusätzlich zu den weichen Ereignissen davor. 1 entspricht einer voll
// ausgebildeten Totalreflexion (k = −1) bzw. einem Verfehlen mit D = −1: größer als
// die Strafen nahe an jeder Versagensgrenze, sodass ein hart versagter Strahl teurer ist als ein
// gerade eben weich versagter. Mit dem Standard w_fail = 10⁶ Wellen² kostet ein hart versagter
// Strahl so viel wie ein Wellenfrontfehler von 1.000 Wellen (≈ 0,55 mm OPD) auf demselben Strahl.
inline constexpr double kHardFailurePenalty = 1.0;

// Standardgewicht w_fail [Wellen²] (ResidualSettings::fail_weight): Eine Strafgröße von 10⁻³
// (z. B. k = −10⁻³ knapp hinter dem Grenzwinkel) auf allen Strahlen kostet 1 Welle²; Strafen
// dominieren also, solange Strahlen merklich versagen, und verschwinden ohne Versagen.
inline constexpr double kDefaultFailWeight = 1e6;

// Gewichtsfaktor der Strafresiduen eines Tupels: √(w_fail · ŵ_f · ŵ_λ) (0 für ein Tupel mit
// Gewicht 0, dessen Strahlen dann auch nicht bestraft werden). Speichertyp B des Batches wie
// residual_scale.
template <class B>
MKO_HD inline B penalty_scale(const SystemBatchView<B>& b, int f, int l, B fail_weight) {
    const B w = tuple_weight(b, f, l);
    if (!(w > B(0))) {
        return B(0);
    }
    return msqrt(fail_weight * w);
}

// Strafgröße π_k eines Strahls aus Status und gesammelter Strafe der weichen Ereignisse
// (SoftFailure::penalty): 0 für Ok, `soft_penalty` für weich versagt, `soft_penalty` +
// kHardFailurePenalty für hart versagt.
template <class R>
MKO_HD inline R failure_penalty(TraceStatus st, R soft_penalty) {
    if (st == TraceStatus::Ok) {
        return R(0);
    }
    if (is_soft_failure(st)) {
        return soft_penalty;
    }
    return soft_penalty + R(kHardFailurePenalty);
}

// Strafresiduum p_k = scale · √(W_i V_j) · π_k mit scale = penalty_scale des Tupels.
template <class R>
MKO_HD inline R ray_penalty(R scale, const PupilSample<R>& s, TraceStatus st, R soft_penalty) {
    return scale * residual_weight(s) * failure_penalty(st, soft_penalty);
}

} // namespace mko
