#pragma once

// Pro-Strahl-Teil des Modus EVAL (M8-T1), gemeinsam für die CPU-Referenz (eval_cpu) und den
// Kernel EVAL (ADR-0005). Header-only, MKO_HD, templatisiert über R; ohne Ausnahmen,
// Allokation und std-Container. Festlegungen: docs/numerik.md, Abschnitt 7 ("Auswertung auf
// dem Raster, M8-T1"); Ablauf im Kernel: docs/architektur.md, Abschnitt "Modus EVAL".
//
// Im Modus EVAL ist jedes Versagen hart (docs/numerik.md, Abschnitt 11).

#include "mikroptim/batch/system_batch_view.hpp"
#include "mikroptim/config.hpp"
#include "mikroptim/math/scalar.hpp"
#include "mikroptim/math/vec3.hpp"
#include "mikroptim/trace/ray.hpp"
#include "mikroptim/trace/reference_sphere.hpp"
#include "mikroptim/trace/surface_ops.hpp"
#include "mikroptim/trace/trace.hpp"

#include <cstdint>

namespace mko {

// Lage der Referenzsphäre um den idealen Bildpunkt M_f (1 Byte, Werte stabil).
enum class EvalReference : std::uint8_t {
    // Sphäre um M_f durch die Mitte der paraxialen Austrittspupille E = (0, 0, z_AP), Radius
    // |M_f − E| (Standard des Modus EVAL, Endbewertung nach docs/numerik.md, Abschnitt 7).
    ExitPupil = 0,
    // Sphäre um M_f durch den Scheitel der letzten Fläche, Radius |M_f| (wie MERIT und
    // RESIDUALS, M1-T7).
    Vertex = 1,
};

// Zahl der Feldstrahlen pro Tupel (Hauptstrahl und Nachbarstrahlen für die Fokuslagen) und
// ihre Nummern.
inline constexpr int kEvalFieldRays = 4;
inline constexpr int kEvalChiefRay = 0;        // Pupillenmitte (0, 0)
inline constexpr int kEvalTangentialPlus = 1;  // (0, +δ)
inline constexpr int kEvalTangentialMinus = 2; // (0, −δ)
inline constexpr int kEvalSagittal = 3;        // (+δ, 0)
// Werte pro Feldstrahl in der Ausgabe: x, y (Bildebene, relativ zum idealen Bildpunkt),
// d_x, d_y, d_z (Richtung im Bildraum).
inline constexpr int kEvalFieldRayValues = 5;

// Normierte Pupillenkoordinaten (x, y) des Feldstrahls j mit dem Abstand δ der Nachbarstrahlen.
template <class R>
MKO_HD inline void eval_field_ray_point(int j, R delta, R& x, R& y) {
    x = R(0);
    y = R(0);
    if (j == kEvalTangentialPlus) {
        y = delta;
    } else if (j == kEvalTangentialMinus) {
        y = -delta;
    } else if (j == kEvalSagittal) {
        x = delta;
    }
}

// Startstrahl der Variante v für Feld f, Wellenlänge l und den Rasterpunkt (x, y) in
// normierten Pupillenkoordinaten (x² + y² <= 1), lokale Koordinaten von Fläche 0. Dieselben
// Formeln wie batch_source_ray mit x = ρ sin φ und y = ρ cos φ, aber ohne den Umweg über
// (ρ, φ) (keine Rundung durch sin und cos des Azimuts):
// - kollimiert: p₀ = (a x, a y, 0), d₀ = (0, sin θ_f, cos θ_f), OPL₀ = p₀·d₀,
// - Punktquelle: p₀ = q, d₀ = (s x, s y, √(1 − s²(x² + y²))) mit s = NA / index[v][0][l],
//   OPL₀ = 0.
template <class R, class B>
MKO_HD inline Ray<R> eval_source_ray(const SystemBatchView<B>& b, int v, int f, int l, R x, R y) {
    Ray<R> ray;
    if (b.source == SourceKind::Point) {
        const R s = R(b.source_na) / R(b.index[b.shape.index_at(v, 0, l)]);
        const R dx = s * x;
        const R dy = s * y;
        ray.p = Vec3<R>{R(b.source_point[0]), R(b.source_point[1]), R(b.source_point[2])};
        ray.d = Vec3<R>{dx, dy, msqrt(R(1) - (dx * dx + dy * dy))};
        ray.opl = R(0);
        return ray;
    }
    const R a = R(b.pupil_radius[v]);
    const R theta = R(b.field_angle[b.shape.field_at(v, f)]);
    ray.p = Vec3<R>{a * x, a * y, R(0)};
    ray.d = Vec3<R>{R(0), msin(theta), mcos(theta)};
    ray.opl = dot(ray.p, ray.d);
    return ray;
}

// Schritt auf die Referenzsphäre durch die Austrittspupille (Modus ExitPupil). Die Sphäre hat
// den Mittelpunkt `center` = M und die vorzeichenbehaftete Krümmung κ = σ / R mit
// R = |M − E| (E Mitte der Austrittspupille) und σ = +1 für M_z >= E_z, sonst −1; κ = 0 für
// eine Austrittspupille im Unendlichen (bildseitig telezentrisch, R → ∞).
//
// Mit q = p − M, b = −q·d und t = −|q × d|² (= b² − |q|² für |d| = 1) schneidet der Strahl
// die Kappe durch E bei s = b − σ √(R² + t) (docs/numerik.md, Abschnitt 7). Addiert wird
// n_image · (s + σ R), also ohne den für alle Strahlen eines Feldes gleichen Anteil −σ n R,
// der beim Kolbenabzug ohnehin wegfällt:
//
//     s + σ R = b − κ t / (√(1 + κ² t) + 1).
//
// Die Form enthält keine Differenz großer Zahlen (R² gegen |q|²) und geht für κ → 0 stetig in
// b über (Grenzfall einer Sphäre mit unendlichem Radius). Status Missed, wenn 1 + κ² t < 0
// (die Gerade verfehlt die Sphäre) oder das Ergebnis NaN ist; der Strahl bleibt dann
// unverändert. Bei Ok wird nur die OPL fortgeschrieben, nicht die Position.
template <class R>
MKO_HD inline TraceStatus exit_pupil_sphere_step(Ray<R>& ray, Vec3<R> center, R kappa, R n_image) {
    const Vec3<R> q = ray.p - center;
    const R b = -dot(q, ray.d);
    const Vec3<R> c{q.y * ray.d.z - q.z * ray.d.y, q.z * ray.d.x - q.x * ray.d.z,
                    q.x * ray.d.y - q.y * ray.d.x};
    const R t = -dot(c, c);
    const R disc = R(1) + kappa * kappa * t;
    if (!(disc >= R(0))) {
        return TraceStatus::Missed;
    }
    const R step = b - kappa * t / (msqrt(disc) + R(1));
    if (!(step == step)) {
        return TraceStatus::Missed;
    }
    ray.opl += n_image * step;
    return TraceStatus::Ok;
}

// Ergebnis eines Strahls im Modus EVAL.
template <class R>
struct EvalRay {
    TraceStatus status;
    int failed_surface; // −1 ohne Versagen, S für Bildraum bzw. Referenzsphäre
    R opl;              // OPL bis zur Referenzsphäre (Modus ExitPupil: ohne −σ n R)
    Vec3<R> image;      // Schnittpunkt mit der Bildebene (lokale Koordinaten, z = 0)
    Vec3<R> d;          // Richtung im Bildraum (nach der letzten Fläche)
};

// Ein Strahl des Modus EVAL: Startstrahl (eval_source_ray), trace_ray über `table`
// (Flächentabelle der Variante v mit Wellenlängenindex `table_l`, wie batch_ray_opl), Prüfung
// d_z > 0 nach der letzten Fläche (sonst Backward an Fläche S), Schnittpunkt mit der
// Bildebene im Abstand thickness[S−1] (transfer_to_image_plane auf einer Kopie) und Schritt
// auf die Referenzsphäre um `center`: Vertex mit reference_sphere_step (dieselben Operationen
// wie trace_to_reference_sphere), ExitPupil mit exit_pupil_sphere_step und der Krümmung
// `kappa`. Versagt die Referenzsphäre, ist failed_surface = S. Bei Versagen sind opl, image
// und d nicht zu verwenden.
template <class R, class B, class SurfaceTable>
MKO_HD inline EvalRay<R> eval_ray(const SystemBatchView<B>& b, const SurfaceTable& table,
                                  int table_l, int v, int f, int l, R x, R y, Vec3<R> center,
                                  EvalReference reference, R kappa) {
    EvalRay<R> out{};
    Ray<R> ray = eval_source_ray(b, v, f, l, x, y);
    int failed = -1;
    TraceStatus st = trace_ray(table, table_l, ray, nullptr, &failed);
    const int S = table.surfaces;
    if (st == TraceStatus::Ok && !(ray.d.z > R(0))) {
        st = TraceStatus::Backward;
        failed = S;
    }
    if (st == TraceStatus::Ok) {
        Ray<R> image = ray;
        (void)transfer_to_image_plane(table, table_l, image); // d_z > 0 ist geprüft
        out.image = image.p;
        out.d = ray.d;
        const R n_image = R(table.index[(S - 1) * table.wavelengths + table_l]);
        st = reference == EvalReference::Vertex
                 ? reference_sphere_step(ray, center, n_image)
                 : exit_pupil_sphere_step(ray, center, kappa, n_image);
        if (st != TraceStatus::Ok) {
            failed = S;
        }
    }
    out.status = st;
    out.failed_surface = failed;
    out.opl = ray.opl;
    return out;
}

} // namespace mko
