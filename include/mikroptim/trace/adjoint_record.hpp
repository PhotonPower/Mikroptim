#pragma once

// Aufzeichnung des Vorwärtsdurchlaufs für den adjungierten Modus (M5-T1), gemeinsam für CPU und
// CUDA-Kernel (ADR-0005): header-only, MKO_HD, templatisiert über R, feste Größe, ohne
// Ausnahmen, Allokation und std-Container. Festlegungen: docs/architektur.md, Abschnitt
// "Modus ADJOINT: Vorwärtsdurchlauf mit Aufzeichnung (M5-T1)"; welche Größen der
// Rückwärtsdurchlauf (M5-T2) nutzt: docs/numerik.md, Abschnitt 5.
//
// Minimaler Satz: Pro Strahl und Fläche s werden nur der Zustand nach der Fläche (Schnittpunkt
// p[s] und Richtung d[s], 6 Zahlen) und der genommene Zweig (1 Byte) gespeichert, dazu einmal pro
// Strahl der Schritt auf die Referenzsphäre (s_ref und Zweig). Alle lokalen Zwischengrößen einer
// Flächenoperation (s₀, s, H, B, √D, Normale, cos I, k, √k, u, Profil, ∇Φ, Radikand; an der
// Referenzsphäre b, g, √(b² − g), σ) folgen aus dem Eingangszustand p[s − 1], d[s − 1] und der
// Flächentabelle in O(1) (recompute_surface_step, recompute_reference_step); die OPL geht in die
// Ableitungen nur linear ein und wird nicht gebraucht.
//
// Drei Formen:
// - AdjointRecord<R, C>: alle Zustände, Kapazität C (Kapazitätsklassen 16, 32, 64 für kleine S),
//   (6·sizeof(R) + 1)·C Byte pro Strahl.
// - CheckpointRecord<R, C, k>: der vollständige Zwischenzustand (Strahl mit OPL, Fortschritt,
//   Akkumulator des weichen Versagens) nur nach jeder k-ten Fläche, Zweige für alle Flächen; der
//   Rückwärtsdurchlauf rechnet jeden Abschnitt einmal neu vorwärts (replay_segment, bitgleich,
//   weil derselbe trace_segment-Aufruf mit demselben Zustand läuft).
// - SegmentStates<R, k>: Zustände eines neu gerechneten Abschnitts.

#include "mikroptim/config.hpp"
#include "mikroptim/math/scalar.hpp"
#include "mikroptim/math/vec3.hpp"
#include "mikroptim/trace/ghost_plate.hpp"
#include "mikroptim/trace/ray.hpp"
#include "mikroptim/trace/reference_sphere.hpp"
#include "mikroptim/trace/surface_extrema.hpp"
#include "mikroptim/trace/surface_ops.hpp"
#include "mikroptim/trace/surface_table.hpp"
#include "mikroptim/trace/surface_transform.hpp"
#include "mikroptim/trace/trace.hpp"

namespace mko {

// Kapazitätsklassen der Aufzeichnung in den Kerneln (CUDA kennt kein dynamisch bemessenes Local
// Memory; der Kernel ist pro Klasse instanziiert und der Host wählt die kleinste passende).
inline constexpr int kRecordCapacitySmall = 16;
inline constexpr int kRecordCapacityMedium = 32;
inline constexpr int kRecordCapacityLarge = kMaxSurfaces;

// Kleinste Kapazitätsklasse für S Flächen (S <= kMaxSurfaces).
MKO_HD constexpr int record_capacity_class(int surfaces) {
    return surfaces <= kRecordCapacitySmall    ? kRecordCapacitySmall
           : surfaces <= kRecordCapacityMedium ? kRecordCapacityMedium
                                               : kRecordCapacityLarge;
}

// Abstand der Checkpoints (Flächen pro Abschnitt) der CheckpointRecord in den Kerneln.
inline constexpr int kCheckpointInterval = 8;

// ---- Vollständige Aufzeichnung ---------------------------------------------------------------

// Alle Zustände eines Strahls (Recorder-Schnittstelle in trace.hpp). Eintrag s (0 <= s < count)
// ist der Zustand nach Fläche s: Schnittpunkt p[s] in lokalen Koordinaten von Fläche s, Richtung
// d[s] nach der Brechung, Zweig branch[s] (kBranch…; branch[0] = 0). Eintrag 0 ist der
// Startzustand. Bei hartem Versagen an Fläche s ist count = s. image_s, image_branch: Schritt auf
// die Referenzsphäre (nur gesetzt, wenn er nicht hart versagt, sonst 0). soft_mode: der Trace lief
// mit Akkumulator des weichen Versagens (die Neuberechnung muss dann dieselben Fortsetzungen
// nehmen). Flächen s >= C werden nicht aufgezeichnet (der Host wählt C >= S).
template <class R, int kCapacity = kMaxSurfaces>
struct AdjointRecord {
    static_assert(kCapacity >= 2 && kCapacity <= kMaxSurfaces, "Kapazität 2 … kMaxSurfaces");

    int count;
    bool soft_mode;
    Vec3<R> p[kCapacity];
    Vec3<R> d[kCapacity];
    unsigned char branch[kCapacity];
    R image_s;
    unsigned char image_branch;

    MKO_HD static constexpr int capacity() { return kCapacity; }

    MKO_HD void start(const Ray<R>& ray, const TraceProgress& /*at*/, const SoftFailure<R>* soft) {
        count = 1;
        soft_mode = soft != nullptr;
        p[0] = ray.p;
        d[0] = ray.d;
        branch[0] = kBranchRegular;
        image_s = R(0);
        image_branch = kBranchRegular;
    }
    MKO_HD void surface(const Ray<R>& ray, const TraceProgress& at, unsigned char b,
                        const SoftFailure<R>* /*soft*/) {
        const int s = at.surface;
        if (s < kCapacity) {
            count = s + 1;
            p[s] = ray.p;
            d[s] = ray.d;
            branch[s] = b;
        }
    }
    MKO_HD void image(R s_ref, unsigned char b, const Ray<R>& /*ray*/) {
        image_s = s_ref;
        image_branch = b;
    }
};

// ---- Checkpoints -------------------------------------------------------------------------------

// Vollständiger Zwischenzustand nach einer Fläche: Strahl (mit OPL), Fortschritt und Stand des
// Akkumulators (im harten Modus {0, Ok}). Von hier setzt trace_segment den Trace bitgleich fort.
template <class R>
struct TraceCheckpoint {
    Ray<R> ray;
    TraceProgress at;
    SoftFailure<R> soft;
};

// Aufzeichnung mit Checkpoints nach den Flächen 0, k, 2k, … (checkpoint[c] = Zustand nach Fläche
// c·k) und den Zweigen aller Flächen. count, soft_mode, image_s, image_branch wie AdjointRecord.
// Abschnitt c umfasst die Zustände c·k … min((c + 1)·k, count) − 1; replay_segment rechnet ihn aus
// checkpoint[c] neu.
template <class R, int kCapacity = kMaxSurfaces, int kInterval = kCheckpointInterval>
struct CheckpointRecord {
    static_assert(kCapacity >= 2 && kCapacity <= kMaxSurfaces, "Kapazität 2 … kMaxSurfaces");
    static_assert(kInterval >= 1 && kCapacity % kInterval == 0,
                  "Kapazität muss ein Vielfaches des Checkpoint-Abstands sein");

    int count;
    bool soft_mode;
    TraceCheckpoint<R> checkpoint[kCapacity / kInterval];
    unsigned char branch[kCapacity];
    R image_s;
    unsigned char image_branch;

    MKO_HD static constexpr int capacity() { return kCapacity; }
    MKO_HD static constexpr int interval() { return kInterval; }
    MKO_HD static constexpr int checkpoints() { return kCapacity / kInterval; }

    MKO_HD void start(const Ray<R>& ray, const TraceProgress& at, const SoftFailure<R>* soft) {
        count = 1;
        soft_mode = soft != nullptr;
        save(0, ray, at, soft);
        branch[0] = kBranchRegular;
        image_s = R(0);
        image_branch = kBranchRegular;
    }
    MKO_HD void surface(const Ray<R>& ray, const TraceProgress& at, unsigned char b,
                        const SoftFailure<R>* soft) {
        const int s = at.surface;
        if (s < kCapacity) {
            count = s + 1;
            branch[s] = b;
            if (s % kInterval == 0) {
                save(s / kInterval, ray, at, soft);
            }
        }
    }
    MKO_HD void image(R s_ref, unsigned char b, const Ray<R>& /*ray*/) {
        image_s = s_ref;
        image_branch = b;
    }

private:
    MKO_HD void save(int c, const Ray<R>& ray, const TraceProgress& at,
                     const SoftFailure<R>* soft) {
        checkpoint[c].ray = ray;
        checkpoint[c].at = at;
        checkpoint[c].soft = soft != nullptr ? *soft : SoftFailure<R>{R(0), TraceStatus::Ok};
    }
};

// Zustände p, d eines Abschnitts first … first + count − 1 (höchstens kLength), Recorder für
// replay_segment.
template <class R, int kLength>
struct SegmentStates {
    int first;
    int count;
    Vec3<R> p[kLength];
    Vec3<R> d[kLength];

    MKO_HD void start(const Ray<R>& ray, const TraceProgress& at, const SoftFailure<R>* /*soft*/) {
        first = at.surface;
        count = 1;
        p[0] = ray.p;
        d[0] = ray.d;
    }
    MKO_HD void surface(const Ray<R>& ray, const TraceProgress& at, unsigned char /*b*/,
                        const SoftFailure<R>* /*soft*/) {
        const int i = at.surface - first;
        if (i < kLength) {
            count = i + 1;
            p[i] = ray.p;
            d[i] = ray.d;
        }
    }
    MKO_HD void image(R /*s_ref*/, unsigned char /*b*/, const Ray<R>& /*ray*/) {}
};

// Rechnet Abschnitt `segment` einer CheckpointRecord neu vorwärts: Zustand aus
// checkpoint[segment] laden, trace_segment über die Flächen bis min((segment + 1)·k, count) − 1
// (mit Akkumulator, wenn der ursprüngliche Durchlauf ihn hatte), Zustände nach `out`. Derselbe
// Aufruf wie im ursprünglichen Durchlauf mit demselben Zustand, also bitgleich (Zustände, OPL,
// Akkumulator, Fortschritt). `end_state` (optional) erhält den Zustand am Ende des Abschnitts
// (Tests: gleich checkpoint[segment + 1]). Rückgabe: Status von trace_segment (Ok für jeden
// aufgezeichneten Abschnitt, weil count vor einer hart versagenden Fläche endet).
template <class R, class SurfaceTable, int kCapacity, int kInterval>
MKO_HD inline TraceStatus replay_segment(const SurfaceTable& sys, int wavelength_index,
                                         const CheckpointRecord<R, kCapacity, kInterval>& rec,
                                         int segment, SegmentStates<R, kInterval>& out,
                                         TraceCheckpoint<R>* end_state = nullptr) {
    TraceCheckpoint<R> c = rec.checkpoint[segment];
    SoftFailure<R>* const soft = rec.soft_mode ? &c.soft : nullptr;
    out.start(c.ray, c.at, soft);
    const int segment_end = (segment + 1) * kInterval;
    const int end = segment_end < rec.count ? segment_end : rec.count;
    NoSurfaceObserver* const no_observer = nullptr;
    const TraceStatus st =
        trace_segment(sys, wavelength_index, c.ray, c.at, end, &out, nullptr, no_observer, soft);
    if (end_state != nullptr) {
        *end_state = c;
    }
    return st;
}

// ---- Zugriff in absteigender Reihenfolge (Muster des Rückwärtsdurchlaufs) -------------------

// Ruft fn(s, p[s], d[s], branch[s]) für s = count − 1 … 0 auf.
template <class R, class SurfaceTable, int kCapacity, class Fn>
MKO_HD inline void for_each_state_reverse(const SurfaceTable& /*sys*/, int /*wavelength_index*/,
                                          const AdjointRecord<R, kCapacity>& rec, Fn&& fn) {
    for (int s = rec.count - 1; s >= 0; --s) {
        fn(s, rec.p[s], rec.d[s], rec.branch[s]);
    }
}

// Dasselbe für eine CheckpointRecord: Abschnitte in absteigender Reihenfolge je einmal neu
// vorwärts rechnen (replay_segment), dann ihre Zustände absteigend. Kosten: ein zusätzlicher
// Vorwärtsdurchlauf; Speicher: eine SegmentStates<R, k> auf dem Stack.
template <class R, class SurfaceTable, int kCapacity, int kInterval, class Fn>
MKO_HD inline void for_each_state_reverse(const SurfaceTable& sys, int wavelength_index,
                                          const CheckpointRecord<R, kCapacity, kInterval>& rec,
                                          Fn&& fn) {
    SegmentStates<R, kInterval> seg;
    for (int c = (rec.count - 1) / kInterval; c >= 0; --c) {
        replay_segment(sys, wavelength_index, rec, c, seg);
        for (int i = seg.count - 1; i >= 0; --i) {
            const int s = seg.first + i;
            fn(s, seg.p[i], seg.d[i], rec.branch[s]);
        }
    }
}

// ---- Lokale Zwischengrößen (Neuberechnung aus dem minimalen Satz) ---------------------------

// Nummer der Geisterplatte, die Fläche s nutzt (Zahl der Flächen vom Typ GhostPlate unter
// 1 … s − 1; im Rückwärtsdurchlauf auch absteigend mitzählbar).
template <class SurfaceTable>
MKO_HD inline int ghost_index_before(const SurfaceTable& sys, int s) {
    int g = 0;
    for (int i = 1; i < s; ++i) {
        if (sys.type[i] == SurfaceKind::GhostPlate) {
            ++g;
        }
    }
    return g;
}

// Alle lokalen Größen der Operation an Fläche s (numerik.md, Abschnitte 5 und 6), aus dem
// Eingangszustand (p_in, d_in) = Zustand nach Fläche s − 1 neu berechnet. Der Rückwärtsdurchlauf
// (M5-T2) bildet daraus die lokalen Ableitungen. Felder, die für den Flächentyp nicht gelten,
// sind 0.
template <class R>
struct SurfaceStepState {
    SurfaceKind kind;
    TraceStatus status;   // propagate_surface bei der Neuberechnung (Ok für aufgezeichnete Flächen)
    unsigned char branch; // genommene Fortsetzung (gleich der aufgezeichneten)
    R t;                  // Abstand der Scheitel (thickness[s − 1])
    R c;                  // wirksame Krümmung (surface_curvature)
    R n_before, n_after, mu;
    Vec3<R> p_in, d_in;
    // Dezentrierung und Verkippung der Fläche (M8-T3): Ist `transformed` gesetzt, gelten alle
    // folgenden Zwischengrößen im gekippten System der Fläche, mit dem Eingang
    // p_local = Rᵀ (p_in − t e_z − Δ), d_local = Rᵀ d_in und dem Transfer um 0; der Ausgang
    // (p_out, d_out) steht wie im Trace im nominellen System.
    bool transformed;
    SurfaceTransform<R> transform;
    Vec3<R> p_local, d_local;
    // Transfer in die Tangentialebene
    R s0;
    Vec3<R> p_plane;
    // Schnitt und Brechung (Stop, Plane, Sphere)
    R H, B, D;
    R root_d;       // √D (0 im Zweig kBranchSoftMissed)
    R s;            // Schnittparameter H/(B + √D) bzw. B/c (fortgesetzt)
    R normal_scale; // 1/|N| im Zweig kBranchSoftMissed, sonst 1
    Vec3<R> normal; // Einheitsnormale im Schnittpunkt
    R cos_i, k;
    R root_k; // √k (0 im Zweig kBranchSoftTIR)
    // Geisterplatte
    R inv_r2, u, profile_t, profile_dt_du, dn, radial, phase, grad_x, grad_y, dx, dy, radicand;
    R root_radicand; // √Radikand (0 im Zweig kBranchSoftTIR)
    // Ausgang (bitgleich zum Vorwärtsdurchlauf: dieselbe propagate_surface)
    Vec3<R> p_out, d_out;
};

// Neuberechnung der Operation an Fläche s (1 <= s < S) bei Wellenlänge `wavelength_index` aus
// dem Zustand (p_in, d_in) nach Fläche s − 1; `ghost` = ghost_index_before(sys, s), `soft_mode`
// wie im Vorwärtsdurchlauf (AdjointRecord::soft_mode). Ausgang, Status und Zweig kommen aus
// propagate_surface (bitgleich zum Vorwärtsdurchlauf), die Zwischengrößen aus denselben
// Formeln wie in surface_ops.hpp und ghost_plate.hpp (auf der CPU bitgleich zu den internen
// Werten; auf dem Gerät können sie wegen FMA-Kontraktion um Rundungen abweichen, was für
// Ableitungen ohne Belang ist).
template <class R, class SurfaceTable>
MKO_HD inline SurfaceStepState<R>
recompute_surface_step(const SurfaceTable& sys, int wavelength_index, int s, int ghost,
                       const Vec3<R>& p_in, const Vec3<R>& d_in, bool soft_mode) {
    const int L = sys.wavelengths;
    SurfaceStepState<R> st{};
    st.kind = sys.type[s];
    st.t = R(sys.thickness[s - 1]);
    st.c = surface_curvature(st.kind, R(sys.curvature[s]));
    st.n_before = R(sys.index[(s - 1) * L + wavelength_index]);
    st.n_after = R(sys.index[s * L + wavelength_index]);
    st.mu = st.n_before / st.n_after;
    st.p_in = p_in;
    st.d_in = d_in;

    GhostPlateParams<R> plate{};
    const GhostPlateParams<R>* plate_ptr = nullptr;
    if (st.kind == SurfaceKind::GhostPlate && ghost < sys.ghost_plates) {
        plate.coeff = sys.ghost_coeff + ghost * sys.ghost_terms;
        plate.terms = sys.ghost_terms;
        plate.norm_radius = R(sys.ghost_norm_radius[ghost]);
        plate.n_glass = R(sys.ghost_index[ghost * L + wavelength_index]);
        plate_ptr = &plate;
    }
    st.transformed = surface_transform_of(sys, s, st.transform);
    SoftFailure<R> scratch{R(0), TraceStatus::Ok};
    Ray<R> ray{p_in, d_in, R(0)};
    unsigned char branch = kBranchRegular;
    st.status = propagate_surface(ray, st.t, st.kind, R(sys.curvature[s]), st.n_before, st.n_after,
                                  plate_ptr, soft_mode ? &scratch : nullptr, &branch,
                                  st.transformed ? &st.transform : nullptr);
    st.branch = branch;
    st.p_out = ray.p;
    st.d_out = ray.d;

    // Eingang im System der Fläche (wie propagate_surface)
    Ray<R> plane{p_in, d_in, R(0)};
    R t_local = st.t;
    if (st.transformed) {
        plane.p.z -= st.t;
        ray_to_surface(plane, st.transform);
        t_local = R(0);
    }
    st.p_local = plane.p;
    st.d_local = plane.d;
    const Vec3<R> d_loc = plane.d;
    st.s0 = transfer_to_tangent_plane(plane, t_local);
    st.p_plane = plane.p;
    const R x = plane.p.x;
    const R y = plane.p.y;
    if (st.kind == SurfaceKind::GhostPlate) {
        st.normal = Vec3<R>{R(0), R(0), R(1)};
        st.normal_scale = R(1);
        st.cos_i = d_loc.z;
        if (plate_ptr != nullptr) {
            st.inv_r2 = R(1) / (plate.norm_radius * plate.norm_radius);
            st.u = (x * x + y * y) * st.inv_r2;
            const GhostProfile<R> prof = ghost_profile(plate.coeff, plate.terms, st.u);
            st.profile_t = prof.t;
            st.profile_dt_du = prof.dt_du;
            st.dn = plate.n_glass - st.n_before;
            st.radial = st.dn * prof.dt_du * R(2) * st.inv_r2;
            st.phase = st.dn * prof.t;
            st.grad_x = st.radial * x;
            st.grad_y = st.radial * y;
            st.dx = (st.n_before * d_loc.x + st.grad_x) / st.n_after;
            st.dy = (st.n_before * d_loc.y + st.grad_y) / st.n_after;
            st.radicand = R(1) - st.dx * st.dx - st.dy * st.dy;
            st.root_radicand = (branch & kBranchSoftTIR) != 0 ? R(0) : msqrt(st.radicand);
        }
        return st;
    }
    const R c = st.c;
    st.H = c * (x * x + y * y);
    st.B = d_loc.z - c * (x * d_loc.x + y * d_loc.y);
    st.D = st.B * st.B - c * st.H;
    const bool missed = (branch & kBranchSoftMissed) != 0;
    if (missed) {
        st.root_d = R(0);
        st.s = st.B / c;
    } else {
        st.root_d = msqrt(st.D);
        st.s = st.H / (st.B + st.root_d);
    }
    // Schnittpunkt (die Brechung bewegt ihn nicht), im System der Fläche
    const Vec3<R> q = st.transformed ? st.p_plane + st.s * d_loc : st.p_out;
    Vec3<R> N{-c * q.x, -c * q.y, R(1) - c * q.z};
    st.normal_scale = R(1);
    if (missed) {
        st.normal_scale = R(1) / norm(N);
        N = st.normal_scale * N;
    }
    st.normal = N;
    st.cos_i = dot(d_loc, N);
    st.k = R(1) - st.mu * st.mu * (R(1) - st.cos_i * st.cos_i);
    st.root_k = (branch & kBranchSoftTIR) != 0 ? R(0) : msqrt(st.k);
    return st;
}

// Lokale Größen des Schritts auf die Referenzsphäre (numerik.md, Abschnitt 7) aus dem Zustand
// nach der letzten Fläche.
template <class R>
struct ReferenceStepState {
    TraceStatus status;   // reference_sphere_step bei der Neuberechnung
    unsigned char branch; // kBranchSoftMissed: b² − g < 0, s_ref = b
    Vec3<R> p_in, d_in, center;
    R n_image;
    Vec3<R> q;    // p − M
    R b, g, disc; // b = −q·d, g = p·(p − 2M), disc = b² − g
    R sigma;      // +1 für M_z >= 0, sonst −1
    R root;       // √disc (0 im Zweig kBranchSoftMissed)
    R s;          // s_ref (bitgleich zum Vorwärtsdurchlauf)
    Vec3<R> p_out;
};

// Neuberechnung des Schritts auf die Referenzsphäre um `center` im Medium n_image aus dem
// Zustand (p_in, d_in) nach der letzten Fläche. s_ref, Ausgang, Status und Zweig kommen aus
// reference_sphere_step (bitgleich), die Zwischengrößen aus denselben Formeln.
template <class R>
MKO_HD inline ReferenceStepState<R> recompute_reference_step(const Vec3<R>& p_in,
                                                             const Vec3<R>& d_in, Vec3<R> center,
                                                             R n_image, bool soft_mode) {
    ReferenceStepState<R> st{};
    st.p_in = p_in;
    st.d_in = d_in;
    st.center = center;
    st.n_image = n_image;
    SoftFailure<R> scratch{R(0), TraceStatus::Ok};
    Ray<R> ray{p_in, d_in, R(0)};
    R s_ref = R(0);
    st.status = reference_sphere_step(ray, center, n_image, &s_ref, soft_mode ? &scratch : nullptr);
    st.branch = st.status == TraceStatus::SoftMissed ? kBranchSoftMissed : kBranchRegular;
    st.s = s_ref;
    st.p_out = ray.p;
    st.q = p_in - center;
    st.b = -dot(st.q, d_in);
    st.g = dot(p_in, p_in - R(2) * center);
    st.disc = st.b * st.b - st.g;
    st.sigma = center.z >= R(0) ? R(1) : R(-1);
    st.root = st.branch == kBranchSoftMissed ? R(0) : msqrt(st.disc);
    return st;
}

} // namespace mko
