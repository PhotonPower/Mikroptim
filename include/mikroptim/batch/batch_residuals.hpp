#pragma once

// Residuenvektor pro Variante eines SystemBatch: Ablage (ResidualLayout), Ergebnis
// (ResidualResult) und CPU-Referenz (residuals_cpu) für den Kernel RESIDUALS (M2-T2).
// Reines C++ ohne CUDA-Typen. Ablage und Definition: docs/architektur.md, Abschnitt
// "Kernel RESIDUALS"; docs/numerik.md, Abschnitt 7.

#include "mikroptim/batch/batch_memory.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/surface_ops.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace mko {

// Einstellungen eines Residuenlaufs (für alle Varianten und Felder gleich).
struct ResidualSettings {
    int rings = 10; // N, radiale Ringe der Quadratur (1 … kMaxPupilRings)
    int spokes = 8; // M, Speichen (Achsfelder: 1)
};

// Ablage eines Feldes im Residuenvektor einer Variante (trivial kopierbar, liegt für den
// Kernel als Feld [F] im Gerätespeicher).
struct ResidualFieldPlan {
    std::int64_t offset; // Beginn des Tupels (f, l = 0) im Residuenvektor der Variante
    int samples;         // n_f = sample_count der Quadratur des Feldes (N oder N · M)
    PupilMode mode;      // Pupillenart des Feldes (aus BatchTopology::pupil_mode)
};

static_assert(std::is_trivially_copyable_v<ResidualFieldPlan>);

// Ablage des Residuenvektors einer Variante: Konkatenation der Tupel (f, l), f-major, dann l,
// dann Stützstelle k = i · M + j der Quadratur des Feldes. Achsfelder haben n_f = N, alle
// anderen N · M Einträge pro Wellenlänge:
//
//     offset(f, l) = L · Σ_{f' < f} n_f' + l · n_f,   rays_per_variant = L · Σ_f n_f.
//
// Gleich für alle Varianten eines Batches, weil die Pupillenart pro Feld eine Eigenschaft der
// Topologie ist (BatchTopology::pupil_mode). rays_per_variant stimmt mit
// mko::rays_per_variant(F, L, N · M, Achsfelder, N) überein.
class ResidualLayout {
public:
    ResidualLayout() = default;
    // Wirft std::invalid_argument bei ungültiger Topologie, nicht festgelegten Pupillenarten
    // (topology.pupil_mode leer) und ungültigen Einstellungen (rings ∉ [1, 16], spokes < 1,
    // N · M > 2³¹ − 1).
    ResidualLayout(const BatchTopology& topology, const ResidualSettings& settings = {});

    int fields() const { return fields_; }
    int wavelengths() const { return wavelengths_; }
    const ResidualSettings& settings() const { return settings_; }
    // Zahl der Felder mit Pupillenart Axis.
    int axis_fields() const { return axis_fields_; }

    // Die folgenden Zugriffe prüfen f und l (std::out_of_range).
    PupilMode pupil_mode(int f) const;
    int samples(int f) const;                               // n_f
    const PupilQuadrature<double>& quadrature(int f) const; // make_pupil_quadrature(N, M, mode)
    std::int64_t offset(int f, int l) const;                // Beginn des Tupels (f, l)

    // Residuen (= Strahlen) pro Variante.
    std::int64_t rays_per_variant() const { return rays_per_variant_; }
    // Ablage pro Feld für den Kernel.
    std::span<const ResidualFieldPlan> plan() const { return plan_; }

private:
    void check_field(int f) const;

    int fields_ = 0;
    int wavelengths_ = 0;
    int axis_fields_ = 0;
    ResidualSettings settings_{};
    std::int64_t rays_per_variant_ = 0;
    std::vector<ResidualFieldPlan> plan_;
    std::vector<PupilQuadrature<double>> quadrature_;
};

// Residuen, OPD und Status für `variants` Varianten (CPU-Referenz und Download vom Gerät).
// Residuen der Variante v liegen ab v · stride; die ersten rays_per_variant Einträge sind die
// Strahlresiduen in der Ablage von ResidualLayout, die übrigen stride − rays_per_variant
// gehören weiteren Residuen (Paraxial, Strafen, ab M3) und sind hier 0. OPD und Status liegen
// ab v · rays_per_variant.
struct ResidualResult {
    int variants = 0;
    std::int64_t rays_per_variant = 0;
    std::int64_t stride = 0;          // Residuen pro Variante (>= rays_per_variant)
    std::vector<double> residuals;    // [V · stride], r_k [mm]; 0 bei versagten Strahlen
    std::vector<double> opd;          // [V · rays] OPD nach Kolbenabzug [mm] (0 bei Versagen);
                                      // leer, wenn nicht angefordert
    std::vector<std::uint8_t> status; // [V · rays] TraceStatus; leer, wenn nicht angefordert

    double residual(int v, std::int64_t i) const {
        return residuals[static_cast<std::size_t>(v) * static_cast<std::size_t>(stride) +
                         static_cast<std::size_t>(i)];
    }
    double opd_at(int v, std::int64_t i) const { return opd[ray_index(v, i)]; }
    TraceStatus status_at(int v, std::int64_t i) const {
        return static_cast<TraceStatus>(status[ray_index(v, i)]);
    }
    std::size_t ray_index(int v, std::int64_t i) const {
        return static_cast<std::size_t>(v) * static_cast<std::size_t>(rays_per_variant) +
               static_cast<std::size_t>(i);
    }
    // Zahl der Strahlen mit Status ≠ Ok (alle Varianten).
    std::size_t failed_count() const;
};

// CPU-Referenz des Kernels RESIDUALS für Variante v: für jedes Tupel (f, l) und jede
// Stützstelle k batch_ray_opl über surface_table(v), dann opd_statistics (Kolbenabzug, hartes
// Versagen mit Renormierung, Summation in aufsteigender Reihenfolge k) und
// r_k = ray_residual(residual_scale(f, l), sample_k, OPD_k). Ohne gültigen Strahl in einem
// Tupel sind dort alle OPD und Residuen 0, der Status bleibt gesetzt.
// Ergebnis mit variants = 1, stride = rays_per_variant, OPD und Status immer gefüllt.
// Für eine aus einem System gebaute Variante ist die OPD bitgleich zu evaluate_wavefront
// (Bildpunkte aus field_target und Arbeitsabstand).
// Wirft std::out_of_range (v), std::invalid_argument (Layout passt nicht zur Topologie,
// Variante verlässt ein Achsfeld: check_pupil_modes; Punktquelle mit NA außerhalb von
// (0, index[v][0][l])).
ResidualResult residuals_cpu(const SystemBatchHost& batch, int v, const ResidualLayout& layout);

// Wie oben für alle Varianten (CPU-Pfad ohne GPU, z. B. CI); Ablage wie der Download vom
// Gerät (stride = rays_per_variant).
ResidualResult residuals_cpu(const SystemBatchHost& batch, const ResidualLayout& layout);

// Prüft, ob `layout` zur Topologie passt (gleiche Felder, Wellenlängen und Pupillenarten);
// wirft std::invalid_argument.
void check_residual_layout(const BatchTopology& topology, const ResidualLayout& layout);

// RMS-Wellenfront und versagte Strahlen eines Tupels (Feld f, Wellenlänge l) einer Variante,
// berechnet aus OPD und Status eines ResidualResult (CPU-Referenz oder Download vom Gerät),
// z. B. für die Ausgabe von `mikroptim trace --gpu` (M2-T5).
struct TupleWavefront {
    int field = 0;
    int wavelength = 0;
    int samples = 0;           // n_f
    double valid_weight = 0.0; // Σ W_k der Strahlen mit Status Ok
    double rms = 0.0;          // √(Σ_ok W_k OPD_k² / valid_weight) [mm]; NaN ohne gültigen Strahl
    int failed = 0;            // Strahlen mit Status ≠ Ok

    // Mindestens ein gültiger Strahl (sonst ist rms NaN).
    bool valid() const { return valid_weight > 0.0; }
};

// Statistik aller Tupel der Variante v, Ablage [f · L + l] wie WavefrontResult::maps. Summiert
// wie opd_statistics (Pass 2) in aufsteigender Reihenfolge k mit den Gewichten W_k der
// Quadratur des Feldes; für das Ergebnis von residuals_cpu ist rms daher bitgleich zu
// evaluate_wavefront. Verlangt OPD und Status im Ergebnis und dieselbe Ablage wie `layout`.
// Wirft std::invalid_argument (OPD oder Status fehlen, Größen passen nicht zum Layout),
// std::out_of_range (v).
std::vector<TupleWavefront> tuple_wavefronts(const ResidualResult& result,
                                             const ResidualLayout& layout, int v);

} // namespace mko
