#pragma once

// FP32-Vorfilterung (M7-T2) für den Multistart: Kriterien, Rohdaten pro Variante,
// konservative Entscheidung "behalten/aussortieren" und die CPU-Referenz in float. Reines C++
// ohne CUDA-Typen. Formeln, Fehlerabschätzung und Margen: docs/numerik.md, Abschnitt 12
// ("FP32-Vorfilterung"); Einsatz: docs/optimierung.md, Abschnitt 8; GPU-Pfad:
// mko::cuda::TraceEngine::prefilter_variants.
//
// VORFILTERUNG: Die Werte dieses Pfads (FP32) sind nur eine Grobbewertung. Sie dürfen nie in
// Merit, Residuen, Jacobi-Matrix oder Optimierung einfließen (AGENTS.md §6.1, ADR-0003); eine
// behaltene Variante wird danach in FP64 (oder df64) neu bewertet. Die Entscheidung ist so
// gebaut, dass eine Variante nur aussortiert wird, wenn sie die Kriterien auch in FP64 sicher
// erfüllt.

#include "mikroptim/batch/batch_merit.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"

#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace mko {

// Einheitsrundung von float (u = 2⁻²⁴).
inline constexpr double kFloatUnitRoundoff = 0x1p-24;

// Kriterien zum Aussortieren (konfigurierbar). Eine Variante wird aussortiert, wenn eines
// erfüllt ist. Dieselbe Regel ohne Margen auf FP64-Werte angewandt ist die "Wahrheit"
// (prefilter_decide_exact).
struct PrefilterCriteria {
    // (a) Strahlversagen: failed > max_failed_fraction · R_v (R_v Strahlen pro Variante).
    // Werte >= 1 schalten das Kriterium ab.
    double max_failed_fraction = 0.5;
    // (b) Merit über einer festen Schwelle: F > max_merit (F = Σ r² + Anteil weiterer
    // Residuen, Einheit der Merit, in M2 mm²). +∞ schaltet ab.
    double max_merit = std::numeric_limits<double>::infinity();
    // (c) Merit relativ zur besten Variante des Batches: F > max_merit_factor · F_best. F_best
    // ist die kleinste Merit der Varianten, die (a) nicht aussortiert und die mindestens einen
    // gültigen Strahl haben. +∞ schaltet ab.
    double max_merit_factor = std::numeric_limits<double>::infinity();
};

// Sicherheitsmargen der FP32-Seite (docs/numerik.md, Abschnitt 12).
struct PrefilterMargins {
    // η (Einheit eines Richtungskosinus): Strahlen, deren Prüfgrößen (d_z, cos I = √D,
    // cos I' = √k, √Radikand der Geisterplatte, Diskriminante und Nenner der Referenzsphäre)
    // näher als η an der Versagensgrenze liegen, gelten als unsicher: Sie zählen nicht als
    // sicher versagt und gehen nicht in die untere Schranke der Merit ein. 0,15 entspricht
    // Winkeln über 81,4°; die Stresstests zeigen bei 0,05 einen falsch sicheren Strahl, bei 0,1
    // und darüber keinen (numerik.md, Abschnitt 12).
    double boundary = 0.15;
    // κ und m: Fehlermodell der OPD pro sicher gültigem Strahl |OPD_FP32 − OPD_FP64| <=
    // κ · u · Λ_v + m · |OPD| (u = 2⁻²⁴, Λ_v = größte optische Länge eines sicher gültigen
    // Strahls der Variante). Gemessen: κ <= 4,7 bei m = 0,01 (numerik.md, Abschnitt 12).
    double opl_error_factor = 32.0;
    double merit_relative = 0.01;
};

struct PrefilterOptions {
    PrefilterCriteria criteria{};
    PrefilterMargins margins{};
};

// Rohdaten der Vorfilterung pro Variante (Kernel bzw. prefilter_stats_cpu), in FP32 gerechnet
// und als double abgelegt.
struct PrefilterStats {
    int variants = 0;
    std::int64_t rays_per_variant = 0;
    // F32_v = Σ r_k² über die sicher gültigen Strahlen (Kolben pro Tupel nur über diese),
    // ohne Anteil weiterer Residuen.
    std::vector<double> merit;
    // Λ_v = größte optische Länge |opl_offset| + |OPL − opl_offset| eines sicher gültigen
    // Strahls [mm] (Maßstab der Rundungsfehler); 0 ohne sicher gültigen Strahl.
    std::vector<double> optical_length;
    std::vector<std::int64_t> failed;           // sicher versagte Strahlen
    std::vector<std::int64_t> uncertain_ok;     // in FP32 gültig, aber nahe der Grenze
    std::vector<std::int64_t> uncertain_failed; // in FP32 versagt, aber nahe der Grenze

    std::int64_t uncertain(int v) const {
        const auto i = static_cast<std::size_t>(v);
        return uncertain_ok[i] + uncertain_failed[i];
    }
};

// Gründe für das Aussortieren (Bitmaske); 0 = behalten.
enum PrefilterReason : std::uint8_t {
    kPrefilterKeep = 0,
    kPrefilterFailedRays = 1,    // Kriterium (a)
    kPrefilterMeritAbsolute = 2, // Kriterium (b)
    kPrefilterMeritRelative = 4, // Kriterium (c)
};

// Ergebnis der Vorfilterung: Maske und Werte.
struct PrefilterResult {
    PrefilterStats stats;
    // Schranken der FP64-Merit (einschließlich des Anteils weiterer Residuen):
    // merit_lower <= F_FP64 <= merit_upper unter den Annahmen von numerik.md, Abschnitt 12;
    // merit_upper = +∞ bei unsicheren Strahlen (deren OPL in FP64 unbekannt ist).
    std::vector<double> merit_lower;
    std::vector<double> merit_upper;
    std::vector<std::uint8_t> reject; // Bitmaske PrefilterReason pro Variante, 0 = behalten
    // Obere Schranke von F_best für Kriterium (c) und die Variante, die sie liefert; +∞ und −1,
    // wenn keine Variante sicher in die Referenzmenge gehört oder (c) abgeschaltet ist.
    double reference_merit = std::numeric_limits<double>::infinity();
    int reference_variant = -1;
    int kept = 0; // Zahl der behaltenen Varianten

    bool keep(int v) const { return reject[static_cast<std::size_t>(v)] == kPrefilterKeep; }
};

// Prüft Kriterien und Margen: Brüche, Schwellen und Faktoren >= 0 (nicht NaN), η und κ endlich
// und >= 0, m in [0, 1). Wirft std::invalid_argument.
void check_prefilter_options(const PrefilterOptions& options);

// Konservative Entscheidung aus den Rohdaten (Formeln: docs/numerik.md, Abschnitt 12):
//
//   Δ_v = κ u Λ_v √(Σ_(f,l) s_(f,l)²)    (s = residual_scale des Tupels)
//   L_v = (max(0, √F32_v − Δ_v) / (1 + m))² + E_v,   U_v = ((√F32_v + Δ_v) / (1 − m))² + E_v
//   (a) failed_v > max_failed_fraction · R_v
//   (b) L_v > max_merit
//   (c) L_v > max_merit_factor · min_u U_u  über die Varianten u ohne unsichere Strahlen, die (a)
//       nicht erfüllen und mindestens einen gültigen Strahl haben.
//
// `extra_merit` (E_v) ist leer oder hat V Werte in FP64 (z. B. Σ e² der paraxialen
// Nebenbedingungen). `batch` und `layout` liefern R_v und s_(f,l). Wirft std::invalid_argument
// bei unpassenden Größen oder ungültigen Optionen (Brüche oder Faktoren < 0 bzw. NaN, η < 0,
// κ < 0, m ∉ [0, 1)).
PrefilterResult prefilter_decide(PrefilterStats stats, const SystemBatchHost& batch,
                                 const ResidualLayout& layout, const PrefilterOptions& options,
                                 std::span<const double> extra_merit = {});

// Dieselbe Regel ohne Margen auf FP64-Werte ("Wahrheit" der Abnahme): merit[v] ohne Anteil
// weiterer Residuen (merit_cpu oder TraceEngine im Modus MERIT ohne extra_merit), failed[v],
// dazu E_v aus `extra_merit` (leer oder V Werte). Rückgabe: Bitmaske PrefilterReason pro
// Variante. Wirft std::invalid_argument wie prefilter_decide.
std::vector<std::uint8_t> prefilter_decide_exact(const MeritResult& fp64,
                                                 const PrefilterCriteria& criteria,
                                                 std::span<const double> extra_merit = {});

// Strahlen einer Variante in der Vorfilterung (CPU, float), Ablage von ResidualLayout:
// Status, Grenzabstand, Klasse und OPD (Kolben pro Tupel über die sicher gültigen Strahlen,
// 0 sonst), alles als double bzw. Byte. Für Tests und Diagnose.
struct PrefilterRayData {
    std::vector<std::uint8_t> status;   // TraceStatus in float
    std::vector<double> margin;         // Grenzabstand
    std::vector<double> opd;            // OPD [mm] der sicher gültigen Strahlen, sonst 0
    std::vector<double> optical_length; // |opl_offset| + |OPL − opl_offset| [mm], sonst 0
    std::vector<std::uint8_t> certain;  // 1: Entscheidung sicher (Grenzabstand > η)
};

// Wirft wie residuals_cpu.
PrefilterRayData prefilter_rays_cpu(const SystemBatchHost& batch, int v,
                                    const ResidualLayout& layout, double boundary_margin);

// CPU-Referenz der Rohdaten des Kernels in float (gleiche Pro-Strahl-Funktion
// prefilter_ray, Summen sequenziell in aufsteigender Reihenfolge k statt als Baum; daher nicht
// bitgleich zum Kernel). Wirft wie residuals_cpu und bei boundary_margin < 0 oder NaN.
PrefilterStats prefilter_stats_cpu(const SystemBatchHost& batch, const ResidualLayout& layout,
                                   double boundary_margin);

// prefilter_stats_cpu und prefilter_decide in einem Aufruf.
PrefilterResult prefilter_cpu(const SystemBatchHost& batch, const ResidualLayout& layout,
                              const PrefilterOptions& options,
                              std::span<const double> extra_merit = {});

} // namespace mko
