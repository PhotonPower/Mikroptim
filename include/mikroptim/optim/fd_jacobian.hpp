#pragma once

// Jacobi-Matrix per finiten Differenzen (M3-T2, Host). Mehrere Läufe (gebündelter LM) liegen
// mit ihren FD-Mustern hintereinander in einem Batch; Schablone und Formel pro Variable aus
// fd_pattern (variant_generator.hpp) und fd_derivative (fd_formula.hpp). Ablage, Pfade und
// Genauigkeit: docs/optimierung.md, Abschnitt „Finite-Differenzen-Jacobi“. Der GPU-Pfad mit
// J im Gerätespeicher steht in mikroptim/cuda/fd_jacobian.hpp.

#include "mikroptim/batch/batch_extrema.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/optim/extra_residuals.hpp"
#include "mikroptim/optim/trace_extra_residuals.hpp"
#include "mikroptim/optim/variable_set.hpp"
#include "mikroptim/optim/variant_generator.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mko {

// Größen der Jacobi-Matrizen mehrerer Läufe. Der Residuenvektor einer Variante hat
// R = R_v + E Zeilen: die R_v = ResidualLayout::trace_rows Zeilen des Traces (Strahlresiduen,
// im weichen Modus danach die Strafresiduen, M4-T2), danach die E Zusatzresiduen
// (ExtraResiduals, dahinter die trace-abhängigen TraceExtraResiduals, M4-T3), wie der
// Ausgabepuffer der TraceEngine mit extra_residuals = E.
struct JacobianShape {
    int runs = 0;              // Läufe
    int variables = 0;         // P
    std::int64_t ray_rows = 0; // R_v (Strahl- und Strafresiduen, trace_rows)
    int extra_rows = 0;        // E (Zusatzresiduen einschließlich der trace-abhängigen)

    std::int64_t rows() const { return ray_rows + extra_rows; } // R
    // Einträge von J (runs·P·R), von r₀ (runs·R) und der Status-Wechsel (runs·P).
    std::size_t jacobian_size() const;
    std::size_t residual_size() const;
    std::size_t column_count() const;

    friend bool operator==(const JacobianShape&, const JacobianShape&) = default;
};

// Einstellungen der Jacobi-Rechnung.
struct FdJacobianOptions {
    ResidualSettings quadrature{};       // Pupillenquadratur der Strahlresiduen
    FdScheme scheme = FdScheme::Central; // Central (Standard) oder Forward
    ExtraResiduals extra{};              // Zusatzresiduen (count = 0: keine)
    // Trace-abhängige Zusatzresiduen (M4-T3, count = 0: keine): für jede FD-Variante aus den
    // Max-Reduktionen ihres eigenen Traces (CPU: surface_extrema_cpu, GPU: Extrema der Tranche),
    // im Residuenvektor hinter `extra`.
    TraceExtraResiduals trace_extra{};
    // Host-Threads für die Zusatzresiduen aller FD-Varianten (evaluate_extra_residuals aus
    // paraxial_residuals.hpp, <= 0: Hardware); das Ergebnis ist für jede Zahl bitgleich.
    int extra_threads = 0;

    // Zahl aller Zusatzresiduen E = extra.count + trace_extra.count.
    int extra_count() const { return extra.count + trace_extra.count; }
};

// FD-Batch mehrerer Läufe: Lauf i belegt die Varianten [i·n, (i + 1)·n) mit
// n = fd_variant_count(scheme, P) im Muster von write_fd_variants (Variante i·n = x₀ des Laufs).
struct FdJacobianBatch {
    SystemBatchHost batch;
    std::vector<FdPattern> patterns; // [runs]
    int runs = 0;
    int variants_per_run = 0; // n
};

// Baut den FD-Batch für runs = x0.size()/P Läufe; x0 enthält die Startvektoren hintereinander
// (Lauf i: x0[i·P … (i + 1)·P)). Vorlage: hat `base` eine Variante, gilt sie für alle Läufe,
// sonst muss `base` genau `runs` Varianten haben (Vorlage pro Lauf, z. B. Homotopie-Zweige
// mit verschiedener Apertur). Die gemeinsamen Felder kommen aus `base`.
// Wirft std::invalid_argument bei P = 0, leerem x0, x0.size() kein Vielfaches von P, falscher
// Variantenzahl von `base` oder unzulässigem x₀ (wie fd_pattern).
FdJacobianBatch make_fd_jacobian_batch(const VariableSet& set, const SystemBatchHost& base,
                                       std::span<const double> x0,
                                       FdScheme scheme = FdScheme::Central);

// Jacobi-Matrizen, Residuen und Diagnose mehrerer Läufe auf dem Host.
//
// Ablage spaltenweise pro Lauf (Spalte p zusammenhängend, Lauf-Index außen):
//     J_i(r, p) = jacobian[(i·P + p)·R + r],   r₀_i(r) = residuals[i·R + r],
// also pro Lauf eine R × P-Matrix spaltenweise mit führender Dimension R und Abstand R·P
// zwischen den Läufen (passt zu cublasDgemmStridedBatched für JᵀJ und Jᵀr, M3-T4).
// status_changes[i·P + p]: Strahlen, deren Gültigkeit (Status Ok oder nicht) sich zwischen
// x₀ und einem der gestörten Punkte der Spalte p ändert (Diagnose: Im weichen Modus, M4-T2,
// sind Residuen und Strafen dort stetig, die Ableitung springt; im harten Modus springt das
// Residuum selbst).
struct FdJacobian {
    JacobianShape shape;
    FdScheme scheme = FdScheme::Central;
    std::vector<double> jacobian;             // [runs][P][R]
    std::vector<double> residuals;            // [runs][R], r(x₀) jedes Laufs
    std::vector<std::int64_t> status_changes; // [runs][P]
    std::vector<FdPattern> patterns;          // [runs], Punkte, Schritte und Schablonen

    // Zugriffe ohne Prüfung der Indizes (Ablage siehe oben).
    double at(int run, std::int64_t row, int p) const {
        return jacobian[column_offset(run, p) + static_cast<std::size_t>(row)];
    }
    std::span<const double> column(int run, int p) const {
        return std::span<const double>(jacobian).subspan(column_offset(run, p),
                                                         static_cast<std::size_t>(shape.rows()));
    }
    std::span<const double> residual(int run) const {
        const auto r = static_cast<std::size_t>(shape.rows());
        return std::span<const double>(residuals).subspan(static_cast<std::size_t>(run) * r, r);
    }
    std::int64_t status_change(int run, int p) const {
        return status_changes[static_cast<std::size_t>(run) *
                                  static_cast<std::size_t>(shape.variables) +
                              static_cast<std::size_t>(p)];
    }
    std::size_t column_offset(int run, int p) const {
        return (static_cast<std::size_t>(run) * static_cast<std::size_t>(shape.variables) +
                static_cast<std::size_t>(p)) *
               static_cast<std::size_t>(shape.rows());
    }
};

// Differenzbildung auf dem Host aus den Residuen aller Varianten des FD-Batches:
// `rays` mit Strahl- und Strafresiduen (Ablage wie ResidualResult, R_v = trace_rows(),
// beliebiger stride ≥ R_v, die Zeilen danach werden nicht gelesen) und Status; `extra` [V·E] aus
// evaluate_extra_residuals(extra, fd.batch, …) (paraxial_residuals.hpp, Ablage
// [(v)·E + i], E = extra_count). Für jede Spalte J = fd_derivative(Schablone, r₀,
// r_a, r_b, d_a, d_b) elementweise. Wirft std::invalid_argument, wenn die Größen nicht zum
// Batch passen oder der Status fehlt.
void fd_jacobian_from_residuals(const FdJacobianBatch& fd, const ResidualResult& rays,
                                std::span<const double> extra, int extra_count, FdJacobian& out);

// Zusatzresiduen aller Varianten des FD-Batches in der Ablage [v · E + i] mit
// E = options.extra_count(): evaluate_extra_residuals für options.extra, dahinter die
// trace-abhängigen Residuen aus `extrema` (Max-Reduktionen aller Varianten des Batches, nur
// nötig bei trace_extra.count > 0). Host-Threads options.extra_threads.
std::vector<double> fd_extra_residuals(const FdJacobianOptions& options,
                                       const SystemBatchHost& batch,
                                       const SurfaceExtremaResult& extrema);

// CPU-Pfad (CI ohne GPU und Referenz): make_fd_jacobian_batch, residuals_cpu für alle
// Varianten, bei trace-abhängigen Zusatzresiduen surface_extrema_cpu für alle Varianten,
// fd_extra_residuals (Host-Threads options.extra_threads), fd_jacobian_from_residuals. Wirft
// wie die Bausteine (std::invalid_argument auch bei extra.count < 0 oder count > 0 ohne
// evaluate, ebenso für trace_extra).
FdJacobian fd_jacobian_cpu(const VariableSet& set, const SystemBatchHost& base,
                           std::span<const double> x0, const FdJacobianOptions& options = {});

} // namespace mko
