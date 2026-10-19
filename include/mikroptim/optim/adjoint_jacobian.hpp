#pragma once

// Adjungierte Jacobi-Matrix (M5-T2, Host). Die Zeilen des Traces (Strahl- und Strafresiduen)
// kommen aus dem Rückwärtsdurchlauf pro Strahl (include/mikroptim/trace/adjoint_backward.hpp,
// include/mikroptim/merit/ray_jacobian.hpp) mit dem Kolbenabzug als linearer Projektion; die
// Zusatzresiduen (ExtraResiduals, TraceExtraResiduals) per finiten Differenzen auf dem Host
// (hybrider Ansatz). Ablage wie die FD-Jacobi (fd_jacobian.hpp), damit der LM (M5-T3) sie
// austauschen kann. Formeln: docs/numerik.md, Abschnitt "Rückwärtsdurchlauf (M5-T2)"; Ablage und
// Pfade: docs/architektur.md, Abschnitt "Modus ADJOINT: Jacobi-Matrix (M5-T2)". Der GPU-Pfad
// steht in mikroptim/cuda/adjoint_jacobian.hpp.

#include "mikroptim/batch/batch_extrema.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/merit/ray_jacobian.hpp"
#include "mikroptim/optim/extra_residuals.hpp"
#include "mikroptim/optim/fd_jacobian.hpp"
#include "mikroptim/optim/trace_extra_residuals.hpp"
#include "mikroptim/optim/variable_set.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mko {

// Spaltenzuordnung der Parameter einer Flächentabelle (Besitzer der AdjointParameterMap) für
// Batches mit der Topologie der Variablenmenge und den Wellenlängen von `batch` (Steigungen der
// Glasvariablen: (1, ω_l, ω_l²) mit ω_l = buchdahl_omega(λ_l), wie VariableSet::apply).
// Wirft std::invalid_argument, wenn die Topologie nicht passt oder die Buchdahl-Koordinate bei
// einer Wellenlänge nicht definiert ist (nur mit Glasvariablen).
class AdjointParameters {
public:
    AdjointParameters() = default;
    AdjointParameters(const VariableSet& set, const SystemBatchHost& batch);

    // Sicht auf die eigenen Vektoren (gültig, solange das Objekt lebt und nicht verschoben wird).
    AdjointParameterMap view() const;

    int variables() const { return variables_; }
    std::span<const int> curvature() const { return curvature_; }  // [S]
    std::span<const int> thickness() const { return thickness_; }  // [S]
    std::span<const int> glass() const { return glass_; }          // [S · 3]
    std::span<const double> glass_slope() const { return slope_; } // [L · 3]
    std::span<const int> ghost() const { return ghost_; }          // [G · K]
    int surfaces() const { return surfaces_; }
    int wavelengths() const { return wavelengths_; }
    int ghost_plates() const { return ghost_plates_; }
    int ghost_terms() const { return ghost_terms_; }

private:
    int variables_ = 0;
    int surfaces_ = 0;
    int wavelengths_ = 0;
    int ghost_plates_ = 0;
    int ghost_terms_ = 0;
    std::vector<int> curvature_;
    std::vector<int> thickness_;
    std::vector<int> glass_;
    std::vector<double> slope_;
    std::vector<int> ghost_;
};

// Einstellungen der adjungierten Jacobi-Rechnung (wie FdJacobianOptions ohne Schema: Die
// Zusatzzeilen nutzen immer das zentrale Schema mit den Schablonen an Schranken, fd_pattern).
struct AdjointJacobianOptions {
    ResidualSettings quadrature{};     // Pupillenquadratur und Strahlversagen der Strahlresiduen
    ExtraResiduals extra{};            // Zusatzresiduen (count = 0: keine)
    TraceExtraResiduals trace_extra{}; // trace-abhängige Zusatzresiduen (M4-T3), hinter `extra`
    int extra_threads = 0;             // Host-Threads für die Zusatzresiduen (<= 0: Hardware)
    // Host-Threads der CPU-Referenz für die Trace-Zeilen (Paare aus Variante und Feld, M5-T3;
    // <= 0: Hardware). Das Ergebnis ist für jede Zahl bitgleich (jedes Paar schreibt nur seine
    // eigenen Zeilen). Der GPU-Pfad ignoriert den Wert.
    int threads = 0;

    int extra_count() const { return extra.count + trace_extra.count; }
};

// Optionen der adjungierten Jacobi mit Quadratur, Zusatzresiduen und extra_threads der
// FD-Optionen (das Schema gilt nur für die FD-Jacobi; die Zusatzzeilen sind immer zentral). Für
// den LM mit JacobianMethod::Adjoint (M5-T3).
AdjointJacobianOptions adjoint_options_of(const FdJacobianOptions& options);

// Jacobi-Matrizen und Residuen mehrerer Läufe. Ablage wie FdJacobian:
//     J_i(r, p) = jacobian[(i·P + p)·R + r],   r₀_i(r) = residuals[i·R + r],
// R = shape.rows() = trace_rows + E. status[i · rays + k] = TraceStatus des Strahls k von Lauf i.
struct AdjointJacobian {
    JacobianShape shape;
    std::vector<double> jacobian;     // [runs][P][R]
    std::vector<double> residuals;    // [runs][R]
    std::vector<std::uint8_t> status; // [runs][rays]
    std::int64_t rays_per_variant = 0;

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
    std::size_t column_offset(int run, int p) const {
        return (static_cast<std::size_t>(run) * static_cast<std::size_t>(shape.variables) +
                static_cast<std::size_t>(p)) *
               static_cast<std::size_t>(shape.rows());
    }
    // Zahl der hart versagten Strahlen aller Läufe.
    std::size_t hard_failed_count() const;
};

// Batch mit einer Variante pro Lauf am Punkt x₀ des Laufs (Vorlage: eine Variante für alle Läufe
// oder eine pro Lauf, wie make_fd_jacobian_batch; Variante i = Vorlage + apply(x₀_i), bitgleich
// zur Variante x₀ des FD-Batches). Wirft std::invalid_argument bei P = 0, leerem x0, Länge kein
// Vielfaches von P oder falscher Variantenzahl der Vorlage, sonst wie VariableSet::apply.
SystemBatchHost make_adjoint_batch(const VariableSet& set, const SystemBatchHost& base,
                                   std::span<const double> x0);

// Zusatzzeilen der Jacobi-Matrix per zentraler finiter Differenzen auf dem Host (hybrider
// Ansatz): Residuen [runs][E] am Punkt x₀ und Spalten [runs][P][E].
struct ExtraRowsJacobian {
    int runs = 0;
    int variables = 0;
    int rows = 0;                  // E
    std::vector<double> residuals; // [runs][E]
    std::vector<double> jacobian;  // [runs][P][E]
};

// Optionen der FD-Jacobi mit denselben Zusatzresiduen (zentrales Schema).
FdJacobianOptions fd_options_of(const AdjointJacobianOptions& options);

// Zusatzzeilen aus den Zusatzresiduen aller Varianten des FD-Batches `fd` (Ablage [v · E + i],
// z. B. fd_extra_residuals): r₀ aus Variante i·n, Spalten mit fd_derivative nach den Mustern.
ExtraRowsJacobian extra_rows_from_fd(const FdJacobianBatch& fd, std::span<const double> extra,
                                     int extra_count);

// Zusatzzeilen ohne trace-abhängige Residuen (options.trace_extra.count == 0, sonst
// std::invalid_argument), parallel über die Läufe auf options.extra_threads Host-Threads (M5-T3):
// pro Lauf fd_pattern (zentral) und eine Variante als Arbeitsfläche (Kopie der Vorlage des Laufs),
// in die nacheinander x₀ und die 2P FD-Punkte geschrieben werden (VariableSet::apply wie
// write_fd_variants), jeweils ExtraResiduals::evaluate, dann fd_derivative wie
// extra_rows_from_fd. Bitgleich zum Weg über den FD-Batch aller Läufe, aber ohne ihn aufzubauen
// (die Auswertung pro Punkt bleibt). Ohne Zusatzresiduen (E = 0) leer.
ExtraRowsJacobian extra_rows_untraced(const VariableSet& set, const SystemBatchHost& base,
                                      std::span<const double> x0,
                                      const AdjointJacobianOptions& options);

// Zusatzzeilen auf der CPU: ohne trace-abhängige Residuen extra_rows_untraced, sonst
// make_fd_jacobian_batch, bei trace-abhängigen Residuen
// surface_extrema_cpu jeder FD-Variante, fd_extra_residuals, extra_rows_from_fd. Ohne
// Zusatzresiduen (E = 0) leer, ohne Batch. Wirft wie fd_jacobian_cpu (u. a. bei x₀ außerhalb der
// Schranken).
ExtraRowsJacobian extra_rows_cpu(const VariableSet& set, const SystemBatchHost& base,
                                 std::span<const double> x0, const AdjointJacobianOptions& options);

// Schreibt die Zusatzzeilen in die Ablage einer Jacobi-Matrix (Zeilen trace_rows … R − 1 jeder
// Spalte und von r₀). Wirft std::invalid_argument bei unpassenden Größen.
void insert_extra_rows(const ExtraRowsJacobian& extra, AdjointJacobian& out);

// CPU-Referenz (CI ohne GPU): make_adjoint_batch, residuals_cpu für r₀ der Trace-Zeilen, pro
// Strahl Vorwärtsdurchlauf mit AdjointRecord und Rückwärtsdurchlauf (ray_gradient_rows), pro
// Tupel der Kolbenabzug als Projektion
//     ∂r_k/∂x = s_{f,l} √(W_i V_j) (∂OPL_k/∂x − Σ_j W_j ∂OPL_j/∂x / W_ok)
// (Summen über die Strahlen im Kolben in aufsteigender Reihenfolge k wie opd_statistics), die
// Strafzeilen p_k direkt; dann die Zusatzzeilen (extra_rows_cpu). Wirft wie residuals_cpu und
// extra_rows_cpu, std::logic_error, wenn der Trace mit Aufzeichnung einen anderen Status liefert.
AdjointJacobian adjoint_jacobian_cpu(const VariableSet& set, const SystemBatchHost& base,
                                     std::span<const double> x0,
                                     const AdjointJacobianOptions& options = {});

} // namespace mko
