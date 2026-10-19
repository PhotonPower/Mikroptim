#pragma once

// Gebündelter Levenberg-Marquardt (M3-T4, Host). Viele unabhängige Läufe werden gleichzeitig
// iteriert; jeder Lauf hat eigenen Zustand (x, F, Dämpfung μ und ν, Status). Die Steuerlogik
// (Marquardt-Skalierung, Nielsen-Dämpfung, Schranken mit aktiver Menge, Abbruch) ist einmal
// implementiert (levenberg_marquardt); die Rechnungen pro Punkt liefert ein austauschbares
// Backend (LmBackend): CPU (optimize_lm_cpu, hier), GPU (mko::cuda::optimize_lm_gpu in
// mikroptim/cuda/levenberg_marquardt.hpp) oder eine analytische Residuenfunktion (Tests).
// Verfahren, Determinismus und Backends: docs/optimierung.md, Abschnitt 4.

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/optim/extra_residuals.hpp"
#include "mikroptim/optim/fd_jacobian.hpp"
#include "mikroptim/optim/variable_set.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace mko {

// Status eines Laufs.
enum class LmStatus {
    Running, // noch aktiv (nur während der Iteration sichtbar, z. B. im Rückruf)
    // Die Kriterien werden nach jeder neuen Jacobi-Matrix am ungedämpften Modell geprüft
    // (Gauß-Newton-Schritt Δx_GN mit aktiver Menge), damit ein nur wegen großer Dämpfung
    // kleiner Schritt nicht als Konvergenz gilt (docs/optimierung.md, Abschnitt 4).
    ConvergedMerit,    // vorhergesagte Abnahme L(0) − L(Δx_GN) ≤ ftol·F, oder F = 0
    ConvergedStep,     // max_k |Δx_k|/s_k ≤ xtol für Δx_GN oder nach einer Ablehnung für den
                       // gedämpften Schritt
    ConvergedGradient, // projizierter skalierter Gradient max_k |g_k|/√(A_kk·F) ≤ gtol
    MaxIterations,     // max_iterations Zyklen ohne Konvergenz
    DampingLimit,      // μ > max_damping: kein Abstieg mehr gefunden
    NumericalError,    // F am Startpunkt oder A, g nicht endlich
};

// Name für Meldungen ("converged_merit" usw.).
const char* to_string(LmStatus status);
// true für die drei Converged-Zustände.
bool lm_converged(LmStatus status);

// Protokoll eines Zyklus eines Laufs (Rückruf LmOptions::on_iteration, z. B. für das
// Iterationsprotokoll der CLI, M3-T5).
struct LmIterationInfo {
    int run = 0;       // Index des Laufs
    int iteration = 0; // Zyklen des Laufs nach diesem (1, 2, …)
    LmStatus status = LmStatus::Running;
    bool evaluated = false;            // Versuchspunkt ausgewertet (sonst Cholesky gescheitert)
    bool accepted = false;             // Schritt angenommen
    double merit = 0.0;                // F nach dem Zyklus (bei Ablehnung unverändert)
    double trial_merit = 0.0;          // F am Versuchspunkt (NaN ohne Auswertung)
    double gain_ratio = 0.0;           // ρ = (F − F_neu)/(L(0) − L(Δx)) (NaN ohne Auswertung)
    double damping = 0.0;              // μ, mit dem der Schritt gerechnet wurde
    double step_norm = 0.0;            // max_k |Δx_k|/s_k des Versuchsschritts
    std::int64_t failed_rays = 0;      // versagte Strahlen am Versuchspunkt (weich und hart)
    std::int64_t hard_failed_rays = 0; // davon hart versagt (M4-T2)
    std::span<const double> x;         // Punkt nach dem Zyklus (gültig nur während des Rückrufs)
};

using LmIterationFn = std::function<void(const LmIterationInfo&)>;

// Verfahren der Jacobi-Matrix der optischen Backends (M5-T3, docs/optimierung.md, Abschnitt 4,
// „Jacobi-Verfahren im LM“). Beide liefern J in derselben Ablage (JacobianShape); die
// Zusatzzeilen (LmOptions::jacobian.extra, trace_extra) sind in beiden Fällen zentrale finite
// Differenzen (hybrider Ansatz, M5-T2). Die finiten Differenzen bleiben dauerhaft als Orakel
// erhalten (ADR-0009).
enum class JacobianMethod {
    FiniteDifferences, // FD-Batch mit 2P + 1 (bzw. P + 1) Varianten pro Lauf (M3-T2)
    Adjoint,           // Rückwärtsdurchlauf pro Strahl (M5-T2); jacobian.scheme gilt hier nicht
};

// Name für Meldungen und die CLI ("fd", "adjoint").
const char* to_string(JacobianMethod method);
// Umkehrung von to_string; leer bei unbekanntem Namen.
std::optional<JacobianMethod> jacobian_method_from_string(std::string_view name);

// Einstellungen. Die Standardwerte sind die des Plans (docs/optimierung.md, Abschnitt 4).
struct LmOptions {
    int max_iterations = 100;      // Zyklen pro Lauf (Versuchsschritte inkl. verworfener)
    double ftol = 1e-10;           // relative Abnahme von F, die das Modell noch vorhersagt
    double xtol = 1e-10;           // skalierte Schrittnorm, s_k = max(|x_k|, x_typ,k)
    double gtol = 1e-10;           // projizierter skalierter Gradient (Kosinus zwischen r und J_k)
    double initial_damping = 1e-3; // μ₀, relativ zu D = diag(JᵀJ) (dimensionslos)
    double max_damping = 1e16;     // μ darüber: Abbruch DampingLimit
    // Untergrenze der Marquardt-Skalierung: D_kk = max(A_kk, diag_floor·max_j(A_jj s_j²)/s_k²),
    // damit Spalten ohne Einfluss (A_kk = 0) das gedämpfte System nicht singulär machen.
    double diag_floor = 1e-12;
    // Schritt verwerfen, wenn am Versuchspunkt mehr Strahlen hart versagen als bei x (hart
    // versagte Strahlen sind maskiert, r = 0, und könnten F senken). Weich versagte Strahlen
    // (M4-T2) zählen nicht: Ihre Residuen und Strafen sind stetig und stehen in F.
    bool reject_new_failures = true;
    // Host-Threads für die Schritte (Cholesky pro Lauf) und das CPU-Backend; ≤ 0: nach der
    // Arbeit des Aufrufs (lm_host_threads). Das Ergebnis ist für jede Zahl bitgleich (keine
    // Reduktion über Läufe oder Varianten).
    int threads = 0;
    // Jacobi-Matrix und Residuen der optischen Backends: Pupillenquadratur und Strahlversagen
    // (jacobian.quadrature, Standard weich mit w_fail = kDefaultFailWeight, M4-T2), FD-Schema,
    // Zusatzresiduen (z. B. make_paraxial_residuals), Host-Threads der Zusatzresiduen
    // (extra_threads ≤ 0: nach der Zahl der Varianten, lm_host_threads).
    FdJacobianOptions jacobian{};
    // Verfahren der Jacobi-Matrix (M5-T3): finite Differenzen (Standard) oder adjungiert. Mit
    // Adjoint gelten aus `jacobian` die Quadratur, die Zusatzresiduen und extra_threads; das
    // Schema (scheme) nur für die FD-Jacobi.
    JacobianMethod jacobian_method = JacobianMethod::FiniteDifferences;
    // Äußeres Strafverfahren der optischen Treiber (optimize_lm_cpu, cuda::optimize_lm_gpu,
    // cuda::optimize_lm_multi_gpu;
    // weiches Versagen, M4-T2, docs/optimierung.md, Abschnitt 4): Endet ein Lauf mit versagten
    // Strahlen, wird er vom Endpunkt aus mit w_fail · fail_weight_factor neu gestartet, solange
    // Strahlen versagen und das Gewicht max_fail_weight nicht übersteigt. fail_weight_factor <= 1,
    // w_fail = 0 oder hartes Versagen: ein Durchgang wie bisher.
    double fail_weight_factor = 10.0;
    double max_fail_weight = 1e14;
    LmIterationFn on_iteration; // optional, nach jedem Zyklus pro Lauf in Laufreihenfolge
};

// Ergebnis eines Laufs.
struct LmRunResult {
    std::vector<double> x;             // Endpunkt (zulässig)
    double merit = 0.0;                // F(x) = ‖r(x)‖² einschließlich Straf- und Zusatzresiduen
    std::int64_t failed_rays = 0;      // versagte Strahlen bei x (weich und hart)
    std::int64_t hard_failed_rays = 0; // davon hart versagt
    LmStatus status = LmStatus::Running;
    int iterations = 0;          // Zyklen (Versuchsschritte inkl. verworfener)
    int accepted = 0;            // angenommene Schritte
    int jacobians = 0;           // Auswertungen der Jacobi-Matrix
    double damping = 0.0;        // μ am Ende
    double gradient_norm = 0.0;  // projizierter skalierter Gradient der letzten Jacobi-Matrix
    double model_decrease = 0.0; // (L(0) − L(Δx_GN))/F der letzten Jacobi-Matrix (NaN, wenn
                                 // das Modell nicht lösbar war oder keine Jacobi-Matrix vorlag)
    std::vector<double> history; // history[0] = F(x₀), history[k] = F nach Zyklus k; bei
                                 // mehreren Runden des Strafverfahrens aneinandergehängt (jede
                                 // Runde beginnt mit F am Endpunkt unter dem neuen w_fail)
    double fail_weight = 0.0;    // w_fail der letzten Runde (optische Treiber, weicher Modus)
    int penalty_rounds = 1;      // Runden des Strafverfahrens (1: kein Neustart)
};

struct LmResult {
    std::vector<LmRunResult> runs;
    int cycles = 0; // Zyklen der Schleife über alle Läufe
};

// Rechnungen pro Punkt für die Steuerlogik. Läufe werden über ihren Index (0 … runs − 1)
// angesprochen; `runs` ist aufsteigend, `x` enthält die Punkte hintereinander (x[j·P …] für
// runs[j]). Die Ergebnisse eines Laufs dürfen nur von seinem Index und seinem Punkt abhängen,
// nicht von den übrigen Läufen im Aufruf (Bitgleichheit Gruppe gegen Einzellauf).
class LmBackend {
public:
    virtual ~LmBackend() = default;
    // A_j = JᵀJ (P × P spaltenweise, a[j·P² + q·P + p]) und g_j = Jᵀr (g[j·P + p]) an x_j.
    virtual void normal_equations(std::span<const int> runs, std::span<const double> x,
                                  std::span<double> a, std::span<double> g) = 0;
    // F_j = ‖r(x_j)‖², versagte Strahlen (Status ≠ Ok) und davon hart versagte an x_j (im
    // harten Modus gleich; reject_new_failures vergleicht die hart versagten, M4-T2).
    virtual void merit(std::span<const int> runs, std::span<const double> x,
                       std::span<double> merit, std::span<std::int64_t> failed,
                       std::span<std::int64_t> hard_failed) = 0;
};

// Schranken und typische Größen (Länge P; ±∞ für offene Schranken, typical > 0).
struct LmBounds {
    std::span<const double> lower;
    std::span<const double> upper;
    std::span<const double> typical;
};

// Steuerlogik für runs = x0.size()/P Läufe mit Startpunkten x0 [runs·P] (zulässig, endlich).
// Wirft std::invalid_argument bei ungültigen Argumenten oder Einstellungen; Ausnahmen des
// Backends werden weitergereicht.
LmResult levenberg_marquardt(LmBackend& backend, int variables, const LmBounds& bounds,
                             std::span<const double> x0, const LmOptions& options = {});

// ---- Optische Probleme -------------------------------------------------------------------

// Zahl der Host-Threads für `work` Arbeitseinheiten: `requested` > 0 wie angegeben, sonst
// ⌊work/min_work_per_thread⌋, mindestens 1, höchstens hardware_concurrency. Ein Thread kostet
// beim Start bis zu einigen Millisekunden (gemessen unter Windows); kleine Aufgaben (wenige
// Varianten pro Zyklus) laufen daher im aufrufenden Thread. Die Ergebnisse hängen nie von der
// Threadzahl ab.
int lm_host_threads(int requested, std::int64_t work, std::int64_t min_work_per_thread);

// Arbeitseinheiten pro Thread: Zusatzresiduen (Varianten), Strahlen × Flächen im CPU-Backend,
// Gleitkommaoperationen der Schritte (Läufe · P³).
inline constexpr std::int64_t kLmExtraPerThread = 4096;
inline constexpr std::int64_t kLmRaySurfacesPerThread = 200000;
inline constexpr std::int64_t kLmStepFlopsPerThread = 2000000;

// Prüft, dass `base` eine Variante (für alle Läufe) oder genau eine pro Lauf hat
// (runs = x0_size/P, x0_size ein positives Vielfaches von P); wirft std::invalid_argument.
void check_lm_base(const SystemBatchHost& base, int variables, std::size_t x0_size);

// Vorlage pro Lauf: `base` mit einer Variante (für alle Läufe) bleibt unverändert; sonst muss
// `base` eine Variante pro Lauf haben, und der Batch enthält die Varianten runs[j] (gemeinsame
// Felder aus `base`). Wirft std::invalid_argument bzw. std::out_of_range.
SystemBatchHost lm_template_batch(const SystemBatchHost& base, std::span<const int> runs);
// Batch mit einer Variante pro Lauf: Kopie der Vorlage des Laufs, dann set.apply(x_j).
SystemBatchHost lm_point_batch(const VariableSet& set, const SystemBatchHost& base,
                               std::span<const int> runs, std::span<const double> x);
// Σ_i e_i² der Zusatzresiduen jeder Variante (aufsteigend in i; leer bei count = 0), auf
// `threads` Host-Threads (evaluate_extra_residuals).
std::vector<double> extra_merit_values(const ExtraResiduals& extra, const SystemBatchHost& batch,
                                       int threads = 0);

// CPU-Backend (CI ohne GPU und Referenz): J wie fd_jacobian_cpu (Varianten auf
// options.threads Host-Threads, bitgleich) bzw. mit JacobianMethod::Adjoint wie
// adjoint_jacobian_cpu (Tupel auf options.threads Host-Threads, bitgleich), A und g mit
// normal_equations_host, F wie merit_cpu plus Σ e² der Zusatzresiduen. `set`, `base` und `options`
// müssen leben, solange das Backend benutzt wird.
class CpuLmBackend final : public LmBackend {
public:
    CpuLmBackend(const VariableSet& set, const SystemBatchHost& base, const LmOptions& options);
    void normal_equations(std::span<const int> runs, std::span<const double> x, std::span<double> a,
                          std::span<double> g) override;
    void merit(std::span<const int> runs, std::span<const double> x, std::span<double> merit,
               std::span<std::int64_t> failed, std::span<std::int64_t> hard_failed) override;

private:
    const VariableSet& set_;
    const SystemBatchHost& base_;
    const LmOptions& options_;
    ResidualLayout layout_;
};

// Erzeugt das Backend einer Runde für die Einstellungen `options` und die Vorlage `base` (eine
// Variante für alle Läufe der Runde oder eine pro Lauf). Beide leben bis zum Ende der Runde.
using LmBackendFactory = std::function<std::unique_ptr<LmBackend>(const LmOptions& options,
                                                                  const SystemBatchHost& base)>;

// Gemeinsamer Ablauf der optischen Treiber (CPU, eine GPU, mehrere GPUs): levenberg_marquardt
// mit dem Backend der Fabrik,
// danach im weichen Modus das äußere Strafverfahren (LmOptions::fail_weight_factor): Läufe mit
// versagten Strahlen (außer NumericalError) starten an ihrem Endpunkt mit dem zehnfachen
// (fail_weight_factor-fachen) w_fail neu, gemeinsam in einem Aufruf. Ergebnis pro Lauf: x,
// merit, versagte Strahlen, Status, μ, Gradientenmaß und Modellabnahme der letzten Runde,
// Zähler summiert, history aneinandergehängt; on_iteration erhält den ursprünglichen
// Laufindex und fortlaufende Zyklen. Die Entscheidung pro Lauf hängt nur von seinem Ergebnis ab
// (Bitgleichheit Gruppe gegen Einzellauf bleibt).
LmResult optimize_lm_rounds(const LmBackendFactory& make_backend, const VariableSet& set,
                            const SystemBatchHost& base, std::span<const double> x0,
                            const LmOptions& options);

// Optimiert runs = x0.size()/P Läufe auf der CPU (optimize_lm_rounds mit CpuLmBackend). `base`
// hat eine Variante (für alle Läufe) oder eine pro Lauf; die Zusatzresiduen kommen aus
// options.jacobian.extra.
LmResult optimize_lm_cpu(const VariableSet& set, const SystemBatchHost& base,
                         std::span<const double> x0, const LmOptions& options = {});

} // namespace mko
