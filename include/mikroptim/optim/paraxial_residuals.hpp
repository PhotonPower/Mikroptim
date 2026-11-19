#pragma once

// Paraxiale Nebenbedingungen als Zusatzresiduen (M3-T3): Brennweite, Abgleichlänge,
// Arbeitsabstand, Bildebene im paraxialen Fokus und Petzvalkrümmung (M4-T4), mit dem
// Paraxialtrace aller Varianten eines Batches auf dem Host (parallel); dazu die Verletzung der
// Sinusbedingung (M11-T4), die zusätzlich je Zone einen realen meridionalen Strahl auf dem Host
// verfolgt. Definitionen: docs/optimierung.md, Abschnitte 2.2, 2.3 und 2.7;
// Schema der Datei: docs/dateiformat.md, „Nebenbedingungen“.

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/optim/extra_residuals.hpp"
#include "mikroptim/paraxial/paraxial.hpp"

#include <span>
#include <vector>

namespace mko {

struct SystemDescription;

// ---- Paraxialtrace über Batch-Varianten ------------------------------------------------------

// Paraxialdaten der Variante v bei λ_ref: Krümmungen und Dicken der Variante, Brechzahlen
// paraxial_index, Pupillenradius als Blendenradius; Geisterplatten als dünne Elemente mit
// φ_G = −2 (n_G − n_m) t₂ aus ghost_coeff, ghost_norm_radius, paraxial_ghost_index und dem
// Medium des Spalts n_m = paraxial_index der Geisterfläche (docs/numerik.md, Abschnitte 6, 9).
// Für eine mit make_system_batch gebaute Variante gleich compute_paraxial(system, catalog).
// Wirft std::out_of_range (v) und std::invalid_argument (wie compute_paraxial: nicht endliche
// Werte, Brechzahl <= 0 – z. B. paraxial_index in einem von Hand gebauten Batch nicht gesetzt –,
// Pupillenradius <= 0).
ParaxialData batch_paraxial(const SystemBatchHost& batch, int v);

// Paraxialdaten aller Varianten, verteilt auf `threads` Host-Threads (<= 0: so viele, wie die
// Hardware meldet) in fester, zusammenhängender Aufteilung. Jede Variante wird unabhängig
// gerechnet; das Ergebnis ist bitgleich für jede Threadzahl. Wirft die Ausnahme der kleinsten
// Variante, die eine wirft.
std::vector<ParaxialData> batch_paraxial_all(const SystemBatchHost& batch, int threads = 0);

// Verletzung der Sinusbedingung (OSC, Conrady; conrady_osc in mikroptim/paraxial/
// sine_condition.hpp) der Variante v in der Pupillenzone ρ (0 < ρ ≤ 1): realer meridionaler
// Strahl parallel zur Achse in der Höhe ρ a (a = pupil_radius) durch die Tabellenflächen der
// Variante bei λ_ref (Brechzahlen paraxial_index, Geisterplatten mit paraxial_ghost_index und
// ihrem Dickenprofil, ohne Dezentrierung und Verkippung), `p` = batch_paraxial(batch, v).
// Gleich AberrationReport::osc (ρ = 1) bzw. osc_zone (ρ = 1/√2) für eine Variante ohne
// Geisterplatten. NaN, wenn der Strahl versagt (hart, auch TIR), die Achse nicht schneidet oder
// das System afokal ist. Wirft std::out_of_range (v), std::invalid_argument (Punktquelle, ρ).
double batch_sine_condition(const SystemBatchHost& batch, int v, const ParaxialData& p, double rho);

// ---- Nebenbedingungen --------------------------------------------------------------------

// Eine aufgelöste paraxiale Nebenbedingung. Residuum r = √w · q mit q nach Typ und Modus
// (docs/optimierung.md, Abschnitt 2.2):
//   FocalLength     q = f_s (1 − f_s φ)          (= f_s (f − f_s)/f, linear in φ, mm)
//   WorkingDistance q = t[v][S−1] − w_s
//   ParfocalLength  q = offset + Σ_{j=reference_surface}^{S−1} t[v][j] − L_s
//   ParaxialFocus   q = a (A + t[v][S−1] C / n'_L) (Randstrahlhöhe in der Bildebene, Soll 0)
//   Petzval         q = |κ_P| − κ_max               (κ_P = −n'_L P mit Geisterplatten, 1/mm)
//   SineCondition   q = OSC(ρ = zone)                (batch_sine_condition; 0, wenn NaN)
// Modus Min: q ← min(q, 0) (aktiv nur unterhalb des Solls), Max: q ← max(q, 0). Petzval ist
// nur mit Max definiert: r = √w · max(0, |κ_P| − κ_max), w in mm². SineCondition nur als
// Gleichung (Soll 0), w dimensionslos [Wellen²], ein Eintrag pro Zone.
struct ParaxialConstraint {
    ConstraintType type = ConstraintType::FocalLength;
    ConstraintMode mode = ConstraintMode::Equal;
    double weight = kDefaultConstraintWeight; // w >= 0
    double target = 0.0;       // Sollwert f_s, L_s bzw. w_s [mm]; bei Petzval κ_max [1/mm]
    int reference_surface = 0; // nur ParfocalLength: Fläche im Batch (Tabellenfläche)
    double offset = 0.0;       // nur ParfocalLength: Anlagefläche vor deren Scheitel [mm]
    int item = -1;             // Eintrag in constraints.items, −1 wenn von Hand gebaut
    // Verschiebung θ ≥ 0 des Augmented Lagrangian (M11-T2, nur Modus min/max) in der Einheit
    // von q: Obergrenze √w · max(q + θ, 0), Untergrenze √w · min(q − θ, 0); 0 = reine Strafe.
    double shift = 0.0;
    double zone = 0.0; // nur SineCondition: Pupillenzone ρ ∈ (0, 1]
};

struct ParaxialConstraints {
    std::vector<ParaxialConstraint> items;

    int size() const { return static_cast<int>(items.size()); }
    // Braucht eine der Bedingungen den Paraxialtrace (FocalLength, ParaxialFocus, Petzval,
    // SineCondition)?
    bool needs_trace() const;
};

// Löst die paraxialen Einträge von system.constraints in Dateireihenfolge auf: Sollwerte aus
// `target` bzw. bei Petzval κ_max aus max_curvature (constraint_bound), Standardwerte fehlender
// Felder (Petzval: Modus Max), Referenzfläche der Abgleichlänge von der Dateifläche auf die
// Tabellenfläche abgebildet (make_surface_map), Verschiebung aus ConstraintSpec::shift (Länge 1,
// sonst 0). Ein Eintrag sine_condition wird zu einer Bedingung pro Zone (zones_or_default, in
// Dateireihenfolge, alle mit derselben Nummer `item`). Ohne Abschnitt leer.
// Wirft FormatError wie check_constraints.
ParaxialConstraints make_paraxial_constraints(const SystemDescription& system);

// Prüft die Bedingungen gegen eine Batch-Topologie: Gewicht endlich und >= 0, Sollwert
// endlich (bei FocalLength ≠ 0, bei Petzval ≥ 0, bei SineCondition 0), Modus erlaubt
// (FocalLength, ParaxialFocus und SineCondition nur Equal, Petzval nur Max), Referenzfläche in
// [0, S) und keine Geisterplatte, ParaxialFocus und SineCondition nur bei kollimierter Quelle,
// Zone der SineCondition in (0, 1], Verschiebung endlich und ≥ 0, bei einer Gleichung 0.
// Wirft std::invalid_argument mit der Nummer der Bedingung.
void check_paraxial_constraints(const ParaxialConstraints& constraints,
                                const BatchTopology& topology);

// Abweichung q einer Bedingung für Variante v (Tabelle in docs/optimierung.md, Abschnitt 2.2,
// bei Petzval |κ_P| − κ_max); `p` wie bei paraxial_residual. Prüft nur den Variantenindex.
double paraxial_deviation(const ParaxialConstraint& c, const SystemBatchHost& batch, int v,
                          const ParaxialData& p);

// Verletzung g einer einseitigen Bedingung zur Abweichung q (> 0: verletzt): q bei Modus max
// (auch Petzval), −q bei min; bei einer Gleichung |q|.
double paraxial_violation(const ParaxialConstraint& c, double q);

// Residuum einer Bedingung für Variante v; `p` sind deren Paraxialdaten (batch_paraxial),
// die nur FocalLength, ParaxialFocus, Petzval und SineCondition lesen. Prüft nichts außer dem
// Variantenindex. Mit Verschiebung θ = c.shift ≠ 0 (nur einseitig) ist es √w · max(0, g + θ)
// mit dem Vorzeichen des Modus (docs/optimierung.md, Abschnitt 2.6).
double paraxial_residual(const ParaxialConstraint& c, const SystemBatchHost& batch, int v,
                         const ParaxialData& p);

// Alle Residuen der Variante v nach out (out.size() == constraints.size(), sonst
// std::invalid_argument); höchstens ein Paraxialtrace, nur wenn needs_trace().
void paraxial_residuals(const ParaxialConstraints& constraints, const SystemBatchHost& batch, int v,
                        std::span<double> out);

// Zusatzresiduen für den Residuenvektor (M3-T2, M3-T4): count = constraints.size(), evaluate
// = paraxial_residuals (threadsicher, hält nur eine unveränderliche Kopie der Bedingungen).
// Die Bedingungen werden bei jedem Aufruf gegen die Topologie des Batches geprüft
// (check_paraxial_constraints; billig gegen den Trace).
ExtraResiduals make_paraxial_residuals(ParaxialConstraints constraints);

// Zusatzresiduen der Varianten [first, first + count) (count < 0: bis zum Ende), verteilt auf
// `threads` Host-Threads wie batch_paraxial_all: out[(v − first) · extra.count + i].
// extra.evaluate muss threadsicher sein. Bitgleich für jede Threadzahl. Wirft
// std::invalid_argument bei falscher Länge von out, std::out_of_range bei ungültigem Bereich.
void evaluate_extra_residuals(const ExtraResiduals& extra, const SystemBatchHost& batch,
                              std::span<double> out, int first = 0, int count = -1,
                              int threads = 0);

} // namespace mko
