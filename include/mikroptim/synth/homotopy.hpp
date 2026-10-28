#pragma once

// Homotopie-Controller der Systemsynthese (M6-T3, Host; docs/optimierung.md, Abschnitt 7).
//
// Zustand: Systembeschreibung (Variablenwerte, Topologie, Strafgewichte pro Platte) und ein Punkt
// der Homotopie (HomotopyPoint: numerische Apertur, Feldanteil, Spektrumstufe, Faktoren auf das
// Strafgewicht μ und die Glättung ε der Geisterplatten, Faktor auf die Gewichte der einseitigen
// Nebenbedingungen). Ein Schritt erhöht einen Parameter um Δ, optimiert (LM) und bewertet das
// Ergebnis; die Schrittweite wird nach dem Ergebnis vergrößert, beibehalten oder halbiert (dann
// vom letzten guten Zustand neu). Fällt Δ unter Δ_min und meldet die Diagnose der Geisterplatten
// Bedarf, wird ein Einfüge-Operator angewandt (innere Homotopie über die Dicke des neuen
// Elements, danach Penalty-Fortsetzung). Am Ende ε-Fortsetzung, Prüfung des Ziels am realen
// System (alle Geisterkoeffizienten 0) und bei Bedarf μ-Fortsetzung.
//
// Die Steuerlogik (run_homotopy) ist unabhängig von der Optik: Lösen, Diagnose, Einfügen und
// Zielprüfung liefert ein Modell (HomotopyModel). Das optische Modell steht in
// mikroptim/synth/optical_homotopy.hpp; Tests verwenden synthetische Modelle.

#include "mikroptim/model/system.hpp"
#include "mikroptim/optim/ghost_diagnosis.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"

#include <cstdint>
#include <functional>
#include <ostream>
#include <span>
#include <string>
#include <vector>

namespace mko {

// ---------------------------------------------------------------------------------------------
// Homotopie-Parameter und Schrittweitensteuerung
// ---------------------------------------------------------------------------------------------

// Ein Punkt der Homotopie. Die Bedeutung legt das Modell fest; im optischen Modell:
struct HomotopyPoint {
    double aperture = 0.0;        // numerische Apertur NA (kollimiert: Pupillenradius f_soll·NA)
    double field = 1.0;           // Anteil der Feldwinkel der Startdatei, 0 … 1
    int spectrum = 0;             // Spektrumstufe (Index in der Liste des Modells)
    double ghost_weight = 1.0;    // Faktor auf μ aller Strafen der Geisterplatten
    double ghost_smoothing = 1.0; // Faktor auf ε der Strafen der Geisterplatten
    double penalty_scale = 1.0;   // Faktor auf die Gewichte der einseitigen Nebenbedingungen
    double extrema_beta = 0.0;    // β der geglätteten Strahl-Extrema (M11-T1b); 0 = hart

    friend bool operator==(const HomotopyPoint&, const HomotopyPoint&) = default;
};

// Schrittweitensteuerung eines stetigen Parameters (Apertur, Feld).
struct HomotopyStepControl {
    double initial = 0.02; // Δ₀ (auch nach jedem Einfügen)
    double min = 0.002;    // Δ_min: darunter Einfügen bzw. Abbruch
    double max = 0.1;      // Δ_max
    double grow = 2.0;     // Δ ← min(grow·Δ, max) nach einem schnellen Schritt
    double shrink = 0.5;   // Δ ← shrink·Δ nach einer Ablehnung
};

// ---------------------------------------------------------------------------------------------
// Schnittstelle zum Modell
// ---------------------------------------------------------------------------------------------

// Ergebnis einer Optimierung an einem Punkt (ein LM-Aufruf mit Strafverfahren).
struct HomotopySolve {
    LmStatus status = LmStatus::Running;
    int iterations = 0;       // LM-Zyklen (alle Runden des Strafverfahrens)
    double start_merit = 0.0; // F am Startpunkt [Wellen²]
    double merit = 0.0;       // F am Ende [Wellen²] (alle Residuen)
    // Anteile von F für die Bewertung: Strahlresiduen (Wellenfront, beim weichen Versagen mit
    // den Strafresiduen) und L1-Strafe der Geisterplatten [Wellen²]. Die Nebenbedingungen zählen
    // nicht: Ihr Beitrag am Optimum (≈ λ²/(4w) einer aktiven Strafe) hängt von der Wahl der
    // Gewichte ab, nicht von der Güte des Systems.
    double wavefront = 0.0;
    double ghost_penalty = 0.0;
    double rms = 0.0;             // größter RMS-Wellenfrontfehler über Felder und Wellenlängen mit
                                  // Gewicht > 0, mit Geisterplatten wie im LM [Wellen bei λ des
                                  // Tupels]; NaN ohne gültigen Strahl
    double ghost_sum = 0.0;       // Σ|a| aller Geisterkoeffizienten [mm]
    std::int64_t failed_rays = 0; // versagte Strahlen am Ende (weich und hart)
};

// Diagnose (Bedarf an einem neuen Element).
struct HomotopyNeed {
    GhostOperator recommendation = GhostOperator::None;
    std::string text;         // Empfehlung in einem Satz
    GhostDiagnosis diagnosis; // vollständiger Bericht (optisches Modell)
};

// Ein angewandter Einfüge-Operator: Startsysteme der Zweige (Operator mit Dicke 0 eingefügt, die
// Merit des Ausgangssystems unverändert bis auf Rundung bzw. die Störung des Sattelpunkts) und die
// Werte der inneren Homotopie (z. B. Mittendicke des Meniskus).
struct HomotopyInsertion {
    GhostOperator op = GhostOperator::None;
    std::vector<SystemDescription> branches; // mindestens ein Zweig
    std::vector<double> stages;              // innere Homotopie (leer: keine)
    int first_surface = -1;                  // erste neue Dateifläche
    int new_surfaces = 0;
    int plate = -1;   // Quellplatte der Diagnose
    std::string text; // Beschreibung für das Protokoll
    // Daten des Modells für set_stage und finish (optisches Modell: Luftspalt hinter dem
    // Element bei Dicke 0, kleinster Luftspalt, Einträge mechanischer Bedingungen mit
    // Standardauswahl)
    double gap = 0.0;
    double min_gap = 0.0;
    std::vector<int> default_constraints;
};

// Prüfung des Ziels am realen System.
struct HomotopyTupleRms {
    int field = 0;
    int wavelength = 0;
    double field_angle = 0.0; // [deg] (kollimiert) bzw. 0
    double wavelength_nm = 0.0;
    double rms = 0.0; // [Wellen bei λ des Tupels]; NaN ohne gültigen Strahl
    int failed = 0;   // versagte Strahlen (hartes Versagen wie die Auswertung)
};

struct HomotopyGoal {
    bool reached = false;
    double target = 0.0;                  // Ziel [Wellen]
    double max_rms = 0.0;                 // größter RMS der bewerteten Tupel [Wellen]
    std::vector<HomotopyTupleRms> tuples; // Tupel mit Feld- und Spektralgewicht > 0
};

// Modell der Homotopie. `system` ist immer die Basisbeschreibung des Zustands (Werte der
// Variablen, Topologie, Strafgewichte pro Platte); der Punkt bestimmt, was das Modell daraus
// macht (optisch: Pupillenradius, Feldwinkel, Spektrum, Gewichte).
class HomotopyModel {
public:
    virtual ~HomotopyModel() = default;
    // Optimiert ab den Variablenwerten in `system` bei `point` und schreibt das Ergebnis zurück.
    virtual HomotopySolve solve(SystemDescription& system, const HomotopyPoint& point) = 0;
    // Diagnose der Geisterplatten am aktuellen Punkt (ohne Optimierung).
    virtual HomotopyNeed diagnose(const SystemDescription& system, const HomotopyPoint& point) = 0;
    // Wendet den empfohlenen Operator an (need.recommendation ≠ None).
    virtual HomotopyInsertion insert(const SystemDescription& system, const HomotopyPoint& point,
                                     const HomotopyNeed& need) = 0;
    // Setzt den Wert der inneren Homotopie (z. B. Dicke des neuen Elements).
    virtual void set_stage(SystemDescription& system, const HomotopyInsertion& insertion,
                           double value) = 0;
    // Abschluss eines Einfügens (z. B. Nebenbedingungen auf die neuen Flächen ausdehnen).
    virtual void finish_insertion(SystemDescription& system,
                                  const HomotopyInsertion& insertion) = 0;
    // Ziel am realen System prüfen: RMS ≤ target pro Feld und Wellenlänge.
    virtual HomotopyGoal goal(const SystemDescription& system, const HomotopyPoint& point,
                              double target) = 0;
    // Zahl der Glaselemente (Abbruchkriterium max_elements).
    virtual int elements(const SystemDescription& system) const = 0;
    // Hat das System einseitige Nebenbedingungen (Penalty-Fortsetzung nötig)?
    virtual bool one_sided_constraints(const SystemDescription& system) const = 0;

    // ---- Strahlsuche (M6-T4, docs/optimierung.md, Abschnitt 8); Standards ohne Bündelung ----

    // Optimiert mehrere Zustände am selben Punkt. Das Ergebnis jedes Systems muss dasselbe sein
    // wie bei einem einzelnen solve (die Strahlsuche hängt nicht von der Gruppierung ab).
    // `groups` (optional) erhält pro System die Nummer seines gemeinsamen Optimierer-Aufrufs
    // (0, 1, … in der Reihenfolge des ersten Auftretens). Standard: solve nacheinander, jedes
    // System eine eigene Gruppe.
    virtual std::vector<HomotopySolve> solve_all(std::span<SystemDescription* const> systems,
                                                 const HomotopyPoint& point,
                                                 std::vector<int>* groups = nullptr);
    // Varianten eines Einfügens: `need` selbst zuerst, dann höchstens max − 1 weitere
    // Empfehlungen (z. B. andere Lagen oder Gläser der Diagnose). Standard: nur `need`.
    virtual std::vector<HomotopyNeed> insertion_options(const SystemDescription& system,
                                                        const HomotopyPoint& point,
                                                        const HomotopyNeed& need, int max);
    // Abstand zweier Zustände am Punkt für die Deduplizierung (dimensionslos, z. B. größte
    // relative Abweichung der Variablen); +∞, wenn sie nicht vergleichbar sind (andere
    // Topologie). Standard: +∞ (keine Deduplizierung).
    virtual double distance(const SystemDescription& a, const SystemDescription& b,
                            const HomotopyPoint& point);
    // `count` gestörte Kopien des Startsystems `start` eines Zweigs von `insertion` (Multistart),
    // Kopie j = 1 … count mit dem Zufallsstrom variant_seed(seed, j). Standard: keine.
    virtual std::vector<SystemDescription> perturbed_starts(const SystemDescription& start,
                                                            const HomotopyInsertion& insertion,
                                                            const HomotopyPoint& point, int count,
                                                            std::uint64_t seed);
    // Vorfilter der Startsysteme eines Einfügens (Grobbewertung, z. B. FP32): true = behalten.
    // Standard: alle behalten.
    virtual std::vector<bool> prefilter(std::span<const SystemDescription* const> starts,
                                        const HomotopyPoint& point);
};

// ---------------------------------------------------------------------------------------------
// Protokoll
// ---------------------------------------------------------------------------------------------

enum class HomotopyStepKind {
    Start,            // Optimierung am Startpunkt
    Penalty,          // Stufe der Penalty-Fortsetzung (einseitige Bedingungen)
    Aperture,         // Schritt der numerischen Apertur
    Field,            // Schritt des Feldanteils
    Spectrum,         // Spektrumstufe
    Insertion,        // Einfüge-Operator angewandt (ohne Optimierung)
    InsertionStage,   // Stufe der inneren Homotopie eines Zweigs
    Smoothing,        // Stufe der ε-Fortsetzung
    ExtremaSmoothing, // Stufe der β-Fortsetzung der Strahl-Extrema (M11-T1b)
    GhostWeight,      // Stufe der μ-Fortsetzung
    Goal,             // Prüfung des Ziels am realen System (ohne Optimierung)
};

// "start", "penalty", "aperture", "field", "spectrum", "insertion", "insertion_stage",
// "smoothing", "ghost_weight", "goal"
const char* to_string(HomotopyStepKind kind);

// Entscheidung der Schrittweitensteuerung.
enum class HomotopyAction {
    None,   // keine (Fortsetzungsstufen, Einfügen, Ziel)
    Grow,   // angenommen, schnell konvergiert: Δ vergrößert
    Keep,   // angenommen, Δ bleibt
    Shrink, // abgelehnt: Δ verkleinert, weiter ab dem letzten guten Zustand
    Insert, // abgelehnt, Δ < Δ_min mit Bedarf: Einfüge-Operator
    Stop,   // abgelehnt ohne Ausweg
};

// "none", "grow", "keep", "shrink", "insert", "stop"
const char* to_string(HomotopyAction action);

struct HomotopyStep {
    int index = 0; // fortlaufend ab 0
    HomotopyStepKind kind = HomotopyStepKind::Start;
    HomotopyPoint point; // Parameter des Schritts (beim Versuch der Versuchspunkt)
    double delta = 0.0;  // Schrittweite des Parameters (Apertur, Feld), sonst 0
    int branch = -1;     // Zweig der inneren Homotopie (sonst −1)
    double stage = 0.0;  // Wert der inneren Homotopie (sonst 0)
    bool solved = false; // Optimierung ausgeführt (solve gültig)
    HomotopySolve solve;
    bool accepted = false; // Schritt angenommen (Zustand übernommen)
    HomotopyAction action = HomotopyAction::None;
    double next_delta = 0.0;                            // Schrittweite danach (Apertur, Feld)
    GhostOperator recommendation = GhostOperator::None; // Diagnose (Einfügen, Ziel)
    std::string operation;                              // ausgeführter Operator bzw. Bemerkung
    int elements = 0;                                   // Glaselemente nach dem Schritt
    bool goal_checked = false;
    HomotopyGoal goal;    // bei Goal
    double seconds = 0.0; // Laufzeit des Schritts (nicht deterministisch)
};

enum class HomotopyStatus {
    GoalReached,    // Ziel am Endpunkt erreicht
    GoalNotReached, // Endpunkt erreicht, Ziel verfehlt, kein Einfügen möglich
    StepTooSmall,   // Δ < Δ_min ohne Bedarf (kein Ausweg)
    MaxElements,    // Einfügen nötig, aber höchste Elementzahl erreicht
    MaxSteps,       // höchste Zahl von Optimierungen erreicht
    StartRejected,  // Startpunkt nicht annehmbar und kein Einfügen möglich
};

// "goal_reached", "goal_not_reached", "step_too_small", "max_elements", "max_steps",
// "start_rejected"
const char* to_string(HomotopyStatus status);

// ---------------------------------------------------------------------------------------------
// Steuerung
// ---------------------------------------------------------------------------------------------

struct HomotopyOptions {
    // Ziel der Homotopie (Endpunkt); die übrigen Parameter des Startpunkts bleiben.
    double target_aperture = 0.0;   // ≤ 0: Apertur des Startpunkts
    double target_field = -1.0;     // < 0: Feldanteil des Startpunkts
    int target_spectrum = -1;       // < 0: Spektrumstufe des Startpunkts
    double target_rms = 1.0 / 14.0; // Beugungsziel [Wellen] pro Feld und Wellenlänge (Maréchal)

    // Bewertung: Ein Schritt ist annehmbar, wenn F endlich ist, kein Strahl versagt, der Status
    // nicht NumericalError ist und die Bewertungsmerit F_B = wavefront + ghost_penalty
    // (HomotopySolve) höchstens merit_threshold beträgt (≤ 0: (merit_factor · target_rms)²; mit
    // Faktor 1 heißt das: gewichteter RMS-Wellenfrontfehler mit Geisterplatten plus Kosten der
    // Geisterkorrektur höchstens das Beugungsziel).
    double merit_threshold = 0.0;
    double merit_factor = 1.0;
    // Schnell: konvergiert (lm_converged) nach höchstens fast_iterations Zyklen.
    int fast_iterations = 50;
    HomotopyStepControl aperture_step{};
    HomotopyStepControl field_step{0.125, 1.0 / 64.0, 0.5, 2.0, 0.5};

    // Penalty-Fortsetzung der einseitigen Bedingungen am Start und nach jedem Einfügen:
    // Faktoren penalty_factor^{−(K−1)}, …, 1/penalty_factor, 1 (K = penalty_stages).
    int penalty_stages = 3;
    double penalty_factor = 10.0;
    // ε-Fortsetzung am Ende: smoothing_stages Stufen mit Faktor smoothing_factor auf ε.
    int smoothing_stages = 1;
    double smoothing_factor = 0.1;
    // β-Fortsetzung der Strahl-Extrema (M11-T1b, ADR-0013): extrema_beta > 0 startet mit
    // geglätteten Extrema (β [1/mm] für die Höhe, dasselbe β für cos I); am Endpunkt
    // extrema_stages Stufen mit Faktor extrema_factor, danach hart. 0 = aus (Standard).
    double extrema_beta = 0.0;
    int extrema_stages = 2;
    double extrema_factor = 10.0;
    // μ-Fortsetzung am Ende, solange das Ziel verfehlt ist: höchstens ghost_weight_stages
    // Stufen mit Faktor ghost_weight_factor auf μ.
    int ghost_weight_stages = 2;
    double ghost_weight_factor = 10.0;

    // Abbruch
    int max_elements = 6; // höchstens so viele Glaselemente (Einfügen darüber: MaxElements)
    int max_steps = 400;  // höchstens so viele Optimierungen (solve)

    // Rückruf nach jedem Protokolleintrag (z. B. Fortschritt in der CLI)
    std::function<void(const HomotopyStep&)> on_step;
};

struct HomotopyResult {
    HomotopyStatus status = HomotopyStatus::GoalNotReached;
    std::string message;      // Bericht in einem Satz
    SystemDescription system; // Basisbeschreibung am Ende
    HomotopyPoint point;      // Punkt am Ende
    HomotopyPoint target;     // angestrebter Endpunkt (Apertur, Feld, Spektrum)
    std::vector<HomotopyStep> steps;
    HomotopyGoal goal; // letzte Zielprüfung (reached = false ohne Prüfung)
    int elements = 0;  // Glaselemente am Ende
    int start_elements = 0;
    int insertions = 0;           // angewandte Operatoren
    int solves = 0;               // Optimierungen
    double merit_threshold = 0.0; // verwendetes F_max
    double seconds = 0.0;         // Laufzeit (nicht deterministisch)
};

// Läuft die Homotopie vom Startzustand (system, start) zum Ziel aus `options`. Reihenfolge
// (docs/optimierung.md, Abschnitt 7): Start (Penalty-Fortsetzung, bei Bedarf Einfügen),
// Apertur, Feld, Spektrumstufen, Abschluss (ε-Fortsetzung, Ziel, μ-Fortsetzung, bei Bedarf
// Einfügen). Deterministisch, wenn das Modell es ist (außer den Laufzeiten). Wirft
// std::invalid_argument bei ungültigen Optionen; Ausnahmen des Modells werden weitergereicht.
HomotopyResult run_homotopy(HomotopyModel& model, SystemDescription system,
                            const HomotopyPoint& start, const HomotopyOptions& options = {});

// F_max aus den Optionen (merit_threshold bzw. (merit_factor · target_rms)²).
double homotopy_merit_threshold(const HomotopyOptions& options);

// Prüft die Optionen wie run_homotopy (Ziel, Schwelle, Stufen, Faktoren, Schrittweiten); wirft
// std::invalid_argument.
void check_homotopy_options(const HomotopyOptions& options);

// Endpunkt aus Startpunkt und Optionen (target_aperture > 0, target_field ≥ 0,
// target_spectrum ≥ 0 ersetzen die Werte des Starts); wirft std::invalid_argument, wenn die
// Spektrumstufe vor der des Starts liegt oder ein Zielparameter nicht endlich ist.
HomotopyPoint homotopy_target(const HomotopyPoint& start, const HomotopyOptions& options);

// Protokoll als Text (eine Zeile pro Schritt, Zusammenfassung) bzw. JSON
// (Format kSynthFormat, docs/cli.md; nicht endliche Zahlen als null).
inline constexpr const char* kSynthFormat = "mikroptim.synth/1";
void write_homotopy_step_header(std::ostream& out);
void write_homotopy_step_line(const HomotopyStep& step, std::ostream& out);
void write_homotopy_summary(const HomotopyResult& result, std::ostream& out);
std::string homotopy_step_json(const HomotopyStep& step, int indent = -1);
std::string homotopy_result_json(const HomotopyResult& result, int indent = 2);

} // namespace mko
