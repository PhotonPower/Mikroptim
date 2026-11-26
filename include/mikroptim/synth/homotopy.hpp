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
#include "mikroptim/optim/augmented_lagrangian.hpp"
#include "mikroptim/optim/ghost_diagnosis.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"

#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
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
    // Spektrumbreite der stetigen Spektrum-Homotopie (M11-T5f), 0 … 1: Wellenlängen
    // λ_ref + w·(λ_i − λ_ref); 1 = Spektrum der Datei (ohne stetige Homotopie immer 1)
    double spectrum_width = 1.0;

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
    // Augmented Lagrangian (M11-T2): max √w |Δθ| der Fortschreibung nach dem Lauf
    // (HomotopyModel::update_shifts); NaN ohne Augmented Lagrangian
    double shift_change = std::numeric_limits<double>::quiet_NaN();
    // Aperturgrenze (M6-T15): größter RMS-Wellenfrontfehler [Wellen] der voll gewichteten Tupel
    // (Feld- mal Spektralgewicht ≥ 0,5 × größtes Gewicht; Zielspezifikation: 436–656 nm, nicht
    // die schwach gewichteten 405/707 nm), mit Geisterplatten wie `rms`. NaN ohne gültigen Strahl
    // bzw. wenn das Modell ihn nicht liefert; der Controller setzt NaN ohne aperture_rms_limit.
    double rms_main = std::numeric_limits<double>::quiet_NaN();
};

// Diagnose (Bedarf an einem neuen Element).
struct HomotopyNeed {
    GhostOperator recommendation = GhostOperator::None;
    std::string text;         // Empfehlung in einem Satz
    GhostDiagnosis diagnosis; // vollständiger Bericht (optisches Modell)
    // Farbregel (M11-T5c): Operator B an der Linse ab dieser Dateifläche, ohne Geisterdiagnose
    // (−1: Empfehlung aus der Diagnose)
    int split_lens = -1;
    // Weitere Linsen der Farbregel in absteigender Eignung (falls split_lens am Zustand des
    // Einfügens nicht teilbar ist)
    std::vector<int> split_candidates{};
    // Feldregel (M6-T8): Operator A im Luftspalt nach dieser Dateifläche, ohne Geisterdiagnose
    // (−1: Empfehlung aus der Diagnose), weitere Spalte in absteigender Eignung
    int field_gap = -1;
    std::vector<int> field_candidates{};
    // Kittglied-Modul der Farbregel (M6-T11): Null-Kittglied im Luftspalt nach dieser
    // Dateifläche (−1: kein Modul), weitere Spalte in absteigender Eignung
    int module_gap = -1;
    std::vector<int> module_candidates{};
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
    bool doublet = false; // Null-Kittglied der Farbregel (M6-T11): innere Homotopie beider Teile
};

// Kandidat des Rückbaus (M6-T9): ein Element (Glasblock), das versuchsweise über die
// Rückbau-Homotopie entfernt wird.
struct HomotopyRemoval {
    int first_surface = -1; // erste Dateifläche des Blocks
    int elements = 1;       // Glaselemente des Blocks (Kittglied mit k Gläsern: k)
    double score = 0.0;     // Beitrag (Rangfolge: kleinster zuerst; Modell)
    std::string text;       // Beschreibung für das Protokoll
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
    // Augmented Lagrangian (M11-T2): schreibt die Verschiebungen θ der einseitigen Bedingungen
    // am Zustand `system` (nach einer Optimierung bei `point`) fort, θ ← min(max(0, θ + g),
    // max_shift/√w), und liefert max √w |Δθ|. Standard: nichts fortzuschreiben, 0.
    virtual double update_shifts(SystemDescription& system, const HomotopyPoint& point,
                                 double max_shift);

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
    // Prädiktor (M11-T3): Startsystem für den nächsten Punkt aus den beiden letzten angenommenen
    // Zuständen derselben Topologie, x = x_cur + ratio · (x_cur − x_prev) (Sekante; ratio =
    // Verhältnis der Schrittweiten), auf die Schranken projiziert. Leer, wenn nicht möglich
    // (andere Topologie). Standard: leer.
    virtual std::optional<SystemDescription> predict(const SystemDescription& previous,
                                                     const SystemDescription& current,
                                                     const HomotopyPoint& point, double ratio);
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

    // ---- Rückbau (M6-T9, docs/optimierung.md, Abschnitt 7); Standards ohne Rückbau ----

    // Entfernbare Elemente am Zustand, in der Reihenfolge der Versuche (geringster Beitrag
    // zuerst). Standard: keine.
    virtual std::vector<HomotopyRemoval> removal_candidates(const SystemDescription& system,
                                                            const HomotopyPoint& point);
    // Stufe der Rückbau-Homotopie mit Anteil `fraction` ∈ (0, 1] des verbleibenden Wegs zum
    // Null-Element, angewandt auf `current` (Ergebnis der vorigen Stufe bzw. `before`, den
    // Zustand vor der ersten Stufe). Warnungen (z. B. angehobene Schranken) kommen an `warnings`.
    // Wirft std::invalid_argument bzw. FormatError, wenn es nicht geht. Standard: wirft.
    virtual SystemDescription collapse(const SystemDescription& before,
                                       const SystemDescription& current,
                                       const HomotopyRemoval& removal, double fraction,
                                       std::vector<std::string>& warnings);
    // Entfernt das Null-Element nach der letzten Stufe (`collapsed`); `before` wie bei collapse.
    // Standard: wirft std::invalid_argument.
    virtual SystemDescription remove(const SystemDescription& before,
                                     const SystemDescription& collapsed,
                                     const HomotopyRemoval& removal,
                                     std::vector<std::string>& warnings);
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
    SpectrumWidth,    // Schritt der stetigen Spektrumbreite (M11-T5f)
    ExtremaSmoothing, // Stufe der β-Fortsetzung der Strahl-Extrema (M11-T1b)
    GhostWeight,      // Stufe der μ-Fortsetzung
    Goal,             // Prüfung des Ziels am realen System (ohne Optimierung)
    ApertureField,    // Schritt der gemeinsamen Apertur-/Feld-Homotopie (M6-T8)
    RemovalStage,     // Stufe der Rückbau-Homotopie eines Kandidaten (M6-T9)
    Removal,          // Entscheidung über einen Rückbau (ohne Optimierung; M6-T9)
};

// "start", "penalty", "aperture", "field", "spectrum", "insertion", "insertion_stage",
// "smoothing", "spectrum_width", "extrema_smoothing", "ghost_weight", "goal", "aperture_field",
// "removal_stage", "removal"
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
    double target_aperture = 0.0;        // ≤ 0: Apertur des Startpunkts
    double target_field = -1.0;          // < 0: Feldanteil des Startpunkts
    double target_spectrum_width = -1.0; // < 0: Spektrumbreite des Startpunkts (M11-T5f)
    int target_spectrum = -1;            // < 0: Spektrumstufe des Startpunkts
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
    // Vergrößern mit Abstand zur Schwelle (M6-T13): > 0 vergrößert Δ in den stetigen Phasen auch
    // nach einem annehmbaren, langsamen Schritt (Keep), wenn F_B ≤ grow_margin · F_max. Anlass:
    // Die LM-Läufe enden fast immer mit max_iterations (F10), kein Schritt gilt als schnell; die
    // stetige Spektrumbreite kroch so mit Δ_min = 1/1024 weiter, obwohl F_B auf ein Drittel von
    // F_max sank (Bericht M6-T13). 0 (Standard): aus, bitgleich.
    double grow_margin = 0.0;
    HomotopyStepControl aperture_step{};
    HomotopyStepControl field_step{0.125, 1.0 / 64.0, 0.5, 2.0, 0.5};
    // Stetige Spektrumbreite (M11-T5f): kleiner Anfangsschritt, weil die Merit etwa quadratisch
    // mit der Breite wächst (Zielspezifikation: F_B 1,8 schon bei einem Viertel der Breite)
    HomotopyStepControl spectrum_step{1.0 / 32.0, 1.0 / 1024.0, 0.25, 2.0, 0.5};
    // Gemeinsame Apertur-/Feld-Homotopie (M6-T8, Befund F14 aus M6-T6b): Unterscheiden sich
    // Apertur und Feldanteil von Start und Ziel, führt eine Phase beide über einen Parameter
    // t ∈ [0, 1] zum Ziel: NA(t) = NA₀ + t·(NA₁ − NA₀), φ(t) = φ₀ + t^p·(φ₁ − φ₀) mit
    // p = joint_field_exponent (1: Diagonale im (NA, φ)-Raum; p < 1: das Feld läuft voraus).
    // Schrittweite in t (joint_step), sonst wie die Aperturphase. Standard aus (erst Apertur,
    // dann Feld). Nur im Controller ohne Strahlsuche.
    bool joint_aperture_field = false;
    double joint_field_exponent = 1.0;
    HomotopyStepControl joint_step{1.0 / 32.0, 1.0 / 256.0, 0.25, 2.0, 0.5};
    // Einfügen nur mit Verbesserung (M6-T8, Befund F16): In den stetigen Phasen wird ein
    // Einfügen nur übernommen, wenn F_B nach der Penalty-Fortsetzung kleiner ist als F_B des
    // guten Zustands davor; sonst wird es verworfen und die nächste Variante des Modells
    // (insertion_options, höchstens insertion_attempts) versucht. Verbessert keine, endet der
    // Lauf wie ohne Bedarf. Standard aus. Nur im Controller ohne Strahlsuche.
    // Startphase (M6-T11b): Einfügungen, solange der Startpunkt nicht annehmbar ist, werden
    // ebenso geprüft, mit Spielraum: übernommen, wenn F_B danach kleiner ist als davor ODER
    // höchstens F_max (endlich, ohne versagte Strahlen); sonst nächste Variante, verbessert
    // keine, Abbruch StartRejected.
    bool insertion_must_improve = false;
    int insertion_attempts = 4;
    // Abschluss (M6-T13): mit insertion_must_improve auch die Einfügungen nach verfehltem Ziel
    // am Endpunkt nur übernehmen, wenn F_B kleiner wird als am Zustand vor dem Abschluss;
    // verbessert keine Variante, endet der Lauf mit GoalNotReached an diesem Zustand. Anlass:
    // Lauf H2 (Bericht M6-T13) – das Ziel pro Tupel war verfehlt (größter RMS 0,126 Wellen bei
    // F_B < F_max), jedes Einfügen im Abschluss machte das System schlechter (F_B bis 300 Wellen²),
    // bis die Elementzahl erreicht war. Standard aus.
    bool insertion_improve_finish = false;
    // Rückbau (M6-T9, Befund F16): Elemente versuchsweise über die Rückbau-Homotopie entfernen
    // (HomotopyModel::removal_candidates, collapse, remove). Ein Kandidat: prune_stages Stufen
    // mit Anteil 1/K, 1/(K − 1), …, 1, nach jeder Stufe eine Optimierung am Punkt; übernommen,
    // wenn das Ergebnis annehmbar ist mit F_B ≤ prune_merit_factor · F_max und (mit
    // prune_tolerance ε ≥ 0) F_B ≤ (1 + ε) · F_B vor dem Rückbau; sonst Zustand zurück und der
    // nächste Kandidat (höchstens prune_attempts pro Runde). Nach jedem Erfolg neue Runde, bis
    // keiner der Kandidaten einer Runde entfernt werden kann; nie unter prune_min_elements.
    // prune: nach jeder stetigen bzw. Spektrumphase, in der eingefügt wurde, vor dem Abschluss
    // und nach einem Abbruch (am letzten guten Zustand). prune_at_limit (Elementbudget): verlangt
    // ein Einfügen mehr als max_elements, zuerst eine Rückbau-Runde (je Punkt höchstens einmal);
    // entfernt sie etwas, geht der Lauf ohne Einfügen weiter, sonst Abbruch max_elements. Beide
    // Standard aus (ohne sie bitgleich).
    bool prune = false;
    bool prune_at_limit = false;
    int prune_stages = 3;
    int prune_attempts = 4;
    double prune_merit_factor = 1.0; // (0, 1]
    double prune_tolerance = -1.0;   // < 0: aus
    int prune_min_elements = 1;
    // Relative Annahme des Rückbaus (M6-T15, Befund F26): δ = prune_relative ≥ 0 hebt die
    // Grenze auf max(prune_merit_factor · F_max, min((1 + δ) · F_B,vor, F_max)) an; F_B,vor ist
    // F_B am Beginn des Rückbaus (vor dem ersten Entfernen dieses Aufrufs), die Verschlechterung
    // über alle Entfernungen eines Rückbaus bleibt so ≤ (1 + δ). prune_tolerance gilt danach wie
    // bisher. < 0: aus (bitgleich).
    double prune_relative = -1.0;
    // Sperre im Abschluss (M6-T15, F26): nach einem übernommenen Rückbau in diesem Lauf fügt der
    // Abschluss bei verfehltem Ziel nicht ein (Ende GoalNotReached am Zustand nach der
    // μ-Fortsetzung), statt die entfernten Elemente sofort wieder einzufügen (M6-T14b, Z1S).
    bool prune_finish_lock = false;
    // Aperturgrenze (M6-T15): > 0 verlangt in den Schritten der Apertur (auch der gemeinsamen
    // Apertur-/Feld-Homotopie) zusätzlich HomotopySolve::rms_main ≤ aperture_rms_limit [Wellen];
    // sonst gilt der Schritt als nicht erreicht (Δ halbiert, unter Δ_min Einfügen bzw. Ende).
    // ≤ 0: aus (bitgleich). Nur im Controller ohne Strahlsuche.
    double aperture_rms_limit = 0.0;

    // Penalty-Fortsetzung der einseitigen Bedingungen am Start und nach jedem Einfügen:
    // Faktoren penalty_factor^{−(K−1)}, …, 1/penalty_factor, 1 (K = penalty_stages).
    int penalty_stages = 3;
    double penalty_factor = 10.0;
    // Augmented Lagrangian (M11-T2, docs/optimierung.md, Abschnitt 2.6): rounds > 0 ersetzt die
    // Penalty-Fortsetzung durch höchstens `rounds` Optimierungen mit dem festen Faktor
    // al_penalty_scale auf die Gewichte (bis max √w |Δθ| ≤ tolerance); nach jeder Optimierung
    // (auch der stetigen Schritte) werden die Verschiebungen θ fortgeschrieben. Standard aus.
    // Nicht mit der Strahlsuche.
    AugmentedLagrangianOptions augmented_lagrangian{};
    double al_penalty_scale = 1.0;
    // ε-Fortsetzung am Ende: smoothing_stages Stufen mit Faktor smoothing_factor auf ε.
    int smoothing_stages = 1;
    double smoothing_factor = 0.1;
    // β-Fortsetzung der Strahl-Extrema (M11-T1b, ADR-0013): extrema_beta > 0 startet mit
    // geglätteten Extrema (β [1/mm] für die Höhe, dasselbe β für cos I); am Endpunkt
    // extrema_stages Stufen mit Faktor extrema_factor, danach hart. 0 = aus (Standard).
    double extrema_beta = 0.0;
    int extrema_stages = 2;
    // Prädiktor der stetigen Phasen (M11-T3): Start jedes Schritts aus der Sekante der beiden
    // letzten angenommenen Zustände (HomotopyModel::predict); Ablehnung wie ohne Prädiktor (Δ
    // sinkt; eine Wiederholung ohne Prädiktor brachte in der Messung nichts). Nur im Controller
    // ohne Strahlsuche.
    bool predictor = false;
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
    int removal_attempts = 0;     // Rückbau-Versuche (M6-T9)
    int removals = 0;             // übernommene Rückbauten (M6-T9)
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
