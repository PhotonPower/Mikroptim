#pragma once

// Kandidatenverwaltung der Systemsynthese (M6-T4, Host; docs/optimierung.md, Abschnitt 8):
// Strahlsuche (beam search) über die Zweige des Homotopie-Controllers.
//
// Der Strahl ist eine Menge von höchstens B Kandidaten (Systembeschreibung, letzte Optimierung,
// Elementzahl, Herkunft im Stammbaum), die im Gleichschritt an einem gemeinsamen Homotopie-Punkt
// stehen: Jeder Schritt (Apertur, Feld, Spektrum, Fortsetzungen) wird für alle Kandidaten am
// selben Punkt gerechnet, die Optimierungen gemeinsam über HomotopyModel::solve_all (das
// optische Modell bündelt Kandidaten gleicher Topologie in einem LM-Aufruf). Verzweigt wird beim
// Einfügen: Jeder Kandidat, der nicht weiterkommt und Bedarf hat, erzeugt Kinder (Varianten des
// Operators aus der Diagnose × Startpunkte am Sattelpunkt × gestörte Starts); nach der inneren
// Homotopie aller Kinder werden die besten B aus Kindern und übrigen Kandidaten behalten
// (Rangfolge, Deduplizierung). Mit B = 1 und ohne Varianten trifft die Strahlsuche dieselben
// Entscheidungen wie run_homotopy (bis auf die Reihenfolge der Stufen im Protokoll).

#include "mikroptim/synth/homotopy.hpp"

#include <cstdint>
#include <functional>
#include <ostream>
#include <string>
#include <vector>

namespace mko {

// ---------------------------------------------------------------------------------------------
// Einstellungen
// ---------------------------------------------------------------------------------------------

struct BeamOptions {
    // Ziel, Schwelle, Schrittweiten, Fortsetzungen und Abbruch wie beim einzweigigen Controller.
    // max_steps zählt die Optimierungen aller Kandidaten; homotopy.on_step wird nicht benutzt
    // (stattdessen on_step unten).
    HomotopyOptions homotopy{};
    int beam_width = 2; // B: höchstens so viele Kandidaten im Strahl
    // Varianten eines Einfügens aus der Diagnose (HomotopyModel::insertion_options: weitere
    // Lagen, andere Gläser); 1: nur die Empfehlung.
    int alternatives = 1;
    // Zusätzliche gestörte Starts pro Zweig (HomotopyModel::perturbed_starts) und Startwert ihrer
    // Zufallszahlen; Zweig n (fortlaufend über den Lauf) erhält variant_seed(seed, n).
    int perturbed_starts = 0;
    std::uint64_t seed = 0;
    // Vorfilter der Starts eines Einfügens (HomotopyModel::prefilter, optisch: FP32-Vorfilter).
    bool prefilter = false;
    // Duplikat: Abstand (HomotopyModel::distance) zu einem besser platzierten Kandidaten
    // höchstens dedup_tolerance; < 0 schaltet die Deduplizierung ab.
    double dedup_tolerance = 1e-4;
};

// ---------------------------------------------------------------------------------------------
// Stammbaum und Protokoll
// ---------------------------------------------------------------------------------------------

enum class BeamNodeStatus {
    Kept,      // im Strahl (am Ende: Ziel nicht geprüft oder verfehlt)
    Finished,  // im Strahl, Ziel am realen System erreicht
    Discarded, // verworfen (Rang, Duplikat, Vorfilter, kein Ausweg)
    Expanded,  // durch Einfügen in Kinder aufgeteilt
};

// "kept", "finished", "discarded", "expanded"
const char* to_string(BeamNodeStatus status);

// Ein Kandidat im Stammbaum. Kandidaten entstehen nur an Verzweigungen (Start, Einfügen); die
// Schritte dazwischen ändern seinen Zustand (point, merit, …), nicht seine Nummer.
struct BeamNode {
    int id = 0;
    int parent = -1; // −1: Wurzel (Startsystem)
    std::vector<int> children;
    std::string operation; // Herkunft: "start" bzw. Operator, Variante, Zweig, Start
    GhostOperator op = GhostOperator::None;
    int option = -1;              // Variante des Einfügens (Index in insertion_options)
    int branch = -1;              // Zweig des Operators (Sattelpunkt)
    int start = 0;                // 0: Startpunkt des Zweigs, j ≥ 1: gestörter Start j
    HomotopyPoint created;        // Punkt der Entstehung
    int created_step = -1;        // Protokolleintrag der Entstehung (Einfügen), −1 bei der Wurzel
    HomotopyPoint point;          // Punkt der letzten Optimierung des Kandidaten
    bool solved = false;          // mindestens eine Optimierung
    double merit = 0.0;           // F der letzten Optimierung [Wellen²]
    double evaluation = 0.0;      // F_B = Wellenfront + L1-Strafe [Wellen²]
    double rms = 0.0;             // größter RMS mit Geisterplatten [Wellen]
    bool acceptable = false;      // letzte Optimierung annehmbar
    std::int64_t failed_rays = 0; // versagte Strahlen der letzten Optimierung
    int elements = 0;
    int solves = 0; // Optimierungen des Kandidaten
    bool goal_checked = false;
    HomotopyGoal goal; // letzte Zielprüfung
    BeamNodeStatus status = BeamNodeStatus::Kept;
    std::string reason;  // Grund des Status (verworfen, expandiert, fertig)
    int final_step = -1; // Protokolleintrag der Statusänderung (−1: aktiv bis zum Ende)
};

// Ein Protokolleintrag: Schritt eines Kandidaten (Felder wie beim einzweigigen Controller) mit
// Kandidat und gemeinsamem Optimierer-Aufruf.
struct BeamStep {
    int candidate = -1; // Kandidat (BeamNode::id)
    int group = -1;     // Nummer des gemeinsamen Optimierer-Aufrufs (fortlaufend), −1 ohne
    HomotopyStep step;  // step.index = Nummer des Eintrags im Protokoll der Strahlsuche
};

struct BeamResult {
    HomotopyStatus status = HomotopyStatus::GoalNotReached;
    std::string message;
    int best = -1;            // Kandidat des Ergebnisses
    SystemDescription system; // Basisbeschreibung des besten Kandidaten am Ende
    HomotopyPoint point;      // gemeinsamer Punkt am Ende
    HomotopyPoint target;
    HomotopyGoal goal; // letzte Zielprüfung des besten Kandidaten
    int elements = 0;
    int start_elements = 0;
    int insertions = 0; // angewandte Operatoren (alle Kandidaten und Varianten)
    int solves = 0;     // Optimierungen (alle Kandidaten)
    int groups = 0;     // Optimierer-Aufrufe (gemeinsame Gruppen)
    double merit_threshold = 0.0;
    int beam_width = 0;
    std::uint64_t seed = 0;
    std::vector<int> beam; // Kandidaten im Strahl am Ende, nach Rang
    std::vector<BeamStep> steps;
    std::vector<BeamNode> nodes; // Stammbaum, Index = id
    double seconds = 0.0;        // Laufzeit (nicht deterministisch)
};

// Läuft die Strahlsuche vom Startzustand (system, start) zum Ziel aus options.homotopy (Phasen
// wie run_homotopy: Start, Apertur, Feld, Spektrum, Abschluss). Deterministisch, wenn das Modell
// es ist: gleiche Eingaben und gleicher Startwert ergeben dasselbe Protokoll und dieselben
// Systeme (außer den Laufzeiten); die Gruppierung der Optimierungen ändert kein Ergebnis.
// Wirft std::invalid_argument bei ungültigen Optionen; Ausnahmen des Modells werden
// weitergereicht. on_step wird nach jedem Protokolleintrag gerufen.
BeamResult run_beam_search(HomotopyModel& model, SystemDescription system,
                           const HomotopyPoint& start, const BeamOptions& options = {},
                           const std::function<void(const BeamStep&)>& on_step = {});

// Rangfolge zweier Kandidaten am selben Punkt (true: a vor b): endliche Merit, annehmbar vor
// nicht annehmbar, unter annehmbaren weniger Glaselemente und dann kleineres F, unter nicht
// annehmbaren weniger versagte Strahlen und dann kleineres F, zuletzt die kleinere Nummer.
bool beam_rank_before(const BeamNode& a, const BeamNode& b);

// Protokoll als Text: Kopf und eine Zeile pro Eintrag (Kandidat, Gruppe, dann wie
// write_homotopy_step_line), Zusammenfassung, Stammbaum als eingerückter Baum.
void write_beam_step_header(std::ostream& out);
void write_beam_step_line(const BeamStep& step, std::ostream& out);
void write_beam_summary(const BeamResult& result, std::ostream& out);
void write_beam_tree(const BeamResult& result, std::ostream& out);
// Stammbaum als Graphviz-DOT (digraph, ein Knoten pro Kandidat, Kanten Eltern → Kind).
void write_beam_dot(const BeamResult& result, std::ostream& out);
// JSON-Objekt des Ergebnisses (Felder wie homotopy_result_json, dazu Strahl, Stammbaum
// `candidates`, Kandidat und Gruppe pro Schritt; nicht endliche Zahlen als null).
std::string beam_result_json(const BeamResult& result, int indent = 2);

} // namespace mko
