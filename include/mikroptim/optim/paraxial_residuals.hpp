#pragma once

// Paraxiale Nebenbedingungen als Zusatzresiduen (M3-T3): Brennweite, Abgleichlänge,
// Arbeitsabstand, Bildebene im paraxialen Fokus und Petzvalkrümmung (M4-T4), mit dem
// Paraxialtrace aller Varianten eines Batches auf dem Host (parallel). Definitionen:
// docs/optimierung.md, Abschnitte 2.2 und 2.3;
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

// ---- Nebenbedingungen --------------------------------------------------------------------

// Eine aufgelöste paraxiale Nebenbedingung. Residuum r = √w · q mit q nach Typ und Modus
// (docs/optimierung.md, Abschnitt 2.2):
//   FocalLength     q = f_s (1 − f_s φ)          (= f_s (f − f_s)/f, linear in φ, mm)
//   WorkingDistance q = t[v][S−1] − w_s
//   ParfocalLength  q = offset + Σ_{j=reference_surface}^{S−1} t[v][j] − L_s
//   ParaxialFocus   q = a (A + t[v][S−1] C / n'_L) (Randstrahlhöhe in der Bildebene, Soll 0)
//   Petzval         q = |κ_P| − κ_max               (κ_P = −n'_L P mit Geisterplatten, 1/mm)
// Modus Min: q ← min(q, 0) (aktiv nur unterhalb des Solls), Max: q ← max(q, 0). Petzval ist
// nur mit Max definiert: r = √w · max(0, |κ_P| − κ_max), w in mm².
struct ParaxialConstraint {
    ConstraintType type = ConstraintType::FocalLength;
    ConstraintMode mode = ConstraintMode::Equal;
    double weight = kDefaultConstraintWeight; // w >= 0
    double target = 0.0;       // Sollwert f_s, L_s bzw. w_s [mm]; bei Petzval κ_max [1/mm]
    int reference_surface = 0; // nur ParfocalLength: Fläche im Batch (Tabellenfläche)
    double offset = 0.0;       // nur ParfocalLength: Anlagefläche vor deren Scheitel [mm]
    int item = -1;             // Eintrag in constraints.items, −1 wenn von Hand gebaut
};

struct ParaxialConstraints {
    std::vector<ParaxialConstraint> items;

    int size() const { return static_cast<int>(items.size()); }
    // Braucht eine der Bedingungen den Paraxialtrace (FocalLength, ParaxialFocus, Petzval)?
    bool needs_trace() const;
};

// Löst die paraxialen Einträge von system.constraints in Dateireihenfolge auf: Sollwerte aus
// `target` bzw. bei Petzval κ_max aus max_curvature (constraint_bound), Standardwerte fehlender
// Felder (Petzval: Modus Max), Referenzfläche der Abgleichlänge von der Dateifläche auf die
// Tabellenfläche abgebildet (make_surface_map). Ohne Abschnitt leer.
// Wirft FormatError wie check_constraints.
ParaxialConstraints make_paraxial_constraints(const SystemDescription& system);

// Prüft die Bedingungen gegen eine Batch-Topologie: Gewicht endlich und >= 0, Sollwert
// endlich (bei FocalLength ≠ 0, bei Petzval ≥ 0), Modus erlaubt (FocalLength und ParaxialFocus
// nur Equal, Petzval nur Max), Referenzfläche in [0, S) und keine Geisterplatte, ParaxialFocus
// nur bei kollimierter Quelle.
// Wirft std::invalid_argument mit der Nummer der Bedingung.
void check_paraxial_constraints(const ParaxialConstraints& constraints,
                                const BatchTopology& topology);

// Residuum einer Bedingung für Variante v; `p` sind deren Paraxialdaten (batch_paraxial),
// die nur FocalLength, ParaxialFocus und Petzval lesen. Prüft nichts außer dem Variantenindex.
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
