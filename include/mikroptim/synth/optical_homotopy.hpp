#pragma once

// Optisches Modell des Homotopie-Controllers (M6-T3, Host; docs/optimierung.md, Abschnitt 7):
// realisiert einen Homotopie-Punkt auf einer Systembeschreibung (Pupillenradius, Feldwinkel,
// Spektrum, Gewichte mit Vorhalt), optimiert mit dem gebündelten LM (CPU oder über einen
// austauschbaren Optimierer, z. B. cuda::optimize_lm_gpu), diagnostiziert die Geisterplatten,
// wendet die Einfüge-Operatoren A (Meniskus, innere Homotopie über seine Dicke, zwei Starts am
// Sattelpunkt) und B (Kittglied mit freiem Glas) an und prüft das Ziel am realen System.

#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/optim/insertion.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/synth/homotopy.hpp"

#include <functional>
#include <span>
#include <vector>

namespace mko {

// Optimierer eines Schritts (ein Lauf): Standard optimize_lm_cpu.
using HomotopyOptimizer =
    std::function<LmResult(const VariableSet& set, const SystemBatchHost& base,
                           std::span<const double> x0, const LmOptions& options)>;

struct OpticalHomotopyOptions {
    // LM-Einstellungen jedes Schritts: Zyklen, Quadratur und Strahlversagen
    // (lm.jacobian.quadrature), Threads; jacobian.extra und jacobian.trace_extra setzt das Modell
    // (paraxiale Bedingungen, L1-Strafe, Glas-Hülle, mechanische Bedingungen wie
    // `mikroptim optimize`).
    LmOptions lm{};
    HomotopyOptimizer optimizer; // leer: optimize_lm_cpu
    // Spektrumstufen (leer: nur das Spektrum der Startdatei, Stufe 0). Stufe k ersetzt das
    // Spektrum der Beschreibung.
    std::vector<Spectrum> spectra;

    // Vorhalt der einseitigen Bedingungen (die Strafe erreicht die Grenze nur asymptotisch,
    // docs/optimierung.md, Abschnitte 2.3–2.5): Mindestdicken um mechanical_margin größer,
    // Höchstdicken kleiner, Strahlhöhen-Zugabe (clearance) größer [mm]; Einfallswinkel um
    // angle_margin [deg] kleiner; κ_max der Petzval-Bedingung mal (1 − petzval_margin);
    // Sicherheitsabstand der Glas-Hülle plus glass_hull_margin (normiert).
    double mechanical_margin = 5e-5;
    double angle_margin = 0.01;
    double petzval_margin = 1e-3;
    double glass_hull_margin = 0.0;

    // Operator A: Mittendicke des Meniskus am Ende der inneren Homotopie [mm] und Zahl der
    // Stufen (Dicken t·2^{−(J−1)}, …, t/2, t); der Luftspalt hinter dem Meniskus wird um die Dicke
    // kürzer, aber nicht kürzer als min_air_gap (bzw. die untere Schranke seiner
    // Dickenvariable), sonst wird das System länger.
    double element_thickness = 0.5;
    int element_stages = 5;
    // Nach der inneren Homotopie wird die Mittendicke des Meniskus Variable mit den Schranken
    // [element_thickness, element_max_thickness] (≤ element_thickness: bleibt fest).
    double element_max_thickness = 5.0;
    double min_air_gap = 0.1;
    // Zwei Starts am Sattelpunkt (meniscus_saddle_starts mit (δ, −δ)); δ = 0: ein Zweig mit c₀
    // aus dem Plan.
    double saddle_delta = 0.002;
    MeniscusInitOptions meniscus{}; // Krümmungsbetrag, Schranken der neuen Krümmungsvariablen
    // Operator B
    SplitInitOptions split{};
    // Strafgewicht μ der Quellplatte nach dem Einfügen mal penalty_raise.
    double penalty_raise = 10.0;
    // Mechanische Bedingungen mit Standardauswahl (bzw. "all") gelten nach dem Einfügen auch für
    // die neuen Flächen (sonst bleiben sie bei den alten Flächen, renumber_surfaces).
    bool constrain_new_elements = true;
    GhostDiagnosisOptions diagnosis{};
};

class OpticalHomotopyModel final : public HomotopyModel {
public:
    // `start`: geladenes Startsystem (Kataloge, Basisbeschreibung). Kollimierte Quelle verlangt
    // target.focal_length (Pupillenradius a = f_soll · NA); wirft std::invalid_argument sonst
    // oder bei ungültigen Optionen.
    OpticalHomotopyModel(const LoadedSystem& start, OpticalHomotopyOptions options);

    // Startpunkt aus der Startbeschreibung: NA = a/f_soll (kollimiert) bzw. die NA der
    // Punktquelle, Feldanteil 1, Spektrumstufe 0, alle Faktoren 1.
    HomotopyPoint start_point() const;
    // Basisbeschreibung am Start (Glasvariablen in Buchdahl-Materialien umgewandelt).
    const SystemDescription& start_system() const { return start_; }

    // Beschreibung am Punkt: Pupillenradius f_soll·NA (bzw. NA der Punktquelle) und
    // Normierungsradien der Geisterplatten im selben Verhältnis wie der Pupillenradius zur
    // Basis, Feldwinkel × Feldanteil, Spektrum der Stufe, μ und ε der Geisterplatten mal ihren
    // Faktoren, Gewichte der einseitigen Bedingungen mal penalty_scale, mit `margins` der
    // Vorhalt.
    SystemDescription realize(const SystemDescription& system, const HomotopyPoint& point,
                              bool margins = true) const;
    // Geladenes System (Kataloge des Starts, ohne Dateizugriff).
    LoadedSystem load(const SystemDescription& realized) const;

    HomotopySolve solve(SystemDescription& system, const HomotopyPoint& point) override;
    HomotopyNeed diagnose(const SystemDescription& system, const HomotopyPoint& point) override;
    HomotopyInsertion insert(const SystemDescription& system, const HomotopyPoint& point,
                             const HomotopyNeed& need) override;
    void set_stage(SystemDescription& system, const HomotopyInsertion& insertion,
                   double value) override;
    void finish_insertion(SystemDescription& system, const HomotopyInsertion& insertion) override;
    HomotopyGoal goal(const SystemDescription& system, const HomotopyPoint& point,
                      double target) override;
    int elements(const SystemDescription& system) const override;
    bool one_sided_constraints(const SystemDescription& system) const override;

private:
    LoadedSystem like_; // Kataloge
    SystemDescription start_;
    OpticalHomotopyOptions options_;
    double focal_ = 0.0; // f_soll (kollimiert)
};

// Wahr für Nebenbedingungen, die nur auf einer Seite wirken (Penalty-Fortsetzung, Vorhalt):
// paraxiale Bedingungen mit Modus min/max, Petzval, Glas-Hülle, mechanische Bedingungen.
bool is_one_sided(const ConstraintSpec& spec);

// Zahl der Glaselemente: Dateiflächen vor der letzten, nach denen kein Luftmedium folgt (ein
// Kittglied zählt zwei).
int glass_elements(const SystemDescription& system);

} // namespace mko
