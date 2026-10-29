#pragma once

// Optisches Modell des Homotopie-Controllers (M6-T3, Host; docs/optimierung.md, Abschnitt 7):
// realisiert einen Homotopie-Punkt auf einer Systembeschreibung (Pupillenradius, Feldwinkel,
// Spektrum, Gewichte mit Vorhalt), optimiert mit dem gebündelten LM (CPU oder über einen
// austauschbaren Optimierer, z. B. cuda::optimize_lm_gpu), diagnostiziert die Geisterplatten,
// wendet die Einfüge-Operatoren A (Meniskus, innere Homotopie über seine Dicke, zwei Starts am
// Sattelpunkt) und B (Kittglied mit freiem Glas) an und prüft das Ziel am realen System.

#include "mikroptim/batch/batch_prefilter.hpp"
#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/optim/insertion.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/synth/homotopy.hpp"

#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace mko {

// Optimierer eines Schritts (ein Lauf): Standard optimize_lm_cpu.
using HomotopyOptimizer =
    std::function<LmResult(const VariableSet& set, const SystemBatchHost& base,
                           std::span<const double> x0, const LmOptions& options)>;

// FP32-Vorfilter der Strahlsuche (M7-T2): Grobbewertung aller Varianten von `batch` mit der
// Quadratur `quadrature`, den Kriterien `options` und den Zusatzresiduen `extra_merit` (FP64,
// eine Zahl pro Variante). Standard prefilter_cpu; GPU z. B. TraceEngine::prefilter_variants.
using HomotopyPrefilterFn = std::function<PrefilterResult(
    const SystemBatchHost& batch, const ResidualSettings& quadrature,
    const PrefilterOptions& options, std::span<const double> extra_merit)>;

// Kriterien des Vorfilters der Strahlsuche: mehr als die Hälfte der Strahlen versagt oder Merit
// mehr als das Zehnfache der besten Variante desselben Einfügens.
inline PrefilterOptions homotopy_prefilter_defaults() {
    PrefilterOptions o;
    o.criteria.max_failed_fraction = 0.5;
    o.criteria.max_merit_factor = 10.0;
    return o;
}

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
    // Farbregel (M11-T5c): Dominiert die Farbe, empfiehlt diagnose Operator B an der Linse mit
    // dem größten Beitrag zum Farblängsfehler ΔW020 (Aberrationsbericht, Katalogglas); Zielglas
    // und Kittkrümmung aus der Glasvorauswahl (split_for_color). Vor der Geisterdiagnose;
    // Standard aus. Farbe dominiert, wenn der größte RMS über Felder und Wellenlängen (getracte
    // Wellenfront am Punkt) mindestens color_min_rms Wellen ist und
    //   (a) der RMS einer Wellenlänge mindestens color_ratio-mal der RMS bei λ_ref desselben
    //       Feldes ist (liegt λ_ref nicht im Spektrum, monochromatisch bei λ_ref nachgerechnet)
    //       (gut bei hoher NA, wo Seidel-Sphärik durch höhere Ordnungen ausgeglichen ist), oder
    //   (b) |ΔW020| ≥ color_seidel und |ΔW020| ≥ |W040| (Seidel-Summen, gut bei kleiner NA).
    bool color_rule = false;
    double color_ratio = 2.0;
    double color_seidel = 0.25;
    double color_min_rms = 0.05;
    // Strafgewicht μ der Quellplatte nach dem Einfügen mal penalty_raise.
    double penalty_raise = 10.0;
    // Mechanische Bedingungen mit Standardauswahl (bzw. "all") gelten nach dem Einfügen auch für
    // die neuen Flächen (sonst bleiben sie bei den alten Flächen, renumber_surfaces).
    bool constrain_new_elements = true;
    GhostDiagnosisOptions diagnosis{};

    // Strahlsuche (M6-T4, docs/optimierung.md, Abschnitt 8)
    // Kandidaten gleicher Topologie gemeinsam in einem LM-Aufruf (solve_all); false: jeder
    // Kandidat einzeln (gleiche Ergebnisse, nur zum Vergleich).
    bool group_runs = true;
    // Gestörte Starts: Krümmungen der neuen Flächen ± start_perturbation [1/mm] (gleichverteilt).
    double start_perturbation = 0.01;
    // Vorfilter der Starts eines Einfügens (leer: prefilter_cpu).
    PrefilterOptions prefilter = homotopy_prefilter_defaults();
    HomotopyPrefilterFn prefilter_fn;
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

    // Strahlsuche (M6-T4): Gruppen gleicher Topologie, Varianten aus der Diagnose, Abstand,
    // gestörte Starts, FP32-Vorfilter.
    std::vector<HomotopySolve> solve_all(std::span<SystemDescription* const> systems,
                                         const HomotopyPoint& point,
                                         std::vector<int>* groups = nullptr) override;
    std::vector<HomotopyNeed> insertion_options(const SystemDescription& system,
                                                const HomotopyPoint& point,
                                                const HomotopyNeed& need, int max) override;
    double distance(const SystemDescription& a, const SystemDescription& b,
                    const HomotopyPoint& point) override;
    // Sekante über die Variablen (VariableSet am Punkt), projiziert auf die Schranken; leer bei
    // anderer Topologie oder wenn die Werte nicht geschrieben werden können.
    std::optional<SystemDescription> predict(const SystemDescription& previous,
                                             const SystemDescription& current,
                                             const HomotopyPoint& point, double ratio) override;
    std::vector<SystemDescription> perturbed_starts(const SystemDescription& start,
                                                    const HomotopyInsertion& insertion,
                                                    const HomotopyPoint& point, int count,
                                                    std::uint64_t seed) override;
    std::vector<bool> prefilter(std::span<const SystemDescription* const> starts,
                                const HomotopyPoint& point) override;

private:
    std::optional<HomotopyNeed> color_need(const SystemDescription& system,
                                           const HomotopyPoint& point) const;
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
