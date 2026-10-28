#pragma once

// Mechanische Nebenbedingungen (M4-T3, Host): Durchmesser, Mitten- und Randdicke, maximaler
// Einfallswinkel als einseitige Residuen r = √w · max(0, g). Bedingungen ohne Strahlbezug
// (Mittendicke, Randdicke mit festem Halbdurchmesser) sind gewöhnliche Zusatzresiduen
// (ExtraResiduals); Strahlhöhe, Einfallswinkel und Randdicke bei der freien Öffnung kommen aus
// den Max-Reduktionen derselben Trace-Auswertung (TraceExtraResiduals). Formeln, Vorzeichen,
// Glattheit und Gewichte: docs/optimierung.md, Abschnitt 2.5; Schema: docs/dateiformat.md.

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/optim/extra_residuals.hpp"
#include "mikroptim/optim/trace_extra_residuals.hpp"

#include <vector>

namespace mko {

struct SystemDescription;

// Pfeilhöhe einer Sphäre mit Krümmung c im Abstand h von der Achse (docs/numerik.md, §2: z in
// Tracerichtung, c > 0 mit Mittelpunkt auf der +z-Seite, dann sag > 0):
//
//     sag(h) = c h² / (1 + √(1 − c² h²)),   1 − c² h² = (1 − |c| h)(1 + |c| h),
//
// ohne Auslöschung für alle |c| h < 1 und exakt 0 für c = 0. Für |c| h ≥ 1 (Halbdurchmesser
// über dem Krümmungsradius, nicht herstellbar) die Fortsetzung c h²: stetig mit der Pfeilhöhe
// 1/c der Halbkugel bei |c| h = 1, danach weiter wachsend, sodass die Randdicke eine endliche,
// mit h fallende Strafe liefert (die Ableitung ist bei |c| h → 1 von unten unbeschränkt).
double sag(double curvature, double height);

// Randdicke des Spalts zwischen einer Fläche (Krümmung c_front) und der nächsten (c_back) mit
// Mittendicke t im Abstand h von der Achse: t − sag(c_front, h) + sag(c_back, h). Bikonvexe
// Linse (c_front > 0 > c_back): t − |sag_1| − |sag_2|.
double edge_thickness(double thickness, double c_front, double c_back, double height);

// Eine mechanische Bedingung für eine Dateifläche bzw. einen Spalt (aufgelöst aus einem Eintrag
// von constraints.items).
struct MechanicalConstraint {
    ConstraintType type = ConstraintType::MinCenterThickness;
    int file_surface = 0; // Dateifläche i (Dicken: Spalt i → i + 1)
    int surface = 0;      // Tabellenfläche m(i)
    int next_surface = 0; // Tabellenfläche m(i + 1) (nur Dicken; sonst gleich surface)
    double weight = 0.0;  // w ≥ 0 (1/mm² bei Längen, dimensionslos beim Einfallswinkel)
    // Grenze: Halbdurchmesser [mm] (MaxSemiDiameter), Dicke [mm] (Mitten- und Randdicke) bzw.
    // cos θ_max (MaxIncidenceAngle).
    double limit = 0.0;
    double angle_deg = 0.0; // θ_max [deg] (nur MaxIncidenceAngle)
    double clearance = 0.0; // Fassungszugabe [mm] (MaxSemiDiameter, MinEdgeThickness)
    EdgeHeight height = EdgeHeight::SemiDiameter; // nur MinEdgeThickness
    // Mechanische Halbdurchmesser der Dateiflächen i und i + 1 (nur MinEdgeThickness mit
    // height = SemiDiameter) [mm].
    double semi_diameter = 0.0;
    double next_semi_diameter = 0.0;
    int item = -1; // Eintrag in constraints.items (−1: von Hand gebaut)

    friend bool operator==(const MechanicalConstraint&, const MechanicalConstraint&) = default;
};

// Wahr, wenn die Bedingung die Extrema des Traces braucht (MaxSemiDiameter, MaxIncidenceAngle,
// MinEdgeThickness mit height = Rays).
bool needs_extrema(const MechanicalConstraint& c);

struct MechanicalConstraints {
    // In der Reihenfolge der Einträge, pro Eintrag die Flächen in der Reihenfolge von
    // constraint_surfaces.
    std::vector<MechanicalConstraint> items;
    // Glättung der Extrema (constraints.extrema_smoothing, M11-T1)
    ExtremaSmoothing smoothing{};

    int size() const { return static_cast<int>(items.size()); }
    bool empty() const { return items.empty(); }
    bool needs_extrema() const;
};

// Bedingungen aus der Systembeschreibung (alle mechanischen Einträge von `constraints`, geprüft
// mit check_constraints): Flächen aus constraint_surfaces, Tabellenflächen über die SurfaceMap,
// fehlender semi_diameter bei MaxSemiDiameter aus surfaces[i].semi_diameter, limit bei
// MaxIncidenceAngle = cos(angle · π/180). Wirft FormatError bei ungültigem Abschnitt.
MechanicalConstraints make_mechanical_constraints(const SystemDescription& system);

// Prüft gegen eine Topologie: mechanischer Typ, Tabellenflächen existieren und sind keine
// Geisterplatten, bei Dicken 0 ≤ surface < next_surface < S, Gewicht, Grenze, clearance und
// Halbdurchmesser endlich (w, clearance ≥ 0; Halbdurchmesser > 0). Wirft std::invalid_argument.
void check_mechanical_constraints(const MechanicalConstraints& constraints,
                                  const BatchTopology& topology);

// Mittendicke des Spalts der Variante v: Summe der Tabellendicken m(i) … m(i + 1) − 1
// (aufsteigend; ein Luftspalt mit Geisterplatten zählt ganz).
double mechanical_center_thickness(const MechanicalConstraint& c, const SystemBatchHost& batch,
                                   int v);

// Halbdurchmesser h der Randdicke: max der beiden mechanischen Halbdurchmesser bzw. der beiden
// größten Strahlhöhen (Extrema, nur bei height = Rays nötig) plus clearance.
double mechanical_edge_height(const MechanicalConstraint& c, const VariantExtrema* extrema);

// Istgröße der Bedingung in Variante v: größte Strahlhöhe + clearance [mm] (MaxSemiDiameter),
// Mittendicke [mm], Randdicke [mm] bzw. kleinster cos I (MaxIncidenceAngle). `extrema` sind die
// Extrema der Variante (nur nötig, wenn needs_extrema(c); sonst darf es nullptr sein).
// Wirft std::invalid_argument, wenn Extrema gebraucht werden und fehlen.
double mechanical_value(const MechanicalConstraint& c, const SystemBatchHost& batch, int v,
                        const VariantExtrema* extrema);

// Verletzung g (> 0: verletzt) zur Istgröße: value − limit bei MaxSemiDiameter und
// MaxCenterThickness, limit − value bei MinCenterThickness, MinEdgeThickness und
// MaxIncidenceAngle (cos I ≤ cos θ_max ⇔ I ≥ θ_max).
double mechanical_violation(const MechanicalConstraint& c, double value);

// Residuum √w · max(0, g) der Variante v.
double mechanical_residual(const MechanicalConstraint& c, const SystemBatchHost& batch, int v,
                           const VariantExtrema* extrema);

// Größter Einfallswinkel [deg] zu cos I (arccos, cos I auf [−1, 1] begrenzt).
double incidence_angle_deg(double cos_incidence);

// Aufteilung für den Residuenvektor: `geometric` (ohne Extrema, gewöhnliche Zusatzresiduen) und
// `traced` (aus den Extrema), je in der Reihenfolge der Bedingungen; geometric_items[k] bzw.
// traced_items[k] ist der Index der Bedingung (in MechanicalConstraints::items) von Zeile k.
// Beide Funktionen sind threadsicher.
struct MechanicalResiduals {
    ExtraResiduals geometric;
    TraceExtraResiduals traced;
    std::vector<int> geometric_items;
    std::vector<int> traced_items;
};
MechanicalResiduals make_mechanical_residuals(const MechanicalConstraints& constraints);

} // namespace mko
