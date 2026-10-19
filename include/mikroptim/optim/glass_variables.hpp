#pragma once

// Kontinuierliches Glas als Variable (M6-T1, Host): Umwandlung der Startgläser in
// Buchdahl-Materialien und Strafe außerhalb der Glas-Hülle als Zusatzresiduen
// (docs/optimierung.md, Abschnitte 1 und 2.4; Hülle: mikroptim/glass/glass_hull.hpp).
//
// Ablauf: Systemdatei mit Variablen `glass` laden, convert_glass_variables (Katalogglas →
// Buchdahl-Fit, n₀, ν₁, ν₂ werden Variablen, ν₃ bleibt fest), Brechzahltabelle neu
// (prepare_system), Batch und VariableSet bauen; Glas-Hülle aus den Katalogen
// (make_glass_hull), Strafe aus der Nebenbedingung `glass_hull` (make_glass_hull_penalty) und
// als Zusatzresiduen (make_glass_hull_residuals, kombinierbar über concat_extra_residuals).

#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/glass/buchdahl_fit.hpp"
#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/glass/glass_hull.hpp"
#include "mikroptim/model/material.hpp"
#include "mikroptim/optim/extra_residuals.hpp"
#include "mikroptim/optim/variable_set.hpp"

#include <array>
#include <span>
#include <string>
#include <vector>

namespace mko {

struct SystemDescription;

// Bericht über ein umgewandeltes Material.
struct GlassConversion {
    int surface = 0;                         // Dateifläche (Medium danach)
    MaterialKind from = MaterialKind::Air;   // ursprüngliche Art
    std::string name;                        // Katalogname (bei Catalog)
    double n0 = 0.0;                         // Ergebnis
    std::array<double, 3> nu{0.0, 0.0, 0.0}; // (ν₁, ν₂, ν₃)
    double max_fit_error = 0.0;              // max |n_Buchdahl − n| des Fits (0 ohne Fit)
};

// Ersetzt das Material jeder Fläche mit einer Glasvariable durch ein Buchdahl-Material:
// Katalogglas → fit_buchdahl(glass, dispersion_terms, fit) (Standard 3 Terme über 400–700 nm,
// dieselben Einstellungen wie die Glas-Hülle; ν₃ des Fits bleibt in der Optimierung fest);
// konstantes Material n → n₀ = n, ν = 0; Buchdahl bleibt unverändert. Rückgabe: ein Eintrag pro
// Glasfläche (Reihenfolge der Variablen). Danach müssen Brechzahltabelle und Batch neu gebaut
// werden (prepare_system). Wirft FormatError bei ungültigem Abschnitt `variables`, GlassError
// bei unbekanntem Glas oder wenn der Fitbereich nicht in range_nm des Glases liegt.
std::vector<GlassConversion> convert_glass_variables(SystemDescription& system,
                                                     const GlassCatalog& catalog,
                                                     const BuchdahlFitOptions& fit = {},
                                                     int dispersion_terms = 3);

// Ein bestraftes Glaselement.
struct GlassHullElement {
    int file_surface = 0;  // Dateifläche (Medium danach)
    int batch_surface = 0; // Tabellenfläche, buchdahl_coeff[v][batch_surface][0 … 2]

    friend bool operator==(const GlassHullElement&, const GlassHullElement&) = default;
};

// Strafe außerhalb der Glas-Hülle: drei Residuen pro Glaselement, der Vektor
//
//     r = √w · max(0, φ(u)) · g(u),   u = normierte Koordinaten (n₀, ν₁, ν₂) des Elements,
//
// mit φ = d(u, Hülle) außerhalb (euklidisch, normiert) und bei einer dreidimensionalen Hülle
// φ = signed_distance + δ (innen minus der Abstand zum Rand, plus Sicherheitsabstand δ), und
// g = ∇φ (Einheitsvektor: außen (u − P(u))/d mit dem nächsten Punkt P(u) der Hülle, innen die
// Normale der nächsten Randebene). Ohne δ ist r = √w (u − P(u)) außen und 0 innen, mit δ > 0
// beginnt die Strafe δ vor dem Rand (auf den Facetten exakt der Abstand zur nach innen
// versetzten Fläche). |r|² = w max(0, φ)² (Strafbeitrag, „√w · Abstand“ des Plans) ist C¹.
// Warum ein Vektor statt des Betrags: d² ist stückweise quadratisch (Abstand zu einer Ecke,
// Kante oder Facette) und u − P(u) stückweise linear; das Gauß-Newton-Modell der Strafe ist
// dann in jedem Stück exakt. Mit dem Betrag √w · d fehlte die Krümmung w (I − g gᵀ) quer zum
// Gradienten an Ecken und Kanten, LM liefe im Zickzack (docs/optimierung.md, Abschnitt 2.4).
inline constexpr int kGlassHullResidualsPerElement = 3;

struct GlassHullPenalty {
    GlassHull hull;
    double weight = 0.0; // w ≥ 0 [Wellen² pro normierte Einheit²]
    double margin = 0.0; // δ ≥ 0 (normiert), > 0 nur bei dimension() == 3
    std::vector<GlassHullElement> elements;
    int item = -1; // Eintrag in constraints.items (−1: von Hand gebaut oder keiner)

    int size() const { return static_cast<int>(elements.size()); } // Glaselemente
    // Zahl der Residuen: 3 pro Glaselement
    int residual_count() const { return kGlassHullResidualsPerElement * size(); }
    bool empty() const { return elements.empty(); }
};

// Strafe aus der Systembeschreibung: Eintrag `glass_hull` von `constraints` (geprüft mit
// check_constraints; Gewicht und δ mit Standardwerten), ein Element pro Glasfläche von `set`
// (Reihenfolge der Variablen). Ohne Eintrag oder ohne Glasvariablen leer (elements leer).
// Wirft FormatError bei ungültigem Abschnitt, std::invalid_argument bei leerer Hülle oder
// δ > 0 mit einer Hülle niedrigerer Dimension.
GlassHullPenalty make_glass_hull_penalty(const SystemDescription& system, const VariableSet& set,
                                         GlassHull hull);

// Prüft Gewicht (endlich, ≥ 0), δ (endlich, ≥ 0, > 0 nur bei dreidimensionaler Hülle), Hülle
// (nicht leer, wenn es Elemente gibt) und die Tabellenflächen gegen die Topologie. Wirft
// std::invalid_argument.
void check_glass_hull_penalty(const GlassHullPenalty& penalty, const BatchTopology& topology);

// Koordinaten (n₀, ν₁, ν₂) der Tabellenfläche s in Variante v (buchdahl_coeff).
Point3 batch_glass_coordinates(const SystemBatchHost& batch, int v, int s);

// max(0, φ(u(x))) für Rohkoordinaten x = (n₀, ν₁, ν₂) (normiert, ohne Gewicht).
double glass_hull_excess(const GlassHullPenalty& penalty, const Point3& x);
// Residuenvektor √w · max(0, φ) · g (siehe oben).
Point3 glass_hull_residual(const GlassHullPenalty& penalty, const Point3& x);
// Strafbeitrag |r|² = w · max(0, φ)² und sein Gradient nach den Rohkoordinaten,
// 2 w max(0, φ) g_j / s_j (für Tests; die Jacobi-Matrix entsteht wie bei allen Zusatzresiduen
// per FD).
double glass_hull_penalty(const GlassHullPenalty& penalty, const Point3& x);
Point3 glass_hull_penalty_gradient(const GlassHullPenalty& penalty, const Point3& x);

// Residuen der Variante v (out.size() == penalty.residual_count()): je Element die drei
// Komponenten von r, Elemente in ihrer Reihenfolge. Wirft std::invalid_argument (Länge),
// std::out_of_range (v, Flächen).
void glass_hull_residuals(const GlassHullPenalty& penalty, const SystemBatchHost& batch, int v,
                          std::span<double> out);

// Zusatzresiduen für den Residuenvektor (count = residual_count(), threadsicher), kombinierbar
// mit make_paraxial_residuals und make_ghost_penalty_residuals über concat_extra_residuals.
ExtraResiduals make_glass_hull_residuals(GlassHullPenalty penalty);

} // namespace mko
