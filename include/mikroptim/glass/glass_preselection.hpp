#pragma once

// Algebraische Glasvorauswahl für Farbkorrektur an dünnen Elementen (M11-T5,
// docs/optimierung-erweiterungen.md, Abschnitt 5). Host-Code, Bibliothek Mikroptim::glass.
//
// Dispersion im Buchdahl-Modell (docs/numerik.md, Abschnitt 10): n(ω) = n₀ + ν₁ω + ν₂ω² + ν₃ω³.
// Ein dünnes Element hat die Brechkraft φ(ω) = (n(ω) − 1) K, also
//
//     φ(ω) = φ₀ (1 + η₁ω + η₂ω² + η₃ω³),   η_j = ν_j / (n₀ − 1)
//
// (Dispersionskoeffizienten nach Robb und Mercado). Für m Elemente mit Gewichten w_i = h_i²
// (Randstrahlhöhe, normiert) ist die gewichtete Brechkraft Σ w_i φ_i(ω) farbfrei bis zur
// Ordnung k, wenn
//
//     Σ w_i φ_i = φ,   Σ w_i φ_i η_ij = 0   (j = 1 … k).
//
// m = k + 1: lineares System in φ_i. k = 1 Achromat (Paare), k = 2 Apochromat (Tripel). Rest
// der ersten nicht korrigierten Ordnung (sekundäres bzw. tertiäres Spektrum):
// |Σ w_i φ_i η_{i,k+1}| / |φ|. Summe der Einzelbrechkräfte Σ |w_i φ_i| / |φ| (klein = flache
// Radien, Zhang §3.2).

#include "mikroptim/glass/buchdahl_fit.hpp"
#include "mikroptim/glass/catalog.hpp"

#include <array>
#include <span>
#include <string>
#include <vector>

namespace mko {

// Dispersion eines Glases aus dem 3-Term-Buchdahl-Fit.
struct GlassDispersion {
    std::string name;
    double n0 = 0.0;
    std::array<double, 3> eta{}; // η₁, η₂, η₃
};

// η aus einem Buchdahl-Fit (n₀ > 1 verlangt; sonst std::invalid_argument).
GlassDispersion glass_dispersion(std::string name, const BuchdahlFit& fit);

// Dispersionen aller Gläser des Katalogs (3-Term-Fit mit `options`); Gläser, deren Bereich den
// Fitbereich nicht abdeckt oder deren Fit scheitert, werden übersprungen (Namen in `skipped`).
std::vector<GlassDispersion> catalog_dispersions(const GlassCatalog& catalog,
                                                 const BuchdahlFitOptions& options = {},
                                                 std::vector<std::string>* skipped = nullptr);

struct GlassPreselectionOptions {
    int order = 1;               // k: 1 Achromat (Paare), 2 Apochromat (Tripel)
    std::vector<double> weights; // w_i = h_i², Länge k + 1; leer: alle 1
    double power = 1.0;          // φ (Vorzeichen und Betrag; Ergebnis skaliert linear)
    double max_power_sum = 10.0; // verwerfen, wenn Σ|w_i φ_i|/|φ| größer
    int max_results = 20;
    // Indizes (in der Glasliste), die in jeder Kombination vorkommen müssen (z. B. das Glas der
    // aufzuspaltenden Linse); höchstens k + 1, keine Wiederholung.
    std::vector<int> fixed;
    // Relative Determinante |det| / Π‖Zeile‖, darunter gilt das System als singulär.
    double singular_tolerance = 1e-9;
};

struct PreselectedGlasses {
    std::vector<int> glasses;   // Indizes in der Glasliste, Reihenfolge der Elemente
    std::vector<double> powers; // φ_i (ohne Gewicht)
    double power_sum = 0.0;     // Σ|w_i φ_i| / |φ|
    double residual = 0.0;      // |Σ w_i φ_i η_{i,k+1}| / |φ|
};

// Alle Kombinationen (Paare bzw. Tripel ohne Wiederholung, mit den festen Gläsern) lösen,
// filtern (singulär, Σ|w φ|/|φ| > max_power_sum) und nach dem Rest, bei Gleichheit nach der
// Summe der Brechkräfte, sortieren; höchstens max_results. Deterministisch (feste Reihenfolge).
// Wirft std::invalid_argument bei ungültigen Optionen.
std::vector<PreselectedGlasses> preselect_glasses(std::span<const GlassDispersion> glasses,
                                                  const GlassPreselectionOptions& options = {});

} // namespace mko
