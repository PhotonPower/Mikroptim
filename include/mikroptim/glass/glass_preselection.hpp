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
#include "mikroptim/glass/glass_properties.hpp"

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
    // Spektrum [nm] für den spektralen Rest (M11-T5g); leer: Rang nach dem Koeffizienten der
    // Ordnung k + 1 wie bisher. Sonst Rang nach spectral_residual: Streuung (RMS um den
    // Mittelwert, gleich gewichtet) des relativen Brechkraftrests
    //     ρ(ω) = Σ w_i φ_i (η_{i,k+1} ω^{k+1} + … + η_{i,3} ω³) / φ
    // über die Buchdahl-Koordinaten ω der Wellenlängen, also der Farblängsfehler nach der besten
    // Fokussierung (mit ν₃ statt nur dem führenden Term). Mindestens zwei Wellenlängen > 146,074
    // nm.
    std::vector<double> wavelengths_nm;
};

struct PreselectedGlasses {
    std::vector<int> glasses;   // Indizes in der Glasliste, Reihenfolge der Elemente
    std::vector<double> powers; // φ_i (ohne Gewicht)
    double power_sum = 0.0;     // Σ|w_i φ_i| / |φ|
    double residual = 0.0;      // |Σ w_i φ_i η_{i,k+1}| / |φ|
    // Streuung von ρ(ω) über options.wavelengths_nm (NaN ohne Spektrum)
    double spectral_residual = 0.0;
};

// Alle Kombinationen (Paare bzw. Tripel ohne Wiederholung, mit den festen Gläsern) lösen,
// filtern (singulär, Σ|w φ|/|φ| > max_power_sum) und nach dem Rest (mit Spektrum: dem
// spektralen Rest), bei Gleichheit nach der Summe der Brechkräfte, sortieren; höchstens
// max_results. Deterministisch (feste Reihenfolge).
// Wirft std::invalid_argument bei ungültigen Optionen.
std::vector<PreselectedGlasses> preselect_glasses(std::span<const GlassDispersion> glasses,
                                                  const GlassPreselectionOptions& options = {});

// ---------------------------------------------------------------------------------------------
// Filter der Kandidaten (M11-T5g): dieselbe Bereichsbeschreibung wie die Glasabfrage (M10-T5,
// GlassPropertyFilter auf n_d, ν_d, P_g,F, ΔP_g,F), damit Vorauswahl und glass_query
// übereinstimmen. Regeln der Wissensbasis und Transmission wertet mikroptim.design aus und reicht
// das Ergebnis als Namensliste (`only`) durch.
// ---------------------------------------------------------------------------------------------

struct GlassCandidateFilter {
    GlassPropertyFilter properties;
    std::vector<std::string> only;    // nicht leer: nur diese Namen
    std::vector<std::string> exclude; // diese Namen nie
    bool empty() const {
        return !properties.n_d && !properties.v_d && !properties.p_gF && !properties.dp_gF &&
               only.empty() && exclude.empty();
    }
};

// Gläser aus `glasses` (Reihenfolge bleibt), die den Filter erfüllen; Kennzahlen aus `catalog`
// (glass_properties). Mit Bereichen fallen Gläser heraus, deren Bereich die Linien g … C nicht
// abdeckt (wie list_glass_properties). Gläser in `keep` (z. B. die festen Gläser der Vorauswahl)
// bleiben immer. Wirft GlassError, wenn ein Glas nicht im Katalog steht.
std::vector<GlassDispersion> filter_dispersions(std::span<const GlassDispersion> glasses,
                                                const GlassCatalog& catalog,
                                                const GlassCandidateFilter& filter,
                                                std::span<const std::string> keep = {});

// ---------------------------------------------------------------------------------------------
// Startradien des ganzen Glieds (M11-T5g, docs/numerik.md, Abschnitt 15): m dünne Elemente in
// Kontakt (Kittglied bzw. -tripel), Blende am Glied, kollimiert einfallend. Mit den Brechkräften
// K_i und Brechzahlen n_i liegt die Krümmungsfolge bis auf die Biegung c₁ fest:
//
//     c_{i+1} = c_i − K_i/(n_i − 1),
//
// Seidel-Summen nach Welford (Aberrations of Optical Systems, Kap. 7, dünne Linse), mit
// Formfaktor B_i = (c_i + c_{i+1})/(c_i − c_{i+1}), Konjugationsfaktor C_i = (u_i + u_i′)/(u_i −
// u_i′), u_i = −y Σ_{j<i} K_j, u_i′ = u_i − y K_i, Randstrahlhöhe y und Lagrange-Invariante H:
//
//     S_I  = Σ (y⁴K³/4) [(n/(n−1))² + (n+2)/(n(n−1)²) B² + 4(n+1)/(n(n−1)) B C + (3n+2)/n C²]
//     S_II = Σ (y²K²H/2) [(n+1)/(n(n−1)) B + (2n+1)/n C]
//
// (Vorzeichen von S_II zu H = n(ū y − u ȳ) wie im Aberrationsbericht, numerik.md §14). S_I ist
// quadratisch, S_II linear in c₁.
// ---------------------------------------------------------------------------------------------

enum class MemberBending {
    // S_I = 0, wenn reell lösbar (von beiden Lösungen die flachere, kleinere Σ c_i²; in der
    // Bruteforce-Abnahme nach Optimierung bei 248 von 298 Paaren die bessere), sonst der Scheitel
    // der Parabel (kleinstes |S_I|).
    MinSpherical,
    // S_II = 0 (komafrei bei Blende am Glied).
    ZeroComa,
};

struct ThinMemberShape {
    std::vector<double> curvatures; // m + 1 Flächen, vorn nach hinten [1/mm]
    double spherical = 0.0;         // S_I bei y = 1 (skaliert mit y⁴)
    double coma = 0.0;              // S_II bei y = 1, H = 1 (skaliert mit y²H)
    bool spherical_zero = false;    // S_I = 0 war reell erreichbar
    // Koeffizienten S_I = a c₁² + b c₁ + d und S_II = s c₁ + t (y = 1, H = 1)
    std::array<double, 3> spherical_coefficients{};
    std::array<double, 2> coma_coefficients{};
};

// Biegung des dünnen Glieds mit Brechzahlen `indices` (n_i > 1, z. B. n₀ des Buchdahl-Fits bei
// 546,074 nm) und Brechkräften `powers` (z. B. PreselectedGlasses::powers). Wirft
// std::invalid_argument bei ungleichen Längen, n ≤ 1, nicht endlichen Werten oder (ZeroComa)
// wenn S_II nicht von c₁ abhängt.
ThinMemberShape thin_member_shape(std::span<const double> indices, std::span<const double> powers,
                                  MemberBending bending = MemberBending::MinSpherical);

// Seidel-Summen S_I, S_II (y = 1, H = 1) des dünnen Glieds mit den Krümmungen `curvatures`
// (m + 1 Flächen; die Brechkräfte folgen daraus).
std::array<double, 2> thin_member_seidel(std::span<const double> indices,
                                         std::span<const double> curvatures);

} // namespace mko
