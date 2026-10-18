#pragma once

// Kleinste-Quadrate-Fit des Buchdahl-Modells an eine Brechzahlfunktion, typischerweise an
// die Sellmeier-Brechzahl eines Katalogglases (docs/numerik.md, Abschnitt 10). Host-Code:
// Fehler werden als GlassError gemeldet.
//
// Termzahl: `dispersion_terms` zählt die Dispersionskoeffizienten ohne n0.
//   3 Terme: n = n0 + nu1 w + nu2 w^2 + nu3 w^3  (vier Unbekannte),
//   2 Terme: n = n0 + nu1 w + nu2 w^2, nu3 = 0   (drei Unbekannte).
// Die 2-Term-Koeffizienten (n0, nu1, nu2) sind die Glasvariablen der Optimierung, der
// 3-Term-Fit dient der Bewertung (numerik.md, Abschnitt 10).

#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/model/material.hpp"

#include <array>
#include <functional>
#include <vector>

namespace mko {

// Verteilung der Fit-Stützstellen über [lambda_min_nm, lambda_max_nm].
enum class BuchdahlFitNodes {
    // Tschebyschow-Knoten erster Art in lambda (ohne Randpunkte, zu den Rändern verdichtet).
    // Der Kleinste-Quadrate-Fit kommt damit dem gleichmäßig besten (Minimax-)Fit nahe;
    // Standard, weil die Abnahme den maximalen Fehler bewertet.
    Chebyshev,
    // Äquidistant in lambda, beide Randpunkte eingeschlossen.
    Equidistant,
};

struct BuchdahlFitOptions {
    double lambda_min_nm = 400.0; // Fitbereich [nm]
    double lambda_max_nm = 700.0;
    int samples = 301; // Anzahl der Fit-Stützstellen (mindestens dispersion_terms + 1)
    BuchdahlFitNodes nodes = BuchdahlFitNodes::Chebyshev;
    // Äquidistantes Prüfgitter für max_abs_error und rms_error, Randpunkte eingeschlossen.
    // 3001 Punkte = 0,1 nm Schritt über 400-700 nm (mindestens 2).
    int check_samples = 3001;
};

struct BuchdahlFit {
    int dispersion_terms = 0;                // 2 oder 3
    double n0 = 0.0;                         // Brechzahl bei lambda0 = 546,074 nm
    std::array<double, 3> nu{0.0, 0.0, 0.0}; // (nu1, nu2, nu3); nu3 = 0 bei 2 Termen
    double max_abs_error = 0.0;              // max |n_Buchdahl - n| auf dem Prüfgitter
    double rms_error = 0.0;     // Wurzel des mittleren Fehlerquadrats auf dem Prüfgitter
    BuchdahlFitOptions options; // verwendete Einstellungen

    // Brechzahl des angepassten Modells bei lambda_nm (buchdahl_index, ohne Bereichsprüfung).
    double index(double lambda_nm) const;
};

// Brechzahl als Funktion der Wellenlänge in nm.
using IndexFunction = std::function<double(double lambda_nm)>;

// Fit-Stützstellen [nm], aufsteigend. Wirft GlassError bei ungültigen Optionen.
std::vector<double> buchdahl_fit_nodes_nm(const BuchdahlFitOptions& options);

// Prüfgitter [nm]: lambda_i = min + (max - min) i / (check_samples - 1), letzter Punkt = max.
// Wirft GlassError bei ungültigen Optionen.
std::vector<double> buchdahl_check_grid_nm(const BuchdahlFitOptions& options);

// Fit an eine beliebige Brechzahlfunktion. Lösung des überbestimmten linearen Systems per
// Householder-QR in FP64 (keine Normalgleichungen). Wirft GlassError bei ungültiger Termzahl,
// ungültigen Optionen (Bereich leer, nicht endlich oder nicht oberhalb von 146,074 nm, zu
// wenige Stützstellen) und nicht endlichen Funktionswerten.
BuchdahlFit fit_buchdahl(const IndexFunction& index_nm, int dispersion_terms,
                         const BuchdahlFitOptions& options = {});

// Fit an die Sellmeier-Brechzahl eines Katalogglases. Wirft zusätzlich GlassError, wenn
// [lambda_min_nm, lambda_max_nm] nicht in glass.range_nm liegt.
BuchdahlFit fit_buchdahl(const Glass& glass, int dispersion_terms,
                         const BuchdahlFitOptions& options = {});

// Material vom Typ Buchdahl mit den Fitkoeffizienten (für make_index_table).
MaterialSpec to_buchdahl_material(const BuchdahlFit& fit);

} // namespace mko
