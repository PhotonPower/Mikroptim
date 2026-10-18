#pragma once

// Paraxiale Kenngrößen eines Systems (Host, docs/numerik.md, Abschnitt 9).
// Rechenkern: mikroptim/paraxial/ynu_trace.hpp (header-only, auch für die GPU).

#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/glass/index_table.hpp"
#include "mikroptim/model/system.hpp"

#include <array>
#include <span>

namespace mko {

// Paraxialdaten bei einer Wellenlänge. Alle Längen in mm, Lagen als z-Koordinaten entlang
// +z (Tracerichtung) relativ zum Scheitel der genannten Fläche. Definitionen und Vorzeichen:
// docs/numerik.md, Abschnitt 9.
struct ParaxialData {
    // Systemmatrix von Fläche 0 (vor der Brechung) bis zur letzten Fläche (nach der Brechung)
    // in reduzierten Winkeln: (y_L, n'_L u'_L) = M (y_0, u_0), M = {A, B, C, D}, det M = 1.
    std::array<double, 4> abcd{1.0, 0.0, 0.0, 1.0};

    // Brechkraft phi = -C [1/mm]. Exakt 0: afokales System (afocal = true).
    double power = 0.0;
    bool afocal = false;

    // Effektive Brennweite 1/phi. Bildraum Luft: gleich -y_0 / u'_L. Afokal: +inf.
    double focal_length = 0.0;
    // Hintere Schnittweite: letzte Fläche -> hinterer Brennpunkt F'. Afokal: NaN.
    double back_focal_distance = 0.0;
    // Vordere Schnittweite: Fläche 0 -> vorderer Brennpunkt F (meist negativ). Afokal: NaN.
    double front_focal_distance = 0.0;
    // Vordere Hauptebene H relativ zu Fläche 0. Afokal: NaN.
    double front_principal_plane = 0.0;
    // Hintere Hauptebene H' relativ zur letzten Fläche. Afokal: NaN.
    double rear_principal_plane = 0.0;
    // Defokus der Bildebene: back_focal_distance - Dicke der letzten Fläche (> 0: paraxialer
    // Brennpunkt liegt hinter der Bildebene). Afokal: NaN.
    double image_defocus = 0.0;

    // Petzvalsumme P [1/mm], Krümmung der Petzvalschale kappa_P = -n'_L P [1/mm] und
    // Petzvalradius 1/kappa_P [mm] (+inf bei kappa_P = 0).
    double petzval_sum = 0.0;
    double petzval_curvature = 0.0;
    double petzval_radius = 0.0;

    // Austrittspupille = paraxiales Bild der Blende (Fläche 0) durch alle Flächen.
    // Die Eintrittspupille ist die Blende selbst (z = 0 an Fläche 0, Radius = stop).
    // exit_pupil_at_infinity: D = 0 (bildseitig telezentrisch); dann Lage und Vergrößerung
    // NaN, Radius +inf.
    bool exit_pupil_at_infinity = false;
    // Lage relativ zur letzten Fläche
    double exit_pupil_position = 0.0;
    // Pupillenvergrößerung (Bildgröße / Blendengröße, vorzeichenbehaftet) = 1/D
    double exit_pupil_magnification = 0.0;
    // Radius der Austrittspupille = Blendenradius * |1/D|
    double exit_pupil_semi_diameter = 0.0;
};

// Paraxialdaten aus nackten Arrays gleicher Länge (Fläche 0 = Blende, Medium vor Fläche 0 Luft):
// Krümmungen [1/mm], Dicken nach jeder Fläche [mm] (die letzte ist der Abstand zur Bildebene),
// Brechzahlen nach jeder Fläche, Blendenradius [mm].
// Wirft std::invalid_argument bei leeren oder ungleich langen Arrays, nicht endlichen Werten,
// Brechzahlen <= 0 oder Blendenradius <= 0.
ParaxialData compute_paraxial(std::span<const double> curvatures,
                              std::span<const double> thicknesses, std::span<const double> indices,
                              double stop_semi_diameter);

// Wie oben, zusätzlich mit dünnen Elementen ohne Krümmung (Geisterplatten): `power[s]` ist
// ihre Brechkraft φ_G an Fläche s, `petzval[s]` ihr Beitrag φ_G / (n_m n_G) zur Petzvalsumme
// (beide 0 für gewöhnliche Flächen). Leere Spannen bedeuten „keine“, sonst Länge wie
// `curvatures`. Fehler wie oben.
ParaxialData compute_paraxial(std::span<const double> curvatures,
                              std::span<const double> thicknesses, std::span<const double> indices,
                              double stop_semi_diameter, std::span<const double> power,
                              std::span<const double> petzval);

// Paraxialdaten eines Systems bei Wellenlänge `wavelength_index` der Brechzahltabelle
// (`indices` aus make_system_index_table bzw. make_index_table für die Medien des Systems).
// Geisterplatten gehen als dünne Elemente der Brechkraft φ_G = −2 (n_G − 1) t₂ an ihrer Lage
// im Luftspalt ein (Trace über die Tabellenflächen, make_surface_map), mit dem Beitrag φ_G/n_G
// zur Petzvalsumme (docs/numerik.md, Abschnitte 6 und 9); n_G aus indices.ghost_n.
// Wirft std::invalid_argument, wenn die Tabelle nicht zur Flächen- oder Geisterplattenzahl
// passt, und std::out_of_range bei ungültigem Wellenlängenindex.
ParaxialData compute_paraxial(const SystemDescription& system, const IndexTable& indices,
                              int wavelength_index);

// Paraxialdaten eines Systems bei beliebiger Wellenlänge [nm]; Brechzahlen (auch der Gläser der
// Geisterplatten) über material_index, Geisterplatten wie oben.
// Wirft GlassError (unbekanntes Glas, Wellenlänge außerhalb des Bereichs).
ParaxialData compute_paraxial(const SystemDescription& system, const GlassCatalog& catalog,
                              double wavelength_nm);

// Wie oben bei der Referenzwellenlänge `spectrum.reference_nm` (sie muss nicht unter den
// `wavelengths_nm` vorkommen).
ParaxialData compute_paraxial(const SystemDescription& system, const GlassCatalog& catalog);

} // namespace mko
