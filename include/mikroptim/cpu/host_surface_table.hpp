#pragma once

// Flächentabelle auf dem Host (std::vector), liefert die SurfaceTableView für trace_ray
// (docs/architektur.md, Abschnitt "Sequenzieller Trace").

#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/glass/index_table.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/trace/surface_table.hpp"

#include <vector>

namespace mko {

// Besitzt die Flächendaten einer Variante in FP64. Invariante (von make_surface_table
// hergestellt; wer die Felder selbst füllt, ist dafür verantwortlich):
// 2 <= surfaces <= kMaxSurfaces, wavelengths >= 1, type/curvature/thickness/semi_diameter
// haben surfaces Einträge, index hat surfaces · wavelengths Einträge (Layout wie IndexTable),
// ghost_plates = Zahl der Flächen vom Typ GhostPlate, ghost_coeff hat G · K, ghost_index G · L
// und ghost_norm_radius G Einträge (Layout wie SurfaceTableView).
struct HostSurfaceTable {
    int surfaces = 0;
    int wavelengths = 0;
    std::vector<SurfaceKind> type;
    std::vector<double> curvature;     // 1/mm, 0 bei Stop, Plane und GhostPlate
    std::vector<double> thickness;     // mm, thickness[S−1] = Abstand zur Bildebene
    std::vector<double> index;         // index[s * L + l], Medium nach Fläche s
    std::vector<double> semi_diameter; // mm, nicht Teil der Sicht; [0] = Pupillenradius a;
                                       // bei Geisterplatten der Normierungsradius
    // Geisterplatten (Geisterplatte g = g-te Fläche vom Typ GhostPlate)
    int ghost_plates = 0;                  // G
    int ghost_terms = 0;                   // K = größte Ordnung, kürzere Profile mit 0 ergänzt
    std::vector<double> ghost_coeff;       // [g * K + k − 1] = a_k [mm]
    std::vector<double> ghost_index;       // [g * L + l] = n_G des virtuellen Glases
    std::vector<double> ghost_norm_radius; // [g] = r_g [mm]

    // Sicht auf die Vektoren; gültig, solange die Tabelle lebt und nicht verändert wird.
    SurfaceTableView<double> view() const {
        return SurfaceTableView<double>{surfaces,
                                        wavelengths,
                                        type.data(),
                                        curvature.data(),
                                        thickness.data(),
                                        index.data(),
                                        ghost_plates,
                                        ghost_terms,
                                        ghost_coeff.data(),
                                        ghost_index.data(),
                                        ghost_norm_radius.data()};
    }
};

// Brechzahltabelle eines Systems: Zeilen der Dateiflächen (make_index_table mit den Medien der
// Flächen und den Wellenlängen von system.spectrum) und die Brechzahlen der Gläser aller
// Geisterplatten (ghost_n). Wirft GlassError; die Meldung nennt Fläche bzw. Geisterplatte.
IndexTable make_system_index_table(const SystemDescription& system, const GlassCatalog& catalog);

// Baut die Tabelle aus Systembeschreibung und Brechzahltabelle (make_system_index_table oder,
// ohne Geisterplatten, make_index_table mit den Wellenlängen von system.spectrum).
// Geisterplatten werden als Flächen vom Typ GhostPlate eingefügt (Abbildung: make_surface_map):
// Der Luftspalt nach Dateifläche i (Dicke T) wird an den Abständen d_1 <= d_2 <= … geteilt,
// thickness = d_1, d_2 − d_1, …, T − d_last; das Medium nach jeder Geisterplatte ist das des
// Spalts. Wirft std::invalid_argument, wenn die Flächenzahlen nicht übereinstimmen, die Zahl der
// Tabellenflächen außerhalb von [2, kMaxSurfaces] liegt, die Tabelle keine Wellenlängen hat
// oder die Brechzahlen der Geisterplatten fehlen.
HostSurfaceTable make_surface_table(const SystemDescription& system, const IndexTable& index);

} // namespace mko
