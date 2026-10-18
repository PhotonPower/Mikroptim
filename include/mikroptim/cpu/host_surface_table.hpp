#pragma once

// Flächentabelle auf dem Host (std::vector), liefert die SurfaceTableView für trace_ray
// (docs/architektur.md, Abschnitt "Sequenzieller Trace").

#include "mikroptim/glass/index_table.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/trace/surface_table.hpp"

#include <vector>

namespace mko {

// Besitzt die Flächendaten einer Variante in FP64. Invariante (von make_surface_table
// hergestellt; wer die Felder selbst füllt, ist dafür verantwortlich):
// 2 <= surfaces <= kMaxSurfaces, wavelengths >= 1, type/curvature/thickness/semi_diameter
// haben surfaces Einträge, index hat surfaces · wavelengths Einträge (Layout wie IndexTable).
struct HostSurfaceTable {
    int surfaces = 0;
    int wavelengths = 0;
    std::vector<SurfaceKind> type;
    std::vector<double> curvature;     // 1/mm, 0 bei Stop und Plane
    std::vector<double> thickness;     // mm, thickness[S−1] = Abstand zur Bildebene
    std::vector<double> index;         // index[s * L + l], Medium nach Fläche s
    std::vector<double> semi_diameter; // mm, nicht Teil der Sicht; [0] = Pupillenradius a

    // Sicht auf die Vektoren; gültig, solange die Tabelle lebt und nicht verändert wird.
    SurfaceTableView<double> view() const {
        return SurfaceTableView<double>{surfaces,         wavelengths,      type.data(),
                                        curvature.data(), thickness.data(), index.data()};
    }
};

// Baut die Tabelle aus Systembeschreibung und Brechzahltabelle (make_index_table mit den
// Wellenlängen von system.spectrum). Wirft std::invalid_argument, wenn die Flächenzahlen nicht
// übereinstimmen, die Flächenzahl außerhalb von [2, kMaxSurfaces] liegt oder die Tabelle keine
// Wellenlängen hat.
HostSurfaceTable make_surface_table(const SystemDescription& system, const IndexTable& index);

} // namespace mko
