#pragma once

// Brechzahltabelle n[Fläche][Wellenlänge] (docs/architektur.md, Abschnitt "Brechzahltabelle").

#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/model/material.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace mko {

// Zeilenweise abgelegt: n[s * wavelengths + l] ist die Brechzahl des Mediums nach Fläche s
// bei Wellenlänge l (docs/numerik.md, Abschnitt 2). Das Medium vor Fläche 0 ist Luft.
struct IndexTable {
    int surfaces = 0;
    int wavelengths = 0;
    std::vector<double> n;

    double at(int s, int l) const {
        return n[static_cast<std::size_t>(s) * static_cast<std::size_t>(wavelengths) +
                 static_cast<std::size_t>(l)];
    }
};

// Brechzahl eines Mediums bei lambda_nm: Air -> 1, Constant -> n, Catalog -> Sellmeier aus
// dem Katalog (mit Bereichsprüfung), Buchdahl -> Formel aus docs/numerik.md, Abschnitt 10.
// Wirft GlassError bei unbekanntem Glas, Wellenlänge außerhalb des Gültigkeitsbereichs
// oder ungültigen Parametern.
double material_index(const MaterialSpec& medium, double lambda_nm, const GlassCatalog& catalog);

// Tabelle für alle Medien (Medium s = Medium nach Fläche s) und Wellenlängen [nm].
// Wirft GlassError; die Meldung nennt Fläche und Wellenlänge.
IndexTable make_index_table(std::span<const MaterialSpec> media,
                            std::span<const double> wavelengths_nm, const GlassCatalog& catalog);

} // namespace mko
