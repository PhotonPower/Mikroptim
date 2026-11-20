#pragma once

// Reales Testsystem ohne Geisterplatten für die Rückkehr zu Geisterplatten (M6-T10): der frei
// erfundene Mehrlinser testsystem_mehrlinser.json mit Variablen und Nebenbedingungen.

#include "mikroptim/cpu/load_system.hpp"
#include "mikroptim/io/system_json.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/model/variables.hpp"
#include "residual_test_systems.hpp"

#include <filesystem>
#include <string>

namespace mko::test {

// Variablen: Krümmungen "all", Luftspalte 3 und 5 als Liste, Glasdicke 1 einzeln,
// Arbeitsabstand; Nebenbedingungen: Brennweite 40 mm, Mindestluftspalt. Mit `small` nur 546 nm
// und zwei Felder (kurze Läufe in den CLI-Tests).
inline SystemDescription ghost_return_real_system(bool small = false) {
    SystemDescription d = load_residual_system("testsystem_mehrlinser.json").system;
    if (small) {
        d.spectrum.wavelengths_nm = {546.07};
        d.spectrum.weights = {1.0};
        d.source.field_angles_deg = {0.0, 3.0};
        d.source.field_weights = {1.0, 1.0};
    }
    Variables v;
    VariableSpec c;
    c.type = VariableType::Curvature;
    c.selection = SurfaceSelection::All;
    v.items.push_back(c);
    VariableSpec t;
    t.type = VariableType::Thickness;
    t.selection = SurfaceSelection::List;
    t.surfaces = {3, 5};
    t.min = 0.1;
    t.max = 20.0;
    v.items.push_back(t);
    VariableSpec g;
    g.type = VariableType::Thickness;
    g.selection = SurfaceSelection::One;
    g.surfaces = {1};
    g.min = 0.5;
    g.max = 6.0;
    v.items.push_back(g);
    VariableSpec w;
    w.type = VariableType::WorkingDistance;
    w.selection = SurfaceSelection::None;
    v.items.push_back(w);
    d.variables = v;
    Constraints cs;
    ConstraintSpec f;
    f.type = ConstraintType::FocalLength;
    f.weight = 1.0;
    cs.items.push_back(f);
    ConstraintSpec m;
    m.type = ConstraintType::MinCenterThickness;
    m.weight = 100.0;
    m.medium = ConstraintMedium::Air;
    m.thickness = 0.05;
    cs.items.push_back(m);
    d.constraints = cs;
    d.target = Target{};
    d.target->focal_length = 40.0;
    validate_system(d);
    return d;
}

// Als Datei (Kataloge mit absolutem Pfad, Referenzsphäre image_infinite).
inline void write_ghost_return_real_system(const std::filesystem::path& file, bool small = false) {
    SystemDescription d = ghost_return_real_system(small);
    d.glass_catalogs = {std::string(MKO_SOURCE_DIR) + "/data/glass/basis.json"};
    d.reference_sphere = ReferenceSphereKind::ImageInfinite;
    write_system_file(d, file);
}

} // namespace mko::test
