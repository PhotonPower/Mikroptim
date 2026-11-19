// Operatoren der Entwurfssteuerung (M10-T4): allgemeine Umnummerierung mit entfernten Flächen,
// Rückbau eines Glasblocks, Glastausch, Geisterplatten-Wörterbuch und die Vorlagen
// aplanatische Schalenlinse und Null-Kittglied (docs/optimierung.md, Abschnitt 13).

#include "mikroptim/optim/design_operators.hpp"

#include "mikroptim/cpu/host_surface_table.hpp"
#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/glass/index_table.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/model/variables.hpp"
#include "mikroptim/paraxial/paraxial.hpp"
#include "mikroptim/trace/trace.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace mko {

namespace {

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument(message);
}

std::string num(double v) {
    std::ostringstream os;
    os.precision(15);
    os << v;
    return os.str();
}

std::size_t sz(int i) {
    return static_cast<std::size_t>(i);
}

bool is_surface_type(VariableType t) {
    return t == VariableType::Curvature || t == VariableType::Thickness || t == VariableType::Glass;
}

// Mechanische Bedingungen eines Spalts (Fläche i = Spalt i → i + 1).
bool is_gap_constraint(ConstraintType t) {
    return t == ConstraintType::MinCenterThickness || t == ConstraintType::MaxCenterThickness ||
           t == ConstraintType::MinEdgeThickness;
}

void check_finite(double v, const char* what) {
    if (!std::isfinite(v)) {
        invalid(std::string(what) + " muss endlich sein, ist " + num(v));
    }
}

double curvature_of(const Surface& s) {
    return s.type == SurfaceType::Sphere ? s.curvature : 0.0;
}

// Variableneinträge mit "all" in die explizite Liste der aufgelösten Flächen umwandeln.
void explicit_variable_lists(SystemDescription& system) {
    if (!system.variables) {
        return;
    }
    const std::vector<ResolvedVariable> resolved = resolve_variables(system);
    std::vector<VariableSpec>& items = system.variables->items;
    for (std::size_t k = 0; k < items.size(); ++k) {
        VariableSpec& spec = items[k];
        if (!is_surface_type(spec.type) || spec.selection != SurfaceSelection::All) {
            continue;
        }
        std::vector<int> list;
        for (const ResolvedVariable& r : resolved) {
            if (r.item == static_cast<int>(k) && r.component == 0) {
                list.push_back(r.surface);
            }
        }
        spec.surfaces = std::move(list);
        spec.selection = SurfaceSelection::List;
    }
}

// Bezüge eines Eintrags abbilden (−1 entfällt), doppelte über `seen` entfernen; die Auswahl
// bleibt One bei genau einem Bezug, sonst List.
void map_list(std::vector<int>& list, SurfaceSelection& selection, const std::vector<int>& map,
              std::set<int>& seen) {
    std::vector<int> out;
    for (int s : list) {
        if (s < 0 || s >= static_cast<int>(map.size())) {
            invalid("remap_surfaces: Flächenbezug " + std::to_string(s) + " außerhalb von [0, " +
                    std::to_string(map.size() - 1) + "]");
        }
        const int n = map[sz(s)];
        if (n < 0 || !seen.insert(n).second) {
            continue;
        }
        out.push_back(n);
    }
    if (selection == SurfaceSelection::One && out.size() != 1) {
        selection = SurfaceSelection::List;
    }
    list = std::move(out);
}

// Einträge eines Plattenbezugs abbilden (Index der Platten).
void map_plates(std::vector<int>& plates, const std::vector<int>& plate_new_of_old) {
    for (int& g : plates) {
        g = plate_new_of_old.at(sz(g));
    }
}

// Obere Schranke der Dickenvariable des Spalts `gap` (bzw. des Arbeitsabstands, wenn `gap` die
// letzte Fläche ist) auf dessen Dicke anheben, falls sie darunter liegt: Rückbau und
// Rückbau-Homotopie verlängern Spalte (vereinigter Spalt, abgezogene Glasdicke).
void widen_upper_bound(SystemDescription& system, int gap) {
    if (!system.variables) {
        return;
    }
    const bool last = gap == static_cast<int>(system.surfaces.size()) - 1;
    const double t = system.surfaces[sz(gap)].thickness;
    for (VariableSpec& spec : system.variables->items) {
        const bool covers = last ? spec.type == VariableType::WorkingDistance
                                 : spec.type == VariableType::Thickness &&
                                       (spec.selection == SurfaceSelection::All ||
                                        std::find(spec.surfaces.begin(), spec.surfaces.end(),
                                                  gap) != spec.surfaces.end());
        if (covers && spec.max && *spec.max < t) {
            spec.max = t;
        }
    }
}

double total_glass_thickness(const SystemDescription& system, int first, int last) {
    double t = 0.0;
    for (int s = first; s < last; ++s) {
        t += system.surfaces[sz(s)].thickness;
    }
    return t;
}

} // namespace

// ---------------------------------------------------------------------------------------------

SystemDescription remap_surfaces(const SystemDescription& system, const SurfaceRemap& remap,
                                 std::vector<Surface> new_surfaces) {
    const int n_old = static_cast<int>(system.surfaces.size());
    const int n_new = static_cast<int>(new_surfaces.size());
    if (static_cast<int>(remap.surface.size()) != n_old ||
        static_cast<int>(remap.gap.size()) != n_old || n_old < 1) {
        invalid("remap_surfaces: Abbildungen passen nicht zur Flächenzahl " +
                std::to_string(n_old));
    }
    for (int i = 0; i < n_old; ++i) {
        if (remap.surface[sz(i)] >= n_new || remap.gap[sz(i)] >= n_new ||
            remap.surface[sz(i)] < -1 || remap.gap[sz(i)] < -1) {
            invalid("remap_surfaces: Ziel von Fläche " + std::to_string(i) +
                    " außerhalb der neuen Flächen");
        }
    }
    if (remap.surface[0] != 0) {
        invalid("remap_surfaces: die Blende muss Fläche 0 bleiben");
    }

    SystemDescription out = system;
    explicit_variable_lists(out);
    if (out.constraints) {
        for (ConstraintSpec& c : out.constraints->items) {
            if (is_mechanical(c.type) && (c.surface_selection == SurfaceSelection::None ||
                                          c.surface_selection == SurfaceSelection::All)) {
                c.surfaces = constraint_surfaces(system, c);
                c.surface_selection = SurfaceSelection::List;
            }
        }
    }
    out.surfaces = std::move(new_surfaces);

    if (out.variables) {
        std::vector<VariableSpec> kept;
        std::set<int> seen_curv, seen_thick, seen_glass;
        for (VariableSpec& spec : out.variables->items) {
            if (is_surface_type(spec.type)) {
                if (spec.type == VariableType::Thickness) {
                    map_list(spec.surfaces, spec.selection, remap.gap, seen_thick);
                } else {
                    map_list(spec.surfaces, spec.selection, remap.surface,
                             spec.type == VariableType::Curvature ? seen_curv : seen_glass);
                }
                if (spec.surfaces.empty()) {
                    continue;
                }
            }
            kept.push_back(std::move(spec));
        }
        out.variables->items = std::move(kept);
    }
    for (GhostPlate& g : out.ghost_plates) {
        const int n = remap.gap.at(sz(g.after_surface));
        if (n < 0) {
            invalid("remap_surfaces: Geisterplatte nach Fläche " + std::to_string(g.after_surface) +
                    " läge in einem entfernten Spalt");
        }
        g.after_surface = n;
    }
    if (out.constraints) {
        std::vector<ConstraintSpec> kept;
        for (ConstraintSpec& c : out.constraints->items) {
            c.shift.clear();
            if (is_mechanical(c.type)) {
                std::set<int> seen;
                map_list(c.surfaces, c.surface_selection,
                         is_gap_constraint(c.type) ? remap.gap : remap.surface, seen);
                if (c.surfaces.empty()) {
                    continue;
                }
            }
            if (c.reference_surface) {
                const int n = remap.surface.at(sz(*c.reference_surface));
                if (n < 0) {
                    invalid("remap_surfaces: Bezugsfläche " + std::to_string(*c.reference_surface) +
                            " einer Nebenbedingung wird entfernt");
                }
                c.reference_surface = n;
            }
            kept.push_back(std::move(c));
        }
        out.constraints->items = std::move(kept);
    }
    return out;
}

void freeze_variables(SystemDescription& system, std::span<const int> curvature_surfaces,
                      std::span<const int> thickness_gaps, std::span<const int> glass_surfaces) {
    if (!system.variables) {
        return;
    }
    explicit_variable_lists(system);
    std::vector<VariableSpec> kept;
    for (VariableSpec& spec : system.variables->items) {
        if (is_surface_type(spec.type)) {
            const std::span<const int> frozen =
                spec.type == VariableType::Curvature   ? curvature_surfaces
                : spec.type == VariableType::Thickness ? thickness_gaps
                                                       : glass_surfaces;
            std::erase_if(spec.surfaces, [&](int s) {
                return std::find(frozen.begin(), frozen.end(), s) != frozen.end();
            });
            if (spec.surfaces.empty()) {
                continue;
            }
            if (spec.selection == SurfaceSelection::One && spec.surfaces.size() != 1) {
                spec.selection = SurfaceSelection::List;
            }
        }
        kept.push_back(std::move(spec));
    }
    system.variables->items = std::move(kept);
}

// ---------------------------------------------------------------------------------------------

int glass_block_end(const SystemDescription& system, int first) {
    const int S = static_cast<int>(system.surfaces.size());
    if (first < 1 || first > S - 2) {
        invalid("Glasblock: Fläche " + std::to_string(first) + " außerhalb von [1, " +
                std::to_string(S - 2) + "]");
    }
    if (system.surfaces[sz(first - 1)].material.kind != MaterialKind::Air) {
        invalid("Glasblock: vor Fläche " + std::to_string(first) +
                " liegt Glas (Kittglied ab seiner ersten Fläche angeben)");
    }
    if (system.surfaces[sz(first)].material.kind == MaterialKind::Air) {
        invalid("Glasblock: nach Fläche " + std::to_string(first) + " liegt kein Glas");
    }
    int last = first + 1;
    while (last < S && system.surfaces[sz(last)].material.kind != MaterialKind::Air) {
        ++last;
    }
    if (last >= S) {
        invalid("Glasblock ab Fläche " + std::to_string(first) +
                ": hinter der letzten Fläche "
                "liegt Glas");
    }
    return last;
}

bool is_null_element(const SystemDescription& system, int first) {
    const int last = glass_block_end(system, first);
    const double c = curvature_of(system.surfaces[sz(first)]);
    for (int s = first; s <= last; ++s) {
        if (curvature_of(system.surfaces[sz(s)]) != c) {
            return false;
        }
        if (s < last && system.surfaces[sz(s)].thickness != 0.0) {
            return false;
        }
    }
    return true;
}

RemovalResult remove_element(const SystemDescription& system, int first) {
    const int last = glass_block_end(system, first);
    const int S = static_cast<int>(system.surfaces.size());
    const int removed = last - first + 1;

    std::vector<Surface> surfaces;
    surfaces.reserve(sz(S - removed));
    for (int j = 0; j < S; ++j) {
        if (j < first || j > last) {
            surfaces.push_back(system.surfaces[sz(j)]);
        }
    }
    // Spalt davor + Glas + Spalt danach, fortlaufend summiert
    const double before_last =
        system.surfaces[sz(first - 1)].thickness + total_glass_thickness(system, first, last);
    double merged = system.surfaces[sz(first - 1)].thickness;
    for (int s = first; s <= last; ++s) {
        merged += system.surfaces[sz(s)].thickness;
    }
    surfaces[sz(first - 1)].thickness = merged;

    SurfaceRemap map;
    map.surface.resize(sz(S));
    map.gap.resize(sz(S));
    for (int j = 0; j < S; ++j) {
        const int shifted = j < first ? j : (j > last ? j - removed : -1);
        map.surface[sz(j)] = shifted;
        map.gap[sz(j)] = shifted;
    }
    // Der vereinigte Spalt; ist der Block die Frontgruppe, wird er zum Arbeitsabstand (keine
    // Dickenbezüge an der letzten Fläche)
    map.gap[sz(last)] = last == S - 1 ? -1 : first - 1;
    if (last == S - 1) {
        map.gap[sz(first - 1)] = -1;
    }

    RemovalResult r;
    r.new_of_old = map.surface;
    r.first_removed = first;
    r.removed_surfaces = removed;
    r.merged_gap = first - 1;
    r.system = remap_surfaces(system, map, std::move(surfaces));
    for (std::size_t g = 0; g < system.ghost_plates.size(); ++g) {
        if (system.ghost_plates[g].after_surface == last) {
            r.system.ghost_plates[g].distance = before_last + system.ghost_plates[g].distance;
        }
    }
    widen_upper_bound(r.system, first - 1);
    validate_system(r.system);
    return r;
}

SystemDescription collapse_element(const SystemDescription& system, int first, double fraction) {
    const int last = glass_block_end(system, first);
    if (!(fraction > 0.0 && fraction <= 1.0)) {
        invalid("collapse_element: Anteil " + num(fraction) + " außerhalb von (0, 1]");
    }
    // Variablen des Blocks zuerst festsetzen (die Auflösung prüft die Schranken am alten System)
    SystemDescription out = system;
    std::vector<int> block, gaps;
    for (int s = first; s <= last; ++s) {
        block.push_back(s);
        if (s < last) {
            gaps.push_back(s);
        }
    }
    freeze_variables(out, block, gaps, gaps);
    const double c_first = curvature_of(system.surfaces[sz(first)]);
    double taken = 0.0;
    for (int s = first; s < last; ++s) {
        Surface& surf = out.surfaces[sz(s)];
        const double t = fraction == 1.0 ? 0.0 : (1.0 - fraction) * surf.thickness;
        taken += surf.thickness - t;
        surf.thickness = t;
    }
    for (int s = first + 1; s <= last; ++s) {
        Surface& surf = out.surfaces[sz(s)];
        const double c = curvature_of(surf);
        const double c_new = fraction == 1.0 ? c_first : (1.0 - fraction) * c + fraction * c_first;
        if (surf.type == SurfaceType::Plane && c_new != 0.0) {
            surf.type = SurfaceType::Sphere;
        }
        if (surf.type == SurfaceType::Sphere) {
            surf.curvature = c_new;
        }
    }
    out.surfaces[sz(last)].thickness += taken;
    widen_upper_bound(out, last);
    validate_system(out);
    return out;
}

// ---------------------------------------------------------------------------------------------

void replace_glass(SystemDescription& system, int surface, const MaterialSpec& glass) {
    const int S = static_cast<int>(system.surfaces.size());
    if (surface < 1 || surface > S - 2) {
        invalid("replace_glass: Fläche " + std::to_string(surface) + " außerhalb von [1, " +
                std::to_string(S - 2) + "]");
    }
    if (system.surfaces[sz(surface)].material.kind == MaterialKind::Air) {
        invalid("replace_glass: nach Fläche " + std::to_string(surface) + " liegt kein Glas");
    }
    if (glass.kind == MaterialKind::Air) {
        invalid("replace_glass: das neue Medium muss ein Glas sein");
    }
    const int s[] = {surface};
    freeze_variables(system, {}, {}, s);
    system.surfaces[sz(surface)].material = glass;
    validate_system(system);
}

// ---------------------------------------------------------------------------------------------

GhostDictionaryResult place_ghost_dictionary(const SystemDescription& system,
                                             const GhostDictionaryPlacement& p) {
    const int S = static_cast<int>(system.surfaces.size());
    const int i = p.after_surface;
    if (i < 0 || i > S - 2) {
        invalid("place_ghost_dictionary: after_surface " + std::to_string(i) +
                " außerhalb von [0, " + std::to_string(S - 2) + "]");
    }
    const Surface& gap = system.surfaces[sz(i)];
    if (gap.material.kind != MaterialKind::Air) {
        invalid("place_ghost_dictionary: nach Fläche " + std::to_string(i) +
                " liegt kein Luftspalt");
    }
    check_finite(p.distance, "place_ghost_dictionary: distance");
    if (p.distance < 0.0 || p.distance > gap.thickness) {
        invalid("place_ghost_dictionary: distance " + num(p.distance) + " außerhalb von [0, " +
                num(gap.thickness) + "]");
    }
    if (p.glasses.empty()) {
        invalid("place_ghost_dictionary: mindestens ein Glas nötig");
    }
    for (const MaterialSpec& g : p.glasses) {
        if (g.kind == MaterialKind::Air) {
            invalid("place_ghost_dictionary: Geisterplatten brauchen ein Glas");
        }
    }
    if (p.order < 1) {
        invalid("place_ghost_dictionary: Ordnung " + std::to_string(p.order) + " < 1");
    }
    if (p.penalty_weight && !(std::isfinite(*p.penalty_weight) && *p.penalty_weight > 0.0)) {
        invalid("place_ghost_dictionary: penalty_weight muss endlich und > 0 sein");
    }
    const double r_g = p.norm_radius > 0.0
                           ? p.norm_radius
                           : std::max(gap.semi_diameter, system.surfaces[sz(i + 1)].semi_diameter);
    check_finite(r_g, "place_ghost_dictionary: norm_radius");

    const int G = static_cast<int>(system.ghost_plates.size());
    const int m = static_cast<int>(p.glasses.size());
    int k = 0;
    while (k < G && (system.ghost_plates[sz(k)].after_surface < i ||
                     (system.ghost_plates[sz(k)].after_surface == i &&
                      system.ghost_plates[sz(k)].distance <= p.distance))) {
        ++k;
    }
    GhostDictionaryResult r;
    r.plate_new_of_old.resize(sz(G));
    for (int g = 0; g < G; ++g) {
        r.plate_new_of_old[sz(g)] = g < k ? g : g + m;
    }
    for (int j = 0; j < m; ++j) {
        r.new_plates.push_back(k + j);
    }
    r.system = system;
    std::vector<GhostPlate> plates;
    for (int g = 0; g < k; ++g) {
        plates.push_back(system.ghost_plates[sz(g)]);
    }
    for (const MaterialSpec& glass : p.glasses) {
        GhostPlate gp;
        gp.after_surface = i;
        gp.distance = p.distance;
        gp.glass = glass;
        gp.norm_radius = r_g;
        gp.coefficients.assign(sz(p.order), 0.0);
        plates.push_back(std::move(gp));
    }
    for (int g = k; g < G; ++g) {
        plates.push_back(system.ghost_plates[sz(g)]);
    }
    r.system.ghost_plates = std::move(plates);

    // Variablen ghost_coefficient
    if (!r.system.variables) {
        r.system.variables = Variables{};
    }
    bool covered = false;
    for (VariableSpec& spec : r.system.variables->items) {
        if (spec.type != VariableType::GhostCoefficient) {
            continue;
        }
        if (spec.plate_selection == SurfaceSelection::One ||
            spec.plate_selection == SurfaceSelection::List) {
            map_plates(spec.plates, r.plate_new_of_old);
        } else {
            covered = true;
        }
    }
    if (!covered) {
        VariableSpec spec;
        spec.type = VariableType::GhostCoefficient;
        spec.selection = SurfaceSelection::None;
        spec.plate_selection = m == 1 ? SurfaceSelection::One : SurfaceSelection::List;
        spec.plates = r.new_plates;
        spec.comment = "Geisterplatten-Wörterbuch (place_ghost_dictionary)";
        r.system.variables->items.push_back(std::move(spec));
    }

    // Strafe ghost_penalty
    if (!r.system.constraints) {
        r.system.constraints = Constraints{};
    }
    std::vector<ConstraintSpec>& items = r.system.constraints->items;
    const ConstraintSpec* template_entry = nullptr;
    bool penalty_all = false;
    int first_list = -1;
    for (std::size_t e = 0; e < items.size(); ++e) {
        ConstraintSpec& c = items[e];
        if (c.type != ConstraintType::GhostPenalty) {
            continue;
        }
        if (template_entry == nullptr) {
            template_entry = &c;
        }
        if (c.plate_selection == SurfaceSelection::One ||
            c.plate_selection == SurfaceSelection::List) {
            map_plates(c.plates, r.plate_new_of_old);
            if (first_list < 0) {
                first_list = static_cast<int>(e);
            }
        } else if (p.penalty_weight) {
            // "all" deckt sonst auch die neuen Platten ab: Liste der alten Platten
            c.plates = r.plate_new_of_old;
            c.plate_selection = SurfaceSelection::List;
            if (first_list < 0) {
                first_list = static_cast<int>(e);
            }
        } else {
            penalty_all = true;
        }
    }
    if (p.penalty_weight) {
        ConstraintSpec c;
        c.type = ConstraintType::GhostPenalty;
        c.weight = *p.penalty_weight;
        if (template_entry != nullptr) {
            c.order_weight = template_entry->order_weight;
            c.order_factor = template_entry->order_factor;
            c.smoothing = template_entry->smoothing;
        }
        c.plate_selection = m == 1 ? SurfaceSelection::One : SurfaceSelection::List;
        c.plates = r.new_plates;
        c.comment = "Geisterplatten-Wörterbuch (place_ghost_dictionary)";
        items.push_back(std::move(c));
    } else if (!penalty_all) {
        if (first_list < 0) {
            invalid("place_ghost_dictionary: keine Strafe ghost_penalty im System; "
                    "penalty_weight angeben");
        }
        ConstraintSpec& c = items[sz(first_list)];
        c.plates.insert(c.plates.end(), r.new_plates.begin(), r.new_plates.end());
        std::sort(c.plates.begin(), c.plates.end());
        c.plate_selection = SurfaceSelection::List;
    }
    for (ConstraintSpec& c : items) {
        c.shift.clear();
    }
    validate_system(r.system);
    return r;
}

// ---------------------------------------------------------------------------------------------

namespace {

// Einfügen von `inserted` Flächen nach Dateifläche i (Spalt nach i wird geteilt): Abbildung,
// Dickenbezüge und mechanische Dicken-Bedingungen des Spalts gelten für das Teilstück nach der
// letzten neuen Fläche mit (gleicher Eintrag), Geisterplatten im Spalt nach der Lage.
SystemDescription insert_after(const SystemDescription& system, int i,
                               std::vector<Surface> inserted, double position,
                               double inserted_thickness, InsertionResult& r) {
    const int S = static_cast<int>(system.surfaces.size());
    const int m = static_cast<int>(inserted.size());
    std::vector<Surface> surfaces;
    surfaces.reserve(sz(S + m));
    for (int j = 0; j <= i; ++j) {
        surfaces.push_back(system.surfaces[sz(j)]);
    }
    surfaces.back().thickness = position;
    for (Surface& s : inserted) {
        surfaces.push_back(std::move(s));
    }
    for (int j = i + 1; j < S; ++j) {
        surfaces.push_back(system.surfaces[sz(j)]);
    }
    SurfaceRemap map;
    map.surface.resize(sz(S));
    for (int j = 0; j < S; ++j) {
        map.surface[sz(j)] = j <= i ? j : j + m;
    }
    map.gap = map.surface;
    r.new_of_old = map.surface;
    r.first_new_surface = i + 1;
    r.new_surfaces = m;
    SystemDescription out = remap_surfaces(system, map, std::move(surfaces));
    const int rear = i + m; // Fläche vor dem hinteren Teilstück des Spalts
    // Dickenbezüge und Spalt-Bedingungen: auch das hintere Teilstück (direkt hinter i)
    if (out.variables) {
        for (VariableSpec& spec : out.variables->items) {
            if (spec.type != VariableType::Thickness) {
                continue;
            }
            const auto it = std::find(spec.surfaces.begin(), spec.surfaces.end(), i);
            if (it != spec.surfaces.end()) {
                spec.surfaces.insert(it + 1, rear);
                spec.selection = SurfaceSelection::List;
            }
        }
    }
    if (out.constraints) {
        for (ConstraintSpec& c : out.constraints->items) {
            if (!is_gap_constraint(c.type)) {
                continue;
            }
            const auto it = std::find(c.surfaces.begin(), c.surfaces.end(), i);
            if (it != c.surfaces.end()) {
                c.surfaces.insert(it + 1, rear);
                c.surface_selection = SurfaceSelection::List;
            }
        }
    }
    for (std::size_t g = 0; g < system.ghost_plates.size(); ++g) {
        if (system.ghost_plates[g].after_surface != i) {
            continue;
        }
        const double d = system.ghost_plates[g].distance;
        if (d <= position) {
            continue;
        }
        if (d < position + inserted_thickness) {
            invalid("Einfügen: Geisterplatte " + std::to_string(g) + " (distance " + num(d) +
                    ") läge im eingefügten Glas");
        }
        out.ghost_plates[g].after_surface = rear;
        out.ghost_plates[g].distance = d - position - inserted_thickness;
    }
    return out;
}

void add_curvature_variables(SystemDescription& system, const NewCurvatureVariables& vars,
                             std::vector<int> surfaces, const char* comment) {
    if (!vars.enabled) {
        return;
    }
    VariableSpec spec;
    spec.type = VariableType::Curvature;
    spec.selection = surfaces.size() == 1 ? SurfaceSelection::One : SurfaceSelection::List;
    spec.surfaces = std::move(surfaces);
    spec.min = vars.min;
    spec.max = vars.max;
    spec.comment = comment;
    if (!system.variables) {
        system.variables = Variables{};
    }
    system.variables->items.push_back(std::move(spec));
}

} // namespace

AplanaticShellResult insert_aplanatic_shell(const SystemDescription& system,
                                            const GlassCatalog& catalog,
                                            const AplanaticShell& shell) {
    const int S = static_cast<int>(system.surfaces.size());
    const int i = shell.after_surface;
    if (system.source.type != SourceType::Collimated || !is_coaxial(system)) {
        invalid("insert_aplanatic_shell: nur für koaxiale Systeme mit kollimierter Quelle");
    }
    if (i < 1 || i > S - 1) {
        invalid("insert_aplanatic_shell: after_surface " + std::to_string(i) +
                " außerhalb von [1, " + std::to_string(S - 1) + "]");
    }
    const Surface& gap = system.surfaces[sz(i)];
    if (gap.material.kind != MaterialKind::Air) {
        invalid("insert_aplanatic_shell: nach Fläche " + std::to_string(i) +
                " liegt kein Luftspalt");
    }
    if (shell.glass.kind == MaterialKind::Air) {
        invalid("insert_aplanatic_shell: die Schale braucht ein Glas");
    }
    check_finite(shell.position, "insert_aplanatic_shell: position");
    check_finite(shell.thickness, "insert_aplanatic_shell: thickness");
    check_finite(shell.semi_diameter, "insert_aplanatic_shell: semi_diameter");
    const double T = gap.thickness;
    if (shell.position < 0.0 || !(shell.thickness > 0.0) || shell.position + shell.thickness > T) {
        invalid("insert_aplanatic_shell: Schale (position " + num(shell.position) + ", Dicke " +
                num(shell.thickness) + ") passt nicht in den Spalt der Dicke " + num(T));
    }

    // Realer Randstrahl ρ = 1 auf der Achse bei λ_ref, ohne Geisterplatten
    const double lambda = system.spectrum.reference_nm;
    SystemDescription ref = system;
    ref.ghost_plates.clear();
    ref.spectrum.wavelengths_nm = {lambda};
    ref.spectrum.weights = {1.0};
    const IndexTable index = make_system_index_table(ref, catalog);
    const HostSurfaceTable table = make_surface_table(ref, index);
    const double a = system.surfaces[0].semi_diameter;
    Ray<double> ray = collimated_ray<double>(a, 0.0, 1.0, 0.0);
    auto rec = std::make_unique<TraceRecord<double>>();
    int failed = -1;
    if (trace_ray(table.view(), 0, ray, rec.get(), &failed) != TraceStatus::Ok) {
        invalid("insert_aplanatic_shell: der reale Randstrahl versagt an Fläche " +
                std::to_string(failed));
    }
    const Vec3<double>& p = rec->p[sz(i)];
    const Vec3<double>& d = rec->d[sz(i)];
    if (d.y == 0.0) {
        invalid("insert_aplanatic_shell: der Randstrahl ist nach Fläche " + std::to_string(i) +
                " achsparallel (kein Bildpunkt)");
    }
    const double L = p.z - p.y * d.z / d.y; // Achsenschnittpunkt ab dem Scheitel von i
    const double n_m = material_index(gap.material, lambda, catalog);
    const double n_g = material_index(shell.glass, lambda, catalog);
    const double q = L - shell.position;
    const double q_img = q * n_m / n_g;
    const double r1 = q * n_m / (n_m + n_g);
    const double r2 = q_img - shell.thickness;
    if (!(std::isfinite(q) && q != 0.0 && std::isfinite(r2) && r2 != 0.0)) {
        invalid("insert_aplanatic_shell: Schnittpunkt Q = " + num(q) + " mm bzw. R₂ = " + num(r2) +
                " mm ungeeignet (Schale am Bildpunkt)");
    }
    const double semi =
        shell.semi_diameter > 0.0
            ? shell.semi_diameter
            : std::max(gap.semi_diameter,
                       i + 1 < S ? system.surfaces[sz(i + 1)].semi_diameter : gap.semi_diameter);

    Surface first;
    first.type = SurfaceType::Sphere;
    first.curvature = 1.0 / r1;
    first.thickness = shell.thickness;
    first.material = shell.glass;
    first.semi_diameter = semi;
    first.comment = "aplanatische Schale, aplanatische Fläche";
    Surface second;
    second.type = SurfaceType::Sphere;
    second.curvature = 1.0 / r2;
    second.thickness = T - shell.position - shell.thickness;
    second.material = gap.material;
    second.semi_diameter = semi;
    second.comment = "aplanatische Schale, konzentrische Fläche";

    AplanaticShellResult r;
    r.n_glass = n_g;
    r.intercept = q;
    r.image_intercept = q_img;
    r.c_aplanatic = first.curvature;
    r.c_concentric = second.curvature;
    r.insertion.system = insert_after(system, i, {std::move(first), std::move(second)},
                                      shell.position, shell.thickness, r.insertion);
    SystemDescription& out = r.insertion.system;
    add_curvature_variables(out, shell.curvature_variables, {i + 1, i + 2}, "aplanatische Schale");
    if (shell.refocus) {
        const ParaxialData before = compute_paraxial(system, catalog);
        const ParaxialData after = compute_paraxial(out, catalog);
        const double defocus = system.surfaces.back().thickness - before.back_focal_distance;
        const double wd = after.back_focal_distance + defocus;
        if (!(std::isfinite(wd) && wd > 0.0)) {
            invalid("insert_aplanatic_shell: Arbeitsabstand nach dem Nachfokussieren " + num(wd) +
                    " mm");
        }
        out.surfaces.back().thickness = wd;
    }
    r.working_distance = out.surfaces.back().thickness;
    validate_system(out);
    return r;
}

InsertionResult insert_null_doublet(const SystemDescription& system, const NullDoublet& doublet) {
    if (doublet.back_glass.kind == MaterialKind::Air) {
        invalid("insert_null_doublet: das zweite Glas fehlt");
    }
    MeniscusInsertion m;
    m.after_surface = doublet.after_surface;
    m.position = doublet.position;
    m.curvature = doublet.curvature;
    m.glass = doublet.front_glass;
    m.semi_diameter = doublet.semi_diameter;
    m.curvature_variables.enabled = false;
    InsertionResult a = insert_meniscus(system, m);
    LensSplit s;
    s.lens_surface = a.first_new_surface;
    s.curvature = doublet.curvature;
    s.fraction = 0.5;
    s.semi_diameter = doublet.semi_diameter;
    s.curvature_variable.enabled = false;
    InsertionResult b = split_lens(a.system, s);
    const int f = a.first_new_surface;
    b.system.surfaces[sz(f)].comment = "Null-Kittglied, erste Fläche";
    b.system.surfaces[sz(f + 1)].comment = "Null-Kittglied, Kittfläche";
    b.system.surfaces[sz(f + 1)].material = doublet.back_glass;
    b.system.surfaces[sz(f + 2)].comment = "Null-Kittglied, letzte Fläche";
    add_curvature_variables(b.system, doublet.curvature_variables, {f, f + 1, f + 2},
                            "Null-Kittglied");
    InsertionResult r;
    r.system = std::move(b.system);
    r.new_of_old = a.new_of_old;
    for (int& j : r.new_of_old) {
        j = b.new_of_old[sz(j)];
    }
    r.first_new_surface = f;
    r.new_surfaces = 3;
    validate_system(r.system);
    return r;
}

} // namespace mko
