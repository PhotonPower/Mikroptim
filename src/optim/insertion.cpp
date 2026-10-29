// Einfüge-Operatoren der Systemsynthese (M6-T2): Umnummerierung der Flächenbezüge, Operator A
// (Meniskus ohne Brechkraft), Operator B (Kittglied), Initialisierung aus dem Diagnosebericht
// (docs/optimierung.md, Abschnitt 6).

#include "mikroptim/optim/insertion.hpp"

#include "mikroptim/glass/glass_preselection.hpp"

#include "mikroptim/glass/catalog.hpp"
#include "mikroptim/glass/index_table.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/model/variables.hpp"
#include "mikroptim/optim/ghost_diagnosis.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
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

// Fügt `added` in die Flächenauswahl jedes Eintrags vom Typ `type` ein, der `surface` enthält
// (direkt dahinter; One wird zur Liste). Die Einträge sind nach renumber_surfaces explizit.
void extend_entries(SystemDescription& system, VariableType type, int surface, int added) {
    if (!system.variables) {
        return;
    }
    for (VariableSpec& spec : system.variables->items) {
        if (spec.type != type) {
            continue;
        }
        const auto it = std::find(spec.surfaces.begin(), spec.surfaces.end(), surface);
        if (it == spec.surfaces.end()) {
            continue;
        }
        spec.surfaces.insert(it + 1, added);
        spec.selection = SurfaceSelection::List;
    }
}

// Mechanische Bedingungen eines Spalts (Mitten- und Randdicke, Fläche i = Spalt i → i + 1).
bool is_gap_constraint(ConstraintType t) {
    return t == ConstraintType::MinCenterThickness || t == ConstraintType::MaxCenterThickness ||
           t == ConstraintType::MinEdgeThickness;
}

// Fügt den Spalt `added` in jeden mechanischen Spalt-Eintrag ein, der den geteilten Spalt `gap`
// enthält (direkt dahinter; One wird zur Liste). Die Einträge sind nach renumber_surfaces
// explizit.
void extend_gap_constraints(SystemDescription& system, int gap, int added) {
    if (!system.constraints) {
        return;
    }
    for (ConstraintSpec& c : system.constraints->items) {
        if (!is_gap_constraint(c.type)) {
            continue;
        }
        const auto it = std::find(c.surfaces.begin(), c.surfaces.end(), gap);
        if (it == c.surfaces.end()) {
            continue;
        }
        c.surfaces.insert(it + 1, added);
        c.surface_selection = SurfaceSelection::List;
    }
}

void add_curvature_entry(SystemDescription& system, const NewCurvatureVariables& vars,
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

void check_finite(double v, const char* what) {
    if (!std::isfinite(v)) {
        invalid(std::string(what) + " muss endlich sein, ist " + num(v));
    }
}

// Platten, die ein Eintrag ghost_penalty abdeckt (None/All: alle).
std::vector<int> penalty_plates(const ConstraintSpec& c, int plates) {
    if (c.plate_selection == SurfaceSelection::None || c.plate_selection == SurfaceSelection::All) {
        std::vector<int> all(sz(plates));
        for (int g = 0; g < plates; ++g) {
            all[sz(g)] = g;
        }
        return all;
    }
    return c.plates;
}

// Eintrag ghost_penalty, der die Platte abdeckt (−1: keiner).
int penalty_entry(const SystemDescription& system, int plate) {
    if (!system.constraints) {
        return -1;
    }
    const int plates = static_cast<int>(system.ghost_plates.size());
    const std::vector<ConstraintSpec>& items = system.constraints->items;
    for (std::size_t e = 0; e < items.size(); ++e) {
        if (items[e].type != ConstraintType::GhostPenalty) {
            continue;
        }
        const std::vector<int> covered = penalty_plates(items[e], plates);
        if (std::find(covered.begin(), covered.end(), plate) != covered.end()) {
            return static_cast<int>(e);
        }
    }
    return -1;
}

double sign_of(double v) {
    return v < 0.0 ? -1.0 : 1.0;
}

} // namespace

// ---------------------------------------------------------------------------------------------

SystemDescription renumber_surfaces(const SystemDescription& system,
                                    std::span<const int> new_of_old,
                                    std::vector<Surface> new_surfaces) {
    const int n_old = static_cast<int>(system.surfaces.size());
    const int n_new = static_cast<int>(new_surfaces.size());
    if (static_cast<int>(new_of_old.size()) != n_old || n_old < 1) {
        invalid("renumber_surfaces: Abbildung hat " + std::to_string(new_of_old.size()) +
                " Einträge, das System " + std::to_string(n_old) + " Flächen");
    }
    if (new_of_old.front() != 0 || new_of_old.back() != n_new - 1) {
        invalid("renumber_surfaces: Blende muss Fläche 0 und die letzte Fläche die letzte "
                "bleiben (Abbildung " +
                std::to_string(new_of_old.front()) + " … " + std::to_string(new_of_old.back()) +
                ", neue Flächenzahl " + std::to_string(n_new) + ")");
    }
    for (int i = 1; i < n_old; ++i) {
        if (!(new_of_old[sz(i)] > new_of_old[sz(i - 1)])) {
            invalid("renumber_surfaces: Abbildung nicht streng monoton bei Fläche " +
                    std::to_string(i));
        }
    }
    const auto map = [&](int i) {
        if (i < 0 || i >= n_old) {
            invalid("renumber_surfaces: Flächenbezug " + std::to_string(i) + " außerhalb von [0, " +
                    std::to_string(n_old - 1) + "]");
        }
        return new_of_old[sz(i)];
    };

    // "all" in die alte, aufgelöste Flächenliste umwandeln (vor dem Tausch der Flächen)
    const std::vector<ResolvedVariable> resolved = resolve_variables(system);
    SystemDescription out = system;
    out.surfaces = std::move(new_surfaces);
    if (out.variables) {
        std::vector<VariableSpec>& items = out.variables->items;
        for (std::size_t k = 0; k < items.size(); ++k) {
            VariableSpec& spec = items[k];
            if (!is_surface_type(spec.type)) {
                continue; // Arbeitsabstand (letzte Fläche bleibt letzte), Geisterkoeffizienten
            }
            if (spec.selection == SurfaceSelection::All) {
                std::vector<int> list;
                for (const ResolvedVariable& r : resolved) {
                    if (r.item == static_cast<int>(k) && r.component == 0) {
                        list.push_back(r.surface);
                    }
                }
                spec.surfaces = std::move(list);
                spec.selection = SurfaceSelection::List;
            }
            for (int& s : spec.surfaces) {
                s = map(s);
            }
        }
    }
    for (GhostPlate& g : out.ghost_plates) {
        g.after_surface = map(g.after_surface);
    }
    if (out.constraints) {
        for (ConstraintSpec& c : out.constraints->items) {
            if (is_mechanical(c.type)) {
                // Standardauswahl bzw. "all" → explizite Liste der alten Flächen (wie bei den
                // Variablen: eingefügte Flächen werden nicht unbemerkt eingeschränkt)
                if (c.surface_selection == SurfaceSelection::None ||
                    c.surface_selection == SurfaceSelection::All) {
                    c.surfaces = constraint_surfaces(system, c);
                    c.surface_selection = SurfaceSelection::List;
                }
                for (int& s : c.surfaces) {
                    s = map(s);
                }
            }
            if (c.reference_surface) {
                c.reference_surface = map(*c.reference_surface);
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------------

InsertionResult insert_meniscus(const SystemDescription& system, const MeniscusInsertion& m) {
    const int S = static_cast<int>(system.surfaces.size());
    const int i = m.after_surface;
    if (i < 0 || i > S - 2) {
        invalid("insert_meniscus: after_surface " + std::to_string(i) + " außerhalb von [0, " +
                std::to_string(S - 2) + "] (nach der letzten Fläche liegt der Arbeitsabstand)");
    }
    const Surface& gap = system.surfaces[sz(i)];
    if (gap.material.kind != MaterialKind::Air) {
        invalid("insert_meniscus: das Medium nach Fläche " + std::to_string(i) +
                " ist kein Luftspalt");
    }
    if (m.glass.kind == MaterialKind::Air) {
        invalid("insert_meniscus: der Meniskus braucht ein Glas");
    }
    check_finite(m.position, "insert_meniscus: position");
    check_finite(m.curvature, "insert_meniscus: curvature");
    check_finite(m.thickness, "insert_meniscus: thickness");
    check_finite(m.semi_diameter, "insert_meniscus: semi_diameter");
    const double T = gap.thickness;
    const double rear = T - m.position - m.thickness;
    if (m.position < 0.0 || m.thickness < 0.0 || rear < 0.0) {
        invalid("insert_meniscus: Meniskus (position " + num(m.position) + ", Dicke " +
                num(m.thickness) + ") passt nicht in den Luftspalt der Dicke " + num(T));
    }
    const double semi = m.semi_diameter > 0.0
                            ? m.semi_diameter
                            : std::max(gap.semi_diameter, system.surfaces[sz(i + 1)].semi_diameter);

    std::vector<Surface> surfaces;
    surfaces.reserve(sz(S + 2));
    for (int j = 0; j <= i; ++j) {
        surfaces.push_back(system.surfaces[sz(j)]);
    }
    surfaces.back().thickness = m.position;
    Surface front;
    front.type = SurfaceType::Sphere;
    front.curvature = m.curvature;
    front.thickness = m.thickness;
    front.material = m.glass;
    front.semi_diameter = semi;
    front.comment = "Meniskus (Operator A), Vorderfläche";
    Surface back = front;
    back.thickness = rear;
    back.material = MaterialSpec{};
    back.comment = "Meniskus (Operator A), Rückfläche";
    surfaces.push_back(std::move(front));
    surfaces.push_back(std::move(back));
    for (int j = i + 1; j < S; ++j) {
        surfaces.push_back(system.surfaces[sz(j)]);
    }
    InsertionResult r;
    r.new_of_old.resize(sz(S));
    for (int j = 0; j < S; ++j) {
        r.new_of_old[sz(j)] = j <= i ? j : j + 2;
    }
    r.first_new_surface = i + 1;
    r.new_surfaces = 2;
    r.system = renumber_surfaces(system, r.new_of_old, std::move(surfaces));

    // Geisterplatten im geteilten Spalt (after_surface ist schon i, weil i auf i abgebildet wird)
    for (std::size_t g = 0; g < system.ghost_plates.size(); ++g) {
        if (system.ghost_plates[g].after_surface != i) {
            continue;
        }
        GhostPlate& gp = r.system.ghost_plates[g];
        const double d = gp.distance;
        if (d <= m.position) {
            continue; // vor dem Meniskus
        }
        if (d < m.position + m.thickness) {
            invalid("insert_meniscus: Geisterplatte " + std::to_string(g) + " (distance " + num(d) +
                    ") läge im Glas des Meniskus");
        }
        gp.after_surface = i + 2;
        gp.distance = d - m.position - m.thickness;
    }
    extend_entries(r.system, VariableType::Thickness, i, i + 2);
    extend_gap_constraints(r.system, i, i + 2);
    add_curvature_entry(r.system, m.curvature_variables, {i + 1, i + 2}, "Meniskus (Operator A)");
    validate_system(r.system);
    return r;
}

std::array<SystemDescription, 2> meniscus_saddle_starts(const SystemDescription& system,
                                                        int first_surface, double delta_front,
                                                        double delta_back) {
    const int S = static_cast<int>(system.surfaces.size());
    if (first_surface < 1 || first_surface + 1 > S - 1) {
        invalid("meniscus_saddle_starts: Fläche " + std::to_string(first_surface) +
                " ist nicht die Vorderfläche eines Meniskus");
    }
    for (int j = first_surface; j <= first_surface + 1; ++j) {
        if (system.surfaces[sz(j)].type != SurfaceType::Sphere) {
            invalid("meniscus_saddle_starts: Fläche " + std::to_string(j) + " ist keine Sphäre");
        }
    }
    check_finite(delta_front, "meniscus_saddle_starts: delta_front");
    check_finite(delta_back, "meniscus_saddle_starts: delta_back");
    std::array<SystemDescription, 2> out{system, system};
    for (int k = 0; k < 2; ++k) {
        const double s = k == 0 ? 1.0 : -1.0;
        out[sz(k)].surfaces[sz(first_surface)].curvature += s * delta_front;
        out[sz(k)].surfaces[sz(first_surface + 1)].curvature += s * delta_back;
    }
    return out;
}

// ---------------------------------------------------------------------------------------------

InsertionResult split_lens(const SystemDescription& system, const LensSplit& s) {
    const int S = static_cast<int>(system.surfaces.size());
    const int i = s.lens_surface;
    if (i < 1 || i > S - 2) {
        invalid("split_lens: lens_surface " + std::to_string(i) + " außerhalb von [1, " +
                std::to_string(S - 2) + "]");
    }
    const Surface& lens = system.surfaces[sz(i)];
    if (lens.material.kind == MaterialKind::Air) {
        invalid("split_lens: nach Fläche " + std::to_string(i) + " liegt Luft, keine Linse");
    }
    check_finite(s.curvature, "split_lens: curvature");
    check_finite(s.fraction, "split_lens: fraction");
    check_finite(s.semi_diameter, "split_lens: semi_diameter");
    if (s.fraction < 0.0 || s.fraction > 1.0) {
        invalid("split_lens: fraction muss in [0, 1] liegen, ist " + num(s.fraction));
    }
    const double t1 = s.fraction * lens.thickness;
    const double t2 = lens.thickness - t1;
    const double semi = s.semi_diameter > 0.0 ? s.semi_diameter
                                              : std::min(lens.semi_diameter,
                                                         system.surfaces[sz(i + 1)].semi_diameter);

    std::vector<Surface> surfaces;
    surfaces.reserve(sz(S + 1));
    for (int j = 0; j <= i; ++j) {
        surfaces.push_back(system.surfaces[sz(j)]);
    }
    surfaces.back().thickness = t1;
    Surface cement;
    cement.type = SurfaceType::Sphere;
    cement.curvature = s.curvature;
    cement.thickness = t2;
    cement.material = lens.material;
    cement.semi_diameter = semi;
    cement.comment = "Kittfläche (Operator B)";
    surfaces.push_back(std::move(cement));
    for (int j = i + 1; j < S; ++j) {
        surfaces.push_back(system.surfaces[sz(j)]);
    }
    InsertionResult r;
    r.new_of_old.resize(sz(S));
    for (int j = 0; j < S; ++j) {
        r.new_of_old[sz(j)] = j <= i ? j : j + 1;
    }
    r.first_new_surface = i + 1;
    r.new_surfaces = 1;
    r.system = renumber_surfaces(system, r.new_of_old, std::move(surfaces));
    // Geisterplatten liegen nur in Luftspalten: keine im geteilten Glas
    extend_entries(r.system, VariableType::Thickness, i, i + 1);
    extend_entries(r.system, VariableType::Glass, i, i + 1);
    extend_gap_constraints(r.system, i, i + 1);
    add_curvature_entry(r.system, s.curvature_variable, {i + 1}, "Kittfläche (Operator B)");
    validate_system(r.system);
    return r;
}

void free_glass(SystemDescription& system, int surface,
                const std::optional<MaterialSpec>& start_glass) {
    const int S = static_cast<int>(system.surfaces.size());
    if (surface < 1 || surface >= S) {
        invalid("free_glass: Fläche " + std::to_string(surface) + " außerhalb von [1, " +
                std::to_string(S - 1) + "]");
    }
    if (start_glass) {
        if (start_glass->kind == MaterialKind::Air) {
            invalid("free_glass: Startglas darf nicht Luft sein");
        }
        system.surfaces[sz(surface)].material = *start_glass;
    }
    if (system.surfaces[sz(surface)].material.kind == MaterialKind::Air) {
        invalid("free_glass: nach Fläche " + std::to_string(surface) + " liegt Luft");
    }
    for (const ResolvedVariable& v : resolve_variables(system)) {
        if (v.type == VariableType::Glass && v.surface == surface) {
            return; // schon Variable
        }
    }
    VariableSpec spec;
    spec.type = VariableType::Glass;
    spec.selection = SurfaceSelection::One;
    spec.surfaces = {surface};
    spec.comment = "Glas freigegeben (Operator B)";
    if (!system.variables) {
        system.variables = Variables{};
    }
    system.variables->items.push_back(std::move(spec));
}

double achromat_split_curvature(double c_front, double c_back, double abbe_front,
                                double abbe_back) {
    check_finite(c_front, "achromat_split_curvature: c_front");
    check_finite(c_back, "achromat_split_curvature: c_back");
    check_finite(abbe_front, "achromat_split_curvature: abbe_front");
    check_finite(abbe_back, "achromat_split_curvature: abbe_back");
    if (abbe_front == abbe_back) {
        invalid("achromat_split_curvature: gleiche Abbezahlen, kein Achromat möglich");
    }
    // c_s = c₁ − φ₁/(n − 1) mit φ₁ = (n − 1)(c₁ − c₂) ν₁/(ν₁ − ν₂)
    return c_front - (c_front - c_back) * abbe_front / (abbe_front - abbe_back);
}

double abbe_number(const MaterialSpec& material, const GlassCatalog& catalog) {
    const double nf = material_index(material, kLineF, catalog);
    const double nd = material_index(material, kLineD, catalog);
    const double nc = material_index(material, kLineC, catalog);
    const double dfc = nf - nc;
    if (dfc == 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return (nd - 1.0) / dfc;
}

// ---------------------------------------------------------------------------------------------

void scale_ghost_penalty(SystemDescription& system, int plate, double factor) {
    if (!(std::isfinite(factor) && factor > 0.0)) {
        invalid("scale_ghost_penalty: Faktor muss endlich und > 0 sein, ist " + num(factor));
    }
    const int plates = static_cast<int>(system.ghost_plates.size());
    if (plate < 0 || plate >= plates) {
        invalid("scale_ghost_penalty: Geisterplatte " + std::to_string(plate) + " existiert nicht");
    }
    const int e = penalty_entry(system, plate);
    if (e < 0) {
        invalid("scale_ghost_penalty: keine Strafe ghost_penalty für Geisterplatte " +
                std::to_string(plate));
    }
    std::vector<ConstraintSpec>& items = system.constraints->items;
    ConstraintSpec& c = items[sz(e)];
    const double weight = c.weight_or_default() * factor;
    std::vector<int> covered = penalty_plates(c, plates);
    if (covered.size() == 1) {
        c.weight = weight;
        return;
    }
    // Platte aus dem gemeinsamen Eintrag lösen, eigener Eintrag mit erhöhtem Gewicht
    ConstraintSpec own = c;
    covered.erase(std::find(covered.begin(), covered.end(), plate));
    c.plate_selection = covered.size() == 1 ? SurfaceSelection::One : SurfaceSelection::List;
    c.plates = std::move(covered);
    own.plate_selection = SurfaceSelection::One;
    own.plates = {plate};
    own.weight = weight;
    own.comment = "Strafgewicht der Geisterplatte " + std::to_string(plate) + " um den Faktor " +
                  num(factor) + " erhöht (Einfüge-Operator)";
    items.insert(items.begin() + e + 1, std::move(own));
}

std::optional<double> ghost_penalty_weight(const SystemDescription& system, int plate) {
    const int e = penalty_entry(system, plate);
    if (e < 0) {
        return std::nullopt;
    }
    return system.constraints->items[sz(e)].weight_or_default();
}

double ghost_profile_t4(const GhostPlate& plate) {
    double sum = 0.0;
    for (std::size_t j = 0; j < plate.coefficients.size(); ++j) {
        const double k = static_cast<double>(j + 1);
        // q_k = (−1)^k C(k, 2) C(k + 2, 2) = (−1)^k · k(k − 1)/2 · (k + 2)(k + 1)/2
        const double q =
            ((j + 1) % 2 == 0 ? 1.0 : -1.0) * (k * (k - 1.0) / 2.0) * ((k + 2.0) * (k + 1.0) / 2.0);
        sum += plate.coefficients[j] * q;
    }
    const double r2 = plate.norm_radius * plate.norm_radius;
    return sum / (r2 * r2);
}

namespace {

// Lage aus dem Bericht und Abgleich mit dem System.
const GhostPositionReport& position_of(const SystemDescription& system,
                                       const GhostDiagnosis& diagnosis, int requested, int& index) {
    index = requested < 0 ? diagnosis.recommended_position : requested;
    if (index < 0 || index >= static_cast<int>(diagnosis.positions.size())) {
        invalid("Einfüge-Operator: keine (gültige) Lage im Diagnosebericht (Index " +
                std::to_string(index) + ")");
    }
    const GhostPositionReport& pr = diagnosis.positions[sz(index)];
    if (pr.active_plates.empty()) {
        invalid("Einfüge-Operator: an der Lage nach Fläche " + std::to_string(pr.after_surface) +
                " ist keine Geisterplatte aktiv");
    }
    for (const int g : pr.plates) {
        if (g < 0 || g >= static_cast<int>(system.ghost_plates.size()) ||
            g >= static_cast<int>(diagnosis.plates.size()) ||
            system.ghost_plates[sz(g)].after_surface != pr.after_surface ||
            system.ghost_plates[sz(g)].distance != pr.distance) {
            invalid("Einfüge-Operator: Diagnosebericht passt nicht zum System (Geisterplatte " +
                    std::to_string(g) + ")");
        }
    }
    if (pr.after_surface < 0 || pr.after_surface > static_cast<int>(system.surfaces.size()) - 2) {
        invalid("Einfüge-Operator: Lage nach Fläche " + std::to_string(pr.after_surface) +
                " passt nicht zum System");
    }
    return pr;
}

} // namespace

MeniscusPlan meniscus_from_diagnosis(const SystemDescription& system,
                                     const GhostDiagnosis& diagnosis,
                                     const MeniscusInitOptions& options) {
    MeniscusPlan plan;
    const GhostPositionReport& pr = position_of(system, diagnosis, options.position, plan.position);
    int best = pr.active_plates.front();
    for (const int g : pr.active_plates) {
        if (diagnosis.plates[sz(g)].sum_abs > diagnosis.plates[sz(best)].sum_abs) {
            best = g;
        }
    }
    plan.plate = best;
    const GhostPlate& gp = system.ghost_plates[sz(best)];
    const GhostPlateReport& rep = diagnosis.plates[sz(best)];
    plan.t4 = ghost_profile_t4(gp);
    // Vorzeichen: (n_G − n_m)·t₄ (Phase der sphärischen Korrektur), sonst φ_G
    const double dn = rep.glass.n_ref - 1.0;
    double sign = 1.0;
    const char* from = "Vorgabe +";
    if (plan.t4 != 0.0 && dn != 0.0) {
        sign = sign_of(plan.t4 * dn);
        from = "(n_G − 1)·t4";
    } else if (rep.power != 0.0) {
        sign = sign_of(rep.power);
        from = "phi_G";
    }
    // Betrag: Vorgabe, sonst die nähere Spaltfläche, sonst die andere, sonst 0
    const int i = pr.after_surface;
    double magnitude = options.curvature_magnitude;
    const char* mag_from = "Vorgabe";
    if (!(magnitude > 0.0)) {
        const double T = system.surfaces[sz(i)].thickness;
        const bool front_nearer = pr.distance <= 0.5 * T;
        const double c_near = system.surfaces[sz(front_nearer ? i : i + 1)].curvature;
        const double c_other = system.surfaces[sz(front_nearer ? i + 1 : i)].curvature;
        magnitude = c_near != 0.0 ? std::abs(c_near) : std::abs(c_other);
        mag_from = c_near != 0.0 ? "nähere Spaltfläche"
                                 : (c_other != 0.0 ? "andere Spaltfläche" : "Planplatte");
    }
    plan.insertion.after_surface = i;
    plan.insertion.position = pr.distance;
    plan.insertion.curvature = sign * magnitude;
    plan.insertion.glass = gp.glass;
    plan.insertion.thickness = options.thickness;
    plan.insertion.curvature_variables = options.curvature_variables;
    std::ostringstream os;
    os << "Operator A nach Fläche " << i << " bei " << num(pr.distance) << " mm, Glas der Platte "
       << best << " (" << rep.glass.name << "), c0 = " << num(plan.insertion.curvature)
       << " (Betrag: " << mag_from << ", Vorzeichen: " << from << ", t4 = " << num(plan.t4) << ")";
    plan.reason = os.str();
    return plan;
}

namespace {

// Gemeinsamer Teil von Operator B (M6-T2, M11-T5): Zielglas (ggf. aus der Glasvorauswahl),
// Kittkrümmung (dünnes Achromat mit Rücknahme bei zu kleiner Randdicke), Begründung. `plan`
// bringt Platte und Zielglas der Diagnose mit (bei der Farbregel: Platte −1).
SplitPlan finish_split(const SystemDescription& system, const GlassCatalog& catalog, int lens,
                       SplitPlan plan, const SplitInitOptions& options, bool anomalous,
                       const std::string& source) {
    const MaterialSpec& lens_glass = system.surfaces[sz(lens)].material;
    plan.secondary_spectrum = std::numeric_limits<double>::quiet_NaN();
    // Glasvorauswahl (M11-T5): Partner des Linsenglases aus dem Katalog
    double preselect_front_fraction = std::numeric_limits<double>::quiet_NaN();
    if (options.preselect && lens_glass.kind == MaterialKind::Catalog) {
        const std::vector<GlassDispersion> g = catalog_dispersions(catalog);
        const auto it = std::find_if(g.begin(), g.end(), [&](const GlassDispersion& d) {
            return d.name == lens_glass.name;
        });
        if (it != g.end()) {
            GlassPreselectionOptions po;
            po.fixed = {static_cast<int>(it - g.begin())};
            po.max_power_sum = options.preselect_max_power_sum;
            po.max_results = 1;
            const std::vector<PreselectedGlasses> best = preselect_glasses(g, po);
            if (!best.empty()) {
                plan.preselected = true;
                plan.target_glass = MaterialSpec{};
                plan.target_glass.kind = MaterialKind::Catalog;
                plan.target_glass.name = g[sz(best[0].glasses[1])].name;
                plan.secondary_spectrum = best[0].residual;
                // Anteil des vorderen Teils an der Brechkraft (φ = 1): hinten frei → vorn Linse
                preselect_front_fraction =
                    options.free_part == 1 ? best[0].powers[0] : best[0].powers[1];
            }
        }
    }
    plan.abbe_target = abbe_number(plan.target_glass, catalog);

    const Surface& front = system.surfaces[sz(lens)];
    const Surface& back = system.surfaces[sz(lens + 1)];
    const double c1 = front.curvature;
    const double c2 = back.curvature;
    const double mean = 0.5 * (c1 + c2);
    double cs = mean;
    std::string cs_from = "Mittel der Außenkrümmungen";
    plan.achromat_curvature = std::numeric_limits<double>::quiet_NaN();
    if (options.achromat_curvature && std::isfinite(plan.abbe_lens) &&
        std::isfinite(plan.abbe_target) && plan.abbe_lens != plan.abbe_target) {
        const double front_abbe = options.free_part == 1 ? plan.abbe_lens : plan.abbe_target;
        const double back_abbe = options.free_part == 1 ? plan.abbe_target : plan.abbe_lens;
        plan.achromat_curvature = achromat_split_curvature(c1, c2, front_abbe, back_abbe);
        if (plan.preselected) {
            // dünnes Achromat in η (Glasvorauswahl): φ_vorn = Anteil · φ, c_s = c₁ − Anteil (c₁ −
            // c₂)
            plan.achromat_curvature = c1 - preselect_front_fraction * (c1 - c2);
        }
        // Randdicken der Teile bei der Höhe h (Pfeilhöhe c h²/(1 + √(1 − c²h²)))
        const double t1 = options.fraction * front.thickness;
        const double t2 = front.thickness - t1;
        const auto sag = [](double ci, double h) {
            return ci * h * h / (1.0 + std::sqrt(1.0 - ci * ci * h * h));
        };
        // Prüfhöhe: kleinerer Halbdurchmesser, höchstens 0,99/|c| der Außenflächen; hat schon
        // die ungeteilte Linse dort eine zu kleine Randdicke (inkonsistente Halbdurchmesser),
        // die größte Höhe, bei der sie passt (Bisektion)
        double h0 = std::min(front.semi_diameter, back.semi_diameter);
        for (const double ci : {c1, c2}) {
            if (ci != 0.0) {
                h0 = std::min(h0, 0.99 / std::abs(ci));
            }
        }
        const auto lens_edge = [&](double h) { return front.thickness + sag(c2, h) - sag(c1, h); };
        if (lens_edge(h0) < options.min_edge_thickness) {
            double lo = 0.0;
            double hi = h0;
            for (int it = 0; it < 60; ++it) {
                const double mid = 0.5 * (lo + hi);
                (lens_edge(mid) >= options.min_edge_thickness ? lo : hi) = mid;
            }
            h0 = lo;
        }
        const auto feasible = [&](double c) {
            const double h = c != 0.0 ? std::min(h0, 0.99 / std::abs(c)) : h0;
            const double e1 = t1 + sag(c, h) - sag(c1, h);
            const double e2 = t2 + sag(c2, h) - sag(c, h);
            return e1 >= options.min_edge_thickness && e2 >= options.min_edge_thickness;
        };
        cs = plan.achromat_curvature;
        cs_from = "dünnes Achromat";
        if (!feasible(cs)) {
            // größtes λ ∈ [0, 1] mit mean + λ (c_A − mean) zulässig
            double lo = 0.0;
            double hi = 1.0;
            for (int it = 0; it < 60; ++it) {
                const double mid = 0.5 * (lo + hi);
                if (feasible(mean + mid * (plan.achromat_curvature - mean))) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            cs = mean + lo * (plan.achromat_curvature - mean);
            cs_from = "dünnes Achromat (c_s = " + num(plan.achromat_curvature) +
                      ") wegen der Randdicke zum Mittel hin zurückgenommen, Anteil " + num(lo);
        }
    }
    plan.split.lens_surface = lens;
    plan.split.curvature = cs;
    plan.split.fraction = options.fraction;
    plan.split.curvature_variable = options.curvature_variable;
    plan.free_surface = lens + options.free_part;
    std::ostringstream os;
    os << "Operator " << (anomalous ? "B (anomal)" : "B") << ": Linse " << lens << "/" << lens + 1
       << " (nu_d = " << num(plan.abbe_lens) << "), Zielglas "
       << (plan.preselected ? "aus der Glasvorauswahl " + plan.target_glass.name +
                                  " (sekundäres Spektrum " + num(plan.secondary_spectrum) + ")"
                            : "der Platte " + std::to_string(plan.plate))
       << (source.empty() ? std::string() : " [" + source + "]")
       << " (nu_d = " << num(plan.abbe_target) << "), c_s = " << num(cs) << " (" << cs_from
       << "), freigegeben: Teil " << options.free_part << " (Fläche " << plan.free_surface << ")";
    plan.reason = os.str();
    return plan;
}

} // namespace

SplitPlan split_from_diagnosis(const SystemDescription& system, const GlassCatalog& catalog,
                               const GhostDiagnosis& diagnosis, const SplitInitOptions& options) {
    if (options.free_part != 0 && options.free_part != 1) {
        invalid("split_from_diagnosis: free_part muss 0 oder 1 sein");
    }
    SplitPlan plan;
    const GhostPositionReport& pr = position_of(system, diagnosis, options.position, plan.position);
    const int S = static_cast<int>(system.surfaces.size());
    const int i = pr.after_surface;
    const auto is_lens = [&](int front) {
        return front >= 1 && front + 1 <= S - 1 &&
               system.surfaces[sz(front)].material.kind != MaterialKind::Air;
    };
    const double T = system.surfaces[sz(i)].thickness;
    int lens = -1;
    const bool before = is_lens(i - 1);
    const bool after = is_lens(i + 1);
    if (before && after) {
        lens = pr.distance <= T - pr.distance ? i - 1 : i + 1;
    } else if (before) {
        lens = i - 1;
    } else if (after) {
        lens = i + 1;
    } else {
        invalid("split_from_diagnosis: am Luftspalt nach Fläche " + std::to_string(i) +
                " liegt keine Linse");
    }
    const MaterialSpec& lens_glass = system.surfaces[sz(lens)].material;
    plan.abbe_lens = abbe_number(lens_glass, catalog);

    // Zielglas unter den aktiven Platten
    const bool anomalous = pr.recommendation == GhostOperator::BAnomalous;
    double best_score = -std::numeric_limits<double>::infinity();
    for (const int g : pr.active_plates) {
        const GhostGlassData& gd = diagnosis.plates[sz(g)].glass;
        double score = 0.0;
        if (anomalous) {
            score = gd.delta_partial;
        } else {
            const double abbe = abbe_number(system.ghost_plates[sz(g)].glass, catalog);
            score = std::abs(1.0 / abbe - 1.0 / plan.abbe_lens);
        }
        if (std::isfinite(score) && score > best_score) {
            best_score = score;
            plan.plate = g;
        }
    }
    if (plan.plate < 0) {
        plan.plate = pr.active_plates.front();
    }
    plan.target_glass = system.ghost_plates[sz(plan.plate)].glass;
    return finish_split(system, catalog, lens, plan, options, anomalous, "");
}

SplitPlan split_for_color(const SystemDescription& system, const GlassCatalog& catalog, int lens,
                          const SplitInitOptions& options) {
    const int S = static_cast<int>(system.surfaces.size());
    if (lens < 1 || lens + 1 > S - 1 ||
        system.surfaces[sz(lens)].material.kind == MaterialKind::Air) {
        invalid("split_for_color: an Fläche " + std::to_string(lens) + " beginnt keine Linse");
    }
    if (options.free_part != 0 && options.free_part != 1) {
        invalid("split_for_color: free_part muss 0 oder 1 sein");
    }
    SplitInitOptions o = options;
    o.preselect = true;
    SplitPlan plan;
    plan.abbe_lens = abbe_number(system.surfaces[sz(lens)].material, catalog);
    plan.target_glass = system.surfaces[sz(lens)].material; // ersetzt die Vorauswahl
    plan = finish_split(system, catalog, lens, plan, o, false, "Farbregel");
    if (!plan.preselected) {
        invalid("split_for_color: kein Partner für das Glas der Linse an Fläche " +
                std::to_string(lens) + " in der Glasvorauswahl");
    }
    return plan;
}

} // namespace mko
