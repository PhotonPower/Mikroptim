// Abbildung Variablenvektor ↔ SystemBatch (docs/optimierung.md, „Variablenabbildung“).

#include "mikroptim/optim/variable_set.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace mko {

namespace {

SurfaceKind kind_of(SurfaceType type) {
    switch (type) {
    case SurfaceType::Stop:
        return SurfaceKind::Stop;
    case SurfaceType::Plane:
        return SurfaceKind::Plane;
    case SurfaceType::Sphere:
        return SurfaceKind::Sphere;
    }
    return SurfaceKind::Plane;
}

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument("VariableSet: " + message);
}

} // namespace

SurfaceMap checked_surface_map(const SystemDescription& system, const BatchTopology& topology) {
    SurfaceMap map = make_surface_map(system);
    if (static_cast<int>(topology.type.size()) != topology.surfaces ||
        topology.surfaces != map.table_surfaces()) {
        invalid("die Topologie hat " + std::to_string(topology.surfaces) +
                " Flächen, die Systemdatei ergibt " + std::to_string(map.table_surfaces()) +
                " Tabellenflächen");
    }
    if (topology.ghost_plates != static_cast<int>(system.ghost_plates.size())) {
        invalid("Zahl der Geisterplatten der Topologie passt nicht zur Systemdatei");
    }
    for (int j = 0; j < topology.surfaces; ++j) {
        const int file = map.file_of_table[static_cast<std::size_t>(j)];
        const SurfaceKind expected =
            file >= 0 ? kind_of(system.surfaces[static_cast<std::size_t>(file)].type)
                      : SurfaceKind::GhostPlate;
        if (topology.type[static_cast<std::size_t>(j)] != expected) {
            invalid("Flächentyp der Tabellenfläche " + std::to_string(j) +
                    " passt nicht zur Systemdatei");
        }
    }
    return map;
}

std::string variable_name(VariableType type, int file_surface) {
    switch (type) {
    case VariableType::Curvature:
        return "c[" + std::to_string(file_surface) + "]";
    case VariableType::Thickness:
        return "t[" + std::to_string(file_surface) + "]";
    case VariableType::WorkingDistance:
        return "w";
    case VariableType::GhostCoefficient:
        break; // Name aus ghost_variable_name
    }
    return "?";
}

std::string ghost_variable_name(int plate, int order) {
    return "a[" + std::to_string(plate) + "," + std::to_string(order) + "]";
}

VariableSet::VariableSet(const SystemDescription& system, const BatchTopology& topology)
    : topology_(topology), file_surfaces_(static_cast<int>(system.surfaces.size())) {
    validate_topology(topology_);
    const std::vector<ResolvedVariable> resolved = resolve_variables(system);
    const SurfaceMap map = checked_surface_map(system, topology_);

    file_types_.reserve(system.surfaces.size());
    for (const Surface& s : system.surfaces) {
        file_types_.push_back(s.type);
    }
    for (const GhostPlate& g : system.ghost_plates) {
        ghost_orders_.push_back(static_cast<int>(g.coefficients.size()));
    }

    variables_.reserve(resolved.size());
    for (const ResolvedVariable& r : resolved) {
        Variable var;
        var.type = r.type;
        var.file_surface = r.surface;
        var.lower = r.lower;
        var.upper = r.upper;
        var.typical = r.typical;
        var.name = variable_name(r.type, r.surface);
        const std::size_t i = static_cast<std::size_t>(r.surface);
        switch (r.type) {
        case VariableType::Curvature:
            var.batch_surface = map.table_of_file[i];
            break;
        case VariableType::Thickness: {
            // Dicke der Dateifläche i = Spalt bis zur Dateifläche i + 1. Liegen Geisterplatten
            // darin, ist er im Batch geteilt (make_surface_table: d_1, d_2 − d_1, …, T − d_m);
            // die Variable wirkt auf das letzte Teilstück, die Abstände d_g bleiben fest.
            var.batch_surface = map.table_of_file[i + 1] - 1;
            const int last_ghost = map.ghost_of_table[static_cast<std::size_t>(var.batch_surface)];
            if (last_ghost >= 0) {
                var.split = true;
                var.offset = system.ghost_plates[static_cast<std::size_t>(last_ghost)].distance;
                // Die Platte muss im Spalt bleiben (validate_system: distance ≤ Dicke).
                var.lower = std::max(var.lower, var.offset);
            }
            break;
        }
        case VariableType::WorkingDistance:
            // Arbeitsabstand = thickness[S−1] (einzige Quelle, docs/architektur.md); nach der
            // letzten Fläche liegen keine Geisterplatten (validate_system).
            var.batch_surface = topology_.surfaces - 1;
            break;
        case VariableType::GhostCoefficient:
            // ghost_coeff[v][g][k − 1]; K der Topologie ist die größte Ordnung aller Platten
            // (kürzere Profile mit 0 ergänzt), also k ≤ ghost_terms.
            if (r.order > topology_.ghost_terms) {
                invalid("Ordnung " + std::to_string(r.order) + " der Geisterplatte " +
                        std::to_string(r.plate) + " größer als K = " +
                        std::to_string(topology_.ghost_terms) + " der Topologie");
            }
            var.ghost_plate = r.plate;
            var.ghost_order = r.order;
            var.batch_surface = map.table_of_ghost[static_cast<std::size_t>(r.plate)];
            var.name = ghost_variable_name(r.plate, r.order);
            break;
        }
        lower_.push_back(var.lower);
        upper_.push_back(var.upper);
        typical_.push_back(var.typical);
        variables_.push_back(std::move(var));
    }
}

const Variable& VariableSet::operator[](int k) const {
    if (k < 0 || k >= size()) {
        throw std::out_of_range("VariableSet: Variable " + std::to_string(k) +
                                " außerhalb von [0, " + std::to_string(size()) + ")");
    }
    return variables_[static_cast<std::size_t>(k)];
}

void VariableSet::check_size(std::size_t n, const char* what) const {
    if (n != variables_.size()) {
        invalid(std::string(what) + " hat Länge " + std::to_string(n) + ", erwartet " +
                std::to_string(variables_.size()));
    }
}

void VariableSet::check_batch(const SystemBatchHost& batch) const {
    if (!(batch.topology() == topology_)) {
        invalid("Batch mit anderer Topologie als bei der Erzeugung der Variablenmenge");
    }
}

void VariableSet::check_description(const SystemDescription& system) const {
    bool same = static_cast<int>(system.surfaces.size()) == file_surfaces_;
    for (std::size_t i = 0; same && i < system.surfaces.size(); ++i) {
        same = system.surfaces[i].type == file_types_[i];
    }
    if (!same) {
        invalid("Systembeschreibung mit anderen Flächen als bei der Erzeugung der "
                "Variablenmenge");
    }
    same = system.ghost_plates.size() == ghost_orders_.size();
    for (std::size_t g = 0; same && g < ghost_orders_.size(); ++g) {
        same = static_cast<int>(system.ghost_plates[g].coefficients.size()) == ghost_orders_[g];
    }
    if (!same) {
        invalid("Systembeschreibung mit anderen Geisterplatten als bei der Erzeugung der "
                "Variablenmenge");
    }
}

std::vector<double> VariableSet::extract(const SystemBatchHost& batch, int v) const {
    std::vector<double> x(variables_.size());
    extract(batch, v, x);
    return x;
}

void VariableSet::extract(const SystemBatchHost& batch, int v, std::span<double> x) const {
    check_batch(batch);
    check_size(x.size(), "x");
    for (std::size_t k = 0; k < variables_.size(); ++k) {
        const Variable& var = variables_[k];
        if (var.type == VariableType::Curvature) {
            x[k] = batch.curvature(v, var.batch_surface);
        } else if (var.type == VariableType::GhostCoefficient) {
            x[k] = batch.ghost_coeff(v, var.ghost_plate, var.ghost_order - 1);
        } else {
            const double t = batch.thickness(v, var.batch_surface);
            x[k] = var.split ? var.offset + t : t;
        }
    }
}

void VariableSet::apply(SystemBatchHost& batch, int v, std::span<const double> x) const {
    check_batch(batch);
    check_size(x.size(), "x");
    for (std::size_t k = 0; k < x.size(); ++k) {
        if (!std::isfinite(x[k])) {
            invalid("Wert von " + variables_[k].name + " ist nicht endlich");
        }
    }
    (void)batch.curvature(v, 0); // prüft v vor der ersten Änderung (std::out_of_range)
    for (std::size_t k = 0; k < variables_.size(); ++k) {
        const Variable& var = variables_[k];
        if (var.type == VariableType::Curvature) {
            batch.curvature(v, var.batch_surface) = x[k];
        } else if (var.type == VariableType::GhostCoefficient) {
            batch.ghost_coeff(v, var.ghost_plate, var.ghost_order - 1) = x[k];
        } else {
            // derselbe Ausdruck wie in make_surface_table (Ende des Spalts − distance)
            batch.thickness(v, var.batch_surface) = var.split ? x[k] - var.offset : x[k];
        }
    }
}

std::vector<double> VariableSet::extract(const SystemDescription& system) const {
    check_description(system);
    std::vector<double> x(variables_.size());
    for (std::size_t k = 0; k < variables_.size(); ++k) {
        const Variable& var = variables_[k];
        if (var.type == VariableType::GhostCoefficient) {
            x[k] = system.ghost_plates[static_cast<std::size_t>(var.ghost_plate)]
                       .coefficients[static_cast<std::size_t>(var.ghost_order - 1)];
            continue;
        }
        const Surface& s = system.surfaces[static_cast<std::size_t>(var.file_surface)];
        x[k] = var.type == VariableType::Curvature ? s.curvature : s.thickness;
    }
    return x;
}

void VariableSet::apply_to_description(SystemDescription& system, std::span<const double> x) const {
    check_description(system);
    check_size(x.size(), "x");
    for (std::size_t k = 0; k < x.size(); ++k) {
        if (!std::isfinite(x[k])) {
            invalid("Wert von " + variables_[k].name + " ist nicht endlich");
        }
    }
    for (std::size_t k = 0; k < variables_.size(); ++k) {
        const Variable& var = variables_[k];
        if (var.type == VariableType::GhostCoefficient) {
            system.ghost_plates[static_cast<std::size_t>(var.ghost_plate)]
                .coefficients[static_cast<std::size_t>(var.ghost_order - 1)] = x[k];
            continue;
        }
        Surface& s = system.surfaces[static_cast<std::size_t>(var.file_surface)];
        (var.type == VariableType::Curvature ? s.curvature : s.thickness) = x[k];
    }
}

bool VariableSet::feasible(std::span<const double> x) const {
    check_size(x.size(), "x");
    for (std::size_t k = 0; k < x.size(); ++k) {
        if (!(x[k] >= lower_[k] && x[k] <= upper_[k])) {
            return false;
        }
    }
    return true;
}

void VariableSet::clamp_to_bounds(std::span<double> x) const {
    check_size(x.size(), "x");
    for (std::size_t k = 0; k < x.size(); ++k) {
        x[k] = std::min(std::max(x[k], lower_[k]), upper_[k]);
    }
}

} // namespace mko
