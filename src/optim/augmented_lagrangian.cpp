// Augmented Lagrangian für einseitige Bedingungen (M11-T2), docs/optimierung.md, Abschnitt 2.6.

#include "mikroptim/optim/augmented_lagrangian.hpp"

#include "mikroptim/batch/batch_extrema.hpp"
#include "mikroptim/model/system.hpp"
#include "mikroptim/optim/mechanical_constraints.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "mikroptim/optim/trace_extra_residuals.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>

namespace mko {

double update_shift(double theta, double violation, double weight, double max_shift) {
    if (!(weight > 0.0)) {
        return 0.0;
    }
    const double t = std::max(0.0, theta + violation);
    return std::min(t, max_shift / std::sqrt(weight));
}

bool has_shiftable_constraints(const SystemDescription& system) {
    if (!system.constraints) {
        return false;
    }
    return std::any_of(system.constraints->items.begin(), system.constraints->items.end(),
                       [](const ConstraintSpec& c) { return supports_shift(c); });
}

namespace {

void add_entry(ShiftUpdate& u, ShiftEntry e, double max_shift) {
    e.new_shift = update_shift(e.shift, e.violation, e.weight, max_shift);
    const double s = std::sqrt(e.weight);
    u.max_change = std::max(u.max_change, s * std::abs(e.new_shift - e.shift));
    u.max_violation = std::max(u.max_violation, s * std::max(e.violation, 0.0));
    u.entries.push_back(e);
}

} // namespace

ShiftUpdate shift_update(const SystemDescription& system, const SystemBatchHost& batch, int v,
                         const ResidualSettings& quadrature, double max_shift) {
    if (!(max_shift >= 0.0)) {
        throw std::invalid_argument("Augmented Lagrangian: Schranke der Verschiebung muss "
                                    "mindestens 0 sein");
    }
    ShiftUpdate u;
    const ParaxialConstraints paraxial = make_paraxial_constraints(system);
    check_paraxial_constraints(paraxial, batch.topology());
    const MechanicalConstraints mech = make_mechanical_constraints(system);
    check_mechanical_constraints(mech, batch.topology());

    const ParaxialData p = paraxial.needs_trace() ? batch_paraxial(batch, v) : ParaxialData{};
    for (const ParaxialConstraint& c : paraxial.items) {
        if (c.mode == ConstraintMode::Equal) {
            continue;
        }
        ShiftEntry e;
        e.item = c.item;
        e.type = c.type;
        e.weight = c.weight;
        e.violation = paraxial_violation(c, paraxial_deviation(c, batch, v, p));
        e.shift = c.shift;
        add_entry(u, e, max_shift);
    }

    std::optional<SurfaceExtremaResult> extrema;
    if (mech.needs_extrema()) {
        extrema = surface_extrema_cpu(batch, v, ResidualLayout(batch.topology(), quadrature),
                                      mech.smoothing);
    }
    const VariantExtrema ve = extrema ? variant_extrema(*extrema, 0) : VariantExtrema{};
    int item = -1;
    int index = 0;
    for (const MechanicalConstraint& c : mech.items) {
        index = c.item == item ? index + 1 : 0;
        item = c.item;
        ShiftEntry e;
        e.item = c.item;
        e.index = index;
        e.type = c.type;
        e.file_surface = c.file_surface;
        e.weight = c.weight;
        e.violation =
            mechanical_violation(c, mechanical_value(c, batch, v, extrema ? &ve : nullptr));
        e.shift = c.shift;
        add_entry(u, e, max_shift);
    }
    return u;
}

void apply_shift_update(SystemDescription& system, const ShiftUpdate& update) {
    if (update.entries.empty()) {
        return;
    }
    if (!system.constraints) {
        throw std::invalid_argument("Augmented Lagrangian: Beschreibung ohne Nebenbedingungen");
    }
    std::vector<ConstraintSpec>& items = system.constraints->items;
    // Längen pro Eintrag: größter Index + 1
    std::vector<std::size_t> length(items.size(), 0);
    for (const ShiftEntry& e : update.entries) {
        if (e.item < 0 || static_cast<std::size_t>(e.item) >= items.size() || e.index < 0 ||
            !supports_shift(items[static_cast<std::size_t>(e.item)])) {
            throw std::invalid_argument("Augmented Lagrangian: Eintrag " + std::to_string(e.item) +
                                        " passt nicht zur Beschreibung");
        }
        std::size_t& n = length[static_cast<std::size_t>(e.item)];
        n = std::max(n, static_cast<std::size_t>(e.index) + 1);
    }
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (length[i] > 0) {
            items[i].shift.assign(length[i], 0.0);
        }
    }
    for (const ShiftEntry& e : update.entries) {
        items[static_cast<std::size_t>(e.item)].shift[static_cast<std::size_t>(e.index)] =
            e.new_shift;
    }
}

void clear_shifts(SystemDescription& system) {
    if (!system.constraints) {
        return;
    }
    for (ConstraintSpec& c : system.constraints->items) {
        c.shift.clear();
    }
}

std::vector<ShiftValue> shift_values(const SystemDescription& system) {
    std::vector<ShiftValue> out;
    if (!system.constraints) {
        return out;
    }
    const std::vector<ConstraintSpec>& items = system.constraints->items;
    for (std::size_t i = 0; i < items.size(); ++i) {
        for (std::size_t k = 0; k < items[i].shift.size(); ++k) {
            out.push_back(ShiftValue{static_cast<int>(i), static_cast<int>(k), items[i].type,
                                     items[i].shift[k]});
        }
    }
    return out;
}

void set_shift_values(SystemDescription& system, const std::vector<ShiftValue>& values) {
    if (values.empty()) {
        return;
    }
    const auto bad = [](const ShiftValue& v, const std::string& why) {
        throw std::invalid_argument("Verschiebung für Eintrag " + std::to_string(v.item) +
                                    ", Residuum " + std::to_string(v.index) + ": " + why);
    };
    if (!system.constraints) {
        bad(values.front(), "die Beschreibung hat keine Nebenbedingungen");
    }
    const std::vector<ConstraintSpec>& items = system.constraints->items;
    ShiftUpdate u;
    for (const ShiftValue& v : values) {
        if (v.item < 0 || static_cast<std::size_t>(v.item) >= items.size()) {
            bad(v, "Eintrag fehlt");
        }
        const ConstraintSpec& spec = items[static_cast<std::size_t>(v.item)];
        if (!supports_shift(spec)) {
            bad(v, "der Eintrag trägt keine Verschiebung");
        }
        if (spec.type != v.type) {
            bad(v, "Typ " + std::string(to_string(v.type)) + " statt " +
                       std::string(to_string(spec.type)));
        }
        const std::size_t count =
            is_mechanical(spec.type) ? constraint_surfaces(system, spec).size() : 1;
        if (v.index < 0 || static_cast<std::size_t>(v.index) >= count) {
            bad(v, "Index außerhalb (" + std::to_string(count) + " Residuen)");
        }
        if (!(v.shift >= 0.0) || !std::isfinite(v.shift)) {
            bad(v, "θ muss endlich und mindestens 0 sein");
        }
        ShiftEntry e;
        e.item = v.item;
        e.index = v.index;
        e.type = v.type;
        e.new_shift = v.shift;
        u.entries.push_back(e);
    }
    apply_shift_update(system, u);
}

} // namespace mko
