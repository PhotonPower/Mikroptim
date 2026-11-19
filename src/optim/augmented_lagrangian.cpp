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

} // namespace mko
