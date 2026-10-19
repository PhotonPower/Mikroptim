// Geglättete L1-Strafe der Geisterkoeffizienten (docs/optimierung.md, Abschnitte 2 und 3).

#include "mikroptim/optim/ghost_penalty.hpp"

#include "mikroptim/model/constraints.hpp"
#include "mikroptim/model/system.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace mko {

namespace {

std::size_t to_size(int i) {
    return static_cast<std::size_t>(i);
}

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument("Geisterplatten-Strafe: " + message);
}

} // namespace

double ghost_order_weight(int order, double order_weight, double order_factor) {
    double w = order_weight;
    for (int i = 0; i < order; ++i) {
        w *= order_factor;
    }
    return w;
}

double ghost_penalty_residual(double a, double weight, double smoothing) {
    const double s = std::hypot(a, smoothing);
    return std::sqrt(weight) * a / std::sqrt(smoothing + s);
}

double ghost_penalty_square(double a, double weight, double smoothing) {
    // weight · (s − ε) ohne Auslöschung: s − ε = a² / (ε + s)
    const double s = std::hypot(a, smoothing);
    return weight * (a * a / (smoothing + s));
}

double ghost_penalty_derivative(double a, double weight, double smoothing) {
    const double s = std::hypot(a, smoothing);
    const double d = smoothing + s;
    return std::sqrt(weight) * (smoothing + 0.5 * s + 0.5 * smoothing * smoothing / s) /
           (d * std::sqrt(d));
}

GhostPenalty make_ghost_penalty(const SystemDescription& system) {
    check_constraints(system);
    GhostPenalty out;
    if (!system.constraints) {
        return out;
    }
    const int ghosts = static_cast<int>(system.ghost_plates.size());
    // Eintrag je Platte (check_constraints garantiert höchstens einen)
    std::vector<int> item_of(to_size(ghosts), -1);
    const std::vector<ConstraintSpec>& items = system.constraints->items;
    for (std::size_t i = 0; i < items.size(); ++i) {
        const ConstraintSpec& c = items[i];
        if (c.type != ConstraintType::GhostPenalty) {
            continue;
        }
        if (c.plate_selection == SurfaceSelection::One ||
            c.plate_selection == SurfaceSelection::List) {
            for (const int g : c.plates) {
                item_of[to_size(g)] = static_cast<int>(i);
            }
        } else {
            for (int g = 0; g < ghosts; ++g) {
                item_of[to_size(g)] = static_cast<int>(i);
            }
        }
    }
    for (int g = 0; g < ghosts; ++g) {
        const int i = item_of[to_size(g)];
        if (i < 0) {
            continue;
        }
        const ConstraintSpec& c = items[to_size(i)];
        const double mu = c.weight_or_default();
        const int orders = static_cast<int>(system.ghost_plates[to_size(g)].coefficients.size());
        for (int k = 1; k <= orders; ++k) {
            GhostPenaltyTerm t;
            t.plate = g;
            t.order = k;
            t.weight = mu * ghost_order_weight(k, c.order_weight_or_default(),
                                               c.order_factor_or_default());
            t.smoothing = c.smoothing_or_default();
            t.item = i;
            out.terms.push_back(t);
        }
    }
    return out;
}

void check_ghost_penalty(const GhostPenalty& penalty, const BatchTopology& topology) {
    for (std::size_t i = 0; i < penalty.terms.size(); ++i) {
        const GhostPenaltyTerm& t = penalty.terms[i];
        const std::string where = "Term " + std::to_string(i) + " (Platte " +
                                  std::to_string(t.plate) + ", Ordnung " + std::to_string(t.order) +
                                  "): ";
        if (t.plate < 0 || t.plate >= topology.ghost_plates) {
            invalid(where + "Platte außerhalb von [0, " + std::to_string(topology.ghost_plates) +
                    ")");
        }
        if (t.order < 1 || t.order > topology.ghost_terms) {
            invalid(where + "Ordnung außerhalb von [1, " + std::to_string(topology.ghost_terms) +
                    "]");
        }
        if (!(std::isfinite(t.weight) && t.weight >= 0.0)) {
            invalid(where + "Gewicht muss endlich und mindestens 0 sein");
        }
        if (!(std::isfinite(t.smoothing) && t.smoothing > 0.0)) {
            invalid(where + "Glättung muss endlich und größer als 0 sein");
        }
    }
}

void ghost_penalty_residuals(const GhostPenalty& penalty, const SystemBatchHost& batch, int v,
                             std::span<double> out) {
    if (out.size() != penalty.terms.size()) {
        invalid("Ausgabe hat " + std::to_string(out.size()) + " Einträge, erwartet " +
                std::to_string(penalty.terms.size()));
    }
    for (std::size_t i = 0; i < penalty.terms.size(); ++i) {
        const GhostPenaltyTerm& t = penalty.terms[i];
        out[i] = ghost_penalty_residual(batch.ghost_coeff(v, t.plate, t.order - 1), t.weight,
                                        t.smoothing);
    }
}

double ghost_penalty_merit(const GhostPenalty& penalty, const SystemBatchHost& batch, int v,
                           int plate) {
    double sum = 0.0;
    for (const GhostPenaltyTerm& t : penalty.terms) {
        if (plate >= 0 && t.plate != plate) {
            continue;
        }
        const double r = ghost_penalty_residual(batch.ghost_coeff(v, t.plate, t.order - 1),
                                                t.weight, t.smoothing);
        sum += r * r;
    }
    return sum;
}

ExtraResiduals make_ghost_penalty_residuals(GhostPenalty penalty) {
    ExtraResiduals out;
    out.count = penalty.size();
    if (out.count == 0) {
        return out;
    }
    out.evaluate = [p = std::move(penalty)](const SystemBatchHost& batch, int v,
                                            std::span<double> values) {
        ghost_penalty_residuals(p, batch, v, values);
    };
    return out;
}

double ghost_activation_gradient(double rms_waves, double delta_n, double reference_nm, int order) {
    const double lambda_mm = reference_nm * 1e-6;
    return 2.0 * rms_waves * delta_n / (lambda_mm * std::sqrt(2.0 * order + 1.0));
}

} // namespace mko
