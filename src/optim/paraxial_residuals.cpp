// Paraxiale Nebenbedingungen als Zusatzresiduen (docs/optimierung.md, Abschnitte 2.2 und 2.3).

#include "mikroptim/optim/paraxial_residuals.hpp"

#include "mikroptim/model/system.hpp"
#include "mikroptim/trace/ghost_plate.hpp"

#include "parallel_variants.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace mko {

namespace {

std::size_t to_size(int n) {
    return static_cast<std::size_t>(n);
}

void check_variant(const SystemBatchHost& batch, int v) {
    if (v < 0 || v >= batch.variants()) {
        throw std::out_of_range("Paraxial-Nebenbedingung: Variante " + std::to_string(v) +
                                " außerhalb von [0, " + std::to_string(batch.variants()) + ")");
    }
}

[[noreturn]] void fail_constraint(int i, const ParaxialConstraint& c, const std::string& what) {
    throw std::invalid_argument("Paraxial-Nebenbedingung " + std::to_string(i) + " (" +
                                std::string(to_string(c.type)) + "): " + what);
}

// Abweichung nach Modus: Gleichung q, Untergrenze min(q, 0), Obergrenze max(q, 0).
double apply_mode(ConstraintMode mode, double q) {
    switch (mode) {
    case ConstraintMode::Equal:
        return q;
    case ConstraintMode::Min:
        return std::min(q, 0.0);
    case ConstraintMode::Max:
        return std::max(q, 0.0);
    }
    return q;
}

} // namespace

// ---- Paraxialtrace über Batch-Varianten ------------------------------------------------------

ParaxialData batch_paraxial(const SystemBatchHost& batch, int v) {
    check_variant(batch, v);
    const SystemBatchView<double> b = batch.view();
    const BatchShape& sh = b.shape;
    const int S = sh.surfaces;
    // Feste Felder auf dem Stack (S <= kMaxSurfaces); Layout wie eine Variante im Batch.
    std::array<double, kMaxSurfaces> c{}, t{}, n{}, power{}, petzval{};
    for (int s = 0; s < S; ++s) {
        c[to_size(s)] = b.curvature[sh.surface_at(v, s)];
        t[to_size(s)] = b.thickness[sh.surface_at(v, s)];
        n[to_size(s)] = b.paraxial_index[sh.surface_at(v, s)];
    }
    // Geisterplatten: dünne Elemente im Medium des Spalts (Brechzahl nach der Geisterfläche),
    // wie compute_paraxial(system, …) über die Tabellenflächen.
    for (int g = 0; g < sh.ghost_plates; ++g) {
        const int s = b.ghost_surface[g];
        const double n_medium = n[to_size(s)];
        const double n_glass = b.paraxial_ghost_index[sh.ghost_at(v, g)];
        const double* coeff = b.ghost_coeff + sh.ghost_coeff_at(v, g, 0);
        const double r = b.ghost_norm_radius[g];
        power[to_size(s)] = ghost_plate_power(coeff, sh.ghost_terms, r, n_glass, n_medium);
        petzval[to_size(s)] = ghost_plate_petzval(coeff, sh.ghost_terms, r, n_glass, n_medium);
    }
    const std::size_t count = to_size(S);
    return compute_paraxial(std::span<const double>(c.data(), count),
                            std::span<const double>(t.data(), count),
                            std::span<const double>(n.data(), count), b.pupil_radius[v],
                            std::span<const double>(power.data(), count),
                            std::span<const double>(petzval.data(), count));
}

std::vector<ParaxialData> batch_paraxial_all(const SystemBatchHost& batch, int threads) {
    std::vector<ParaxialData> out(to_size(batch.variants()));
    detail::parallel_for_variants(0, batch.variants(), threads,
                                  [&](int v) { out[to_size(v)] = batch_paraxial(batch, v); });
    return out;
}

// ---- Nebenbedingungen --------------------------------------------------------------------

bool ParaxialConstraints::needs_trace() const {
    return std::any_of(items.begin(), items.end(), [](const ParaxialConstraint& c) {
        return c.type == ConstraintType::FocalLength || c.type == ConstraintType::ParaxialFocus ||
               c.type == ConstraintType::Petzval;
    });
}

ParaxialConstraints make_paraxial_constraints(const SystemDescription& system) {
    check_constraints(system);
    ParaxialConstraints out;
    if (!system.constraints) {
        return out;
    }
    const SurfaceMap map = make_surface_map(system);
    const std::vector<ConstraintSpec>& items = system.constraints->items;
    for (std::size_t i = 0; i < items.size(); ++i) {
        const ConstraintSpec& spec = items[i];
        if (!is_paraxial(spec.type)) {
            continue;
        }
        ParaxialConstraint c;
        c.type = spec.type;
        c.mode = spec.mode_or_default();
        c.weight = spec.weight_or_default();
        c.target = *constraint_bound(system, spec); // von check_constraints garantiert
        if (spec.type == ConstraintType::ParfocalLength) {
            c.reference_surface = map.table_surface(spec.reference_surface_or_default());
            c.offset = spec.offset_or_default();
        }
        c.item = static_cast<int>(i);
        out.items.push_back(c);
    }
    return out;
}

void check_paraxial_constraints(const ParaxialConstraints& constraints,
                                const BatchTopology& topology) {
    for (int i = 0; i < constraints.size(); ++i) {
        const ParaxialConstraint& c = constraints.items[to_size(i)];
        if (!is_paraxial(c.type)) {
            fail_constraint(i, c, "kein paraxialer Typ");
        }
        if (c.mode != ConstraintMode::Equal && c.mode != ConstraintMode::Min &&
            c.mode != ConstraintMode::Max) {
            fail_constraint(i, c, "unbekannter Modus");
        }
        if (!(std::isfinite(c.weight) && c.weight >= 0.0)) {
            fail_constraint(i, c, "Gewicht muss endlich und mindestens 0 sein");
        }
        if (!std::isfinite(c.target)) {
            fail_constraint(i, c, "Sollwert muss endlich sein");
        }
        switch (c.type) {
        case ConstraintType::FocalLength:
            if (c.target == 0.0) {
                fail_constraint(i, c, "Sollbrennweite darf nicht 0 sein");
            }
            [[fallthrough]];
        case ConstraintType::ParaxialFocus:
            if (c.mode != ConstraintMode::Equal) {
                fail_constraint(i, c, "nur als Gleichung definiert");
            }
            if (c.type == ConstraintType::ParaxialFocus &&
                topology.source != SourceKind::Collimated) {
                fail_constraint(i, c, "verlangt eine kollimierte Quelle");
            }
            break;
        case ConstraintType::ParfocalLength:
            if (c.reference_surface < 0 || c.reference_surface >= topology.surfaces) {
                fail_constraint(i, c,
                                "Referenzfläche " + std::to_string(c.reference_surface) +
                                    " außerhalb von [0, " + std::to_string(topology.surfaces) +
                                    ")");
            }
            if (topology.type[to_size(c.reference_surface)] == SurfaceKind::GhostPlate) {
                fail_constraint(i, c, "Referenzfläche ist eine Geisterplatte");
            }
            if (!std::isfinite(c.offset)) {
                fail_constraint(i, c, "Versatz muss endlich sein");
            }
            break;
        case ConstraintType::WorkingDistance:
            break;
        case ConstraintType::Petzval:
            if (c.mode != ConstraintMode::Max) {
                fail_constraint(i, c, "nur als Obergrenze definiert (Modus max)");
            }
            if (c.target < 0.0) {
                fail_constraint(i, c, "Grenze κ_max muss mindestens 0 sein");
            }
            break;
        }
    }
}

double paraxial_residual(const ParaxialConstraint& c, const SystemBatchHost& batch, int v,
                         const ParaxialData& p) {
    check_variant(batch, v);
    const int S = batch.surfaces();
    double q = 0.0;
    switch (c.type) {
    case ConstraintType::FocalLength:
        // f_s (1 − f_s φ) = f_s² (φ_s − φ): linear in φ, stetig durch φ = 0, nahe dem Soll
        // gleich f − f_s bis auf Terme zweiter Ordnung.
        q = c.target * (1.0 - c.target * p.power);
        break;
    case ConstraintType::WorkingDistance:
        q = batch.thickness(v, S - 1) - c.target;
        break;
    case ConstraintType::ParfocalLength: {
        double length = c.offset;
        for (int s = c.reference_surface; s < S; ++s) {
            length += batch.thickness(v, s);
        }
        q = length - c.target;
        break;
    }
    case ConstraintType::ParaxialFocus: {
        // Randstrahl mit Einfallshöhe a: Höhe a A und Winkel a C / n'_L nach der letzten
        // Fläche, in der Bildebene a (A + t C / n'_L).
        const double A = p.abcd[0];
        const double C = p.abcd[2];
        q = batch.pupil_radius(v) *
            (A + batch.thickness(v, S - 1) * C / batch.paraxial_index(v, S - 1));
        break;
    }
    case ConstraintType::Petzval:
        // |κ_P| − κ_max mit κ_P = −n'_L P einschließlich der Geisterplatten (batch_paraxial);
        // Modus Max: aktiv nur oberhalb der Grenze.
        q = std::abs(p.petzval_curvature) - c.target;
        break;
    }
    return std::sqrt(c.weight) * apply_mode(c.mode, q);
}

void paraxial_residuals(const ParaxialConstraints& constraints, const SystemBatchHost& batch, int v,
                        std::span<double> out) {
    if (out.size() != to_size(constraints.size())) {
        throw std::invalid_argument("Paraxial-Nebenbedingungen: Ausgabe hat " +
                                    std::to_string(out.size()) + " Einträge, erwartet " +
                                    std::to_string(constraints.size()));
    }
    check_variant(batch, v);
    const ParaxialData p = constraints.needs_trace() ? batch_paraxial(batch, v) : ParaxialData{};
    for (int i = 0; i < constraints.size(); ++i) {
        out[to_size(i)] = paraxial_residual(constraints.items[to_size(i)], batch, v, p);
    }
}

ExtraResiduals make_paraxial_residuals(ParaxialConstraints constraints) {
    ExtraResiduals extra;
    extra.count = constraints.size();
    extra.evaluate = [cons = std::move(constraints)](const SystemBatchHost& batch, int v,
                                                     std::span<double> out) {
        check_paraxial_constraints(cons, batch.topology());
        paraxial_residuals(cons, batch, v, out);
    };
    return extra;
}

void evaluate_extra_residuals(const ExtraResiduals& extra, const SystemBatchHost& batch,
                              std::span<double> out, int first, int count, int threads) {
    const int total = batch.variants();
    if (first < 0 || first > total || count > total - first) {
        throw std::out_of_range("Zusatzresiduen: Variantenbereich außerhalb des Batches");
    }
    if (count < 0) {
        count = total - first;
    }
    if (extra.count < 0) {
        throw std::invalid_argument("Zusatzresiduen: negative Anzahl");
    }
    const std::size_t per = to_size(extra.count);
    if (out.size() != per * to_size(count)) {
        throw std::invalid_argument("Zusatzresiduen: Ausgabe hat " + std::to_string(out.size()) +
                                    " Einträge, erwartet " + std::to_string(per * to_size(count)));
    }
    if (per == 0) {
        return;
    }
    if (!extra.evaluate) {
        throw std::invalid_argument("Zusatzresiduen: keine Auswertefunktion");
    }
    detail::parallel_for_variants(first, count, threads, [&](int v) {
        extra.evaluate(batch, v, out.subspan(to_size(v - first) * per, per));
    });
}

} // namespace mko
