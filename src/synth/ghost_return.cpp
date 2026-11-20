// Rückkehr vom realen System zu Geisterplatten (M6-T10), docs/optimierung.md, Abschnitt 7,
// „Rückkehr zu Geisterplatten“.

#include "mikroptim/synth/ghost_return.hpp"

#include "mikroptim/batch/batch_extrema.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/model/constraints.hpp"
#include "mikroptim/model/variables.hpp"
#include "mikroptim/optim/design_operators.hpp"
#include "mikroptim/optim/extra_residuals.hpp"
#include "mikroptim/optim/ghost_diagnosis.hpp"
#include "mikroptim/optim/ghost_penalty.hpp"
#include "mikroptim/optim/glass_snap.hpp"
#include "mikroptim/optim/mechanical_constraints.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "mikroptim/paraxial/aberrations.hpp"
#include "mikroptim/trace/surface_table.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

namespace mko {

namespace {

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument("return_to_ghost_plates: " + message);
}

std::size_t sz(int i) {
    return static_cast<std::size_t>(i);
}

std::string num(double v) {
    std::ostringstream os;
    os.precision(6);
    os << v;
    return os.str();
}

// F wie im LM ohne Glas-Hülle (wie `mikroptim insert`).
double cpu_merit(const LoadedSystem& l, const ResidualSettings& q) {
    const SystemBatchHost b = make_system_batch(l, 1);
    MechanicalResiduals mechanical =
        make_mechanical_residuals(make_mechanical_constraints(l.system));
    const ExtraResiduals extra =
        concat_extra_residuals({make_paraxial_residuals(make_paraxial_constraints(l.system)),
                                make_ghost_penalty_residuals(make_ghost_penalty(l.system)),
                                std::move(mechanical.geometric)});
    return make_cpu_merit(q, extra, 0, std::move(mechanical.traced))(b).at(0);
}

// Dickenvariablen des Spalts nach `gap` entfernen ("all" wird zur Liste der übrigen Spalte).
// Liefert, ob eine entfernt wurde.
bool remove_thickness_variable(SystemDescription& d, int gap) {
    if (!d.variables) {
        return false;
    }
    const int n = static_cast<int>(d.surfaces.size());
    bool removed = false;
    std::vector<VariableSpec> out;
    for (VariableSpec v : d.variables->items) {
        if (v.type != VariableType::Thickness) {
            out.push_back(std::move(v));
            continue;
        }
        if (v.selection == SurfaceSelection::All) {
            v.selection = SurfaceSelection::List;
            v.surfaces.clear();
            for (int s = 0; s < n - 1; ++s) {
                v.surfaces.push_back(s);
            }
        }
        const auto it = std::find(v.surfaces.begin(), v.surfaces.end(), gap);
        if (it == v.surfaces.end()) {
            out.push_back(std::move(v));
            continue;
        }
        v.surfaces.erase(it);
        removed = true;
        if (v.surfaces.empty()) {
            continue;
        }
        if (v.selection == SurfaceSelection::One) {
            v.selection = SurfaceSelection::List;
        }
        if (v.surfaces.size() == 1) {
            v.selection = SurfaceSelection::One;
        }
        out.push_back(std::move(v));
    }
    d.variables->items = std::move(out);
    return removed;
}

struct GapCandidate {
    int gap = 0;
    double height = 0.0; // |y| (Mittel)
    double ratio = 0.0;  // |ȳ/y| (Mittel)
};

} // namespace

std::string_view to_string(GhostReturnLocation location) {
    switch (location) {
    case GhostReturnLocation::Stop:
        return "stop";
    case GhostReturnLocation::Aperture:
        return "aperture";
    case GhostReturnLocation::Field:
        return "field";
    case GhostReturnLocation::Front:
        return "front";
    }
    return "?";
}

std::vector<GhostReturnLocation> parse_ghost_return_locations(std::string_view text) {
    std::vector<GhostReturnLocation> out;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string_view name =
            text.substr(start, comma == std::string_view::npos ? text.npos : comma - start);
        GhostReturnLocation l = GhostReturnLocation::Stop;
        if (name == "stop") {
            l = GhostReturnLocation::Stop;
        } else if (name == "aperture") {
            l = GhostReturnLocation::Aperture;
        } else if (name == "field") {
            l = GhostReturnLocation::Field;
        } else if (name == "front") {
            l = GhostReturnLocation::Front;
        } else {
            throw std::invalid_argument("unbekannte Lage \"" + std::string(name) +
                                        "\" (stop, aperture, field, front)");
        }
        if (std::find(out.begin(), out.end(), l) != out.end()) {
            throw std::invalid_argument("Lage \"" + std::string(name) + "\" doppelt");
        }
        out.push_back(l);
        if (comma == std::string_view::npos) {
            break;
        }
        start = comma + 1;
    }
    return out;
}

std::string ghost_return_json_text(const GhostReturnResult& r) {
    using Json = nlohmann::ordered_json;
    const auto number = [](double v) { return std::isfinite(v) ? Json(v) : Json(nullptr); };
    Json positions = Json::array();
    for (const GhostReturnPosition& p : r.positions) {
        positions.push_back(Json{{"location", std::string(to_string(p.location))},
                                 {"after_surface", p.after_surface},
                                 {"distance", number(p.distance)},
                                 {"norm_radius", number(p.norm_radius)},
                                 {"marginal_height", number(p.marginal_height)},
                                 {"ratio", number(p.ratio)},
                                 {"plates", p.plates},
                                 {"gap_fixed", p.gap_fixed}});
    }
    const double rel = r.merit_before != 0.0
                           ? std::abs(r.merit_after - r.merit_before) / std::abs(r.merit_before)
                           : std::abs(r.merit_after - r.merit_before);
    const Json j{{"positions", positions},
                 {"notes", r.notes},
                 {"merit_before", number(r.merit_before)},
                 {"merit_after", number(r.merit_after)},
                 {"relative_change", number(rel)}};
    return j.dump();
}

GhostReturnResult return_to_ghost_plates(const LoadedSystem& real,
                                         const GhostReturnOptions& options) {
    const SystemDescription& d0 = real.system;
    if (!d0.ghost_plates.empty()) {
        invalid("das System hat schon Geisterplatten");
    }
    if (options.glasses.empty()) {
        invalid("mindestens ein Glas nötig");
    }
    for (const MaterialSpec& g : options.glasses) {
        if (g.kind == MaterialKind::Air) {
            invalid("Geisterplatten brauchen ein Glas");
        }
    }
    if (options.locations.empty()) {
        invalid("keine Lage angegeben");
    }
    if (options.order < 1 || options.order > kMaxGhostTerms) {
        invalid("Ordnung " + std::to_string(options.order) + " außerhalb von [1, " +
                std::to_string(kMaxGhostTerms) + "]");
    }
    const auto positive = [](double v) { return std::isfinite(v) && v > 0.0; };
    if (!positive(options.penalty_weight) || !positive(options.order_weight) ||
        !positive(options.order_factor) || !positive(options.smoothing)) {
        invalid("Gewichte und Glättung der Strafe müssen endlich und > 0 sein");
    }
    if (!(std::isfinite(options.min_gap) && options.min_gap >= 0.0)) {
        invalid("min_gap muss endlich und >= 0 sein");
    }
    if (!positive(options.norm_margin) ||
        !(std::isfinite(options.norm_field_scale) && options.norm_field_scale >= 0.0)) {
        invalid("norm_margin muss > 0, norm_field_scale >= 0 sein");
    }
    if (options.rings < 1 || options.spokes < 1) {
        invalid("Quadratur mit rings, spokes >= 1");
    }
    const int S = static_cast<int>(d0.surfaces.size());
    const auto air_gap = [&](int s) {
        return d0.surfaces[sz(s)].material.kind == MaterialKind::Air &&
               d0.surfaces[sz(s)].thickness > 0.0;
    };

    // Innere Luftspalte (nach Fläche 1 … S − 2) mit paraxialen Strahlhöhen
    std::vector<GapCandidate> inner;
    std::string aberration_error;
    if (d0.source.type == SourceType::Collimated) {
        AberrationOptions ao;
        ao.real_rays = false;
        ao.focal_shift = false;
        try {
            const AberrationReport rep = compute_aberrations(d0, real.catalog, ao);
            const auto ratio = [&](int s) {
                const SurfaceAberration& a = rep.surfaces.at(sz(s));
                return std::abs(a.y) > 0.0 ? std::abs(a.ybar) / std::abs(a.y)
                                           : std::numeric_limits<double>::infinity();
            };
            for (int s = 1; s + 1 < S; ++s) {
                if (!air_gap(s) || !(d0.surfaces[sz(s)].thickness >= options.min_gap)) {
                    continue;
                }
                GapCandidate c;
                c.gap = s;
                c.height = 0.5 * (std::abs(rep.surfaces.at(sz(s)).y) +
                                  std::abs(rep.surfaces.at(sz(s + 1)).y));
                c.ratio = 0.5 * (ratio(s) + ratio(s + 1));
                inner.push_back(c);
            }
        } catch (const std::invalid_argument& e) {
            aberration_error = e.what();
        }
    } else {
        aberration_error = "Punktquelle (kein Aberrationsbericht)";
    }

    GhostReturnResult r;
    std::vector<GhostReturnPosition> chosen;
    for (const GhostReturnLocation loc : options.locations) {
        std::optional<GapCandidate> pick;
        if (loc == GhostReturnLocation::Stop) {
            if (S >= 2 && air_gap(0)) {
                GapCandidate c;
                c.gap = 0; // Blendenspalt: |ȳ/y| = 0 an der Blende
                pick = c;
            }
        } else {
            for (const GapCandidate& c : inner) {
                bool better = !pick;
                if (pick) {
                    switch (loc) {
                    case GhostReturnLocation::Aperture:
                        better = c.height > pick->height;
                        break;
                    case GhostReturnLocation::Field:
                        better = std::isfinite(c.ratio) &&
                                 (!std::isfinite(pick->ratio) || c.ratio > pick->ratio);
                        break;
                    case GhostReturnLocation::Front:
                        better = c.gap > pick->gap;
                        break;
                    case GhostReturnLocation::Stop:
                        break;
                    }
                } else if (loc == GhostReturnLocation::Field) {
                    better = std::isfinite(c.ratio);
                }
                if (better) {
                    pick = c;
                }
            }
        }
        const std::string name(to_string(loc));
        if (!pick) {
            r.notes.push_back(name + ": kein passender Luftspalt" +
                              (aberration_error.empty() || loc == GhostReturnLocation::Stop
                                   ? std::string()
                                   : " (" + aberration_error + ")"));
            continue;
        }
        const auto same = [&](const GhostReturnPosition& p) {
            return p.after_surface == pick->gap;
        };
        if (std::any_of(chosen.begin(), chosen.end(), same)) {
            r.notes.push_back(name + ": Spalt nach Fläche " + std::to_string(pick->gap) +
                              " schon belegt, übersprungen");
            continue;
        }
        GhostReturnPosition p;
        p.location = loc;
        p.after_surface = pick->gap;
        p.distance = 0.5 * d0.surfaces[sz(pick->gap)].thickness;
        p.marginal_height = pick->height;
        p.ratio = pick->ratio;
        chosen.push_back(p);
    }
    if (chosen.empty()) {
        invalid("keine Lage passt");
    }

    // Strafe vor dem Einsetzen: ein Eintrag mit "all" deckt danach alle Platten ab
    SystemDescription d = d0;
    if (!d.constraints) {
        d.constraints = Constraints{};
    }
    const bool has_penalty =
        std::any_of(d.constraints->items.begin(), d.constraints->items.end(),
                    [](const ConstraintSpec& c) { return c.type == ConstraintType::GhostPenalty; });
    if (!has_penalty) {
        ConstraintSpec c;
        c.type = ConstraintType::GhostPenalty;
        c.weight = options.penalty_weight;
        c.order_weight = options.order_weight;
        c.order_factor = options.order_factor;
        c.smoothing = options.smoothing;
        c.comment = "Rückkehr zu Geisterplatten (M6-T10)";
        d.constraints->items.push_back(std::move(c));
    }
    for (GhostReturnPosition& p : chosen) {
        GhostDictionaryPlacement g;
        g.after_surface = p.after_surface;
        g.distance = p.distance;
        g.glasses = options.glasses;
        g.order = options.order;
        g.norm_radius = 1.0; // vorläufig; Koeffizienten 0, unten aus den Strahlhöhen
        GhostDictionaryResult gr = place_ghost_dictionary(d, g);
        for (GhostReturnPosition& q : chosen) {
            for (int& k : q.plates) {
                k = gr.plate_new_of_old.at(sz(k));
            }
        }
        p.plates = gr.new_plates;
        d = std::move(gr.system);
    }
    if (options.fix_plate_gaps) {
        for (GhostReturnPosition& p : chosen) {
            p.gap_fixed = remove_thickness_variable(d, p.after_surface);
        }
    }

    // Normierungsradien aus den realen Strahlhöhen an den Platten
    ResidualSettings q;
    q.rings = options.rings;
    q.spokes = options.spokes;
    {
        SystemDescription h = d;
        if (h.source.type == SourceType::Collimated) {
            for (double& a : h.source.field_angles_deg) {
                a *= options.norm_field_scale;
            }
        }
        const LoadedSystem lh = rebuild_loaded_system(real, std::move(h));
        const SystemBatchHost b = make_system_batch(lh, 1);
        ResidualSettings hard = q;
        hard.failure = FailureMode::Hard;
        const ResidualLayout layout(b.topology(), hard);
        const SurfaceExtremaResult ex = surface_extrema_cpu(b, 0, layout);
        const SurfaceMap map = make_surface_map(lh.system);
        for (GhostReturnPosition& p : chosen) {
            double hmax = 0.0;
            for (const int g : p.plates) {
                hmax = std::max(hmax, ex.max_height(0, map.table_of_ghost.at(sz(g))));
            }
            double rg = options.norm_margin * hmax;
            if (!(rg > 0.0) || !std::isfinite(rg)) {
                rg = std::max(d.surfaces[sz(p.after_surface)].semi_diameter,
                              d.surfaces[sz(p.after_surface + 1)].semi_diameter);
                r.notes.push_back(std::string(to_string(p.location)) +
                                  ": kein gültiger Strahl an der Platte, r_g = Halbdurchmesser " +
                                  num(rg) + " mm");
            }
            p.norm_radius = rg;
            for (const int g : p.plates) {
                d.ghost_plates.at(sz(g)).norm_radius = rg;
            }
        }
    }
    validate_system(d);

    r.merit_before = cpu_merit(real, q);
    const LoadedSystem after = rebuild_loaded_system(real, d);
    r.merit_after = cpu_merit(after, q);
    r.system = std::move(d);
    r.positions = std::move(chosen);
    return r;
}

} // namespace mko
