// Erweiterung des Glas-Wörterbuchs der Geisterplatten (M6-T12), docs/optimierung.md,
// Abschnitt 5 „Erweitertes Wörterbuch“.

#include "mikroptim/synth/ghost_dictionary.hpp"

#include "mikroptim/glass/glass_preselection.hpp"
#include "mikroptim/glass/glass_properties.hpp"
#include "mikroptim/glass/index_table.hpp"
#include "mikroptim/optim/design_operators.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace mko {

namespace {

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument("Glas-Wörterbuch: " + message);
}

bool same_material(const MaterialSpec& a, const MaterialSpec& b) {
    if (a.kind != b.kind) {
        return false;
    }
    switch (a.kind) {
    case MaterialKind::Catalog:
        return a.name == b.name;
    case MaterialKind::Constant:
        return a.n == b.n;
    case MaterialKind::Buchdahl:
        return a.n0 == b.n0 && a.nu == b.nu;
    case MaterialKind::Air:
        break;
    }
    return true;
}

// n_d und ν_d eines Materials (NaN, wenn nicht auswertbar).
struct NdVd {
    double n_d = std::numeric_limits<double>::quiet_NaN();
    double v_d = std::numeric_limits<double>::quiet_NaN();
};

NdVd nd_vd(const MaterialSpec& m, const GlassCatalog& catalog) {
    NdVd r;
    try {
        const double nd = material_index(m, kLineD, catalog);
        const double nf = material_index(m, kLineF, catalog);
        const double nc = material_index(m, kLineC, catalog);
        r.n_d = nd;
        r.v_d = (nd - 1.0) / (nf - nc);
    } catch (const std::exception&) {
        // nicht auswertbar (Bereich, Modell): kein Vergleich
    }
    return r;
}

bool duplicate(const NdVd& a, double n_d, double v_d, const GhostDictionaryGlassOptions& o) {
    return std::isfinite(a.n_d) && std::isfinite(a.v_d) && std::abs(a.n_d - n_d) < o.duplicate_nd &&
           std::abs(a.v_d - v_d) < o.duplicate_vd;
}

GhostGlassClass classify(const GlassProperties& p, const GhostDictionaryGlassOptions& o) {
    if (std::abs(p.dp_gF) >= o.anomalous_dpgf) {
        return GhostGlassClass::Anomalous;
    }
    const bool flint = p.n_d < 1.60 ? p.v_d < 55.0 : p.v_d < 50.0;
    return flint ? GhostGlassClass::Flint : GhostGlassClass::Other;
}

std::string fmt(double v, int digits) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*f", digits, v);
    return buf;
}

std::string sci(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.3e", v);
    return buf;
}

} // namespace

std::string_view to_string(GhostGlassClass c) {
    switch (c) {
    case GhostGlassClass::Flint:
        return "flint";
    case GhostGlassClass::Anomalous:
        return "anomalous";
    case GhostGlassClass::Other:
        break;
    }
    return "other";
}

std::vector<MaterialSpec> ghost_dictionary_materials(const SystemDescription& system) {
    std::vector<MaterialSpec> out;
    for (const GhostPlate& p : system.ghost_plates) {
        if (std::none_of(out.begin(), out.end(),
                         [&](const MaterialSpec& m) { return same_material(m, p.glass); })) {
            out.push_back(p.glass);
        }
    }
    return out;
}

std::vector<GhostDictionaryGlass> select_ghost_dictionary_glasses(
    const SystemDescription& system, const GlassCatalog& catalog,
    const GhostDictionaryGlassOptions& o) {
    if (o.count < 0) {
        invalid("Zahl der Zusatzgläser < 0");
    }
    for (std::size_t i = 0; i < o.names.size(); ++i) {
        const std::string& n = o.names[i];
        if (!catalog.contains(n)) {
            invalid("Glas \"" + n + "\" nicht in den Katalogen");
        }
        if (std::find(o.names.begin(), o.names.begin() + static_cast<std::ptrdiff_t>(i), n) !=
            o.names.begin() + static_cast<std::ptrdiff_t>(i)) {
            invalid("Glas \"" + n + "\" doppelt genannt");
        }
    }
    const std::vector<MaterialSpec> dict = ghost_dictionary_materials(system);

    // Rang aller Katalog-Gläser: kleinster spektraler Rest eines Achromats mit einem anderen
    // Katalogglas (unabhängig vom Wörterbuch, damit die Auswahl bei jedem Aufruf dieselbe ist)
    const std::vector<GlassDispersion> cat = catalog_dispersions(catalog);
    std::vector<double> best(cat.size(), std::numeric_limits<double>::infinity());
    std::vector<double> best_sum(cat.size(), std::numeric_limits<double>::quiet_NaN());
    std::vector<std::string> best_partner(cat.size());
    if (cat.size() >= 2) {
        GlassPreselectionOptions po;
        po.order = 1;
        po.max_power_sum = o.max_power_sum;
        po.max_results = static_cast<int>(cat.size() * (cat.size() - 1) / 2);
        po.wavelengths_nm = o.wavelengths_nm;
        for (const PreselectedGlasses& r : preselect_glasses(cat, po)) {
            for (std::size_t m = 0; m < 2; ++m) {
                const auto c = static_cast<std::size_t>(r.glasses[m]);
                const auto other = static_cast<std::size_t>(r.glasses[1 - m]);
                if (r.spectral_residual < best[c]) {
                    best[c] = r.spectral_residual;
                    best_sum[c] = r.power_sum;
                    best_partner[c] = cat[other].name;
                }
            }
        }
    }

    // „im Wörterbuch“: Katalogname einer Platte oder Duplikat eines Modells (z. B. der
    // Buchdahl-Fit des Glases im Startsystem)
    std::vector<NdVd> model_props;
    for (const MaterialSpec& m : dict) {
        if (m.kind != MaterialKind::Catalog) {
            model_props.push_back(nd_vd(m, catalog));
        }
    }
    const auto present = [&](const std::string& name, double n_d, double v_d) {
        const bool by_name = std::any_of(dict.begin(), dict.end(), [&](const MaterialSpec& m) {
            return m.kind == MaterialKind::Catalog && m.name == name;
        });
        const bool by_model = std::any_of(model_props.begin(), model_props.end(),
                                          [&](const NdVd& d) { return duplicate(d, n_d, v_d, o); });
        return by_name || by_model;
    };

    std::vector<GhostDictionaryGlass> chosen;
    std::vector<NdVd> chosen_props;
    const auto make = [&](const std::string& name, bool automatic) {
        GhostDictionaryGlass g;
        g.name = name;
        g.automatic = automatic;
        const GlassProperties p = glass_properties(catalog.at(name));
        g.n_d = p.n_d;
        g.v_d = p.v_d;
        g.dp_gF = p.dp_gF;
        g.glass_class = classify(p, o);
        g.present = present(name, p.n_d, p.v_d);
        g.spectral_residual = std::numeric_limits<double>::quiet_NaN();
        g.power_sum = std::numeric_limits<double>::quiet_NaN();
        for (std::size_t c = 0; c < cat.size(); ++c) {
            if (cat[c].name == name && std::isfinite(best[c])) {
                g.spectral_residual = best[c];
                g.power_sum = best_sum[c];
                g.partner = best_partner[c];
            }
        }
        chosen_props.push_back(NdVd{p.n_d, p.v_d});
        return g;
    };
    for (const std::string& n : o.names) {
        chosen.push_back(make(n, false));
    }

    // Rangfolgen der beiden Klassen
    std::vector<std::size_t> flints;
    std::vector<std::size_t> anomalous;
    std::vector<GlassProperties> props(cat.size());
    for (std::size_t c = 0; c < cat.size(); ++c) {
        if (!std::isfinite(best[c])) {
            continue;
        }
        try {
            props[c] = glass_properties(catalog.at(cat[c].name));
        } catch (const std::exception&) {
            continue; // Bereich deckt g … C nicht ab
        }
        const GhostGlassClass k = classify(props[c], o);
        if (k == GhostGlassClass::Flint) {
            flints.push_back(c);
        } else if (k == GhostGlassClass::Anomalous) {
            anomalous.push_back(c);
        }
    }
    const auto by_rank = [&](std::size_t a, std::size_t b) {
        return best[a] != best[b] ? best[a] < best[b] : a < b;
    };
    std::stable_sort(flints.begin(), flints.end(), by_rank);
    std::stable_sort(anomalous.begin(), anomalous.end(), by_rank);
    std::size_t fi = 0;
    std::size_t ai = 0;
    const auto take = [&](const std::vector<std::size_t>& from, std::size_t& i) {
        while (i < from.size()) {
            const std::size_t c = from[i++];
            const bool dup =
                std::any_of(chosen_props.begin(), chosen_props.end(), [&](const NdVd& d) {
                    return duplicate(d, props[c].n_d, props[c].v_d, o);
                });
            if (!dup) {
                chosen.push_back(make(cat[c].name, true));
                return true;
            }
        }
        return false;
    };
    bool flint_turn = true;
    for (int k = 0; k < o.count; ++k) {
        bool ok = flint_turn ? take(flints, fi) : take(anomalous, ai);
        if (!ok) {
            ok = flint_turn ? take(anomalous, ai) : take(flints, fi);
        }
        if (!ok) {
            break; // beide Klassen erschöpft
        }
        flint_turn = !flint_turn;
    }
    return chosen;
}

GhostDictionaryExtension extend_ghost_dictionaries(const SystemDescription& system,
                                                   const GlassCatalog& catalog,
                                                   const GhostDictionaryGlassOptions& options) {
    GhostDictionaryExtension r;
    r.system = system;
    r.glasses = select_ghost_dictionary_glasses(system, catalog, options);
    if (system.ghost_plates.empty() || r.glasses.empty()) {
        return r;
    }
    // Lagen in Reihenfolge der Platten (after_surface, distance)
    struct Position {
        int after_surface;
        double distance;
    };
    std::vector<Position> positions;
    for (const GhostPlate& p : system.ghost_plates) {
        if (std::none_of(positions.begin(), positions.end(), [&](const Position& q) {
                return q.after_surface == p.after_surface && q.distance == p.distance;
            })) {
            positions.push_back({p.after_surface, p.distance});
        }
    }
    for (const Position& q : positions) {
        const GhostPlate* first = nullptr;
        std::vector<MaterialSpec> glasses;
        for (const GhostPlate& p : r.system.ghost_plates) {
            if (p.after_surface == q.after_surface && p.distance == q.distance && first == nullptr) {
                first = &p;
            }
        }
        for (const GhostDictionaryGlass& g : r.glasses) {
            const bool present = std::any_of(
                r.system.ghost_plates.begin(), r.system.ghost_plates.end(), [&](const GhostPlate& p) {
                    if (p.after_surface != q.after_surface || p.distance != q.distance) {
                        return false;
                    }
                    if (p.glass.kind == MaterialKind::Catalog) {
                        return p.glass.name == g.name;
                    }
                    return duplicate(nd_vd(p.glass, catalog), g.n_d, g.v_d, options);
                });
            if (!present) {
                MaterialSpec m;
                m.kind = MaterialKind::Catalog;
                m.name = g.name;
                glasses.push_back(std::move(m));
            }
        }
        if (glasses.empty() || first == nullptr) {
            continue;
        }
        GhostDictionaryPlacement pl;
        pl.after_surface = q.after_surface;
        pl.distance = q.distance;
        pl.order = static_cast<int>(first->coefficients.size());
        pl.norm_radius = first->norm_radius;
        pl.glasses = std::move(glasses);
        const int n = static_cast<int>(pl.glasses.size());
        GhostDictionaryResult gr = place_ghost_dictionary(r.system, pl);
        r.system = std::move(gr.system);
        r.added_plates += n;
        ++r.positions;
    }
    return r;
}

std::string ghost_dictionary_text(const std::vector<GhostDictionaryGlass>& glasses) {
    std::string t;
    for (const GhostDictionaryGlass& g : glasses) {
        const char* cls = g.glass_class == GhostGlassClass::Flint       ? "Flint"
                          : g.glass_class == GhostGlassClass::Anomalous ? "anomal"
                                                                        : "Kron";
        t += (t.empty() ? "" : ", ") + g.name + " (" + cls + ", n_d " + fmt(g.n_d, 4) + ", v_d " +
             fmt(g.v_d, 2) + ", dP_gF " + fmt(g.dp_gF, 4) +
             (g.partner.empty() ? std::string()
                                : ", Partner " + g.partner + ", Rest " + sci(g.spectral_residual)) +
             (g.automatic ? "" : ", genannt") +
             (g.present ? ", schon im Wörterbuch" : "") + ")";
    }
    return t.empty() ? "keine" : t;
}

} // namespace mko
