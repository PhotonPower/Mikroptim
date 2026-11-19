#include "mikroptim/glass/glass_properties.hpp"

#include <utility>

namespace mko {

GlassProperties glass_properties(const Glass& glass) {
    if (!glass.in_range(kLineG) || !glass.in_range(kLineC)) {
        throw GlassError("Glas " + glass.name + ": Bereich deckt die Linien g bis C nicht ab");
    }
    GlassProperties p;
    p.name = glass.name;
    p.n_d = glass.index(kLineD);
    p.n_F = glass.index(kLineF);
    p.n_C = glass.index(kLineC);
    p.n_g = glass.index(kLineG);
    const double dfc = p.n_F - p.n_C;
    p.v_d = (p.n_d - 1.0) / dfc;
    p.p_gF = (p.n_g - p.n_F) / dfc;
    p.dp_gF = p.p_gF - (kPgfNormalA + kPgfNormalB * p.v_d);
    p.range_nm = glass.range_nm;
    return p;
}

bool GlassPropertyFilter::matches(const GlassProperties& p) const {
    return (!n_d || n_d->contains(p.n_d)) && (!v_d || v_d->contains(p.v_d)) &&
           (!p_gF || p_gF->contains(p.p_gF)) && (!dp_gF || dp_gF->contains(p.dp_gF));
}

std::vector<GlassProperties> list_glass_properties(const GlassCatalog& catalog,
                                                   const GlassPropertyFilter& filter,
                                                   std::vector<std::string>* skipped) {
    std::vector<GlassProperties> out;
    for (const Glass& g : catalog.glasses()) {
        if (!g.in_range(kLineG) || !g.in_range(kLineC)) {
            if (skipped != nullptr) {
                skipped->push_back(g.name);
            }
            continue;
        }
        GlassProperties p = glass_properties(g);
        if (filter.matches(p)) {
            out.push_back(std::move(p));
        }
    }
    return out;
}

} // namespace mko
