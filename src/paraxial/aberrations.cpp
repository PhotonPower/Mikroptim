#include "mikroptim/paraxial/aberrations.hpp"

#include "mikroptim/glass/index_table.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace mko {
namespace {

struct Rays {
    std::vector<double> y, u_before, u_after, ybar, ubar_before, ubar_after, n;
};

// Paraxialer Rand- und Hauptstrahl über die Dateiflächen bei einer Wellenlänge.
// n[s] = Medium nach Fläche s, vor Fläche 0 Luft. Randstrahl y0 = a, u0 = 0; Hauptstrahl
// y0 = 0, u0 = tan θ (Blende = Fläche 0).
Rays trace_rays(const SystemDescription& system, const GlassCatalog& catalog, double lambda_nm,
                double a, double tan_theta) {
    const std::size_t S = system.surfaces.size();
    Rays r;
    r.y.resize(S);
    r.u_before.resize(S);
    r.u_after.resize(S);
    r.ybar.resize(S);
    r.ubar_before.resize(S);
    r.ubar_after.resize(S);
    r.n.resize(S);
    double y = a, u = 0.0, yb = 0.0, ub = tan_theta, n_before = 1.0;
    for (std::size_t s = 0; s < S; ++s) {
        const Surface& surf = system.surfaces[s];
        if (s > 0) {
            const double t = system.surfaces[s - 1].thickness;
            y += t * u;
            yb += t * ub;
        }
        const double n_after = material_index(surf.material, lambda_nm, catalog);
        const double c = surf.type == SurfaceType::Sphere ? surf.curvature : 0.0;
        const double phi = c * (n_after - n_before);
        r.y[s] = y;
        r.ybar[s] = yb;
        r.u_before[s] = u;
        r.ubar_before[s] = ub;
        u = (n_before * u - y * phi) / n_after;
        ub = (n_before * ub - yb * phi) / n_after;
        r.u_after[s] = u;
        r.ubar_after[s] = ub;
        r.n[s] = n_after;
        n_before = n_after;
    }
    return r;
}

double n_before_of(const Rays& r, std::size_t s) {
    return s == 0 ? 1.0 : r.n[s - 1];
}

double curvature_of(const Surface& surf) {
    return surf.type == SurfaceType::Sphere ? surf.curvature : 0.0;
}

// S_I einer Fläche (Welford): −A² y Δ(u/n).
double seidel_spherical(const Rays& r, std::size_t s, double c) {
    const double n = n_before_of(r, s);
    const double A = n * (r.u_before[s] + r.y[s] * c);
    const double d = r.u_after[s] / r.n[s] - r.u_before[s] / n;
    return -A * A * r.y[s] * d;
}

} // namespace

SeidelWaves AberrationReport::waves() const {
    const double lam = reference_nm * 1e-6; // mm
    SeidelWaves w;
    w.W040 = S[0] / 8.0 / lam;
    w.W131 = S[1] / 2.0 / lam;
    w.W222 = S[2] / 2.0 / lam;
    w.W220 = S[3] / 4.0 / lam;
    w.W311 = S[4] / 2.0 / lam;
    w.dW020 = C[0] / 2.0 / lam;
    w.dW111 = C[1] / lam;
    w.dW040 = spherochromatism / 8.0 / lam;
    return w;
}

SeidelWaves AberrationReport::waves(const SurfaceAberration& s) const {
    const double lam = reference_nm * 1e-6;
    SeidelWaves w;
    w.W040 = s.S[0] / 8.0 / lam;
    w.W131 = s.S[1] / 2.0 / lam;
    w.W222 = s.S[2] / 2.0 / lam;
    w.W220 = s.S[3] / 4.0 / lam;
    w.W311 = s.S[4] / 2.0 / lam;
    w.dW020 = s.C[0] / 2.0 / lam;
    w.dW111 = s.C[1] / lam;
    w.dW040 = s.spherochromatism / 8.0 / lam;
    return w;
}

AberrationReport compute_aberrations(const SystemDescription& system, const GlassCatalog& catalog,
                                     const AberrationOptions& options) {
    if (system.source.type != SourceType::Collimated) {
        throw std::invalid_argument(
            "Aberrationsbericht: nur für kollimierte Quelle (Objekt im Unendlichen)");
    }
    if (system.surfaces.empty() || system.surfaces[0].type != SurfaceType::Stop) {
        throw std::invalid_argument("Aberrationsbericht: Fläche 0 muss die Blende sein");
    }
    const std::vector<double>& wl = system.spectrum.wavelengths_nm;
    AberrationReport rep;
    rep.reference_nm = options.reference_nm.value_or(system.spectrum.reference_nm);
    rep.short_nm = options.short_nm.value_or(*std::min_element(wl.begin(), wl.end()));
    rep.long_nm = options.long_nm.value_or(*std::max_element(wl.begin(), wl.end()));
    double theta = 0.0;
    for (double f : system.source.field_angles_deg) {
        theta = std::max(theta, std::abs(f));
    }
    rep.field_angle_deg = options.field_angle_deg.value_or(theta);
    if (!(std::isfinite(rep.field_angle_deg) && std::abs(rep.field_angle_deg) < 90.0) ||
        !(rep.reference_nm > 0.0 && rep.short_nm > 0.0 && rep.long_nm > 0.0)) {
        throw std::invalid_argument("Aberrationsbericht: ungültiger Feldwinkel oder Wellenlänge");
    }
    rep.stop_semi_diameter = system.surfaces[0].semi_diameter;
    const double a = rep.stop_semi_diameter;
    const double tan_theta = std::tan(rep.field_angle_deg * std::numbers::pi / 180.0);
    for (const GhostPlate& g : system.ghost_plates) {
        if (std::any_of(g.coefficients.begin(), g.coefficients.end(),
                        [](double x) { return x != 0.0; })) {
            ++rep.ghost_plates_ignored;
        }
    }

    const Rays r = trace_rays(system, catalog, rep.reference_nm, a, tan_theta);
    const Rays rs = trace_rays(system, catalog, rep.short_nm, a, tan_theta);
    const Rays rl = trace_rays(system, catalog, rep.long_nm, a, tan_theta);
    // H = n (ū y − u ȳ) an Fläche 0 vor der Brechung (Luft)
    rep.lagrange = tan_theta * a;
    const double H = rep.lagrange;

    const std::size_t S = system.surfaces.size();
    rep.surfaces.resize(S);
    for (std::size_t s = 0; s < S; ++s) {
        SurfaceAberration& out = rep.surfaces[s];
        const double c = curvature_of(system.surfaces[s]);
        const double n = n_before_of(r, s);
        const double np = r.n[s];
        out.surface = static_cast<int>(s);
        out.curvature = c;
        out.n_before = n;
        out.n_after = np;
        out.y = r.y[s];
        out.u_before = r.u_before[s];
        out.u_after = r.u_after[s];
        out.ybar = r.ybar[s];
        out.ubar_before = r.ubar_before[s];
        out.ubar_after = r.ubar_after[s];
        out.A = n * (out.u_before + out.y * c);
        out.Abar = n * (out.ubar_before + out.ybar * c);
        const double du_n = out.u_after / np - out.u_before / n;
        const double d_inv_n = 1.0 / np - 1.0 / n;
        out.S[0] = -out.A * out.A * out.y * du_n;
        out.S[1] = -out.A * out.Abar * out.y * du_n;
        out.S[2] = -out.Abar * out.Abar * out.y * du_n;
        out.S[3] = -H * H * c * d_inv_n;
        if (out.A != 0.0) {
            out.S[4] = out.Abar / out.A * (out.S[2] + out.S[3]);
        } else if (out.Abar == 0.0 || (out.S[2] == 0.0 && out.S[3] == 0.0)) {
            out.S[4] = 0.0;
        } else {
            out.S[4] = std::numeric_limits<double>::quiet_NaN();
            out.distortion_defined = false;
            rep.distortion_defined = false;
        }
        // δn = n(λ_kurz) − n(λ_lang); Δ(δn/n) mit n bei λ_ref
        const double dn_after = rs.n[s] - rl.n[s];
        const double dn_before = s == 0 ? 0.0 : rs.n[s - 1] - rl.n[s - 1];
        const double d_dn_n = dn_after / np - dn_before / n;
        out.C[0] = out.A * out.y * d_dn_n;
        out.C[1] = out.Abar * out.y * d_dn_n;
        out.spherochromatism = seidel_spherical(rs, s, c) - seidel_spherical(rl, s, c);
        for (int k = 0; k < 5; ++k) {
            rep.S[k] += out.S[k];
        }
        rep.C[0] += out.C[0];
        rep.C[1] += out.C[1];
        rep.spherochromatism += out.spherochromatism;
    }
    return rep;
}

} // namespace mko
