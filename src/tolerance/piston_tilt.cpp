// Abzug von Kolben und Kippung über die Pupillenquadratur (M8-T4, Kippungsabzug M11-T6b);
// docs/numerik.md, Abschnitt 16.

#include "mikroptim/tolerance/piston_tilt.hpp"

#include "mikroptim/trace/surface_ops.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace mko {

PistonTiltFit::PistonTiltFit(const PupilQuadrature<double>& q, std::span<const std::uint8_t> use) {
    const int n = sample_count(q);
    if (use.size() != static_cast<std::size_t>(n)) {
        throw std::invalid_argument("PistonTiltFit: Maske braucht " + std::to_string(n) +
                                    " Einträge");
    }
    const auto N = static_cast<std::size_t>(n);
    x_.resize(N);
    y_.resize(N);
    w_.resize(N);
    use_.assign(use.begin(), use.end());
    // Normalgleichungen der Anpassung c₀ + t_x x + t_y y (symmetrisch, obere Hälfte), Summen in
    // aufsteigender Reihenfolge k
    double a00 = 0.0, a01 = 0.0, a02 = 0.0, a11 = 0.0, a12 = 0.0, a22 = 0.0;
    double w_all = 0.0;
    for (int k = 0; k < n; ++k) {
        const PupilSample<double> s = sample(q, k);
        const auto ku = static_cast<std::size_t>(k);
        x_[ku] = s.rho * std::sin(s.phi);
        y_[ku] = s.rho * std::cos(s.phi);
        w_[ku] = s.weight;
        w_all += s.weight;
        if (use_[ku] == 0) {
            continue;
        }
        const double w = s.weight;
        const double x = x_[ku];
        const double y = y_[ku];
        a00 += w;
        a01 += w * x;
        a02 += w * y;
        a11 += w * (x * x);
        a12 += w * (x * y);
        a22 += w * (y * y);
    }
    w_all_ = w_all;
    a00_ = a00;
    if (!(a00 > 0.0)) {
        return;
    }
    // Cholesky A = L Lᵀ; bei (nahezu) singulärer Matrix nur der Kolben
    const double eps = 1e-12 * a00;
    l00_ = std::sqrt(a00);
    l10_ = a01 / l00_;
    l20_ = a02 / l00_;
    const double d11 = a11 - l10_ * l10_;
    if (d11 > eps) {
        l11_ = std::sqrt(d11);
        l21_ = (a12 - l20_ * l10_) / l11_;
        const double d22 = a22 - l20_ * l20_ - l21_ * l21_;
        if (d22 > eps) {
            l22_ = std::sqrt(d22);
            tilt_ = true;
        }
    }
}

std::array<double, 3> PistonTiltFit::solve(double b0, double b1, double b2) const {
    if (!tilt_) {
        return {b0 / a00_, 0.0, 0.0};
    }
    // L z = b, Lᵀ c = z
    const double z0 = b0 / l00_;
    const double z1 = (b1 - l10_ * z0) / l11_;
    const double z2 = (b2 - l20_ * z0 - l21_ * z1) / l22_;
    const double ty = z2 / l22_;
    const double tx = (z1 - l21_ * ty) / l11_;
    const double c0 = (z0 - l10_ * tx - l20_ * ty) / l00_;
    return {c0, tx, ty};
}

TiltProjection::TiltProjection(const ResidualLayout& layout, std::span<const std::uint8_t> status)
    : rays_(layout.rays_per_variant()) {
    if (status.size() != static_cast<std::size_t>(rays_)) {
        throw std::invalid_argument("TiltProjection: Status braucht " + std::to_string(rays_) +
                                    " Einträge");
    }
    std::vector<std::uint8_t> use;
    for (int f = 0; f < layout.fields(); ++f) {
        const PupilQuadrature<double>& q = layout.quadrature(f);
        const auto n = static_cast<std::size_t>(layout.samples(f));
        for (int l = 0; l < layout.wavelengths(); ++l) {
            const auto o = static_cast<std::size_t>(layout.offset(f, l));
            use.resize(n);
            for (std::size_t k = 0; k < n; ++k) {
                use[k] = in_piston(static_cast<TraceStatus>(status[o + k])) ? 1 : 0;
            }
            Tuple t;
            t.offset = layout.offset(f, l);
            t.fit = PistonTiltFit(q, use);
            if (!t.fit.valid() || !t.fit.tilt()) {
                continue; // nur Kolben: die Residuen sind bereits kolbenfrei
            }
            t.root.resize(n);
            for (std::size_t k = 0; k < n; ++k) {
                t.root[k] = std::sqrt(t.fit.weight(static_cast<int>(k)));
            }
            tuples_.push_back(std::move(t));
        }
    }
}

void TiltProjection::apply(std::span<double> rows) const {
    if (rows.size() < static_cast<std::size_t>(rays_)) {
        throw std::invalid_argument("TiltProjection: zu wenige Zeilen");
    }
    for (const Tuple& t : tuples_) {
        const int n = t.fit.samples();
        double* r = rows.data() + t.offset;
        double b0 = 0.0, b1 = 0.0, b2 = 0.0;
        for (int k = 0; k < n; ++k) {
            if (!t.fit.used(k)) {
                continue;
            }
            const double sr = t.root[static_cast<std::size_t>(k)] * r[k];
            b0 += sr;
            b1 += t.fit.x(k) * sr;
            b2 += t.fit.y(k) * sr;
        }
        const std::array<double, 3> c = t.fit.solve(b0, b1, b2);
        for (int k = 0; k < n; ++k) {
            if (!t.fit.used(k)) {
                continue;
            }
            r[k] -= t.root[static_cast<std::size_t>(k)] *
                    (c[0] + c[1] * t.fit.x(k) + c[2] * t.fit.y(k));
        }
    }
}

} // namespace mko
