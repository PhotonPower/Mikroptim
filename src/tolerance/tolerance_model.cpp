// Stichproben eines Toleranzsatzes und ihre Anwendung auf einen SystemBatch (M8-T4,
// docs/optimierung.md, Abschnitt „Monte-Carlo-Tolerierung“).

#include "mikroptim/tolerance/tolerance_model.hpp"

#include "mikroptim/optim/variable_set.hpp"
#include "mikroptim/optim/variant_generator.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace mko {

namespace {

using Mat3 = std::array<std::array<double, 3>, 3>;

// R = R_y(β) · R_x(α) (aktiv, rechtshändig; docs/numerik.md, Abschnitt 2).
Mat3 rotation(double alpha, double beta) {
    const double ca = std::cos(alpha);
    const double sa = std::sin(alpha);
    const double cb = std::cos(beta);
    const double sb = std::sin(beta);
    return Mat3{{{cb, sb * sa, sb * ca}, {0.0, ca, -sa}, {-sb, cb * sa, cb * ca}}};
}

Mat3 multiply(const Mat3& a, const Mat3& b) {
    Mat3 c{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    return c;
}

std::size_t to_size(int i) {
    return static_cast<std::size_t>(i);
}

} // namespace

double tolerance_deviate(std::mt19937_64& engine, ToleranceDistribution distribution,
                         double cutoff) {
    if (distribution == ToleranceDistribution::Uniform) {
        return 2.0 * unit_uniform(engine()) - 1.0; // exakt: 2u und 2u − 1
    }
    if (!(cutoff >= kMinToleranceCutoff && cutoff <= kMaxToleranceCutoff)) {
        throw std::invalid_argument("tolerance_deviate: Abschneiden c muss in [0.5, 6] liegen");
    }
    for (;;) {
        double sum = 0.0;
        for (int i = 0; i < 12; ++i) {
            sum += unit_uniform(engine());
        }
        const double z = sum - 6.0;
        if (std::abs(z) <= cutoff) {
            return z / cutoff;
        }
    }
}

SurfacePlacement move_surface(const SurfacePlacement& p, double z_s, const ElementMotion& motion) {
    SurfacePlacement out = p;
    if (motion.alpha == 0.0 && motion.beta == 0.0) {
        out.dx = p.dx + motion.dx;
        out.dy = p.dy + motion.dy;
        return out;
    }
    const Mat3 re = rotation(motion.alpha, motion.beta);
    // Scheitel relativ zum Drehpunkt: W = V − P mit V = (dx, dy, z_s + dz), P = (0, 0, z_P)
    const double l = z_s - motion.pivot_z;
    const std::array<double, 3> w{p.dx, p.dy, l + p.dz};
    out.dx = re[0][0] * w[0] + re[0][1] * w[1] + re[0][2] * w[2] + motion.dx;
    out.dy = re[1][0] * w[0] + re[1][1] * w[1] + re[1][2] * w[2] + motion.dy;
    out.dz = (re[2][0] * w[0] + re[2][1] * w[1] + re[2][2] * w[2]) - l;
    const Mat3 m = multiply(re, rotation(p.alpha, p.beta));
    const double nx = m[0][2];
    const double ny = m[1][2];
    const double nz = m[2][2];
    out.alpha = std::atan2(-ny, std::hypot(nx, nz));
    out.beta = std::atan2(nx, nz);
    return out;
}

ToleranceModel::ToleranceModel(const SystemDescription& system, const BatchTopology& topology)
    : tolerances_(resolve_tolerances(system)), topology_(topology) {
    if (!topology.non_coaxial) {
        throw std::invalid_argument(
            "ToleranceModel: verlangt einen nicht koaxialen Batch (make_non_coaxial)");
    }
    const SurfaceMap map = checked_surface_map(system, topology);
    file_surfaces_ = map.file_surfaces();
    table_of_file_ = map.table_of_file;
    vertex_z_.resize(to_size(file_surfaces_));
    double z = 0.0;
    for (int s = 0; s < file_surfaces_; ++s) {
        vertex_z_[to_size(s)] = z;
        z += system.surfaces[to_size(s)].thickness;
    }
    offset_.reserve(tolerances_.size());
    for (const ResolvedTolerance& t : tolerances_) {
        offset_.push_back(draws_);
        draws_ += t.draws();
    }
}

int ToleranceModel::draw_offset(int k) const {
    if (k < 0 || k >= static_cast<int>(offset_.size())) {
        throw std::out_of_range("ToleranceModel::draw_offset: Toleranz " + std::to_string(k) +
                                " existiert nicht");
    }
    return offset_[to_size(k)];
}

void ToleranceModel::deviates(std::uint64_t seed, std::uint64_t index, std::span<double> xi) const {
    if (xi.size() != to_size(draws_)) {
        throw std::invalid_argument("ToleranceModel::deviates: Länge " + std::to_string(xi.size()) +
                                    " statt " + std::to_string(draws_));
    }
    std::mt19937_64 engine(variant_seed(seed, index));
    std::size_t d = 0;
    for (const ResolvedTolerance& t : tolerances_) {
        for (int k = 0; k < t.draws(); ++k) {
            xi[d++] = tolerance_deviate(engine, t.distribution, t.cutoff);
        }
    }
}

std::vector<double> ToleranceModel::deviates(std::uint64_t seed, std::uint64_t index) const {
    std::vector<double> xi(to_size(draws_));
    deviates(seed, index, xi);
    return xi;
}

void ToleranceModel::check_batch(const SystemBatchHost& batch) const {
    if (!(batch.topology() == topology_)) {
        throw std::invalid_argument(
            "ToleranceModel: Batch hat eine andere Topologie als der Toleranzsatz");
    }
}

void ToleranceModel::apply(SystemBatchHost& batch, int v, std::span<const double> xi) const {
    check_batch(batch);
    if (xi.size() != to_size(draws_)) {
        throw std::invalid_argument("ToleranceModel::apply: Länge " + std::to_string(xi.size()) +
                                    " statt " + std::to_string(draws_));
    }
    if (v < 0 || v >= batch.variants()) {
        throw std::out_of_range("ToleranceModel::apply: Variante " + std::to_string(v) +
                                " existiert nicht");
    }
    const std::size_t n = to_size(file_surfaces_);
    // Lage pro Dateifläche, Start: Transformation der Variante (nominell, z. B. aus der Datei)
    std::vector<SurfacePlacement> place(n);
    std::vector<char> touched(n, 0); // Transformation zu schreiben
    std::vector<char> moved(n, 0);   // Elementbewegung (Dicken ändern sich um ±δz)
    for (std::size_t s = 1; s < n; ++s) {
        const int ts = table_of_file_[s];
        place[s].dx = batch.decenter_tilt(v, ts, 0);
        place[s].dy = batch.decenter_tilt(v, ts, 1);
        place[s].alpha = batch.decenter_tilt(v, ts, 2);
        place[s].beta = batch.decenter_tilt(v, ts, 3);
    }
    std::vector<double> dn(n, 0.0);
    std::vector<double> abbe(n, 0.0);
    std::vector<char> glass(n, 0);
    std::vector<char> has_abbe(n, 0);
    std::vector<std::size_t> elements; // Toleranzen mit Elementbewegung, in Reihenfolge

    for (std::size_t k = 0; k < tolerances_.size(); ++k) {
        const ResolvedTolerance& t = tolerances_[k];
        // + 0.0 macht aus −0 (Grenze 0, ξ < 0) +0, damit Null-Toleranzen den Batch bitgleich
        // lassen
        const double x1 = t.limit * xi[to_size(offset_[k])] + 0.0;
        const double x2 = t.draws() > 1 ? t.limit * xi[to_size(offset_[k]) + 1] + 0.0 : 0.0;
        const std::size_t s = to_size(t.surface);
        switch (t.type) {
        case ToleranceType::Radius: {
            double& c = batch.curvature(v, table_of_file_[s]);
            c = t.radius_fringes ? c + x1 : c / (1.0 + x1);
            break;
        }
        case ToleranceType::Thickness:
            // letztes Teilstück des Spalts nach Fläche s (vor der nächsten Dateifläche)
            batch.thickness(v, table_of_file_[s + 1] - 1) += x1;
            break;
        case ToleranceType::Index:
            dn[s] = x1;
            glass[s] = 1;
            break;
        case ToleranceType::Abbe:
            abbe[s] = x1;
            glass[s] = 1;
            has_abbe[s] = 1;
            break;
        case ToleranceType::Decenter:
            place[s].dx += x1;
            place[s].dy += x2;
            touched[s] = 1;
            break;
        case ToleranceType::Tilt:
            if (place[s].alpha == 0.0 && place[s].beta == 0.0) {
                place[s].alpha = x1;
                place[s].beta = x2;
            } else {
                // Kippung um den (ggf. schon dezentrierten) Scheitel der Fläche
                ElementMotion m;
                m.alpha = x1;
                m.beta = x2;
                SurfacePlacement local = place[s];
                local.dx = 0.0;
                local.dy = 0.0;
                local = move_surface(local, 0.0, m);
                place[s].alpha = local.alpha;
                place[s].beta = local.beta;
            }
            touched[s] = 1;
            break;
        case ToleranceType::ElementDecenter:
        case ToleranceType::ElementTilt:
            elements.push_back(k);
            break;
        }
    }
    // Elementbewegungen nach den Toleranzen der Einzelflächen (Bewegung des schon gestörten
    // Elements als starrer Körper)
    for (const std::size_t k : elements) {
        const ResolvedTolerance& t = tolerances_[k];
        const double x1 = t.limit * xi[to_size(offset_[k])] + 0.0;
        const double x2 = t.limit * xi[to_size(offset_[k]) + 1] + 0.0;
        ElementMotion m;
        if (t.type == ToleranceType::ElementDecenter) {
            m.dx = x1;
            m.dy = x2;
        } else {
            m.alpha = x1;
            m.beta = x2;
            m.pivot_z = t.pivot_z;
        }
        for (int s = t.first; s <= t.last; ++s) {
            const std::size_t su = to_size(s);
            place[su] = move_surface(place[su], vertex_z_[su], m);
            touched[su] = 1;
            moved[su] = 1;
        }
    }
    // Transformationen und Dickenänderungen der Elementbewegungen
    for (std::size_t s = 1; s < n; ++s) {
        if (!touched[s]) {
            continue;
        }
        const int ts = table_of_file_[s];
        const SurfacePlacement& p = place[s];
        batch.set_surface_transform(v, ts, p.dx, p.dy, p.alpha, p.beta);
        if (moved[s] && p.dz != 0.0) {
            batch.thickness(v, ts - 1) += p.dz; // letztes Teilstück vor der Fläche
            batch.thickness(v, ts) -= p.dz;     // erstes Teilstück danach (bzw. Arbeitsabstand)
        }
    }
    // Brechzahlen der gestörten Gläser
    const int wavelengths = batch.wavelengths();
    for (std::size_t s = 0; s < n; ++s) {
        if (!glass[s]) {
            continue;
        }
        const int ts = table_of_file_[s];
        const double n_ref = batch.paraxial_index(v, ts);
        const double scale = 1.0 + abbe[s];
        for (int l = 0; l < wavelengths; ++l) {
            double& nl = batch.index(v, ts, l);
            nl = has_abbe[s] ? (n_ref + dn[s]) + (nl - n_ref) / scale : nl + dn[s];
        }
        batch.paraxial_index(v, ts) = n_ref + dn[s];
    }
}

void ToleranceModel::write_samples(const SystemBatchHost& nominal, int base_variant,
                                   std::uint64_t seed, std::uint64_t first_index,
                                   SystemBatchHost& out, int first, int count) const {
    check_batch(nominal);
    check_batch(out);
    if (count < 0 || first < 0 || first + count > out.variants()) {
        throw std::out_of_range("ToleranceModel::write_samples: Bereich passt nicht in den Batch");
    }
    if (&nominal == &out) {
        throw std::invalid_argument("ToleranceModel::write_samples: Ziel und Vorlage sind derselbe "
                                    "Batch");
    }
    std::vector<double> xi(to_size(draws_));
    for (int j = 0; j < count; ++j) {
        out.copy_variant_from(nominal, base_variant, first + j);
        deviates(seed, first_index + static_cast<std::uint64_t>(j), xi);
        apply(out, first + j, xi);
    }
}

} // namespace mko
