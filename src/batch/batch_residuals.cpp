// Residuenvektor pro Variante: Ablage und CPU-Referenz des Kernels RESIDUALS.

#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/merit/opd.hpp"
#include "mikroptim/merit/ray_residual.hpp"

#include <climits>
#include <limits>
#include <stdexcept>
#include <string>

namespace mko {

namespace {

std::size_t to_size(std::int64_t n) {
    return static_cast<std::size_t>(n);
}

} // namespace

// ---- ResidualLayout ----------------------------------------------------------------------

ResidualLayout::ResidualLayout(const BatchTopology& topology, const ResidualSettings& settings)
    : fields_(topology.fields), wavelengths_(topology.wavelengths), settings_(settings) {
    validate_topology(topology);
    if (topology.pupil_mode.size() != static_cast<std::size_t>(topology.fields)) {
        throw std::invalid_argument("ResidualLayout: Pupillenart pro Feld ist nicht festgelegt "
                                    "(BatchTopology::pupil_mode)");
    }
    if (settings.spokes > 0 && settings.rings > INT_MAX / settings.spokes) {
        throw std::invalid_argument("ResidualLayout: zu viele Stützstellen pro Feld");
    }
    std::int64_t offset = 0;
    for (int f = 0; f < fields_; ++f) {
        const PupilMode mode = topology.pupil_mode[static_cast<std::size_t>(f)];
        // Prüft rings und spokes (std::invalid_argument)
        const PupilQuadrature<double> q =
            make_pupil_quadrature(settings.rings, settings.spokes, mode);
        const int n = sample_count(q);
        quadrature_.push_back(q);
        plan_.push_back(ResidualFieldPlan{offset, n, mode});
        offset += static_cast<std::int64_t>(wavelengths_) * n;
        if (mode == PupilMode::Axis) {
            ++axis_fields_;
        }
    }
    rays_per_variant_ = offset;
}

void ResidualLayout::check_field(int f) const {
    if (f < 0 || f >= fields_) {
        throw std::out_of_range("ResidualLayout: Feld " + std::to_string(f) + " existiert nicht");
    }
}

PupilMode ResidualLayout::pupil_mode(int f) const {
    check_field(f);
    return plan_[static_cast<std::size_t>(f)].mode;
}

int ResidualLayout::samples(int f) const {
    check_field(f);
    return plan_[static_cast<std::size_t>(f)].samples;
}

const PupilQuadrature<double>& ResidualLayout::quadrature(int f) const {
    check_field(f);
    return quadrature_[static_cast<std::size_t>(f)];
}

std::int64_t ResidualLayout::offset(int f, int l) const {
    check_field(f);
    if (l < 0 || l >= wavelengths_) {
        throw std::out_of_range("ResidualLayout: Wellenlänge " + std::to_string(l) +
                                " existiert nicht");
    }
    const ResidualFieldPlan& p = plan_[static_cast<std::size_t>(f)];
    return p.offset + static_cast<std::int64_t>(l) * p.samples;
}

void check_residual_layout(const BatchTopology& topology, const ResidualLayout& layout) {
    bool same = layout.fields() == topology.fields &&
                layout.wavelengths() == topology.wavelengths &&
                topology.pupil_mode.size() == static_cast<std::size_t>(topology.fields);
    for (int f = 0; same && f < layout.fields(); ++f) {
        same = layout.pupil_mode(f) == topology.pupil_mode[static_cast<std::size_t>(f)];
    }
    if (!same) {
        throw std::invalid_argument(
            "Residuen: ResidualLayout passt nicht zur Topologie des Batches (Felder, "
            "Wellenlängen oder Pupillenarten)");
    }
}

// ---- Ergebnis ----------------------------------------------------------------------------

std::size_t ResidualResult::failed_count() const {
    std::size_t n = 0;
    for (const std::uint8_t s : status) {
        n += s != static_cast<std::uint8_t>(TraceStatus::Ok) ? 1 : 0;
    }
    return n;
}

std::vector<TupleWavefront> tuple_wavefronts(const ResidualResult& result,
                                             const ResidualLayout& layout, int v) {
    if (v < 0 || v >= result.variants) {
        throw std::out_of_range("tuple_wavefronts: Variante " + std::to_string(v) +
                                " existiert nicht");
    }
    const std::size_t total =
        static_cast<std::size_t>(result.variants) * to_size(result.rays_per_variant);
    if (result.rays_per_variant != layout.rays_per_variant()) {
        throw std::invalid_argument("tuple_wavefronts: Ergebnis passt nicht zum ResidualLayout");
    }
    if (result.opd.size() != total || result.status.size() != total) {
        throw std::invalid_argument("tuple_wavefronts: OPD und Status pro Strahl werden "
                                    "gebraucht (im Lauf anfordern)");
    }
    std::vector<TupleWavefront> out;
    out.reserve(static_cast<std::size_t>(layout.fields()) *
                static_cast<std::size_t>(layout.wavelengths()));
    for (int f = 0; f < layout.fields(); ++f) {
        const PupilQuadrature<double>& q = layout.quadrature(f);
        const int n = layout.samples(f);
        for (int l = 0; l < layout.wavelengths(); ++l) {
            const std::int64_t o = layout.offset(f, l);
            TupleWavefront t;
            t.field = f;
            t.wavelength = l;
            t.samples = n;
            double sum_sq = 0.0;
            for (int k = 0; k < n; ++k) {
                const std::size_t i = result.ray_index(v, o + k);
                if (static_cast<TraceStatus>(result.status[i]) == TraceStatus::Ok) {
                    const double w = sample(q, k).weight;
                    const double e = result.opd[i];
                    t.valid_weight += w;
                    sum_sq += w * e * e; // wie opd_statistics
                } else {
                    ++t.failed;
                }
            }
            if (t.valid_weight > 0.0) {
                t.rms = msqrt(sum_sq / t.valid_weight);
            } else {
                t.valid_weight = 0.0;
                t.rms = std::numeric_limits<double>::quiet_NaN();
            }
            out.push_back(t);
        }
    }
    return out;
}

// ---- CPU-Referenz ------------------------------------------------------------------------

namespace {

// Schreibt die Residuen, OPD und Status der Variante v ab den Zeigern (Ablage des Layouts).
void variant_residuals(const SystemBatchHost& batch, int v, const ResidualLayout& layout,
                       double* residuals, double* opd, std::uint8_t* status) {
    const SystemBatchView<double> view = batch.view();
    const SurfaceTableView<double> table = view.surface_table(v);
    std::vector<double> opl;
    std::vector<double> weight;
    std::vector<TraceStatus> st;
    std::vector<PupilSample<double>> pupil;
    for (int f = 0; f < layout.fields(); ++f) {
        const PupilQuadrature<double>& q = layout.quadrature(f);
        const int n = sample_count(q);
        const auto nu = static_cast<std::size_t>(n);
        pupil.resize(nu);
        weight.resize(nu);
        opl.resize(nu);
        st.resize(nu);
        for (std::size_t k = 0; k < nu; ++k) {
            pupil[k] = sample(q, static_cast<int>(k));
            weight[k] = pupil[k].weight;
        }
        const Vec3<double> center = batch_image_point(view, table, v, f);
        for (int l = 0; l < layout.wavelengths(); ++l) {
            const std::size_t o = to_size(layout.offset(f, l));
            for (std::size_t k = 0; k < nu; ++k) {
                st[k] = batch_ray_opl(view, table, l, v, f, l, pupil[k], center, &opl[k]);
            }
            // Kolbenabzug mit hartem Versagen; OPD versagter Strahlen (und aller Strahlen eines
            // Tupels ohne gültigen Strahl) wird 0.
            opd_statistics(opl.data(), weight.data(), st.data(), n, opd + o);
            const double scale = residual_scale(view, f, l);
            for (std::size_t k = 0; k < nu; ++k) {
                residuals[o + k] = ray_residual(scale, pupil[k], opd[o + k]);
                status[o + k] = static_cast<std::uint8_t>(st[k]);
            }
        }
    }
}

void check_variant(const SystemBatchHost& batch, int v, const ResidualLayout& layout) {
    if (v < 0 || v >= batch.variants()) {
        throw std::out_of_range("residuals_cpu: Variante " + std::to_string(v) +
                                " existiert nicht");
    }
    check_residual_layout(batch.topology(), layout);
    check_pupil_modes(batch, v, 1);
    if (batch.source() == SourceKind::Point) {
        // Gleiche Prüfung wie source_ray (cpu_trace.cpp): sin_max = NA / n in (0, 1)
        for (int l = 0; l < batch.wavelengths(); ++l) {
            const double sin_max = batch.source_na / batch.index(v, 0, l);
            if (!(sin_max > 0.0 && sin_max < 1.0)) {
                throw std::invalid_argument(
                    "residuals_cpu: NA der Punktquelle muss in (0, n) liegen");
            }
        }
    }
}

ResidualResult make_result(int variants, const ResidualLayout& layout) {
    ResidualResult r;
    r.variants = variants;
    r.rays_per_variant = layout.rays_per_variant();
    r.stride = r.rays_per_variant;
    const std::size_t total = static_cast<std::size_t>(variants) * to_size(r.rays_per_variant);
    r.residuals.assign(total, 0.0);
    r.opd.assign(total, 0.0);
    r.status.assign(total, static_cast<std::uint8_t>(TraceStatus::Ok));
    return r;
}

} // namespace

ResidualResult residuals_cpu(const SystemBatchHost& batch, int v, const ResidualLayout& layout) {
    check_variant(batch, v, layout);
    ResidualResult r = make_result(1, layout);
    variant_residuals(batch, v, layout, r.residuals.data(), r.opd.data(), r.status.data());
    return r;
}

ResidualResult residuals_cpu(const SystemBatchHost& batch, const ResidualLayout& layout) {
    for (int v = 0; v < batch.variants(); ++v) {
        check_variant(batch, v, layout);
    }
    ResidualResult r = make_result(batch.variants(), layout);
    for (int v = 0; v < batch.variants(); ++v) {
        const std::size_t o = r.ray_index(v, 0);
        variant_residuals(batch, v, layout, r.residuals.data() + o, r.opd.data() + o,
                          r.status.data() + o);
    }
    return r;
}

} // namespace mko
