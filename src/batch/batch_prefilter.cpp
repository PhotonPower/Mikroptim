// FP32-Vorfilterung (M7-T2): konservative Entscheidung und CPU-Referenz in float.
// VORFILTERUNG: Die float-Werte dienen nur dem Aussortieren und fließen nie in Merit, Jacobi
// oder Optimierung ein (AGENTS.md §6.1, ADR-0003; docs/numerik.md, Abschnitt 12).

#include "mikroptim/batch/batch_prefilter.hpp"

#include "mikroptim/merit/prefilter_ray.hpp"
#include "mikroptim/merit/ray_residual.hpp"
#include "mikroptim/pupil/quadrature.hpp"
#include "mikroptim/trace/surface_table.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace mko {

namespace {

std::size_t to_size(std::int64_t n) {
    return static_cast<std::size_t>(n);
}

// Prüft die Optionen (std::invalid_argument).
void check_criteria(const PrefilterCriteria& c) {
    if (!(c.max_failed_fraction >= 0.0)) {
        throw std::invalid_argument("Vorfilterung: max_failed_fraction muss >= 0 sein");
    }
    if (!(c.max_merit >= 0.0)) {
        throw std::invalid_argument("Vorfilterung: max_merit muss >= 0 sein");
    }
    if (!(c.max_merit_factor >= 0.0)) {
        throw std::invalid_argument("Vorfilterung: max_merit_factor muss >= 0 sein");
    }
}

void check_boundary(double boundary) {
    if (!(boundary >= 0.0) || !std::isfinite(boundary)) {
        throw std::invalid_argument("Vorfilterung: Grenzabstand η muss endlich und >= 0 sein");
    }
}

} // namespace

void check_prefilter_options(const PrefilterOptions& o) {
    check_criteria(o.criteria);
    check_boundary(o.margins.boundary);
    if (!(o.margins.opl_error_factor >= 0.0) || !std::isfinite(o.margins.opl_error_factor)) {
        throw std::invalid_argument("Vorfilterung: opl_error_factor muss endlich und >= 0 sein");
    }
    if (!(o.margins.merit_relative >= 0.0 && o.margins.merit_relative < 1.0)) {
        throw std::invalid_argument("Vorfilterung: merit_relative muss in [0, 1) liegen");
    }
}

namespace {

void check_extra(std::span<const double> extra, int variants) {
    if (!extra.empty() && extra.size() != static_cast<std::size_t>(variants)) {
        throw std::invalid_argument("Vorfilterung: extra_merit muss leer sein oder " +
                                    std::to_string(variants) + " Werte haben");
    }
}

double extra_at(std::span<const double> extra, int v) {
    return extra.empty() ? 0.0 : extra[static_cast<std::size_t>(v)];
}

// Kriterium (a) mit der Zahl versagter Strahlen.
bool too_many_failed(std::int64_t failed, std::int64_t rays, double fraction) {
    return static_cast<double>(failed) > fraction * static_cast<double>(rays);
}

// Flächentabelle einer Variante in float (wie die Kopie im Shared Memory des Kernels).
struct FloatTable {
    std::vector<float> curvature, thickness, index, ghost_coeff, ghost_index, ghost_radius;
    SurfaceTableView<float> view{};

    explicit FloatTable(const SurfaceTableView<double>& t) {
        const auto conv = [](const double* p, std::size_t n, std::vector<float>& out) {
            out.resize(n);
            for (std::size_t i = 0; i < n; ++i) {
                out[i] = static_cast<float>(p[i]);
            }
        };
        const auto S = static_cast<std::size_t>(t.surfaces);
        const auto L = static_cast<std::size_t>(t.wavelengths);
        const auto G = static_cast<std::size_t>(t.ghost_plates);
        const auto K = static_cast<std::size_t>(t.ghost_terms);
        conv(t.curvature, S, curvature);
        conv(t.thickness, S, thickness);
        conv(t.index, S * L, index);
        if (G > 0) {
            conv(t.ghost_coeff, G * K, ghost_coeff);
            conv(t.ghost_index, G * L, ghost_index);
            conv(t.ghost_norm_radius, G, ghost_radius);
        }
        view = SurfaceTableView<float>{t.surfaces,
                                       t.wavelengths,
                                       t.type,
                                       curvature.data(),
                                       thickness.data(),
                                       index.data(),
                                       t.ghost_plates,
                                       t.ghost_terms,
                                       G > 0 ? ghost_coeff.data() : nullptr,
                                       G > 0 ? ghost_index.data() : nullptr,
                                       G > 0 ? ghost_radius.data() : nullptr};
    }
};

PupilQuadrature<float> to_float(const PupilQuadrature<double>& q) {
    PupilQuadrature<float> r{};
    r.rings = q.rings;
    r.spokes = q.spokes;
    r.mode = q.mode;
    for (int i = 0; i < q.rings; ++i) {
        r.u[i] = static_cast<float>(q.u[i]);
        r.w[i] = static_cast<float>(q.w[i]);
    }
    return r;
}

// Rohdaten einer Variante; füllt optional die Strahldaten (Ablage von ResidualLayout).
struct VariantStats {
    double merit = 0.0;
    double optical_length = 0.0;
    std::int64_t failed = 0;
    std::int64_t uncertain_ok = 0;
    std::int64_t uncertain_failed = 0;
};

VariantStats variant_stats(const SystemBatchHost& batch, int v, const ResidualLayout& layout,
                           double boundary, PrefilterRayData* rays) {
    const SystemBatchView<double> view = batch.view();
    const FloatTable table(view.surface_table(v));
    const auto eta = static_cast<float>(boundary);
    VariantStats out;
    double merit = 0.0;            // Summe der Tupel wie die zweite Stufe des Kernels (double)
    float optical = 0.0f;          // Maximum
    std::vector<float> opl;        // OPL − Versatz der Stützstellen eines Tupels
    std::vector<std::uint8_t> cls; // 0 sicher gültig, 1 sicher versagt, 2/3 unsicher
    for (int f = 0; f < layout.fields(); ++f) {
        const PupilQuadrature<float> q = to_float(layout.quadrature(f));
        const int n = sample_count(q);
        const Vec3<float> center = batch_image_point(view, table.view, v, f);
        opl.assign(static_cast<std::size_t>(n), 0.0f);
        cls.assign(static_cast<std::size_t>(n), 0);
        for (int l = 0; l < layout.wavelengths(); ++l) {
            const float offset = prefilter_opl_offset(table.view, l);
            const std::int64_t o = layout.offset(f, l);
            float weight_ok = 0.0f;
            float weighted_opl = 0.0f;
            for (int k = 0; k < n; ++k) {
                const auto ku = static_cast<std::size_t>(k);
                const PupilSample<float> s = sample(q, k);
                const PrefilterRay<float> r =
                    prefilter_ray(view, table.view, l, v, f, l, s, center, offset);
                const bool ok = r.status == TraceStatus::Ok;
                const bool certain = r.margin > eta;
                cls[ku] = static_cast<std::uint8_t>(ok ? (certain ? 0 : 2) : (certain ? 1 : 3));
                opl[ku] = r.opl;
                if (cls[ku] == 0) {
                    weight_ok += s.weight;
                    weighted_opl += s.weight * r.opl;
                    const float len = std::abs(offset) + std::abs(r.opl);
                    optical = len > optical ? len : optical;
                }
                out.failed += cls[ku] == 1 ? 1 : 0;
                out.uncertain_ok += cls[ku] == 2 ? 1 : 0;
                out.uncertain_failed += cls[ku] == 3 ? 1 : 0;
                if (rays != nullptr) {
                    const std::size_t i = to_size(o + k);
                    rays->status[i] = static_cast<std::uint8_t>(r.status);
                    rays->margin[i] = static_cast<double>(r.margin);
                    rays->certain[i] = certain ? 1 : 0;
                    rays->optical_length[i] =
                        cls[ku] == 0 ? static_cast<double>(std::abs(offset) + std::abs(r.opl))
                                     : 0.0;
                }
            }
            const bool valid = weight_ok > 0.0f;
            const float piston = valid ? weighted_opl / weight_ok : 0.0f;
            const auto scale = static_cast<float>(residual_scale(view, f, l));
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                const auto ku = static_cast<std::size_t>(k);
                float e = 0.0f;
                if (valid && cls[ku] == 0) {
                    e = opl[ku] - piston;
                }
                const float r = ray_residual(scale, sample(q, k), e);
                sum += r * r;
                if (rays != nullptr) {
                    rays->opd[to_size(o + k)] = static_cast<double>(e);
                }
            }
            merit += static_cast<double>(sum);
        }
    }
    out.merit = merit;
    out.optical_length = static_cast<double>(optical);
    return out;
}

void resize_stats(PrefilterStats& s, int variants, std::int64_t rays) {
    const auto n = static_cast<std::size_t>(variants);
    s.variants = variants;
    s.rays_per_variant = rays;
    s.merit.assign(n, 0.0);
    s.optical_length.assign(n, 0.0);
    s.failed.assign(n, 0);
    s.uncertain_ok.assign(n, 0);
    s.uncertain_failed.assign(n, 0);
}

} // namespace

PrefilterResult prefilter_decide(PrefilterStats stats, const SystemBatchHost& batch,
                                 const ResidualLayout& layout, const PrefilterOptions& options,
                                 std::span<const double> extra_merit) {
    check_prefilter_options(options);
    check_residual_layout(batch.topology(), layout);
    const int V = stats.variants;
    const auto n = static_cast<std::size_t>(V);
    if (V != batch.variants() || stats.rays_per_variant != layout.rays_per_variant() ||
        stats.merit.size() != n || stats.optical_length.size() != n || stats.failed.size() != n ||
        stats.uncertain_ok.size() != n || stats.uncertain_failed.size() != n) {
        throw std::invalid_argument("prefilter_decide: Rohdaten passen nicht zu Batch und Layout");
    }
    check_extra(extra_merit, V);

    // √(Σ_(f,l) s²): residual_scale hängt nur von (f, l) und den gemeinsamen Gewichten ab
    const SystemBatchView<double> view = batch.view();
    double scale_sq = 0.0;
    for (int f = 0; f < layout.fields(); ++f) {
        for (int l = 0; l < layout.wavelengths(); ++l) {
            const double s = residual_scale(view, f, l);
            scale_sq += s * s;
        }
    }
    const double scale_norm = std::sqrt(scale_sq);
    const PrefilterCriteria& c = options.criteria;
    const PrefilterMargins& m = options.margins;
    const std::int64_t R = stats.rays_per_variant;
    const bool soft_failure = layout.soft_failure();

    PrefilterResult res;
    res.merit_lower.assign(n, 0.0);
    res.merit_upper.assign(n, 0.0);
    res.reject.assign(n, kPrefilterKeep);
    for (int v = 0; v < V; ++v) {
        const auto i = static_cast<std::size_t>(v);
        const double root = std::sqrt(std::max(stats.merit[i], 0.0));
        const double delta =
            m.opl_error_factor * kFloatUnitRoundoff * stats.optical_length[i] * scale_norm;
        const double lower = std::max(0.0, root - delta) / (1.0 + m.merit_relative);
        const double upper = (root + delta) / (1.0 - m.merit_relative);
        const double extra = extra_at(extra_merit, v);
        res.merit_lower[i] = lower * lower + extra;
        // Unsichere Strahlen: ihre OPL in FP64 ist unbekannt, also keine obere Schranke. NaN
        // in den Rohdaten (z. B. Überlauf) macht beide Schranken unbrauchbar. Weiches Versagen
        // (M4-T2): Versagte Strahlen tragen in FP64 Straf- und (weich versagt) Wellenfront-
        // residuen, die der Vorfilter nicht rechnet; dann ebenfalls keine obere Schranke. Die
        // untere bleibt gültig, weil beide Anteile >= 0 sind und weitere Strahlen im Kolben die
        // Quadratsumme nur vergrößern (docs/numerik.md, Abschnitt 12).
        const bool finite = std::isfinite(root) && std::isfinite(delta);
        const bool penalties = soft_failure && stats.failed[i] > 0;
        res.merit_upper[i] = stats.uncertain(v) == 0 && finite && !penalties
                                 ? upper * upper + extra
                                 : std::numeric_limits<double>::infinity();
        if (!finite) {
            res.merit_lower[i] = extra;
        }
        if (too_many_failed(stats.failed[i], R, c.max_failed_fraction)) {
            res.reject[i] |= kPrefilterFailedRays;
        }
        if (res.merit_lower[i] > c.max_merit) {
            res.reject[i] |= kPrefilterMeritAbsolute;
        }
    }
    // Referenz für (c): Varianten ohne unsichere Strahlen (FP64 entscheidet ihre Strahlen
    // gleich), die (a) nicht erfüllen und mindestens einen gültigen Strahl haben.
    if (std::isfinite(c.max_merit_factor)) {
        for (int v = 0; v < V; ++v) {
            const auto i = static_cast<std::size_t>(v);
            if (stats.uncertain(v) != 0 || (res.reject[i] & kPrefilterFailedRays) != 0 ||
                stats.failed[i] >= R) {
                continue;
            }
            if (res.merit_upper[i] < res.reference_merit) {
                res.reference_merit = res.merit_upper[i];
                res.reference_variant = v;
            }
        }
        if (res.reference_variant >= 0) {
            const double limit = c.max_merit_factor * res.reference_merit;
            for (int v = 0; v < V; ++v) {
                const auto i = static_cast<std::size_t>(v);
                if (res.merit_lower[i] > limit) {
                    res.reject[i] |= kPrefilterMeritRelative;
                }
            }
        }
    }
    res.kept = static_cast<int>(std::count(res.reject.begin(), res.reject.end(), kPrefilterKeep));
    res.stats = std::move(stats);
    return res;
}

std::vector<std::uint8_t> prefilter_decide_exact(const MeritResult& fp64,
                                                 const PrefilterCriteria& criteria,
                                                 std::span<const double> extra_merit) {
    check_criteria(criteria);
    const int V = fp64.variants;
    const auto n = static_cast<std::size_t>(V);
    if (fp64.merit.size() != n || fp64.failed.size() != n) {
        throw std::invalid_argument("prefilter_decide_exact: Größen passen nicht");
    }
    check_extra(extra_merit, V);
    const std::int64_t R = fp64.rays_per_variant;
    std::vector<std::uint8_t> reject(n, kPrefilterKeep);
    double best = std::numeric_limits<double>::infinity();
    for (int v = 0; v < V; ++v) {
        const auto i = static_cast<std::size_t>(v);
        const double merit = fp64.merit[i] + extra_at(extra_merit, v);
        const bool failed = too_many_failed(fp64.failed[i], R, criteria.max_failed_fraction);
        if (failed) {
            reject[i] |= kPrefilterFailedRays;
        }
        if (merit > criteria.max_merit) {
            reject[i] |= kPrefilterMeritAbsolute;
        }
        if (!failed && fp64.failed[i] < R && merit < best) {
            best = merit;
        }
    }
    if (std::isfinite(criteria.max_merit_factor) && std::isfinite(best)) {
        const double limit = criteria.max_merit_factor * best;
        for (int v = 0; v < V; ++v) {
            const auto i = static_cast<std::size_t>(v);
            if (fp64.merit[i] + extra_at(extra_merit, v) > limit) {
                reject[i] |= kPrefilterMeritRelative;
            }
        }
    }
    return reject;
}

PrefilterRayData prefilter_rays_cpu(const SystemBatchHost& batch, int v,
                                    const ResidualLayout& layout, double boundary_margin) {
    check_residual_variant(batch, v, layout);
    check_boundary(boundary_margin);
    const std::size_t rays = to_size(layout.rays_per_variant());
    PrefilterRayData d;
    d.status.assign(rays, 0);
    d.margin.assign(rays, 0.0);
    d.opd.assign(rays, 0.0);
    d.optical_length.assign(rays, 0.0);
    d.certain.assign(rays, 0);
    (void)variant_stats(batch, v, layout, boundary_margin, &d);
    return d;
}

PrefilterStats prefilter_stats_cpu(const SystemBatchHost& batch, const ResidualLayout& layout,
                                   double boundary_margin) {
    check_boundary(boundary_margin);
    for (int v = 0; v < batch.variants(); ++v) {
        check_residual_variant(batch, v, layout);
    }
    PrefilterStats s;
    resize_stats(s, batch.variants(), layout.rays_per_variant());
    for (int v = 0; v < batch.variants(); ++v) {
        const auto i = static_cast<std::size_t>(v);
        const VariantStats vs = variant_stats(batch, v, layout, boundary_margin, nullptr);
        s.merit[i] = vs.merit;
        s.optical_length[i] = vs.optical_length;
        s.failed[i] = vs.failed;
        s.uncertain_ok[i] = vs.uncertain_ok;
        s.uncertain_failed[i] = vs.uncertain_failed;
    }
    return s;
}

PrefilterResult prefilter_cpu(const SystemBatchHost& batch, const ResidualLayout& layout,
                              const PrefilterOptions& options,
                              std::span<const double> extra_merit) {
    check_prefilter_options(options);
    return prefilter_decide(prefilter_stats_cpu(batch, layout, options.margins.boundary), batch,
                            layout, options, extra_merit);
}

} // namespace mko
