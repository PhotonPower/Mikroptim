// Monte-Carlo-Tolerierung (M8-T4): Tranchen von Stichproben, Kompensator, Bewertung aus der
// Quadratur, Kennzahlen (docs/optimierung.md, Abschnitt „Monte-Carlo-Tolerierung“).

#include "mikroptim/tolerance/tolerance_analysis.hpp"

#include "mikroptim/tolerance/piston_tilt.hpp"
#include "mikroptim/trace/ray.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace mko {

namespace {

std::size_t to_size(std::int64_t i) {
    return static_cast<std::size_t>(i);
}

constexpr double kTwoPi = 2.0 * std::numbers::pi;
const double kNaN = std::numeric_limits<double>::quiet_NaN();

} // namespace

void tolerance_trace_cpu(const SystemBatchHost& batch, const ResidualLayout& layout,
                         ResidualResult& out, int requested_threads) {
    const int variants = batch.variants();
    const std::int64_t rays = layout.rays_per_variant();
    out = ResidualResult{};
    out.variants = variants;
    out.rays_per_variant = rays;
    out.penalty_rows = 0;
    out.stride = rays;
    out.opd.assign(to_size(rays) * to_size(variants), 0.0);
    out.status.assign(to_size(rays) * to_size(variants), 0);
    const std::int64_t work = rays * static_cast<std::int64_t>(variants) * batch.surfaces();
    const int threads = std::min(lm_host_threads(requested_threads, work, kLmRaySurfacesPerThread),
                                 std::max(variants, 1));
    const auto run = [&](int begin, int end) {
        for (int v = begin; v < end; ++v) {
            const ResidualResult r = residuals_cpu(batch, v, layout);
            std::copy(r.opd.begin(), r.opd.end(), out.opd.begin() + out.ray_index(v, 0));
            std::copy(r.status.begin(), r.status.end(),
                      out.status.begin() + static_cast<std::ptrdiff_t>(out.ray_index(v, 0)));
        }
    };
    if (threads <= 1) {
        run(0, variants);
        return;
    }
    std::vector<std::thread> pool;
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(threads));
    for (int t = 0; t < threads; ++t) {
        const int begin = static_cast<int>(static_cast<std::int64_t>(variants) * t / threads);
        const int end = static_cast<int>(static_cast<std::int64_t>(variants) * (t + 1) / threads);
        pool.emplace_back([&, t, begin, end] {
            try {
                run(begin, end);
            } catch (...) {
                errors[static_cast<std::size_t>(t)] = std::current_exception();
            }
        });
    }
    for (std::thread& th : pool) {
        th.join();
    }
    for (const std::exception_ptr& e : errors) {
        if (e) {
            std::rethrow_exception(e);
        }
    }
}

std::vector<double> tolerance_tuple_weights(const SystemBatchHost& batch) {
    double wf = 0.0;
    for (int f = 0; f < batch.fields(); ++f) {
        wf += batch.field_weight(f);
    }
    double wl = 0.0;
    for (int l = 0; l < batch.wavelengths(); ++l) {
        wl += batch.wavelength_weight(l);
    }
    std::vector<double> out;
    for (int f = 0; f < batch.fields(); ++f) {
        for (int l = 0; l < batch.wavelengths(); ++l) {
            const double a = wf > 0.0 ? batch.field_weight(f) / wf : 0.0;
            const double b = wl > 0.0 ? batch.wavelength_weight(l) / wl : 0.0;
            out.push_back(a * b);
        }
    }
    return out;
}

namespace {

// Gemeinsamer Ablauf einer Tranche.
class Runner {
public:
    Runner(const SystemDescription& system, const SystemBatchHost& nominal,
           const ToleranceSettings& settings)
        : settings_(settings), base_(make_non_coaxial(nominal)), model_(system, base_.topology()) {
        ResidualSettings eval = settings.quadrature;
        eval.failure = FailureMode::Hard;
        layout_ = ResidualLayout(base_.topology(), eval);
        compensator_ = settings.compensator
                           ? *settings.compensator
                           : (system.tolerances ? system.tolerances->compensator_or_default()
                                                : kDefaultCompensator);
        if (compensator_ == CompensatorType::WorkingDistance) {
            SystemDescription wd = system;
            Variables vars;
            VariableSpec spec;
            spec.type = VariableType::WorkingDistance;
            spec.selection = SurfaceSelection::None;
            vars.items.push_back(spec);
            wd.variables = vars;
            wd.constraints.reset();
            wd.tolerances.reset();
            set_ = VariableSet(wd, base_.topology());
            lm_ = settings.lm;
            lm_.jacobian.quadrature.rings = settings.quadrature.rings;
            lm_.jacobian.quadrature.spokes = settings.quadrature.spokes;
            lm_.jacobian.quadrature.failure = settings.quadrature.failure;
            lm_.jacobian.quadrature.fail_weight = settings.quadrature.fail_weight;
            lm_.jacobian.extra = ExtraResiduals{};
            lm_.jacobian.trace_extra = TraceExtraResiduals{};
        }
        weights_ = tolerance_tuple_weights(base_);
        lambda_ref_mm_ = base_.reference_wavelength_nm * 1e-6;
    }

    const ToleranceModel& model() const { return model_; }
    CompensatorType compensator() const { return compensator_; }
    const SystemBatchHost& base() const { return base_; }
    const std::vector<double>& weights() const { return weights_; }

    // Stichproben [first_index, first_index + count); nominal: ξ = 0.
    void run(std::uint64_t first_index, int count, bool nominal,
             std::vector<ToleranceSampleResult>& out) {
        SystemBatchHost batch;
        batch.assign_variants(base_, 0, 1);
        batch.resize(count);
        if (nominal) {
            const std::vector<double> zero(static_cast<std::size_t>(model_.draws()), 0.0);
            for (int j = 0; j < count; ++j) {
                batch.copy_variant_from(base_, 0, j);
                model_.apply(batch, j, zero);
            }
        } else {
            model_.write_samples(base_, 0, settings_.seed, first_index, batch, 0, count);
        }
        std::vector<double> before(static_cast<std::size_t>(count));
        for (int j = 0; j < count; ++j) {
            before[static_cast<std::size_t>(j)] = batch.image_distance(j);
        }
        LmResult lm;
        if (compensator_ == CompensatorType::WorkingDistance) {
            lm = settings_.optimizer ? settings_.optimizer(set_, batch, before, lm_)
                                     : optimize_lm_cpu(set_, batch, before, lm_);
            if (lm.runs.size() != static_cast<std::size_t>(count)) {
                throw std::runtime_error("Tolerierung: Optimierer lieferte " +
                                         std::to_string(lm.runs.size()) + " statt " +
                                         std::to_string(count) + " Läufe");
            }
            for (int j = 0; j < count; ++j) {
                set_.apply(batch, j, lm.runs[static_cast<std::size_t>(j)].x);
            }
        }
        ResidualResult rr;
        if (settings_.trace) {
            settings_.trace(batch, layout_, rr);
        } else {
            tolerance_trace_cpu(batch, layout_, rr, settings_.threads);
        }
        const std::size_t need =
            to_size(layout_.rays_per_variant()) * static_cast<std::size_t>(count);
        if (rr.opd.size() != need || rr.status.size() != need ||
            rr.rays_per_variant != layout_.rays_per_variant()) {
            throw std::runtime_error("Tolerierung: Trace lieferte OPD und Status nicht in der "
                                     "erwarteten Größe");
        }
        for (int j = 0; j < count; ++j) {
            ToleranceSampleResult r = evaluate(batch, rr, j);
            r.working_distance = batch.image_distance(j);
            if (compensator_ == CompensatorType::WorkingDistance) {
                const LmRunResult& run = lm.runs[static_cast<std::size_t>(j)];
                r.compensation = r.working_distance - before[static_cast<std::size_t>(j)];
                r.compensator_status = run.status;
                r.compensator_iterations = run.iterations;
            }
            out.push_back(std::move(r));
        }
    }

private:
    ToleranceSampleResult evaluate(const SystemBatchHost& batch, const ResidualResult& rr,
                                   int v) const {
        ToleranceSampleResult r;
        const int fields = layout_.fields();
        const int wavelengths = layout_.wavelengths();
        double sum = 0.0;
        double strehl_min = std::numeric_limits<double>::infinity();
        for (int f = 0; f < fields; ++f) {
            const PupilQuadrature<double>& q = layout_.quadrature(f);
            const auto n = static_cast<std::size_t>(layout_.samples(f));
            for (int l = 0; l < wavelengths; ++l) {
                const std::size_t at = rr.ray_index(v, layout_.offset(f, l));
                const TupleQuality t =
                    tuple_quality(q, std::span<const double>(rr.opd).subspan(at, n),
                                  std::span<const std::uint8_t>(rr.status).subspan(at, n),
                                  batch.wavelength_nm(l) * 1e-6);
                const double rms = t.rms / lambda_ref_mm_;
                r.rms.push_back(rms);
                r.strehl.push_back(t.strehl);
                r.failed_rays += t.failed;
                const double w = weights_[static_cast<std::size_t>(f * wavelengths + l)];
                sum += w * (rms * rms);
                if (w > 0.0) {
                    strehl_min = std::min(strehl_min, t.strehl);
                }
            }
        }
        r.rms_composite = std::sqrt(sum);
        r.strehl_min = std::isfinite(strehl_min) ? strehl_min : kNaN;
        return r;
    }

    const ToleranceSettings& settings_;
    SystemBatchHost base_;
    ToleranceModel model_;
    ResidualLayout layout_;
    CompensatorType compensator_ = CompensatorType::None;
    VariableSet set_;
    LmOptions lm_;
    std::vector<double> weights_;
    double lambda_ref_mm_ = 0.0;
};

} // namespace

TupleQuality tuple_quality(const PupilQuadrature<double>& q, std::span<const double> opd,
                           std::span<const std::uint8_t> status, double lambda_mm,
                           std::span<double> rest) {
    const int n = sample_count(q);
    if (opd.size() != static_cast<std::size_t>(n) || status.size() != static_cast<std::size_t>(n) ||
        (!rest.empty() && rest.size() != static_cast<std::size_t>(n))) {
        throw std::invalid_argument("tuple_quality: OPD und Status brauchen " + std::to_string(n) +
                                    " Einträge");
    }
    std::fill(rest.begin(), rest.end(), 0.0);
    if (!(lambda_mm > 0.0)) {
        throw std::invalid_argument("tuple_quality: Wellenlänge muss > 0 sein");
    }
    TupleQuality out;
    // Anpassung von Kolben und Kippung über die Strahlen mit Status Ok (piston_tilt.hpp)
    std::vector<std::uint8_t> use(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) {
        const auto ku = static_cast<std::size_t>(k);
        use[ku] = static_cast<TraceStatus>(status[ku]) == TraceStatus::Ok ? 1 : 0;
        if (use[ku] == 0) {
            ++out.failed;
        }
    }
    const PistonTiltFit fit(q, use);
    const double a00 = fit.fit_weight();
    if (!fit.valid()) {
        out.rms = kNaN;
        out.strehl = 0.0;
        return out;
    }
    double b0 = 0.0, b1 = 0.0, b2 = 0.0;
    for (int k = 0; k < n; ++k) {
        if (!fit.used(k)) {
            continue;
        }
        const double w = fit.weight(k);
        const double o = opd[static_cast<std::size_t>(k)];
        b0 += w * o;
        b1 += w * (fit.x(k) * o);
        b2 += w * (fit.y(k) * o);
    }
    const std::array<double, 3> c = fit.solve(b0, b1, b2);
    const double c0 = c[0];
    const double tx = c[1];
    const double ty = c[2];
    out.tilt_x = tx;
    out.tilt_y = ty;
    double sum2 = 0.0;
    double re = 0.0;
    double im = 0.0;
    for (int k = 0; k < n; ++k) {
        const auto ku = static_cast<std::size_t>(k);
        if (static_cast<TraceStatus>(status[ku]) != TraceStatus::Ok) {
            continue;
        }
        const double e = opd[ku] - c0 - tx * fit.x(k) - ty * fit.y(k);
        if (!rest.empty()) {
            rest[ku] = e;
        }
        const double w = fit.weight(k);
        sum2 += w * (e * e);
        const double phase = kTwoPi * (e / lambda_mm);
        re += w * std::cos(phase);
        im += w * std::sin(phase);
    }
    out.rms = std::sqrt(sum2 / a00);
    out.strehl = (re * re + im * im) / (fit.total_weight() * fit.total_weight());
    return out;
}

DistributionStats distribution_stats(std::span<const double> values, bool higher_is_worse) {
    DistributionStats s;
    const std::size_t n = values.size();
    double sum = 0.0;
    s.min = std::numeric_limits<double>::infinity();
    s.max = -std::numeric_limits<double>::infinity();
    for (const double v : values) {
        if (std::isfinite(v)) {
            ++s.count;
            sum += v;
            s.min = std::min(s.min, v);
            s.max = std::max(s.max, v);
        }
    }
    if (s.count == 0) {
        s.mean = s.std = s.min = s.max = kNaN;
    } else {
        s.mean = sum / s.count;
        double sq = 0.0;
        for (const double v : values) {
            if (std::isfinite(v)) {
                const double d = v - s.mean;
                sq += d * d;
            }
        }
        s.std = s.count > 1 ? std::sqrt(sq / (s.count - 1)) : 0.0;
    }
    if (n == 0) {
        s.p50 = s.p90 = s.p95 = s.p99 = kNaN;
        return s;
    }
    // NaN als schlechtester Wert
    const double worst = higher_is_worse ? std::numeric_limits<double>::infinity()
                                         : -std::numeric_limits<double>::infinity();
    std::vector<double> sorted(values.begin(), values.end());
    for (double& v : sorted) {
        if (std::isnan(v)) {
            v = worst;
        }
    }
    std::sort(sorted.begin(), sorted.end());
    const auto pick = [&](std::size_t percent) {
        const std::size_t rank = (percent * n + 99) / 100; // ⌈p n⌉, 1 … n
        const double v = higher_is_worse ? sorted[rank - 1] : sorted[n - rank];
        return std::isfinite(v) ? v : kNaN;
    };
    s.p50 = pick(50);
    s.p90 = pick(90);
    s.p95 = pick(95);
    s.p99 = pick(99);
    return s;
}

void summarize_tolerance_result(ToleranceResult& r) {
    const std::size_t n = r.results.size();
    const std::size_t tuples =
        static_cast<std::size_t>(r.fields) * static_cast<std::size_t>(r.wavelengths);
    std::vector<double> values(n);
    const auto stats = [&](auto get, bool higher_is_worse) {
        for (std::size_t i = 0; i < n; ++i) {
            values[i] = get(r.results[i]);
        }
        return distribution_stats(values, higher_is_worse);
    };
    r.rms_composite = stats([](const ToleranceSampleResult& s) { return s.rms_composite; }, true);
    r.strehl_min = stats([](const ToleranceSampleResult& s) { return s.strehl_min; }, false);
    r.compensation = stats([](const ToleranceSampleResult& s) { return s.compensation; }, true);
    r.rms.clear();
    r.strehl.clear();
    for (std::size_t t = 0; t < tuples; ++t) {
        r.rms.push_back(stats([t](const ToleranceSampleResult& s) { return s.rms[t]; }, true));
        r.strehl.push_back(
            stats([t](const ToleranceSampleResult& s) { return s.strehl[t]; }, false));
    }
    std::size_t good = 0;
    r.failed_samples = 0;
    for (const ToleranceSampleResult& s : r.results) {
        if (s.strehl_min >= r.strehl_threshold) {
            ++good;
        }
        if (s.failed_rays > 0) {
            ++r.failed_samples;
        }
    }
    r.yield = n > 0 ? static_cast<double>(good) / static_cast<double>(n) : kNaN;
}

ToleranceResult run_tolerance_analysis(const SystemDescription& system,
                                       const SystemBatchHost& nominal,
                                       const ToleranceSettings& settings) {
    if (settings.samples < 1) {
        throw std::invalid_argument("Tolerierung: Zahl der Stichproben muss >= 1 sein");
    }
    if (settings.chunk < 1) {
        throw std::invalid_argument("Tolerierung: Tranchengröße muss >= 1 sein");
    }
    if (!std::isfinite(settings.strehl_threshold)) {
        throw std::invalid_argument("Tolerierung: Strehl-Schwelle muss endlich sein");
    }
    if (nominal.variants() != 1) {
        throw std::invalid_argument("Tolerierung: Ausgangsbatch muss genau eine Variante haben");
    }
    Runner runner(system, nominal, settings);
    ToleranceResult r;
    r.samples = settings.samples;
    r.seed = settings.seed;
    r.compensator = runner.compensator();
    r.draws = runner.model().draws();
    r.fields = runner.base().fields();
    r.wavelengths = runner.base().wavelengths();
    for (int l = 0; l < r.wavelengths; ++l) {
        r.wavelength_nm.push_back(runner.base().wavelength_nm(l));
    }
    r.reference_nm = runner.base().reference_wavelength_nm;
    r.tuple_weight = runner.weights();
    r.strehl_threshold = settings.strehl_threshold;

    std::vector<ToleranceSampleResult> nominal_result;
    runner.run(0, 1, true, nominal_result);
    r.nominal = std::move(nominal_result.front());

    r.results.reserve(static_cast<std::size_t>(settings.samples));
    for (int first = 0; first < settings.samples; first += settings.chunk) {
        const int count = std::min(settings.chunk, settings.samples - first);
        runner.run(static_cast<std::uint64_t>(first), count, false, r.results);
        ++r.chunks;
        if (settings.progress) {
            settings.progress(first + count, settings.samples);
        }
    }
    summarize_tolerance_result(r);
    return r;
}

} // namespace mko
