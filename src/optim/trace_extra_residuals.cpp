// Trace-abhängige Zusatzresiduen aus den Max-Reduktionen (docs/optimierung.md, Abschnitt 2.1).

#include "mikroptim/optim/trace_extra_residuals.hpp"

#include "parallel_variants.hpp"

#include <stdexcept>
#include <string>

namespace mko {

namespace {

std::size_t to_size(int n) {
    return static_cast<std::size_t>(n);
}

} // namespace

VariantExtrema variant_extrema(const SurfaceExtremaResult& extrema, int i) {
    if (i < 0 || i >= extrema.variants ||
        extrema.values.size() < extrema.index(i, 0) + 2 * to_size(extrema.surfaces)) {
        throw std::out_of_range("Extrema: Eintrag " + std::to_string(i) + " fehlt (" +
                                std::to_string(extrema.variants) + " Varianten)");
    }
    return VariantExtrema{extrema.surfaces, extrema.values.data() + extrema.index(i, 0)};
}

void require_hard_extrema(const TraceExtraResiduals& extra, const char* who) {
    if (extra.count > 0 && extra.smoothing.enabled()) {
        throw std::invalid_argument(std::string(who) +
                                    ": geglättete Extrema (constraints.extrema_smoothing) gibt es "
                                    "auf der GPU noch nicht (M11-T1b); CPU verwenden");
    }
}

void check_trace_extra_residuals(const TraceExtraResiduals& extra, const char* who) {
    if (extra.count < 0) {
        throw std::invalid_argument(std::string(who) +
                                    ": negative Zahl trace-abhängiger Zusatzresiduen");
    }
    if (extra.count > 0 && !extra.evaluate) {
        throw std::invalid_argument(std::string(who) +
                                    ": trace-abhängige Zusatzresiduen ohne Auswertefunktion");
    }
}

void evaluate_trace_extra_residuals(const TraceExtraResiduals& extra, const SystemBatchHost& batch,
                                    const SurfaceExtremaResult& extrema, int extrema_first,
                                    std::span<double> out, int first, int count, int threads) {
    const int total = batch.variants();
    if (first < 0 || first > total || count > total - first) {
        throw std::out_of_range("Trace-Zusatzresiduen: Variantenbereich außerhalb des Batches");
    }
    if (count < 0) {
        count = total - first;
    }
    check_trace_extra_residuals(extra, "Trace-Zusatzresiduen");
    const std::size_t per = to_size(extra.count);
    if (out.size() != per * to_size(count)) {
        throw std::invalid_argument("Trace-Zusatzresiduen: Ausgabe hat " +
                                    std::to_string(out.size()) + " Einträge, erwartet " +
                                    std::to_string(per * to_size(count)));
    }
    if (per == 0 || count == 0) {
        return;
    }
    if (extrema.surfaces != batch.surfaces()) {
        throw std::invalid_argument("Trace-Zusatzresiduen: Extrema mit " +
                                    std::to_string(extrema.surfaces) + " Flächen, Batch mit " +
                                    std::to_string(batch.surfaces()));
    }
    if (first - extrema_first < 0 || first + count - extrema_first > extrema.variants) {
        throw std::out_of_range("Trace-Zusatzresiduen: Extrema fehlen für Varianten [" +
                                std::to_string(first) + ", " + std::to_string(first + count) + ")");
    }
    detail::parallel_for_variants(first, count, threads, [&](int v) {
        extra.evaluate(batch, v, variant_extrema(extrema, v - extrema_first),
                       out.subspan(to_size(v - first) * per, per));
    });
}

namespace {

// Σ t_i² aufsteigend in i.
double sum_squares(std::span<const double> t) {
    double s = 0.0;
    for (const double ti : t) {
        s += ti * ti;
    }
    return s;
}

} // namespace

double trace_extra_merit(const TraceExtraResiduals& extra, const SystemBatchHost& batch, int v,
                         const VariantExtrema& extrema) {
    check_trace_extra_residuals(extra, "Trace-Zusatzresiduen");
    if (extra.count == 0) {
        return 0.0;
    }
    std::vector<double> t(to_size(extra.count));
    extra.evaluate(batch, v, extrema, t);
    return sum_squares(t);
}

std::vector<double> trace_extra_merit_values(const TraceExtraResiduals& extra,
                                             const SystemBatchHost& batch,
                                             const SurfaceExtremaResult& extrema, int threads) {
    if (extra.count == 0) {
        return {};
    }
    const std::size_t E = to_size(extra.count);
    std::vector<double> values(to_size(batch.variants()) * E);
    evaluate_trace_extra_residuals(extra, batch, extrema, 0, values, 0, -1, threads);
    std::vector<double> out(to_size(batch.variants()), 0.0);
    for (std::size_t v = 0; v < out.size(); ++v) {
        out[v] = sum_squares(std::span<const double>(values).subspan(v * E, E));
    }
    return out;
}

std::vector<double> join_extra_rows(std::span<const double> a, int a_count,
                                    std::span<const double> b, int b_count, int variants) {
    if (a_count < 0 || b_count < 0 || variants < 0 ||
        a.size() != to_size(a_count) * to_size(variants) ||
        b.size() != to_size(b_count) * to_size(variants)) {
        throw std::invalid_argument("Zusatzresiduen: Blöcke passen nicht zusammen");
    }
    const std::size_t na = to_size(a_count);
    const std::size_t nb = to_size(b_count);
    std::vector<double> out(to_size(variants) * (na + nb));
    for (std::size_t v = 0; v < to_size(variants); ++v) {
        double* row = out.data() + v * (na + nb);
        for (std::size_t i = 0; i < na; ++i) {
            row[i] = a[v * na + i];
        }
        for (std::size_t i = 0; i < nb; ++i) {
            row[na + i] = b[v * nb + i];
        }
    }
    return out;
}

} // namespace mko
