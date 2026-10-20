// Modus EVAL (M8-T1) auf dem Host: Raster, Referenzgeometrie, CPU-Referenz, Kennzahlen,
// OPD-Karten und Speicherabschätzung (docs/architektur.md, Abschnitt "Modus EVAL";
// docs/numerik.md, Abschnitt 7).

#include "mikroptim/eval/eval.hpp"
#include "mikroptim/merit/opd.hpp"
#include "mikroptim/merit/ray_residual.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace mko {

namespace {

std::size_t to_size(std::int64_t i) {
    return static_cast<std::size_t>(i);
}

// Löst G c = h für eine symmetrisch positiv definite n×n-Matrix (zeilenweise, n <= 4) per
// Cholesky. false, wenn G nicht positiv definit ist (zu wenige oder entartete Punkte).
bool solve_spd(std::array<double, 16> G, std::array<double, 4>& h, int n) {
    for (int j = 0; j < n; ++j) {
        double d = G[to_size(j * n + j)];
        for (int k = 0; k < j; ++k) {
            d -= G[to_size(j * n + k)] * G[to_size(j * n + k)];
        }
        if (!(d > 1e-14 * G[to_size(j * n + j)]) || !(d > 0.0)) {
            return false;
        }
        const double ljj = std::sqrt(d);
        G[to_size(j * n + j)] = ljj;
        for (int i = j + 1; i < n; ++i) {
            double s = G[to_size(i * n + j)];
            for (int k = 0; k < j; ++k) {
                s -= G[to_size(i * n + k)] * G[to_size(j * n + k)];
            }
            G[to_size(i * n + j)] = s / ljj;
        }
    }
    for (int i = 0; i < n; ++i) { // L y = h
        double s = h[to_size(i)];
        for (int k = 0; k < i; ++k) {
            s -= G[to_size(i * n + k)] * h[to_size(k)];
        }
        h[to_size(i)] = s / G[to_size(i * n + i)];
    }
    for (int i = n - 1; i >= 0; --i) { // Lᵀ c = y
        double s = h[to_size(i)];
        for (int k = i + 1; k < n; ++k) {
            s -= G[to_size(k * n + i)] * h[to_size(k)];
        }
        h[to_size(i)] = s / G[to_size(i * n + i)];
    }
    return true;
}

// Fits der OPD eines Tupels: Basis (1, x, y) bzw. (1, x, y, ρ²) auf den gültigen Punkten.
struct TupleFits {
    bool tilt_ok = false;
    bool defocus_ok = false;
    std::array<double, 4> tilt{};    // c₀, t_x, t_y
    std::array<double, 4> defocus{}; // c₀, t_x, t_y, c_d
};

// Basisfunktionen am Punkt (x, y).
std::array<double, 4> basis(double x, double y) {
    return {1.0, x, y, x * x + y * y};
}

double fit_value(const std::array<double, 4>& c, int n, double x, double y) {
    const std::array<double, 4> phi = basis(x, y);
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        s += c[to_size(i)] * phi[to_size(i)];
    }
    return s;
}

// Ruft fn(x, y, w, spiegel) für jede Kopie eines Rasterpunkts auf: volle Pupille einmal mit
// Gewicht w, halbe Pupille als Paar (±x, y) mit je w/2 (spiegel = −1 für die Kopie bei −x).
template <class Fn>
void for_each_copy(const EvalGrid& grid, const EvalPoint& p, Fn&& fn) {
    if (grid.half) {
        fn(p.x, p.y, 0.5 * p.weight, 1.0);
        fn(-p.x, p.y, 0.5 * p.weight, -1.0);
    } else {
        fn(p.x, p.y, p.weight, 1.0);
    }
}

bool ray_ok(const EvalResult& r, std::size_t i) {
    return static_cast<TraceStatus>(r.status[i]) == TraceStatus::Ok;
}

TupleFits tuple_fits(const EvalResult& r, int v, int f, int l) {
    TupleFits fits;
    std::array<double, 16> G{};
    std::array<double, 4> h{};
    const std::int64_t n = r.samples();
    for (std::int64_t k = 0; k < n; ++k) {
        const std::size_t i = r.ray_index(v, f, l, k);
        if (!ray_ok(r, i)) {
            continue;
        }
        const double e = r.opd[i];
        for_each_copy(r.grid, r.grid.points[to_size(k)], [&](double x, double y, double w, double) {
            const std::array<double, 4> phi = basis(x, y);
            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
                    G[to_size(a * 4 + b)] += w * phi[to_size(a)] * phi[to_size(b)];
                }
                h[to_size(a)] += w * phi[to_size(a)] * e;
            }
        });
    }
    std::array<double, 16> G3{};
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            G3[to_size(a * 3 + b)] = G[to_size(a * 4 + b)];
        }
    }
    fits.tilt = {h[0], h[1], h[2], 0.0};
    fits.tilt_ok = solve_spd(G3, fits.tilt, 3);
    fits.defocus = h;
    fits.defocus_ok = solve_spd(G, fits.defocus, 4);
    return fits;
}

void check_variant(const EvalResult& r, int v) {
    if (v < 0 || v >= r.variants) {
        throw std::out_of_range("EVAL: Variante " + std::to_string(v) + " existiert nicht");
    }
}

} // namespace

// ---- Einstellungen und Raster ----------------------------------------------------------------

void check_eval_settings(const EvalSettings& settings) {
    if (settings.grid < 1 || settings.grid > kMaxEvalGrid) {
        throw std::invalid_argument("EVAL: Rasterkantenlänge muss in [1, " +
                                    std::to_string(kMaxEvalGrid) + "] liegen, ist " +
                                    std::to_string(settings.grid));
    }
    switch (settings.reference) {
    case EvalReference::ExitPupil:
    case EvalReference::Vertex:
        break;
    default:
        throw std::invalid_argument("EVAL: unbekannte Lage der Referenzsphäre");
    }
    if (!(settings.field_ray_offset > 0.0) || !(settings.field_ray_offset <= 0.5)) {
        throw std::invalid_argument("EVAL: Abstand der Feldstrahlen muss in (0, 0,5] liegen");
    }
}

void check_eval_settings(const EvalSettings& settings, const SystemBatchHost& batch) {
    check_eval_settings(settings);
    if (settings.half_pupil && batch.source() == SourceKind::Point &&
        batch.source_point[0] != 0.0) {
        throw std::invalid_argument("EVAL: halbe Pupille nur bei Symmetrie x → −x (Punktquelle "
                                    "mit x ≠ 0)");
    }
    if (settings.half_pupil && !batch.coaxial()) {
        throw std::invalid_argument("EVAL: halbe Pupille nur bei Symmetrie x → −x (System nicht "
                                    "koaxial, ADR-0011)");
    }
}

EvalGrid make_eval_grid(int size, bool half_pupil) {
    if (size < 1 || size > kMaxEvalGrid) {
        throw std::invalid_argument("EVAL: Rasterkantenlänge muss in [1, " +
                                    std::to_string(kMaxEvalGrid) + "] liegen, ist " +
                                    std::to_string(size));
    }
    EvalGrid g;
    g.size = size;
    g.half = half_pupil;
    const std::int64_t N = size;
    const auto inside = [N](std::int64_t i, std::int64_t j) {
        const std::int64_t a = 2 * i + 1 - N;
        const std::int64_t b = 2 * j + 1 - N;
        return a * a + b * b <= N * N;
    };
    for (std::int64_t j = 0; j < N; ++j) {
        for (std::int64_t i = 0; i < N; ++i) {
            g.full_count += inside(i, j) ? 1 : 0;
        }
    }
    const double full = static_cast<double>(g.full_count);
    for (std::int64_t j = 0; j < N; ++j) {
        for (std::int64_t i = 0; i < N; ++i) {
            if (!inside(i, j)) {
                continue;
            }
            const std::int64_t a = 2 * i + 1 - N;
            if (half_pupil && a < 0) {
                continue;
            }
            EvalPoint p;
            p.x = static_cast<double>(a) / static_cast<double>(N);
            p.y = static_cast<double>(2 * j + 1 - N) / static_cast<double>(N);
            p.weight = (half_pupil && a > 0 ? 2.0 : 1.0) / full;
            g.points.push_back(p);
            g.column.push_back(static_cast<std::int32_t>(i));
            g.row.push_back(static_cast<std::int32_t>(j));
        }
    }
    return g;
}

EvalReferenceGeometry eval_reference_geometry(const SystemBatchHost& batch,
                                              EvalReference reference) {
    const int V = batch.variants();
    const int F = batch.fields();
    const int S = batch.surfaces();
    EvalReferenceGeometry geo;
    geo.curvature.assign(to_size(V) * to_size(F), 0.0);
    geo.exit_pupil_position.assign(to_size(V), EvalResult::kNaN);
    for (int v = 0; v < V; ++v) {
        const double w = batch.thickness(v, S - 1);
        double z_ap = 0.0;
        bool infinite = false;
        if (reference == EvalReference::ExitPupil) {
            const ParaxialData p = batch_paraxial(batch, v);
            infinite = p.exit_pupil_at_infinity;
            z_ap = p.exit_pupil_position;
            if (!infinite) {
                if (!std::isfinite(z_ap)) {
                    throw std::invalid_argument("EVAL: Lage der Austrittspupille in Variante " +
                                                std::to_string(v) + " nicht endlich");
                }
                geo.exit_pupil_position[to_size(v)] = z_ap;
            }
        }
        for (int f = 0; f < F; ++f) {
            const double h = batch.field_target(v, f);
            double kappa = 0.0;
            if (!infinite) {
                const double dz = w - z_ap;
                const double radius = std::hypot(h, dz);
                if (!(radius > 0.0) || !std::isfinite(radius)) {
                    if (reference == EvalReference::ExitPupil) {
                        throw std::invalid_argument(
                            "EVAL: Bildpunkt liegt in der Austrittspupille (Variante " +
                            std::to_string(v) + ", Feld " + std::to_string(f) + ")");
                    }
                    kappa = EvalResult::kNaN; // nur zur Information (Modus Vertex)
                } else {
                    kappa = (dz >= 0.0 ? 1.0 : -1.0) / radius;
                }
            }
            geo.curvature[to_size(v) * to_size(F) + to_size(f)] = kappa;
        }
    }
    return geo;
}

// ---- Ergebnis und CPU-Referenz ---------------------------------------------------------------

std::size_t EvalResult::failed_count() const {
    std::size_t n = 0;
    for (const std::uint8_t s : status) {
        n += static_cast<TraceStatus>(s) == TraceStatus::Ok ? 0 : 1;
    }
    return n;
}

void prepare_eval_result(EvalResult& out, const SystemBatchHost& batch,
                         const EvalSettings& settings, EvalGrid grid,
                         EvalReferenceGeometry geometry) {
    const int V = batch.variants();
    const int F = batch.fields();
    const int L = batch.wavelengths();
    out.variants = V;
    out.fields = F;
    out.wavelengths = L;
    out.settings = settings;
    out.grid = std::move(grid);
    out.reference_nm = batch.reference_wavelength_nm;
    out.image_height.resize(to_size(V) * to_size(F));
    for (int v = 0; v < V; ++v) {
        for (int f = 0; f < F; ++f) {
            out.image_height[to_size(v) * to_size(F) + to_size(f)] = batch.field_target(v, f);
        }
    }
    out.reference_curvature = std::move(geometry.curvature);
    out.exit_pupil_position = std::move(geometry.exit_pupil_position);
    const std::size_t tuples = to_size(V) * to_size(F) * to_size(L);
    const std::size_t rays = tuples * to_size(out.grid.samples());
    out.opd.resize(rays);
    out.spot_x.resize(rays);
    out.spot_y.resize(rays);
    out.status.resize(rays);
    out.field_rays.resize(tuples * kEvalFieldRays * kEvalFieldRayValues);
    out.field_status.resize(tuples * kEvalFieldRays);
}

EvalResult eval_cpu(const SystemBatchHost& batch, const EvalSettings& settings) {
    check_eval_settings(settings, batch);
    batch.validate_values();
    EvalResult out;
    prepare_eval_result(out, batch, settings, make_eval_grid(settings.grid, settings.half_pupil),
                        eval_reference_geometry(batch, settings.reference));
    const SystemBatchView<double> view = batch.view();
    const int F = batch.fields();
    const int L = batch.wavelengths();
    const std::int64_t n = out.samples();
    const auto nu = to_size(n);
    std::vector<double> weight(nu);
    for (std::size_t k = 0; k < nu; ++k) {
        weight[k] = out.grid.points[k].weight;
    }
    std::vector<double> opl(nu);
    std::vector<TraceStatus> status(nu);
    std::vector<double> opd(nu);
    for (int v = 0; v < out.variants; ++v) {
        const SurfaceTableView<double> table = view.surface_table(v);
        for (int f = 0; f < F; ++f) {
            const Vec3<double> center = batch_image_point(view, table, v, f);
            const double kappa = out.reference_curvature[to_size(v) * to_size(F) + to_size(f)];
            for (int l = 0; l < L; ++l) {
                const std::size_t base = out.ray_index(v, f, l, 0);
                // Bezugswert der OPL: Hauptstrahl (Pupillenmitte), 0 wenn er versagt. Die
                // Summen des Kolbens laufen über OPL − Bezug (Größe der OPD statt ~100 mm).
                const EvalRay<double> chief =
                    eval_ray(view, table, l, v, f, l, 0.0, 0.0, center, settings.reference, kappa);
                const double offset = chief.status == TraceStatus::Ok ? chief.opl : 0.0;
                for (std::size_t k = 0; k < nu; ++k) {
                    const EvalPoint& p = out.grid.points[k];
                    const EvalRay<double> r = eval_ray(view, table, l, v, f, l, p.x, p.y, center,
                                                       settings.reference, kappa);
                    const bool ok = r.status == TraceStatus::Ok;
                    status[k] = r.status;
                    opl[k] = r.opl - offset;
                    out.status[base + k] = static_cast<std::uint8_t>(r.status);
                    out.spot_x[base + k] = ok ? r.image.x - center.x : EvalResult::kNaN;
                    out.spot_y[base + k] = ok ? r.image.y - center.y : EvalResult::kNaN;
                }
                const OpdStatistics<double> st = opd_statistics(
                    opl.data(), weight.data(), status.data(), static_cast<int>(n), opd.data());
                const bool valid = st.valid_weight > 0.0;
                for (std::size_t k = 0; k < nu; ++k) {
                    out.opd[base + k] =
                        valid && status[k] == TraceStatus::Ok ? opd[k] : EvalResult::kNaN;
                }
                const std::size_t t = out.tuple(v, f, l);
                for (int j = 0; j < kEvalFieldRays; ++j) {
                    double x = 0.0;
                    double y = 0.0;
                    eval_field_ray_point(j, settings.field_ray_offset, x, y);
                    const EvalRay<double> r =
                        eval_ray(view, table, l, v, f, l, x, y, center, settings.reference, kappa);
                    const bool ok = r.status == TraceStatus::Ok;
                    double* fr = out.field_rays.data() +
                                 (t * kEvalFieldRays + to_size(j)) * kEvalFieldRayValues;
                    fr[0] = ok ? r.image.x - center.x : EvalResult::kNaN;
                    fr[1] = ok ? r.image.y - center.y : EvalResult::kNaN;
                    fr[2] = ok ? r.d.x : EvalResult::kNaN;
                    fr[3] = ok ? r.d.y : EvalResult::kNaN;
                    fr[4] = ok ? r.d.z : EvalResult::kNaN;
                    out.field_status[t * kEvalFieldRays + to_size(j)] =
                        static_cast<std::uint8_t>(r.status);
                }
            }
        }
    }
    return out;
}

// ---- Kennzahlen ------------------------------------------------------------------------------

std::vector<EvalTupleStats> eval_statistics(const EvalResult& r, int v) {
    check_variant(r, v);
    std::vector<EvalTupleStats> out;
    out.reserve(to_size(r.fields) * to_size(r.wavelengths));
    const std::int64_t n = r.samples();
    for (int f = 0; f < r.fields; ++f) {
        for (int l = 0; l < r.wavelengths; ++l) {
            EvalTupleStats s;
            s.field = f;
            s.wavelength = l;
            s.rays = n;
            s.image_height = r.image_height[to_size(v) * to_size(r.fields) + to_size(f)];
            // Durchlauf 1: Gewicht, Momente der OPD und des Spots
            double w_ok = 0.0;
            double sum_opd2 = 0.0;
            double sum_x = 0.0;
            double sum_y = 0.0;
            double opd_min = 0.0;
            double opd_max = 0.0;
            bool first = true;
            for (std::int64_t k = 0; k < n; ++k) {
                const std::size_t i = r.ray_index(v, f, l, k);
                if (!ray_ok(r, i)) {
                    ++s.failed;
                    continue;
                }
                const double e = r.opd[i];
                opd_min = first ? e : std::min(opd_min, e);
                opd_max = first ? e : std::max(opd_max, e);
                first = false;
                for_each_copy(r.grid, r.grid.points[to_size(k)],
                              [&](double, double, double w, double mirror) {
                                  w_ok += w;
                                  sum_opd2 += w * e * e;
                                  sum_x += w * (mirror * r.spot_x[i]);
                                  sum_y += w * r.spot_y[i];
                              });
            }
            s.valid_weight = w_ok;
            if (s.valid()) {
                s.rms_opd = std::sqrt(sum_opd2 / w_ok);
                s.pv_opd = opd_max - opd_min;
                s.centroid_x = sum_x / w_ok;
                s.centroid_y = sum_y / w_ok;
                const TupleFits fits = tuple_fits(r, v, f, l);
                if (fits.defocus_ok) {
                    s.tilt_x = fits.defocus[1];
                    s.tilt_y = fits.defocus[2];
                    s.defocus = fits.defocus[3];
                }
                // Durchlauf 2: RMS-Spot um den Schwerpunkt, RMS nach Abzug der Fits
                double sum_spot = 0.0;
                double sum_tilt = 0.0;
                double sum_defocus = 0.0;
                for (std::int64_t k = 0; k < n; ++k) {
                    const std::size_t i = r.ray_index(v, f, l, k);
                    if (!ray_ok(r, i)) {
                        continue;
                    }
                    const double e = r.opd[i];
                    for_each_copy(r.grid, r.grid.points[to_size(k)],
                                  [&](double x, double y, double w, double mirror) {
                                      const double dx = mirror * r.spot_x[i] - s.centroid_x;
                                      const double dy = r.spot_y[i] - s.centroid_y;
                                      sum_spot += w * (dx * dx + dy * dy);
                                      const double et = e - fit_value(fits.tilt, 3, x, y);
                                      const double ed = e - fit_value(fits.defocus, 4, x, y);
                                      sum_tilt += w * et * et;
                                      sum_defocus += w * ed * ed;
                                  });
                }
                s.rms_spot = std::sqrt(sum_spot / w_ok);
                if (fits.tilt_ok) {
                    s.rms_opd_tilt = std::sqrt(sum_tilt / w_ok);
                }
                if (fits.defocus_ok) {
                    s.rms_opd_defocus = std::sqrt(sum_defocus / w_ok);
                }
                if (s.image_height != 0.0) {
                    s.distortion_centroid = 100.0 * s.centroid_y / s.image_height;
                }
            }
            // Feldstrahlen: Hauptstrahl, tangentiales Paar, sagittaler Strahl
            const auto fr_ok = [&](int j) {
                return r.field_ray_status(v, f, l, j) == TraceStatus::Ok;
            };
            if (fr_ok(kEvalChiefRay)) {
                s.chief_x = r.field_ray(v, f, l, kEvalChiefRay, 0);
                s.chief_y = r.field_ray(v, f, l, kEvalChiefRay, 1);
                if (s.image_height != 0.0) {
                    s.distortion_chief = 100.0 * s.chief_y / s.image_height;
                }
            }
            if (fr_ok(kEvalTangentialPlus) && fr_ok(kEvalTangentialMinus)) {
                const double y1 = r.field_ray(v, f, l, kEvalTangentialPlus, 1);
                const double m1 = r.field_ray(v, f, l, kEvalTangentialPlus, 3) /
                                  r.field_ray(v, f, l, kEvalTangentialPlus, 4);
                const double y2 = r.field_ray(v, f, l, kEvalTangentialMinus, 1);
                const double m2 = r.field_ray(v, f, l, kEvalTangentialMinus, 3) /
                                  r.field_ray(v, f, l, kEvalTangentialMinus, 4);
                s.focus_tangential = (y2 - y1) / (m1 - m2);
            }
            if (fr_ok(kEvalSagittal)) {
                const double x3 = r.field_ray(v, f, l, kEvalSagittal, 0);
                const double m3 =
                    r.field_ray(v, f, l, kEvalSagittal, 2) / r.field_ray(v, f, l, kEvalSagittal, 4);
                s.focus_sagittal = -x3 / m3;
            }
            out.push_back(s);
        }
    }
    return out;
}

EvalMap eval_opd_map(const EvalResult& r, int v, int f, int l, OpdRemoval removal) {
    check_variant(r, v);
    if (f < 0 || f >= r.fields || l < 0 || l >= r.wavelengths) {
        throw std::out_of_range("EVAL: Tupel (" + std::to_string(f) + ", " + std::to_string(l) +
                                ") existiert nicht");
    }
    EvalMap map;
    const int N = r.grid.size;
    map.size = N;
    map.values.assign(to_size(N) * to_size(N), EvalResult::kNaN);
    TupleFits fits;
    if (removal != OpdRemoval::Piston) {
        fits = tuple_fits(r, v, f, l);
    }
    const auto value = [&](double e, double x, double y) {
        switch (removal) {
        case OpdRemoval::Piston:
            return e;
        case OpdRemoval::Tilt:
            return fits.tilt_ok ? e - fit_value(fits.tilt, 3, x, y) : EvalResult::kNaN;
        case OpdRemoval::Defocus:
            return fits.defocus_ok ? e - fit_value(fits.defocus, 4, x, y) : EvalResult::kNaN;
        }
        return EvalResult::kNaN;
    };
    const std::int64_t n = r.samples();
    for (std::int64_t k = 0; k < n; ++k) {
        const std::size_t i = r.ray_index(v, f, l, k);
        if (!ray_ok(r, i)) {
            continue;
        }
        const EvalPoint& p = r.grid.points[to_size(k)];
        const auto col = to_size(r.grid.column[to_size(k)]);
        const auto row = to_size(r.grid.row[to_size(k)]);
        map.values[row * to_size(N) + col] = value(r.opd[i], p.x, p.y);
        if (r.grid.half) {
            const std::size_t mirror = to_size(N) - 1 - col;
            map.values[row * to_size(N) + mirror] = value(r.opd[i], -p.x, p.y);
        }
    }
    return map;
}

// ---- Speicher --------------------------------------------------------------------------------

EvalOutputLayout make_eval_output_layout(const BatchTopology& topology, int variants,
                                         std::int64_t samples) {
    validate_topology(topology);
    if (variants < 0) {
        throw std::invalid_argument("make_eval_output_layout: negative Variantenzahl");
    }
    if (samples < 1) {
        throw std::invalid_argument("make_eval_output_layout: mindestens ein Rasterpunkt");
    }
    const auto V = to_size(variants);
    const auto F = to_size(topology.fields);
    const std::size_t tuples = V * F * to_size(topology.wavelengths);
    const std::size_t rays = tuples * to_size(samples);
    EvalOutputLayout o;
    std::size_t offset = 0;
    const auto place = [&offset](BufferRegion& region, std::size_t bytes) {
        region.offset = offset;
        region.bytes = bytes;
        offset += align_bytes(bytes);
    };
    place(o.opd, rays * sizeof(double));
    place(o.spot_x, rays * sizeof(double));
    place(o.spot_y, rays * sizeof(double));
    place(o.status, rays);
    place(o.field_rays, tuples * kEvalFieldRays * kEvalFieldRayValues * sizeof(double));
    place(o.field_status, tuples * kEvalFieldRays);
    place(o.reference_curvature, V * F * sizeof(double));
    place(o.points, to_size(samples) * sizeof(EvalPoint));
    o.total_bytes = offset;
    return o;
}

std::size_t eval_device_bytes(const BatchTopology& topology, int variants, std::int64_t samples) {
    return make_batch_layout(topology, variants).total_bytes +
           make_eval_output_layout(topology, variants, samples).total_bytes;
}

} // namespace mko
