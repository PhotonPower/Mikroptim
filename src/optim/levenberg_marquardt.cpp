// Gebündelter Levenberg-Marquardt (M3-T4): Steuerlogik und CPU-Backend
// (docs/optimierung.md, Abschnitt 4).

#include "mikroptim/optim/levenberg_marquardt.hpp"

#include "mikroptim/batch/batch_merit.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/optim/normal_equations.hpp"
#include "mikroptim/optim/paraxial_residuals.hpp"
#include "parallel_variants.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

namespace mko {

namespace {

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument("Levenberg-Marquardt: " + message);
}

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
// Dämpfung des Gauß-Newton-Modells für die Abbruchkriterien (relativ zu D; hält die Cholesky-
// Zerlegung bei rangdefizientem JᵀJ definiert, ändert gut bestimmte Richtungen nicht).
constexpr double kModelDamping = 1e-12;

// Zustand eines Laufs.
struct RunState {
    std::vector<double> x;
    std::vector<double> trial; // Versuchspunkt clamp(x + δ)
    std::vector<double> a;     // JᵀJ an x, P × P spaltenweise
    std::vector<double> g;     // Jᵀr an x
    std::vector<double> d;     // Marquardt-Skalierung (mit Untergrenze)
    double merit = 0.0;
    std::int64_t failed = 0;
    double mu = 0.0;
    double nu = 2.0;
    LmStatus status = LmStatus::Running;
    bool need_jacobian = true;
    bool rejected = false; // Schritt seit dem letzten angenommenen verworfen
    // Ergebnis des Versuchsschritts im aktuellen Zyklus
    bool solved = false;       // Cholesky erfolgreich
    double step_norm = 0.0;    // max_k |Δx_k|/s_k
    double predicted = 0.0;    // L(0) − L(Δx)
    double step_damping = 0.0; // μ des Schritts
    double gradient_norm = 0.0;
    double model_decrease = kNaN; // (L(0) − L(Δx_GN))/F der letzten Jacobi-Matrix
    int iterations = 0;
    int accepted = 0;
    int jacobians = 0;
    std::vector<double> history;
};

double scale_of(double x, double typical) {
    return std::max(std::abs(x), typical);
}

// Cholesky-Zerlegung M = LLᵀ in place (n × n spaltenweise, nur das untere Dreieck wird gelesen
// und geschrieben). false bei nicht positivem oder nicht endlichem Pivot.
bool cholesky(std::vector<double>& m, int n) {
    const auto N = static_cast<std::size_t>(n);
    for (std::size_t j = 0; j < N; ++j) {
        double diag = m[j * N + j];
        for (std::size_t k = 0; k < j; ++k) {
            diag -= m[k * N + j] * m[k * N + j];
        }
        if (!(diag > 0.0) || !std::isfinite(diag)) {
            return false;
        }
        const double ljj = std::sqrt(diag);
        m[j * N + j] = ljj;
        for (std::size_t i = j + 1; i < N; ++i) {
            double s = m[j * N + i];
            for (std::size_t k = 0; k < j; ++k) {
                s -= m[k * N + i] * m[k * N + j];
            }
            m[j * N + i] = s / ljj;
        }
    }
    return true;
}

// Löst LLᵀ y = b in place (L aus cholesky).
void cholesky_solve(const std::vector<double>& l, int n, std::vector<double>& b) {
    const auto N = static_cast<std::size_t>(n);
    for (std::size_t i = 0; i < N; ++i) { // L z = b
        double s = b[i];
        for (std::size_t k = 0; k < i; ++k) {
            s -= l[k * N + i] * b[k];
        }
        b[i] = s / l[i * N + i];
    }
    for (std::size_t i = N; i-- > 0;) { // Lᵀ y = z
        double s = b[i];
        for (std::size_t k = i + 1; k < N; ++k) {
            s -= l[i * N + k] * b[k];
        }
        b[i] = s / l[i * N + i];
    }
}

// Variable k liegt an einer Schranke, und der Abstiegsrichtung −g_k zeigt nach außen (oder der
// Bereich ist ein Punkt): für den Schritt festgehalten und aus dem Gradientenmaß genommen.
bool bound_blocks(const LmBounds& b, std::size_t k, double x, double dir) {
    if (b.lower[k] == b.upper[k]) {
        return true;
    }
    return (x <= b.lower[k] && dir < 0.0) || (x >= b.upper[k] && dir > 0.0);
}

// Marquardt-Skalierung und Gradientenmaß nach einer neuen Jacobi-Matrix.
void prepare_step(RunState& s, const LmBounds& b, int P, const LmOptions& o) {
    const auto n = static_cast<std::size_t>(P);
    double smax = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        const double sk = scale_of(s.x[k], b.typical[k]);
        smax = std::max(smax, s.a[k * n + k] * sk * sk);
    }
    double gmax = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        const double sk = scale_of(s.x[k], b.typical[k]);
        const double akk = s.a[k * n + k];
        const double floor = smax > 0.0 ? o.diag_floor * smax / (sk * sk) : 1.0 / (sk * sk);
        s.d[k] = std::max(akk, floor);
        if (!bound_blocks(b, k, s.x[k], -s.g[k])) {
            const double denom = std::sqrt(akk * s.merit);
            if (denom > 0.0) {
                gmax = std::max(gmax, std::abs(s.g[k]) / denom);
            }
        }
    }
    s.gradient_norm = gmax;
}

// Schritt mit aktiver Menge für die Dämpfung mu: (A_FF + μ D_FF) δ_F = −g_F über die freien
// Variablen F; Variablen an einer Schranke, deren Schritt nach außen zeigt, kommen in die aktive
// Menge, bis sich nichts mehr ändert. Danach Versuchspunkt clamp(x + δ), Schrittnorm und
// vorhergesagte Abnahme L(0) − L(Δx) = −(2 Δxᵀg + ΔxᵀAΔx) mit dem tatsächlichen Schritt Δx.
void solve_step(RunState& s, const LmBounds& b, int P, double mu) {
    const auto n = static_cast<std::size_t>(P);
    std::vector<char> fixed(n);
    for (std::size_t k = 0; k < n; ++k) {
        fixed[k] = bound_blocks(b, k, s.x[k], -s.g[k]) ? 1 : 0;
    }
    std::vector<double> delta(n, 0.0);
    std::vector<std::size_t> idx;
    std::vector<double> m;
    std::vector<double> rhs;
    s.solved = true;
    s.step_damping = mu;
    for (std::size_t round = 0; round <= n; ++round) {
        idx.clear();
        for (std::size_t k = 0; k < n; ++k) {
            if (!fixed[k]) {
                idx.push_back(k);
            }
        }
        const std::size_t f = idx.size();
        std::fill(delta.begin(), delta.end(), 0.0);
        if (f == 0) {
            break;
        }
        m.assign(f * f, 0.0);
        rhs.assign(f, 0.0);
        for (std::size_t j = 0; j < f; ++j) {
            for (std::size_t i = j; i < f; ++i) { // unteres Dreieck
                m[j * f + i] = s.a[idx[j] * n + idx[i]];
            }
            m[j * f + j] += mu * s.d[idx[j]];
            rhs[j] = -s.g[idx[j]];
        }
        if (!cholesky(m, static_cast<int>(f))) {
            s.solved = false;
            return;
        }
        cholesky_solve(m, static_cast<int>(f), rhs);
        for (std::size_t j = 0; j < f; ++j) {
            delta[idx[j]] = rhs[j];
        }
        bool changed = false;
        for (const std::size_t k : idx) {
            if (bound_blocks(b, k, s.x[k], delta[k])) {
                fixed[k] = 1;
                changed = true;
            }
        }
        if (!changed) {
            break;
        }
    }
    double norm = 0.0;
    std::vector<double> step(n);
    for (std::size_t k = 0; k < n; ++k) {
        const double t = std::min(std::max(s.x[k] + delta[k], b.lower[k]), b.upper[k]);
        s.trial[k] = t;
        step[k] = t - s.x[k];
        norm = std::max(norm, std::abs(step[k]) / scale_of(s.x[k], b.typical[k]));
    }
    s.step_norm = norm;
    double lin = 0.0;
    double quad = 0.0;
    for (std::size_t q = 0; q < n; ++q) {
        lin += step[q] * s.g[q];
        double aq = 0.0;
        for (std::size_t p = 0; p < n; ++p) {
            aq += s.a[q * n + p] * step[p];
        }
        quad += step[q] * aq;
    }
    s.predicted = -(2.0 * lin + quad);
}

void check_options(const LmOptions& o) {
    const auto ok = [](double v) { return std::isfinite(v) && v >= 0.0; };
    if (o.max_iterations < 0) {
        invalid("max_iterations < 0");
    }
    if (!ok(o.ftol) || !ok(o.xtol) || !ok(o.gtol) || !ok(o.diag_floor)) {
        invalid("Toleranzen und diag_floor müssen endlich und >= 0 sein");
    }
    if (!(o.initial_damping > 0.0) || !std::isfinite(o.initial_damping) ||
        !(o.max_damping > o.initial_damping)) {
        invalid("Dämpfung: verlangt 0 < initial_damping < max_damping");
    }
}

void check_problem(int P, const LmBounds& b, std::span<const double> x0) {
    if (P < 1) {
        invalid("keine Variablen");
    }
    const auto n = static_cast<std::size_t>(P);
    if (b.lower.size() != n || b.upper.size() != n || b.typical.size() != n) {
        invalid("Schranken und typische Größen brauchen P Werte");
    }
    for (std::size_t k = 0; k < n; ++k) {
        if (std::isnan(b.lower[k]) || std::isnan(b.upper[k]) || b.lower[k] > b.upper[k]) {
            invalid("ungültige Schranken der Variable " + std::to_string(k));
        }
        if (!(b.typical[k] > 0.0) || !std::isfinite(b.typical[k])) {
            invalid("typische Größe der Variable " + std::to_string(k) + " muss > 0 sein");
        }
    }
    if (x0.empty() || x0.size() % n != 0) {
        invalid("x0 hat " + std::to_string(x0.size()) +
                " Werte, erwartet ein positives Vielfaches von P = " + std::to_string(P));
    }
    for (std::size_t i = 0; i < x0.size(); ++i) {
        const std::size_t k = i % n;
        if (!std::isfinite(x0[i]) || x0[i] < b.lower[k] || x0[i] > b.upper[k]) {
            invalid("Startpunkt von Lauf " + std::to_string(i / n) +
                    " ist nicht endlich oder verletzt die Schranken (Variable " +
                    std::to_string(k) + ")");
        }
    }
}

bool all_finite(std::span<const double> v) {
    return std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); });
}

} // namespace

const char* to_string(LmStatus status) {
    switch (status) {
    case LmStatus::Running:
        return "running";
    case LmStatus::ConvergedMerit:
        return "converged_merit";
    case LmStatus::ConvergedStep:
        return "converged_step";
    case LmStatus::ConvergedGradient:
        return "converged_gradient";
    case LmStatus::MaxIterations:
        return "max_iterations";
    case LmStatus::DampingLimit:
        return "damping_limit";
    case LmStatus::NumericalError:
        return "numerical_error";
    }
    return "?";
}

bool lm_converged(LmStatus status) {
    return status == LmStatus::ConvergedMerit || status == LmStatus::ConvergedStep ||
           status == LmStatus::ConvergedGradient;
}

LmResult levenberg_marquardt(LmBackend& backend, int variables, const LmBounds& bounds,
                             std::span<const double> x0, const LmOptions& options) {
    check_options(options);
    check_problem(variables, bounds, x0);
    const auto P = static_cast<std::size_t>(variables);
    const int runs = static_cast<int>(x0.size() / P);
    std::vector<RunState> state(static_cast<std::size_t>(runs));
    std::vector<int> list;
    std::vector<double> xs;
    std::vector<double> buf_a;
    std::vector<double> buf_g;
    std::vector<double> buf_f;
    std::vector<std::int64_t> buf_failed;

    // Punkte der Läufe in `list` hintereinander (x oder Versuchspunkt).
    const auto gather = [&](bool trial) {
        xs.resize(list.size() * P);
        for (std::size_t j = 0; j < list.size(); ++j) {
            const RunState& s = state[static_cast<std::size_t>(list[j])];
            const std::vector<double>& src = trial ? s.trial : s.x;
            std::copy(src.begin(), src.end(), xs.begin() + static_cast<std::ptrdiff_t>(j * P));
        }
    };
    // Threads für die Schritte von n Läufen (Cholesky ~P³ pro Lauf und Runde)
    const auto step_threads = [&](std::size_t n) {
        const auto p3 = static_cast<std::int64_t>(P * P * P) + 1;
        return lm_host_threads(options.threads, static_cast<std::int64_t>(n) * p3,
                               kLmStepFlopsPerThread);
    };
    const auto evaluate = [&](bool trial) {
        gather(trial);
        buf_f.assign(list.size(), 0.0);
        buf_failed.assign(list.size(), 0);
        backend.merit(list, xs, buf_f, buf_failed);
    };

    // ---- Startpunkte ----
    for (int i = 0; i < runs; ++i) {
        RunState& s = state[static_cast<std::size_t>(i)];
        s.x.assign(x0.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(i) * P),
                   x0.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(i + 1) * P));
        s.trial = s.x;
        s.a.assign(P * P, 0.0);
        s.g.assign(P, 0.0);
        s.d.assign(P, 0.0);
        s.mu = options.initial_damping;
        list.push_back(i);
    }
    evaluate(false);
    for (std::size_t j = 0; j < list.size(); ++j) {
        RunState& s = state[static_cast<std::size_t>(list[j])];
        s.merit = buf_f[j];
        s.failed = buf_failed[j];
        s.history.push_back(s.merit);
        if (!std::isfinite(s.merit)) {
            s.status = LmStatus::NumericalError;
        } else if (s.merit == 0.0) {
            s.status = LmStatus::ConvergedMerit;
        }
    }

    LmResult result;
    std::vector<LmIterationInfo> infos;
    for (;;) {
        // ---- Jacobi-Matrix für Läufe nach einem angenommenen Schritt (und am Anfang) ----
        list.clear();
        for (int i = 0; i < runs; ++i) {
            const RunState& s = state[static_cast<std::size_t>(i)];
            if (s.status == LmStatus::Running && s.need_jacobian &&
                s.iterations < options.max_iterations) {
                list.push_back(i);
            }
        }
        if (!list.empty()) {
            gather(false);
            buf_a.assign(list.size() * P * P, 0.0);
            buf_g.assign(list.size() * P, 0.0);
            backend.normal_equations(list, xs, buf_a, buf_g);
            for (std::size_t j = 0; j < list.size(); ++j) {
                RunState& s = state[static_cast<std::size_t>(list[j])];
                const auto a = std::span<const double>(buf_a).subspan(j * P * P, P * P);
                const auto g = std::span<const double>(buf_g).subspan(j * P, P);
                std::copy(a.begin(), a.end(), s.a.begin());
                std::copy(g.begin(), g.end(), s.g.begin());
                s.need_jacobian = false;
                ++s.jacobians;
                if (!all_finite(a) || !all_finite(g)) {
                    s.status = LmStatus::NumericalError;
                    continue;
                }
                prepare_step(s, bounds, variables, options);
            }
            // Ungedämpftes Modell (Gauß-Newton-Schritt mit aktiver Menge, μ = kModelDamping):
            // vorhergesagte relative Abnahme und Schrittnorm für die Abbruchkriterien.
            detail::parallel_for_variants(
                0, static_cast<int>(list.size()), step_threads(list.size()), [&](int j) {
                    RunState& s =
                        state[static_cast<std::size_t>(list[static_cast<std::size_t>(j)])];
                    if (s.status == LmStatus::Running) {
                        solve_step(s, bounds, variables, kModelDamping);
                    }
                });
            for (const int i : list) {
                RunState& s = state[static_cast<std::size_t>(i)];
                if (s.status != LmStatus::Running) {
                    continue;
                }
                s.model_decrease = s.solved ? s.predicted / s.merit : kNaN;
                if (s.gradient_norm <= options.gtol) {
                    s.status = LmStatus::ConvergedGradient;
                } else if (s.solved && s.predicted >= 0.0 &&
                           s.predicted <= options.ftol * s.merit) {
                    s.status = LmStatus::ConvergedMerit;
                } else if (s.solved && s.step_norm <= options.xtol) {
                    s.status = LmStatus::ConvergedStep;
                }
            }
        }

        // ---- Versuchsschritte (unabhängig pro Lauf, auf Host-Threads) ----
        list.clear();
        for (int i = 0; i < runs; ++i) {
            RunState& s = state[static_cast<std::size_t>(i)];
            if (s.status != LmStatus::Running) {
                continue;
            }
            if (s.iterations >= options.max_iterations) {
                s.status = LmStatus::MaxIterations;
                continue;
            }
            list.push_back(i);
        }
        if (list.empty()) {
            break;
        }
        ++result.cycles;
        detail::parallel_for_variants(
            0, static_cast<int>(list.size()), step_threads(list.size()), [&](int j) {
                RunState& s = state[static_cast<std::size_t>(list[static_cast<std::size_t>(j)])];
                solve_step(s, bounds, variables, s.mu);
            });
        std::vector<int> cycle_runs = list; // Läufe dieses Zyklus (für den Rückruf)
        std::vector<int> eval;
        for (const int i : cycle_runs) {
            RunState& s = state[static_cast<std::size_t>(i)];
            // Kleiner gedämpfter Schritt zählt nur nach einer Ablehnung als Konvergenz (sonst
            // ist er klein, weil μ noch groß ist, und μ sinkt mit den nächsten Schritten).
            if (s.solved && s.rejected && s.step_norm <= options.xtol) {
                s.status = LmStatus::ConvergedStep;
            } else if (s.solved) {
                eval.push_back(i);
            }
        }

        // ---- Versuchspunkte bewerten ----
        list = eval;
        if (!list.empty()) {
            evaluate(true);
        }
        infos.clear();
        std::size_t e = 0;
        for (const int i : cycle_runs) {
            RunState& s = state[static_cast<std::size_t>(i)];
            if (s.status == LmStatus::ConvergedStep) {
                continue; // kein Zyklus gezählt
            }
            LmIterationInfo info;
            info.run = i;
            info.damping = s.step_damping;
            info.step_norm = s.step_norm;
            info.trial_merit = kNaN;
            info.gain_ratio = kNaN;
            ++s.iterations;
            bool accept = false;
            if (s.solved) {
                const double f_new = buf_f[e];
                const std::int64_t failed_new = buf_failed[e];
                ++e;
                info.evaluated = true;
                info.trial_merit = f_new;
                info.failed_rays = failed_new;
                const double rho = (s.merit - f_new) / s.predicted;
                info.gain_ratio = rho;
                accept = std::isfinite(f_new) && s.predicted > 0.0 && f_new < s.merit &&
                         !(options.reject_new_failures && failed_new > s.failed);
                if (accept) {
                    s.x = s.trial;
                    s.merit = f_new;
                    s.failed = failed_new;
                    const double t = 2.0 * rho - 1.0;
                    s.mu *= std::max(1.0 / 3.0, 1.0 - t * t * t);
                    s.nu = 2.0;
                    s.need_jacobian = true;
                    s.rejected = false;
                    ++s.accepted;
                    if (f_new == 0.0) {
                        s.status = LmStatus::ConvergedMerit;
                    }
                }
            }
            if (!accept) {
                s.mu *= s.nu;
                s.nu *= 2.0;
                s.rejected = true;
                if (s.mu > options.max_damping) {
                    s.status = LmStatus::DampingLimit;
                }
            }
            info.accepted = accept;
            s.history.push_back(s.merit);
            if (options.on_iteration) {
                info.iteration = s.iterations;
                info.merit = s.merit;
                info.status = s.status;
                infos.push_back(info);
            }
        }
        for (LmIterationInfo& info : infos) {
            info.x = state[static_cast<std::size_t>(info.run)].x;
            options.on_iteration(info);
        }
    }

    result.runs.resize(static_cast<std::size_t>(runs));
    for (int i = 0; i < runs; ++i) {
        RunState& s = state[static_cast<std::size_t>(i)];
        LmRunResult& r = result.runs[static_cast<std::size_t>(i)];
        r.x = std::move(s.x);
        r.merit = s.merit;
        r.failed_rays = s.failed;
        r.status = s.status;
        r.iterations = s.iterations;
        r.accepted = s.accepted;
        r.jacobians = s.jacobians;
        r.damping = s.mu;
        r.gradient_norm = s.gradient_norm;
        r.model_decrease = s.model_decrease;
        r.history = std::move(s.history);
    }
    return result;
}

// ---- Optische Probleme -------------------------------------------------------------------

int lm_host_threads(int requested, std::int64_t work, std::int64_t min_work_per_thread) {
    if (requested > 0) {
        return requested;
    }
    const std::int64_t hw = std::max(1u, std::thread::hardware_concurrency());
    const std::int64_t by_work = min_work_per_thread > 0 ? work / min_work_per_thread : hw;
    return static_cast<int>(std::clamp<std::int64_t>(by_work, 1, hw));
}

void check_lm_base(const SystemBatchHost& base, int variables, std::size_t x0_size) {
    if (variables < 1) {
        invalid("keine Variablen");
    }
    const auto P = static_cast<std::size_t>(variables);
    if (x0_size == 0 || x0_size % P != 0) {
        invalid("x0 hat " + std::to_string(x0_size) +
                " Werte, erwartet ein positives Vielfaches von P = " + std::to_string(variables));
    }
    const std::size_t runs = x0_size / P;
    if (base.variants() != 1 && static_cast<std::size_t>(base.variants()) != runs) {
        invalid("die Vorlage hat " + std::to_string(base.variants()) +
                " Varianten, erwartet 1 oder " + std::to_string(runs) + " (eine pro Lauf)");
    }
}

SystemBatchHost lm_template_batch(const SystemBatchHost& base, std::span<const int> runs) {
    if (base.variants() == 1) {
        return base;
    }
    SystemBatchHost out = base; // gemeinsame Felder
    out.resize(static_cast<int>(runs.size()));
    for (std::size_t j = 0; j < runs.size(); ++j) {
        const int i = runs[j];
        if (i < 0 || i >= base.variants()) {
            throw std::out_of_range("lm_template_batch: Lauf " + std::to_string(i) +
                                    " hat keine Vorlage (" + std::to_string(base.variants()) +
                                    " Varianten)");
        }
        out.copy_variant_from(base, i, static_cast<int>(j));
    }
    return out;
}

SystemBatchHost lm_point_batch(const VariableSet& set, const SystemBatchHost& base,
                               std::span<const int> runs, std::span<const double> x) {
    const auto P = static_cast<std::size_t>(set.size());
    if (P == 0 || x.size() != runs.size() * P) {
        invalid("lm_point_batch: x passt nicht zu den Läufen");
    }
    if (base.variants() < 1) {
        invalid("lm_point_batch: Vorlage ohne Varianten");
    }
    SystemBatchHost out = base;
    out.resize(static_cast<int>(runs.size()));
    for (std::size_t j = 0; j < runs.size(); ++j) {
        const int tmpl = base.variants() == 1 ? 0 : runs[j];
        if (tmpl < 0 || tmpl >= base.variants()) {
            throw std::out_of_range("lm_point_batch: Lauf " + std::to_string(runs[j]) +
                                    " hat keine Vorlage");
        }
        out.copy_variant_from(base, tmpl, static_cast<int>(j));
        set.apply(out, static_cast<int>(j), x.subspan(j * P, P));
    }
    return out;
}

std::vector<double> extra_merit_values(const ExtraResiduals& extra, const SystemBatchHost& batch,
                                       int threads) {
    if (extra.count == 0) {
        return {};
    }
    const auto E = static_cast<std::size_t>(extra.count);
    std::vector<double> values(static_cast<std::size_t>(batch.variants()) * E);
    evaluate_extra_residuals(extra, batch, values, 0, -1, threads);
    std::vector<double> out(static_cast<std::size_t>(batch.variants()), 0.0);
    for (std::size_t v = 0; v < out.size(); ++v) {
        double s = 0.0;
        for (std::size_t i = 0; i < E; ++i) {
            const double e = values[v * E + i];
            s += e * e;
        }
        out[v] = s;
    }
    return out;
}

CpuLmBackend::CpuLmBackend(const VariableSet& set, const SystemBatchHost& base,
                           const LmOptions& options)
    : set_(set), base_(base), options_(options),
      layout_(base.topology(), options.jacobian.quadrature) {
    if (options.jacobian.extra.count < 0 ||
        (options.jacobian.extra.count > 0 && !options.jacobian.extra.evaluate)) {
        invalid("ungültige Zusatzresiduen");
    }
}

void CpuLmBackend::normal_equations(std::span<const int> runs, std::span<const double> x,
                                    std::span<double> a, std::span<double> g) {
    const SystemBatchHost tmpl = lm_template_batch(base_, runs);
    const FdJacobianBatch fd = make_fd_jacobian_batch(set_, tmpl, x, options_.jacobian.scheme);
    const int V = fd.batch.variants();
    const std::int64_t rv = layout_.rays_per_variant();
    ResidualResult rays;
    rays.variants = V;
    rays.rays_per_variant = rv;
    rays.stride = rv;
    rays.residuals.resize(static_cast<std::size_t>(V) * static_cast<std::size_t>(rv));
    rays.status.resize(rays.residuals.size());
    const std::int64_t ray_work =
        static_cast<std::int64_t>(V) * rv * static_cast<std::int64_t>(fd.batch.surfaces());
    detail::parallel_for_variants(
        0, V, lm_host_threads(options_.threads, ray_work, kLmRaySurfacesPerThread), [&](int v) {
            const ResidualResult one = residuals_cpu(fd.batch, v, layout_);
            const std::size_t o = rays.ray_index(v, 0);
            std::copy(one.residuals.begin(), one.residuals.end(),
                      rays.residuals.begin() + static_cast<std::ptrdiff_t>(o));
            std::copy(one.status.begin(), one.status.end(),
                      rays.status.begin() + static_cast<std::ptrdiff_t>(o));
        });
    const ExtraResiduals& extra = options_.jacobian.extra;
    std::vector<double> ev(static_cast<std::size_t>(V) * static_cast<std::size_t>(extra.count));
    evaluate_extra_residuals(
        extra, fd.batch, ev, 0, -1,
        lm_host_threads(options_.jacobian.extra_threads, V, kLmExtraPerThread));
    FdJacobian jac;
    fd_jacobian_from_residuals(fd, rays, ev, extra.count, jac);
    const auto P = static_cast<std::size_t>(set_.size());
    const auto R = static_cast<std::size_t>(jac.shape.rows());
    const std::int64_t gram_work = static_cast<std::int64_t>(runs.size()) *
                                   static_cast<std::int64_t>(P * P) * static_cast<std::int64_t>(R);
    detail::parallel_for_variants(
        0, static_cast<int>(runs.size()),
        lm_host_threads(options_.threads, gram_work, kLmStepFlopsPerThread), [&](int j) {
            const auto jj = static_cast<std::size_t>(j);
            normal_equations_host(std::span<const double>(jac.jacobian).subspan(jj * P * R, P * R),
                                  jac.residual(j), set_.size(), a.subspan(jj * P * P, P * P),
                                  g.subspan(jj * P, P));
        });
}

void CpuLmBackend::merit(std::span<const int> runs, std::span<const double> x,
                         std::span<double> merit, std::span<std::int64_t> failed) {
    const SystemBatchHost batch = lm_point_batch(set_, base_, runs, x);
    const std::vector<double> extra = extra_merit_values(
        options_.jacobian.extra, batch,
        lm_host_threads(options_.jacobian.extra_threads, batch.variants(), kLmExtraPerThread));
    const std::int64_t ray_work = static_cast<std::int64_t>(batch.variants()) *
                                  layout_.rays_per_variant() *
                                  static_cast<std::int64_t>(batch.surfaces());
    detail::parallel_for_variants(
        0, batch.variants(), lm_host_threads(options_.threads, ray_work, kLmRaySurfacesPerThread),
        [&](int v) {
            const MeritResult m = merit_cpu(batch, v, layout_, false);
            const auto vv = static_cast<std::size_t>(v);
            merit[vv] = extra.empty() ? m.merit[0] : m.merit[0] + extra[vv];
            failed[vv] = m.failed[0];
        });
}

LmResult optimize_lm_cpu(const VariableSet& set, const SystemBatchHost& base,
                         std::span<const double> x0, const LmOptions& options) {
    check_lm_base(base, set.size(), x0.size());
    CpuLmBackend backend(set, base, options);
    const LmBounds bounds{set.lower(), set.upper(), set.typical()};
    return levenberg_marquardt(backend, set.size(), bounds, x0, options);
}

} // namespace mko
