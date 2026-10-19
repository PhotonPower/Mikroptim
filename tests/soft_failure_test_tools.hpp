#pragma once

// Werkzeuge für die Tests des weichen Strahlversagens (M4-T2, CPU und GPU): Merit entlang einer
// fein abgetasteten Geraden durch eine Versagensgrenze und Auswertung der Stetigkeit; die
// beiden Testsysteme der Abnahme (Linien durch Totalreflexion und Verfehlen einer Fläche).

#include "fd_test_systems.hpp"
#include "mikroptim/batch/batch_merit.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "residual_test_systems.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <vector>

namespace mko::test {

// Merit und versagte Strahlen aller Varianten eines Batches (CPU: merit_cpu, GPU: TraceEngine im
// Modus MERIT).
using MeritEvaluator =
    std::function<MeritResult(const SystemBatchHost& batch, const ResidualSettings& settings)>;

inline MeritResult cpu_merit_evaluator(const SystemBatchHost& batch,
                                       const ResidualSettings& settings) {
    return merit_cpu(batch, ResidualLayout(batch.topology(), settings), false);
}

// Punkt t der Geraden in Variante v schreiben (Variante v ist vorher eine Kopie der Vorlage).
using PointSetter = std::function<void(SystemBatchHost&, int, double)>;

// Merit und versagte Strahlen an n Punkten t_i ∈ [t0, t1] der Geraden.
struct LineSample {
    std::vector<double> t;
    std::vector<double> merit;
    std::vector<std::int64_t> failed;
};

inline LineSample sample_line(const SystemBatchHost& base, const PointSetter& set_point,
                              const ResidualSettings& settings, double t0, double t1, int n,
                              const MeritEvaluator& evaluate) {
    SystemBatchHost b = base;
    b.resize(n);
    LineSample s;
    for (int i = 0; i < n; ++i) {
        const double t = t0 + (t1 - t0) * static_cast<double>(i) / static_cast<double>(n - 1);
        b.copy_variant(0, i);
        set_point(b, i, t);
        s.t.push_back(t);
    }
    const MeritResult m = evaluate(b, settings);
    s.merit = m.merit;
    s.failed = m.failed;
    return s;
}

// Sprung der Merit über den Versagensübergang, nach k-facher Verfeinerung des Intervalls, in
// dem sich die Zahl versagter Strahlen zuerst ändert (n Punkte pro Stufe): jumps[k] =
// |F(t_b+1) − F(t_b)| auf Stufe k, width[k] = Intervallbreite. max_ratio = größtes Verhältnis
// eines Nachbarsprungs zum größeren seiner beiden Nachbarn auf der ersten Stufe (ein Sprung ist
// ein einzelner Ausreißer, ein Knick oder eine Wurzelsingularität nicht).
struct ContinuityReport {
    std::vector<double> jumps;
    std::vector<double> width;
    double max_ratio = 0.0;
    double merit_scale = 0.0;
    bool crossed = false;
};

inline ContinuityReport continuity_across_failure(const SystemBatchHost& base,
                                                  const PointSetter& set_point,
                                                  const ResidualSettings& settings, double t0,
                                                  double t1, int n, int levels,
                                                  const MeritEvaluator& evaluate) {
    ContinuityReport rep;
    double a = t0;
    double b = t1;
    for (int level = 0; level <= levels; ++level) {
        const LineSample s = sample_line(base, set_point, settings, a, b, n, evaluate);
        int boundary = -1;
        for (int i = 0; i + 1 < n; ++i) {
            if (s.failed[static_cast<std::size_t>(i)] !=
                s.failed[static_cast<std::size_t>(i + 1)]) {
                boundary = i;
                break;
            }
        }
        if (boundary < 0) {
            return rep; // kein Übergang (mehr) auf der Geraden
        }
        rep.crossed = true;
        const auto bi = static_cast<std::size_t>(boundary);
        rep.jumps.push_back(std::abs(s.merit[bi + 1] - s.merit[bi]));
        rep.width.push_back(s.t[bi + 1] - s.t[bi]);
        if (level == 0) {
            for (const double f : s.merit) {
                rep.merit_scale = std::max(rep.merit_scale, std::abs(f));
            }
            const double floor = 1e-13 * rep.merit_scale;
            for (std::size_t i = 1; i + 2 < s.merit.size(); ++i) {
                const double d = std::abs(s.merit[i + 1] - s.merit[i]);
                const double left = std::abs(s.merit[i] - s.merit[i - 1]);
                const double right = std::abs(s.merit[i + 2] - s.merit[i + 1]);
                rep.max_ratio = std::max(rep.max_ratio, d / (std::max(left, right) + floor));
            }
        }
        a = s.t[bi];
        b = s.t[bi + 1];
    }
    return rep;
}

inline void print_report(const char* name, const ContinuityReport& r) {
    std::printf("[ Info     ] %-34s F ~ %.3e, Nachbarverhältnis max %.2f, Sprung am Übergang:",
                name, r.merit_scale, r.max_ratio);
    for (std::size_t k = 0; k < r.jumps.size(); ++k) {
        std::printf(" %.3e (dt %.1e)", r.jumps[k], r.width[k]);
    }
    std::printf("\n");
}

// Gerade durch die Totalreflexion: Plankonvexlinse tir_problem (M3-T2), der äußerste Ring tritt
// bei t = 0 knapp unter dem Grenzwinkel aus; c₂(t) = c₂·(1 + t), Grenze bei t ≈ 3·10⁻⁶.
struct FailureLine {
    SystemBatchHost base;
    PointSetter set;
    double t0 = 0.0;
    double t1 = 0.0;
};

inline FailureLine total_reflection_line() {
    const FdProblem p = tir_problem();
    const double c2 = p.base.curvature(0, 2);
    return FailureLine{
        p.base, [c2](SystemBatchHost& b, int v, double t) { b.curvature(v, 2) = c2 * (1.0 + t); },
        -2e-5, 4e-5};
}

// Gerade durch das Verfehlen der Vorderfläche: Meniskus aus weiches_versagen_start.json mit
// c₂ = +0,06 (streifend eintretende Strahlen passieren die Rückfläche), c₁(t) = c*·(1 + t) mit
// c* = 1/y des äußersten Strahls des Achsfeldes.
inline FailureLine missed_surface_line() {
    SystemBatchHost base =
        make_system_batch(load_residual_system("weiches_versagen_start.json"), 1);
    base.curvature(0, 2) = 0.06;
    const ResidualLayout layout(base.topology());
    const PupilQuadrature<double>& q = layout.quadrature(0);
    const double c_star = 1.0 / (base.pupil_radius(0) * std::sqrt(q.u[q.rings - 1]));
    return FailureLine{
        base,
        [c_star](SystemBatchHost& b, int v, double t) { b.curvature(v, 1) = c_star * (1.0 + t); },
        -2e-5, 4e-5};
}

// Startpunkte um weiches_versagen_start.json: c₁ (Vorderfläche verfehlt für c₁ > 1/y_max ≈
// 0,1007), c₂ (Totalreflexion an der Rückfläche), w; 108 Punkte, 102 davon mit Versagen.
inline std::vector<double> soft_failure_start_grid() {
    std::vector<double> starts;
    for (const double c1 : {0.095, 0.104, 0.11, 0.1136, 0.12, 0.13}) {
        for (const double c2 : {-0.04, -0.03, -0.02, -0.01, 0.0, 0.02}) {
            for (const double w : {5.0, 13.0, 25.0}) {
                starts.insert(starts.end(), {c1, c2, w});
            }
        }
    }
    return starts;
}

} // namespace mko::test
