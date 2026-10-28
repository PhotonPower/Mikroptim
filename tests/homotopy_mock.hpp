#pragma once

// Synthetisches Modell des Homotopie-Controllers (M6-T3) und der Strahlsuche (M6-T4) für Tests
// der Steuerlogik ohne Optik. Zustand in der Beschreibung: Zahl der Flächen = Elemente,
// surfaces[0].thickness = Apertur der letzten Optimierung, surfaces.back().curvature = Zweig,
// surfaces.back().semi_diameter = Variante des Einfügens (Strahlsuche).

#include "mikroptim/optim/variant_generator.hpp"
#include "mikroptim/synth/homotopy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <vector>

namespace mko::test {

struct MockSpec {
    double threshold = 1.0;     // F_max, wie HomotopyOptions::merit_threshold
    double capacity0 = 10.0;    // Apertur, die ein Element ohne Bedarf erreicht
    double capacity_step = 0.0; // pro weiterem Element
    double fast_delta = 0.05;   // Schritt bis hierher konvergiert schnell
    double max_jump = 1e9;      // größere Schritte werden abgelehnt (F_B = 2·F_max)
    bool need = true;           // Diagnose meldet Bedarf, wenn der Punkt über der Kapazität liegt
    bool one_sided = false;
    double goal_ghost_weight = 1.0; // Ziel erst ab diesem μ-Faktor
    bool goal_never = false;
    double start_merit = -1.0; // ≥ 0: F_B am Startpunkt (vor dem ersten Schritt)

    // Strahlsuche
    double branch_spread = 1.0;    // Krümmung (Marke) von Zweig b: b · branch_spread
    double branch0_capacity = 0.0; // Zweig 0 (Marke 0, mehr als ein Element) erreicht weniger
    int alternatives = 1;          // Varianten, die insertion_options höchstens liefert
    double option_bonus = 0.0;     // Variante k: F_B × (1 − option_bonus · k)
    double perturbation = 0.0;     // gestörte Starts: Marke ± perturbation
    double prefilter_max = std::numeric_limits<double>::infinity(); // Vorfilter: Marke ≤ Wert
    bool group_by_elements = false; // solve_all bündelt Systeme gleicher Flächenzahl
    bool predict = false;           // Prädiktor (M11-T3): predict liefert den aktuellen Zustand
};

struct MockCall {
    HomotopyPoint point;
    double from = 0.0; // Apertur der vorigen Optimierung (aus dem Zustand)
    int elements = 0;
};

class MockModel final : public HomotopyModel {
public:
    explicit MockModel(MockSpec spec) : spec_(spec) {}

    std::vector<MockCall> calls;
    int finishes = 0;
    std::vector<int> batches;           // Größe jedes gemeinsamen Aufrufs (solve_all)
    std::vector<std::uint64_t> seeds;   // Startwerte der gestörten Starts
    std::vector<double> predict_ratios; // Verhältnisse der Aufrufe von predict

    std::optional<SystemDescription> predict(const SystemDescription&,
                                             const SystemDescription& current, const HomotopyPoint&,
                                             double ratio) override {
        if (!spec_.predict) {
            return std::nullopt;
        }
        predict_ratios.push_back(ratio);
        return current;
    }

    static SystemDescription start(double aperture) {
        SystemDescription d;
        Surface s;
        s.thickness = aperture;
        d.surfaces.push_back(s);
        return d;
    }

    double capacity(const SystemDescription& system) const {
        const int e = elements(system);
        double c = spec_.capacity0 + spec_.capacity_step * (e - 1);
        if (e > 1 && system.surfaces.back().curvature == 0.0) {
            c -= spec_.branch0_capacity;
        }
        return c;
    }

    HomotopySolve solve(SystemDescription& system, const HomotopyPoint& p) override {
        const int e = elements(system);
        const double from = system.surfaces.front().thickness;
        calls.push_back({p, from, e});
        const double jump = std::abs(p.aperture - from);
        HomotopySolve s;
        s.status = LmStatus::ConvergedMerit;
        s.iterations = jump <= spec_.fast_delta ? 10 : 100;
        double fb =
            p.aperture <= capacity(system) + 1e-12 ? 0.5 * spec_.threshold : 2.0 * spec_.threshold;
        if (jump > spec_.max_jump + 1e-12) {
            fb = 2.0 * spec_.threshold;
        }
        if (spec_.start_merit >= 0.0 && calls.size() == 1) {
            fb = spec_.start_merit;
        }
        // Zweig 1 ist etwas besser
        fb *= 1.0 - 0.1 * system.surfaces.back().curvature / spec_.branch_spread;
        fb *= 1.0 - spec_.option_bonus * system.surfaces.back().semi_diameter;
        s.wavefront = fb;
        s.merit = fb;
        s.start_merit = fb;
        s.rms = std::sqrt(fb);
        system.surfaces.front().thickness = p.aperture;
        return s;
    }

    std::vector<HomotopySolve> solve_all(std::span<SystemDescription* const> systems,
                                         const HomotopyPoint& point,
                                         std::vector<int>* groups) override {
        if (!spec_.group_by_elements) {
            for (std::size_t i = 0; i < systems.size(); ++i) {
                batches.push_back(1);
            }
            return HomotopyModel::solve_all(systems, point, groups);
        }
        // Gruppen nach Flächenzahl in der Reihenfolge des ersten Auftretens
        std::vector<std::size_t> keys;
        std::vector<int> g(systems.size());
        for (std::size_t i = 0; i < systems.size(); ++i) {
            const std::size_t k = systems[i]->surfaces.size();
            const auto it = std::find(keys.begin(), keys.end(), k);
            g[i] = static_cast<int>(it - keys.begin());
            if (it == keys.end()) {
                keys.push_back(k);
            }
        }
        std::vector<HomotopySolve> out(systems.size());
        for (std::size_t k = 0; k < keys.size(); ++k) {
            int n = 0;
            for (std::size_t i = 0; i < systems.size(); ++i) {
                if (g[i] == static_cast<int>(k)) {
                    out[i] = solve(*systems[i], point);
                    ++n;
                }
            }
            batches.push_back(n);
        }
        if (groups != nullptr) {
            *groups = g;
        }
        return out;
    }

    HomotopyNeed diagnose(const SystemDescription& system, const HomotopyPoint& p) override {
        HomotopyNeed n;
        if (spec_.need && p.aperture > capacity(system) + 1e-12) {
            n.recommendation = GhostOperator::A;
            n.text = "Bedarf";
        } else if (spec_.need && spec_.goal_never) {
            n.recommendation = GhostOperator::A;
            n.text = "Bedarf am Ziel";
        } else {
            n.text = "kein Bedarf";
        }
        return n;
    }

    std::vector<HomotopyNeed> insertion_options(const SystemDescription&, const HomotopyPoint&,
                                                const HomotopyNeed& need, int max) override {
        std::vector<HomotopyNeed> out;
        for (int k = 0; k < std::min(max, spec_.alternatives); ++k) {
            HomotopyNeed n = need;
            n.diagnosis.recommended_position = k;
            n.text = "Lage " + std::to_string(k);
            out.push_back(n);
        }
        return out;
    }

    HomotopyInsertion insert(const SystemDescription& system, const HomotopyPoint&,
                             const HomotopyNeed& need) override {
        HomotopyInsertion ins;
        ins.op = need.recommendation;
        const int option = std::max(0, need.diagnosis.recommended_position);
        for (int b = 0; b < 2; ++b) {
            SystemDescription d = system;
            Surface s;
            s.curvature = b * spec_.branch_spread;
            s.semi_diameter = option;
            d.surfaces.push_back(s);
            ins.branches.push_back(d);
        }
        ins.stages = {0.25, 0.5};
        ins.first_surface = static_cast<int>(system.surfaces.size());
        ins.new_surfaces = 1;
        ins.text = "Mock A";
        return ins;
    }

    void set_stage(SystemDescription& system, const HomotopyInsertion&, double v) override {
        system.surfaces.back().thickness = v;
    }

    void finish_insertion(SystemDescription&, const HomotopyInsertion&) override { ++finishes; }

    HomotopyGoal goal(const SystemDescription& system, const HomotopyPoint& p,
                      double target) override {
        HomotopyGoal g;
        g.target = target;
        g.max_rms = p.aperture <= capacity(system) + 1e-12 &&
                            p.ghost_weight >= spec_.goal_ghost_weight && !spec_.goal_never
                        ? 0.5 * target
                        : 2.0 * target;
        g.reached = g.max_rms <= target;
        g.tuples.push_back({0, 0, 0.0, 546.07, g.max_rms, 0});
        return g;
    }

    int elements(const SystemDescription& system) const override {
        return static_cast<int>(system.surfaces.size());
    }

    bool one_sided_constraints(const SystemDescription&) const override { return spec_.one_sided; }

    double distance(const SystemDescription& a, const SystemDescription& b,
                    const HomotopyPoint&) override {
        if (a.surfaces.size() != b.surfaces.size()) {
            return std::numeric_limits<double>::infinity();
        }
        double d = 0.0;
        for (std::size_t i = 0; i < a.surfaces.size(); ++i) {
            d = std::max({d, std::abs(a.surfaces[i].curvature - b.surfaces[i].curvature),
                          std::abs(a.surfaces[i].semi_diameter - b.surfaces[i].semi_diameter),
                          std::abs(a.surfaces[i].thickness - b.surfaces[i].thickness)});
        }
        return d;
    }

    std::vector<SystemDescription> perturbed_starts(const SystemDescription& start,
                                                    const HomotopyInsertion&, const HomotopyPoint&,
                                                    int count, std::uint64_t seed) override {
        seeds.push_back(seed);
        std::vector<SystemDescription> out;
        for (int j = 1; j <= count; ++j) {
            std::mt19937_64 rng(variant_seed(seed, static_cast<std::uint64_t>(j)));
            SystemDescription d = start;
            d.surfaces.back().curvature += spec_.perturbation * (2.0 * unit_uniform(rng()) - 1.0);
            out.push_back(d);
        }
        return out;
    }

    std::vector<bool> prefilter(std::span<const SystemDescription* const> starts,
                                const HomotopyPoint&) override {
        std::vector<bool> keep;
        for (const SystemDescription* s : starts) {
            keep.push_back(s->surfaces.back().curvature <= spec_.prefilter_max);
        }
        return keep;
    }

private:
    MockSpec spec_;
};

inline HomotopyOptions mock_options(double target) {
    HomotopyOptions o;
    o.target_aperture = target;
    o.merit_threshold = 1.0;
    o.fast_iterations = 50;
    o.aperture_step = {0.02, 0.004, 0.1, 2.0, 0.5};
    o.penalty_stages = 3;
    o.smoothing_stages = 1;
    o.ghost_weight_stages = 2;
    return o;
}

inline HomotopyPoint mock_point(double aperture) {
    HomotopyPoint p;
    p.aperture = aperture;
    return p;
}

} // namespace mko::test
