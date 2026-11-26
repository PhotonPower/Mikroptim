#pragma once

// Synthetisches Modell des Homotopie-Controllers (M6-T3) und der Strahlsuche (M6-T4) für Tests
// der Steuerlogik ohne Optik. Zustand in der Beschreibung: Zahl der Flächen = Elemente,
// surfaces[0].thickness = Apertur der letzten Optimierung, surfaces.back().curvature = Zweig,
// surfaces.back().semi_diameter = Variante des Einfügens (Strahlsuche). Rückbau (M6-T9):
// Elemente mit group "useless" tragen nichts zur Kapazität bei, ebenso ein Element in der
// Rückbau-Homotopie (group "collapsing"); Kandidaten sind die Elemente 1 … n − 1, das letzte
// zuerst.

#include "mikroptim/optim/variant_generator.hpp"
#include "mikroptim/synth/homotopy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
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
    double start_merit = -1.0;  // ≥ 0: F_B am Startpunkt (vor dem ersten Schritt)
    bool insert_throws = false; // insert wirft std::invalid_argument (Einfügen nicht möglich)
    int branch_throws = -1; // >= 0: solve wirft std::runtime_error für Zweig b (M6-T11), 2: beide

    // Strahlsuche
    double branch_spread = 1.0;    // Krümmung (Marke) von Zweig b: b · branch_spread
    double branch0_capacity = 0.0; // Zweig 0 (Marke 0, mehr als ein Element) erreicht weniger
    int alternatives = 1;          // Varianten, die insertion_options höchstens liefert
    double option_bonus = 0.0;     // Variante k: F_B × (1 − option_bonus · k)
    int bad_options = 0;           // Varianten k < bad_options: F_B × 4 nach dem Einfügen (M6-T8)
    double bad_factor = 4.0;       // Faktor der schlechten Varianten (M6-T11b)
    // Systeme mit weniger Elementen: ein versagter Strahl, F_B = failed_merit · F_max (M6-T11b)
    int failed_below_elements = 0;
    double failed_merit = 0.2;
    double perturbation = 0.0;     // gestörte Starts: Marke ± perturbation
    double prefilter_max = std::numeric_limits<double>::infinity(); // Vorfilter: Marke ≤ Wert
    bool group_by_elements = false; // solve_all bündelt Systeme gleicher Flächenzahl
    bool predict = false;           // Prädiktor (M11-T3): predict liefert den aktuellen Zustand
    // Augmented Lagrangian (M11-T2): Rückgaben von update_shifts der Reihe nach (danach 0)
    std::vector<double> shift_changes{};
    // Rückbau relativ (M6-T15): unter der Kapazität F_B = F_max/2 · (1 − element_bonus · (n − 1))
    // mit n Elementen außerhalb der Rückbau-Homotopie (mehr Elemente, kleineres F_B)
    double element_bonus = 0.0;
    // Aperturgrenze (M6-T15): rms_main = 0,05 bis zur Apertur rms_capacity0 + rms_capacity_step ·
    // (n − 1), darüber 0,2; die Diagnose meldet dort Bedarf (wie über der Kapazität)
    double rms_capacity0 = std::numeric_limits<double>::infinity();
    double rms_capacity_step = 0.0;
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
    std::vector<int> batches;               // Größe jedes gemeinsamen Aufrufs (solve_all)
    std::vector<std::uint64_t> seeds;       // Startwerte der gestörten Starts
    std::vector<double> predict_ratios;     // Verhältnisse der Aufrufe von predict
    std::vector<double> shift_points;       // Apertur jedes Aufrufs von update_shifts (M11-T2)
    std::vector<double> collapse_fractions; // Anteile der Aufrufe von collapse (M6-T9)

    static bool contributes(const Surface& s) {
        return !s.group || (*s.group != "useless" && *s.group != "collapsing");
    }

    std::vector<HomotopyRemoval> removal_candidates(const SystemDescription& system,
                                                    const HomotopyPoint&) override {
        std::vector<HomotopyRemoval> out;
        for (int i = static_cast<int>(system.surfaces.size()) - 1; i >= 1; --i) {
            HomotopyRemoval r;
            r.first_surface = i;
            r.score = -i;
            r.text = "Mock-Element " + std::to_string(i);
            out.push_back(r);
        }
        return out;
    }

    SystemDescription collapse(const SystemDescription&, const SystemDescription& current,
                               const HomotopyRemoval& r, double fraction,
                               std::vector<std::string>&) override {
        collapse_fractions.push_back(fraction);
        SystemDescription d = current;
        Surface& s = d.surfaces.at(static_cast<std::size_t>(r.first_surface));
        if (contributes(s)) {
            s.group = "collapsing";
        }
        return d;
    }

    SystemDescription remove(const SystemDescription&, const SystemDescription& collapsed,
                             const HomotopyRemoval& r, std::vector<std::string>&) override {
        SystemDescription d = collapsed;
        d.surfaces.erase(d.surfaces.begin() + r.first_surface);
        return d;
    }

    double update_shifts(SystemDescription&, const HomotopyPoint& p, double) override {
        const std::size_t k = shift_points.size();
        shift_points.push_back(p.aperture);
        return k < spec_.shift_changes.size() ? spec_.shift_changes[k] : 0.0;
    }

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
        int useful = 0;
        for (const Surface& s : system.surfaces) {
            useful += contributes(s) ? 1 : 0;
        }
        double c = spec_.capacity0 + spec_.capacity_step * (useful - 1);
        if (e > 1 && system.surfaces.back().curvature == 0.0) {
            c -= spec_.branch0_capacity;
        }
        return c;
    }

    double rms_capacity(const SystemDescription& system) const {
        return spec_.rms_capacity0 + spec_.rms_capacity_step * (elements(system) - 1);
    }

    HomotopySolve solve(SystemDescription& system, const HomotopyPoint& p) override {
        const int e = elements(system);
        const double from = system.surfaces.front().thickness;
        calls.push_back({p, from, e});
        const double jump = std::abs(p.aperture - from);
        if (spec_.branch_throws >= 0 && e > 1 &&
            (spec_.branch_throws == 2 ||
             system.surfaces.back().curvature == spec_.branch_throws * spec_.branch_spread)) {
            throw std::runtime_error("Brechzahl <= 0");
        }
        HomotopySolve s;
        s.status = LmStatus::ConvergedMerit;
        s.iterations = jump <= spec_.fast_delta ? 10 : 100;
        double fb =
            p.aperture <= capacity(system) + 1e-12 ? 0.5 * spec_.threshold : 2.0 * spec_.threshold;
        if (spec_.element_bonus != 0.0 && p.aperture <= capacity(system) + 1e-12) {
            const auto n = std::count_if(system.surfaces.begin(), system.surfaces.end(),
                                         [](const Surface& x) { return x.group != "collapsing"; });
            fb *= 1.0 - spec_.element_bonus * static_cast<double>(n - 1);
        }
        if (jump > spec_.max_jump + 1e-12) {
            fb = 2.0 * spec_.threshold;
        }
        if (spec_.start_merit >= 0.0 && calls.size() == 1) {
            fb = spec_.start_merit;
        }
        // Zweig 1 ist etwas besser
        fb *= 1.0 - 0.1 * system.surfaces.back().curvature / spec_.branch_spread;
        fb *= 1.0 - spec_.option_bonus * system.surfaces.back().semi_diameter;
        if (e > 1 && system.surfaces.back().semi_diameter < spec_.bad_options) {
            fb *= spec_.bad_factor;
        }
        if (e < spec_.failed_below_elements) {
            fb = spec_.failed_merit * spec_.threshold;
            s.failed_rays = 1;
        }
        s.wavefront = fb;
        s.merit = fb;
        s.start_merit = fb;
        s.rms = std::sqrt(fb);
        s.rms_main = p.aperture <= rms_capacity(system) + 1e-12 ? 0.05 : 0.2;
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
        if (spec_.need && (p.aperture > capacity(system) + 1e-12 ||
                           p.aperture > rms_capacity(system) + 1e-12)) {
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
        if (spec_.insert_throws) {
            throw std::invalid_argument("Linse zu dünn");
        }
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
