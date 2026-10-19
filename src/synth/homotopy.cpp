// Homotopie-Controller (M6-T3): Schrittweitensteuerung, Fortsetzungen, Einfügen, Protokoll.
// Festlegungen: docs/optimierung.md, Abschnitt 7.

#include "mikroptim/synth/homotopy.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mko {

const char* to_string(HomotopyStepKind kind) {
    switch (kind) {
    case HomotopyStepKind::Start:
        return "start";
    case HomotopyStepKind::Penalty:
        return "penalty";
    case HomotopyStepKind::Aperture:
        return "aperture";
    case HomotopyStepKind::Field:
        return "field";
    case HomotopyStepKind::Spectrum:
        return "spectrum";
    case HomotopyStepKind::Insertion:
        return "insertion";
    case HomotopyStepKind::InsertionStage:
        return "insertion_stage";
    case HomotopyStepKind::Smoothing:
        return "smoothing";
    case HomotopyStepKind::GhostWeight:
        return "ghost_weight";
    case HomotopyStepKind::Goal:
        return "goal";
    }
    return "?";
}

const char* to_string(HomotopyAction action) {
    switch (action) {
    case HomotopyAction::None:
        return "none";
    case HomotopyAction::Grow:
        return "grow";
    case HomotopyAction::Keep:
        return "keep";
    case HomotopyAction::Shrink:
        return "shrink";
    case HomotopyAction::Insert:
        return "insert";
    case HomotopyAction::Stop:
        return "stop";
    }
    return "?";
}

const char* to_string(HomotopyStatus status) {
    switch (status) {
    case HomotopyStatus::GoalReached:
        return "goal_reached";
    case HomotopyStatus::GoalNotReached:
        return "goal_not_reached";
    case HomotopyStatus::StepTooSmall:
        return "step_too_small";
    case HomotopyStatus::MaxElements:
        return "max_elements";
    case HomotopyStatus::MaxSteps:
        return "max_steps";
    case HomotopyStatus::StartRejected:
        return "start_rejected";
    }
    return "?";
}

double homotopy_merit_threshold(const HomotopyOptions& options) {
    if (options.merit_threshold > 0.0) {
        return options.merit_threshold;
    }
    const double s = options.merit_factor * options.target_rms;
    return s * s;
}

namespace {

[[noreturn]] void invalid(const std::string& what) {
    throw std::invalid_argument("Homotopie: " + what);
}

bool finite_positive(double v) {
    return std::isfinite(v) && v > 0.0;
}

void check_control(const HomotopyStepControl& c, const char* name) {
    if (!finite_positive(c.initial) || !finite_positive(c.min) || !finite_positive(c.max) ||
        c.min > c.initial || c.initial > c.max) {
        invalid(std::string(name) + ": verlangt 0 < Δ_min ≤ Δ₀ ≤ Δ_max");
    }
    if (!(c.grow >= 1.0) || !std::isfinite(c.grow) || !(c.shrink > 0.0) || !(c.shrink < 1.0)) {
        invalid(std::string(name) + ": verlangt grow ≥ 1 und 0 < shrink < 1");
    }
}

void check_options_impl(const HomotopyOptions& o) {
    if (!finite_positive(o.target_rms)) {
        invalid("target_rms muss endlich und > 0 sein");
    }
    if (!(o.merit_threshold <= 0.0 || std::isfinite(o.merit_threshold)) ||
        (o.merit_threshold <= 0.0 && !finite_positive(o.merit_factor))) {
        invalid("Schwelle der Merit ungültig");
    }
    if (o.fast_iterations < 0 || o.penalty_stages < 1 || o.smoothing_stages < 0 ||
        o.ghost_weight_stages < 0 || o.max_elements < 0 || o.max_steps < 1) {
        invalid("Zahl der Stufen, Elemente oder Schritte ungültig");
    }
    if (!(o.penalty_factor > 1.0) || !std::isfinite(o.penalty_factor) ||
        !(o.ghost_weight_factor > 1.0) || !std::isfinite(o.ghost_weight_factor) ||
        !(o.smoothing_factor > 0.0) || !(o.smoothing_factor < 1.0)) {
        invalid("Faktoren der Fortsetzungen ungültig (penalty, μ > 1; 0 < ε-Faktor < 1)");
    }
    check_control(o.aperture_step, "Apertur");
    check_control(o.field_step, "Feld");
}

} // namespace

void check_homotopy_options(const HomotopyOptions& options) {
    check_options_impl(options);
}

HomotopyPoint homotopy_target(const HomotopyPoint& start, const HomotopyOptions& options) {
    HomotopyPoint t = start;
    if (options.target_aperture > 0.0) {
        t.aperture = options.target_aperture;
    }
    if (options.target_field >= 0.0) {
        t.field = options.target_field;
    }
    if (options.target_spectrum >= 0) {
        t.spectrum = options.target_spectrum;
    }
    if (t.spectrum < start.spectrum) {
        invalid("Spektrumstufe des Ziels liegt vor der des Starts");
    }
    if (!std::isfinite(t.aperture) || !std::isfinite(t.field)) {
        invalid("Zielparameter nicht endlich");
    }
    return t;
}

// ---------------------------------------------------------------------------------------------
// Standards der Schnittstelle für die Strahlsuche (M6-T4)
// ---------------------------------------------------------------------------------------------

std::vector<HomotopySolve> HomotopyModel::solve_all(std::span<SystemDescription* const> systems,
                                                    const HomotopyPoint& point,
                                                    std::vector<int>* groups) {
    std::vector<HomotopySolve> out;
    out.reserve(systems.size());
    if (groups != nullptr) {
        groups->clear();
    }
    for (std::size_t i = 0; i < systems.size(); ++i) {
        out.push_back(solve(*systems[i], point));
        if (groups != nullptr) {
            groups->push_back(static_cast<int>(i));
        }
    }
    return out;
}

std::vector<HomotopyNeed> HomotopyModel::insertion_options(const SystemDescription&,
                                                           const HomotopyPoint&,
                                                           const HomotopyNeed& need, int) {
    return {need};
}

double HomotopyModel::distance(const SystemDescription&, const SystemDescription&,
                               const HomotopyPoint&) {
    return std::numeric_limits<double>::infinity();
}

std::vector<SystemDescription> HomotopyModel::perturbed_starts(const SystemDescription&,
                                                               const HomotopyInsertion&,
                                                               const HomotopyPoint&, int,
                                                               std::uint64_t) {
    return {};
}

std::vector<bool> HomotopyModel::prefilter(std::span<const SystemDescription* const> starts,
                                           const HomotopyPoint&) {
    return std::vector<bool>(starts.size(), true);
}

namespace {

// Abbruch der Steuerung (Status und Meldung), intern als Ausnahme durch die Phasen gereicht.
struct Stop {
    HomotopyStatus status;
    std::string message;
};

std::string number(double v, const char* fmt = "%.6g") {
    char buf[64];
    std::snprintf(buf, sizeof buf, fmt, v);
    return buf;
}

struct State {
    SystemDescription system;
    HomotopyPoint point;
};

class Controller {
public:
    Controller(HomotopyModel& model, const HomotopyOptions& options, HomotopyResult& result)
        : model_(model), o_(options), r_(result), threshold_(homotopy_merit_threshold(options)) {
        r_.merit_threshold = threshold_;
    }

    void run(State& s) {
        r_.start_elements = model_.elements(s.system);
        elements_ = r_.start_elements;
        start_phase(s);
        if (r_.target.aperture != s.point.aperture) {
            continuous_phase(s, HomotopyStepKind::Aperture, r_.target.aperture, o_.aperture_step);
        }
        if (r_.target.field != s.point.field) {
            continuous_phase(s, HomotopyStepKind::Field, r_.target.field, o_.field_step);
        }
        spectrum_phase(s);
        finish_phase(s);
    }

private:
    HomotopyModel& model_;
    const HomotopyOptions& o_;
    HomotopyResult& r_;
    double threshold_;
    int elements_ = 0;

    using Clock = std::chrono::steady_clock;

    void emit(HomotopyStep step) {
        step.index = static_cast<int>(r_.steps.size());
        step.elements = elements_;
        r_.steps.push_back(std::move(step));
        if (o_.on_step) {
            o_.on_step(r_.steps.back());
        }
    }

    // Optimierung an `point` mit Protokolleintrag (noch ohne Entscheidung); liefert den Eintrag.
    HomotopyStep solve(SystemDescription& system, const HomotopyPoint& point,
                       HomotopyStepKind kind) {
        if (r_.solves >= o_.max_steps) {
            throw Stop{HomotopyStatus::MaxSteps, "höchste Zahl von Optimierungen erreicht (" +
                                                     std::to_string(o_.max_steps) + ")"};
        }
        const auto t0 = Clock::now();
        HomotopyStep step;
        step.kind = kind;
        step.point = point;
        step.solve = model_.solve(system, point);
        step.solved = true;
        step.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
        ++r_.solves;
        return step;
    }

    bool acceptable(const HomotopySolve& s) const {
        return std::isfinite(s.merit) && s.status != LmStatus::NumericalError &&
               s.failed_rays == 0 && s.wavefront + s.ghost_penalty <= threshold_;
    }

    bool fast(const HomotopySolve& s) const {
        return lm_converged(s.status) && s.iterations <= o_.fast_iterations;
    }

    // Penalty-Fortsetzung am Punkt des Zustands: Faktoren f^{−(K−1)} … 1. Ohne einseitige
    // Bedingungen eine Optimierung mit Faktor 1 (Art `kind`).
    // `note` steht vor der Bemerkung der ersten Stufe.
    HomotopyStep penalty_continuation(State& s, HomotopyStepKind kind,
                                      const std::string& note = {}) {
        HomotopyStep last;
        if (!model_.one_sided_constraints(s.system)) {
            s.point.penalty_scale = 1.0;
            last = solve(s.system, s.point, kind);
            last.accepted = true;
            last.operation = note;
            emit(last);
            return last;
        }
        for (int j = 0; j < o_.penalty_stages; ++j) {
            double scale = 1.0;
            for (int k = j; k < o_.penalty_stages - 1; ++k) {
                scale /= o_.penalty_factor;
            }
            s.point.penalty_scale = scale;
            last = solve(s.system, s.point, HomotopyStepKind::Penalty);
            last.accepted = true;
            last.operation = (j == 0 && !note.empty() ? note + "; " : std::string()) +
                             "Gewichte der einseitigen Bedingungen ×" + number(scale);
            emit(last);
        }
        return last;
    }

    // Bedarf aus der Diagnose: zuerst am abgelehnten Versuch (dort zeigt sich der Bedarf des
    // nächsten Schritts), sonst am letzten guten Zustand.
    HomotopyNeed need_for(const State& good, const State* trial) {
        if (trial != nullptr) {
            HomotopyNeed n = model_.diagnose(trial->system, trial->point);
            if (n.recommendation != GhostOperator::None) {
                return n;
            }
        }
        return model_.diagnose(good.system, good.point);
    }

    // Einfügen am guten Zustand `s` (sein Punkt bleibt): Operator, innere Homotopie pro Zweig,
    // bester Zweig, Abschluss und Penalty-Fortsetzung. Wirft Stop ohne Bedarf bzw. bei der
    // höchsten Elementzahl.
    void insert(State& s, const HomotopyNeed& need, HomotopyStatus no_need_status,
                const std::string& no_need_message) {
        if (need.recommendation == GhostOperator::None) {
            throw Stop{no_need_status,
                       no_need_message + "; keine Geisterplatte aktiv (kein Bedarf)"};
        }
        if (elements_ >= o_.max_elements) {
            throw Stop{HomotopyStatus::MaxElements,
                       no_need_message + "; Bedarf (Operator " +
                           std::string(to_string(need.recommendation)) +
                           "), aber höchste Elementzahl " + std::to_string(o_.max_elements) +
                           " erreicht"};
        }
        const auto t0 = Clock::now();
        HomotopyInsertion ins = model_.insert(s.system, s.point, need);
        if (ins.branches.empty()) {
            invalid("das Modell lieferte keinen Zweig");
        }
        {
            HomotopyStep step;
            step.kind = HomotopyStepKind::Insertion;
            step.point = s.point;
            step.recommendation = need.recommendation;
            step.operation = ins.text;
            step.accepted = true;
            step.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
            emit(step);
        }
        ++r_.insertions;
        // Zweige: innere Homotopie, dann der beste (endliche Merit, wenigste versagte Strahlen,
        // kleinste Merit, bei Gleichstand der kleinere Index)
        int best = -1;
        HomotopySolve best_solve;
        SystemDescription best_system;
        for (std::size_t b = 0; b < ins.branches.size(); ++b) {
            SystemDescription sys = ins.branches[b];
            HomotopySolve last;
            if (ins.stages.empty()) {
                HomotopyStep step = solve(sys, s.point, HomotopyStepKind::InsertionStage);
                step.branch = static_cast<int>(b);
                step.accepted = true;
                last = step.solve;
                emit(step);
            }
            for (const double v : ins.stages) {
                model_.set_stage(sys, ins, v);
                HomotopyStep step = solve(sys, s.point, HomotopyStepKind::InsertionStage);
                step.branch = static_cast<int>(b);
                step.stage = v;
                step.accepted = true;
                last = step.solve;
                emit(step);
            }
            const bool better = [&] {
                if (best < 0) {
                    return true;
                }
                if (std::isfinite(last.merit) != std::isfinite(best_solve.merit)) {
                    return std::isfinite(last.merit);
                }
                if (last.failed_rays != best_solve.failed_rays) {
                    return last.failed_rays < best_solve.failed_rays;
                }
                return last.merit < best_solve.merit;
            }();
            if (better) {
                best = static_cast<int>(b);
                best_solve = last;
                best_system = std::move(sys);
            }
        }
        model_.finish_insertion(best_system, ins);
        s.system = std::move(best_system);
        elements_ = model_.elements(s.system);
        penalty_continuation(s, HomotopyStepKind::Start,
                             "Zweig " + std::to_string(best) + " von " +
                                 std::to_string(ins.branches.size()) + " übernommen");
    }

    // Startpunkt: Penalty-Fortsetzung; nicht annehmbar → Einfügen, solange es Bedarf gibt.
    void start_phase(State& s) {
        HomotopyStep last = penalty_continuation(s, HomotopyStepKind::Start);
        while (!acceptable(last.solve)) {
            const HomotopyNeed need = model_.diagnose(s.system, s.point);
            insert(s, need, HomotopyStatus::StartRejected,
                   "Startpunkt nicht annehmbar (F_B = " +
                       number(last.solve.wavefront + last.solve.ghost_penalty) +
                       " > F_max = " + number(threshold_) + ")");
            last = r_.steps.back();
        }
    }

    static double& parameter(HomotopyPoint& p, HomotopyStepKind kind) {
        return kind == HomotopyStepKind::Aperture ? p.aperture : p.field;
    }

    // Stetiger Parameter (Apertur, Feld) mit Schrittweitensteuerung.
    void continuous_phase(State& s, HomotopyStepKind kind, double target,
                          const HomotopyStepControl& c) {
        double delta = c.initial;
        const char* name = kind == HomotopyStepKind::Aperture ? "Apertur" : "Feldanteil";
        while (parameter(s.point, kind) != target) {
            State trial = s;
            const double from = parameter(s.point, kind);
            double& value = parameter(trial.point, kind);
            value = target > from ? std::min(from + delta, target) : std::max(from - delta, target);
            // Rundungsreste der Summe nicht als eigenen Schritt gehen
            if (std::abs(target - value) <= 1e-6 * c.min) {
                value = target;
            }
            HomotopyStep step = solve(trial.system, trial.point, kind);
            step.delta = delta;
            if (acceptable(step.solve)) {
                step.accepted = true;
                if (fast(step.solve)) {
                    delta = std::min(delta * c.grow, c.max);
                    step.action = HomotopyAction::Grow;
                } else {
                    step.action = HomotopyAction::Keep;
                }
                step.next_delta = delta;
                emit(step);
                s = std::move(trial);
                continue;
            }
            delta *= c.shrink;
            step.next_delta = delta;
            if (delta >= c.min) {
                step.action = HomotopyAction::Shrink;
                emit(step);
                continue;
            }
            // Δ < Δ_min: Einfügen am letzten guten Zustand, wenn es Bedarf gibt
            const HomotopyNeed need = need_for(s, &trial);
            step.recommendation = need.recommendation;
            step.action = need.recommendation == GhostOperator::None || elements_ >= o_.max_elements
                              ? HomotopyAction::Stop
                              : HomotopyAction::Insert;
            step.operation = need.text;
            emit(step);
            insert(s, need, HomotopyStatus::StepTooSmall,
                   std::string(name) + " " + number(from) + ": Schrittweite unter Δ_min = " +
                       number(c.min) + " (letzter Versuch F_B = " +
                       number(step.solve.wavefront + step.solve.ghost_penalty) + ")");
            delta = c.initial;
        }
    }

    // Spektrumstufen: jede Stufe ein Schritt; abgelehnt → Einfügen bei Bedarf, dann erneut.
    void spectrum_phase(State& s) {
        while (s.point.spectrum < r_.target.spectrum) {
            State trial = s;
            ++trial.point.spectrum;
            HomotopyStep step = solve(trial.system, trial.point, HomotopyStepKind::Spectrum);
            if (acceptable(step.solve)) {
                step.accepted = true;
                step.action = HomotopyAction::Keep;
                emit(step);
                s = std::move(trial);
                continue;
            }
            const HomotopyNeed need = need_for(s, &trial);
            step.recommendation = need.recommendation;
            step.action = need.recommendation == GhostOperator::None || elements_ >= o_.max_elements
                              ? HomotopyAction::Stop
                              : HomotopyAction::Insert;
            step.operation = need.text;
            emit(step);
            insert(s, need, HomotopyStatus::StepTooSmall,
                   "Spektrumstufe " + std::to_string(trial.point.spectrum) +
                       " nicht annehmbar (F_B = " +
                       number(step.solve.wavefront + step.solve.ghost_penalty) + ")");
        }
    }

    void check_goal(const State& s) {
        HomotopyStep step;
        step.kind = HomotopyStepKind::Goal;
        step.point = s.point;
        const auto t0 = Clock::now();
        step.goal = model_.goal(s.system, s.point, o_.target_rms);
        step.goal_checked = true;
        step.accepted = step.goal.reached;
        step.operation = std::string(step.goal.reached ? "Ziel erreicht" : "Ziel verfehlt") +
                         ": größter RMS " + number(step.goal.max_rms, "%.5f") + " Wellen (Ziel " +
                         number(o_.target_rms, "%.5f") + ")";
        step.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
        r_.goal = step.goal;
        emit(step);
    }

    // Abschluss am Endpunkt: ε-Fortsetzung, Ziel prüfen; verfehlt: μ-Fortsetzung, danach bei
    // Bedarf Einfügen am Zustand vor dem Abschluss und erneut.
    void finish_phase(State& s) {
        for (;;) {
            const State saved = s;
            for (int j = 0; j < o_.smoothing_stages; ++j) {
                s.point.ghost_smoothing *= o_.smoothing_factor;
                HomotopyStep step = solve(s.system, s.point, HomotopyStepKind::Smoothing);
                step.accepted = true;
                step.operation = "ε ×" + number(s.point.ghost_smoothing);
                emit(step);
            }
            check_goal(s);
            if (r_.goal.reached) {
                return;
            }
            for (int j = 0; j < o_.ghost_weight_stages && !r_.goal.reached; ++j) {
                s.point.ghost_weight *= o_.ghost_weight_factor;
                HomotopyStep step = solve(s.system, s.point, HomotopyStepKind::GhostWeight);
                step.accepted = true;
                step.operation = "μ ×" + number(s.point.ghost_weight);
                emit(step);
                check_goal(s);
            }
            if (r_.goal.reached) {
                return;
            }
            const HomotopyNeed need = model_.diagnose(saved.system, saved.point);
            r_.steps.back().recommendation = need.recommendation;
            const std::string why = "Ziel am Endpunkt verfehlt (größter RMS " +
                                    number(r_.goal.max_rms, "%.5f") + " Wellen)";
            State next = saved;
            insert(next, need, HomotopyStatus::GoalNotReached, why);
            s = std::move(next);
        }
    }
};

} // namespace

HomotopyResult run_homotopy(HomotopyModel& model, SystemDescription system,
                            const HomotopyPoint& start, const HomotopyOptions& options) {
    check_homotopy_options(options);
    HomotopyResult r;
    r.target = homotopy_target(start, options);
    const auto t0 = std::chrono::steady_clock::now();
    State s{std::move(system), start};
    Controller c(model, options, r);
    try {
        c.run(s);
        r.status = r.goal.reached ? HomotopyStatus::GoalReached : HomotopyStatus::GoalNotReached;
        r.message = r.goal.reached ? "Ziel erreicht" : "Ziel verfehlt";
    } catch (const Stop& stop) {
        r.status = stop.status;
        r.message = stop.message;
    }
    r.system = std::move(s.system);
    r.point = s.point;
    r.elements = model.elements(r.system);
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

// ---------------------------------------------------------------------------------------------
// Protokoll als Text und JSON
// ---------------------------------------------------------------------------------------------

namespace {

using Json = nlohmann::ordered_json;

Json num(double v) {
    return std::isfinite(v) ? Json(v) : Json(nullptr);
}

Json point_json(const HomotopyPoint& p) {
    return Json{{"aperture", num(p.aperture)},
                {"field", num(p.field)},
                {"spectrum", p.spectrum},
                {"ghost_weight", num(p.ghost_weight)},
                {"ghost_smoothing", num(p.ghost_smoothing)},
                {"penalty_scale", num(p.penalty_scale)}};
}

Json goal_json(const HomotopyGoal& g) {
    Json tuples = Json::array();
    for (const HomotopyTupleRms& t : g.tuples) {
        tuples.push_back(Json{{"field", t.field},
                              {"wavelength", t.wavelength},
                              {"field_angle_deg", num(t.field_angle)},
                              {"wavelength_nm", num(t.wavelength_nm)},
                              {"rms_waves", num(t.rms)},
                              {"failed_rays", t.failed}});
    }
    return Json{{"reached", g.reached},
                {"target_waves", num(g.target)},
                {"max_rms_waves", num(g.max_rms)},
                {"tuples", tuples}};
}

Json step_object(const HomotopyStep& s) {
    Json j{{"index", s.index}, {"kind", to_string(s.kind)}, {"point", point_json(s.point)}};
    j["delta"] = num(s.delta);
    j["branch"] = s.branch >= 0 ? Json(s.branch) : Json(nullptr);
    j["stage"] = s.branch >= 0 ? num(s.stage) : Json(nullptr);
    if (s.solved) {
        j["lm"] = Json{{"status", to_string(s.solve.status)},
                       {"iterations", s.solve.iterations},
                       {"start_merit", num(s.solve.start_merit)},
                       {"merit", num(s.solve.merit)},
                       {"wavefront", num(s.solve.wavefront)},
                       {"ghost_penalty", num(s.solve.ghost_penalty)},
                       {"rms_waves", num(s.solve.rms)},
                       {"ghost_sum_mm", num(s.solve.ghost_sum)},
                       {"failed_rays", s.solve.failed_rays}};
    } else {
        j["lm"] = nullptr;
    }
    j["accepted"] = s.accepted;
    j["action"] = to_string(s.action);
    j["next_delta"] = num(s.next_delta);
    j["recommendation"] = to_string(s.recommendation);
    j["operation"] = s.operation;
    j["elements"] = s.elements;
    j["goal"] = s.goal_checked ? goal_json(s.goal) : Json(nullptr);
    j["seconds"] = num(s.seconds);
    return j;
}

std::string fmt(const char* f, double v) {
    if (!std::isfinite(v)) {
        return "-";
    }
    char buf[64];
    std::snprintf(buf, sizeof buf, f, v);
    return buf;
}

std::string pad(std::string s, std::size_t width) {
    // Breite in Zeichen (UTF-8-Fortsetzungsbytes zählen nicht)
    std::size_t w = 0;
    for (const char ch : s) {
        if ((static_cast<unsigned char>(ch) & 0xC0) != 0x80) {
            ++w;
        }
    }
    while (w < width) {
        s.insert(s.begin(), ' ');
        ++w;
    }
    return s;
}

} // namespace

std::string homotopy_step_json(const HomotopyStep& step, int indent) {
    return step_object(step).dump(indent);
}

std::string homotopy_result_json(const HomotopyResult& r, int indent) {
    Json steps = Json::array();
    for (const HomotopyStep& s : r.steps) {
        steps.push_back(step_object(s));
    }
    Json j{{"status", to_string(r.status)},
           {"message", r.message},
           {"start_point", r.steps.empty() ? Json(nullptr) : point_json(r.steps.front().point)},
           {"target", point_json(r.target)},
           {"point", point_json(r.point)},
           {"merit_threshold", num(r.merit_threshold)},
           {"start_elements", r.start_elements},
           {"elements", r.elements},
           {"insertions", r.insertions},
           {"solves", r.solves},
           {"goal", goal_json(r.goal)},
           {"seconds", num(r.seconds)},
           {"steps", steps}};
    return j.dump(indent);
}

void write_homotopy_step_header(std::ostream& out) {
    out << "Schritt  Art              NA      Feld  Sp  Delta     Status            Zyklen  "
           "F [W^2]       F_B [W^2]     RMS [W]   Sum|a| [mm]  El  Entscheidung\n";
}

void write_homotopy_step_line(const HomotopyStep& s, std::ostream& out) {
    std::string kind = to_string(s.kind);
    kind.resize(15, ' ');
    out << pad(std::to_string(s.index), 7) << "  " << kind << "  " << fmt("%.4f", s.point.aperture)
        << "  " << fmt("%.3f", s.point.field) << "  " << pad(std::to_string(s.point.spectrum), 2)
        << "  " << (s.delta > 0.0 ? fmt("%.5f", s.delta) : std::string("-      ")) << "  ";
    if (s.solved) {
        std::string st = to_string(s.solve.status);
        st.resize(16, ' ');
        out << st << "  " << pad(std::to_string(s.solve.iterations), 6) << "  "
            << fmt("%.6e", s.solve.merit) << "  "
            << fmt("%.6e", s.solve.wavefront + s.solve.ghost_penalty) << "  "
            << fmt("%.5f", s.solve.rms) << "   " << fmt("%.3e", s.solve.ghost_sum) << "  ";
    } else {
        out << "-                      -  -             -             -         -            ";
    }
    out << pad(std::to_string(s.elements), 2) << "  ";
    if (s.action != HomotopyAction::None) {
        out << to_string(s.action);
        if (s.next_delta > 0.0) {
            out << " (Delta " << fmt("%.5f", s.next_delta) << ")";
        }
    } else {
        out << (s.accepted ? "ok" : "-");
    }
    if (s.branch >= 0) {
        out << ", Zweig " << s.branch << ", Stufe " << fmt("%.5g", s.stage);
    }
    if (!s.operation.empty()) {
        out << ": " << s.operation;
    }
    out << '\n';
}

void write_homotopy_summary(const HomotopyResult& r, std::ostream& out) {
    out << "Status: " << to_string(r.status) << " (" << r.message << ")\n";
    out << "Endpunkt: NA " << fmt("%.4f", r.point.aperture) << ", Feldanteil "
        << fmt("%.3f", r.point.field) << ", Spektrumstufe " << r.point.spectrum << ", mu x"
        << fmt("%.4g", r.point.ghost_weight) << ", eps x" << fmt("%.4g", r.point.ghost_smoothing)
        << " (Ziel NA " << fmt("%.4f", r.target.aperture) << ")\n";
    out << "Glaselemente: " << r.start_elements << " -> " << r.elements << " (" << r.insertions
        << " Einfügungen), Optimierungen: " << r.solves
        << ", F_max = " << fmt("%.6e", r.merit_threshold) << " Wellen^2\n";
    if (!r.goal.tuples.empty()) {
        out << "Reales System (Geisterkoeffizienten 0), RMS-Wellenfront pro Feld und "
               "Wellenlänge (Ziel "
            << fmt("%.5f", r.goal.target) << " Wellen):\n";
        for (const HomotopyTupleRms& t : r.goal.tuples) {
            out << "  Feld " << t.field << " (" << fmt("%.4f", t.field_angle) << " deg), "
                << fmt("%.2f", t.wavelength_nm) << " nm: RMS " << fmt("%.5f", t.rms)
                << " Wellen, versagt " << t.failed << '\n';
        }
    }
}

} // namespace mko
