// Homotopie-Controller (M6-T3): Schrittweitensteuerung, Fortsetzungen, Einfügen, Protokoll.
// Festlegungen: docs/optimierung.md, Abschnitt 7.

#include "mikroptim/synth/homotopy.hpp"

#include "mikroptim/model/format_error.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <optional>
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
    case HomotopyStepKind::SpectrumWidth:
        return "spectrum_width";
    case HomotopyStepKind::ExtremaSmoothing:
        return "extrema_smoothing";
    case HomotopyStepKind::GhostWeight:
        return "ghost_weight";
    case HomotopyStepKind::Goal:
        return "goal";
    case HomotopyStepKind::ApertureField:
        return "aperture_field";
    case HomotopyStepKind::RemovalStage:
        return "removal_stage";
    case HomotopyStepKind::Removal:
        return "removal";
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
    if (!(o.grow_margin >= 0.0 && o.grow_margin <= 1.0)) {
        invalid("grow_margin muss in [0, 1] liegen");
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
    if (!(o.extrema_beta >= 0.0) || !std::isfinite(o.extrema_beta) || o.extrema_stages < 0 ||
        !(o.extrema_factor > 1.0) || !std::isfinite(o.extrema_factor)) {
        invalid("β-Fortsetzung ungültig (β ≥ 0, Stufen ≥ 0, Faktor > 1)");
    }
    const AugmentedLagrangianOptions& al = o.augmented_lagrangian;
    if (al.rounds < 0 || !(al.tolerance >= 0.0) || !std::isfinite(al.tolerance) ||
        !(al.max_shift >= 0.0) || !std::isfinite(al.max_shift) ||
        !finite_positive(o.al_penalty_scale)) {
        invalid("Augmented Lagrangian ungültig (Läufe ≥ 0, Toleranz und Schranke ≥ 0, Faktor > 0)");
    }
    check_control(o.aperture_step, "Apertur");
    check_control(o.field_step, "Feld");
    check_control(o.spectrum_step, "Spektrumbreite");
    check_control(o.joint_step, "Apertur und Feld");
    if (o.insertion_attempts < 1) {
        invalid("insertion_attempts muss ≥ 1 sein");
    }
    if (!finite_positive(o.joint_field_exponent)) {
        invalid("Exponent des Feldes der gemeinsamen Homotopie muss endlich und > 0 sein");
    }
    if (o.prune_stages < 1 || o.prune_attempts < 1 || o.prune_min_elements < 0 ||
        !finite_positive(o.prune_merit_factor) || o.prune_merit_factor > 1.0 ||
        !std::isfinite(o.prune_tolerance)) {
        invalid("Rückbau ungültig (Stufen und Versuche ≥ 1, Mindestzahl ≥ 0, 0 < Faktor ≤ 1, "
                "Toleranz endlich)");
    }
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
    if (options.target_spectrum_width >= 0.0) {
        if (options.target_spectrum_width > 1.0) {
            invalid("Spektrumbreite des Ziels über 1");
        }
        t.spectrum_width = options.target_spectrum_width;
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

std::optional<SystemDescription> HomotopyModel::predict(const SystemDescription&,
                                                        const SystemDescription&,
                                                        const HomotopyPoint&, double) {
    return std::nullopt;
}

double HomotopyModel::update_shifts(SystemDescription&, const HomotopyPoint&, double) {
    return 0.0;
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

std::vector<HomotopyRemoval> HomotopyModel::removal_candidates(const SystemDescription&,
                                                               const HomotopyPoint&) {
    return {};
}

SystemDescription HomotopyModel::collapse(const SystemDescription&, const SystemDescription&,
                                          const HomotopyRemoval&, double,
                                          std::vector<std::string>&) {
    throw std::invalid_argument("Homotopie: das Modell kann keine Elemente zurückbauen");
}

SystemDescription HomotopyModel::remove(const SystemDescription&, const SystemDescription&,
                                        const HomotopyRemoval&, std::vector<std::string>&) {
    throw std::invalid_argument("Homotopie: das Modell kann keine Elemente zurückbauen");
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
        try {
            phases(s);
        } catch (const Stop&) {
            // Rückbau (M6-T9) auch am Ende eines abgebrochenen Laufs (am letzten guten Zustand);
            // ein weiterer Abbruch darin (z. B. max_steps) beendet nur den Rückbau
            if (o_.prune) {
                try {
                    maybe_prune(s, true, "am Ende");
                } catch (const Stop&) {
                }
            }
            throw;
        }
    }

private:
    void phases(State& s) {
        start_phase(s);
        if (o_.joint_aperture_field && r_.target.aperture != s.point.aperture &&
            r_.target.field != s.point.field) {
            joint_phase(s);
            maybe_prune(s, false);
        }
        if (r_.target.aperture != s.point.aperture) {
            single_phase(s, HomotopyStepKind::Aperture, r_.target.aperture, o_.aperture_step);
            maybe_prune(s, false);
        }
        if (r_.target.field != s.point.field) {
            single_phase(s, HomotopyStepKind::Field, r_.target.field, o_.field_step);
            maybe_prune(s, false);
        }
        if (r_.target.spectrum_width != s.point.spectrum_width) {
            single_phase(s, HomotopyStepKind::SpectrumWidth, r_.target.spectrum_width,
                         o_.spectrum_step);
            maybe_prune(s, false);
        }
        if (s.point.spectrum < r_.target.spectrum) {
            spectrum_phase(s);
            maybe_prune(s, false);
        }
        maybe_prune(s, true);
        finish_phase(s);
    }

    HomotopyModel& model_;
    const HomotopyOptions& o_;
    HomotopyResult& r_;
    double threshold_;
    int elements_ = 0;

    using Clock = std::chrono::steady_clock;

    // F_B des zuletzt übernommenen Zustands (ohne Zweige der inneren Homotopie; M6-T8)
    double good_fb_ = std::numeric_limits<double>::quiet_NaN();
    // Rückbau (M6-T9): Zahl der Einfügungen beim letzten Rückbau, ob schon einer lief, Punkt des
    // letzten Rückbaus am Elementbudget
    int insertions_at_prune_ = 0;
    bool pruned_ = false;
    std::optional<HomotopyPoint> limit_prune_point_;

    void emit(HomotopyStep step) {
        step.index = static_cast<int>(r_.steps.size());
        step.elements = elements_;
        if (step.solved && step.accepted && step.kind != HomotopyStepKind::InsertionStage) {
            good_fb_ = step.solve.wavefront + step.solve.ghost_penalty;
        }
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
        if (o_.augmented_lagrangian.rounds > 0) {
            // Augmented Lagrangian (M11-T2): Fortschreibung nach jedem LM-Lauf
            step.solve.shift_change =
                model_.update_shifts(system, point, o_.augmented_lagrangian.max_shift);
        }
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
        if (o_.augmented_lagrangian.rounds > 0) {
            // Augmented Lagrangian (M11-T2): festes Gewicht, Läufe bis max √w |Δθ| ≤ Toleranz
            s.point.penalty_scale = o_.al_penalty_scale;
            for (int j = 0; j < o_.augmented_lagrangian.rounds; ++j) {
                last = solve(s.system, s.point, HomotopyStepKind::Penalty);
                last.accepted = true;
                last.operation = (j == 0 && !note.empty() ? note + "; " : std::string()) +
                                 "Augmented Lagrangian, Lauf " + std::to_string(j + 1) +
                                 ", Gewichte ×" + number(o_.al_penalty_scale) +
                                 ", max √w|Δθ| = " + number(last.solve.shift_change);
                emit(last);
                if (!(last.solve.shift_change > o_.augmented_lagrangian.tolerance)) {
                    break;
                }
            }
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
    // choose_at: Zweige zusätzlich an diesem Punkt bewerten (Spektrumphase: an der verworfenen
    // Stufe, sonst wählt die monochromatische Merit immer den Zweig ohne Farbkorrektur)
    // before_fb (M6-T8, insertion_must_improve): F_B des guten Zustands; dann wird ein Einfügen
    // nur übernommen, wenn F_B danach kleiner ist, sonst die nächste Variante versucht.
    // start (M6-T11b, Startphase): zusätzlich übernommen, wenn F_B danach höchstens F_max ist
    // (Spielraum, z. B. nach einem Start mit versagten Strahlen und kleinem F_B).
    void insert(State& s, const HomotopyNeed& need, HomotopyStatus no_need_status,
                const std::string& no_need_message, const HomotopyPoint* choose_at = nullptr,
                const double* before_fb = nullptr, bool start = false) {
        if (need.recommendation == GhostOperator::None) {
            throw Stop{no_need_status,
                       no_need_message + "; keine Geisterplatte aktiv (kein Bedarf)"};
        }
        if (elements_ >= o_.max_elements) {
            // Elementbudget (M6-T9): erst zurückbauen, je Punkt höchstens einmal; entfernt der
            // Rückbau etwas, geht der Lauf ohne Einfügen weiter (die Phase versucht neu)
            if (o_.prune_at_limit && limit_prune_point_ != s.point) {
                limit_prune_point_ = s.point;
                if (prune(s, "Elementbudget " + std::to_string(o_.max_elements)) > 0) {
                    return;
                }
            }
            throw Stop{HomotopyStatus::MaxElements,
                       no_need_message + "; Bedarf (Operator " +
                           std::string(to_string(need.recommendation)) +
                           "), aber höchste Elementzahl " + std::to_string(o_.max_elements) +
                           " erreicht"};
        }
        if (before_fb == nullptr || !o_.insertion_must_improve) {
            insert_once(s, need, no_need_status, no_need_message, choose_at);
            return;
        }
        const std::vector<HomotopyNeed> options =
            model_.insertion_options(s.system, s.point, need, o_.insertion_attempts);
        std::string tried;
        for (const HomotopyNeed& option : options) {
            const State backup = s;
            const int elements = elements_;
            const double good = good_fb_;
            insert_once(s, option, no_need_status, no_need_message, choose_at);
            const HomotopySolve& last = r_.steps.back().solve;
            const double after = last.wavefront + last.ghost_penalty;
            if (std::isfinite(after) && last.failed_rays == 0 &&
                (after < *before_fb || (start && after <= threshold_))) {
                return;
            }
            s = backup;
            elements_ = elements;
            good_fb_ = good;
            HomotopyStep note;
            note.kind = HomotopyStepKind::Insertion;
            note.point = s.point;
            note.recommendation = option.recommendation;
            note.operation = "Einfügen verworfen: F_B danach " + number(after) + " ≥ " +
                             number(*before_fb) + " davor" +
                             (start ? " und > F_max = " + number(threshold_) : std::string()) +
                             " (Einfügen nur mit Verbesserung" +
                             (start ? ", Startphase" : "") + ")";
            emit(note);
            tried += (tried.empty() ? "" : ", ") + number(after);
        }
        throw Stop{no_need_status, no_need_message + "; kein Einfügen verbessert F_B (" +
                                       std::to_string(options.size()) + " Varianten, F_B " + tried +
                                       " ≥ " + number(*before_fb) +
                                       (start ? ", alle > F_max = " + number(threshold_)
                                              : std::string()) +
                                       ")"};
    }

    void insert_once(State& s, const HomotopyNeed& need, HomotopyStatus no_need_status,
                     const std::string& no_need_message, const HomotopyPoint* choose_at) {
        const auto t0 = Clock::now();
        HomotopyInsertion ins;
        try {
            ins = model_.insert(s.system, s.point, need);
        } catch (const std::invalid_argument& e) {
            // Einfügen nicht möglich (z. B. Linse zu dünn zum Teilen, Flächentabelle voll):
            // geordnet beenden wie ohne Bedarf, mit Grund (vorher Absturz von synth, M6-T6)
            throw Stop{no_need_status, no_need_message + "; Einfügen (Operator " +
                                           std::string(to_string(need.recommendation)) +
                                           ") nicht möglich: " + e.what()};
        }
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
        // Mehrere Zweige (M6-T11): wirft die Optimierung eines Zweigs (z. B. Brechzahl einer
        // Glasvariable ≤ 0 im Zweig „Startradien des Glieds“, Lauf F3), wird nur dieser Zweig
        // verworfen; werfen alle, die erste Ausnahme. Ein einzelner Zweig wie bisher.
        std::exception_ptr first_error;
        for (std::size_t b = 0; b < ins.branches.size(); ++b) {
            SystemDescription sys = ins.branches[b];
            HomotopySolve last;
            try {
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
                if (choose_at != nullptr) {
                    // am Bewertungspunkt weiteroptimieren und diesen Stand behalten: der nächste
                    // Versuch der Stufe startet dann vom farbig optimierten System
                    SystemDescription& probe = sys;
                    HomotopyStep step = solve(probe, *choose_at, HomotopyStepKind::InsertionStage);
                    step.branch = static_cast<int>(b);
                    step.accepted = true;
                    step.operation = "Bewertung des Zweigs an der verworfenen Spektrumstufe";
                    last = step.solve;
                    emit(step);
                }
            } catch (const std::exception& e) {
                if (ins.branches.size() < 2) {
                    throw;
                }
                if (!first_error) {
                    first_error = std::current_exception();
                }
                HomotopyStep note;
                note.kind = HomotopyStepKind::InsertionStage;
                note.point = s.point;
                note.branch = static_cast<int>(b);
                note.operation = "Zweig verworfen: " + std::string(e.what());
                emit(note);
                continue;
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
        if (best < 0 && first_error) {
            std::rethrow_exception(first_error);
        }
        model_.finish_insertion(best_system, ins);
        s.system = std::move(best_system);
        elements_ = model_.elements(s.system);
        penalty_continuation(s, HomotopyStepKind::Start,
                             "Zweig " + std::to_string(best) + " von " +
                                 std::to_string(ins.branches.size()) + " übernommen");
    }

    // Rückbau nach einer Phase (final = false: nur, wenn seit dem letzten Rückbau eingefügt
    // wurde) bzw. vor dem Abschluss oder nach einem Abbruch (final = true: auch, wenn noch keiner
    // lief).
    void maybe_prune(State& s, bool final, const char* reason = nullptr) {
        if (!o_.prune) {
            return;
        }
        if (r_.insertions == insertions_at_prune_ && (pruned_ || !final)) {
            return;
        }
        prune(s, reason != nullptr ? reason : final ? "vor dem Abschluss" : "nach der Phase");
    }

    // Rückbau-Runden am guten Zustand `s`: pro Runde die Kandidaten des Modells der Reihe nach
    // (höchstens prune_attempts), der erste Erfolg beendet die Runde; ohne Erfolg Ende. Liefert
    // die Zahl der entfernten Kandidaten.
    int prune(State& s, const std::string& reason) {
        pruned_ = true;
        insertions_at_prune_ = r_.insertions;
        int removed = 0;
        for (;;) {
            const std::vector<HomotopyRemoval> candidates =
                model_.removal_candidates(s.system, s.point);
            int tried = 0;
            bool success = false;
            for (const HomotopyRemoval& c : candidates) {
                if (elements_ - c.elements < o_.prune_min_elements) {
                    continue;
                }
                if (tried >= o_.prune_attempts) {
                    break;
                }
                ++tried;
                if (try_removal(s, c, reason)) {
                    ++removed;
                    success = true;
                    break;
                }
            }
            if (!success) {
                return removed;
            }
        }
    }

    // Ein Kandidat: Rückbau-Homotopie (prune_stages Stufen, Optimierung nach jeder Stufe),
    // Entfernen des Null-Elements, Annahme nach F_B. Bei Ablehnung bleibt `s` unverändert.
    bool try_removal(State& s, const HomotopyRemoval& c, const std::string& reason) {
        ++r_.removal_attempts;
        const double before = good_fb_;
        double limit = o_.prune_merit_factor * threshold_;
        if (o_.prune_tolerance >= 0.0 && std::isfinite(before)) {
            limit = std::min(limit, (1.0 + o_.prune_tolerance) * before);
        }
        State trial = s;
        HomotopySolve last;
        std::string why;
        std::vector<std::string> warnings;
        try {
            const int K = o_.prune_stages;
            for (int k = 0; k < K; ++k) {
                const double f = 1.0 / static_cast<double>(K - k);
                trial.system = model_.collapse(s.system, trial.system, c, f, warnings);
                HomotopyStep step =
                    solve(trial.system, trial.point, HomotopyStepKind::RemovalStage);
                step.operation = c.text + ", Rückbau-Stufe " + std::to_string(k + 1) + "/" +
                                 std::to_string(K) + " (Anteil " + number(f) + ")";
                last = step.solve;
                emit(step);
                if (!std::isfinite(last.merit) || last.status == LmStatus::NumericalError) {
                    why = "Optimierung ohne endliches Ergebnis";
                    break;
                }
            }
            if (why.empty()) {
                trial.system = model_.remove(s.system, trial.system, c, warnings);
            }
        } catch (const std::invalid_argument& e) {
            why = e.what();
        } catch (const FormatError& e) {
            why = e.what();
        }
        const double fb = last.wavefront + last.ghost_penalty;
        if (why.empty() && last.failed_rays > 0) {
            why = std::to_string(last.failed_rays) + " versagte Strahlen";
        }
        if (why.empty() && !(fb <= limit)) {
            why = "F_B " + number(fb) + " > " + number(limit);
        }
        HomotopyStep note;
        note.kind = HomotopyStepKind::Removal;
        note.point = s.point;
        if (why.empty()) {
            s.system = std::move(trial.system);
            elements_ = model_.elements(s.system);
            good_fb_ = fb;
            ++r_.removals;
            note.accepted = true;
            note.operation = c.text + " entfernt (" + reason + "): F_B " + number(fb) + " ≤ " +
                             number(limit) + ", vorher " + number(before);
        } else {
            note.operation = "Rückbau verworfen (" + reason + "): " + c.text + ": " + why;
        }
        for (const std::string& w : warnings) {
            note.operation += "; Warnung: " + w;
        }
        emit(note);
        return why.empty();
    }

    // Startpunkt: Penalty-Fortsetzung; nicht annehmbar → Einfügen, solange es Bedarf gibt.
    // Mit insertion_must_improve (M6-T11b) auch hier nur Einfügungen, die F_B verbessern oder
    // danach höchstens F_max haben (vorher stieg F_B in M6-T11, Lauf F7, von 0,011 auf 0,046).
    void start_phase(State& s) {
        HomotopyStep last = penalty_continuation(s, HomotopyStepKind::Start);
        while (!acceptable(last.solve)) {
            const HomotopyNeed need = model_.diagnose(s.system, s.point);
            const double before = last.solve.wavefront + last.solve.ghost_penalty;
            insert(s, need, HomotopyStatus::StartRejected,
                   "Startpunkt nicht annehmbar (F_B = " + number(before) +
                       " > F_max = " + number(threshold_) + ")",
                   nullptr, o_.insertion_must_improve ? &before : nullptr, true);
            last = r_.steps.back();
        }
    }

    static double& parameter(HomotopyPoint& p, HomotopyStepKind kind) {
        return kind == HomotopyStepKind::Aperture ? p.aperture
               : kind == HomotopyStepKind::Field  ? p.field
                                                  : p.spectrum_width;
    }

    // Ein Parameter (Apertur, Feld, Spektrumbreite) ist selbst der Wert der Phase.
    void single_phase(State& s, HomotopyStepKind kind, double target,
                      const HomotopyStepControl& c) {
        const char* name = kind == HomotopyStepKind::Aperture ? "Apertur"
                           : kind == HomotopyStepKind::Field  ? "Feldanteil"
                                                              : "Spektrumbreite";
        continuous_phase(s, kind, name, parameter(s.point, kind), target, c,
                         [kind](HomotopyPoint& p, double v) { parameter(p, kind) = v; });
    }

    // Gemeinsame Apertur-/Feld-Homotopie (M6-T8): t = 0 … 1 auf der Bahn
    // NA(t) = NA₀ + t·(NA₁ − NA₀), φ(t) = φ₀ + t^p·(φ₁ − φ₀); bei t = 1 genau das Ziel.
    void joint_phase(State& s) {
        const double a0 = s.point.aperture;
        const double a1 = r_.target.aperture;
        const double f0 = s.point.field;
        const double f1 = r_.target.field;
        const double p = o_.joint_field_exponent;
        continuous_phase(s, HomotopyStepKind::ApertureField, "Apertur und Feld, t", 0.0, 1.0,
                         o_.joint_step, [=](HomotopyPoint& q, double t) {
                             if (t >= 1.0) {
                                 q.aperture = a1;
                                 q.field = f1;
                                 return;
                             }
                             q.aperture = a0 + t * (a1 - a0);
                             q.field = f0 + (p == 1.0 ? t : std::pow(t, p)) * (f1 - f0);
                         });
    }

    // Stetiger Parameter mit Schrittweitensteuerung: Wert der Phase von `start` bis `target`,
    // `apply` setzt einen Wert in den Punkt.
    void continuous_phase(State& s, HomotopyStepKind kind, const char* name, double start,
                          double target, const HomotopyStepControl& c,
                          const std::function<void(HomotopyPoint&, double)>& apply) {
        double delta = c.initial;
        double current = start; // Wert des guten Zustands s
        // Prädiktor (M11-T3): letzter angenommener Zustand vor s und sein Parameterwert
        std::optional<State> prev;
        double prev_value = 0.0;
        while (current != target) {
            State trial = s;
            const double from = current;
            double value =
                target > from ? std::min(from + delta, target) : std::max(from - delta, target);
            // Rundungsreste der Summe nicht als eigenen Schritt gehen
            if (std::abs(target - value) <= 1e-6 * c.min) {
                value = target;
            }
            apply(trial.point, value);
            bool predicted = false;
            if (o_.predictor && prev && from != prev_value) {
                const double ratio = (value - from) / (from - prev_value);
                if (std::optional<SystemDescription> p =
                        model_.predict(prev->system, s.system, trial.point, ratio)) {
                    trial.system = std::move(*p);
                    predicted = true;
                }
            }
            HomotopyStep step = solve(trial.system, trial.point, kind);
            step.delta = delta;
            if (predicted) {
                step.operation = "Prädiktor (Sekante)";
            }
            if (acceptable(step.solve)) {
                step.accepted = true;
                if (fast(step.solve) ||
                    (o_.grow_margin > 0.0 &&
                     step.solve.wavefront + step.solve.ghost_penalty <=
                         o_.grow_margin * threshold_)) {
                    delta = std::min(delta * c.grow, c.max);
                    step.action = HomotopyAction::Grow;
                } else {
                    step.action = HomotopyAction::Keep;
                }
                step.next_delta = delta;
                emit(step);
                prev = s;
                prev_value = from;
                current = value;
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
            step.action = need.recommendation == GhostOperator::None ||
                                  (elements_ >= o_.max_elements && !o_.prune_at_limit)
                              ? HomotopyAction::Stop
                              : HomotopyAction::Insert;
            step.operation = need.text;
            emit(step);
            prev.reset(); // neue Topologie: Sekante neu aufbauen
            const double before = good_fb_;
            insert(s, need, HomotopyStatus::StepTooSmall,
                   std::string(name) + " " + number(from) + ": Schrittweite unter Δ_min = " +
                       number(c.min) + " (letzter Versuch F_B = " +
                       number(step.solve.wavefront + step.solve.ghost_penalty) + ")",
                   // Spektrumbreite: Zweige am verworfenen, farbigeren Punkt wählen
                   kind == HomotopyStepKind::SpectrumWidth ? &trial.point : nullptr,
                   std::isfinite(before) ? &before : nullptr);
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
            step.action = need.recommendation == GhostOperator::None ||
                                  (elements_ >= o_.max_elements && !o_.prune_at_limit)
                              ? HomotopyAction::Stop
                              : HomotopyAction::Insert;
            step.operation = need.text;
            emit(step);
            insert(s, need, HomotopyStatus::StepTooSmall,
                   "Spektrumstufe " + std::to_string(trial.point.spectrum) +
                       " nicht annehmbar (F_B = " +
                       number(step.solve.wavefront + step.solve.ghost_penalty) + ")",
                   &trial.point);
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

    // β-Fortsetzung der Strahl-Extrema (M11-T1b): β wächst um extrema_factor je Stufe, zuletzt
    // hart (β = 0). Ohne geglättete Extrema nichts.
    void extrema_phase(State& s) {
        if (s.point.extrema_beta <= 0.0) {
            return;
        }
        for (int j = 0; j <= o_.extrema_stages; ++j) {
            s.point.extrema_beta =
                j < o_.extrema_stages ? s.point.extrema_beta * o_.extrema_factor : 0.0;
            HomotopyStep step = solve(s.system, s.point, HomotopyStepKind::ExtremaSmoothing);
            step.accepted = true;
            step.operation = s.point.extrema_beta > 0.0 ? "β = " + number(s.point.extrema_beta)
                                                        : std::string("Extrema hart");
            emit(step);
        }
    }

    // Abschluss am Endpunkt: ε-Fortsetzung, Ziel prüfen; verfehlt: μ-Fortsetzung, danach bei
    // Bedarf Einfügen am Zustand vor dem Abschluss und erneut.
    void finish_phase(State& s) {
        for (;;) {
            extrema_phase(s);
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
    s.point.extrema_beta = options.extrema_beta;
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
                {"penalty_scale", num(p.penalty_scale)},
                {"extrema_beta", num(p.extrema_beta)},
                {"spectrum_width", num(p.spectrum_width)}};
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
        if (std::isfinite(s.solve.shift_change)) {
            j["lm"]["shift_change"] = s.solve.shift_change; // nur mit Augmented Lagrangian
        }
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
           {"insertions", r.insertions}};
    if (r.removal_attempts > 0) {
        // nur mit Rückbau (M6-T9), sonst wie vorher
        j["removal_attempts"] = r.removal_attempts;
        j["removals"] = r.removals;
    }
    j["solves"] = r.solves;
    j["goal"] = goal_json(r.goal);
    j["seconds"] = num(r.seconds);
    j["steps"] = std::move(steps);
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
        << " Einfügungen";
    if (r.removal_attempts > 0) {
        out << ", " << r.removals << " Rückbauten in " << r.removal_attempts << " Versuchen";
    }
    out << "), Optimierungen: " << r.solves << ", F_max = " << fmt("%.6e", r.merit_threshold)
        << " Wellen^2\n";
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
