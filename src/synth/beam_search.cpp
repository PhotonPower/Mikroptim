// Kandidatenverwaltung (M6-T4): Strahlsuche über die Zweige des Homotopie-Controllers im
// Gleichschritt, gemeinsame Optimierung pro Topologie, Auswahl der besten B, Deduplizierung,
// Stammbaum und Protokoll. Festlegungen: docs/optimierung.md, Abschnitt 8.

#include "mikroptim/synth/beam_search.hpp"

#include "mikroptim/optim/variant_generator.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mko {

const char* to_string(BeamNodeStatus status) {
    switch (status) {
    case BeamNodeStatus::Kept:
        return "kept";
    case BeamNodeStatus::Finished:
        return "finished";
    case BeamNodeStatus::Discarded:
        return "discarded";
    case BeamNodeStatus::Expanded:
        return "expanded";
    }
    return "?";
}

bool beam_rank_before(const BeamNode& a, const BeamNode& b) {
    const bool fa = std::isfinite(a.merit);
    const bool fb = std::isfinite(b.merit);
    if (fa != fb) {
        return fa;
    }
    if (a.acceptable != b.acceptable) {
        return a.acceptable;
    }
    if (a.acceptable) {
        if (a.elements != b.elements) {
            return a.elements < b.elements;
        }
    } else if (a.failed_rays != b.failed_rays) {
        return a.failed_rays < b.failed_rays;
    }
    if (fa && a.merit != b.merit) {
        return a.merit < b.merit;
    }
    return a.id < b.id;
}

namespace {

[[noreturn]] void invalid(const std::string& what) {
    throw std::invalid_argument("Strahlsuche: " + what);
}

std::string number(double v, const char* fmt = "%.6g") {
    char buf[64];
    std::snprintf(buf, sizeof buf, fmt, v);
    return buf;
}

// Abbruch (Status und Meldung), intern als Ausnahme durch die Phasen gereicht.
struct Stop {
    HomotopyStatus status;
    std::string message;
};

// Kandidat im Strahl: Nummer im Stammbaum, Basisbeschreibung am guten Zustand.
struct Cand {
    int id = 0;
    SystemDescription system;
};

// Kind eines Einfügens während der inneren Homotopie.
struct Child {
    Cand cand;
    int insertion = 0; // Index in der Liste der Einfügungen des Ereignisses
    int branches = 1;  // Zweige des Operators (für die Bemerkung)
};

// Kandidat, der einfügen soll: Bedarf und Grund (für die Meldung ohne Ausweg).
struct Request {
    Cand cand;
    HomotopyNeed need;
    std::string why;
};

using Clock = std::chrono::steady_clock;

void check_beam_options(const BeamOptions& o) {
    check_homotopy_options(o.homotopy);
    if (o.beam_width < 1 || o.alternatives < 1 || o.perturbed_starts < 0) {
        invalid("verlangt Strahlbreite ≥ 1, Varianten ≥ 1 und gestörte Starts ≥ 0");
    }
    if (std::isnan(o.dedup_tolerance)) {
        invalid("Toleranz der Deduplizierung ist NaN");
    }
}

class BeamController {
public:
    BeamController(HomotopyModel& model, const BeamOptions& options, BeamResult& result,
                   const std::function<void(const BeamStep&)>& on_step)
        : model_(model), o_(options), h_(options.homotopy), r_(result), on_step_(on_step),
          threshold_(homotopy_merit_threshold(options.homotopy)) {
        r_.merit_threshold = threshold_;
        r_.beam_width = o_.beam_width;
        r_.seed = o_.seed;
    }

    std::vector<Cand> beam;
    HomotopyPoint point;

    void run(SystemDescription system) {
        BeamNode root;
        root.id = 0;
        root.operation = "start";
        root.created = point;
        root.point = point;
        root.elements = model_.elements(system);
        r_.start_elements = root.elements;
        r_.nodes.push_back(root);
        beam.push_back(Cand{0, std::move(system)});
        start_phase();
        if (r_.target.aperture != point.aperture) {
            continuous_phase(HomotopyStepKind::Aperture, r_.target.aperture, h_.aperture_step);
        }
        if (r_.target.field != point.field) {
            continuous_phase(HomotopyStepKind::Field, r_.target.field, h_.field_step);
        }
        spectrum_phase();
        finish_phase();
    }

    BeamNode& node(int id) { return r_.nodes[static_cast<std::size_t>(id)]; }

    // Strahl nach Rang ordnen (Ende des Laufs).
    void sort_beam() {
        std::stable_sort(beam.begin(), beam.end(), [&](const Cand& a, const Cand& b) {
            return beam_rank_before(node(a.id), node(b.id));
        });
    }

private:
    HomotopyModel& model_;
    const BeamOptions& o_;
    const HomotopyOptions& h_;
    BeamResult& r_;
    const std::function<void(const BeamStep&)>& on_step_;
    double threshold_;
    std::uint64_t seed_counter_ = 0;

    void emit(BeamStep step) {
        step.step.index = static_cast<int>(r_.steps.size());
        r_.steps.push_back(std::move(step));
        if (on_step_) {
            on_step_(r_.steps.back());
        }
    }

    bool acceptable(const HomotopySolve& s) const {
        return std::isfinite(s.merit) && s.status != LmStatus::NumericalError &&
               s.failed_rays == 0 && s.wavefront + s.ghost_penalty <= threshold_;
    }

    bool fast(const HomotopySolve& s) const {
        return lm_converged(s.status) && s.iterations <= h_.fast_iterations;
    }

    // Letzte Optimierung eines Kandidaten im Stammbaum festhalten.
    void record(int id, const HomotopySolve& s, const HomotopyPoint& p, int elements) {
        BeamNode& n = node(id);
        n.point = p;
        n.solved = true;
        n.merit = s.merit;
        n.evaluation = s.wavefront + s.ghost_penalty;
        n.rms = s.rms;
        n.failed_rays = s.failed_rays;
        n.acceptable = acceptable(s);
        n.elements = elements;
    }

    void close(int id, BeamNodeStatus status, std::string reason) {
        BeamNode& n = node(id);
        n.status = status;
        n.reason = std::move(reason);
        n.final_step = static_cast<int>(r_.steps.size()) - 1;
    }

    // Gemeinsame Optimierung der Systeme am Punkt; ein Protokolleintrag pro Kandidat (noch
    // nicht ausgegeben, ohne Entscheidung).
    std::vector<BeamStep> solve(const std::vector<SystemDescription*>& systems,
                                const std::vector<int>& ids, const HomotopyPoint& p,
                                HomotopyStepKind kind) {
        const int n = static_cast<int>(systems.size());
        if (r_.solves + n > h_.max_steps) {
            throw Stop{HomotopyStatus::MaxSteps, "höchste Zahl von Optimierungen erreicht (" +
                                                     std::to_string(h_.max_steps) + ")"};
        }
        const auto t0 = Clock::now();
        std::vector<int> groups;
        const std::vector<HomotopySolve> solves = model_.solve_all(systems, p, &groups);
        const double seconds = std::chrono::duration<double>(Clock::now() - t0).count();
        if (solves.size() != systems.size() || groups.size() != systems.size()) {
            invalid("solve_all lieferte nicht ein Ergebnis pro System");
        }
        int used = 0;
        std::vector<BeamStep> out(systems.size());
        for (std::size_t i = 0; i < systems.size(); ++i) {
            BeamStep& b = out[i];
            b.candidate = ids[i];
            b.group = r_.groups + groups[i];
            used = std::max(used, groups[i] + 1);
            b.step.kind = kind;
            b.step.point = p;
            b.step.solve = solves[i];
            b.step.solved = true;
            b.step.seconds = seconds; // Laufzeit des gemeinsamen Aufrufs
            b.step.elements = model_.elements(*systems[i]);
            ++node(ids[i]).solves;
        }
        r_.groups += used;
        r_.solves += n;
        return out;
    }

    // Penalty-Fortsetzung der Kandidaten `cs` am Punkt (wie run_homotopy, alle Kandidaten
    // derselben Stufe gemeinsam); notes[i] steht vor der Bemerkung der ersten Stufe.
    void penalty_continuation(std::vector<Cand>& cs, HomotopyStepKind kind,
                              const std::vector<std::string>& notes) {
        std::vector<std::size_t> plain;
        std::vector<std::size_t> staged;
        for (std::size_t i = 0; i < cs.size(); ++i) {
            (model_.one_sided_constraints(cs[i].system) ? staged : plain).push_back(i);
        }
        const auto run = [&](const std::vector<std::size_t>& idx, HomotopyStepKind k, double scale,
                             bool first, bool with_scale) {
            if (idx.empty()) {
                return;
            }
            HomotopyPoint p = point;
            p.penalty_scale = scale;
            std::vector<SystemDescription*> sys;
            std::vector<int> ids;
            for (const std::size_t i : idx) {
                sys.push_back(&cs[i].system);
                ids.push_back(cs[i].id);
            }
            std::vector<BeamStep> steps = solve(sys, ids, p, k);
            for (std::size_t j = 0; j < idx.size(); ++j) {
                BeamStep& s = steps[j];
                s.step.accepted = true;
                const std::string& note = notes[idx[j]];
                if (with_scale) {
                    s.step.operation = (first && !note.empty() ? note + "; " : std::string()) +
                                       "Gewichte der einseitigen Bedingungen ×" + number(scale);
                } else {
                    s.step.operation = note;
                }
                record(s.candidate, s.step.solve, p, s.step.elements);
                emit(std::move(s));
            }
        };
        run(plain, kind, 1.0, true, false);
        for (int j = 0; j < h_.penalty_stages; ++j) {
            double scale = 1.0;
            for (int k = j; k < h_.penalty_stages - 1; ++k) {
                scale /= h_.penalty_factor;
            }
            run(staged, HomotopyStepKind::Penalty, scale, j == 0, true);
        }
    }

    // Bedarf aus der Diagnose: zuerst am abgelehnten Versuch, sonst am guten Zustand.
    HomotopyNeed need_for(const SystemDescription& good, const HomotopyPoint& good_point,
                          const SystemDescription& trial, const HomotopyPoint& trial_point) {
        HomotopyNeed n = model_.diagnose(trial, trial_point);
        if (n.recommendation != GhostOperator::None) {
            return n;
        }
        return model_.diagnose(good, good_point);
    }

    // Duplikat von einem der Kandidaten `kept` (besser platziert) am Punkt `p`? Dann verworfen.
    bool discard_duplicate(const Cand& c, const std::vector<Cand>& kept, const HomotopyPoint& p) {
        if (o_.dedup_tolerance < 0.0) {
            return false;
        }
        for (const Cand& k : kept) {
            const double d = model_.distance(k.system, c.system, p);
            if (d <= o_.dedup_tolerance) {
                close(c.id, BeamNodeStatus::Discarded,
                      "Duplikat von #" + std::to_string(k.id) + " (Abstand " + number(d, "%.3g") +
                          ")");
                return true;
            }
        }
        return false;
    }

    // Rangfolge und Auswahl der besten B (mit Deduplizierung) aus `pool` am Punkt `p`; die
    // übrigen werden verworfen. Rückgabe in Rangfolge.
    std::vector<Cand> select(std::vector<Cand> pool, const HomotopyPoint& p) {
        std::stable_sort(pool.begin(), pool.end(), [&](const Cand& a, const Cand& b) {
            return beam_rank_before(node(a.id), node(b.id));
        });
        std::vector<Cand> kept;
        for (Cand& c : pool) {
            if (static_cast<int>(kept.size()) >= o_.beam_width) {
                close(c.id, BeamNodeStatus::Discarded,
                      "Rang hinter den besten " + std::to_string(o_.beam_width));
                continue;
            }
            if (!discard_duplicate(c, kept, p)) {
                kept.push_back(std::move(c));
            }
        }
        return kept;
    }

    // Strahl nach Rang ordnen und Duplikate entfernen (nach jedem gemeinsamen Schritt: Zweige,
    // die erst später ins selbe Minimum laufen).
    void dedup_beam() {
        sort_beam();
        if (o_.dedup_tolerance < 0.0 || beam.size() < 2) {
            return;
        }
        std::vector<Cand> kept; // Kopien: Der Strahl bleibt bei einer Ausnahme des Modells gültig
        for (const Cand& c : beam) {
            if (!discard_duplicate(c, kept, point)) {
                kept.push_back(c);
            }
        }
        beam = std::move(kept);
    }

    // Einfügen bei den Kandidaten `requests` am Punkt `p` (ihr guter Zustand): Varianten des
    // Operators, Zweige, gestörte Starts, Vorfilter, innere Homotopie aller Kinder gemeinsam,
    // Auswahl der besten B aus `stay` und den Kindern, Abschluss und Penalty-Fortsetzung der
    // gewählten Kinder. Ohne Überlebende Stop mit dem Grund des ersten Kandidaten ohne Ausweg.
    // Der Strahl wird erst bei Erfolg ersetzt; danach steht er bei `p`.
    void expand(std::vector<Cand> stay, std::vector<Request> requests, const HomotopyPoint& p,
                HomotopyStatus no_need_status) {
        std::optional<Stop> first_stop;
        std::vector<HomotopyInsertion> insertions;
        std::vector<Child> children;
        for (Request& req : requests) {
            const int pid = req.cand.id;
            if (req.need.recommendation == GhostOperator::None) {
                Stop s{no_need_status, req.why + "; keine Geisterplatte aktiv (kein Bedarf)"};
                close(pid, BeamNodeStatus::Discarded, s.message);
                if (!first_stop) {
                    first_stop = std::move(s);
                }
                continue;
            }
            if (model_.elements(req.cand.system) >= h_.max_elements) {
                Stop s{HomotopyStatus::MaxElements,
                       req.why + "; Bedarf (Operator " +
                           std::string(to_string(req.need.recommendation)) +
                           "), aber höchste Elementzahl " + std::to_string(h_.max_elements) +
                           " erreicht"};
                close(pid, BeamNodeStatus::Discarded, s.message);
                if (!first_stop) {
                    first_stop = std::move(s);
                }
                continue;
            }
            const std::vector<HomotopyNeed> options =
                model_.insertion_options(req.cand.system, p, req.need, o_.alternatives);
            const std::size_t first_child = children.size();
            for (std::size_t k = 0; k < options.size(); ++k) {
                const auto t0 = Clock::now();
                HomotopyInsertion ins = model_.insert(req.cand.system, p, options[k]);
                if (ins.branches.empty()) {
                    invalid("das Modell lieferte keinen Zweig");
                }
                BeamStep step;
                step.candidate = pid;
                step.step.kind = HomotopyStepKind::Insertion;
                step.step.point = p;
                step.step.recommendation = options[k].recommendation;
                step.step.operation =
                    ins.text + (options.size() > 1 ? " [Variante " + std::to_string(k) + " von " +
                                                         std::to_string(options.size()) + "]"
                                                   : std::string());
                step.step.accepted = true;
                step.step.elements = model_.elements(req.cand.system);
                step.step.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
                emit(std::move(step));
                const int created_step = static_cast<int>(r_.steps.size()) - 1;
                ++r_.insertions;
                const int ins_index = static_cast<int>(insertions.size());
                const int nbranches = static_cast<int>(ins.branches.size());
                for (int b = 0; b < nbranches; ++b) {
                    const SystemDescription& start = ins.branches[static_cast<std::size_t>(b)];
                    std::vector<SystemDescription> starts{start};
                    const std::uint64_t seed = variant_seed(o_.seed, seed_counter_++);
                    if (o_.perturbed_starts > 0) {
                        std::vector<SystemDescription> more =
                            model_.perturbed_starts(start, ins, p, o_.perturbed_starts, seed);
                        for (SystemDescription& m : more) {
                            starts.push_back(std::move(m));
                        }
                    }
                    for (std::size_t j = 0; j < starts.size(); ++j) {
                        BeamNode n;
                        n.id = static_cast<int>(r_.nodes.size());
                        n.parent = pid;
                        n.op = ins.op;
                        n.option = static_cast<int>(k);
                        n.branch = b;
                        n.start = static_cast<int>(j);
                        n.created = p;
                        n.point = p;
                        n.created_step = created_step;
                        n.elements = model_.elements(starts[j]);
                        n.operation = "Operator " + std::string(to_string(ins.op)) +
                                      (options.size() > 1 ? ", Variante " + std::to_string(k)
                                                          : std::string()) +
                                      ", Zweig " + std::to_string(b) +
                                      (j > 0 ? ", Start " + std::to_string(j) : std::string());
                        node(pid).children.push_back(n.id);
                        r_.nodes.push_back(n);
                        children.push_back(
                            Child{Cand{n.id, std::move(starts[j])}, ins_index, nbranches});
                    }
                }
                insertions.push_back(std::move(ins));
            }
            if (children.size() > first_child) {
                close(pid, BeamNodeStatus::Expanded,
                      std::to_string(children.size() - first_child) + " Kinder");
            } else {
                close(pid, BeamNodeStatus::Discarded, "Einfügen ohne Startpunkt");
            }
        }

        // Vorfilter der Starts jedes Einfügens
        if (o_.prefilter && !children.empty()) {
            std::vector<Child> kept;
            for (int e = 0; e < static_cast<int>(insertions.size()); ++e) {
                std::vector<const SystemDescription*> starts;
                std::vector<std::size_t> idx;
                for (std::size_t i = 0; i < children.size(); ++i) {
                    if (children[i].insertion == e) {
                        starts.push_back(&children[i].cand.system);
                        idx.push_back(i);
                    }
                }
                const std::vector<bool> keep = model_.prefilter(starts, p);
                if (keep.size() != starts.size()) {
                    invalid("prefilter lieferte nicht eine Entscheidung pro Start");
                }
                for (std::size_t j = 0; j < idx.size(); ++j) {
                    if (!keep[j]) {
                        close(children[idx[j]].cand.id, BeamNodeStatus::Discarded,
                              "Vorfilter (Grobbewertung der Starts)");
                    }
                }
            }
            for (Child& c : children) {
                if (node(c.cand.id).status == BeamNodeStatus::Kept) {
                    kept.push_back(std::move(c));
                }
            }
            children = std::move(kept);
        }

        // Innere Homotopie: Stufe j aller Kinder gemeinsam
        std::size_t stages = 0;
        for (const Child& c : children) {
            stages = std::max(
                stages, std::max<std::size_t>(
                            1, insertions[static_cast<std::size_t>(c.insertion)].stages.size()));
        }
        for (std::size_t j = 0; j < stages; ++j) {
            std::vector<SystemDescription*> sys;
            std::vector<int> ids;
            std::vector<double> values;
            std::vector<int> branch;
            for (Child& c : children) {
                const HomotopyInsertion& ins = insertions[static_cast<std::size_t>(c.insertion)];
                const std::size_t count = std::max<std::size_t>(1, ins.stages.size());
                if (j >= count) {
                    continue;
                }
                double v = 0.0;
                if (!ins.stages.empty()) {
                    v = ins.stages[j];
                    model_.set_stage(c.cand.system, ins, v);
                }
                sys.push_back(&c.cand.system);
                ids.push_back(c.cand.id);
                values.push_back(v);
                branch.push_back(node(c.cand.id).branch);
            }
            std::vector<BeamStep> steps = solve(sys, ids, p, HomotopyStepKind::InsertionStage);
            for (std::size_t i = 0; i < steps.size(); ++i) {
                BeamStep& s = steps[i];
                s.step.branch = branch[i];
                s.step.stage = values[i];
                s.step.accepted = true;
                record(s.candidate, s.step.solve, p, s.step.elements);
                emit(std::move(s));
            }
        }

        // Auswahl aus den übrigen Kandidaten und den Kindern
        std::vector<Cand> pool = std::move(stay);
        std::vector<int> child_insertion(r_.nodes.size(), -1);
        std::vector<int> child_branches(r_.nodes.size(), 1);
        for (Child& c : children) {
            child_insertion[static_cast<std::size_t>(c.cand.id)] = c.insertion;
            child_branches[static_cast<std::size_t>(c.cand.id)] = c.branches;
            pool.push_back(std::move(c.cand));
        }
        if (pool.empty()) {
            throw first_stop ? *first_stop
                             : Stop{no_need_status, "keine Kandidaten nach dem Einfügen"};
        }
        std::vector<Cand> kept = select(std::move(pool), p);
        // Abschluss der gewählten Kinder und Penalty-Fortsetzung
        std::vector<Cand> fresh;
        std::vector<std::string> notes;
        std::vector<Cand> old;
        std::vector<int> order; // Position im Rang: ≥ 0 alt, < 0 neu (−1 − Index in fresh)
        for (Cand& c : kept) {
            const int e = child_insertion[static_cast<std::size_t>(c.id)];
            if (e < 0) {
                order.push_back(static_cast<int>(old.size()));
                old.push_back(std::move(c));
                continue;
            }
            const BeamNode& n = node(c.id);
            model_.finish_insertion(c.system, insertions[static_cast<std::size_t>(e)]);
            node(c.id).elements = model_.elements(c.system);
            std::string note = "Zweig " + std::to_string(n.branch) + " von " +
                               std::to_string(child_branches[static_cast<std::size_t>(c.id)]) +
                               " übernommen";
            if (o_.alternatives > 1 || o_.perturbed_starts > 0 || o_.beam_width > 1) {
                note = "Kandidat #" + std::to_string(c.id) + " (" + n.operation + ") übernommen";
            }
            notes.push_back(std::move(note));
            order.push_back(-1 - static_cast<int>(fresh.size()));
            fresh.push_back(std::move(c));
        }
        const HomotopyPoint before = point;
        point = p;
        try {
            penalty_continuation(fresh, HomotopyStepKind::Start, notes);
        } catch (const Stop&) {
            point = before; // Abbruch: Strahl und Punkt wie vor dem Einfügen
            throw;
        }
        std::vector<Cand> next;
        for (const int k : order) {
            next.push_back(k >= 0 ? std::move(old[static_cast<std::size_t>(k)])
                                  : std::move(fresh[static_cast<std::size_t>(-1 - k)]));
        }
        beam = std::move(next);
        dedup_beam();
    }

    // Startpunkt: Penalty-Fortsetzung; nicht annehmbare Kandidaten fügen ein, solange es Bedarf
    // gibt.
    void start_phase() {
        penalty_continuation(beam, HomotopyStepKind::Start, std::vector<std::string>(1));
        for (;;) {
            // Kopien: Bei einem Abbruch bleibt der Strahl unverändert.
            std::vector<Cand> stay;
            std::vector<Request> requests;
            for (const Cand& c : beam) {
                const BeamNode& n = node(c.id);
                if (n.acceptable) {
                    stay.push_back(c);
                    continue;
                }
                HomotopyNeed need = model_.diagnose(c.system, point);
                const std::string why =
                    "Startpunkt nicht annehmbar (F_B = " + number(n.evaluation) +
                    " > F_max = " + number(threshold_) + ")";
                requests.push_back(Request{c, std::move(need), why});
            }
            if (requests.empty()) {
                return;
            }
            expand(std::move(stay), std::move(requests), point, HomotopyStatus::StartRejected);
        }
    }

    static double& parameter(HomotopyPoint& p, HomotopyStepKind kind) {
        return kind == HomotopyStepKind::Aperture ? p.aperture : p.field;
    }

    // Kandidaten, die einen Versuch nicht geschafft haben, fügen ein (am guten Zustand), die
    // übrigen bleiben am guten Zustand stehen (Gleichschritt). `trials` sind die Versuche in der
    // Reihenfolge des Strahls, `steps` ihre Protokolleinträge (werden hier ausgegeben).
    void insert_rejected(std::vector<SystemDescription>& trials, const HomotopyPoint& tp,
                         std::vector<BeamStep>& steps, const std::string& why_prefix,
                         bool with_delta) {
        std::vector<Cand> stay;
        std::vector<Request> requests;
        for (std::size_t i = 0; i < beam.size(); ++i) {
            BeamStep& s = steps[i];
            if (acceptable(s.step.solve)) {
                s.step.accepted = false;
                s.step.action = HomotopyAction::None;
                s.step.operation = "annehmbar, bleibt am guten Zustand (Gleichschritt: andere "
                                   "Kandidaten fügen ein)";
                emit(std::move(s));
                stay.push_back(beam[i]);
                continue;
            }
            HomotopyNeed need = need_for(beam[i].system, point, trials[i], tp);
            s.step.recommendation = need.recommendation;
            s.step.action = need.recommendation == GhostOperator::None ||
                                    model_.elements(beam[i].system) >= h_.max_elements
                                ? HomotopyAction::Stop
                                : HomotopyAction::Insert;
            s.step.operation = need.text;
            std::string why = why_prefix + (with_delta ? " (letzter Versuch F_B = " : " (F_B = ") +
                              number(s.step.solve.wavefront + s.step.solve.ghost_penalty) + ")";
            emit(std::move(s));
            requests.push_back(Request{beam[i], std::move(need), std::move(why)});
        }
        // Kopien: Bei einem Abbruch bleibt der Strahl unverändert.
        expand(std::move(stay), std::move(requests), point, HomotopyStatus::StepTooSmall);
    }

    // Stetiger Parameter (Apertur, Feld) mit gemeinsamer Schrittweitensteuerung.
    void continuous_phase(HomotopyStepKind kind, double target, const HomotopyStepControl& c) {
        double delta = c.initial;
        const char* name = kind == HomotopyStepKind::Aperture ? "Apertur" : "Feldanteil";
        while (parameter(point, kind) != target) {
            HomotopyPoint tp = point;
            const double from = parameter(point, kind);
            double& value = parameter(tp, kind);
            value = target > from ? std::min(from + delta, target) : std::max(from - delta, target);
            if (std::abs(target - value) <= 1e-6 * c.min) {
                value = target;
            }
            std::vector<SystemDescription> trials;
            for (const Cand& b : beam) {
                trials.push_back(b.system);
            }
            std::vector<SystemDescription*> sys;
            std::vector<int> ids;
            for (std::size_t i = 0; i < beam.size(); ++i) {
                sys.push_back(&trials[i]);
                ids.push_back(beam[i].id);
            }
            std::vector<BeamStep> steps = solve(sys, ids, tp, kind);
            bool all = true;
            bool all_fast = true;
            for (BeamStep& s : steps) {
                s.step.delta = delta;
                all = all && acceptable(s.step.solve);
                all_fast = all_fast && fast(s.step.solve);
            }
            if (all) {
                if (all_fast) {
                    delta = std::min(delta * c.grow, c.max);
                }
                for (std::size_t i = 0; i < steps.size(); ++i) {
                    BeamStep& s = steps[i];
                    s.step.accepted = true;
                    s.step.action = all_fast ? HomotopyAction::Grow : HomotopyAction::Keep;
                    s.step.next_delta = delta;
                    record(s.candidate, s.step.solve, tp, s.step.elements);
                    emit(std::move(s));
                    beam[i].system = std::move(trials[i]);
                }
                point = tp;
                dedup_beam();
                continue;
            }
            delta *= c.shrink;
            if (delta >= c.min) {
                for (BeamStep& s : steps) {
                    s.step.next_delta = delta;
                    s.step.action = HomotopyAction::Shrink;
                    if (acceptable(s.step.solve)) {
                        s.step.operation = "annehmbar, verworfen (Gleichschritt: anderer "
                                           "Kandidat abgelehnt)";
                    }
                    emit(std::move(s));
                }
                continue;
            }
            for (BeamStep& s : steps) {
                s.step.next_delta = delta;
            }
            insert_rejected(trials, tp, steps,
                            std::string(name) + " " + number(from) +
                                ": Schrittweite unter Δ_min = " + number(c.min),
                            true);
            delta = c.initial;
        }
    }

    // Spektrumstufen: jede Stufe ein gemeinsamer Schritt; abgelehnte Kandidaten fügen ein.
    void spectrum_phase() {
        while (point.spectrum < r_.target.spectrum) {
            HomotopyPoint tp = point;
            ++tp.spectrum;
            std::vector<SystemDescription> trials;
            for (const Cand& b : beam) {
                trials.push_back(b.system);
            }
            std::vector<SystemDescription*> sys;
            std::vector<int> ids;
            for (std::size_t i = 0; i < beam.size(); ++i) {
                sys.push_back(&trials[i]);
                ids.push_back(beam[i].id);
            }
            std::vector<BeamStep> steps = solve(sys, ids, tp, HomotopyStepKind::Spectrum);
            bool all = true;
            for (const BeamStep& s : steps) {
                all = all && acceptable(s.step.solve);
            }
            if (all) {
                for (std::size_t i = 0; i < steps.size(); ++i) {
                    BeamStep& s = steps[i];
                    s.step.accepted = true;
                    s.step.action = HomotopyAction::Keep;
                    record(s.candidate, s.step.solve, tp, s.step.elements);
                    emit(std::move(s));
                    beam[i].system = std::move(trials[i]);
                }
                point = tp;
                dedup_beam();
                continue;
            }
            insert_rejected(trials, tp, steps,
                            "Spektrumstufe " + std::to_string(tp.spectrum) + " nicht annehmbar",
                            false);
        }
    }

    // Zielprüfung aller Kandidaten; true, wenn mindestens einer das Ziel erreicht.
    bool check_goals() {
        bool any = false;
        for (const Cand& c : beam) {
            BeamStep step;
            step.candidate = c.id;
            step.step.kind = HomotopyStepKind::Goal;
            step.step.point = point;
            const auto t0 = Clock::now();
            step.step.goal = model_.goal(c.system, point, h_.target_rms);
            step.step.goal_checked = true;
            step.step.accepted = step.step.goal.reached;
            step.step.elements = model_.elements(c.system);
            step.step.operation =
                std::string(step.step.goal.reached ? "Ziel erreicht" : "Ziel verfehlt") +
                ": größter RMS " + number(step.step.goal.max_rms, "%.5f") + " Wellen (Ziel " +
                number(h_.target_rms, "%.5f") + ")";
            step.step.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
            BeamNode& n = node(c.id);
            n.goal = step.step.goal;
            n.goal_checked = true;
            any = any || step.step.goal.reached;
            emit(std::move(step));
        }
        return any;
    }

    // Alle Kandidaten gemeinsam am Punkt optimieren (Fortsetzungsstufe).
    void solve_beam(HomotopyStepKind kind, const std::string& operation) {
        std::vector<SystemDescription*> sys;
        std::vector<int> ids;
        for (Cand& c : beam) {
            sys.push_back(&c.system);
            ids.push_back(c.id);
        }
        std::vector<BeamStep> steps = solve(sys, ids, point, kind);
        for (BeamStep& s : steps) {
            s.step.accepted = true;
            s.step.operation = operation;
            record(s.candidate, s.step.solve, point, s.step.elements);
            emit(std::move(s));
        }
    }

    // Abschluss am Endpunkt: ε-Fortsetzung, Ziel; verfehlt: μ-Fortsetzung, danach Einfügen am
    // Zustand vor dem Abschluss und erneut. Endet, sobald ein Kandidat das Ziel erreicht.
    void finish_phase() {
        for (;;) {
            // β-Fortsetzung der Strahl-Extrema (M11-T1b), zuletzt hart
            if (point.extrema_beta > 0.0) {
                for (int j = 0; j <= h_.extrema_stages; ++j) {
                    point.extrema_beta =
                        j < h_.extrema_stages ? point.extrema_beta * h_.extrema_factor : 0.0;
                    solve_beam(HomotopyStepKind::ExtremaSmoothing,
                               point.extrema_beta > 0.0 ? "β = " + number(point.extrema_beta)
                                                        : std::string("Extrema hart"));
                }
            }
            const std::vector<Cand> saved = beam;
            const HomotopyPoint saved_point = point;
            for (int j = 0; j < h_.smoothing_stages; ++j) {
                point.ghost_smoothing *= h_.smoothing_factor;
                solve_beam(HomotopyStepKind::Smoothing, "ε ×" + number(point.ghost_smoothing));
            }
            if (check_goals()) {
                return;
            }
            for (int j = 0; j < h_.ghost_weight_stages; ++j) {
                point.ghost_weight *= h_.ghost_weight_factor;
                solve_beam(HomotopyStepKind::GhostWeight, "μ ×" + number(point.ghost_weight));
                if (check_goals()) {
                    return;
                }
            }
            // Ziel verfehlt: Einfügen am Zustand vor dem Abschluss
            std::vector<Request> requests;
            for (const Cand& c : saved) {
                HomotopyNeed need = model_.diagnose(c.system, saved_point);
                for (auto it = r_.steps.rbegin(); it != r_.steps.rend(); ++it) {
                    if (it->candidate == c.id && it->step.kind == HomotopyStepKind::Goal) {
                        it->step.recommendation = need.recommendation;
                        break;
                    }
                }
                const std::string why = "Ziel am Endpunkt verfehlt (größter RMS " +
                                        number(node(c.id).goal.max_rms, "%.5f") + " Wellen)";
                requests.push_back(Request{c, std::move(need), why});
            }
            expand({}, std::move(requests), saved_point, HomotopyStatus::GoalNotReached);
        }
    }
};

} // namespace

BeamResult run_beam_search(HomotopyModel& model, SystemDescription system,
                           const HomotopyPoint& start, const BeamOptions& options,
                           const std::function<void(const BeamStep&)>& on_step) {
    check_beam_options(options);
    BeamResult r;
    r.target = homotopy_target(start, options.homotopy);
    const auto t0 = Clock::now();
    BeamController c(model, options, r, on_step);
    c.point = start;
    bool reached = false;
    try {
        c.run(std::move(system));
        reached = true;
    } catch (const Stop& stop) {
        r.status = stop.status;
        r.message = stop.message;
    }
    // Ergebnis: bester Kandidat (am Ziel: der beste unter denen, die es erreicht haben)
    c.sort_beam();
    int best = -1;
    for (const Cand& k : c.beam) {
        BeamNode& n = c.node(k.id);
        if (reached && n.goal_checked && n.goal.reached) {
            n.status = BeamNodeStatus::Finished;
            n.reason = "Ziel erreicht";
            if (best < 0) {
                best = k.id;
            }
        } else {
            // im Strahl am Ende (nach einem Abbruch wiederhergestellt): behalten, mit dem Grund
            // des Abbruchs, falls der Kandidat ihn ausgelöst hat
            if (n.status == BeamNodeStatus::Expanded) {
                n.reason = "Abbruch während des Einfügens";
            } else if (n.status == BeamNodeStatus::Kept) {
                n.reason = n.goal_checked ? "Ziel verfehlt" : "Lauf beendet";
            }
            n.status = BeamNodeStatus::Kept;
        }
        r.beam.push_back(k.id);
    }
    if (reached) {
        r.status = HomotopyStatus::GoalReached;
        r.message = "Ziel erreicht";
    }
    if (best < 0 && !c.beam.empty()) {
        best = c.beam.front().id;
    }
    r.best = best;
    for (const Cand& k : c.beam) {
        if (k.id == best) {
            r.system = k.system;
        }
    }
    r.point = c.point;
    if (best >= 0) {
        const BeamNode& n = c.node(best);
        r.goal = n.goal_checked ? n.goal : HomotopyGoal{};
        r.elements = model.elements(r.system);
    }
    r.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    return r;
}

// ---------------------------------------------------------------------------------------------
// Protokoll als Text, DOT und JSON
// ---------------------------------------------------------------------------------------------

namespace {

using Json = nlohmann::ordered_json;

Json num(double v) {
    return std::isfinite(v) ? Json(v) : Json(nullptr);
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

Json point_json(const HomotopyPoint& p) {
    return Json{{"aperture", num(p.aperture)},
                {"field", num(p.field)},
                {"spectrum", p.spectrum},
                {"ghost_weight", num(p.ghost_weight)},
                {"ghost_smoothing", num(p.ghost_smoothing)},
                {"extrema_beta", num(p.extrema_beta)},
                {"spectrum_width", num(p.spectrum_width)},
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

Json node_json(const BeamNode& n) {
    return Json{{"id", n.id},
                {"parent", n.parent >= 0 ? Json(n.parent) : Json(nullptr)},
                {"children", n.children},
                {"operation", n.operation},
                {"operator", to_string(n.op)},
                {"option", n.option >= 0 ? Json(n.option) : Json(nullptr)},
                {"branch", n.branch >= 0 ? Json(n.branch) : Json(nullptr)},
                {"start", n.start},
                {"created", point_json(n.created)},
                {"created_step", n.created_step >= 0 ? Json(n.created_step) : Json(nullptr)},
                {"point", point_json(n.point)},
                {"merit", n.solved ? num(n.merit) : Json(nullptr)},
                {"evaluation", n.solved ? num(n.evaluation) : Json(nullptr)},
                {"rms_waves", n.solved ? num(n.rms) : Json(nullptr)},
                {"failed_rays", n.failed_rays},
                {"acceptable", n.acceptable},
                {"elements", n.elements},
                {"solves", n.solves},
                {"goal", n.goal_checked ? goal_json(n.goal) : Json(nullptr)},
                {"status", to_string(n.status)},
                {"reason", n.reason},
                {"final_step", n.final_step >= 0 ? Json(n.final_step) : Json(nullptr)}};
}

std::string point_text(const HomotopyPoint& p) {
    return "NA " + fmt("%.4f", p.aperture) + ", Feld " + fmt("%.3f", p.field) + ", Sp " +
           std::to_string(p.spectrum);
}

const char* status_text(BeamNodeStatus s) {
    switch (s) {
    case BeamNodeStatus::Kept:
        return "behalten";
    case BeamNodeStatus::Finished:
        return "fertig";
    case BeamNodeStatus::Discarded:
        return "verworfen";
    case BeamNodeStatus::Expanded:
        return "expandiert";
    }
    return "?";
}

void tree_lines(const BeamResult& r, int id, const std::string& prefix, bool last, bool root,
                std::ostream& out) {
    const BeamNode& n = r.nodes[static_cast<std::size_t>(id)];
    out << prefix;
    if (!root) {
        out << (last ? "`-- " : "|-- ");
    }
    out << "#" << n.id << " " << n.operation << " @ " << point_text(n.created) << ": ";
    if (n.solved) {
        out << "F " << fmt("%.4e", n.merit) << ", F_B " << fmt("%.4e", n.evaluation) << " @ "
            << point_text(n.point);
    } else {
        out << "nicht optimiert";
    }
    out << ", El " << n.elements << ", " << n.solves << " Opt. [" << status_text(n.status);
    if (!n.reason.empty()) {
        out << ": " << n.reason;
    }
    if (n.id == r.best) {
        out << "; Ergebnis";
    }
    out << "]\n";
    const std::string child_prefix = root ? prefix : prefix + (last ? "    " : "|   ");
    for (std::size_t i = 0; i < n.children.size(); ++i) {
        tree_lines(r, n.children[i], child_prefix, i + 1 == n.children.size(), false, out);
    }
}

std::string dot_escape(const std::string& s) {
    std::string o;
    for (const char c : s) {
        if (c == '"' || c == '\\') {
            o += '\\';
        }
        o += c;
    }
    return o;
}

} // namespace

void write_beam_step_header(std::ostream& out) {
    out << "Kand  Gr    ";
    write_homotopy_step_header(out);
}

void write_beam_step_line(const BeamStep& s, std::ostream& out) {
    out << pad("#" + std::to_string(s.candidate), 4) << "  "
        << (s.group >= 0 ? pad(std::to_string(s.group), 4) : std::string("   -")) << "  ";
    write_homotopy_step_line(s.step, out);
}

void write_beam_summary(const BeamResult& r, std::ostream& out) {
    out << "Status: " << to_string(r.status) << " (" << r.message << ")\n";
    out << "Strahlsuche: B = " << r.beam_width << ", Seed " << r.seed << ", " << r.nodes.size()
        << " Kandidaten, " << r.insertions << " Einfügungen, " << r.solves << " Optimierungen in "
        << r.groups << " Aufrufen\n";
    out << "Ergebnis: Kandidat #" << r.best << ", Strahl am Ende:";
    for (const int id : r.beam) {
        out << " #" << id;
    }
    out << '\n';
    out << "Endpunkt: NA " << fmt("%.4f", r.point.aperture) << ", Feldanteil "
        << fmt("%.3f", r.point.field) << ", Spektrumstufe " << r.point.spectrum << ", mu x"
        << fmt("%.4g", r.point.ghost_weight) << ", eps x" << fmt("%.4g", r.point.ghost_smoothing)
        << " (Ziel NA " << fmt("%.4f", r.target.aperture) << ")\n";
    out << "Glaselemente: " << r.start_elements << " -> " << r.elements
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

void write_beam_tree(const BeamResult& r, std::ostream& out) {
    out << "Stammbaum (Kandidat, Herkunft @ Punkt der Entstehung: letzte Merit @ Punkt, "
           "Elemente, Optimierungen [Status]):\n";
    if (!r.nodes.empty()) {
        tree_lines(r, 0, "", true, true, out);
    }
}

void write_beam_dot(const BeamResult& r, std::ostream& out) {
    out << "digraph stammbaum {\n  rankdir=TB;\n  node [shape=box, fontname=\"Helvetica\"];\n";
    for (const BeamNode& n : r.nodes) {
        // Zeilen einzeln maskieren, dann mit dem DOT-Zeilenumbruch \n verbinden
        const std::string label = dot_escape("#" + std::to_string(n.id) + " " + n.operation) +
                                  "\\n" + dot_escape(point_text(n.created)) + "\\n" +
                                  dot_escape("El " + std::to_string(n.elements) + ", F " +
                                             (n.solved ? fmt("%.3e", n.merit) : std::string("-"))) +
                                  "\\n" + dot_escape(status_text(n.status));
        const char* color = n.status == BeamNodeStatus::Finished    ? "palegreen"
                            : n.status == BeamNodeStatus::Discarded ? "lightgray"
                            : n.status == BeamNodeStatus::Expanded  ? "lightblue"
                                                                    : "khaki";
        out << "  n" << n.id << " [label=\"" << label << "\", style=filled, fillcolor=" << color
            << (n.id == r.best ? ", penwidth=3" : "") << "];\n";
    }
    for (const BeamNode& n : r.nodes) {
        if (n.parent >= 0) {
            out << "  n" << n.parent << " -> n" << n.id << ";\n";
        }
    }
    out << "}\n";
}

std::string beam_result_json(const BeamResult& r, int indent) {
    Json steps = Json::array();
    for (const BeamStep& s : r.steps) {
        Json j = Json::parse(homotopy_step_json(s.step));
        Json o{{"candidate", s.candidate}, {"group", s.group >= 0 ? Json(s.group) : Json(nullptr)}};
        for (auto it = j.begin(); it != j.end(); ++it) {
            o[it.key()] = it.value();
        }
        steps.push_back(std::move(o));
    }
    Json nodes = Json::array();
    for (const BeamNode& n : r.nodes) {
        nodes.push_back(node_json(n));
    }
    Json j{{"status", to_string(r.status)},
           {"message", r.message},
           {"start_point", r.nodes.empty() ? Json(nullptr) : point_json(r.nodes.front().created)},
           {"target", point_json(r.target)},
           {"point", point_json(r.point)},
           {"merit_threshold", num(r.merit_threshold)},
           {"start_elements", r.start_elements},
           {"elements", r.elements},
           {"insertions", r.insertions},
           {"solves", r.solves},
           {"goal", goal_json(r.goal)},
           {"beam_search", Json{{"beam_width", r.beam_width},
                                {"seed", r.seed},
                                {"best", r.best >= 0 ? Json(r.best) : Json(nullptr)},
                                {"beam", r.beam},
                                {"groups", r.groups},
                                {"candidates", nodes}}},
           {"seconds", num(r.seconds)},
           {"steps", steps}};
    return j.dump(indent);
}

} // namespace mko
