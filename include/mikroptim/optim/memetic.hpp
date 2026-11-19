#pragma once

// Memetische globale Suche in fester Topologie (M11-T7, Host): CMA-ES über normierte Variablen,
// jedes Individuum kurz mit dem gebündelten Levenberg-Marquardt verfeinert (alle Individuen einer
// Generation in einem Aufruf, auf der GPU ein Block pro Variante, ADR-0004), die Verteilung aus
// den verfeinerten Punkten fortgeschrieben (Lamarck). Optional sortiert der FP32-Vorfilter
// (M7-T2) hoffnungslose Starts vorab aus. Zum Vergleich derselbe Ablauf mit zufällig gestörten
// Starts ohne Lernen (perturbed_vector wie mko::test::perturbed_starts und `optimize --runs`).
// Budget, Normierung und Formeln: docs/optimierung.md, Abschnitt 15.

#include "mikroptim/batch/batch_prefilter.hpp"
#include "mikroptim/batch/batch_residuals.hpp"
#include "mikroptim/batch/system_batch.hpp"
#include "mikroptim/optim/levenberg_marquardt.hpp"
#include "mikroptim/optim/variable_set.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace mko {

enum class GlobalSearchMethod {
    Cma,             // CMA-ES mit LM pro Individuum (memetisch)
    PerturbedStarts, // gleichverteilt gestörte Starts um x₀ ohne Lernen (Vergleich)
};

// Name für Meldungen und die CLI ("cma", "starts").
const char* to_string(GlobalSearchMethod method);
// Umkehrung von to_string; leer bei unbekanntem Namen.
std::optional<GlobalSearchMethod> global_search_method_from_string(std::string_view name);

// Kriterien des Vorfilters: mehr als die Hälfte der Strahlen versagt oder Merit über dem
// Zehnfachen der besten Variante der Kandidaten (wie die Strahlsuche, M6-T4).
inline PrefilterOptions memetic_prefilter_defaults() {
    PrefilterOptions o;
    o.criteria.max_failed_fraction = 0.5;
    o.criteria.max_merit_factor = 10.0;
    return o;
}

struct MemeticOptions {
    GlobalSearchMethod method = GlobalSearchMethod::Cma;
    // Individuen pro Generation λ; 0: cma_default_lambda(P) = 4 + ⌊3 ln P⌋.
    int population = 0;
    // Höchstens so viele Generationen (≥ 1).
    int generations = 10;
    // LM-Zyklen pro Individuum (LmOptions::max_iterations des Aufrufs einer Generation, ≥ 1).
    int lm_iterations = 30;
    // Budget an LM-Zyklen über alle Individuen (Σ LmRunResult::iterations); 0: population ·
    // generations · lm_iterations. Eine Generation startet nur, wenn mindestens ein Zyklus pro
    // Individuum ins Restbudget passt, und bekommt höchstens ⌊Rest/λ⌋ Zyklen pro Individuum.
    std::int64_t cycle_budget = 0;
    // Startschrittweite σ₀ in normierten Koordinaten u_k = x_k/d_k, d_k = max(|x₀_k|, x_typ,k):
    // relative Störung wie PerturbationScale::relative. Bei PerturbedStarts die halbe Breite der
    // Gleichverteilung (y_k = x₀_k + σ₀ |x₀_k| ξ, ξ ∈ [−1, 1)).
    double sigma = 0.05;
    std::uint64_t seed = 0;
    // Lamarck (Standard): Verteilung aus den verfeinerten Punkten; sonst Baldwin (aus den Starts,
    // Rang nach dem verfeinerten Ergebnis).
    bool lamarckian = true;
    // Nur Lamarck: Pfad der Schrittweite p_σ aus den gezogenen Starts statt aus den verfeinerten
    // Punkten. Standard aus: Die verfeinerten Punkte liegen weit außerhalb von σ, die
    // Schrittweitenregel vergrößert σ dann fast jede Generation, und die Suche erkundet weiter
    // (am Mehrlinser F = 1,4·10⁻⁴ statt 1,8·10⁻²; docs/optimierung.md, Abschnitt 15).
    bool step_size_from_starts = false;
    // Individuum 0 der Generation 0 ist x₀ selbst (ungestört).
    bool include_start = true;
    // Nur CMA-ES: Kandidat 0 jeder Generation g ≥ 1 ist der verfeinerte Punkt des bisher besten
    // Individuums (Elitismus); er wird mit weiteren L_g Zyklen fortgesetzt.
    bool elitist = true;
    // FP32-Vorfilter (nur Vorfilterung, AGENTS.md §6.1): pro Generation oversampling · λ Kandidaten
    // ziehen, die ersten λ behaltenen in Kandidatenreihenfolge optimieren; fehlen welche, die
    // aussortierten mit der kleinsten unteren Schranke merit_lower (Gleichstand: Index).
    bool prefilter = false;
    int prefilter_oversampling = 2;
    PrefilterOptions prefilter_options = memetic_prefilter_defaults();
};

// Prüft die Einstellungen (Population 0 oder ≥ 2, Generationen ≥ 1, Zyklen ≥ 1, Budget ≥ 0,
// σ₀ endlich und > 0, Überabtastung ≥ 1, Vorfilteroptionen); wirft std::invalid_argument.
void check_memetic_options(const MemeticOptions& options);

// Gebündelter LM einer Generation (Standard optimize_lm_cpu; GPU z. B. cuda::optimize_lm_gpu).
using MemeticOptimizer =
    std::function<LmResult(const VariableSet& set, const SystemBatchHost& base,
                           std::span<const double> x0, const LmOptions& options)>;
// FP32-Vorfilter der Kandidaten (Standard prefilter_cpu; GPU TraceEngine::prefilter_variants),
// Signatur wie HomotopyPrefilterFn.
using MemeticPrefilterFn = std::function<PrefilterResult(
    const SystemBatchHost& batch, const ResidualSettings& quadrature,
    const PrefilterOptions& options, std::span<const double> extra_merit)>;

// Ein verfeinertes Individuum.
struct MemeticIndividual {
    int generation = 0;
    int candidate = 0;         // Index unter den Kandidaten der Generation
    std::vector<double> start; // Startpunkt des LM (zulässig)
    LmRunResult run;           // Ergebnis des LM
};

struct MemeticGeneration {
    int generation = 0;
    int individuals = 0;     // λ
    int candidates = 0;      // gezogene Kandidaten (mit Vorfilter oversampling · λ)
    int prefilter_kept = 0;  // vom Vorfilter behalten (ohne Vorfilter = candidates)
    int lm_iterations = 0;   // Zyklen pro Individuum dieser Generation
    std::int64_t cycles = 0; // verbrauchte LM-Zyklen
    double sigma = 0.0;      // σ beim Ziehen (CMA), sonst σ₀
    double best_merit = 0.0; // beste Merit der Generation (Auswahlregel memetic_better)
    int best = -1;           // Index des besten Individuums der Generation in `individuals`
};

struct MemeticResult {
    std::vector<MemeticIndividual> individuals; // Generation für Generation
    std::vector<MemeticGeneration> generations;
    int best = -1;           // bestes Individuum insgesamt (−1: keine endliche Merit)
    std::int64_t cycles = 0; // Σ LmRunResult::iterations aller Individuen
    std::int64_t budget = 0; // wirksames Budget
    int lambda = 0;

    const MemeticIndividual& best_individual() const {
        return individuals.at(static_cast<std::size_t>(best));
    }
};

// Auswahlregel (wie `mikroptim optimize`): endliche Merit vor nicht endlicher, dann weniger
// versagte Strahlen, dann kleinere Merit. true, wenn a echt besser als b ist.
bool memetic_better(const LmRunResult& a, const LmRunResult& b);

// Globale Suche ab x₀ (zulässig, Länge P). `base` hat genau eine Variante; `lm` liefert Quadratur,
// Zusatzresiduen, Strahlversagen und Abbruchkriterien des LM (max_iterations wird pro Generation
// gesetzt, on_iteration erhält den fortlaufenden Index des Individuums in `individuals`). Mit
// gleichen Eingaben ist das Ergebnis bitgleich (der LM ist bitgleich, CMA-ES deterministisch).
// Wirft std::invalid_argument bei ungültigen Argumenten.
MemeticResult memetic_search(const VariableSet& set, const SystemBatchHost& base,
                             std::span<const double> x0, const LmOptions& lm,
                             const MemeticOptions& options, const MemeticOptimizer& optimizer = {},
                             const MemeticPrefilterFn& prefilter = {});

} // namespace mko
