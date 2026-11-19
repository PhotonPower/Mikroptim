#pragma once

// Unterbefehl `mikroptim sensitivity` (docs/cli.md, M10-T3): Empfindlichkeit der Bildgüte für
// jede einzelne Toleranz (Toleranzsatz der Datei oder Mikroptim-Standardsatz: Dezentrierung und
// Verkippung pro Element, Radien, Dicken, Brechzahlen), direkt bei ±δ und aus der Jacobi-Matrix
// der Wellenfrontreste, dazu einfache Herstellbarkeitsprüfungen. Text oder JSON
// `mikroptim.sensitivity/1`; CPU oder mit --gpu über mko::cuda::TraceEngine.
// Interner Header der Bibliothek mikroptim_cli (gelinkt in das Programm und die Tests).

#include "cli/cli_util.hpp"

#include <ostream>
#include <span>
#include <string_view>

namespace mko::cli {

// Formatkennung der JSON-Ausgabe.
inline constexpr std::string_view kSensitivityFormat = "mikroptim.sensitivity/1";

// Hilfetext des Unterbefehls.
void print_sensitivity_usage(std::ostream& out);

// Führt `mikroptim sensitivity <args...>` aus; `args` sind die Argumente nach "sensitivity".
// Exit-Codes wie `mikroptim trace` (kExitOk, kExitError, kExitUsage, kExitNoGpu). Befunde der
// Herstellbarkeit ändern den Exit-Code nicht.
int run_sensitivity_command(std::span<const std::string_view> args, std::ostream& out,
                            std::ostream& err);

} // namespace mko::cli
