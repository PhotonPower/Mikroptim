#pragma once

// Unterbefehl `mikroptim trace` (docs/cli.md): Paraxialdaten, RMS-Wellenfront pro Feld und
// Wellenlänge und versagte Strahlen eines Systems, als Text oder als JSON `mikroptim.trace/1`.
// Interner Header der Bibliothek mikroptim_cli (gelinkt in das Programm und die Tests).

#include <ostream>
#include <span>
#include <string_view>

namespace mko::cli {

// Formatkennung der JSON-Ausgabe.
inline constexpr std::string_view kTraceFormat = "mikroptim.trace/1";

// Exit-Codes von run_trace_command (und damit von `mikroptim trace`).
inline constexpr int kExitOk = 0;    // Auswertung ausgegeben (auch ohne auswertbare Wellenfront)
inline constexpr int kExitError = 1; // Datei nicht lesbar, Formatfehler, Glasfehler, ...
inline constexpr int kExitUsage = 2; // ungültige Kommandozeile

// Hilfetext des Unterbefehls.
void print_trace_usage(std::ostream& out);

// Führt `mikroptim trace <args...>` aus; `args` sind die Argumente nach "trace".
// Ergebnis auf `out`, Warnungen und Fehlermeldungen auf `err`. Wirft keine Ausnahmen aus dem
// Laden oder Auswerten: Fehler werden als Meldung auf `err` mit Exit-Code kExitError gemeldet.
int run_trace_command(std::span<const std::string_view> args, std::ostream& out, std::ostream& err);

} // namespace mko::cli
