# ADR-0013: Glattes Maximum im Kernel; keine Bitgleichheit bei Verteilung auf mehrere GPUs

- **Status:** angenommen; ändert ADR-0007 (Satz zur Verteilung auf mehrere GPUs)
- **Datum:** 2026-10-28

## Kontext

Die trace-abhängigen Nebenbedingungen (größte Strahlhöhe, Randdicke bei der freien Öffnung, Einfallswinkel) beruhen auf Maxima über Strahlen und Flächen. Sie sind nur stückweise glatt; der Levenberg-Marquardt kriecht an den Knicken und endet in M6-T3 und M6-T6 fast immer mit `max_iterations` (Befund F10). Ein glattes Maximum (Log-Sum-Exp) behebt das, braucht aber `exp` und `log`.

Diese Funktionen sind in CUDA und in den C-Bibliotheken nicht korrekt gerundet. Ergebnisse sind damit zwischen CPU und GPU und zwischen sm_60 und sm_86 nicht bitgleich. ADR-0007 sagt bisher: „die Verteilung auf mehrere GPUs ändert Ergebnisse pro Variante nicht“; M7-T3 hat das als Abnahme.

Erwogen wurden (docs/optimierung-erweiterungen.md, Abschnitt 1):

- A: Log-Sum-Exp im Kernel mit den Bibliotheksfunktionen,
- B: exakte Top-K-Reduktion im Kernel, Log-Sum-Exp auf dem Host,
- C: eigene `exp`/`log` als `MKO_HD` aus Grundoperationen.

## Entscheidung

1. **Variante A.** Pro Fläche wird neben dem harten Maximum (bzw. kleinsten cos I) im Kernel das glatte Maximum m + (1/β)·ln Σ e^{β(h − m)} reduziert, mit dem harten Maximum m als Verschiebung. Die Reduktion ist eine deterministische Baumreduktion ohne Gleitkomma-Atomics (ADR-0007 bleibt dafür gültig).
2. **Rechentyp der glatten Reduktion:** FP64, auch wenn der Trace in df64 läuft. Die Reduktion ist nicht OPD-kritisch.
3. **β** hat die Einheit 1/Einheit der Größe und ist Homotopie-Parameter. Der Endwert erfüllt ln N/β ≤ Vorhalt (Standard 5·10⁻⁵ mm), damit die Vorhalt-Regeln des Homotopie-Controllers gültig bleiben.
4. **Schalter:** Das harte Maximum bleibt Standard, bis Messungen an den Abnahmesystemen von M4-T3 und M6-T3 den Wechsel begründen.
5. **Bitgleichheit bei Verteilung auf mehrere GPUs ist keine Anforderung mehr.** Das gilt allgemein, nicht nur für das glatte Maximum. Verbindlich bleiben:
   - bitgleiche Ergebnisse bei gleichen Eingaben auf demselben Gerät, unabhängig von Tranchen und Batchgröße (ADR-0007),
   - Übereinstimmung zwischen CPU und GPU sowie zwischen Geräten innerhalb dokumentierter Toleranzen aus `tests/tolerances.hpp`.
6. Neue Toleranz `kSmoothMaxCpuGpuRelative` für das glatte Maximum; der Wert wird bei der Umsetzung (M11-T1) gemessen und begründet.

## Konsequenzen

- LM-Konvergenz an aktiven trace-abhängigen Bedingungen soll sich verbessern (zu messen in M11-T1).
- Läufe auf mehreren GPUs dürfen sich von Einzelläufen in den letzten Bits unterscheiden. Über viele Einfügungen kann ein Lauf deshalb einen anderen Weg nehmen (wie bei CPU gegen GPU schon beobachtet, M6-T6 F8). Reproduzierbar ist ein Lauf über dieselbe Gerätekonfiguration oder über das Protokoll.
- Die Tests `MultiTraceEngine.RealDevicesBitIdenticalToSingle` und `ResultFingerprint` sind keine Abnahme mehr. Sie bleiben als Information, solange sie für die bestehenden Kernel grün sind; neue Tests vergleichen mehrere GPUs gegen Toleranzen. Die Anpassung ist eine eigene Aufgabe.
- Die Kalibrierung der Anteile in `MultiTraceEngine` muss keine Rücksicht mehr auf Bitgleichheit nehmen; spätere Optimierungen pro Architektur (etwa Registergrenzen oder FMA-Verhalten nur für sm_60) sind erlaubt, solange die Toleranzen eingehalten werden.
- Mit eingeschaltetem glatten Maximum ändern sich die Werte der trace-abhängigen Strafen geringfügig; Ergebnisse sind nicht bitgleich zu früheren Läufen. Das Protokoll vermerkt den Schalter und β.
- Zusätzlicher Local Memory pro Thread für die zweite Reduktion; die Reserve in hardware.md ist zu prüfen.

## Verworfene Alternativen

- **B (Top-K exakt, Log-Sum-Exp auf dem Host):** hätte die Bitgleichheit erhalten, kostet aber K Werte pro Fläche und Thread im Local Memory und einen Fehler durch abgeschnittene Strahlen. Nach dem Verzicht auf Bitgleichheit über mehrere GPUs ohne Vorteil.
- **C (eigene exp/log):** Entwicklungs- und Testaufwand für Fehlerschranken ohne Nutzen, sobald Bitgleichheit nicht mehr verlangt ist.
- **Bitgleichheit über mehrere GPUs beibehalten und auf das glatte Maximum verzichten:** lässt F10 ungelöst.
