# ADR-0005: Gemeinsamer Mathe-Kern für CPU und GPU

- **Status:** angenommen
- **Datum:** 2026-10-18

## Kontext
CPU-Referenz und GPU-Backend sollen dieselbe Physik rechnen; doppelte Implementierungen laufen auseinander.

## Entscheidung
Alle Pro-Strahl-Rechnungen sind header-only, `MKO_HD`, ohne Ausnahmen, Allokation, STL-Container und virtuelle Funktionen. CPU-Tracer und Kernel rufen dieselben Funktionen.

## Konsequenzen
Der CPU/GPU-Vergleich prüft Compiler- und Ausführungsunterschiede, nicht die Physik. Die Physik wird daher durch **unabhängige** Orakel geprüft: analytische Lösungen (Planplatte, exakter trigonometrischer Strahl, aplanatische Punkte), geschlossene Paraxialformeln, und finite Differenzen gegen Adjungierte.
