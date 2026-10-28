# ADR-0007: Deterministische Reduktionen

- **Status:** angenommen; Satz zur Verteilung auf mehrere GPUs geändert durch [ADR-0013](0013-glattes-maximum-und-mehrere-gpus.md)
- **Datum:** 2026-10-18

## Kontext
Gleitkomma-Atomics liefern je nach Ausführungsreihenfolge unterschiedliche Summen. Das macht Optimierungsläufe nicht reproduzierbar und Tests instabil.

## Entscheidung
Reduktionen für Merit, Residuen und Jacobi-Matrizen: Baumreduktion im Block, Ergebnis pro Block in den globalen Speicher, Summation in fester Reihenfolge. Gleiche Eingaben auf derselben GPU ergeben bitgleiche Ergebnisse; die Verteilung auf mehrere GPUs ändert Ergebnisse pro Variante nicht.

## Konsequenzen
Etwas mehr Code und ein zweiter Reduktionsschritt; vernachlässigbare Kosten.

## Änderung durch ADR-0013
Die Zusage „die Verteilung auf mehrere GPUs ändert Ergebnisse pro Variante nicht“ gilt nicht mehr. Verbindlich bleibt die Bitgleichheit auf demselben Gerät; zwischen Geräten gelten die Toleranzen aus `tests/tolerances.hpp`.
