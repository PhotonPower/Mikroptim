# ADR-0007: Deterministische Reduktionen

- **Status:** angenommen
- **Datum:** 2026-10-18

## Kontext
Gleitkomma-Atomics liefern je nach Ausführungsreihenfolge unterschiedliche Summen. Das macht Optimierungsläufe nicht reproduzierbar und Tests instabil.

## Entscheidung
Reduktionen für Merit, Residuen und Jacobi-Matrizen: Baumreduktion im Block, Ergebnis pro Block in den globalen Speicher, Summation in fester Reihenfolge. Gleiche Eingaben auf derselben GPU ergeben bitgleiche Ergebnisse; die Verteilung auf mehrere GPUs ändert Ergebnisse pro Variante nicht.

## Konsequenzen
Etwas mehr Code und ein zweiter Reduktionsschritt; vernachlässigbare Kosten.
