# ADR-0009: Erst finite Differenzen, dann Adjungierte

- **Status:** angenommen
- **Datum:** 2026-10-18

## Kontext
LM braucht die Jacobi-Matrix. Finite Differenzen kosten (P+1) bis 2P Traces, sind aber trivial korrekt und nutzen die Batch-Infrastruktur. Die adjungierte Methode kostet etwa das Zwei- bis Dreifache eines Traces.

## Entscheidung
M3 implementiert zentrale finite Differenzen über gebündelte Varianten. M5 implementiert den adjungierten Modus pro Strahl. Die finite-Differenzen-Variante bleibt dauerhaft als Test-Orakel erhalten.

## Konsequenzen
Frühe Ende-zu-Ende-Optimierung ohne aufwendige Ableitungen; spätere Beschleunigung mit gesicherter Referenz.
