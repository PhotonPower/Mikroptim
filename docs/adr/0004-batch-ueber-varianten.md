# ADR-0004: Parallelisierung über Systemvarianten

- **Status:** angenommen
- **Datum:** 2026-10-18

## Kontext
Mit Gauß-Quadratur braucht die Merit eines einzelnen Systems nur einige tausend Strahlen. Ein Kernel pro System wäre durch den Start-Overhead dominiert. Viele Varianten fallen ohnehin an: Jacobi-Spalten, Multistart, Homotopie-Zweige, Tolerierung.

## Entscheidung
Ein Kernelaufruf verarbeitet einen Batch von Varianten gleicher Topologie. Ein Block pro (Variante, Feld, Wellenlänge), Threads über Pupillenstützstellen. Die Systemtabelle der Variante liegt im Shared Memory. Der Optimierer iteriert viele Läufe gleichzeitig (gebündelter LM).

## Konsequenzen
Varianten verschiedener Topologie laufen in getrennten Batches. Speicherbedarf wächst mit der Batchgröße; sie wird zur Laufzeit aus dem freien Speicher bestimmt.
