# ADR-0003: FP64 als Standard, templatisierter Zahlentyp

- **Status:** angenommen
- **Datum:** 2026-10-18

## Kontext
Ziel ist eine OPD-Genauigkeit von ca. 1 nm bei optischen Weglängen um 100 mm (relativ 10⁻⁸ bis 10⁻⁹). FP32 (ε ≈ 6·10⁻⁸) reicht nicht. GeForce-Karten rechnen FP64 mit 1/64 der FP32-Rate, die P100 mit 1/2.

## Entscheidung
Der Mathe-Kern ist über den Zahlentyp `R` templatisiert. Standard ist `double`. Ab M7 kommt `df64` (Paar aus zwei `float`) für GeForce-Karten dazu. `float` ist nur in ausdrücklich gekennzeichneten Vorfilterungspfaden erlaubt. Keine fast-math-Optionen auf Host oder Device; FMA-Kontraktion ist erlaubt.

## Konsequenzen
Neue Zahlentypen liefern eigene Überladungen (z. B. `msqrt`). CPU/GPU-Vergleiche nutzen Toleranzen statt Bitgleichheit.
