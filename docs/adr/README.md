# Architekturentscheidungen (ADR)

Jede grundsätzliche Entscheidung bekommt eine nummerierte Datei nach [vorlage.md](vorlage.md). Entscheidungen werden nicht gelöscht, sondern durch neue ADRs ersetzt („Ersetzt durch ADR-00xx“).

| Nr. | Titel | Status |
|---|---|---|
| [0001](0001-sprache-und-build.md) | C++20, CMake, Linux und Windows | angenommen |
| [0002](0002-cuda-12.md) | CUDA 12.x statt 13 wegen Tesla P100 | angenommen |
| [0003](0003-gleitkomma.md) | FP64 als Standard, templatisierter Zahlentyp | angenommen |
| [0004](0004-batch-ueber-varianten.md) | Parallelisierung über Systemvarianten | angenommen |
| [0005](0005-gemeinsamer-mathe-kern.md) | Gemeinsamer Mathe-Kern für CPU und GPU | angenommen |
| [0006](0006-blende-erste-flaeche.md) | Blende als erste Fläche, kollimierte Quelle | angenommen |
| [0007](0007-deterministische-reduktionen.md) | Deterministische Reduktionen | angenommen |
| [0008](0008-dateiformat-json.md) | Eigenes JSON-Systemformat, Raytatouille per Adapter | angenommen |
| [0009](0009-ableitungen.md) | Erst finite Differenzen, dann Adjungierte | angenommen |
| [0010](0010-normalgleichungen-ohne-cublas.md) | Normalgleichungen mit eigenem Kernel, Cholesky auf dem Host | vorgeschlagen |
| [0011](0011-dezentrierung-verkippung.md) | Dezentrierung und Verkippung einzelner Flächen | angenommen |
