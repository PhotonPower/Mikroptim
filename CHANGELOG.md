# Changelog

Format nach [Keep a Changelog](https://keepachangelog.com/de/1.1.0/), Versionierung nach [SemVer](https://semver.org/lang/de/).

## [Unreleased]

### Hinzugefügt
- M0: Projektgerüst mit CMake (CPU und optional CUDA), CI, Mathe-Kern für Transfer, Sphärenschnitt und Brechung, analytische Tests, CUDA-Smoke-Test, Dokumentation, ADRs und Meilensteinplan für Agenten.
- M1-T2: Glaskatalog `mikroptim.glass/1` (Bibliothek `Mikroptim::glass`, nlohmann/json 3.12.0), Sellmeier- und Buchdahl-Formeln header-only für CPU und GPU, Brechzahltabelle n[Fläche][Wellenlänge] für die Materialien `AIR`, Katalogname, `constant` und `buchdahl`.
