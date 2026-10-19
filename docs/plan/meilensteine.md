# Meilensteinplan für Agenten

Jede Aufgabe hat eine ID (`M<n>-T<k>`), Abhängigkeiten und Abnahmekriterien. Eine Aufgabe ist ein PR (siehe AGENTS.md). Aufgaben innerhalb eines Meilensteins ohne gegenseitige Abhängigkeit können parallel bearbeitet werden.

## Übersicht

```mermaid
flowchart LR
    M0[M0 Gerüst] --> M1[M1 Datenmodell, Glas, CPU-Referenz]
    M1 --> M2[M2 CUDA-Vorwärtskernel FP64]
    M2 --> M3[M3 Batch, FD-Jacobi, gebündelter LM]
    M3 --> M4[M4 Vollständige Merit, Geisterplatten]
    M4 --> M5[M5 Adjungierte Ableitungen]
    M4 --> M6[M6 Synthese]
    M5 --> M6
    M2 --> M7[M7 Performance, df64, Multi-GPU]
    M4 --> M8[M8 Auswertung, Tolerierung]
    M6 --> M9[M9 Python, Raytatouille-Adapter]
    M8 --> M9
```

M7 kann ab M2 parallel laufen, wird aber erst gebraucht, wenn die Laufzeiten der Synthese stören. Die P100 deckt FP64 bis dahin ab.

---

## M0 – Projektgerüst ✅

Erledigt mit dem ersten Commit: Verzeichnisstruktur, CMake mit Presets (CPU, CUDA), CI (Linux, Windows, CUDA-Kompilierung), Mathe-Kern für Transfer, Sphärenschnitt und Brechung, analytische Tests, CUDA-Smoke-Test, Dokumentation, ADRs, dieser Plan.

---

## M1 – Datenmodell, Glas, CPU-Referenztracer ✅

Erledigt mit den PRs #1–#6, #8 und #10 (Oktober 2026).

Ziel: Ein System aus JSON laden und auf der CPU exakt tracen, mit Paraxialdaten und RMS-Wellenfront.

### M1-T1 – Systembeschreibung einlesen und schreiben
- **Abhängig von:** M0
- **Inhalt:** nlohmann/json per FetchContent (feste Version). Datenmodell `mko::SystemDescription` (Host), Einlesen und Schreiben von `mikroptim.system/1` nach docs/dateiformat.md, Validierung mit Fehlermeldungen inklusive JSON-Pfad.
- **Abnahme:**
  - Roundtrip (lesen → schreiben → lesen) ist für alle Dateien in `data/systems/` verlustfrei.
  - Mindestens 8 Negativtests (fehlende Pflichtfelder, falsche Typen, unbekannte Hauptversion, `stop` nicht an erster Stelle, unbekanntes Glas, ungleiche Längen von Wellenlängen und Gewichten …) mit sinnvoller Meldung.

### M1-T2 – Glaskatalog und Brechzahlen
- **Abhängig von:** M0
- **Inhalt:** `mikroptim.glass/1` laden, Sellmeier auswerten, Brechzahltabelle n[Fläche][Wellenlänge] für ein System erzeugen. Materialien `AIR`, Katalogname, `constant`, `buchdahl`.
- **Abnahme:**
  - n_d (587,56 nm) für N-BK7 = 1,51680, CaF₂ = 1,43385, Quarzglas = 1,45846, jeweils ±1·10⁻⁵.
  - Fehler außerhalb von `range_nm` wird gemeldet.

### M1-T3 – Buchdahl-Fit
- **Abhängig von:** M1-T2
- **Inhalt:** Kleinste-Quadrate-Fit von (n₀, ν₁, ν₂, ν₃) an Sellmeier über 400–700 nm nach numerik.md, Abschnitt 10.
- **Abnahme:** Maximaler Fitfehler mit 3 Koeffizienten < 1·10⁻⁵ für alle Gläser in `basis.json`; Fehler mit 2 Koeffizienten wird berichtet (Test protokolliert ihn).

### M1-T4 – Pupillenquadratur
- **Abhängig von:** M0
- **Inhalt:** Header-only, `MKO_HD`: Gauß-Legendre-Knoten und Gewichte für N ≤ 16 (vorab berechnete Tabellen oder exakte Berechnung auf dem Host), Abbildung Index → (ρ, φ, Gewicht), halbe und volle Pupille, Achsfall.
- **Abnahme:**
  - Mittelwerte aller rotationssymmetrischen Zernike-Polynome bis zur garantierten Ordnung werden exakt integriert (Fehler < 1·10⁻¹⁴); Orthogonalität der Zernike-Polynome R_n^m cos(mφ) bis n = 2N−1 auf der halben Pupille.
  - Summe der Gewichte = 1.

### M1-T5 – Sequenzieller CPU-Trace
- **Abhängig von:** M1-T1, M1-T2
- **Inhalt:** `trace_ray<R>` nach architektur.md (header-only, `MKO_HD`) über eine `SurfaceTable`-Sicht; kollimierte Quelle mit OPL₀ = **p₀**·**d₀** und Punktquelle; Statusmaske; CPU-Tracer als Schleife darüber.
- **Abnahme:**
  - Planplatte (`planplatte_bk7.json`): OPL jedes Strahls gegen geschlossene Formel, Fehler < 1·10⁻¹¹ mm.
  - Einzelfläche: exakter trigonometrischer Strahl (wie in `test_surface_ops.cpp`) über das ganze System-API.
  - Aplanatische Punkte: Punktquelle im Glas (n) im Abstand R/n vom Krümmungsmittelpunkt einer Kugelfläche. Alle gebrochenen Strahlen zeigen, rückwärts verlängert, auf den virtuellen Bildpunkt im Abstand n·R vom Mittelpunkt (Abstand Gerade–Punkt < 1·10⁻¹² mm) für Aperturen bis NA 0,95.
  - Strahlumkehr: Ein durch ein mehrlinsiges System getracter Strahl, rückwärts getract, kehrt auf < 1·10⁻¹⁰ mm zum Start zurück.

### M1-T6 – Paraxialoptik
- **Abhängig von:** M1-T1, M1-T2
- **Inhalt:** y-nu-Trace, Brennweite, Schnittweite, Hauptebenen, Petzvalsumme und -krümmung, paraxiale Pupillenlage.
- **Abnahme:**
  - Dicke Linse: Brennweite nach Linsenmacherformel (Fehler < 1·10⁻¹² relativ); `einzellinse_bk7.json` ergibt f ≈ 101,05 mm, Schnittweite ≈ 99,05 mm.
  - Petzvalsumme gegen Formel in numerik.md.
  - Grenzwert: Achsschnittpunkt realer Strahlen für h → 0 konvergiert gegen die paraxiale Schnittweite.

### M1-T7 – Referenzsphäre, OPD, RMS-Wellenfront
- **Abhängig von:** M1-T4, M1-T5
- **Inhalt:** Referenzsphäre und Kolbenabzug nach numerik.md, Abschnitt 7; RMS-Wellenfront pro Feld und Wellenlänge über die Quadratur.
- **Abnahme:**
  - Aplanatischer Fall (Punktquelle, Referenz im exakten Bildpunkt): RMS-OPD < 1·10⁻¹⁰ mm.
  - Kleine Apertur: sphärische Aberration der Einzellinse stimmt mit der Seidel-Näherung W₀₄₀ auf < 1 % überein (NA ≤ 0,05).
  - Achsial verschobener Referenzpunkt bei einer perfekten Kugelwelle: OPD stimmt mit der exakten geometrischen Formel überein (< 1·10⁻¹² mm).

### M1-T8 – CLI `mikroptim trace`
- **Abhängig von:** M1-T6, M1-T7
- **Inhalt:** `mikroptim trace <system.json>` gibt Paraxialdaten, RMS-Wellenfront (in Wellen) pro Feld und Wellenlänge, Anzahl versagter Strahlen aus. Optional `--json` für maschinenlesbare Ausgabe.
- **Abnahme:** Ausgabe für `einzellinse_bk7.json` in docs/ als Beispiel dokumentiert; Test prüft die JSON-Ausgabe.

---

## M2 – CUDA-Vorwärtskernel FP64 ✅ (P100 ausstehend)

Umgesetzt mit den PRs #9, #12, #14, #16, #17, #19 (Oktober 2026), getestet auf der RTX 3090. Offen: GPU-Tests und Benchmark auf der Tesla P100 (Befehl in docs/benchmarks.md).

Ziel: Dieselben Ergebnisse wie die CPU, für viele Varianten gleichzeitig.

### M2-T1 – SystemBatch
- **Abhängig von:** M1-T5
- **Inhalt:** SoA-Layout nach architektur.md, Host-Container und Device-Sicht, Upload, Speicherabschätzung, maximale Batchgröße aus `cudaMemGetInfo`.
- **Abnahme:** Unit-Tests für Indexierung und Speicherabschätzung; Upload und Download sind verlustfrei.

### M2-T2 – Kernel `RESIDUALS`
- **Abhängig von:** M2-T1, M1-T7
- **Inhalt:** Grid-Abbildung (Block pro Variante, Feld, Wellenlänge), Systemtabelle im Shared Memory, Pupillenstützstellen im Thread, maskiertes Versagen, Kolbenabzug per Block-Reduktion, Ausgabe des Residuenvektors pro Variante.
- **Abnahme:** |GPU − CPU| ≤ `kCpuGpuFp64` für alle Strahlen von mindestens 1.000 zufällig gestörten Varianten der Testsysteme, auf RTX 3090 und P100.

### M2-T3 – Kernel `MERIT` mit deterministischer Reduktion
- **Abhängig von:** M2-T2
- **Inhalt:** gewichtete Quadratsumme pro Variante, Max-Reduktionen für Strahlhöhe und Einfallswinkel pro Fläche.
- **Abnahme:** Zehn Wiederholungen ergeben bitgleiche Ergebnisse; Merit stimmt mit der Summe der `RESIDUALS`-Ausgabe überein (< 1·10⁻¹³ relativ).

### M2-T4 – Host-API `TraceEngine`
- **Abhängig von:** M2-T2
- **Inhalt:** reiner C++-Header nach architektur.md; Fehlerbehandlung, Gerätewahl, Tranchen bei Speichermangel.
- **Abnahme:** CLI und Tests nutzen ausschließlich diese API; Batch größer als der Speicher wird korrekt in Tranchen verarbeitet (Test mit künstlich begrenztem Speicher).

### M2-T5 – CLI-Integration
- **Abhängig von:** M2-T4
- **Inhalt:** `mikroptim trace --gpu`.
- **Abnahme:** Ausgabe identisch zur CPU-Variante (innerhalb der Toleranz).

### M2-T6 – Benchmark
- **Abhängig von:** M2-T3
- **Inhalt:** `bench/trace_throughput` (Flächenzahl 10–40, Varianten 1–10.000).
- **Abnahme:** Messungen für RTX 3090 und P100 in docs/benchmarks.md. Abweichung von der Schätzung in docs/hardware.md um mehr als Faktor 3 wird im PR erklärt.

---

## M3 – Batch-Varianten, finite-Differenzen-Jacobi, gebündelter LM ✅

Umgesetzt mit den PRs #18, #20, #22, #26, #29 (Oktober 2026); JᵀJ mit eigenem Kernel statt cuBLAS (ADR-0010, vorgeschlagen).

Ziel: Erste Ende-zu-Ende-Optimierung.

### M3-T1 – Variablen und Variantengenerator
- **Abhängig von:** M2-T4
- **Inhalt:** Schema für `variables` in der Systemdatei (Krümmungen, Dicken, Arbeitsabstand; Glas folgt in M6), Abbildung Variablenvektor ↔ Batch, Erzeugung gestörter Varianten.
- **Abnahme:** Roundtrip Vektor → Batch → Vektor; Schema in docs/dateiformat.md dokumentiert.

### M3-T2 – Finite-Differenzen-Jacobi
- **Abhängig von:** M3-T1
- **Inhalt:** zentrale Differenzen, Schrittweitenwahl nach optimierung.md, Jacobi pro Lauf.
- **Abnahme:** Vergleich gegen Richardson-Extrapolation auf einem Testsystem: relative Abweichung < 1·10⁻⁶.

### M3-T3 – Paraxiale Nebenbedingungen und Residuenaufbau
- **Abhängig von:** M1-T6, M3-T1
- **Inhalt:** Brennweite, Abgleichlänge, Arbeitsabstand als Residuen; Paraxialtrace für alle Varianten (Host, parallel).
- **Abnahme:** Unit-Tests pro Nebenbedingung.

### M3-T4 – Gebündelter Levenberg-Marquardt
- **Abhängig von:** M3-T2, M3-T3
- **Inhalt:** cuBLAS `DgemmStridedBatched`, cuSOLVER `potrfBatched`/`potrsBatched`, Marquardt-Skalierung, Nielsen-Dämpfung, Schranken mit aktiver Menge, Abbruchkriterien.
- **Abnahme:**
  - Einzellinse N-BK7 mit fester Brennweite und kleiner Apertur auf minimale sphärische Aberration: Formfaktor q = (c₁+c₂)/(c₁−c₂) im Bereich 0,74 ± 0,05 (Dünnlinsentheorie: q = 2(n²−1)/(n+2)).
  - Ein optimiertes System, alle Variablen um 1 % gestört, findet zurück (Merit relativ < 1·10⁻⁶ über dem Optimum).
  - 1.000 gleichzeitige Läufe liefern bitgleich dieselben Ergebnisse wie dieselben Läufe einzeln.

### M3-T5 – CLI `mikroptim optimize`
- **Abhängig von:** M3-T4
- **Inhalt:** Optimierung einer Systemdatei, Ausgabe des Ergebnisses als neue Systemdatei, Iterationsprotokoll.
- **Abnahme:** Beispiel in docs/ dokumentiert.

---

## M4 – Vollständige Merit und Geisterplatten ✅

Umgesetzt mit den PRs #15, #27, #28, #33, #35, #39 (Oktober 2026).

### M4-T1 – Gewichtete Wellenfrontresiduen
- **Abhängig von:** M3-T4
- **Inhalt:** Feld- und Spektralgewichte, Normierung auf λ_ref, Gewichte √(W_i V_j).
- **Abnahme:** Merit gleich gewichteter Summe der RMS-Werte aus M1-T7 (< 1·10⁻¹² relativ).

### M4-T2 – Weiches Strahlversagen
- **Abhängig von:** M4-T1
- **Inhalt:** nach numerik.md, Abschnitt 11, in allen Optimierungsmodi.
- **Abnahme:** Startsystem, bei dem 30 % der Strahlen versagen, wird ohne manuelles Eingreifen in einen versagensfreien Zustand optimiert; Merit ist entlang einer Geraden durch den Versagensübergang stetig (Test mit fein abgetasteter Linie).

### M4-T3 – Mechanische Nebenbedingungen
- **Abhängig von:** M2-T3, M4-T1
- **Inhalt:** Durchmesser, Mitten- und Randdicke (Pfeilhöhe analytisch), maximaler Einfallswinkel, aus den Max-Reduktionen des Kernels.
- **Abnahme:** Unit-Tests; ein Testlauf hält alle Grenzen ein.

### M4-T4 – Petzval-Nebenbedingung
- **Abhängig von:** M1-T6
- **Abnahme:** Testsystem mit aktiver Nebenbedingung erreicht |κ_P| ≤ κ_max.

### M4-T5 – Flächentyp Geisterplatte
- **Abhängig von:** M1-T5
- **Inhalt:** nach numerik.md, Abschnitt 6, CPU und GPU; Dateiformat für `ghost_plates`.
- **Abnahme:**
  - Lineare Phase Φ = a·y lenkt exakt um Δ(n d_y) = a ab.
  - Gradient des Dickenprofils gegen finite Differenzen (< 1·10⁻¹⁰ relativ).
  - Geisterplatte mit gefitteter idealer Fokussierphase bei NA 0,3: RMS-OPD < 1·10⁻⁶ mm.
  - Paraxiale Brechkraft φ_G nach Formel stimmt mit dem Grenzwert realer Strahlen überein.

### M4-T6 – L1-Strafe und Diagnosebericht
- **Abhängig von:** M4-T5, M3-T4
- **Inhalt:** geglättete L1-Residuen, Gewichte pro Ordnung, Strafgewicht μ, Diagnosebericht nach optimierung.md, Abschnitt 5.
- **Abnahme:**
  - Exaktheit: Ein System, das sein Ziel alleine erreicht, endet mit allen |a| < 1·10⁻⁸ (μ oberhalb der Schwelle).
  - Bedarf: Einzellinse bei NA 0,5 mit Beugungsbegrenzungsziel aktiviert Geisterkoeffizienten; der Diagnosebericht nennt Position, Ordnung und Glastyp.

---

## M5 – Adjungierte Ableitungen

### M5-T1 – Vorwärtsdurchlauf mit Zustandsspeicherung
- **Abhängig von:** M4-T5
- **Inhalt:** `TraceRecord` im Local Memory, optional Checkpointing.

### M5-T2 – Rückwärtsdurchlauf
- **Abhängig von:** M5-T1
- **Inhalt:** lokale Ableitungen jeder Flächenoperation (Transfer, Schnitt, Brechung, Geisterplatte, Referenzsphäre), Gradientenzeile pro Residuum bezüglich aller Variablen, inklusive Kolbenabzug als lineare Projektion.
- **Abnahme:** relative Abweichung zur finite-Differenzen-Jacobi ≤ 1·10⁻⁶ für alle Variablentypen.

### M5-T3 – Integration in LM
- **Abhängig von:** M5-T2
- **Abnahme:** Gleiche Optimierungsergebnisse wie mit finiten Differenzen; Jacobi mindestens 10× schneller für P ≥ 60 auf beiden GPUs.

---

## M6 – Systemsynthese

### M6-T1 – Kontinuierliches Glas als Variable ✅ (#36)
- **Abhängig von:** M1-T3, M3-T4
- **Inhalt:** (n₀, ν₁, ν₂) als Variablen, konvexe Hülle der Katalog-Gläser (normierte Koordinaten), Strafterm außerhalb.
- **Abnahme:** Hüllentests; Optimierung bleibt innerhalb der Hülle.

### M6-T2 – Einfüge-Operatoren A und B
- **Abhängig von:** M4-T6
- **Inhalt:** nach optimierung.md, Abschnitt 6, inklusive Initialisierung aus dem Diagnosebericht.
- **Abnahme:** Merit nach dem Einfügen relativ unverändert (|ΔF| ≤ 1·10⁻¹²). Operator B macht aus einer Einzellinse bei zwei Wellenlängen ein Kittglied, das den Farblängsfehler um mehr als den Faktor 10 reduziert.

### M6-T3 – Homotopie-Controller
- **Abhängig von:** M6-T2
- **Inhalt:** Parameter NA, Feld, Spektrum, μ; Schrittweitensteuerung; Protokoll jedes Schritts.
- **Abnahme:** Ausgehend von einer Linse bei NA 0,3, monochromatisch, entsteht automatisch ein beugungsbegrenztes System bei NA 0,6 auf der Achse.

### M6-T4 – Kandidatenverwaltung
- **Abhängig von:** M6-T3
- **Inhalt:** Strahlsuche über Zweige (Sattelpunkt-Abstiege, Einfügungen), gemeinsame Batches pro Topologie.
- **Abnahme:** Läufe sind mit festem Seed reproduzierbar; Protokoll zeigt den Stammbaum der Kandidaten.

### M6-T5 – Glas-Einrasten ✅ (#38)
- **Abhängig von:** M6-T1
- **Abnahme:** Nach dem Einrasten nur Katalog-Gläser; Merit-Verschlechterung wird berichtet.

### M6-T6 – Lauf auf der Zielspezifikation
- **Abhängig von:** M6-T4, M6-T5
- **Inhalt:** stufenweiser Lauf nach optimierung.md, Abschnitt 7, bis zur vollen Spezifikation aus docs/spezifikation.md.
- **Abnahme:** Bericht unter `docs/ergebnisse/` mit Kandidaten, Linsenzahl, Wellenfrontfehler pro Feld und Wellenlänge, Laufzeit. Forschungsaufgabe ohne festen Zielwert; Vergleich mit den Startsystemen des Maintainers.

---

## M7 – Performance

### M7-T1 – Zahlentyp df64 ✅ (#23)
- **Abhängig von:** M2-T3
- **Abnahme:** |df64 − FP64| ≤ `kDf64VsFp64` für alle Tests der Testsuite; Durchsatz auf der RTX 3090 gemessen.

### M7-T2 – FP32-Vorfilterung ✅ (#25)
- **Inhalt:** schneller Pfad zum Aussortieren von Varianten mit massivem Strahlversagen oder schlechter Merit im Multistart.
- **Abnahme:** keine falsch aussortierten Varianten in einem Testsatz (Vergleich mit FP64).

### M7-T3 – Mehrere GPUs ✅ (#30, nur mit einer GPU getestet)
- **Abhängig von:** M2-T4
- **Abnahme:** Ergebnisse pro Variante bitgleich zum Einzel-GPU-Lauf; Verteilung nach gemessenem Durchsatz.

### M7-T4 – Profiling und Tuning
- **Inhalt:** Nsight Compute, Registerdruck, Belegung, Blockgröße.
- **Abnahme:** Profilberichte und Ergebnisse in docs/benchmarks.md.

---

## M8 – Auswertung und Tolerierung

### M8-T1 – Modus `EVAL` ✅ (#32)
- **Abhängig von:** M4-T1
- **Inhalt:** dichtes kartesisches Pupillenraster, OPD-Karten, Spotdiagramme, Feldkrümmung, Verzeichnung.

### M8-T2 – PSF, MTF, Strehl ✅ (#34)
- **Abhängig von:** M8-T1
- **Inhalt:** skalare Beugung über cuFFT. Hinweis: Bei NA 0,95 ist die skalare Näherung ungenau; ein vektorielles Modell (Richards-Wolf) ist als M8-T5 vorgesehen.
- **Abnahme:** Perfektes System bei kleiner NA: erste Nullstelle der PSF bei 0,61·λ/NA (< 1 %); Strehl gegen Maréchal-Näherung bei kleinen Aberrationen.

### M8-T3 – Dezentrierung und Verkippung
- **Inhalt:** allgemeine Flächentransformationen, volle Pupille (Symmetrie entfällt). Erfordert eine ADR für die Erweiterung der Konventionen.

### M8-T4 – Monte-Carlo-Tolerierung
- **Abhängig von:** M8-T2, M8-T3
- **Inhalt:** Toleranzsatz (Radien, Dicken, Brechzahlen, Dezentrierung, Verkippung), Stichproben als Varianten im Batch, Kompensatoren (z. B. Arbeitsabstand).
- **Abnahme:** Verteilung von Strehl und RMS-Wellenfront über 10.000 Stichproben; Ergebnis reproduzierbar mit festem Seed.

### M8-T5 – Vektorielle PSF (optional) ✅ (#37)
- **Inhalt:** Richards-Wolf für hohe NA.
- **Abnahme (in der Umsetzung festgelegt):** kleine NA → skalare PSF mit Abweichung zweiter Ordnung in der NA; NA 0,95 aberrationsfrei gegen unabhängige Quadratur des Richards-Wolf-Integrals (Intensität und Komponenten, Halbwertsbreiten in x und y relativ < 10⁻³) und gegen die geschlossene Form der Energieanteile; Energieerhaltung, Symmetrien (Spiegelung, Drehung der Polarisation), GPU = CPU, Determinismus; Strehl bei kleinen Aberrationen gegen Maréchal (docs/numerik.md, „Vektorielle PSF“).

---

## M9 – Python und Integration

### M9-T1 – Python-Bindings
- **Inhalt:** nanobind; Systeme laden, tracen, optimieren, Ergebnisse als NumPy.
- **Abnahme:** pytest-Suite, Wheel-Build für Linux und Windows.

### M9-T2 – Adapter Raytatouille
- **Inhalt:** Konvertierung `*.rtt.json` ↔ `mikroptim.system/1` für den unterstützten Teilumfang (sequenziell, koaxial, sphärisch).
- **Abnahme:** Roundtrip für Testsysteme; nicht abbildbare Elemente werden klar gemeldet.

---

## Querschnittsaufgaben

- CI aktuell halten; GPU-Tests lokal vor jedem Merge von CUDA-Code.
- docs/ und CHANGELOG.md mit jeder Aufgabe pflegen.
- Neue ADR bei jeder grundsätzlichen Entscheidung.
