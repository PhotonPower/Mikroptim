# M6-T10: Stufenweise Synthese mit Rückbau am realen System und Rückkehr zu Geisterplatten

Aufgabe: [Meilensteinplan](../plan/meilensteine.md#m6-t10--stufenweise-synthese-mit-rückbau-am-realen-system-und-rückkehr-zu-geisterplatten). Ausgangslage: [Bericht M6-T9](m6-t9-rueckbau.md) – der Rückbau wirkt am realen System (Z2, ohne Geisterplatten), mit Geisterplatten kaum (F17); vorgeschlagen war der Ablauf „Stufe → Z2 → Rückbau → nächste Stufe vom verkleinerten System aus“, dem die Rückkehr vom realen System zu Geisterplatten fehlte. Alle Zahlen aus Läufen auf dem Rechner des Maintainers (RTX 3090, geteilt mit bis zu vier gleichzeitigen Läufen dieser Aufgabe und anderen Agenten – Laufzeiten grob), FP64, Release mit CUDA 12.9, Referenzsphäre `image_infinite`, 16 Ringe × 8 Speichen, FD-Jacobi. Gläser nur `basis.json` und die frei erfundenen Testgläser `synthetisch.json` – die Systeme sind nicht herstellbar; es werden keine Linsendaten realer Objektive verwendet.

**Kurzfassung.**

- **Umgesetzt** (ohne Option unverändert): Rückkehr zu Geisterplatten `return_to_ghost_plates` ([optimierung.md, „Rückkehr zu Geisterplatten“](../optimierung.md#rückkehr-zu-geisterplatten-m6-t10)): Wörterbücher (SYN-36, CaF₂, SYN-30, K = 10, Koeffizienten 0) in der Mitte des inneren Luftspalts mit der größten Randstrahlhöhe und des Spalts mit dem größten |ȳ/y|, Spalte fest, r_g aus den realen Strahlhöhen, Strafe wie im Startsystem; `mikroptim insert --ghost-return`, `mikroptim synth --start-from-real`; Skript `--zyklus feld,farbe,apertur`. Merit-Invarianz: Testsysteme ≤ 1,1·10⁻¹⁵ relativ, Messläufe |ΔF| ≤ 1,1·10⁻¹⁰ relativ (Rundungsboden sehr gut korrigierter Systeme, ≈ 10⁻¹³ Wellen²).
- **Feld bei hoher NA (5-Linser, NA 0,869):** Mit Geisterplatten, Feldregel und Neustarts öffnet sich das Feld auf **65 %** (12 El., Z2: 0,041/0,070/0,089 Wellen) – praktisch wie in M6-T9 ohne Geisterplatten (64 %, 11 El.). Die zurückgekehrten Platten melden in den monochromatischen Stufen nie Bedarf; alle 53 Einfügungen kamen aus der Feldregel (37 wegen fehlender Verbesserung verworfen).
- **Apertur bei vollem Feld (9-Linser, NA 0,5, Lauf E):** NA **0,5 → 0,6725** bei vollem Feld, 16 El., Z2 0,030/0,071/0,078 Wellen, Strehl ≥ 0,81 (Rand knapp über λ/14); im zweiten Durchlauf NA 0,700 mit 0,091 Wellen am Rand – die höchste NA bei vollem Feld bisher (M6-T6b: 0,5 beugungsbegrenzt, V6 0,565 vor der Farbe).
- **Apertur bei 65 % Feld (Lauf D):** 0,869 → 0,874, Stillstand.
- **Farbe** (436–656 nm) aus den monochromatischen Systemen (Läufe A, B, C): nicht beugungsbegrenzt, 0,36–1,8 Wellen am Rand, schlechter als V7 aus M6-T6b (11 El., ≤ 0,26). Der Startpunkt der Farbstufe ist nicht annehmbar (F_B 0,13–0,49 Wellen² ≫ F_max), der Controller fügt bis zum Budget ein.
- **Rückbau nach den Stufen:** in keiner der zehn Stufen ein Element entfernt (0 von 195 Versuchen). Die Stufen enden an der Schwelle (F_B ≈ F_max) bzw. weit darüber, der Rückbau verlangt F_B ≤ 0,5·F_max. Der in M6-T9 beobachtete Gewinn (25 → 5 El.) setzte ein System mit Abstand zur Schwelle voraus.
- Ziel (NA 0,95, volles Feld, 436–656 nm) nicht erreicht.

## 1. Umsetzung

| Teil | Wo | Wirkung |
|---|---|---|
| `return_to_ghost_plates` | `include/mikroptim/synth/ghost_return.hpp` | Lagen `aperture` (größtes \|y\|), `field` (größtes \|ȳ/y\|), wahlweise `front`, `stop`; Wörterbuch je Lage in der Spaltmitte, Spalte fest, r_g = 1,02 × größte reale Strahlhöhe (alle Felder × `norm_field_scale`, Wellenlängen), neuer Eintrag `ghost_penalty` (μ = 47,2, κ = 1, ε = 10⁻⁶) nur ohne vorhandenen; Bericht mit F vorher/nachher |
| `mikroptim insert --ghost-return` | `src/cli/insert_command.cpp` | dazu `--ghost-glass`, `--ghost-order`, `--ghost-weight`, `--ghost-positions`, `--norm-field`; JSON `details.positions`, `notes` |
| `mikroptim synth --start-from-real` | `src/cli/synth_command.cpp` | Rückkehr vor dem Lauf, r_g für max(Start-, Zielfeld); JSON `options.start_from_real` nur mit Option |
| Skript `--zyklus` | `tools/m6t6_zielspezifikation.py` | je Stufe (feld, farbe, apertur): reales System mit vollem Feld → `insert --ghost-return` → Stufe mit `synth_loop` (Neustarts, μ-Halbierung, erzwungenes Einfügen; Farbe in zwei Teilstufen 2 bzw. 5 Wellenlängen, Operator B) → Z2 (2.000 Zyklen, Spektrum der Stufe, Glas-Hülle in drei Stufen) → Rückbau am realen System (`synth --prune` am erreichten Punkt, ρ = 0,5, 30 Versuche, höchstens die Elemente von Z2) → Z2 nach dem Rückbau, bewertet; Stufen am Ziel und ohne Fortschritt am selben Zustand werden übersprungen; `--zyklus-wiederholen`, `--zyklus-runden`, `--geister-lagen`, `--zyklus-tabelle` |

Beim 5-Linser setzt die Rückkehr die Platten in die Spalte nach Fläche 4 (`aperture`, r_g = 3,98 mm) und 8 (`field`, zugleich der letzte innere Spalt, r_g = 3,18 mm), beim 9-Linser nach Fläche 8 (4,77 mm) und 16 (1,24 mm).

Behoben während der Messung: Der Rückbau-Lauf konnte bei nicht annehmbarem Startpunkt zuerst einfügen (Lauf D: 13 → 14 El.); er ist jetzt auf die Elementzahl von Z2 begrenzt. Ein zweiter Durchlauf wiederholte Stufen ohne Fortschritt am selben Zustand (B, C); sie werden jetzt übersprungen (die betroffenen Teilläufe sind entfernt, die Tabellen enthalten nur abgeschlossene Stufen).

## 2. Messung

Z2 = Stufenergebnis ohne Geisterplatten nachoptimiert; „Rückbau“ = Elemente nach dem Rückbau-Lauf und Z2 danach (bewertet, `mikroptim eval --grid 256 --reference exit-pupil`). RMS in Wellen bei λ des Tupels bei 0/70/100 % des erreichten Feldanteils, größter Wert über die Wellenlängen; Strehl = kleinster FFT-Strehl. Zeit = Summe der Programmaufrufe der Stufe (GPU geteilt, grob).

| Lauf | Stufe | Start | NA | Feld | Spektrum | El. Stufe → Rückbau | RMS 0/70/100 % | Strehl min | Zeit [s] |
|---|---|---|---|---|---|---|---|---|---|
| A | c0_feld | 5-Linser M6-T9 (NA 0,869, 23 %) | 0,869 | **0,653** | 546 nm | 12 → 12 | 0,041/0,070/0,089 | 0,81 | 7.890 |
| A | c1_farbe | A c0 | 0,869 | 0,653 | 436–656 | 14 → 14 | 0,480/0,613/0,806 | 0,01 | 1.901 |
| D | c0_apertur | A c0 | 0,874 | 0,653 | 546 nm | 13 → 13 | 0,042/0,073/0,091 | 0,82 | 1.858 |
| D | c1_apertur | D c0 | 0,874 | 0,653 | 546 nm | 14 → 14 | 0,042/0,070/0,090 | 0,82 | 713 |
| B | c0_farbe | 9-Linser M6-T9 (NA 0,5, 100 %) | 0,50 | 1,00 | 436–656 | 15 → 15 | 0,650/0,998/1,763 | 0,00 | 3.204 |
| B | c1_apertur | B c0 | 0,50 | 1,00 | 436–656 | 18 → 18 | 0,191/0,311/0,532 | 0,00 | 2.802 |
| C | c0_farbe | 6-Linser M6-T9 (NA 0,5, 100 %) | 0,50 | 1,00 | 436–656 | 12 → 12 | 0,759/0,864/0,996 | 0,00 | 1.554 |
| C | c1_apertur | C c0 | 0,50 | 1,00 | 436–656 | 14 → 14 | 0,352/0,520/0,709 | 0,00 | 2.860 |
| E | c0_apertur | 9-Linser M6-T9 | **0,6725** | **1,00** | 546 nm | 16 → 16 | **0,030/0,071/0,078** | 0,81 | 3.802 |
| E | c1_apertur | E c0 | 0,700 | 1,00 | 546 nm | 16 → 16 | 0,025/0,077/0,091 | 0,75 | 1.184 |

Optionen aller Läufe: `--feldregel --einfuegen-verbessern 4`, Elementbudget A 12, B 16, C 14, D 14, E 16; A, B, C, E zusätzlich `--farbregel --farb-budget 3`. Die Aperturstufen von B und C begannen am polychromatischen Ergebnis der Farbstufe (Reihenfolge feld, farbe, apertur; „Farbe früh“).

A c0 bei vollem Feld (ohne Nachoptimierung): 0,041/0,328/2,71 Wellen (M6-T9 JFV855R_z2_feld: 0,17/1,8 bei 70/100 %).

E c0 (`E_9linser_apertur/c0_apertur_z2r_s0.json`), Tupel bei 546 nm: RMS 0,030/0,071/0,078 Wellen, Strehl (FFT) 0,966/0,856/0,811. Der zweite Durchlauf (E c1, eine Runde) öffnet bis NA 0,700, nachoptimiert aber mit 0,091 Wellen am Rand.

## 3. Vergleich

| Größe | M6-T6b (#105) | M6-T8 (#108) | M6-T9 (#110) | M6-T10 |
|---|---|---|---|---|
| beugungsbegrenzt, volles Feld, 546 nm | V6-Z2: 9 El., NA 0,50, ≤ 0,033 | F1: 25 El., NA 0,50, ≤ 0,059 | 9 El., NA 0,50, ≤ 0,038 (6 El. ≤ 0,068) | E: 16 El., **NA 0,6725**, ≤ 0,078 (knapp über 0,0714) |
| hohe NA, größtes Feld | NA 0,855, 10 %, 5 El. | NA 0,869, 23 %, 25 El. | NA 0,869, 64 %, 11 El. (ohne Platten) | NA 0,869, **65 %**, 12 El. (mit Platten), ≤ 0,089; NA 0,874 mit 13 El. |
| polychromatisch 436–656, volles Feld | V7: 11 El., NA 0,50, ≤ 0,26 | – | – | B/C: 12–18 El., NA 0,50, 0,53–1,8 (schlechter) |
| Rückbau | – | – | am realen System 25 → 9 bzw. 5 | 0 von 151 Versuchen (Stufen enden an der Schwelle) |

## 4. Bewertung

1. **Rückkehr zu Geisterplatten** funktioniert technisch: Merit unverändert, die Platten liegen dort, wo die Synthese sie braucht, und die Zyklen laufen durch. Sie bringt aber in den monochromatischen Stufen nichts gegenüber M6-T9: Die Platten bleiben exakt null (L1-Strafe), solange das reale System das Ziel erreicht; an der Schwelle meldet die Diagnose keinen Bedarf, und die Feldregel übernimmt (A: 53 von 53 Einfügungen aus der Feldregel). Das 65-%-Feld ist dasselbe Ergebnis wie ohne Platten (64 %). In der Farbstufe melden die Platten Bedarf, empfehlen aber Operator A (Meniskus), nicht B; die Farbregel greift nur vereinzelt.
2. **Der Rückbau braucht Abstand zur Schwelle.** In M6-T9 wirkte er, weil die 25-Linser am Teilfeld weit unter F_max lagen. Eine Stufe der Homotopie endet dagegen definitionsgemäß dort, wo F_B die Schwelle erreicht; mit ρ = 0,5 ist dann kein Element entbehrlich. Der Zyklus „Stufe → Rückbau“ verkleinert so nichts; sinnvoll wäre der Rückbau an einem Punkt hinter der Stufe zurück (z. B. 80–90 % des erreichten Feldes bzw. der NA), bevor die nächste Stufe beginnt (Vorschlag, nicht umgesetzt).
3. **Apertur bei vollem Feld ist der nutzbare Gewinn** (Lauf E): Ab dem 9-Linser erreicht die Aperturstufe mit Geisterplatten und Feldregel NA 0,6725 bei vollem Feld (16 El.), nachoptimiert am Rand knapp über λ/14. Bei 65 % Feld und NA 0,869 (D) bleibt die Apertur stehen.
4. **Farbe aus einem monochromatischen Entwurf** bleibt der Engpass (F13/F15): Der Sprung von 546 nm auf 480/644 nm macht den Startpunkt unannehmbar (F_B 0,13–0,49 Wellen²), und die Farbstufen füllen das Budget mit Menisken. Die besten polychromatischen Ergebnisse liefern weiterhin „Farbe früh“ bei kleinem Feld (V3) bzw. V7.

## 5. Befunde

| Nr. | Stand nach M6-T10 |
|---|---|
| F14 | bei NA 0,869 auf 65 % Feld (mit und ohne zurückgekehrte Platten gleich); die Platten bleiben in der Feldstufe passiv, die Feldregel trägt |
| F15 | Apertur nach den Farbstufen: weiterhin kein Fortschritt (B, C: NA 0,5 bleibt, das polychromatische System liegt über der Schwelle) |
| F16 | Einfügen nur mit Verbesserung verwirft viele Einfügungen (A: 37 von 53), das Budget begrenzt den Rest; überzählige Elemente entstehen kaum noch, der Rückbau findet keine |
| F17 | bestätigt und verallgemeinert: Der Rückbau wirkt nur mit Abstand zur Schwelle – am realen System direkt nach einer Stufe ebenso wenig wie mit Geisterplatten (0 von 195) |

## 6. Tests

`tests/test_ghost_return.cpp`: `GhostReturn.MeritUnchangedAndPlatesAtExpectedGaps` (Orakel Aberrationsbericht, Merit ≤ 1,1·10⁻¹⁵ relativ), `…NormRadiusFromRealRaysInCollimatedBeam`, `…ExistingPenaltyAndOrderOptions`, `…LocationsParseAndErrors`; `CliInsert.GhostReturnMatchesLibrary`; `CliSynth.StartFromReal` (ohne Option kein Eintrag `start_from_real`, keine Platten); Python `test_tool_m6t6.py::test_system_state_and_stage_done`, `::test_return_to_ghosts_command`, `::test_cycle_run_stops_at_goal_or_stall`.

## 7. Dateien

```
python tools/m6t6_zielspezifikation.py --gpu --zyklus feld,farbe --start docs/ergebnisse/m6-t9/JFV855R_z2_rueckbau/z2_s0.json --feldregel --einfuegen-verbessern 4 --elementbudget 12 --farbregel --farb-budget 3 --out docs/ergebnisse/m6-t10/A_5linser
python tools/m6t6_zielspezifikation.py --gpu --zyklus feld,farbe,apertur --start docs/ergebnisse/m6-t9/F1R_z2_rueckbau/z2_s0.json --feldregel --einfuegen-verbessern 4 --elementbudget 16 --farbregel --farb-budget 3 --out docs/ergebnisse/m6-t10/B_9linser
python tools/m6t6_zielspezifikation.py --gpu --zyklus feld,farbe,apertur --start docs/ergebnisse/m6-t9/F1R_z2_rueckbau_f1/z2_s0.json --feldregel --einfuegen-verbessern 4 --elementbudget 14 --farbregel --farb-budget 3 --out docs/ergebnisse/m6-t10/C_6linser
python tools/m6t6_zielspezifikation.py --gpu --zyklus apertur --zyklus-runden 3 --start docs/ergebnisse/m6-t10/A_5linser/c0_feld_z2r_s0.json --feldregel --einfuegen-verbessern 4 --elementbudget 14 --out docs/ergebnisse/m6-t10/D_5linser_apertur
python tools/m6t6_zielspezifikation.py --gpu --zyklus apertur --zyklus-runden 3 --start docs/ergebnisse/m6-t9/F1R_z2_rueckbau/z2_s0.json --feldregel --einfuegen-verbessern 4 --elementbudget 16 --farbregel --farb-budget 3 --out docs/ergebnisse/m6-t10/E_9linser_apertur
python tools/m6t6_zielspezifikation.py --zyklus-tabelle docs/ergebnisse/m6-t10/E_9linser_apertur
```

A lief zuerst mit `feld,farbe,apertur`; die Aperturstufe ab dem polychromatischen Ergebnis wurde wie in B und C ohne Aussicht abgebrochen. D und E liefen zuerst mit 8 Runden und wurden nach 3 Runden beendet (Laufzeit; jede weitere Runde brachte höchstens +0,005 NA), der zweite Durchlauf von E nach einer Runde (`--zyklus-runden 1`; deshalb nennt `m6t10_zyklus.json` von E für c0 nur die Runde r0, die Protokolle r0–r2 liegen bei). Einmal brach ein synth-Lauf von E mit „cudaDeviceSynchronize(): an illegal memory access“ (`src/cuda/residuals.cu:354`, 16 Elemente plus 6 Platten, vier Läufe gleichzeitig auf der GPU) ab; nach der Fortsetzung lief derselbe Schritt durch (nicht reproduziert, Notiz im PR). Die Verzeichnisse sind mit `--archiv` verkleinert (Zusammenfassung `m6t10_zyklus.json`, Protokolle der synth-Läufe, Z2-Dateien und Bewertungen).
