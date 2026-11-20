# M6-T9: Elementbudget und Rückbau in der Synthese

Aufgabe: [Meilensteinplan](../plan/meilensteine.md#m6-t9--elementbudget-und-rückbau). Ausgangslage: [Bericht M6-T8](m6-t8-apertur-feld.md) – die Feldregel öffnet ab der Einzellinse bei NA 0,5 das volle Feld beugungsbegrenzt, braucht aber 25 Elemente (F1); bei NA ≈ 0,87 kommt sie mit 25 Elementen auf 23 % Feld (JFV855); Befund F16 aus [M6-T6b](m6-t6b-zielspezifikation-adr0017.md) („mehr Elemente verschlechtern“). Alle Zahlen aus Läufen auf dem Rechner des Maintainers (RTX 3090, geteilt mit bis zu fünf gleichzeitigen Läufen, FP64, Release mit CUDA 12.9), Referenzsphäre `image_infinite`, 16 Ringe × 8 Speichen, FD-Jacobi. Gläser nur `basis.json` und die frei erfundenen Testgläser `synthetisch.json` – die Systeme sind nicht herstellbar; es werden keine Linsendaten realer Objektive verwendet.

**Kurzfassung.**

- **Umgesetzt** (Standard aus, ohne Option bitgleich): Rückbau im Homotopie-Controller (`synth --prune`): Elemente werden über die Rückbau-Homotopie (`collapse_element`, M10-T4) in K = 3 Stufen zum Null-Element gefahren, nach jeder Stufe nachoptimiert und entfernt (`remove_element`); übernommen nur, wenn danach F_B ≤ ρ·F_max (optional ≤ (1 + ε)·F_B vorher). Reihenfolge: geringster Betrag des Seidel-Beitrags des Elements zuerst. Elementbudget (`--element-budget N`): verlangt ein Einfügen mehr als N Elemente, zuerst ein Rückbau-Versuch. Festlegungen in [optimierung.md, „Rückbau und Elementbudget“](../optimierung.md#rückbau-und-elementbudget-m6-t9).
- **F1 (NA 0,5, volles Feld):** Rückbau am realen System von F1 (Z2, 25 El., ohne Geisterplatten) mit ρ = 0,5: **9 Elemente**, nachoptimiert RMS **0,012/0,037/0,038 Wellen** bei 0/70/100 % Feld, Strehl 0,99/0,95/0,94 – besser als F1 mit 25 El. (0,039/0,022/0,059) und gleich viele Elemente wie der 9-Linser aus M6-T6 (0,015/0,024/0,033). Mit ρ = 1: **6 Elemente**, 0,041/0,038/0,068 Wellen (Strehl ≥ 0,84), noch beugungsbegrenzt.
- **JFV855 (NA 0,869, 23 % Feld):** Rückbau am realen System: **25 → 5 Elemente**, RMS 0,011/0,012/0,015 Wellen am Teilfeld (vorher 0,022/0,054/0,087). Von diesem 5-Linser aus öffnet die Feldstufe (Feldregel, Einfügen nur mit Verbesserung, Budget 12) das Feld bei NA 0,869 auf **64 %** (11 El.; vorher 23 % mit 25 El.); nachoptimiert 0,028/0,077/0,085 Wellen am Teilfeld, am Rand knapp über dem Ziel. F16 bestätigt: Die überzähligen Elemente der Feldregel verstellten den Weg.
- **Grenzen:** Mit Geisterplatten (Rückbau am Ende der Feldstufe statt am realen System) und während der Stufen ist der Rückbau schwach (F1: 25 → 23 bzw. 15 El.; JFV855: 0 von 6 bzw. 96 Versuchen), und ein Lauf ab der Einzellinse mit Rückbau in allen Stufen kam nur auf 58 % Feld (Abschnitt 3).

## 1. Umsetzung

| Option | Controller / Modell | Wirkung |
|---|---|---|
| `--prune` | `HomotopyOptions::prune` | Rückbau-Runden nach jeder Phase, in der eingefügt wurde, vor dem Abschluss und am letzten guten Zustand eines abgebrochenen Laufs |
| `--element-budget N` | `max_elements = N`, `prune_at_limit` | verlangt ein Einfügen mehr als N Elemente, zuerst eine Rückbau-Runde (je Punkt einmal); entfernt sie etwas, wird nicht eingefügt |
| `--prune-stages K`, `--prune-attempts N`, `--prune-factor ρ`, `--prune-tolerance ε`, `--prune-min-elements N` | `prune_stages` (3), `prune_attempts` (4), `prune_merit_factor` (1), `prune_tolerance` (aus), `prune_min_elements` (1) | Stufen der Rückbau-Homotopie, Kandidaten pro Runde, Annahmegrenze |

Schnittstelle des Modells: `removal_candidates` (Rangfolge), `collapse` (eine Stufe), `remove` (Null-Element entfernen). Das optische Modell nimmt während der Rückbau-Homotopie die Flächen des schrumpfenden Elements aus den mechanischen Bedingungen (sonst kostete z. B. die Randdicke 0,3 mm bei w = 10⁴/mm² 900 Wellen² und zöge die übrigen Variablen) und stellt danach die Standardauswahl wieder her. Protokoll: Schrittarten `removal_stage` (Optimierung einer Stufe) und `removal` (Entscheidung mit F_B, Grenze, Grund). Skript [`tools/m6t6_zielspezifikation.py`](../../tools/m6t6_zielspezifikation.py): `--rueckbau`, `--rueckbau-faktor`, `--rueckbau-versuche`, `--rueckbau-stufen`, `--elementbudget N`, `--nur-rueckbau` (ein synth-Lauf am Punkt der Startdatei, danach Z2), `--rueckbau-feld F`, `--rueckbau-zielfeld F`. Behoben: `remove_dummy_planes` ließ nach einem Rückbau an der Hilfsebene die vereinigte Spaltdicke doppelt in den Variablen („Variable doppelt“).

**Bitgleichheit ohne Option:** Unit-Tests (Abschnitt 4) und der Lauf `V1_basis/p1_r0` aus M6-T6b (Start `p0_start.json`, gleiche Optionen, GPU) mit dem neuen Programm: alle 51 Zeilen des Schrittprotokolls gleich dem archivierten `p1_r0.txt`.

## 2. Messung

Z2 = Ergebnis ohne Geisterplatten und Hilfsebenen monochromatisch nachoptimiert (2.000 Zyklen) und mit `mikroptim eval --grid 256 --reference exit-pupil` bewertet; RMS in Wellen bei 546 nm bei 0/70/100 % des erreichten Feldanteils, Strehl aus der FFT. „vorher“ = Zeilen aus M6-T8. Zeit = Summe der Programmaufrufe (GPU geteilt).

| Lauf | Start | Optionen | NA / Feld | El. vorher → nachher | Z2: RMS 0/70/100 % [Wellen] | Strehl (FFT) | Zeit [s] |
|---|---|---|---|---|---|---|---|
| F1 (M6-T8) | Einzellinse | Feldregel | 0,50 / 1,00 | 25 | 0,039/0,022/0,059 | 0,94/0,98/0,88 | 4.426 |
| 9-Linser (M6-T6, nachoptimiert in #105) | – | – | 0,50 / 1,00 | 9 | 0,015/0,024/0,033 | – | – |
| **F1R_z2** | Z2 von F1 | `--nur-rueckbau`, ρ = 0,5, 6 Versuche | 0,50 / 1,00 | 25 → **9** | **0,012/0,037/0,038** | 0,99/0,95/0,94 | 920 |
| F1R_z2_f1 | Z2 von F1 | `--nur-rueckbau`, ρ = 1, 30 Versuche | 0,50 / 1,00 | 25 → **6** | 0,041/0,038/0,068 | 0,94/0,95/0,84 | 1.386 |
| F1R_p2 | Ende der Feldstufe von F1 (mit Geisterplatten) | `--nur-rueckbau`, ρ = 0,5, 6 Versuche | 0,50 / 1,00 | 25 → 23 | 0,046/0,035/0,071 | 0,92/0,95/0,85 | 622 |
| F1R_p2_alle | ebenso | ρ = 0,5, 30 Versuche | 0,50 / 1,00 | 25 → 15 | 0,059/0,076/0,112 | 0,87/0,81/0,72 | 1.748 |
| F1R_feldregel_rueckbau | Einzellinse | Feldregel, `--rueckbau`, ρ = 0,5, 8 Versuche | 0,50 / 0,58 | p1: 5 → 3; p2_r0: 18 | – (Lauf zerfallen, s. u.) | – | 1.587 |
| JFV855 (M6-T8) | V2 `p1_final` | gemeinsam, Feldregel, Verbesserung 4 | 0,869 / 0,231 | 25 | 0,022/0,054/0,087 | 0,98/0,89/0,78 | 5.996 |
| JFV855R_p2 | Ende der Feldstufe von JFV855 (mit Geisterplatten) | `--nur-rueckbau`, ρ = 0,5, 6 Versuche | 0,869 / 0,231 | 25 → 25 | 0,021/0,054/0,087 | 0,98/0,89/0,78 | 413 |
| **JFV855R_z2** | Z2 von JFV855 | `--nur-rueckbau`, ρ = 0,5, 30 Versuche | 0,869 / 0,231 | 25 → **5** | **0,011/0,012/0,015** | 1,00/0,99/0,99 | 1.415 |
| **JFV855R_z2_feld** | JFV855R_z2 (5 El.) | Feldstufe 0,231 → 1: Feldregel, Verbesserung 4, `--elementbudget 12`, Rückbau ρ = 0,5 | 0,869 / **0,638** | 5 → 11 | 0,028/0,077/0,085 | 0,97/0,79/0,75 | 596 |
| JFV855R_budget12 | V2 `p1_final` | wie JFV855, dazu `--rueckbau`, `--elementbudget 12`, ρ = 0,5 | 0,867 / 0,209 | 5 → 12 | 0,026/0,053/0,083 | 0,97/0,90/0,77 | 3.502 |

Rückbau-Statistik (übernommen / Versuche): F1R_z2 16/45 (139 Optimierungen in 904 s), F1R_z2_f1 19/64, F1R_p2 2/11, F1R_p2_alle 10/75, JFV855R_p2 0/6, JFV855R_z2 20/49, JFV855R_z2_feld 0/11 (am Ende, nach dem Abbruch `step_too_small`), JFV855R_budget12 0/96 über 24 Runden (am Budget), F1R_feldregel_rueckbau 2/92.

Bei vollem Feld (Z2 der Teilfeld-Läufe auf 100 % Feld gesetzt, ohne Nachoptimierung): JFV855 6,0/6,7 Wellen am Rand (M6-T8), JFV855R_z2 1,9/3,3, JFV855R_z2_feld 0,17/1,8.

**Bewertung.**

1. **F1: deutlich weniger Elemente, weiterhin beugungsbegrenzt.** Der Rückbau am realen System von F1 entfernt 16 der 25 Elemente; der 9-Linser ist nachoptimiert auf der Achse besser als der 9-Linser aus M6-T6 (0,012 gegen 0,015) und am Rand etwas schlechter (0,038 gegen 0,033), Strehl ≥ 0,94. Mit ρ = 1 bleiben 6 Elemente mit 0,068 Wellen am Rand (knapp unter λ/14). Entfernt wurden vor allem die dünnen Menisken der Feldregel aus dem hochbrechenden Testglas; die Frontgruppe (CaF₂, SYN-36 an den Flächen 1–4) wurde in jeder Runde versucht und verworfen (F_B 0,015–0,1 Wellen²), ebenso zuletzt die dicken Elemente in der Mitte. Der Rangfolge nach dem Seidel-Beitrag folgt die Annahme nur grob (übernommen wurden Elemente mit 0,2 bis 24 Wellen Beitrag, verworfen die Frontgruppe mit ~3 Wellen); sie entscheidet nur die Reihenfolge, nicht die Annahme.
2. **JFV855: 25 → 5 Elemente, und das Feld öffnet sich weiter (F16).** Am realen System bei NA 0,869 und 23 % Feld tragen 20 der 25 Elemente nichts; der 5-Linser ist sogar besser (RMS ≤ 0,015 statt 0,087 am Rand: die Nachoptimierung konvergiert mit wenigen Variablen, F10). Ab dem 5-Linser kommt die Feldstufe bei derselben NA auf **64 % Feld** mit 11 Elementen (vorher 23 % mit 25); nachoptimiert liegt der Rand mit 0,085 Wellen knapp über dem Ziel. Ziel „Feld über 27 % bei hoher NA“ aus M6-T8: **erreicht** (64 %), das volle Feld nicht.
3. **Mit Geisterplatten und während der Stufen schwach.** Am Ende der Feldstufe (mit Geisterplatten, ρ·F_max = 2,6·10⁻³ Wellen²) ist F_B schon nahe an der Grenze, und die Platten übernehmen nach dem Entfernen Korrektur zu Kosten F_G: F1 nur 25 → 23 (6 Versuche) bzw. 15 (30 Versuche, Z2 dann schlechter: 0,112 Wellen am Rand), JFV855 nichts. Das Elementbudget allein (JFV855R_budget12) begrenzt die Elementzahl (12 statt 25) bei gleichem Ergebnis (NA 0,867, 21 % Feld, Z2 wie JFV855), entfernt aber nichts (0 von 96 Versuchen am Budget): Mitten in der Homotopie liegt jeder Zustand nahe an F_max. Ein Lauf ab der Einzellinse mit Rückbau in allen Stufen (F1R_feldregel_rueckbau) entfernt nach der Aperturstufe zwei Elemente (5 → 3), kommt in der Feldstufe dann nur auf 58 % (18 El.; F1: 71 % in der ersten Runde mit 24 El.); die folgenden Runden des Skripts zerfallen (Neustart nicht annehmbar, erzwungenes Einfügen mit F = 0,2 Wellen², Endzustand F_B = 568), Z2 ist unbrauchbar. Der Rückbau gehört nach diesen Messungen **an das Ende einer Stufe am realen System** (ohne Geisterplatten), nicht in die Stufen mit Geisterplatten.
4. **Empfohlener Ablauf** für die Zielspezifikation: Feldstufe mit Feldregel (Elemente großzügig), dann Z2 und Rückbau am realen System (`--nur-rueckbau` ab `z2_s0.json`), dann die nächste Stufe ab dem verkleinerten System (wie JFV855R_z2 → JFV855R_z2_feld). Dafür fehlt im Skript noch die Rückkehr vom realen System zu Geisterplatten (die Feldstufe ab dem 5-Linser lief ohne Geisterplatten, nur mit der Feldregel).

## 3. Befunde

| Nr. | Stand nach M6-T9 |
|---|---|
| F16 | **bestätigt und behoben, wo der Rückbau am realen System läuft:** F1 25 → 9 (bzw. 6) El., beugungsbegrenzt; JFV855 25 → 5 El., danach 64 % statt 23 % Feld bei NA 0,869 |
| F14 | bei hoher NA weiter geöffnet (64 % Feld bei NA 0,869 ab dem zurückgebauten 5-Linser), volles Feld nicht erreicht |
| F17 (neu) | Rückbau mit Geisterplatten bzw. mitten in der Homotopie entfernt kaum etwas (F_B nahe F_max, die Platten übernehmen die Korrektur); im Lauf ab der Einzellinse mit Rückbau in allen Stufen blieb die Feldstufe bei 58 % |

## 4. Tests

`tests/test_homotopy.cpp`: `HomotopyControl.PruneRemovesSuperfluousKeepsNecessary` (synthetisches Modell: notwendiges Element verworfen und Zustand zurück, überflüssiges entfernt, Anteile 1/3, 1/2, 1, Versuche pro Runde, Mindestzahl, ρ, ε, Text, JSON), `…PruneAfterPhaseWithInsertions` (auch am Ende nach einem Abbruch), `…ElementBudgetPrunesBeforeInserting`, `…PruneOffUnchangedAndInvalidOptions`; `OpticalHomotopy.RemovalOfNullElementRestoresSystem` (Orakel: Null-Meniskus hat Seidel-Beitrag 0 und ist erster Kandidat, Merit nach jeder Stufe und nach dem Entfernen relativ ≤ `kInsertionMeritRelative`, Flächen, Platten und Nebenbedingungen wie vor dem Einfügen); `tests/test_homotopy_optimization.cpp`: `HomotopySmoke.PruneRemovesNullElementKeepsLens` (CPU: Null-Meniskus entfernt, F_B unverändert, die Einzellinse versucht und verworfen); `tests/test_cli_synth.cpp`: `CliSynth.PruneAndElementBudget`; `python/tests/test_tool_m6t6.py::test_synth_options_prune`, `::test_remove_dummy_planes_merged_gap_once`.

## 5. Dateien

```
python tools/m6t6_zielspezifikation.py --gpu --nur-rueckbau --start docs/ergebnisse/m6-t8/F1_feldregel/z2_s0.json --rueckbau-faktor 0.5 --rueckbau-versuche 6 --out docs/ergebnisse/m6-t9/F1R_z2_rueckbau
python tools/m6t6_zielspezifikation.py --gpu --nur-rueckbau --start docs/ergebnisse/m6-t8/F1_feldregel/z2_s0.json --rueckbau-faktor 1.0 --rueckbau-versuche 30 --out docs/ergebnisse/m6-t9/F1R_z2_rueckbau_f1
python tools/m6t6_zielspezifikation.py --gpu --nur-rueckbau --rueckbau-feld 0.23142936250519994 --start docs/ergebnisse/m6-t8/JFV855_gemeinsam_feldregel_verbessern/z2_s0.json --rueckbau-faktor 0.5 --rueckbau-versuche 30 --out docs/ergebnisse/m6-t9/JFV855R_z2_rueckbau
python tools/m6t6_zielspezifikation.py --gpu --nur-rueckbau --start docs/ergebnisse/m6-t9/JFV855R_z2_rueckbau/z2_vollfeld_in.json --rueckbau-feld 0.23142936250519994 --rueckbau-zielfeld 1 --feldregel --einfuegen-verbessern 4 --elementbudget 12 --rueckbau-faktor 0.5 --rueckbau-versuche 30 --out docs/ergebnisse/m6-t9/JFV855R_z2_feld
python tools/m6t6_zielspezifikation.py --gpu --bis p2 --start docs/ergebnisse/m6-t6b/V2_apertur_zuerst/p1_final.json --na-feld 0.95 --gemeinsam --feldregel --einfuegen-verbessern 4 --rueckbau --elementbudget 12 --rueckbau-faktor 0.5 --out docs/ergebnisse/m6-t9/JFV855R_budget12
python tools/m6t6_zielspezifikation.py --gpu --bis p2 --feldregel --rueckbau --rueckbau-faktor 0.5 --rueckbau-versuche 8 --out docs/ergebnisse/m6-t9/F1R_feldregel_rueckbau
```

F1R_p2(_alle) und JFV855R_p2 wie die ersten Zeilen mit `--start …/p2_r11_restart.json` bzw. `…/p2_r23_restart.json` (Ende der Feldstufe mit Geisterplatten). `z2_vollfeld_in.json` ist im Archiv nicht enthalten; es entsteht beim Lauf JFV855R_z2_rueckbau neu. Je Lauf: `m6t6_lauf.json`, `m6t6_zeiten.json`, `lauf.log`, Textprotokolle `p*_r*.txt`, bei Einzelläufen `p2_r0_log.json`, Ergebnis der Stufe, Z2 (`z2_s0.json` und Bewertungen); mit `--archiv` verkleinert (3,3 MB). Glasdaten: nur `basis.json` und die frei erfundenen Testgläser.
