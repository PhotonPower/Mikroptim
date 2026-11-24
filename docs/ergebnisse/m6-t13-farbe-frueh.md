# M6-T13: Synthese mit „Farbe früh“, Feld und Apertur ab F3 öffnen

Aufgabe: [Meilensteinplan](../plan/meilensteine.md#m6-t13--synthese-mit-farbe-früh-feld-und-apertur-ab-f3-öffnen), freigegeben vom Maintainer am 2026-11-22. Ausgangslage: [M6-T11](m6-t11-farbe.md) (nur früh farbig entworfene Systeme lassen sich farbkorrigieren; bestes farbiges System F3: NA 0,6075, 38 % Feld, 16 El., 0,028/0,111/0,130 Wellen) und [M6-T12](m6-t12-glaswoerterbuch.md) (A1W4 → Farbstufe: NA 0,445, 24 % Feld, 11 El., beugungsbegrenzt, Strehl ≥ 0,87). Alle Zahlen aus Läufen auf dem Rechner des Maintainers (RTX 3090), FP64, Release mit CUDA 12.9, Referenzsphäre `image_infinite`, 16 Ringe × 8 Speichen, FD-Jacobi. Gläser nur `basis.json` und die frei erfundenen Testgläser `synthetisch.json`; keine Linsendaten realer Objektive.

**Kurzfassung.**

- **Etappenziel (NA 0,7 bei vollem Feld, 436–656 nm beugungsbegrenzt) nicht erreicht.** Höchste polychromatische NA: **0,6425** bei 41 % Feld (F3ZW, 19 El., 0,032/0,130/0,143 Wellen), bzw. **0,640** bei 41 % Feld mit 16 El. und 0,027/0,118/0,136 Wellen (F3Z, Strehl ≥ 0,66) – gegenüber F3 (NA 0,6075, 38 %) +0,03 NA und +3 % Feld bei gleicher Güte. Das Feld öffnet ab F3 in keinem Lauf über 43 %.
- **Diagnose der Grenze:** Am Feldrand begrenzt die **Kippung der Wellenfront** – Querfarbe und Verzeichnung (−0,5 %, die Merit misst gegen den idealen Bildpunkt f·tan θ). Ohne Kippung liegt F3Z bei 70/100 % Feld für 486–656 nm bei 0,022–0,065 Wellen (beugungsbegrenzt), mit Kippung bei bis zu 0,137. Die Farbregel erkennt das nicht: Die Seidel-Summen am Punkt zeigen Längsfarbe (ΔW020 ≈ −0,5 Wellen) und empfehlen Kittglieder an der Randstrahlhöhe; jedes Einfügen macht F_B schlechter (bis zu vier verworfene Varianten je Runde).
- **„Farbe früh“ ist Standard des Skripts** (`--farbe apertur`: Aperturstufe monochromatisch, dann stetige Farbstufe bei 10 % Feld, Feld und Apertur polychromatisch). Ab der Einzellinse liefert er bei NA 0,5 und 10 % Feld nach 7 Elementen ein **beugungsbegrenztes polychromatisches System** (0,060/0,047/0,057 Wellen, Strehl ≥ 0,87), die polychromatische Feldstufe endet dann aber bei 19 % (Befund F14 bleibt). Der alte monochromatische Pfad bleibt als `--farbe spaet` (bitgleich).
- **Umgesetzt** (Optionen, ohne sie bitgleich): Varianten der Farbregel ohne doppelte Einfügung (`synth --color-variants`), Vergrößern der Schrittweite mit Abstand zur Schwelle (`--grow-margin Q`), Einfügen nur mit Verbesserung auch im Abschluss (`--insertion-improve-finish`), Querfarbe → Kittglied-Modul nach |y·ȳ| (`--color-module-lateral`); unter `--insertion-improve` gilt eine Variante, deren Optimierung wirft, als verworfen statt synth abzubrechen.
- **Erweitertes Wörterbuch** (`--woerterbuch-extra 4`) in den Öffnungsstufen: ab F3 etwas mehr NA (0,6425 gegen 0,640), aber schlechter am Rand und 19 statt 16 El.; ab A1W4 besser (41 % Feld bei 0,101 Wellen am Rand gegen 40 % mit 0,144 Wellen und versagten Strahlen). Kein eindeutiger Gewinn; Kosten etwa ×1,4.
- **Abgebrochen:** alle Läufe am 2026-11-24 17:50 (Vorgabe des Koordinators, Zeitrahmen überschritten). Der Rechner war von 01:01 bis 17:47 im Ruhezustand; danach zeigten die Läufe keinen Fortschritt mehr. Ausgewertet sind die abgeschlossenen Stufen (Abschnitt 3).

## 1. Umsetzung

| Punkt der Aufgabe | Umsetzung | Option |
|---|---|---|
| 1. „Farbe früh“ als Standardpfad | `--farbe spaet\|frueh\|apertur\|anfang`, Standard `apertur`: nach der monochromatischen Aperturstufe (NA 0,3 → 0,5 bei 10 % Feld) die Farbstufe p1f mit stetiger Spektrum-Homotopie direkt zu 436–656 nm, Feld- und Aperturstufe danach polychromatisch, Z2 am Spektrum der Feldstufe; Werkzeuge `COLOR_TOOLS` (Farbregel, Kriterium d, Modul, stetige Farbe, Feldregel, Einfügen nur mit Verbesserung, Farbbudget 4, Varianten, Wachsen 0,8, Abschluss). `spaet` = Standard bis M6-T12 (bitgleich), `frueh`/`anfang` wie die alten Schalter. Begründung: Abschnitt 2 | Skript `--farbe` |
| 3. Diagnose und kleine Verbesserungen | Varianten der Farbregel ohne Rückfall (vorher viermal dieselbe Einfügung), Wachsen mit Abstand zur Schwelle (die stetige Farbstufe kroch mit Δ = 1/1024), Einfügen nur mit Verbesserung im Abschluss (Lauf H2: Einfügungen nach verfehltem Ziel zerstörten das System), werfende Variante verworfen (Lauf H1G8: freie Glasvariable n ≤ 0 brach synth ab), Querfarbe → Modul nach \|y·ȳ\| | `synth --color-variants`, `--grow-margin Q`, `--insertion-improve-finish`, `--color-module-lateral`; Skript `--farb-varianten`, `--wachsen-marge`, `--einfuegen-abschluss`, `--farb-querfarbe` |
| 4. Tests, Doku | Abschnitt 5; optimierung.md („Varianten der Farbregel“, „Farbe früh im Hauptlauf“, Abschnitt 7), cli.md, CHANGELOG, Plan | – |

Einzig die verworfene werfende Variante ist keine eigene Option: Sie wirkt mit `--insertion-improve` und ändert nur Läufe, die vorher mit Fehler endeten (Frage 3 an den Maintainer).

## 2. Hauptlauf ab der Einzellinse: welche „Farbe früh“?

Startsystem der Zielspezifikation (Einzellinse, NA 0,3), `--elementbudget 16`.

| Lauf | Ablauf | Ergebnis | Zeit [s] |
|---|---|---|---|
| H1 (`--farbe apertur`, ohne Wachsen) | p1 NA 0,3 → 0,5 (5 El.) → p1f stetig | p1f: 109 Schritte mit Δ = 1/1024 (Δ wächst nie, alle LM-Läufe `max_iterations`), Breite 0,16 nach ~45 min, F_B sank dabei von 5,1·10⁻³ auf 1,4·10⁻³; **abgebrochen** und als H1G8 fortgesetzt | 1.119 (p1) + ~2.700 |
| H1G8 (`apertur` + `--wachsen-marge 0.8 --farb-varianten --einfuegen-abschluss`) | p1 wie H1, p1f, p2 Feld, p3 Apertur | **p1f: volle Breite 436–656 nm mit 7 El.** (Kittglied nach Operator B, danach Δ bis 1/8); Z1f bei NA 0,5, 10 % Feld: **0,060/0,047/0,057 Wellen, Strehl ≥ 0,871**. p2: Feld 10 → **19,4 %** (11 El.), p3: NA 0,5 → 0,53, Z3 0,103/0,078/0,118 Wellen (Strehl 0,60); **abgebrochen** in p3 | 1.119 + 4.999 + 5.680 + 3.903 |
| H2 (`--farbe anfang`) | 436–656 nm ab dem Start | p1 erreichte NA 0,5 mit 6 El. bei F_B = 4,9·10⁻³ < F_max, aber größter RMS pro Tupel 0,126 Wellen (Ziel verfehlt); jedes Einfügen im Abschluss verschlechterte (F_B bis 305 Wellen², 15 El.), dann Abbruch von synth (Operator B, Glasvariable n ≤ 0) | 5.168 |
| V3 (M6-T6b, `--farbe frueh`, alte Werkzeuge) | Farbe nach der Feldstufe | NA 0,535, 27 % Feld, 12 El., ≤ 0,108 Wellen | 4.895 |
| P (M6-T11, `anfang --gemeinsam`) | – | NA 0,363, 37 % Feld, 8 El., ≤ 0,253 | 8.599 |

**Wahl des Standards `apertur`:** Die Farbe kommt ins System, solange es klein ist (5 El. bei NA 0,5) – genau dort ist sie lokal korrigierbar (M6-T11: ab den großen monochromatischen Ständen nicht mehr). Der Lauf liefert das kleinste beugungsbegrenzte polychromatische System bisher (7 El., NA 0,5, 10 % Feld). `anfang` lässt die Geisterplatten die Farbe tragen (Platten mit Dispersion), das reale System bleibt farbig, und der Abschluss zerstört es. `frueh` (Farbe nach der Feldstufe) erreichte mit älteren Werkzeugen mehr Feld (27 %), baut aber das Feld monochromatisch mit den drei Wörterbuchgläsern auf. Der Engpass des Standardpfads ist danach die **polychromatische Feldstufe** (19 %): F14 gilt auch polychromatisch. Ohne `--wachsen-marge` braucht die stetige Farbstufe Stunden (H1); deshalb ist sie Teil der Werkzeuge des Standards.

## 3. Öffnungsstufen ab F3 und ab A1W4 → Farbstufe

Zyklus `--zyklus feld,apertur --zyklus-runden 4 --farb-budget 6 OPT` (OPT wie M6-T11: `--farb-stetig --farb-ziel voll --farb-modul doublet --farb-chromatisch 0.5 --feldregel --einfuegen-verbessern 4 --rueckbau-punkt 0.85`), Elementbudget 20 (F3) bzw. 18 (A1W4). Bewertung nach dem Rückbau (`mikroptim eval --grid 256`, Referenz Austrittspupille), größter RMS über 436–656 nm bei 0/70/100 % des erreichten Feldanteils, Strehl = kleinster FFT-Strehl über 436–656 nm. Zeit = Summe der Programmaufrufe der Stufe (bis zu acht Läufe gleichzeitig auf der GPU – grob).

| Lauf | Start | Stufe | NA | Feld | Spektrum | El. (Start → Stufe → Rückbau) | RMS 0/70/100 % | Strehl min | Zeit [s] |
|---|---|---|---|---|---|---|---|---|---|
| F3 (M6-T11) | V3 | c1_apertur | 0,6075 | 0,381 | 405–707 | 15 → 16 → 16 | 0,028/0,111/0,130 | 0,64 | 3.421 |
| **F3Z** | F3 | c0_feld | 0,6075 | 0,412 | 405–707 | 16 → 16 → 16 | 0,020/0,114/0,133 | 0,69 | 6.249 |
| **F3Z** | | c1_apertur | **0,640** | 0,412 | 405–707 | 16 → 16 → 16 | 0,027/0,118/0,136 | 0,66 | 7.948 |
| F3Z | | c2_feld | 0,640 | 0,412 | 405–707 | 16 → 17 → 17 | 0,027/0,117/0,134 | 0,67 | 9.705 |
| F3ZV (`--farb-varianten`) | F3 | c0_feld | 0,6075 | **0,428** | 405–707 | 16 → 19 → 19 | 0,033/0,119/0,135 | 0,67 | 11.490 |
| F3ZV | | c1_apertur | 0,6375 | 0,428 | 405–707 | 19 → 19 → 19 | 0,029/0,119/0,134 | 0,65 | 9.000 |
| F3ZW (`--woerterbuch-extra 4`) | F3 | c0_feld | 0,6075 | 0,412 | 405–707 | 16 → 18 → 18 | 0,020/0,116/0,131 | 0,68 | 6.837 |
| F3ZW | | c1_apertur | **0,6425** | 0,412 | 405–707 | 18 → 19 → 19 | 0,032/0,130/0,143 | 0,53 | 13.157 |
| A1W4_farbe (M6-T12) | A1W4 | c0_farbe | 0,445 | 0,241 | 436–656 | 8 → 13 → 11 | 0,025/0,068/0,060 | 0,87 | 8.275 |
| A1Z | A1W4_farbe | c0_feld | 0,445 | 0,397 | 436–656 | 11 → 17 → 18 | 0,026/0,108/0,144 (70 versagte Strahlen) | 0,57 | 17.246 |
| **A1ZW** (`--woerterbuch-extra 4`) | A1W4_farbe | c0_feld | 0,445 | **0,412** | 436–656 | 11 → 16 → 16 | 0,055/0,082/0,101 | **0,77** | 14.251 |

Abgebrochene Stufen ohne Bewertung (Läufe beendet 2026-11-24 17:50): F3Z c3_apertur (ab NA 0,640), F3ZV c2_feld (Feld blieb 42,8 %), F3ZW c2_feld (44,4 % mit 20 El. am Budget), F3ZQ c0_feld (alle Verbesserungen zusammen, im ersten synth-Lauf), A1Z c1_apertur (NA 0,445 → 0,445, am Elementbudget 18), A1ZW c1_apertur (NA 0,445 → 0,4825, 17–18 El.). Die Protokolle der synth-Läufe liegen bei.

Beobachtungen:

- **Apertur ab F3** wächst je Runde um 0,005–0,02 NA und bleibt bei 0,64 stehen; **Feld** öffnet einmal um 3–5 % (38 → 41–43 %) und dann nicht mehr. Alle Stillstände: „kein Einfügen verbessert F_B“ knapp über dem Stand davor.
- **Rückbau bei 85 %** entfernt ab F3 fast nichts (1 von 109 Versuchen, F3ZW c0), ab A1W4 2 bzw. 5 Elemente (A1ZW, A1Z) – dort, wo die Stufe Elemente einfügte, die am Ende nichts tragen.
- **Varianten der Farbregel** (F3ZV): Statt viermal derselben Einfügung (F3Z, wie in M6-T11) versucht der Controller verschiedene Linsen und Modul-Spalte und übernimmt zweimal ein Null-Kittglied (16 → 19 El.); Ergebnis: mehr Feld (42,8 gegen 41,2 %), gleiche Güte, etwas weniger NA.
- **Erweitertes Wörterbuch** (F3ZW, A1ZW): Die Zusatzplatten werden auch hier nie zur Empfehlung (F19 aus M6-T12 bestätigt); ab A1W4 liefert der Lauf das bessere Ergebnis (Rand 0,101 Wellen, Strehl 0,77 gegen 0,144/0,57), ab F3 das schlechtere bei 0,0025 mehr NA. Mit zwei Läufen ist das nicht von der Streuung der Wege zu trennen; als Standard für die Öffnungsstufen nicht empfohlen.

## 4. Diagnose der Grenze

Bewertung des besten Systems F3Z c1 (NA 0,640, 41 % Feld; Feldwinkel 0/1,256/1,792°), RMS in Wellen bei λ, mit und ohne Kippung (Referenz Austrittspupille):

| λ [nm] | Achse | 70 % mit / ohne Kippung | 100 % mit / ohne Kippung | Verzeichnung 100 % |
|---|---|---|---|---|
| 436 | 0,027 | 0,118 / 0,067 | 0,106 / 0,102 | −0,48 % |
| 486 | 0,015 | 0,043 / 0,039 | 0,137 / 0,065 | −0,51 % |
| 546 | 0,021 | 0,035 / 0,027 | 0,105 / 0,042 | −0,42 % |
| 588 | 0,024 | 0,050 / 0,025 | 0,068 / 0,037 | −0,33 % |
| 656 | 0,021 | 0,084 / 0,022 | 0,036 / 0,034 | −0,18 % |

- Ohne Kippung ist das System bis auf 436 nm am Rand (0,102) und 436 nm bei 70 % (0,067, knapp) beugungsbegrenzt. Die Kippung ist **Verzeichnung** (gemeinsamer Anteil, −0,4 %) plus **Querfarbe** (die Kippung ändert sich mit λ von 0,01 bei 656 nm bis 0,12 Wellen bei 486 nm am Rand). Die Merit bestraft beides, weil die Referenzsphäre um den idealen Bildpunkt f·tan θ liegt (ADR-0017, Punkt 3).
- Die **Farbregel** sieht am Punkt (41 % Feld) aus den Seidel-Summen vor allem Längsfarbe (ΔW020 ≈ −0,5, ΔW111 ≈ 0,25 Wellen); sie teilt Linsen bzw. setzt Null-Kittglieder an die größte Randstrahlhöhe. Das korrigiert die Querfarbe nicht. Die neue Option `--color-module-lateral` (Modul nach |y·ȳ|) greift deshalb in F3ZQ nicht (Kriterium |ΔW111|/4 > |ΔW020|/(2√3) am Punkt nicht erfüllt); ihr Nutzen ist ungemessen.
- **Geisterplatten:** Die zurückgekehrten Platten (Blendennähe und Feldlage) melden in den Öffnungsstufen keinen Bedarf; alle Einfügungen kommen aus Farb- und Feldregel (wie M6-T10/T12).
- **F10:** Fast alle LM-Läufe enden mit `max_iterations`; Δ wächst ohne `--grow-margin` nie (H1).

## 5. Tests

| Test | Orakel / Ergebnis |
|---|---|
| `ColorStage.ColorVariantsAreDistinctInsertions` | Dreilinser mit nicht teilbarer Mittellinse; Orakel = Fläche, an der `insert` tatsächlich einfügt: ohne Option mindestens eine Variante gleich der Empfehlung, mit Option alle verschieden (Linsen 1, 5, Modul-Spalte 0, 2, 4), keine nicht teilbare Linse; eine nicht teilbare Empfehlung wirft bei `insert` und entfällt als Variante; mit aktiver Platte letzte Variante = `diagnose` ohne Farbregel |
| `ColorStage.LateralColorModuleOrderedByChiefRayHeight` | Dreilinser mit 20° Feld: Seidel-Orakel ΔW111/4 = 1,80 > ΔW020/(2√3) = 0,84; Modul zuerst, Spalte nach \|y·ȳ\| aus dem Aberrationsbericht (andere Reihenfolge als nach \|y\|); ohne Option Operator B |
| `HomotopyControl.GrowMarginGrowsAfterSlowStepsBelowThreshold` | Mock (langsame Schritte F_B = 0,5 F_max): mit 0,5 Δ 0,02/0,04/0,08/0,1/0,1, alle `grow`; mit 0,4 Folge gleich ohne Option; 1,5 wirft |
| `HomotopyControl.InsertionMustImproveInFinishPhase` | Mock mit verfehltem Ziel: schlechte Varianten verworfen, `goal_not_reached` am Zustand vor dem Abschluss; bessere Variante übernommen; ohne Teiloption wie bisher |
| `HomotopyControl.ThrowingBranchIsDiscarded` (erweitert) | werfende Variante unter `insertion_must_improve` verworfen (Vermerk mit Fehler), geordnetes Ende; ohne Option wirft der Lauf wie bisher |
| `CliSynth.ColorStageOptions` (erweitert) | `--color-variants`, `--color-module-lateral`, `--grow-margin`, `--insertion-improve-finish`: Hilfe, Fehlbedienung Exit 2, JSON nur mit Option, Einzellinse mit `--color-variants` gleiches Ergebnis |
| Python `test_tool_m6t6.py::test_color_mode_default_is_color_early`, `::test_color_mode_late_and_legacy_unchanged`, `::test_apply_color_mode`, `::test_synth_options_color_variants` | Folge der Stufen mit aufgezeichneten Aufrufen (p1 → p1f → p2 → p3; `spaet` p1 … p5 mit `--stall 0` wie bisher; `--farbe-frueh`; `anfang`), Werkzeuge, Durchreichen der Optionen |

Lokal: `ctest --preset release-cuda` (RTX 3090) 1142/1142 bestanden; GCC-Prüfbau (MSYS2, Warnungen als Fehler) 948/948.

## 6. Vergleich

| Größe | M6-T11 | M6-T12 | M6-T13 |
|---|---|---|---|
| höchste NA polychromatisch | F3: 0,6075 bei 38 %, 16 El., ≤ 0,130 | F3W2: 0,575 bei 41 % | **F3Z: 0,640 bei 41 %, 16 El., ≤ 0,136**; F3ZW 0,6425 (19 El., ≤ 0,143) |
| größtes Feld polychromatisch, NA ≥ 0,6 | 38 % | – (F3W4: 43 % bei 0,5525) | 42,8 % bei 0,6375 (F3ZV, 19 El.) |
| beugungsbegrenzt 436–656 nm | – | A1W4_farbe: NA 0,445, 24 %, 11 El., Strehl ≥ 0,87 | Z1f (H1G8): NA 0,5, 10 %, **7 El.**, Strehl ≥ 0,87 |
| ab A1W4 | – | 24 % | A1ZW: 41 % bei NA 0,445, ≤ 0,101, Strehl 0,77 (nicht beugungsbegrenzt) |
| Rückbau | 0/114 | erstmals in Farbstufen | ab F3 1/109, ab A1W4 2 bzw. 5 El. |

## 7. Befunde

| Nr. | Stand nach M6-T13 |
|---|---|
| F13 | „Farbe früh“ ist Standard; ab der Einzellinse polychromatisch beugungsbegrenzt bei NA 0,5 mit 7 El. (10 % Feld) |
| F14 | gilt auch polychromatisch: Feldstufe ab dem 7-Linser endet bei 19 %, ab F3 bei 41–43 % |
| F19 | bestätigt: Zusatzplatten empfehlen auch in den Öffnungsstufen nie ein Einfügen |
| F20 (neu) | Grenze am Feldrand ist die Kippung: Querfarbe und Verzeichnung (−0,5 %); ohne Kippung beugungsbegrenzt bis auf 436 nm. Farbregel und Geisterplatten sehen das nicht |
| F21 (neu) | Varianten der Farbregel waren oft dieselbe Einfügung (Rückfall auf die Empfehlung); behoben mit `--color-variants` |
| F22 (neu) | Stetige Farbstufe kriecht mit Δ_min, weil kein LM-Lauf „schnell“ ist (F10); behoben mit `--grow-margin` |
| F23 (neu) | Abschluss nach verfehltem Ziel fügt auch dann ein, wenn jedes Einfügen verschlechtert (H2: 6 → 15 El., F_B 0,005 → 305); behoben mit `--insertion-improve-finish` |

## 8. Vorschläge (nicht umgesetzt)

1. **Verzeichnung von der Wellenfront trennen:** Kippung des Hauptstrahls bei λ_ref als eigene, schwächer gewichtete Nebenbedingung (Verzeichnung ≤ x %) und die Merit gegen den Hauptstrahl-Durchstoßpunkt bei λ_ref; Querfarbe bliebe als Unterschied der Wellenlängen in der Merit. Merit-Änderung, braucht ADR (ADR-0017, Punkt 3).
2. **Querfarbe aus realen Strahlen** statt Seidel am Punkt für die Farbregel (Kippungsdifferenz am größten Feld), damit `--color-module-lateral` am Feldrand greift.
3. **Einfügen in der Feldlage mit Feldbedarf** (F14) auch polychromatisch: Feldregel-Varianten früher in der Liste.

## 9. Fragen an den Maintainer

1. Standard des Skripts ist jetzt `--farbe apertur` mit allen Werkzeugen (`COLOR_TOOLS`); ältere Läufe brauchen `--farbe spaet`. Einverstanden, oder lieber `frueh` als Standard (mehr Feld, aber monochromatischer Feldaufbau)?
2. Sollen `--color-variants`, `--grow-margin 0.8` und `--insertion-improve-finish` Standard von `synth` werden? (gemessen nie schlechter; ändern alle Referenzläufe mit `--insertion-improve`)
3. Verworfene werfende Variante unter `--insertion-improve` ist ohne eigene Option umgesetzt (ändert nur Läufe, die vorher mit Fehler endeten). In Ordnung?
4. Vorschlag 1 (Verzeichnung als eigene Bedingung) als Aufgabe mit ADR?

## 10. Dateien

```
python tools/m6t6_zielspezifikation.py --gpu --elementbudget 16 --wachsen-marge 0.8 --farb-varianten --einfuegen-abschluss --out docs/ergebnisse/m6-t13/H1G8
python tools/m6t6_zielspezifikation.py --gpu --farbe anfang --elementbudget 16 --out docs/ergebnisse/m6-t13/H2_anfang
python tools/m6t6_zielspezifikation.py --gpu --zyklus feld,apertur --zyklus-runden 4 --start docs/ergebnisse/m6-t11/F3_v3/c1_apertur_z2r_s2.json --elementbudget 20 --farb-budget 6 OPT --out docs/ergebnisse/m6-t13/F3Z
python tools/m6t6_zielspezifikation.py ... wie F3Z ... --farb-varianten --out docs/ergebnisse/m6-t13/F3ZV
python tools/m6t6_zielspezifikation.py ... wie F3Z ... --woerterbuch-extra 4 --out docs/ergebnisse/m6-t13/F3ZW
python tools/m6t6_zielspezifikation.py ... wie F3Z ... --farb-varianten --farb-querfarbe --wachsen-marge 0.8 --einfuegen-abschluss --out docs/ergebnisse/m6-t13/F3ZQ
python tools/m6t6_zielspezifikation.py --gpu --zyklus feld,apertur --zyklus-runden 4 --start docs/ergebnisse/m6-t12/A1W4_farbe_ohne/c0_farbe_z2r_s2.json --elementbudget 18 --farb-budget 6 OPT --out docs/ergebnisse/m6-t13/A1Z
python tools/m6t6_zielspezifikation.py ... wie A1Z ... --woerterbuch-extra 4 --out docs/ergebnisse/m6-t13/A1ZW
```

mit `OPT` wie in Abschnitt 3. H1_apertur enthält den abgebrochenen Lauf H1 (bis zur kriechenden Farbstufe), H1G8 setzt ihn ab p1 fort; die Bewertung der Zwischenstände Z1f (`z1f_*`) entstand nach dem Abbruch aus `p1f_final.json` (Nachoptimierung ohne Platten, 2.000 Zyklen, Spektrum der Stufe); die Dateien von Z3 (aus `p3_r2.json`) gingen beim Verkleinern mit `--archiv` verloren, die Zahlen stammen aus der Bewertung davor. Die Verzeichnisse sind mit `--archiv` verkleinert.

## M6-T13b: Nachmessung mit den neuen Standards

Freigabe Maintainer 2026-11-24 (Antworten auf Abschnitt 9): Frage 1 – `--farbe apertur` bleibt Standard des Skripts; Frage 2 – `--color-variants`, `--grow-margin 0.8` und `--insertion-improve-finish` sind seit M6-T13b Standard von `synth` (Abschalter `--no-color-variants`, `--grow-margin 0`/`--no-grow-margin`, `--no-insertion-improve-finish`, bitgleich zum Stand davor; [optimierung.md](../optimierung.md#weitere-steuerung-der-öffnungsstufen-m6-t13)); Frage 3 – die werfende Variante bleibt ohne eigene Option verworfen; `insert_module` verwirft Schalen mit Q ≤ 0. `--split-preselect` und `--woerterbuch-extra` bleiben aus.

**Lauf H1S:** Standardpfad des Skripts ab der Einzellinse wie H1G8, jetzt in einem Zug mit den neuen Standards (auch die monochromatische Aperturstufe p1 mit Q = 0,8; in H1G8 stammte p1 aus dem Lauf H1 ohne Wachsen):

```
python tools/m6t6_zielspezifikation.py --gpu --elementbudget 16 --out docs/ergebnisse/m6-t13b/H1S
```

| Stufe | H1G8 (M6-T13) | H1S (M6-T13b) |
|---|---|---|
| p1 (NA 0,3 → 0,5, 10 % Feld, mono) | 5 El., 1.119 s | 5 El., **44 s** (ein erzwungenes Einfügen) |
| p1f (Farbe 436–656 nm) | volle Breite mit 7 El., 4.999 s; Z1f 0,060/0,047/0,057 Wellen, Strehl ≥ 0,87 | Breite **0,97** mit 9 El. (Farbbudget 4 erreicht, vier Runden), 588 s; danach mit vollem Spektrum weiter |
| p2 (Feld) | 10 → 19,4 %, 11 El., 5.680 s | 10 → **14,7 %**, 15 El., 299 s |
| p3 (Apertur) | 0,5 → 0,53 (abgebrochen), 3.903 s | 0,5 → **0,5275**, 15 El. (Elementbudget, kein Spalt für ein erzwungenes Einfügen), 513 s |
| Endstand (436–656 nm, RMS 0/70/100 % des Feldanteils) | Z3: 0,103/0,078/0,118, Strehl ≥ 0,60 | Z2: **0,102/0,065/0,115**, Strehl ≥ 0,60 (486 nm am Rand; Achse 0,102 bei 436 nm) |
| Gesamtzeit | ≈ 15.700 s (abgebrochen) | **1.473 s** (vollständig, mit Bewertung) |

Bewertung: `mikroptim eval --grid 256` wie in Abschnitt 3 (Referenz Austrittspupille), Z2 = Nachoptimierung ohne Platten am Spektrum der Feldstufe. Auf 405–707 nm (p7/p8) liegt der Rand bei bis zu 0,41 Wellen (405 nm), wie in M6-T13 nicht korrigiert.

- **Laufzeit:** Mit Q = 0,8 ab dem Start ist der Standardpfad etwa zehnmal so schnell und läuft ohne Abbruch durch (H1G8 brauchte allein für p1f 83 min). Die Zeiten in p1 sanken von 1.119 auf 44 s, vermutlich weil die Aperturstufe nach langsamen Schritten wieder wächst (F22 gilt auch für die Apertur).
- **Güte:** Endstand vergleichbar mit H1G8 Z3 (gleiche NA 0,53, ähnlicher größter RMS, Strehl 0,60), aber bei weniger Feld (14,7 statt 19,4 %) und mit mehr Elementen (15 statt 11). Kein beugungsbegrenzter Zwischenstand wie Z1f: Die schnellere Aperturstufe endet bei einem anderen 5-Linser, dessen Farbstufe mit dem Farbbudget 4 (9 El.) nur 97 % der Breite erreicht – das reale System ist danach farbiger, und Feld- und Aperturstufe verbrauchen das Elementbudget 16.
- **Folgerung:** Die neuen Standards verkürzen die Läufe stark, ändern aber den Weg; der Engpass bleibt die polychromatische Feldstufe (F14). Ein Einzellauf trennt das nicht von der Streuung der Wege (vgl. M6-T11b, Median gestörter Starts). Vorschlag (nicht umgesetzt): Farbbudget der Stufe p1f für `--farbe apertur` an der erreichten Breite messen (etwa 5 statt 4) bzw. mehrere gestörte Starts.

Dateien: `docs/ergebnisse/m6-t13b/H1S` (mit `--archiv` verkleinert, Programmausgabe in `lauf.log`).
