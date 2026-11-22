# M6-T12: Erweitertes Glas-Wörterbuch der Geisterplatten

Aufgabe: [Meilensteinplan](../plan/meilensteine.md#m6-t12--glas-wörterbuch-der-geisterplatten-um-flint--und-anomale-gläser-erweitern), freigegeben vom Maintainer am 2026-11-22. Ausgangslage: [Bericht M6-T11](m6-t11-farbe.md) – die monochromatischen Syntheseläufe bauen nur aus den drei Gläsern des Wörterbuchs (CaF₂, SYN-30, SYN-36) und sind danach nicht farbkorrigierbar (σ_col 16–24 Wellen, F13). Alle Zahlen aus Läufen auf dem Rechner des Maintainers (RTX 3090, bis zu fünf Läufe gleichzeitig – Laufzeiten grob), FP64, Release mit CUDA 12.9, Referenzsphäre `image_infinite`, 16 Ringe × 8 Speichen, FD-Jacobi. Gläser nur `basis.json` und die frei erfundenen Testgläser `synthetisch.json`; keine Linsendaten realer Objektive, keine erfundenen Glasdaten.

**Kurzfassung.**

- **Umgesetzt** (Option, ohne sie bitgleich; [optimierung.md, „Erweitertes Wörterbuch“](../optimierung.md#erweitertes-wörterbuch-m6-t12)): `extend_ghost_dictionaries` setzt an jeder Lage der Geisterplatten Platten mit Koeffizienten 0 aus Zusatzgläsern dazu (Merit unverändert), abwechselnd das beste Flint- und das beste anomale Glas (|ΔP_g,F| ≥ 0,01, `anomalous_pgf`) nach dem spektralen Rest des besten Achromats über 436–656 nm (Rangfolge der Glasvorauswahl M11-T5g). Die Auswahl hängt nicht vom System ab, vorhandene Gläser zählen mit; daher ist die Erweiterung wiederholbar ohne Wirkung. `synth --ghost-dictionary-extra N`, `--ghost-dictionary-glass GLAS`, Skript `--woerterbuch-extra [N]`, Aktion `synth_segment` mit `ghost_dictionary_extra`.
- **Auswahl** mit den Testkatalogen: SYN-32 (Flint, 1,769/48,4), CaF₂ (anomal, schon im Wörterbuch), SYN-13 (Flint, 1,573/47,8), SYN-04 (anomal, FK51-artig, 1,474/88,5). N = 2 setzt eine, N = 4 drei neue Platten je Lage.
- **Monochromatisch (a):** Mit N = 4 baut der Feldregel-Pfad ab der Einzellinse seine Menisken auch aus SYN-32, SYN-13, SYN-04 (Operator A aus deren Platten). Er endet früher (NA 0,445, 24 % Feld, 8 El. statt NA 0,5, 100 %, 25 El.), aber **farbfähig**: Längsfarbe (Fokusspanne g…C) 75 µm statt 298 µm (9-Linser M6-T9), σ_col 8,3 statt 21,5 Wellen (bei vollem Feldwinkel). Die anschließende Farbstufe wie M6-T11 erreicht 436–656 nm **beugungsbegrenzt**: 0,025/0,068/0,060 Wellen (0/70/100 % des Feldes), Strehl ≥ 0,87, 11 El. nach dem Rückbau (der Rückbau entfernt zum ersten Mal in einer Farbstufe zwei Elemente). Ab den monochromatischen Ständen von M6-T9/M6-T10 kam M6-T11 nur auf 0,46–18,7 Wellen. Mit N = 2 (nur SYN-32) verläuft der Pfad wie bisher (NA 0,5, 69 % Feld, 26 El., σ_col 17,6).
- **Farbe früh (b), F3 mit Erweiterung:** etwa gleiche Güte, mehr Feld, weniger NA: N = 2 NA 0,575 bei 41 % Feld, 18 El., 0,022/0,113/0,128 Wellen (M6-T11: NA 0,6075 bei 38 %, 16 El., 0,028/0,111/0,130); N = 4 NA 0,5525 bei 43 %, nach dem Rückbau 13 El., 0,068/0,102/0,110 (Strehl ≥ 0,73). In den polychromatischen Stufen empfehlen die Zusatzplatten selbst nie ein Einfügen; die Einfügungen kommen aus Farbregel und Feldregel. Ein Vorteil zeigt sich nur indirekt (mehr Feld, Rückbau auf 13 El. bei N = 4) und ist nicht von der Streuung der Läufe zu trennen.
- **Standardgröße N = 4** (Skript `--woerterbuch-extra` ohne Zahl), für die monochromatischen Stufen; Kosten +38 % LM-Zeit je Zyklus (12 statt 6 Platten), N = 6 +80 %. In den Farbstufen ab einem schon farbfähigen Stand die Option weglassen (Farbstufe mit Erweiterung: 14 El., 0,039/0,091/0,082 gegen 11 El., 0,025/0,068/0,060 ohne).
- Ziel (NA 0,95, volles Feld, 436–656 nm beugungsbegrenzt) nicht erreicht.

## 1. Umsetzung

| Punkt der Aufgabe | Umsetzung |
|---|---|
| 1. Wörterbuch konfigurierbar erweitern | `include/mikroptim/synth/ghost_dictionary.hpp`: `select_ghost_dictionary_glasses` (Rangfolge, Klassen, genannte Gläser, Merkmal „vorhanden“), `extend_ghost_dictionaries` (eine Platte je neuem Glas und Lage über `place_ghost_dictionary`, Ordnung und r_g der ersten Platte der Lage). CLI `synth --ghost-dictionary-extra N`, `--ghost-dictionary-glass GLAS` (nach `--start-from-real` an den neuen Platten; JSON `options.ghost_dictionary`, Textkopf „Glas-Wörterbuch (M6-T12)“, nur mit Option). Skript `--woerterbuch-extra [N]` (Standard 4) und `--woerterbuch-glas`, an jeden `synth`-Aufruf durchgereicht. `mikroptim.design`: `synth_segment` mit `ghost_dictionary_extra` (1–4) |
| 2. Diagnose/Operatoren | Keine Änderung nötig: Die Diagnose wertet jede Platte mit ihrem Glas aus, Operator A nimmt das Glas der aktivsten Platte, Operator B das Glas mit der größten Abweichung \|1/ν_Platte − 1/ν_Linse\| als Zielglas; die Zusatzplatten sind Katalogglas, Einrasten betrifft nur Glasvariablen. Geprüft mit Orakeln (Abschnitt 4) und im Lauf A1W4 (Menisken aus SYN-32, SYN-13, SYN-04 über Operator A aus deren Platten) |
| 3. Messung | Läufe A1W4, A1W2, A1W4_farbe, A1W4_farbe_ohne, F3W2, F3W4 (Abschnitt 3); Kosten über die Zahl der Platten (Abschnitt 2) |
| 4. Tests, Doku | Abschnitt 4; optimierung.md, cli.md, entwurfssteuerung.md, CHANGELOG, Plan |

**Entwurfsentscheidung „vorhandene Gläser zählen mit“.** `synth` wendet die Option bei jedem Aufruf an, und das Skript ruft `synth` viele Male auf den eigenen Ergebnissen auf. Eine Auswahl relativ zum vorhandenen Wörterbuch („N Gläser, die noch fehlen“) würde bei jedem Aufruf weitere Gläser hinzufügen. Deshalb ist die Rangfolge unabhängig vom System (bester Achromat mit irgendeinem Katalogglas), und ein gewähltes Glas, das schon im Wörterbuch steht (Katalogname oder Modell mit gleichen Kennzahlen wie der Buchdahl-Fit von SYN-36 im Startsystem), bekommt keine neue Platte. Folge: CaF₂ belegt im Startsystem der Zielspezifikation den ersten anomalen Platz, N = 2 setzt nur SYN-32 dazu.

**Befund zur Ausgangslage.** Nach der üblichen Kron-Flint-Grenze ist SYN-36 (ν_d 48,6, n_d 1,80) schon ein (Lanthan-)Flint und SYN-30 (ν_d 22,3) ein Schwerflint mit stark anomaler Teildispersion (ΔP_g,F +0,038). Das alte Wörterbuch enthielt also Flintgläser; es fehlten leichte Flinte als Partner von CaF₂ und ein FK51-artiges Glas. Die Rangfolge nach dem spektralen Rest bevorzugt leichte Flinte (bester Partner jeweils CaF₂); dichte Flinte mit normaler Teildispersion (z. B. SYN-21, ν_d 34,5) lassen sich ausdrücklich nennen.

## 2. Kosten der Erweiterung (Standardgröße)

`optimize --gpu`, 100 Zyklen ohne Stagnationsabbruch, Startpunkt Rückkehr zu Geisterplatten des Laufs F3 (12 El., 7 Wellenlängen, zwei Lagen, K = 10), kleinste von zwei Zeiten bei geteilter GPU:

| N | Platten | Variablen der Platten | Zeit [s] | relativ |
|---|---|---|---|---|
| 0 | 6 | 60 | 1,91 | 1,00 |
| 2 | 8 | 80 | 2,02 | 1,06 |
| 4 | 12 | 120 | 2,64 | 1,38 |
| 6 | 16 | 160 | 3,42 | 1,80 |

Begründung der Standardgröße N = 4: Der Nutzen zeigt sich nur mit den beiden zusätzlichen Gläsern SYN-13 und SYN-04 (A1W4 gegen A1W2, Abschnitt 3); N = 2 lieferte einen Pfad wie ohne Erweiterung. N = 6 kostet fast das Doppelte, ohne gemessenen Nutzen. Der monochromatische Lauf A1W4 war insgesamt schneller als F1 aus M6-T8 (1.956 s gegen 4.426 s), weil er mit weniger Elementen endet.

## 3. Messung

Optionen der Farbstufen wie M6-T11: `OPT` = `--farb-stetig --farb-ziel voll --farb-modul doublet --farb-spektrum-rang --farb-startradien --farb-chromatisch 0.5 --feldregel --einfuegen-verbessern 4 --rueckbau-punkt 0.85`. Monochromatische Stufen: Z2 = ohne Geisterplatten nachoptimiert, RMS aus `synth`-Quadratur 16 × 8 bei 546 nm an den Feldern 0/70/100 % des erreichten Feldanteils. Farbstufen: nach dem Rückbau bewertet (`mikroptim eval --grid 256`), größter RMS über 436–656 nm je Feld, Strehl = kleinster FFT-Strehl über 436–656 nm. Farbkennzahlen wie M6-T11, Abschnitt 3 (Aberrationsbericht bei 436/656 nm, größter Feldwinkel der Datei, ohne Platten).

### 3.1 Monochromatisch (a) und Farbstufe danach

| Lauf | Start | Wörterbuch | NA | Feld | Spektrum | El. | RMS 0/70/100 % | Strehl min | Zeit [s] |
|---|---|---|---|---|---|---|---|---|---|
| F1 (M6-T8) | Einzellinse, Feldregel | 3 Gläser | 0,50 | 1,00 | 546 | 25 | 0,039/0,022/0,059 (Z2, eval) | – | 4.426 |
| 9-Linser (M6-T9) | F1 nach Rückbau | – | 0,50 | 1,00 | 546 | 9 | 0,012/0,037/0,038 | – | – |
| **A1W4** | Einzellinse, Feldregel | + SYN-32, SYN-13, SYN-04 | 0,445 | 0,24 | 546 | 8 | 0,048/0,082/0,083 (Z2) | – | 1.956 |
| A1W2 | Einzellinse, Feldregel | + SYN-32 | 0,50 | 0,69 | 546 | 26 | 0,108/0,442/0,497 (Z2) | – | 8.464 |
| **A1W4_farbe_ohne** | A1W4 Z2, Farbstufe ohne Option | 3 Gläser (Rückkehr) | 0,445 | 0,24 | 436–656 | 8 → 13 → **11** | **0,025/0,068/0,060** | **0,87** | 8.275 |
| A1W4_farbe | A1W4 Z2, Farbstufe mit N = 4 | 3 + 3 Gläser | 0,445 | 0,24 | 436–656 | 8 → 14 → 14 | 0,039/0,091/0,082 | 0,81 | 12.128 |
| M9 (M6-T11) | 9-Linser, Farbstufe | 3 Gläser | 0,50 | 1,00 | 436–656 | 9 → 15 | 0,615/1,266/1,691 | 0,00 | 6.427 |
| M5 (M6-T11) | 5-Linser NA 0,869, 23 % | 3 Gläser | 0,869 | 0,23 | 436–656 | 5 → 12 | 0,632/0,733/0,841 | 0,00 | 6.790 |

A1W4 endet in der Aperturstufe bei NA 0,445 (F1: 0,5) und in der Feldstufe bei 24 % (F1: 100 %); er ist monochromatisch nicht beugungsbegrenzt (Rand 0,083 Wellen). A1W2 hängt an der Elementhöchstzahl 26 und ist schlechter als F1.

Farbkennzahlen der monochromatischen Endstände (436/656 nm, Wellen bei 546 nm):

| System | El. | NA | Gläser | ΔW020 | ΔW111 | Fokusspanne g…C | σ_col |
|---|---|---|---|---|---|---|---|
| 9-Linser M6-T9 | 9 | 0,50 | CaF₂, SYN-30, SYN-36 | 70,6 | 76,3 | 298 µm | 21,5 |
| F1 M6-T8 (Z2) | 25 | 0,50 | ebenso | 106,4 | 79,0 | 454 µm | 24,8 |
| A1W2 (Z2, volles Feld) | 26 | 0,50 | + SYN-32 | 62,0 | 83,9 | 303 µm | 23,2 |
| **A1W4** (Z2, volles Feld) | 8 | 0,445 | + SYN-32, SYN-13, SYN-04 | 13,9 | 32,2 | **75 µm** | **8,3** |
| A1W4 (Z2, erreichtes Feld 24 %) | 8 | 0,445 | ebenso | 13,9 | 7,7 | 75 µm | 2,8 |
| A1W4_farbe_ohne (nach der Farbstufe) | 11 | 0,445 | CaF₂, SYN-11, SYN-30, SYN-36, Glasvariable | 0,19 | 0,29 | 1,8 µm | 0,09 |

Vergleich gleicher Stufen: Am Ende der Aperturstufe (10 % Feld, Bericht beim vollen Feldwinkel) haben F1 (5 El., NA 0,5) und A1W4 (7 El., NA 0,445) ähnliche Längsfarbe (76 bzw. 87 µm); der Unterschied entsteht in der Feldstufe: F1 baut dort 20 Menisken aus SYN-36/SYN-30 (Fokusspanne steigt auf 454 µm), A1W4 bleibt bei 8 Elementen, vier davon aus den Zusatzgläsern. Ob das am Wörterbuch oder am früheren Stillstand der Feldstufe liegt, trennt die Messung nicht; der Endstand ist jedenfalls lokal farbkorrigierbar, anders als alle monochromatischen Stände aus M6-T9/M6-T10 (F13).

### 3.2 Farbe früh (b): F3 mit erweitertem Wörterbuch

Start V3 aus M6-T6b (12 El., NA 0,535, 27 % Feld, 405–707 nm), `--zyklus feld,apertur --zyklus-runden 4 --elementbudget 18 OPT`. RMS über 436–656 nm am letzten Stand; in Klammern das Maximum über alle Wellenlängen einschließlich 405/707 nm (schwach gewichtet) aus der Zyklustabelle:

| Lauf | Wörterbuch | Stufen | NA | Feld | El. | RMS 0/70/100 % | Strehl min | Zeit [s] |
|---|---|---|---|---|---|---|---|---|
| F3 (M6-T11) | 3 Gläser | feld, apertur | 0,535 → **0,6075** | 0,27 → 0,38 | 16 | 0,028/0,111/0,130 (0,149/0,275/0,237) | 0,64 | 9.550 |
| F3W2 | + SYN-32 | feld, apertur, feld, apertur | 0,535 → 0,575 | 0,27 → 0,41 | 18 | **0,022/0,113/0,128** (0,128/0,257/0,203) | **0,67** | 15.514 |
| F3W4 | + SYN-32, SYN-13, SYN-04 | feld, apertur, feld, apertur | 0,535 → 0,5525 | 0,27 → **0,43** | 18 → **13** (Rückbau) | 0,068/0,102/**0,110** (0,089/0,225/0,217) | **0,73** | 17.232 |

Zwischenstände F3W2 (Zyklustabelle `--zyklus-tabelle docs/ergebnisse/m6-t12/F3W2`): c0_feld 39,7 % (17 El.), c1_apertur NA 0,56, c2_feld 41,2 % (19 El.), c3_apertur NA 0,575 (18 El.). F3W4: die Feldstufe c0 erreichte schon im ersten synth-Aufruf 42,8 % (18 El.), danach lag das Elementbudget fest; c1/c3_apertur öffneten nur auf 0,5475 bzw. 0,5525. Im letzten Stand (c3_apertur) entfernte der Rückbau bei 85 % der Stufe fünf Elemente (18 → 13), das Ergebnis ist am Rand besser als F3 aus M6-T11 (0,110 gegen 0,130 Wellen, Strehl 0,73 gegen 0,64) – bei kleinerer NA und größerem Feld. In beiden Läufen kommen die Einfügungen aus der Farbregel (Operator B mit Partner CaF₂ aus der Vorauswahl, Null-Kittglied) und der Feldregel; die Zusatzplatten wurden nicht zur Empfehlung.

## 4. Tests

| Test | Orakel / Ergebnis |
|---|---|
| `GhostDictionary.SelectionFollowsPreselectionRank` | Rangfolge unabhängig nachgerechnet (jedes Glaspaar einzeln in der Vorauswahl, Klassen aus `glass_properties`): Auswahl und Rest gleich (≤ 10⁻¹² relativ); CaF₂ im Startsystem als vorhanden markiert; genannte Gläser zuerst; Fehler bei unbekanntem/doppeltem Namen und N < 0 |
| `GhostDictionary.ExtensionKeepsMeritAndIsIdempotent` | Merit mit aktiven Platten vor/nach der Erweiterung gleich (2,838949660294069·10⁴ beide, Toleranz `kInsertionMeritRelative`); Platten, Variablen (K je Platte), Strafe; zweiter Aufruf ändert nichts; ohne Platten bzw. N = 0 unverändert |
| `GhostDictionary.FlintPlateCorrectsColorOfSinglet` | Dünner Achromat: Flint-Platte (SYN-21) mit δφ_G = −δφ_L vor N-BK7, f = 50 mm: Farbe der Brechkraft 3,06·10⁻⁴ → 1,38·10⁻⁵ /mm (Faktor 22), Platte zerstreuend |
| `GhostDictionary.FlintPlateLeadsToFlintElementOrDoublet` | Flint-Platte allein aktiv → Operator A, Meniskus aus dem Flint; CaF₂ + Flint (Brechkraft 0) → Operator B an der Linse, Zielglas Flint, c_s = −0,0641/mm (dünner Achromat −0,0638/mm); Kittglied mit den Krümmungen des dünnen Achromats: Längsfarbe 0,778 → 0,0052 mm |
| `CliSynth.GhostDictionaryExtra` | Ohne Option kein Eintrag; mit N = 0 Protokoll (ohne Laufzeiten) und Ergebnisdatei **bitgleich** zum Lauf ohne Option; mit N = 2 (Katalog basis.json, kein Flint) CaF₂ als einziges Glas, 2 Platten; Text; Fehlbedienung Exit 2, unbekanntes Glas Exit 1 |
| Python `test_tool_m6t6.py::test_synth_options_ghost_dictionary`, `test_design.py::test_synth_segment_ghost_dictionary_extra` | Optionen durchgereicht, ohne Option unverändert, Schema 1–4 |

Lokal: `ctest --preset release-cuda` (RTX 3090) 1136/1136 bestanden; GCC-Prüfbau (MSYS2, Warnungen als Fehler) 943/943.

## 5. Befunde

| Nr. | Stand nach M6-T12 |
|---|---|
| F13 (Farbe nach dem monochromatischen Aufbau) | **teilweise gelöst:** Mit erweitertem Wörterbuch (N = 4) ist der monochromatische Endstand farbfähig, die Farbstufe wird beugungsbegrenzt (0,068 Wellen, 11 El.) – allerdings bei kleinerer NA (0,445) und kleinem Feld (24 %), weil der monochromatische Pfad früher stehen bleibt |
| F18 (neu) | Das erweiterte Wörterbuch ändert den monochromatischen Pfad stark (Apertur 0,445 statt 0,5, Feld 24 % statt 100 %); die Zusatzplatten haben andere Brechzahlen, Operator A wählt dann andere Gläser und andere Lagen. Ob ein Pfad mit Erweiterung erst ab einer bestimmten Stufe (z. B. nach der Aperturstufe) besser ist, ist nicht gemessen |
| F19 (neu) | In polychromatischen Stufen empfehlen die Zusatzplatten nie ein Einfügen; die Farbregel nimmt ihre Partner aus der Vorauswahl über den ganzen Katalog. Die Erweiterung wirkt dort höchstens indirekt (F3W4: mehr Feld, Rückbau auf 13 El.; A1W4_farbe: schlechter als ohne) |
| F17 (Rückbau braucht Abstand) | erstmals in Farbstufen wirksam: A1W4_farbe_ohne 13 → 11 El., F3W4 (c3_apertur) 18 → 13 El., jeweils bei 85 % der Stufe |

## 6. Fragen an den Maintainer

1. **Standard der Option:** Die Option bleibt standardmäßig aus (Bitgleichheit). Soll `--woerterbuch-extra 4` für die monochromatischen Stufen des Skripts Standard werden? Das ändert alle Referenzläufe.
2. **Auswahlregel:** Vorhandene Gläser zählen mit (CaF₂ belegt einen Platz). Alternative wäre eine Markierung der Zusatzplatten im Dateiformat (z. B. ein Feld `comment` an Geisterplatten, kompatible Erweiterung von `mikroptim.system/1`), dann könnte N „neue Gläser“ bedeuten. Gewünscht?
3. **Flint-Klasse:** Die Rangfolge wählt leichte Flinte (ν_d ≈ 48) als Partner von CaF₂. Soll zusätzlich ein dichtes Flint mit normaler Teildispersion (Regel `dense_flint_for_lca`) fest dazugehören?

## 7. Dateien

```
python tools/m6t6_zielspezifikation.py --gpu --bis p2 --feldregel --woerterbuch-extra 4 --out docs/ergebnisse/m6-t12/A1W4
python tools/m6t6_zielspezifikation.py --gpu --bis p2 --feldregel --woerterbuch-extra 2 --out docs/ergebnisse/m6-t12/A1W2
python tools/m6t6_zielspezifikation.py --gpu --zyklus farbe --start docs/ergebnisse/m6-t12/A1W4/z2_s0.json --elementbudget 14 --farb-budget 6 OPT --out docs/ergebnisse/m6-t12/A1W4_farbe_ohne
python tools/m6t6_zielspezifikation.py --gpu --zyklus farbe --start docs/ergebnisse/m6-t12/A1W4/z2_s0.json --elementbudget 14 --farb-budget 6 OPT --woerterbuch-extra 4 --out docs/ergebnisse/m6-t12/A1W4_farbe
python tools/m6t6_zielspezifikation.py --gpu --zyklus feld,apertur --zyklus-runden 4 --start docs/ergebnisse/m6-t6b/V3_farbe_frueh/p7.json --elementbudget 18 OPT --woerterbuch-extra 2 --out docs/ergebnisse/m6-t12/F3W2
python tools/m6t6_zielspezifikation.py --gpu --zyklus feld,apertur --zyklus-runden 4 --start docs/ergebnisse/m6-t6b/V3_farbe_frueh/p7.json --elementbudget 18 OPT --woerterbuch-extra 4 --out docs/ergebnisse/m6-t12/F3W4
```

mit `OPT` wie in Abschnitt 3. Die Verzeichnisse sind mit `--archiv` verkleinert.
