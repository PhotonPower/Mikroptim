# M11-T4: Sinusbedingung als Residuum – Versuch „Apertur zuerst auf der Achse“

Bericht zur Abnahme von M11-T4 ([Meilensteinplan](../plan/meilensteine.md#m11-t4--sinusbedingung-als-residuum), Formeln: [optimierung.md, Abschnitt 2.7](../optimierung.md#27-sinusbedingung-ab-m11-t4)). Läufe auf dem Rechner des Maintainers (RTX 3090, geteilt mit anderen Agenten), Release-Build mit CUDA, `mikroptim synth` wie im dokumentierten M6-T6-Lauf (`--stall 0`).

**Kurzfassung.**

- Am **Endpunkt der Synthese** (System mit Geisterplatten, gleiche NA) senkt das Residuum den RMS bei 7 % und 10 % Feld von **14,6 bzw. 20,9 Wellen auf 0,80 Wellen**; der Feldfehler wächst mit Residuum nicht mehr über den Achswert (0,79 Wellen). OSC am Rand sinkt von −0,13 auf −3·10⁻⁴.
- Der Versuch kommt **nicht bis NA 0,95**: Sobald die Falle F12 (Fokus im Scheitel der Hilfsebene vor dem Objekt) mit einer schwachen Bedingung `paraxial_focus` ausgeschlossen ist, bleibt der Controller mit und ohne Residuum bei NA 0,37 stehen (kein Bedarf, kein Spalt für ein erzwungenes Einfügen). Der Vergleich mit dem damaligen Wert 0,28 Wellen bei NA 0,95 ist deshalb nicht möglich.
- Nach dem Entfernen der Geisterplatten und Nachoptimierung auf der Achse sind beide Systeme bei NA 0,37 fast beugungsbegrenzt; der Feldzuwachs (quadratisch vom Achswert abgezogen) ist mit Residuum kleiner (0,028/0,040 statt 0,037/0,050 Wellen), der Achs-RMS aber größer (0,085 statt 0,053).
- Ein Lauf ohne Schutz gegen F12 erreicht mit Residuum NA 0,92 (ohne Residuum nur 0,38), aber nur über die Falle (Endpunkt 76 Wellen auf der Achse); nachoptimiert 0,15/0,44/0,92 Wellen bei 0/7/10 % Feld.

## Ablauf

`python tools/m6t6_zielspezifikation.py --osc-versuch [--osc-gewicht W] [--osc-strahlsuche] --gpu` (Ausgabe in `docs/ergebnisse/m11-t4/mit` bzw. `ohne`, im Repository verkleinert auf Start, Synthese-Ergebnis, Nachoptimierung und `m11t4_osc.json`):

1. Start `data/systems/zielspezifikation_start.json` (Einzellinse, NA 0,3, 546 nm, Geisterplatten blendennah und bildnah, Nebenbedingungen von M6-T6 einschließlich Petzval), **nur das Achsfeld**.
2. `paraxial_focus` mit Gewicht 100/mm² gegen die Falle F12: Die Hilfsebene vor dem Objekt ist die letzte Fläche, die Referenzsphäre der Merit hat dort den Radius des Arbeitsabstands, und der LM kann den Fokus in ihren Scheitel legen (M6-T6, Befund F12). Die Bedingung kostet bei 0,1 mm Randstrahlhöhe eine Welle², die Falle (2,7 mm Defokus) rund 700 Wellen²; eine Fokusverschiebung von einigen µm für den Ausgleich der Restsphärik bleibt praktisch frei.
3. Mit Residuum: Eintrag `sine_condition` mit w = (NA η′/λ)²/(72 · 2) = 13,1 (η′ = Bildhöhe bei 10 % Feld; das Residuum kostet so viel wie die Koma dort, optimierung.md, Abschnitt 2.7), Zonen 1/√2 und 1.
4. `synth_loop` des M6-T6-Skripts: NA 0,3 → 0,95, höchstens 12 Runden mit Nachoptimierung der Spalte, μ-Halbierung und erzwungenem Einfügen (wie der Hauptlauf).
5. Bewertung bei 0, 7 und 10 % Feld (0; 0,3045°; 0,4345°; dieselben Winkel wie die Stufe 1 des Hauptlaufs) mit `mikroptim eval --grid 256` (Referenz durch die Austrittspupille), OSC aus `mikroptim optimize --max-iterations 0 --sine-condition 0` (Batch mit Geisterplatten):
   - am Endpunkt der Synthese (mit Geisterplatten),
   - nach Entfernen der Geisterplatten und Hilfsebenen und 2000 LM-Zyklen auf der Achse (Residuum wie in der Synthese).

## Ergebnisse

RMS in Wellen bei 546 nm (Referenz Austrittspupille), OSC in Zone 1/√2 und am Rand.

| Lauf | erreichte NA | Glaselemente | Bewertung | RMS 0 % | RMS 7 % | RMS 10 % | OSC Zone | OSC Rand |
|---|---|---|---|---|---|---|---|---|
| ohne Residuum | 0,375 | 4 | Endpunkt (Geisterplatten) | 1,079 | **14,58** | **20,88** | −0,053 | −0,132 |
| mit Residuum (w = 13,1) | 0,365 | 6 | Endpunkt (Geisterplatten) | 0,794 | **0,796** | **0,798** | 6,8·10⁻⁴ | −3,0·10⁻⁴ |
| ohne Residuum | 0,375 | 4 | nachoptimiert | 0,053 | 0,064 | 0,073 | −0,0074 | −0,0143 |
| mit Residuum | 0,365 | 6 | nachoptimiert | 0,085 | 0,090 | 0,094 | −4,9·10⁻⁴ | 2,7·10⁻⁴ |

Laufzeit 72 s bzw. 120 s, je drei synth-Runden; beide enden mit „kein Fortschritt: erzwungenes Einfügen“ ohne passenden Luftspalt (≥ 0,7 mm).

**Varianten** (gleicher Ablauf, Zahlen aus den Protokollen der Erkundung, Ausgaben nicht im Repository):

| Variante | Lauf | NA | Endpunkt 0/7/10 % | nachoptimiert 0/7/10 % |
|---|---|---|---|---|
| `paraxial_focus` mit Gewicht 10⁴ | ohne | 0,375 | 1,14 / 21,3 / 24,4 | 0,089 / 0,091 / 0,094 |
| | mit | 0,3725 | 0,83 / 0,83 / 0,83 | 0,091 / 0,092 / 0,092 |
| dasselbe mit Strahlsuche (`--osc-strahlsuche`, B = 2, 2 Varianten) | ohne | 0,375 | 1,11 / 21,1 / 25,4 | 0,007 / 0,072 / 0,106 |
| | mit | 0,3725 | 0,83 / 0,83 / 0,83 | 0,091 / 0,092 / 0,092 |
| ohne `paraxial_focus` (Falle F12 möglich) | ohne | 0,38 (4 Elemente) | 1,04 / 18,0 / 34,1 | 0,106 / 0,137 / 0,171 |
| | mit | **0,9225** (9 Elemente) | 76,3 / 78,6 / 81,0 (F12) | 0,155 / 0,443 / 0,916 |

## Bewertung

- **Wirkung des Residuums auf die Synthese.** Am Endpunkt unterscheiden sich die Läufe um mehr als eine Größenordnung im Feld, in allen Varianten: Ohne Residuum wächst der RMS von ≈ 1 auf 15–34 Wellen bei 7–10 % Feld, mit Residuum bleibt er beim Achswert. Der große Feldfehler ohne Residuum stammt zu einem guten Teil aus den Geisterplatten selbst: Ihr rotationssymmetrisches Profil hoher Ordnung korrigiert das Achsbündel, das im Feld verschobene Bündel sieht es versetzt. Das Residuum erfasst das, weil sein Zonenstrahl durch die Platten läuft (OSC am Rand −0,13 ohne Residuum).
- **Nach der Nachoptimierung** ohne Geisterplatten ist bei NA 0,37 kaum noch Koma übrig (OSC −0,014 → nach Seidel 0,03 Wellen RMS bei 10 % Feld). Der Vorteil des Residuums ist dort klein und wird vom schlechteren Achswert aufgehoben; bei dieser NA ist das Ergebnis also nicht „deutlich kleiner“.
- **Abnahme „deutlich kleinerer RMS bei 7 % und 10 % Feld“:** erfüllt für die Systeme, mit denen der Controller arbeitet (Endpunkt der Synthese, Faktor 18 bzw. 26), nicht nachgewiesen für das reale System bei hoher NA, weil der Versuch die hohe NA ohne die Falle F12 nicht erreicht. Der Befund F4 (NA 0,95 mit 5 Elementen in einer Achsoptimierung) ließ sich mit dem heutigen Skript nicht reproduzieren; dass die Synthese ohne `paraxial_focus` hohe NA nur über F12 erreicht, legt nahe, dass auch der damalige Lauf in der Falle stand.
- **Folgerung.** Das Residuum ist billig (zwei Strahlen pro Variante auf dem Host) und hält die Synthese komafrei, auch dort, wo die Geisterplatten den Feldfehler erzeugen. Für den Weg zur hohen NA ist die Falle F12 der Engpass (Referenzsphäre unabhängig von der letzten Fläche, F11), nicht die Koma.

## Dateien

| Datei | Inhalt |
|---|---|
| [`tools/m6t6_zielspezifikation.py`](../../tools/m6t6_zielspezifikation.py) | `--osc-versuch`, `--osc-gewicht W` (Standard `osc_weight_for_field(0.1)` = 13,1; 0 = ohne), `--osc-strahlsuche` |
| `docs/ergebnisse/m11-t4/{ohne,mit}/m11t4_osc.json` | Zusammenfassung: Runden, Punkt, RMS und OSC am Endpunkt und nachoptimiert |
| `docs/ergebnisse/m11-t4/{ohne,mit}/o0_start.json`, `o1_final.json`, `o2.json` | Start, Ergebnis der Synthese, nachoptimiertes System |
