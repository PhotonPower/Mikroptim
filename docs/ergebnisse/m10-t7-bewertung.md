# M10-T7: Bewertung der Entwurfssteuerung (mit M10-T7b)

Bericht zur Aufgabe M10-T7 ([Meilensteinplan](../plan/meilensteine.md#m10-t7--bewertung), [entwurfssteuerung.md, Abschnitt 8](../entwurfssteuerung.md#8-bewertung)). Verglichen werden die Synthese ohne Entscheider (V0, V1), das M6-T6-Skript (V2), die regelbasierte Strategie ohne Modell (V3) und zwei lokale Sprachmodelle über Ollama auf denselben Werkzeugen (`mikroptim.design`). Läufe mit der Claude-API (M10-T6c) und mit einem Menschen (V4) stehen aus (Abschnitt 6).

**Fortschreibung M10-T7b (Abschnitt 9):** Bewertung mit den Operatoren aus M10-T4 wiederholt (Stand `main` 9cc89a7 plus PR), Regel V3 mit Stufen bei Stillstand, Befunde E6–E8 behoben. Ergebnis: **B3 erreicht die Regel mit der Referenzsphäre nach ADR-0017 (V3RS): NA 0,85, RMS 0,060, 4 Elemente** – der erste Lauf, der B3 löst; mit alter Sphäre kommt V3 bis NA 0,833 (alt 0,448). **B2: V3RS bis NA 0,79** (alte Sphäre 0,41; Ziel 0,95). Entscheidende Schritte: Geisterplatten-Wörterbuch und Null-Meniskus mit Dicken-Homotopie an der Stelle größter Randstrahlhöhe. Kein lokales Modell ist besser als die Regel; gpt-oss:20b erreichte B1 nur auf dem Zwischenstand 8d3aa3c. Claude-Läufe weiter ausstehend (kein `ANTHROPIC_API_KEY`). Die Abschnitte 1–8 beschreiben den Stand von M10-T7 (717a8ff).

**Ab M10-T7c (Entscheidung Maintainer 2026-11-20):** Alle Benchmarks rechnen mit `reference_sphere: image_infinite` (ADR-0017); **V3RS ist die Referenz der Regel.** Neue Läufe schreibt das Skript nach `docs/ergebnisse/m10-t7c`; die Varianten `V0`, `V1`, `V3`, `V3AL`, `LM` liefern dort die Kürzel `V0RS`, `V1RS`, `V3RS`, `V3ALRS`, `LMRS_<modell>`, die alte Konvention bleibt als `<V>-alt` abrufbar (Kürzel wie in diesem Bericht). Die Zahlen und Dateien dieses Berichts (Abschnitte 1–9) bleiben unverändert; Einzelheiten in [entwurfssteuerung.md, Abschnitt 8](../entwurfssteuerung.md#8-bewertung) und Abschnitt 10 unten.

**Claude-Läufe (M10-T7c, Abschnitt 11):** Haiku 5.5, Sonnet 5.5 und Opus 5.5 auf B1–B3. Opus erreichte B1 und **B3 mit RMS 0,0145 (Regel V3RS 0,060)**; Haiku erreichte (Zwischenstand Z1) B1 und B3 und öffnete **B2 bis NA 0,835 (Regel 0,79)**. Die Endläufe streuen stark (Opus B1 Ziel, B2 0,46; Haiku und Sonnet ohne Ziel). Entscheidend ist der erste Schritt: Wer vor dem ersten Homotopie-Abschnitt Spalte freigibt und optimiert, landet in einer Falle der Werkzeuge (Befund E14, offen, Frage an den Maintainer) und bleibt bei NA 0,30. Weitere Werkzeugfehler (E15–E18: `abandon` beendete die Sitzung, `optimize`-Variablenauswahl fehlte, `start_rejected` sah im Verlauf wie ein Erfolg aus, strict-Schemas von der API abgelehnt) sind behoben. Kosten aller Läufe 12,63 $ (Budget 20 $).

**M10-T7d (Abschnitt 12):** E14 behoben (`free_gaps` gibt Spalte mit Geisterplatten nicht mehr frei), je drei Claude-Läufe pro Modell und Benchmark. **B1 erreichen alle drei Modelle in 9 von 9 Läufen** (RMS-Median Opus 0,0097, Haiku 0,0112, Sonnet 0,0130; Regel 0,0154). B2/B3 erreicht keiner; auf B2 ist Opus vorn (Median 0,678, bester 0,783). Die Regel V3RS ändert sich mit dem Fix über einen anderen Pfad durch ihre Stufen: B2 0,790 → 0,453, B3 Ziel → 0,600 (neue Referenz, alte Werte daneben). Kosten 7,17 $, kumuliert 19,80 $.

**M10-T7e (Abschnitt 13):** Operator A senkt im kurzen Plattenspalt die Schranke der Teilstücke (Plattenspalte wieder frei), E19 mit zweitem Versuch nach `optimize`, gestörte Starts für die Regel, `insert_module` mit Standardglas und angehobener Dicke. Fehler der Modelle 108 → 11 von 272 bzw. 400 Aktionen. **Erstmals B2 am Ziel (Haiku, NA 0,950, RMS 0,037)**, B3 Opus 2/3 und Haiku 1/3. Die Regel V3RS erreicht ungestört wieder B2 0,790 und B3, mit fünf gestörten Starts aber nur B1 (5/5), B2 Median 0,458, B3 Median 0,410. Kosten 4,02 $ (kumuliert 23,82 $).

**Stand M10-T7: abgeschlossen bis auf V4 (Mensch); B4 ruht** (Entscheidung Maintainer 2026-11-22: die Patentsysteme sind Platzhalter, als Referenz unbrauchbar). **Referenz der Regel seit M6-T11b: Median von V3RS über fünf gestörte Starts** ([Abschnitt 14](#14-m6-t11b-referenz-der-regel-als-median-gestörter-starts)). Die folgenden Absätze beschreiben den ursprünglichen Stand von M10-T7.

**Stand M10-T7: Teilergebnis.** V0, V1 und V3 auf B1–B3 sowie zwei Ollama-Modelle (Instruktion und Thinking) auf B1–B3 sind gerechnet; V2 ist aus dem Bericht M6-T6 übernommen. **Ausstehend:** Claude-Läufe (kein `ANTHROPIC_API_KEY` in der Umgebung, `ant` nicht installiert), V4 (Mensch), B4 (private Patentsysteme).

**Kurzfassung.**

- **B1** (NA 0,3 → 0,6, Achse, 546 nm) erreichen V0, V1, V3 und **gpt-oss:20b** (Thinking). gpt-oss erreicht den kleinsten RMS (0,0063 Wellen gegen 0,0163 bei V0/V3, 4 Elemente): Es gab dem Homotopie-Abschnitt nur 2 neue Elemente und gab nach dem ersten Abschnitt Spalte frei und optimierte, bevor es den zweiten startete. qwen3:14b (Instruktion) kommt nur bis NA 0,44.
- **B2** (50x/0,95, Aperturteil bei 10 % Feld) und **B3** (40x/0,85, Aperturteil bei 10 % Feld) erreicht keine Variante. Alle bleiben bei NA 0,41–0,46 mit 3–5 Elementen stehen (`step_too_small` des Controllers, Befund F3 aus M6-T6: keine Geisterplatte meldet Bedarf). Die Sprachmodelle kommen nicht weiter als V0/V3; gpt-oss erreicht auf B3 NA 0,455 (V0/V3: 0,448), aber mit größerem RMS und 5 statt 3 Elementen.
- **V3 gegen Modell:** Auf diesen Benchmarks zeigt nur gpt-oss auf B1 einen Vorteil (besseres System, gleiche Elementzahl), auf B2/B3 keinen. qwen3:14b ist auf allen drei Benchmarks schlechter als die Regel. Nach ADR-0012, Punkt 10, rechtfertigen diese Ergebnisse (noch) keinen Ausbau der Modellanbindung; die Bewertung mit einem großen Modell (Claude) steht aus und entscheidet das mit.
- **Fehler der Werkzeuge,** die die Modelle gefunden haben (Abschnitt 5), sind behoben: Parameter falschen Typs brachen die Sitzung ab, `split_lens` und `free_gaps` erzeugten Systeme, deren Startwerte unter den Variablenschranken lagen.

## 1. Benchmarks

Startsysteme frei erfunden (AGENTS.md §8). Ziel: NA ≥ Soll-NA (relativ 10⁻³), RMS ≤ 1/14 Welle pro Feld und Wellenlänge, keine versagten Strahlen.

| Nr. | Aufgabe | Start | Soll-NA | Elementbudget V0/V1 |
|---|---|---|---|---|
| B1 | NA 0,3 → 0,6 auf der Achse, 546 nm (Abnahme M6-T3) | `data/systems/einzellinse_homotopie_start.json` | 0,6 | 6 |
| B2 | 50x/0,95 (Zielspezifikation), **Aperturteil**: NA 0,3 → 0,95 bei 10 % Feld (0; 0,30; 0,43°), 546 nm | `zielspezifikation_start.json`, Feldwinkel × 0,1 | 0,95 | 15 |
| B3 | 40x/0,85 trocken (in Anlehnung an Zhang §7.2), **Aperturteil** wie B2 | `bewertung_b3_40x085_start.json` (neu: B2-Start mit allen Längen × 1,25, f = 4,1125 mm), Feldwinkel × 0,1 | 0,85 | 15 |
| B4 | Rekonstruktion von Patentsystemen | privates Verzeichnis | – | **nicht umgesetzt** |

**Warum nur der Aperturteil.** Die Entwurfswerkzeuge können Feld und Spektrum eines Kandidaten nicht stufenweise ändern: `synth_segment` hat noch keine Spektrumstufen (`not_implemented`), und ein Feldanteil am Start lässt sich über die Aktion nicht setzen. Bei vollem Feld verwirft schon V0 den Start bei NA 0,3 (`start_rejected` nach 9 min, F_B = 0,0144 > F_max = 0,0051, Vorversuch). Der Aperturteil bei 10 % Feld ist die erste Stufe der Zielspezifikation („Apertur zuerst“ in M6-T6) und für alle Varianten gleich erreichbar. Der Zustand nennt die Spezifikation trotzdem vollständig (Farbklasse c, Feldklasse 4–5, FZ 2,5 für das verkleinerte Feld).

**B4** braucht Patentsysteme aus einem privaten Verzeichnis; Ort und Format sind offen (entwurfssteuerung.md, Frage 3). Mit dem Claude-Entscheider darf B4 nicht laufen (ADR-0016; der Entscheider verweigert Daten unter `private/`).

## 2. Varianten und Einstellungen

| Variante | Umsetzung | Einstellungen |
|---|---|---|
| V0 | `mikroptim synth <start> --target-na NA --max-elements N` (`run_homotopy`) | Standardwerte (10 × 8 Strahlen, 500 LM-Zyklen pro Schritt, ohne Stagnationsabbruch) |
| V1 | wie V0 mit `--beam 3 --alternatives 2` (Strahlsuche) | wie V0 |
| V2 | `tools/m6t6_zielspezifikation.py` | nur B2; **aus dem Bericht M6-T6 übernommen**, nicht neu gerechnet (Lauf mehrere Stunden auf der CPU) |
| V3 | `mikroptim.design` mit `RuleDecider` | 12 Entscheidungen; Regel: `synth_segment` zur Soll-NA (≤ 4 neue Elemente), dann `free_gaps` + `optimize`, höchstens 4 Abschnitte, dann Farb-/Sphärik-Regeln |
| LM | `mikroptim.design` mit `OllamaDecider`, Rückfall `RuleDecider` | 12 Entscheidungen, Temperatur 0, Seed 0, `num_ctx` 32768, `keep_alive` 10 min, nach jeder Messreihe entladen; qwen3:14b ohne Thinking (Instruktion), gpt-oss:20b mit Thinking |
| V4 | `mikroptim.design` mit `HumanDecider` | **ausstehend** |

Alle Läufe rechnen auf der CPU im Prozess (Python-Wheel, `run_cli`), damit Trace und Sprachmodell nicht um die geteilte RTX 3090 konkurrieren; Ollama 0.40.16 auf der RTX 3090. Modelle: `qwen3:14b` (Digest `bdbd181c33f2…`), `gpt-oss:20b` (Digest `f38aa0c53da5…`). Rechner des Maintainers (24 Kerne), mehrere Läufe parallel; Zeiten sind Orientierung. Code-Stand: `main` 717a8ff (nach M10-T6c, vor M10-T3, M11-T4 und M9-T2) plus die Änderungen dieses PRs.

**Bewertung jedes Endsystems** (Skript, Funktion `assess`): Geisterplatten-Koeffizienten auf 0 (reales System wie die Zielprüfung von `synth`), `mikroptim trace` (Referenzsphäre durch den Scheitel der letzten Fläche, wie Merit und Zielprüfung) und zur Gegenprobe `mikroptim eval --grid 64` mit der Referenzsphäre durch die Austrittspupille (Spalte „RMS AP“; Falle F12 aus M6-T6). NA = Pupillenradius / Soll-Brennweite (wie `synth`). Bei den Sitzungen zählt der beste Kandidat des Baums (Ziel, dann NA, dann RMS, dann Elementzahl).

## 3. Ergebnisse

| Bench. | Variante | Ziel | NA | RMS max [Wellen] | RMS AP | Elemente | LM-Läufe¹ | Zeit [s] | Entsch. | Aktionen abgelehnt² | Entsch. mit Fehler | Tokens Prompt / Ausgabe³ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | V0 | **ja** | 0,600 | 0,0163 | 0,0165 | 4 | 63 | 15 | – | – | – | – |
| B1 | V1 | **ja** | 0,600 | 0,0354 | 0,0354 | 3 | 121 | 35 | – | – | – | – |
| B1 | V2 | – | – | – | – | – | – | – | – | – | – | – |
| B1 | V3 | **ja** | 0,600 | 0,0163 | 0,0165 | 4 | 63 | 15 | 2 | 0 % | 0/2 | – |
| B1 | qwen3:14b | nein | 0,443 | 0,0743 | 0,0747 | 4 | 44 | 358 | 4 | 19 % | 3/4 | 137.750 / 2.437 |
| B1 | gpt-oss:20b | **ja** | 0,600 | **0,0063** | 0,0063 | 4 | 74 | 227 | 5 | 14 % | 2/5 | 51.725 / 21.651 |
| B2 | V0 | nein | 0,405 | 0,0845 | 0,1062 | 3 | 44 | 486 | – | – | – | – |
| B2 | V1 | nein | 0,413 | 0,0890 | 0,1141 | 3 | 139 | 876 | – | – | – | – |
| B2 | V2⁴ | nein | 0,94 (Merit) | 21 (real, Achse) | – | 4 | – | 390 (GPU) | – | – | – | – |
| B2 | V3 | nein | 0,413 | 0,0822 | 0,0998 | 3 | 70 | 646 | 8 | 0 % | 0/8 | – |
| B2 | qwen3:14b | nein | 0,300 | 0,5340 | 0,6310 | 8 | 4 | 757 | 12 | 21 % | 10/12 | 531.008 / 7.895 |
| B2 | gpt-oss:20b | nein | 0,405 | 0,0845 | 0,1062 | 3 | 50 | 1184 | 12 | 21 % | 8/12 | 175.755 / 41.482 |
| B3 | V0 | nein | 0,448 | 0,0747 | 0,0779 | 3 | 47 | 500 | – | – | – | – |
| B3 | V1 | nein | 0,448 | 0,0747 | 0,0779 | 3 | 143 | 827 | – | – | – | – |
| B3 | V2 | – | – | – | – | – | – | – | – | – | – | – |
| B3 | V3 | nein | 0,448 | 0,0747 | 0,0779 | 3 | 48 | 601 | 4 | 20 % | 1/4 | – |
| B3 | qwen3:14b | nein | 0,340 | 0,1142 | 0,1652 | 3 | 23 | 392 | 2 | 0 % | 0/2 | 90.555 / 1.041 |
| B3 | gpt-oss:20b | nein | 0,455 | 0,1074 | 0,1112 | 5 | 76 | 1331 | 12 | 26 % | 10/12 | 124.562 / 40.939 |
| B1–B3 | Claude (Opus 5.5, Sonnet 5.5) | ausstehend | | | | | | | | | | |
| B2 | V4 (Mensch) | ausstehend | | | | | | | | | | |

¹ Optimierungen des Homotopie-Controllers (`solves` von `synth` bzw. aller `synth_segment`) plus `optimize`-Aktionen.
² Anteil der Aktionen mit Fehler an allen vorgeschlagenen Aktionen (nach einem Fehler laufen die übrigen Aktionen einer Antwort nicht mehr). Rückfälle auf die Regel gab es in keinem Lauf: Alle Antworten waren schema-gültig.
³ Ollama zählt Denken und Antwort zusammen in der Ausgabe; Denktext gpt-oss: B1 80.000, B2 149.000, B3 146.000 Zeichen.
⁴ Aus [m6-t6-zielspezifikation.md](m6-t6-zielspezifikation.md), Vergleichszweig „Apertur zuerst“ (`--apertur-zweig`, Strahlsuche B = 2, GPU): NA 0,94 bei 10 % Feld mit 4 Elementen in der Merit, real (ohne Hilfsebene, nachoptimiert) 21 Wellen RMS auf der Achse (Falle F12). Andere Einstellungen (16 Ringe, Petzval und Mechanik, Skript-Gegenmaßnahmen), daher nur eingeschränkt vergleichbar. V2 ist für B1 und B3 nicht definiert (das Skript ist auf die Zielspezifikation zugeschnitten).

Alle Ergebnisse, Protokolle (`log.jsonl`, `session.json`, Startsystem) und besten Systeme liegen unter [`m10-t7/`](m10-t7/). Jede Sitzung lässt sich ohne Modell wiederholen, z. B.

```bash
python -m mikroptim.design replay docs/ergebnisse/m10-t7/B1_LM_gpt-oss-20b_think sitzung_wieder
```

(geprüft: bester Kandidat bitgleich).

**Kosten:** keine (keine Claude-Läufe).

## 4. Beobachtungen

**B1.** Regel und V0 sind identisch: Die Regel ruft genau einen Homotopie-Abschnitt auf, der das Ziel erreicht (2 Entscheidungen). gpt-oss wählte nach zwei abgelehnten Aktionen (Glas-Einrasten ohne Glasvariablen, `split_lens` mit Zielglas ohne `free_part 1`) zweimal `synth_segment` mit `max_new_elements: 2`, jeweils gefolgt von `free_gaps` und `optimize`, und beendete mit `finish`. Ergebnis 0,0063 Wellen, also 2,6-mal kleiner als V0 bei gleicher Elementzahl. qwen3 öffnete mit einem Abschnitt bis NA 0,44, teilte dann die Linse, verwarf diesen Kandidaten selbst (`abandon`) und schlug danach Glas-Einrasten ohne Glasvariablen vor; das Budget an Entscheidungen blieb ungenutzt.

**B2, B3.** Der Engpass ist der Homotopie-Controller, nicht die Entscheidung: Jeder `synth_segment` endet mit `step_too_small` bei NA 0,40–0,46, weil keine Geisterplatte Bedarf meldet (F3), und `free_gaps` + `optimize` dazwischen ändern daran wenig (V3: 4 Abschnitte, NA 0,405 → 0,413). Die Modelle haben keine Aktion, die diesen Engpass löst (`insert_module`, `place_ghost_dictionary` u. a. aus M10-T4 fehlen). Was sie versuchen:

- **gpt-oss** teilt Linsen für Farbkorrektur (B2: 11, B3: 10 erfolgreiche strukturelle Änderungen), obwohl der Benchmark monochromatisch ist – es folgt der Ziel-Farbklasse c im Zustand –, und versucht Gläser einzurasten, bevor es Glasvariablen gibt (12 der 18 abgelehnten Aktionen auf B2/B3 sind `snap_glasses` ohne Glasvariablen). Die strukturellen Änderungen verbessern weder NA noch RMS; auf B3 erreicht es mit 5 Elementen NA 0,455, aber RMS 0,107.
- **qwen3** teilt auf B2 Linsen und fügt Menisken ein (7 strukturelle Änderungen, 8 Elemente), versucht aber nur einmal, die Apertur zu öffnen (scheitert an E6, Abschnitt 5); 8 der 10 abgelehnten Aktionen sind Menisken an Stellen ohne Luftspalt. Auf B3 bricht es nach einem Abschnitt mit `abandon` ab („Keine Glasdaten vorhanden“ – die Glasvorauswahl lieferte für das feste Glas keine Partner).
- Beide Modelle erfinden gelegentlich Glasnamen (`F2`, `null`, `buchdahl(...)`) statt Katalognamen aus dem Zustand oder der Vorauswahl zu nehmen.

**Tokens und Zeit.** qwen3 ruft viele Lese-Werkzeuge auf (B2: 55-mal `glass_preselect`), daher 0,5 Mio. Prompt-Tokens bei 12 Entscheidungen; gpt-oss ruft wenige Werkzeuge auf, denkt aber lange (Ausgabe 40.000 Tokens). Die Rechenzeit der Sitzungen liegt beim 1- bis 2,5-fachen von V0; den Großteil kosten die Homotopie-Abschnitte, nicht das Modell.

**Reproduzierbarkeit.** gpt-oss lieferte auf B1 in zwei Läufen (vor und nach den Korrekturen aus Abschnitt 5, die B1 nicht berührten) dieselben Antworten und dasselbe System (Temperatur 0, Seed 0). Verbindlich bleibt das Protokoll (ADR-0012).

## 5. Befunde an den Werkzeugen (behoben)

Vorläufe mit beiden Modellen haben Fehler in `mikroptim.design` gezeigt. Sie sind in diesem PR behoben (mit Tests), alle Sitzungen der Tabelle wurden danach neu gerechnet:

| Nr. | Befund | Behebung |
|---|---|---|
| E1 | `glass_preselect` mit `element: "achromat"` (gpt-oss) warf `ValueError` aus dem Werkzeugaufruf und brach die Sitzung ab; ebenso `IndexError` | Werkzeugaufrufe (Ollama, Claude) und Aktionen melden `ValueError`, `TypeError`, `LookupError` als Fehler (`invalid_parameter`) an den Entscheider |
| E2 | `split_lens` an einer Linse mit Dickenvariable (Schranke 0,4 bzw. 0,5 mm): die Hälften lagen unter der Schranke, `insert` lehnte das System ab (gpt-oss B2/B3, qwen3 B2: 5–8 Fehler pro Lauf) | Schranke der Dickenvariable des Elements vor dem Teilen auf die dünnere Hälfte senken (auch in der Listenform, die Operator B für beide Hälften anlegt) |
| E3 | `free_gaps` legte für Spalte, die dünner als die Lage ihrer Geisterplatte sind, eine Variable mit Startwert unter der Schranke an; der nächste `synth_segment` scheiterte | solche Spalte bekommen keine Variable |
| E4 | `homotopy_point.aperture` im Zustand war die Soll-NA, nicht die aktuelle; Modell und Regel konnten nicht erkennen, wie weit die Apertur offen ist | aktuelle NA = Pupillenradius / Soll-Brennweite, `spec.na` bleibt die Soll-NA; System-Prompt nennt `synth_segment` zum Öffnen der Apertur |
| E5 | Die Regel wiederholte einen gescheiterten `synth_segment` bis zum Budgetende | ein gescheiterter Abschnitt wird nicht wiederholt |

**In M10-T7 offen, mit M10-T7b behoben (Abschnitt 9.1):**

- **E6:** Nach Einfügungen durch `synth` (Operator A in einem Plattenspalt) liegen Startwerte von Dickenvariablen gelegentlich unter ihren Schranken (`variables.items[…].min: Startwert 0 … liegt unter der unteren Schranke 0,125`); der nächste Aufruf von `synth` oder `insert` lehnt das System ab (B3 V3, qwen3 B2). Ursache im C++-Einfügen bzw. in der Fortschreibung der Variablen; Vorschlag: Schranken beim Einfügen anpassen oder beim Laden auf den Startwert klemmen (mit Warnung).
- **E7:** `snap_glasses` ohne Glasvariablen und Glasnamen, die nicht im Katalog stehen, kosten die Modelle viele Entscheidungen. Eine Vorprüfung im Zustand („einrastbar: nein“, Liste der Katalognamen) würde helfen.
- **E8:** Für monochromatische Benchmarks sollte der Zustand keine Ziel-Farbklasse nennen (gpt-oss verfolgte sie).

## 6. Ausstehend und Anleitung zum Nachholen

### Claude-Läufe (M10-T6c)

Voraussetzung: Zugangsdaten in der Umgebung (`ANTHROPIC_API_KEY` setzen oder `ant auth login`), SDK als Extra, Python-Wheel wie in [python/README.md](../../python/README.md). Budget: höchstens 20 US-$ für alle Läufe zusammen; das Skript summiert in einer gemeinsamen Kostendatei und bricht vor Überschreiten ab (Exit 3). Aus dem Wurzelverzeichnis:

```bash
pip install "mikroptim[claude]"            # bzw. pip install "anthropic>=2.10" zum vorhandenen Wheel
export ANTHROPIC_API_KEY=...               # nie in Dateien oder Protokolle schreiben
KOSTEN=docs/ergebnisse/m10-t7-arbeit/claude_cost.json
python tools/m10t7_bewertung.py --benchmarks B1,B2,B3 --varianten LM --decider claude \
    --model claude-sonnet-5-5 --effort medium --max-usd 20 --cost-file $KOSTEN
python tools/m10t7_bewertung.py --benchmarks B1,B2,B3 --varianten LM --decider claude \
    --model claude-opus-5-5 --effort medium --max-usd 20 --cost-file $KOSTEN
python tools/m10t7_bewertung.py --archiv --tabelle      # Protokolle übernehmen, Tabelle
```

Unter PowerShell die Listen in Anführungszeichen setzen (`--benchmarks 'B1,B2,B3'`). Erwartete Kosten (Schätzung aus den Ollama-Läufen: 50.000–500.000 Prompt-Tokens und 20.000–40.000 Ausgabe-Tokens pro Lauf, mit Caching meist Cache-Lesen): Sonnet 5.5 etwa 0,3–1,5 $ pro Lauf, Opus 5.5 etwa das Doppelte; alle sechs Läufe zusammen voraussichtlich unter 10 $. Die Kosten pro Lauf stehen danach in der Tabelle (Spalte „Kosten“) und in `claude_cost.json`. Bricht ein Lauf an der API ab, steht die Meldung im Ergebnis unter `aborted` (Ergebnisdatei löschen und neu starten). Die API lehnte die abgeleiteten strict-Schemas ab (Fehler 400, zu viele optionale Parameter); seit M10-T7c ist `--no-strict` der Standard (Abschnitt 11), die Antworten werden weiter gegen das Aktionsschema geprüft.

### V4 (Mensch, mindestens B2)

```bash
python tools/m10t7_bewertung.py --benchmarks B2 --varianten V4
```

Der Zustand erscheint als Text, die Antwort wird als JSON (`mikroptim.design.action/1`) eingegeben, Beispiel in [python/README.md](../../python/README.md#entwurfswerkzeuge-m10-t6a); dieselben Prüfungen und Werkzeuge wie für die Modelle. Zum Nachschlagen in einer zweiten Konsole: `python -m mikroptim.design state <sitzung> -c N --json`.

### B4

Erst nach Entscheidung des Maintainers über Verzeichnis und Format der Patentsysteme; nur mit Ollama (lokal).

## 7. Folgerungen

1. Für die Aperturstufe bei kleinem Feld ist der Homotopie-Controller der Engpass (F3); ohne neue Aktionen (M10-T4: `insert_module`, `place_ghost_dictionary`, `remove_element`) können weder Regel noch Modell mehr erreichen als V0. Die Bewertung mit Modellen sollte nach M10-T4 wiederholt werden.
2. gpt-oss:20b (Thinking) nutzt die Werkzeuge sinnvoll, wenn die Aufgabe klein ist (B1), und erzeugt dort ein besseres System als die Regel. qwen3:14b ohne Thinking ist schwächer als die Regel.
3. Die Werkzeugfehler E1–E3 hätten auch einen Menschen behindert; die Läufe mit Modellen sind damit auch ein Test der Werkzeuge.
4. Entscheidung nach ADR-0012, Punkt 10, erst nach den Claude-Läufen und nach M10-T4: Ein Vorteil eines Modells gegenüber der Regel ist bisher nur auf B1 belegt.

## 8. Reproduktion

```bash
python -m pip wheel . -w dist --no-deps && python -m pip install --no-index --find-links dist mikroptim
python tools/m10t7_bewertung.py --varianten V0,V1,V3                        # ohne Modell
python tools/m10t7_bewertung.py --varianten LM --decider ollama --model qwen3:14b --no-think
python tools/m10t7_bewertung.py --varianten LM --decider ollama --model gpt-oss:20b --think
python tools/m10t7_bewertung.py --neu-bewerten --archiv --tabelle
```

Seit M10-T7b schreibt das Skript standardmäßig nach `docs/ergebnisse/m10-t7b` (Abschnitt 9.5), seit M10-T7c nach `docs/ergebnisse/m10-t7c` mit `image_infinite` (Abschnitt 10); für den Stand von M10-T7 `--out docs/ergebnisse/m10-t7 --referenzsphaere last_surface --num-ctx 32768`. Vorhandene Ergebnisse werden übersprungen; zum Neurechnen die Datei `<out>/<B>_<Variante>.json` löschen. Das Arbeitsverzeichnis (`--work`, Standard `docs/ergebnisse/m10-t7-arbeit`) gehört nicht ins Repository.

## 9. M10-T7b: Wiederholung mit den Operatoren aus M10-T4

Aufgabe M10-T7b: Bewertung mit den neuen Operatoren (M10-T4: `insert_module`, `place_ghost_dictionary`, `remove_element`, `replace_glass`, `replace_group`, `start_from_skeleton`) wiederholen, die offenen Befunde E6–E8 beheben und die Regel V3 so erweitern, dass sie die Operatoren nutzt, wenn die Homotopie stehen bleibt. **Code-Stand aller Zahlen dieses Abschnitts:** `main` 9cc89a7 (nach M10-T5 Wissensbasis, M11-T6/T6b/T7, den Folgeentscheidungen #97, M6-T7/ADR-0017 und M11-T5g) plus die Änderungen dieses PRs, außer wo ausdrücklich der Zwischenstand 8d3aa3c genannt ist (nach dem letzten Rebase auf 9da0d30 geprüft: B1 mit V0, V3 und V3RS unverändert); Ergebnisse, Protokolle und beste Systeme unter [`m10-t7b/`](m10-t7b/). Die Zahlen der Abschnitte 1–8 bleiben der Stand 717a8ff (M10-T7) und dienen als „alt“.

### 9.1 Änderungen

**Regel V3 (`RuleDecider`, [entwurfssteuerung.md, Abschnitt 2.1](../entwurfssteuerung.md#21-entscheider)).** Bleibt ein `synth_segment` unter der Soll-NA stehen (in allen Läufen `step_too_small`) oder scheitert er, folgt je einmal die nächste Stufe, danach ein neuer Abschnitt (höchstens 5):

1. `free_gaps` + `optimize` (wie bisher),
2. aplanatische Schale `front.shell_ca` im Arbeitsabstand aus dem Katalogglas mit dem größten n₀ (`mikroptim glasses`); Dicke so, dass der nachfokussierte Arbeitsabstand (w − p)/n′ − t ≈ 1,05 w_min bleibt, sonst übersprungen; dann `free_gaps` + `optimize`,
3. Geisterplatten-Wörterbuch (hochbrechendes Glas und die Katalognamen der vorhandenen Platten) in der Mitte des Luftspalts mit der größten Randstrahlhöhe, sofort gefolgt vom Abschnitt (F3: Bedarf sichtbar machen),
4. Null-Meniskus `generic.null_meniscus` im selben Spalt mit Dicken-Homotopie (4 Stufen à 50 Zyklen, Dicke höchstens so groß, dass der Luftspalt dahinter über seiner Schranke bleibt), dann `free_gaps` + `optimize`,
5. danach `free_gaps` + `optimize` und weitere Abschnitte, solange der letzte die NA um ≥ 0,01 geöffnet hat.

Keine Linsendaten realer Objektive: alle Geometrien entstehen aus den Vorlagen und den Strahldaten des Kandidaten. Variante **V3AL** ist dieselbe Regel mit `augmented_lagrangian: 5` und `sine_condition: 13,1` (Gewicht wie im Versuch M11-T4) in jedem Abschnitt (`synth_segment` kennt beide Parameter jetzt).

**Befunde E6–E8 (Abschnitt 5) behoben:**

| Nr. | Ursache | Behebung | Test |
|---|---|---|---|
| E6 | Operator A in `synth` setzt den Meniskus an die Lage der aktiven Geisterplatte. Eine frühere Dicken-Homotopie hatte Platten ans Spaltende geschoben; beide Teilstücke erben die Dickenvariable des Spalts, das Teilstück der Dicke 0 lag unter deren Schranke, `insert_meniscus` warf `FormatError` (kein `invalid_argument`), `synth` brach ab | `meniscus_from_diagnosis` rückt die Lage auf [b, T − t − b] in den Spalt (bei Dicke 0 merit-neutral); passt der Spalt nicht, `std::invalid_argument` und der Controller endet geordnet | `InsertionInit.MeniscusFromDiagnosisKeepsGapBounds` (C++); B3 V3: der zweite Abschnitt, der in M10-T7 an E6 scheiterte, läuft jetzt (NA 0,4475 → 0,5025) |
| E7 | Modelle riefen `snap_glasses` ohne Glasvariablen und erfundene Glasnamen auf | Zustand `glass_check` (`snappable`, `glass_variables`, `model_glass_elements`, Katalognamen); `snap_glasses` ohne Glasvariablen → `precondition`; Glasnamen außerhalb der Kataloge → `unknown_glass` vor jedem Aufruf | `test_state_glass_check_and_snap_precondition`, `test_unknown_glass_name_is_rejected`, `test_catalog_names_json_and_agf` |
| E8 | Zustand nannte bei monochromatischen Benchmarks die Ziel-Farbklasse c | `spec.monochromatic`; bei einer Wellenlänge ist `target_color_class` `null`, die Farbregel von V3 greift nicht | `test_state_monochromatic_has_no_target_color_class` |

**Weitere Befunde der Läufe (behoben, mit Tests):**

| Nr. | Befund | Behebung |
|---|---|---|
| E9 | Nach `optimize` mit freier Glasvariable (qwen3, B1) lag eine Brechzahl ≤ 0 vor; der Zustand des neuen Kandidaten ließ sich nicht bilden, die Sitzung brach ab | neue Kandidaten werden vor der Aufnahme ausgewertet (Trace und Aberrationsbericht), sonst Fehler `invalid_result` (`test_unevaluable_result_is_rejected`) |
| E10 | `insert_module` mit Dicke: die Nachoptimierung der vorigen Stufe schob den Luftspalt dahinter an seine Schranke, die nächste Stufe unterschritt sie, `optimize` lehnte ab (V3 B3) | der Luftspalt bleibt auf der Schranke, die folgenden Scheitel rücken (`test_thickness_homotopy_keeps_gap_bound`) |
| E11 | Die Regel wiederholte eine scheiternde Stufe bis zum Budgetende | jede Stufe nur einmal, auch wenn sie scheitert |
| E12 | Ollama antwortete gelegentlich mit HTTP 500 (Modell lädt nicht, solange andere Agenten die RTX 3090 belegen); der Lauf brach ab | `OllamaDecider` wiederholt Serverfehler dreimal nach 20 s (`test_ollama_retries_server_error`); das Skript schreibt bei HTTP-Fehlern kein Ergebnis |
| E13 | Nach `replace_group` (gpt-oss, B1) war der Bildraum afokal (u′ = 0); die Feldklasse teilte durch u′², der Zustand brach mit `ZeroDivisionError` ab und mit ihm die Sitzung | keine Bildschale bei u′ = 0 (Feldklasse 1); die Prüfung neuer Kandidaten (E9) bildet jetzt den Zustand ohne Merit und Empfindlichkeiten und meldet jeden Fehler als `invalid_result` (`test_afocal_image_space_has_no_field_shift`); der Lauf wurde danach wiederholt (die ersten elf Antworten des Modells gleich, Temperatur 0, Seed 0) |
| – | Hinweis im Zustand „replace_glass mit einem Katalognamen“ (erste Fassung von `glass_check`) | qwen3 tauschte daraufhin nur noch Gläser (Vorlauf, verworfen); `glass_check.reason` nennt jetzt nur den Befund |

Neu als Werkzeuge (im Teilumfang, für die Regel und die Modelle): `set_bounds` (`wd`, `t[i]`; Startwert wird geklemmt) und `set_constraint` (`weight`, `enabled: false`, `paraxial_focus` neu anlegen).

### 9.2 Ergebnisse alt gegen neu

Bewertung wie in Abschnitt 2 (reales System ohne Geisterplatten, `trace` 10 × 8 in der Konvention der Datei; „RMS AP“ = `eval --grid 64` durch die Austrittspupille, in allen Varianten gleich und damit die vergleichbare Spalte). **Neue Varianten V0RS und V3RS** (Abschnitt 9.3): wie V0 bzw. V3, Start mit `reference_sphere: image_infinite` (ADR-0017, M6-T7). Zeiten auf dem geteilten Rechner (bis zu zehn Läufe und andere Agenten gleichzeitig), nur Orientierung.

| Bench. | Variante | Ziel alt | NA alt | RMS alt | Elem. alt | Zeit alt [s] | Ziel neu | NA neu | RMS neu | RMS AP neu | Elem. neu | Zeit neu [s] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | V0 | **ja** | 0,600 | 0,0163 | 4 | 15 | **ja** | 0,600 | 0,0163 | 0,0165 | 4 | 15 |
| B1 | V1 | **ja** | 0,600 | 0,0354 | 3 | 35 | **ja** | 0,600 | 0,0354 | 0,0354 | 3 | 34 |
| B1 | V3 | **ja** | 0,600 | 0,0163 | 4 | 15 | **ja** | 0,600 | 0,0163 | 0,0165 | 4 | 16 |
| B1 | V3AL | – | – | – | – | – | **ja** | 0,600 | 0,0111 | 0,0115 | 5 | 23 |
| B1 | V0RS | – | – | – | – | – | **ja** | 0,600 | 0,0154 | 0,0156 | 4 | 16 |
| B1 | V3RS | – | – | – | – | – | **ja** | 0,600 | 0,0154 | 0,0156 | 4 | 16 |
| B1 | qwen3:14b | nein | 0,443 | 0,0743 | 4 | 358 | nein | 0,300 | 22,1581 | 21,2078 | 1 | 763 |
| B1 | gpt-oss:20b | **ja** | 0,600 | 0,0063 | 4 | 227 | nein | 0,443 | 0,0761 | 0,0765 | 3 | 1470 |
| B2 | V0 | nein | 0,405 | 0,0845 | 3 | 486 | nein | 0,405 | 0,0845 | 0,1062 | 3 | 415 |
| B2 | V1 | nein | 0,413 | 0,0890 | 3 | 876 | nein | 0,413 | 0,0890 | 0,1141 | 3 | 769 |
| B2 | V2 | nein | 0,94 (Merit) | 21 (real) | 4 | 390 (GPU) | – | – | – | – | – | – |
| B2 | V3 | nein | 0,413 | 0,0822 | 3 | 646 | nein | 0,413 | 0,0818 | 0,0988 | 4 | 754 |
| B2 | V3AL | – | – | – | – | – | nein | 0,410 | 0,0759 | 0,0843 | 4 | 784 |
| B2 | V0RS | – | – | – | – | – | nein | 0,453 | 0,0797 | 0,0802 | 4 | 621 |
| B2 | V3RS | – | – | – | – | – | nein | 0,790 | 0,0907 | 0,0913 | 4 | 1239 |
| B2 | qwen3:14b | nein | 0,300 | 0,5340 | 8 | 757 | nein | 0,300 | 1,7930 | 2,0921 | 1 | 68 |
| B2 | gpt-oss:20b | nein | 0,405 | 0,0845 | 3 | 1184 | nein | 0,530 | 0,0920 | 39,4328 | 5 | 33390 |
| B3 | V0 | nein | 0,448 | 0,0747 | 3 | 500 | nein | 0,448 | 0,0747 | 0,0779 | 3 | 500 |
| B3 | V1 | nein | 0,448 | 0,0747 | 3 | 827 | nein | 0,448 | 0,0747 | 0,0779 | 3 | 805 |
| B3 | V3 | nein | 0,448 | 0,0747 | 3 | 601 | nein | 0,833 | 0,0984 | 0,1027 | 4 | 1135 |
| B3 | V3AL | – | – | – | – | – | nein | 0,525 | 0,0791 | 0,0837 | 6 | 1382 |
| B3 | V0RS | – | – | – | – | – | nein | 0,375 | 0,0799 | 0,0806 | 3 | 482 |
| B3 | V3RS | – | – | – | – | – | **ja** | 0,850 | 0,0595 | 0,0603 | 4 | 1062 |
| B3 | qwen3:14b | nein | 0,340 | 0,1142 | 3 | 392 | nein | 0,300 | 2,2413 | 2,6151 | 1 | 288 |
| B3 | gpt-oss:20b | nein | 0,455 | 0,1074 | 5 | 1331 | nein | 0,300 | 2,2413 | 2,6151 | 1 | 716 |

Weitere Kennzahlen der neuen Läufe (vollständige Tabelle: `python tools/m10t7_bewertung.py --tabelle`):

| Bench. | Variante | LM-Läufe | Entscheidungen | Aktionen abgelehnt | Tokens Prompt / Ausgabe | Denktext [Zeichen] |
|---|---|---|---|---|---|---|
| B1 | V0 | 63 | – | – | – | – |
| B1 | V1 | 121 | – | – | – | – |
| B1 | V3 | 63 | 2 | 0 % | – | – |
| B1 | V3AL | 73 | 6 | 0 % | – | – |
| B1 | V0RS | 63 | – | – | – | – |
| B1 | V3RS | 63 | 2 | 0 % | – | – |
| B1 | qwen3:14b | 0 | 12 | 24 % | 525222 / 7994 | – |
| B1 | gpt-oss:20b | 62 | 12 | 21 % | 133616 / 59537 | 224484 |
| B2 | V0 | 44 | – | – | – | – |
| B2 | V1 | 139 | – | – | – | – |
| B2 | V3 | 69 | 7 | 0 % | – | – |
| B2 | V3AL | 78 | 7 | 0 % | – | – |
| B2 | V0RS | 67 | – | – | – | – |
| B2 | V3RS | 138 | 11 | 0 % | – | – |
| B2 | qwen3:14b | 0 | 2 | 20 % | 46704 / 754 | – |
| B2 | gpt-oss:20b | 76 | 12 | 30 % | 178204 / 138708 | 553376 |
| B3 | V0 | 47 | – | – | – | – |
| B3 | V1 | 143 | – | – | – | – |
| B3 | V3 | 98 | 9 | 0 % | – | – |
| B3 | V3AL | 128 | 11 | 0 % | – | – |
| B3 | V0RS | 42 | – | – | – | – |
| B3 | V3RS | 103 | 5 | 0 % | – | – |
| B3 | qwen3:14b | 0 | 4 | 27 % | 199838 / 3060 | – |
| B3 | gpt-oss:20b | 0 | 6 | 25 % | 81412 / 91496 | 376630 |

**gpt-oss:20b auf B3** brach zweimal bei der 7. Entscheidung mit HTTP 500 von Ollama ab, auch nach drei Wiederholungen (E12); die Zeile ist das Teilergebnis der ersten sechs Entscheidungen (in dieser Zeit 377.000 Zeichen Denktext – vermutlich überschritt der Verlauf den Kontext von 32.768 Tokens). gpt-oss auf B2 brauchte 9,3 h (lange Denkphasen bei voller GPU), die anderen Läufe 0,1–0,7 h.

**Modelle auf dem Zwischenstand 8d3aa3c** (nach M10-T5, vor den Folgeentscheidungen des Maintainers #97 und M6-T7; gleiche Regel und Werkzeuge dieses PRs, archiviert nicht im Repository). Die Prompts unterscheiden sich vom Endstand im Zustand (Gruppen aus #97), die Modelle entschieden deshalb anders:

| Bench. | Modell | Ziel | NA | RMS | RMS AP | Elemente | Zeit [s] | Tokens Prompt / Ausgabe |
|---|---|---|---|---|---|---|---|---|
| B1 | qwen3:14b | nein | 0,300 | 2,5602 | 2,8624 | 1 | 378 | 297345 / 5602 |
| B1 | gpt-oss:20b | **ja** | 0,600 | 0,0061 | 0,0062 | 4 | 2496 | 152512 / 69964 |
| B2 | qwen3:14b | nein | 0,300 | 0,9004 | 1,0477 | 1 | 376 | 365435 / 5687 |
| B2 | gpt-oss:20b | nein | 0,413 | 0,0822 | 0,0998 | 3 | 1790 | 193093 / 82016 |
| B3 | qwen3:14b | nein | 0,300 | 1,0423 | 1,1357 | 1 | 379 | 490324 / 9152 |
| B3 | gpt-oss:20b | nein | 0,448 | 0,0747 | 0,0779 | 3 | 1827 | 169382 / 46945 |

**Kosten:** keine. `ANTHROPIC_API_KEY` ist in der Umgebung nicht gesetzt (nur auf Existenz geprüft); die Claude-Läufe bleiben ausstehend (Abschnitt 6, Budget unverändert höchstens 20 US-$).

### 9.3 Beobachtungen

**B3 (40x/0,85) wird erreicht – mit Regel und neuer Referenzsphäre.** V3RS erreicht das Ziel: NA **0,850**, RMS **0,060** Wellen (AP 0,060), **4 Elemente**, 5 Entscheidungen. Verlauf: 0,375 → 0,39 (Spaltfreigabe) → **0,85** im Abschnitt nach dem Geisterplatten-Wörterbuch im Spalt größter Randstrahlhöhe (der Controller sieht dort Bedarf und fügt Elemente ein). Ohne Regel (V0RS) bleibt der Controller mit der neuen Sphäre bei 0,375 stehen, ehrlicher als V0 (0,448 mit alter Sphäre, siehe M6-T7: der Zweig „Apertur zuerst“ bei 10 % Feld endet dort ebenfalls bei 0,47). Mit der alten Sphäre kommt V3 bis NA 0,833 (4 Elemente, RMS 0,098; alt 0,448): 0,4475 → 0,5025 (Spalte; dieser Abschnitt scheiterte in M10-T7 an E6) → 0,5475 (Wörterbuch) → **0,8125** (Null-Meniskus mit Dicken-Homotopie im Spalt größter Randstrahlhöhe) → 0,8325 (Wiederholung), dann ist das Budget von 5 Abschnitten erschöpft.

**B2 (50x/0,95): von 0,41 auf 0,79.** Mit alter Sphäre bewegen die Stufen die NA kaum (V3: 0,405 → 0,41 → 0,4125 → 0,4125). Die Schale ist dort nie anwendbar: Am Stillstand liegt der Arbeitsabstand auf seiner unteren Schranke (0,15 mm), die Schale verkürzt ihn etwa auf (w − p)/n′ − t. Platz dafür zu schaffen (Schranke des Arbeitsabstands auf 1,1 mm heben und optimieren; Vorlauf mit `set_bounds`/`set_constraint`) scheiterte an der **Falle F12**: Der LM legte den Fokus in den Scheitel der Hilfsebene (Merit-RMS 0,12, durch die Austrittspupille 12 Wellen, mit `paraxial_focus` bis Gewicht 10⁴ unverändert); deshalb ist dieser Weg nicht Teil der Regel. **Mit der neuen Sphäre (V3RS)** öffnet das Wörterbuch die Apertur von 0,465 auf **0,765**, der Null-Meniskus auf **0,79** (4 Elemente, RMS 0,091, AP 0,091); danach lehnt der Controller den Start ab. Das Ziel 0,95 bleibt fern; die Schale ist auch hier nicht anwendbar.

**V3AL** (Augmented Lagrangian + Sinusbedingung, alte Sphäre) hilft der Apertur nicht: B2 0,41 (wie V3, RMS 0,076 statt 0,082), B3 0,525 mit 6 Elementen (die Schale passt dort, weil der Arbeitsabstand nach der Spaltfreigabe noch 0,65 mm ist, bringt aber nur 0,495 → 0,52; Wörterbuch und Null-Meniskus danach je < 0,01). Auf B1 ändert sie den Weg (zweimal `step_too_small`, dann Schale, dann Ziel) und liefert RMS 0,011 statt 0,016 mit einem Element mehr. Fazit: AL und Sinusbedingung verbessern den RMS bei gleicher NA, kosten aber die Freiheit, die der Null-Meniskus-Schritt auf B3 brauchte.

**Einordnung F11/F12.** Die Läufe bestätigen den Befund aus M11-T4: Mit der an die letzte Fläche gebundenen Referenzsphäre ist der Engpass bei hoher NA nicht die Entscheidung, sondern die Falle F12. Die Konventionsänderung hat der Maintainer inzwischen getroffen (ADR-0017, M6-T7, angenommen 2026-11-19); dieser PR ändert keine Konvention, sondern nutzt das neue Dateifeld in den Varianten …RS. Die Benchmark-Starts selbst bleiben ohne Feld (alte Konvention, vergleichbar mit M10-T7); **Empfehlung:** die Bewertung künftig mit `image_infinite` führen (V3RS als neue Referenz der Regel).

**Modelle.** Auf dem Endstand ist kein Modell besser als die Regel. qwen3:14b (ohne Thinking) fügt Kittglieder ohne Glas ein (`glasses fehlt`, B1: 11 Fehler) und verwirft auf B2 und B3 nach zwei bzw. vier Entscheidungen den Startkandidaten (`abandon`); NA 0,30 überall. gpt-oss:20b (Thinking) verfehlt B1 diesmal (NA 0,4425, 3 Elemente; auf dem Zwischenstand 8d3aa3c erreichte es B1 mit dem besten System aller Varianten, RMS 0,0061): Es teilt Linsen, tauscht Gläser und versucht Module – 21 % der Aktionen abgelehnt, meist die Schale ohne Glas oder Module an Stellen ohne Luftspalt. Auf B2 erreicht es NA 0,53 mit 5 Elementen, aber nur in der Falle F12 (Merit-RMS 0,092, durch die Austrittspupille 39 Wellen) – die alte Sphäre täuscht auch die Bewertung von Modellen. Der Null-Meniskus mit Dicke und das Wörterbuch an der Stelle großer Randstrahlhöhe, die der Regel B3 öffnen, wählt keines der Modelle gezielt.

**Werkzeugfehler, die die Modelle trafen,** sind jetzt strukturierte Fehler (`unknown_glass`, `snap_glasses` ohne Glasvariablen, `invalid_result`). Offen und für Folgeaufgaben notiert: Die Vorgabedicke der Schale in `insert_module` (30 % des Spalts) verletzt fast immer die Schranke des Arbeitsabstands; `insert_module` ohne `glasses` (häufigster Fehler beider Modelle, 3–11 pro Lauf) könnte das hochbrechendste Katalogglas bzw. eine Glasregel der Wissensbasis wählen.

### 9.4 Folgerungen

1. **Die neuen Operatoren lösen den Stillstand:** Mit der alten Sphäre öffnet der Null-Meniskus mit Dicken-Homotopie B3 von 0,55 auf 0,81 (V3: 0,833); mit der Sphäre nach ADR-0017 öffnet das Wörterbuch B3 bis zum **Ziel (4 Elemente)** und B2 bis 0,79. Für B2 fehlen weiterhin Schritte jenseits von NA 0,8 (Kandidaten: zweiter Null-Meniskus, Schale nach einem Umbau der Frontgruppe, Wiederholung bei größerem Budget).
2. **V3 gegen Modell:** Die Regel ist auf B2 und B3 klar besser als beide lokalen Modelle; gpt-oss war nur auf dem Zwischenstand bei B1 besser. Nach ADR-0012, Punkt 10, rechtfertigen die Ergebnisse keinen Ausbau der Anbindung lokaler Modelle; die Bewertung mit Claude steht aus.
3. **Augmented Lagrangian + Sinusbedingung** (V3AL) helfen der Apertur nicht, verbessern aber den RMS bei gleicher NA (B1, B2).
4. **Nächste Schritte:** Benchmarks auf `image_infinite` umstellen; Claude-Läufe (Abschnitt 6) auf dem Stand dieses PRs; Vorgabeglas und -dicke in `insert_module`; Kontextgrenze von gpt-oss (B3) prüfen.

### 9.5 Reproduktion

Seit M10-T7c (Abschnitt 10) sind Standardverzeichnis und Standardkonvention andere; für den Stand dieses Abschnitts:

```bash
O=docs/ergebnisse/m10-t7b
python tools/m10t7_bewertung.py --out $O --varianten V0-alt,V1-alt,V3-alt,V3AL-alt,V0RS,V3RS
python tools/m10t7_bewertung.py --out $O --varianten LM-alt --decider ollama --model qwen3:14b --no-think --num-ctx 32768
python tools/m10t7_bewertung.py --out $O --varianten LM-alt --decider ollama --model gpt-oss:20b --think --num-ctx 32768
python tools/m10t7_bewertung.py --out $O --archiv --tabelle
python tools/m10t7_bewertung.py --out $O --vergleich docs/ergebnisse/m10-t7       # alt gegen neu
```

Jede Sitzung lässt sich ohne Modell wiederholen (`python -m mikroptim.design replay docs/ergebnisse/m10-t7b/<B>_<Variante> <ziel>`; geprüft an B1 V3AL: bestes System bitgleich).

## 10. M10-T7c: Benchmarks mit `image_infinite`, V3RS als Referenz

**Entscheidung Maintainer 2026-11-20:** Die Benchmarks der Entwurfssteuerung rechnen ab jetzt mit `reference_sphere: image_infinite` (ADR-0017, Empfehlung aus Abschnitt 9.3); **V3RS ist die neue Referenz der Regel.** Umgesetzt in `tools/m10t7_bewertung.py`, ohne neue Läufe:

- Standard `--referenzsphaere image_infinite` für alle Varianten; das Skript setzt das Feld in der Startkopie (`start_<B>_rs.json` im Arbeitsverzeichnis). Startsysteme unter `data/systems/` und die Ergebnisdateien unter `m10-t7/`, `m10-t7b/` bleiben unverändert.
- Ergebnis-Kürzel nach Konvention: `V0` → `V0RS`, `V1` → `V1RS`, `V3` → `V3RS`, `V3AL` → `V3ALRS`, `LM` → `LMRS_<modell>`. Die alte Konvention (`last_surface`, Startdatei ohne Feld, bitgleich zu M10-T7b) bleibt als `V0-alt`, `V3-alt`, `LM-alt`, … mit den Kürzeln dieses Berichts (`V0`, `V3`, `LM_<modell>`) abrufbar, ebenso global mit `--referenzsphaere last_surface`; `V0RS`/`V3RS` sind weiter gültige Namen. Jede neue Ergebnisdatei nennt die Konvention im Feld `reference_sphere`; die Tabelle (`--tabelle`) nennt V3RS als Referenz der Regel.
- Standard-Ergebnisverzeichnis `docs/ergebnisse/m10-t7c` (Arbeitsverzeichnis `m10-t7c-arbeit`, nicht im Repository); `--vergleich docs/ergebnisse/m10-t7b` stellt gleiche Kürzel gegenüber (z. B. V3RS gegen V3RS aus Abschnitt 9).
- **Ollama-Kontext:** `--num-ctx` (Standard 65.536) statt fest 32.768; begrenzt auf das Modellmaximum laut `/api/show` (`OllamaDecider.model_context_length`), `0` = Maximum. Installiert (geprüft 2026-11-20): gpt-oss:20b 131.072 Tokens, qwen3:14b 40.960 Tokens – qwen3 läuft also mit 40.960, gpt-oss mit 65.536. Hintergrund: Der Abbruch von gpt-oss:20b auf B3 mit HTTP 500 (Abschnitt 9.2, E12) überschritt vermutlich 32.768 Tokens.

**Neue Läufe** (Claude-Läufe aus Abschnitt 6 auf diesem Stand):

```bash
python tools/m10t7_bewertung.py --varianten V0,V1,V3,V3AL                  # V0RS … V3ALRS nach m10-t7c
KOSTEN=docs/ergebnisse/m10-t7c-arbeit/claude_cost.json
python tools/m10t7_bewertung.py --benchmarks B1,B2,B3 --varianten LM --decider claude \
    --model claude-sonnet-5-5 --effort medium --max-usd 20 --cost-file $KOSTEN
python tools/m10t7_bewertung.py --benchmarks B1,B2,B3 --varianten LM --decider claude \
    --model claude-opus-5-5 --effort medium --max-usd 20 --cost-file $KOSTEN
python tools/m10t7_bewertung.py --archiv --tabelle
python tools/m10t7_bewertung.py --vergleich docs/ergebnisse/m10-t7b
```

## 11. M10-T7c: Claude-Läufe

Die Claude-Läufe aus Abschnitt 6 auf dem Stand von M10-T7c (Referenzsphäre `image_infinite`, Abschnitt 10): Claude **Haiku 5.5**, **Sonnet 5.5** und **Opus 5.5** (`--effort medium`, Rückfall auf die Regel, 12 Entscheidungen) auf B1–B3, dazu die Referenz der Regel V3RS neu gerechnet. Die Läufe deckten Fehler der Werkzeuge auf (11.2); sie sind behoben, und alle drei Modelle sind danach neu gerechnet. Ergebnisse, Protokolle und beste Systeme liegen unter [`m10-t7c/`](m10-t7c/) (Kürzel `LMRS_<modell>`; `_vorfix` = vor den Korrekturen, `_z1` … `_z4` = Zwischenstände, 11.3). Gerechnet auf dem Rechner des Maintainers, CPU im Prozess, zum Teil zwei Läufe gleichzeitig; die Zeiten sind nur Orientierung. Code-Stand: `main` f561043 plus dieser PR.

### 11.1 Ergebnisse

| Bench. | Variante | Ziel | NA | RMS max [Wellen] | RMS AP | Elemente | Zeit [s] | Entsch. | abgelehnt | Kosten [$] | erster Schritt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | Regel V3RS | **ja** | 0,600 | 0,0154 | 0,0156 | 4 | 15 | 2 | 0 % | – | öffnen |
| B1 | qwen3:14b¹ | nein | 0,300 | 22,1581 | 21,2078 | 1 | 763 | 12 | 24 % | – | – |
| B1 | gpt-oss:20b¹ | nein | 0,443 | 0,0761 | 0,0765 | 3 | 1470 | 12 | 21 % | – | öffnen |
| B1 | Haiku 5.5 | nein | 0,300 | 0,5107 | 0,5139 | 1 | 85 | 12 | 28 % | 0,02 | vorbereiten |
| B1 | Sonnet 5.5 vor Fix | nein | 0,300 | 0,5107 | 0,5139 | 1 | 22 | 6 | 17 % | 0,10 | vorbereiten |
| B1 | Sonnet 5.5 nach Fix | nein | 0,300 | 0,5107 | 0,5139 | 1 | 47 | 12 | 15 % | 0,18 | vorbereiten |
| B1 | Opus 5.5 vor Fix | **ja** | 0,600 | 0,0011 | 0,0011 | 4 | 139 | 12 | 26 % | 0,53 | öffnen |
| B1 | Opus 5.5 nach Fix | **ja** | 0,600 | 0,0236 | 0,0238 | 3 | 141 | 12 | 14 % | 0,41 | öffnen |
| B2 | Regel V3RS | nein | 0,790 | 0,0907 | 0,0913 | 4 | 1215 | 11 | 0 % | – | öffnen |
| B2 | qwen3:14b¹ | nein | 0,300 | 1,7930 | 2,0921 | 1 | 68 | 2 | 20 % | – | – |
| B2 | gpt-oss:20b¹ | nein | 0,530 | 0,0920 | 39,4328 | 5 | 33390 | 12 | 30 % | – | öffnen |
| B2 | Haiku 5.5 | nein | 0,453 | 0,0802 | 0,0806 | 4 | 521 | 12 | 32 % | 0,02 | öffnen |
| B2 | Sonnet 5.5 vor Fix | nein | 0,300 | 0,3191 | 0,3215 | 1 | 51 | 11 | 18 % | 0,14 | vorbereiten |
| B2 | Sonnet 5.5 nach Fix | nein | 0,300 | 0,3227 | 0,3251 | 1 | 58 | 12 | 0 % | 0,17 | vorbereiten |
| B2 | Opus 5.5 vor Fix | nein | 0,408 | 0,0827 | 0,0835 | 3 | 613 | 12 | 18 % | 0,61 | öffnen |
| B2 | Opus 5.5 nach Fix | nein | 0,458 | 0,0793 | 0,0798 | 4 | 762 | 12 | 23 % | 0,69 | öffnen |
| B3 | Regel V3RS | **ja** | 0,850 | 0,0595 | 0,0603 | 4 | 996 | 5 | 0 % | – | öffnen |
| B3 | qwen3:14b¹ | nein | 0,300 | 2,2413 | 2,6151 | 1 | 288 | 4 | 27 % | – | – |
| B3 | gpt-oss:20b¹ | nein | 0,300 | 2,2413 | 2,6151 | 1 | 716 | 6 | 25 % | – | – |
| B3 | Haiku 5.5 | nein | 0,375 | 0,0799 | 0,0806 | 3 | 520 | 12 | 30 % | 0,02 | öffnen |
| B3 | Sonnet 5.5 vor Fix | nein | 0,300 | 0,3945 | 0,3974 | 1 | 50 | 11 | 19 % | 0,14 | vorbereiten |
| B3 | Sonnet 5.5 nach Fix | nein | 0,510 | 0,0781 | 0,0797 | 4 | 70 | 12 | 0 % | 0,18 | vorbereiten |
| B3 | Opus 5.5 vor Fix | **ja** | 0,850 | 0,0145 | 0,0146 | 4 | 1105 | 12 | 19 % | 0,66 | öffnen |
| B3 | Opus 5.5 nach Fix | nein | 0,300 | 0,3945 | 0,3974 | 1 | 151 | 12 | 21 % | 0,49 | vorbereiten |

¹ Aus M10-T7b (Abschnitt 9.2, Endstand) mit der **alten** Referenzsphäre `last_surface`; der RMS ist daher nur über die Spalte „RMS AP“ vergleichbar (gpt-oss B2: 0,092 im Trace, aber 39 Wellen durch die Austrittspupille, Falle F12). Ein Lauf der lokalen Modelle mit `image_infinite` steht aus.
„erster Schritt“: erste erfolgreiche Lauf-Aktion der Sitzung, `synth_segment` („öffnen“) oder `free_gaps`/`optimize` („vorbereiten“).

**Kosten** (Kostendatei `m10-t7c-arbeit/claude_cost.json` und die Versuche aus 11.3): zusammen **12,63 $** in 489 Anfragen: Opus 10,22 $ (davon 1,43 $ Versuche), Sonnet 2,31 $, Haiku 0,10 $ (dazu ein Rauchtest Haiku vorab, 0,002 $). Pro Lauf kostete Haiku 0,01–0,03 $, Sonnet 0,10–0,20 $ und Opus 0,40–0,70 $. Das Prompt-Caching greift: Cache-Lesen ist der größte Posten der Prompt-Tokens. Das Budget von 20 $ ist eingehalten.

### 11.2 Befunde an den Werkzeugen

| Nr. | Befund | Behebung | Test |
|---|---|---|---|
| E14 | **Behoben in M10-T7d (Abschnitt 12):** `free_gaps` gibt Plattenspalte nicht mehr frei. Befund: `free_gaps` vor dem ersten Abschnitt gibt auch den Spalt nach der Blende frei, in dem die Geisterplatten liegen; seine untere Schranke ist die Plattenlage (E3). Der Optimierer schiebt den Spalt auf diese Schranke (B1: 2 → 1 mm). Danach findet Operator A dort keinen Platz („Spalt nach Fläche 0 (1 mm) zu kurz für zwei Teilstücke“), und `synth` lehnt den Start ab (`start_rejected`, F_B 0,036–0,108 > F_max 0,0051) oder bricht mit Fehler ab. Kein späterer Schritt der Modelle löste das. Ohne Modell reproduziert: B1 mit `synth_segment` direkt oder nach `optimize` (Variablen der Datei) → Ziel; nach `free_gaps` + `optimize` → `start_rejected` | In diesem PR nur für `optimize` mit erweiterter Variablenauswahl: Plattenspalte bleiben fest. `free_gaps` selbst ist unverändert, weil die Regel V3 es benutzt (nach den Abschnitten) und eine Änderung die Referenz V3RS verschöbe; **Frage an den Maintainer** (11.5) | `test_optimize_widening_keeps_ghost_plate_gaps` |
| E15 | `abandon` beendete die Sitzung (`Session.run`), auch mit Restbudget und weiteren offenen Kandidaten (Sonnet B1 nach 6 von 12 Entscheidungen) | `abandon` verwirft nur den Kandidaten. `run` macht mit einem offenen Kandidaten weiter (`resume`; Standard ist der nächste offene Vorfahre, im Bewertungsskript der beste offene Kandidat nach `rank_key`) und meldet das in `last_errors`. Nur `finish` beendet. `abandon` nach einer verändernden Aktion derselben Antwort betrifft den neuen Kandidaten | `test_run_continues_after_abandon`, `test_run_resume_hook_and_end`, `test_resume_after_abandon_picks_best_open_candidate` |
| E16 | `optimize` mit `variables` ≠ `current` meldete `not_implemented` (18 Antworten von Opus und Sonnet vor den Korrekturen; nach einem Fehler laufen die übrigen Aktionen der Antwort nicht mehr) | Umgesetzt: `all_curvatures` (jede sphärische Fläche ohne Krümmungsvariable), `curvatures_and_gaps` (dazu die Luftspalte ohne Geisterplatten), `everything` (dazu die Glasdicken). Gläser bleiben fest; der Bericht nennt die neuen Variablen | `test_optimize_variable_selection`, `test_optimize_widening_keeps_ghost_plate_gaps` |
| E17 | Ein `synth_segment` mit `start_rejected` erzeugt einen Kandidaten ohne geöffnete Apertur und stand im Verlauf des Zustands wie ein Erfolg (`#3: synth_segment → [4]`); Sonnet wiederholte den Abschnitt elfmal | Der Verlauf nennt Ausgang und NA (`(step_too_small, NA 0.4475)`), bei `start_rejected` die Meldung des Controllers | `test_history_names_synth_outcome` |
| E18 | Die API lehnt die strict-Werkzeuge ab: 400 „Schemas contains too many optional parameters (185) … limit: 24“. `submit_decision` allein hat 133, die Lese-Werkzeuge zusammen 52 | Standard ohne `strict` (`ClaudeDecider(strict=False)`, `--strict` wählt es). Die Antwort wird weiter gegen das Aktionsschema geprüft, jetzt auch die Aufzählungen der obersten Ebene (z. B. `optimize.variables`) | `test_strict_is_opt_in_and_answers_still_validated`, `test_strict_off_by_default` |

Weitere Fehler der Modelle, ohne Werkzeugfehler: `insert_module front.shell_ca` ohne `glasses` oder mit einer Dicke, die die Schranke des Arbeitsabstands verletzt (bekannt aus 9.3), `set_bounds` mit `variable: "thickness"` (umgesetzt sind nur `wd` und `t[i]`), `set_constraint` ohne `weight`.

### 11.3 Zwischenstände und Prompt

Die Korrekturen kamen schrittweise. Jeder Zwischenstand wurde gerechnet, weil die Modelle empfindlich auf den System-Prompt reagierten:

| Stand | Änderung gegenüber dem vorigen | gerechnet |
|---|---|---|
| vor Fix | Stand f561043 mit `--no-strict` (vom Koordinator gerechnet) | Sonnet, Opus B1–B3 |
| Z1 | E15, E16 (ohne Ausnahme für Plattenspalte), E18; Prompt: Satz zu `abandon`/`finish` und eine Zeile zu `optimize variables` | Haiku B1–B3, Sonnet B1 |
| Z2 | E17; Prompt: Satz „history nennt den Ausgang …; bei start_rejected … den Start ändern (Struktur, Variablen, Nebenbedingungen)“ | Sonnet B1–B3, Opus B1 (zweimal) bis B3 |
| Z3 | Ausnahme für Plattenspalte in `optimize` (E14) | Opus B1–B3, Sonnet B1 |
| Z4 | Zeile zu `optimize variables` aus dem Prompt entfernt | Opus B1–B3, Sonnet B1 |
| nach Fix | Satz zu `start_rejected` aus dem Prompt entfernt (der Verlauf im Zustand bleibt) | Haiku, Sonnet, Opus B1–B3 |

| Bench. | Variante | Ziel | NA | RMS max [Wellen] | RMS AP | Elemente | Zeit [s] | Entsch. | abgelehnt | Kosten [$] | erster Schritt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | Haiku Z1 | **ja** | 0,600 | 0,0004 | 0,0004 | 4 | 119 | 12 | 0 % | 0,02 | öffnen |
| B1 | Sonnet Z1 | nein | 0,300 | 0,5107 | 0,5139 | 1 | 47 | 12 | 0 % | 0,18 | vorbereiten |
| B1 | Sonnet Z2 | nein | 0,300 | 0,5107 | 0,5139 | 1 | 47 | 12 | 14 % | 0,19 | vorbereiten |
| B1 | Opus Z2 | nein | 0,300 | 0,5107 | 0,5139 | 1 | 139 | 12 | 14 % | 0,53 | vorbereiten |
| B1 | Opus Z2 (Wdh.) | nein | 0,438 | 0,2596 | 0,2637 | 3 | 165 | 12 | 18 % | 0,49 | vorbereiten |
| B1 | Opus Z3 | nein | 0,300 | 0,5107 | 0,5139 | 1 | 141 | 12 | 10 % | 0,54 | vorbereiten |
| B1 | Sonnet Z3 | nein | 0,300 | 0,5107 | 0,5139 | 1 | 68 | 12 | 11 % | 0,20 | vorbereiten |
| B1 | Opus Z4 | nein | 0,300 | 0,5107 | 0,5139 | 1 | 133 | 11 | 20 % | 0,52 | vorbereiten |
| B1 | Sonnet Z4 | nein | 0,300 | 0,5107 | 0,5139 | 1 | 45 | 12 | 4 % | 0,18 | vorbereiten |
| B2 | Haiku Z1 | nein | 0,835 | 0,0865 | 0,0870 | 4 | 980 | 12 | 7 % | 0,02 | öffnen |
| B2 | Sonnet Z2 | nein | 0,375 | 0,0877 | 0,0883 | 2 | 209 | 12 | 14 % | 0,19 | vorbereiten |
| B2 | Opus Z2 | nein | 0,300 | 0,3177 | 0,3201 | 1 | 168 | 12 | 20 % | 0,52 | vorbereiten |
| B2 | Opus Z3 | nein | 0,300 | 0,3177 | 0,3201 | 1 | 146 | 12 | 18 % | 0,51 | vorbereiten |
| B2 | Opus Z4 | nein | 0,300 | 0,3177 | 0,3201 | 1 | 170 | 12 | 15 % | 0,59 | vorbereiten |
| B3 | Haiku Z1 | **ja** | 0,850 | 0,0149 | 0,0151 | 4 | 1041 | 9 | 0 % | 0,01 | öffnen |
| B3 | Sonnet Z2 | nein | 0,300 | 0,3945 | 0,3974 | 1 | 59 | 12 | 4 % | 0,18 | vorbereiten |
| B3 | Opus Z2 | nein | 0,300 | 0,3945 | 0,3974 | 1 | 156 | 12 | 17 % | 0,50 | vorbereiten |
| B3 | Opus Z3 | nein | 0,390 | 0,0854 | 0,0869 | 3 | 176 | 12 | 18 % | 0,56 | vorbereiten |
| B3 | Opus Z4 | nein | 0,383 | 0,0860 | 0,0867 | 2 | 379 | 12 | 22 % | 0,66 | vorbereiten |

**Was entscheidet.** Über alle Claude-Läufe gilt: Jeder Ziel-Treffer begann mit `synth_segment` („öffnen“). Fast jeder Lauf, der zuerst `free_gaps`/`optimize` ausführte („vorbereiten“), blieb bei NA 0,30 – das ist Befund E14. Ausnahmen sind Sonnet B3 nach Fix (0,51) und Opus Z3/Z4 B3 (≈ 0,39). Vor den Korrekturen begann Opus dreimal mit „öffnen“. Mit dem Prompt-Satz zu `start_rejected` (Z2–Z4) begann es in allen zehn Läufen mit „vorbereiten“, auch nachdem die `optimize`-Zeile entfernt war (Z4).

**Versuch** (nicht archiviert, Kosten in der Summe enthalten): Opus B1 auf Stand Z4 ohne den Satz zu `start_rejected` lief zweimal, beide Läufe erreichten das Ziel (RMS 0,0119 und 0,0130, 4 Elemente). Auf Stand Z3 ohne die `optimize`-Zeile lief es einmal und erreichte das Ziel (RMS 0,0033, 3 Elemente). Beide Sätze sind deshalb wieder aus dem Prompt entfernt (Stand „nach Fix“). Die Funktionen (Verlauf, Variablenauswahl) bleiben und sind in [entwurfssteuerung.md](../entwurfssteuerung.md) beschrieben.

Auch auf dem Endstand bleibt die Wahl des ersten Schritts zufällig: Opus öffnete auf B1 und B2 zuerst, auf B3 nicht; Haiku öffnete auf B1 in Z1, im Endlauf nicht. **Einzelläufe ergeben daher keine belastbare Rangfolge der Modelle.** Belastbar ist, dass E14 jeden Lauf entscheidet, der vorbereitet; öffnen allein reicht nicht (Haiku im Endlauf auf B2/B3).

### 11.4 Beobachtungen

- **Opus 5.5** (vor Fix): B1 Ziel mit RMS 0,0011 (Regel 0,0154). **B3 Ziel mit RMS 0,0145, viermal kleiner als die Regel (0,060)**, bei gleicher Elementzahl (4). B2 nur NA 0,41 (Regel 0,79). Der Weg: einen Abschnitt öffnen, dann die Randdicke als Nebenbedingung gewichten (`set_constraint min_edge_thickness`) und optimieren, wiederholt bis zum Budgetende; das senkt den RMS weit unter das Ziel. Nach Fix: B1 Ziel (RMS 0,024, 3 Elemente), B2 NA 0,46, B3 in E14.
- **Haiku 5.5** (Zwischenstand Z1): B1 Ziel mit RMS 0,0004, B3 Ziel mit RMS 0,0149 und **B2 NA 0,835, weiter als die Regel (0,79)**, für 0,01–0,02 $ pro Lauf. Auf B2 öffnete erst `set_constraint` + `free_gaps` (Spalt 7/8) + `optimize variables=curvatures_and_gaps` (E16), danach ein Abschnitt mit Strahlsuche (`beam_width` 3) die Apertur von 0,465 auf 0,835. Im Endlauf bereitete es auf B1 zuerst vor (E14, NA 0,30); auf B2 und B3 öffnete es zuerst, verbrauchte dann aber 9–10 Entscheidungen mit `insert_module front.shell_ca` ohne Glas oder mit verletzter Schranke des Arbeitsabstands (9.3) und kam nur auf NA 0,45 bzw. 0,375.
- **Sonnet 5.5** bereitet in allen Läufen zuerst vor und bleibt damit in E14 (B1 und B2 bei NA 0,30). Vor Fix beendete `abandon` die Sitzung, und `optimize` mit erweiterter Auswahl scheiterte. Nach Fix nutzt es das ganze Budget (12 Entscheidungen; `abandon` springt zum besten offenen Kandidaten), kommt aber nur auf B3 aus der Falle (NA 0,51).
- **Gegen die Regel:** Die Regel ist stabil, mit denselben Zahlen wie in M10-T7b (B1 0,0154, B2 0,79, B3 0,060). Die Modelle sind besser, wenn sie öffnen und dann nacharbeiten: RMS bis zu 40-mal kleiner auf B1 und 4-mal kleiner auf B3, Haiku auf B2 weiter offen. Sie sind viel schlechter, wenn sie vorbereiten. Die lokalen Modelle (M10-T7b, alte Sphäre) erreichen keinen Benchmark.

### 11.5 Folgerungen und Fragen an den Maintainer

1. **E14 zu beheben** ist die wirksamste Verbesserung für alle Entscheider. Das ist eine Frage an den Maintainer, weil `free_gaps` auch die Regel und damit die Referenz V3RS betrifft. Möglich wäre: `free_gaps` ohne Flächenliste lässt Spalte mit Geisterplatten fest (wie jetzt `optimize`), oder Operator A verschiebt bei einem zu kurzen Plattenspalt die Platte bzw. hebt die Schranke, oder der Zustand warnt vor dem ersten Abschnitt.
2. **Mehrere Läufe pro Modell** (z. B. drei) und deren Median, bevor Modelle verglichen werden; die Streuung des ersten Schritts ist groß.
3. Nach ADR-0012, Punkt 10: Große Modelle zeigen gegenüber der Regel einen Vorteil im RMS (Opus und Haiku auf B1/B3) und auf B2 in der NA (Haiku). Die Anbindung lohnt sich, wenn E14 behoben ist. Haiku 5.5 ist dabei mit Abstand am günstigsten.
4. **Offen in M10-T7:** V4 (Mensch, mindestens B2) und B4 (private Patentsysteme, nur lokal).

### 11.6 Reproduktion

```bash
KOSTEN=docs/ergebnisse/m10-t7c-arbeit/claude_cost.json
python tools/m10t7_bewertung.py --varianten V3RS
for m in claude-haiku-5-5 claude-sonnet-5-5 claude-opus-5-5; do
  python tools/m10t7_bewertung.py --benchmarks B1,B2,B3 --varianten LM --decider claude \
      --model $m --effort medium --max-usd 20 --cost-file $KOSTEN
done
python tools/m10t7_bewertung.py --archiv --tabelle
```

`--no-strict` ist seit diesem PR der Standard. Jede Sitzung lässt sich ohne API wiederholen (`python -m mikroptim.design replay docs/ergebnisse/m10-t7c/<B>_<Variante> <ziel>`).

## 12. M10-T7d: E14 behoben, V3RS neu, je drei Claude-Läufe

Freigabe Maintainer 2026-11-20: Befund E14 beheben (11.5, Punkt 1), die Referenz der Regel V3RS neu rechnen und je **drei Läufe** pro Modell und Benchmark (11.5, Punkt 2). Einstellungen wie in Abschnitt 11 (Endstand „nach Fix“): `image_infinite`, `--effort medium`, ohne `strict`, 12 Entscheidungen, Rückfall auf die Regel, System-Prompt unverändert. Ergebnisse, Protokolle und beste Systeme unter [`m10-t7d/`](m10-t7d/), Kürzel `LMRS_<modell>_w<N>` (Wiederholung N). Die drei Modelle liefen gleichzeitig, dazu V3RS und qwen3:14b; die Zeiten sind nur Orientierung. Code-Stand: `main` 7e8112f plus dieser PR.

### 12.1 Behebung von E14

**Regel (neu):** `free_gaps` gibt **keinen Spalt frei, in dem Geisterplatten liegen**, weder ohne noch mit ausdrücklicher Flächenliste; ebenso `optimize` mit `curvatures_and_gaps`/`everything` (wie schon in M10-T7c). Gibt die Aktion nichts Neues frei, entsteht kein Kandidat, und die Folgeaktion läuft auf demselben Kandidaten.

**Warum nicht eine Schranke oder ein Mindestabstand?** Operator A setzt den Meniskus in den Plattenspalt, und **beide Teilstücke erben die untere Schranke der Spaltdicke** (`src/optim/insertion.cpp`). Er braucht deshalb Spaltdicke T ≥ 2 · Schranke (+ Meniskusdicke). Jede Schranke m, die der Optimierer erreichen kann, lässt T = m zu, und dann gilt T < 2m – unabhängig davon, wie m gewählt wird (Plattenlage, doppelte Plattenlage, aktuelle Dicke). Eine Schranke unter der Plattenlage geht auch nicht: Die Platte läge dann hinter dem Spalt, und das System ist nicht ladbar. Ohne Änderung an Operator A ist „Plattenspalte bleiben fest“ deshalb die einzige Variante, die E14 sicher ausschließt. Die Alternative im C++-Kern (Operator A senkt bei zu kurzem Spalt die Schranke der Teilstücke) ist eine Frage an den Maintainer (12.5).

**Test:** `test_free_gaps_keeps_ghost_plate_gaps_e14` (`python/tests/test_design.py`) reproduziert E14 ohne Modell auf B1 (Start wie im Skript, `image_infinite`): `free_gaps` + `optimize` (20 Zyklen), dann `synth_segment` zur NA 0,6. **Vorher:** `free_gaps` gab die Spalte 0 und 2 (beide mit Platten) mit Schranke = Plattenlage frei, der Optimierer schob Spalt 0 auf 1 mm, `synth` meldete `start_rejected` („Spalt nach Fläche 0 (1 mm) zu kurz für zwei Teilstücke mit der unteren Schranke 1 mm der Dickenvariable“), NA 0,30. **Nachher:** Plattenspalte unverändert (2 mm), `synth` startet und meldet `goal_reached` (NA 0,6, 12 s). Angepasst: `test_optimize_widening_keeps_ghost_plate_gaps` (auch `free_gaps` mit Flächenliste gibt den Plattenspalt nicht mehr frei) und `test_free_gaps_skips_gap_thinner_than_plate` (kein neuer Kandidat). Doku: [entwurfssteuerung.md](../entwurfssteuerung.md), Tabelle der Aktionen (`free_gaps`, `optimize`).

**Weitere Änderungen:** `tools/m10t7_bewertung.py --wiederholung 1-3` (eigenes Kürzel `_w<N>`, vorhandene Ergebnisse bleiben) und `--zusammenfassung` (Ziel-Treffer x/n, beste und mediane NA/RMS, Elemente, Kosten pro Lauf); Test `test_repetitions_get_own_tags_and_summary`. `CostLedger` teilt die Kostendatei zwischen gleichzeitigen Prozessen (Sperrdatei, Neulesen vor jeder Anfrage, atomares Ersetzen; vorher überschrieb jeder Prozess die Summe der anderen mit seiner eigenen); Test `test_ledger_shared_between_processes`.

### 12.2 Ergebnisse

Ziel = NA ≥ Soll-NA, RMS ≤ 1/14 Welle, keine versagten Strahlen (Abschnitt 1). Median über die drei Läufe, in Klammern die Spanne. „Bester Lauf“ nach der Rangfolge der Bewertung (Ziel, dann NA, dann RMS).

| Bench. | Variante | Ziel | NA bester Lauf | NA Median (Spanne) | RMS bester Lauf | RMS Median (Spanne) | Elemente Median (Spanne) | Kosten/Lauf [$] Mittel (Spanne) |
|---|---|---|---|---|---|---|---|---|
| B1 | **Regel V3RS neu** | ja | 0,600 | – | 0,0154 | – | 4 | – |
| B1 | Regel V3RS alt (11.1) | ja | 0,600 | – | 0,0154 | – | 4 | – |
| B1 | Haiku 5.5 | **3/3** | 0,600 | 0,600 | 0,0110 | 0,0112 (0,0110–0,0133) | 4 (4–5) | 0,01 (0,01–0,01) |
| B1 | Sonnet 5.5 | **3/3** | 0,600 | 0,600 | 0,0118 | 0,0130 (0,0118–0,0154) | 4 (4–4) | 0,06 (0,03–0,10) |
| B1 | Opus 5.5 | **3/3** | 0,600 | 0,600 | 0,0080 | 0,0097 (0,0080–0,0135) | 4 (4–4) | 0,50 (0,47–0,54) |
| B2 | **Regel V3RS neu** | nein | 0,453 | – | 0,0795 | – | 4 | – |
| B2 | Regel V3RS alt (11.1) | nein | 0,790 | – | 0,0907 | – | 4 | – |
| B2 | Haiku 5.5 | 0/3 | 0,538 | 0,453 (0,453–0,538) | 0,0900 | 0,0802 (0,0797–0,0900) | 4 (4–5) | 0,02 (0,02–0,02) |
| B2 | Sonnet 5.5 | 0/3 | 0,453 | 0,453 (0,453–0,453) | 0,0796 | 0,0796 (0,0796–0,0806) | 4 (4–4) | 0,22 (0,22–0,22) |
| B2 | Opus 5.5 | 0/3 | 0,783 | **0,678** (0,410–0,783) | 0,0957 | 0,0957 (0,0779–0,0972) | 5 (3–6) | 0,70 (0,64–0,80) |
| B3 | **Regel V3RS neu** | nein | 0,600 | – | 0,0836 | – | 8 | – |
| B3 | Regel V3RS alt (11.1) | **ja** | 0,850 | – | 0,0595 | – | 4 | – |
| B3 | Haiku 5.5 | 0/3 | 0,490 | **0,445** (0,438–0,490) | 0,0862 | 0,0804 (0,0788–0,0862) | 4 (3–4) | 0,02 (0,02–0,02) |
| B3 | Sonnet 5.5 | 0/3 | 0,755 | 0,375 (0,375–0,755) | 0,0843 | 0,0801 (0,0795–0,0843) | 3 (3–4) | 0,20 (0,17–0,21) |
| B3 | Opus 5.5 | 0/3 | 0,533 | 0,375 (0,375–0,533) | 0,0867 | 0,0799 (0,0793–0,0867) | 3 (3–4) | 0,66 (0,61–0,71) |
| B1 | qwen3:14b (1 Lauf) | 0/1 | 0,300 | – | 20,4797 | – | 1 | – |
| B2 | qwen3:14b (1 Lauf) | 0/1 | 0,300 | – | 1,1838 | – | 1 | – |
| B3 | qwen3:14b (1 Lauf) | 0/1 | 0,300 | – | 2,5964 | – | 1 | – |

**qwen3:14b** (Ollama, ohne Thinking, `num_ctx` 40.960 = Modellmaximum, erstmals mit `image_infinite`): auf keinem Benchmark über NA 0,30. B1: nach einem `replace_glass` zehn gescheiterte `insert_module` (674 s); B2: ebenso, elf gescheiterte `insert_module`, kein Abschnitt (1023 s); B3: `abandon` des Startkandidaten in der ersten Antwort, damit kein offener Kandidat mehr und Ende der Sitzung (68 s). Wie in M10-T7b ist das lokale Instruktionsmodell schwächer als die Regel.

Einzelläufe (NA / RMS / Elemente / erster Schritt; „öffnen“ = `synth_segment`, sonst „vorbereiten“ mit der genannten Aktion):

| Bench. | Modell | w1 | w2 | w3 |
|---|---|---|---|---|
| B1 | Haiku | **0,600** / 0,0133 / 4 / öffnen | **0,600** / 0,0110 / 5 / `optimize` | **0,600** / 0,0112 / 4 / öffnen |
| B1 | Sonnet | **0,600** / 0,0118 / 4 / `free_gaps` | **0,600** / 0,0154 / 4 / `free_gaps` | **0,600** / 0,0130 / 4 / `free_gaps` |
| B1 | Opus | **0,600** / 0,0097 / 4 / öffnen | **0,600** / 0,0135 / 4 / öffnen | **0,600** / 0,0080 / 4 / öffnen |
| B2 | Haiku | 0,453 / 0,0802 / 4 / `free_gaps` | 0,453 / 0,0797 / 4 / öffnen | 0,538 / 0,0900 / 5 / `free_gaps` |
| B2 | Sonnet | 0,453 / 0,0796 / 4 / `free_gaps` | 0,453 / 0,0796 / 4 / `free_gaps` | 0,453 / 0,0806 / 4 / `free_gaps` |
| B2 | Opus | 0,783 / 0,0957 / 6 / öffnen | 0,678 / 0,0972 / 5 / `free_gaps` | 0,410 / 0,0779 / 3 / `optimize` |
| B3 | Haiku | 0,490 / 0,0862 / 4 / öffnen | 0,438 / 0,0804 / 3 / öffnen | 0,445 / 0,0788 / 4 / `free_gaps` |
| B3 | Sonnet | 0,375 / 0,0795 / 3 / `free_gaps` | 0,375 / 0,0801 / 3 / `free_gaps` | 0,755 / 0,0843 / 4 / `free_gaps` |
| B3 | Opus | 0,375 / 0,0793 / 3 / `set_constraint` | 0,533 / 0,0867 / 4 / öffnen | 0,375 / 0,0799 / 3 / öffnen |

**Kosten dieser Runde** (Kostendatei `m10-t7d-arbeit/claude_cost.json`, Obergrenze 100 $): **7,17 $** in 354 Anfragen – Opus 5,58 $ (149 Anfragen), Sonnet 1,44 $ (85), Haiku 0,15 $ (120). **Kumuliert mit M10-T7c: 19,80 $** von 200 $ Gesamtfreigabe. Pro Lauf: Haiku 0,005–0,024 $, Sonnet 0,03–0,22 $, Opus 0,47–0,80 $.

### 12.3 Einordnung von E14

- **E14 tritt nicht mehr auf.** Von 27 Claude-Läufen begannen 16 mit „vorbereiten“ (`free_gaps`, `optimize` oder `set_constraint` vor dem ersten Abschnitt); in M10-T7c endete das fast immer bei NA 0,30. Jetzt endet kein Lauf bei 0,30; der kleinste Endwert ist 0,375 (B3, ebenso bei Läufen mit „öffnen“).
- **B1 ist für alle Modelle gelöst:** 9/9 Ziel-Treffer (M10-T7c Endstand: 1 von 3). Sonnet bereitete in allen drei Läufen vor und erreichte das Ziel jedes Mal nach 3–5 Entscheidungen (M10-T7c: in sechs Läufen NA 0,30); Haiku w2 ebenso.
- **Der erste Schritt entscheidet nicht mehr:** Auf B2 und B3 liegen Läufe mit „öffnen“ und „vorbereiten“ in denselben Bereichen (Opus B2 0,783 öffnen, 0,678 vorbereiten; Sonnet B3 0,755 vorbereiten).
- **Zwei `start_rejected` anderer Art** (Opus B2 w1 und w2, jeweils mitten in der Sitzung nach einer Strukturänderung): F_B knapp über F_max (0,0057 bzw. 0,0071 > 0,0051), „keine Geisterplatte aktiv (kein Bedarf)“, also kein Einfügen versucht. Das ist nicht E14, sondern die Annahmeschwelle des Startpunkts (neu: **E19, offen**; die Sitzungen liefen weiter).
- Fehler der Modelle ohne Werkzeugfehler (178 Aktionen mit Fehler in 27 Läufen), wie in 9.3 und 11.2: 81 `insert_module`, deren Dicke die Schranke des Arbeitsabstands verletzt, 44 `insert_module` ohne `glasses`, 23 `set_bounds` mit anderem Namen als `wd`/`t[i]`, 22 `set_constraint` auf nicht vorhandene Nebenbedingungen oder ohne `weight`. Die Häufung beim Einfügen kostet vor allem auf B2/B3 viele der 12 Entscheidungen.

### 12.4 V3RS neu gegen alt

Die Regel benutzt `free_gaps` nach jedem Abschnitt und als erste Stufe bei Stillstand. Bisher gab das auch die Plattenspalte frei, jetzt bleiben sie fest. Das ändert den Pfad der Regel, nicht B1:

- **B1** unverändert (Ziel, RMS 0,0154, 4 Elemente, 2 Entscheidungen; die Regel öffnet zuerst).
- **B2: 0,790 → 0,453.** Der zweite Abschnitt nach `free_gaps` + `optimize` öffnete bisher von 0,452 auf 0,465 (+0,013), jetzt nicht. Weil die Regel einen Abschnitt nur nach einem Gewinn ≥ 0,01 wiederholt, wählte sie bisher „wiederholen“ (danach 0,765 und mit dem Wörterbuch 0,790), jetzt das Wörterbuch und den Null-Meniskus, beide ohne Wirkung, und endet bei 0,453.
- **B3: Ziel (0,85) → 0,600 mit 8 Elementen.** Hier öffnete der zweite Abschnitt jetzt *weiter* (0,445 statt 0,390). Damit war die aplanatische Schale anwendbar, und die Regel ging einen anderen Weg durch die Stufen (Schale, Wörterbuch, Null-Meniskus) bis 0,600. Bisher führte das Wörterbuch nach 0,390 direkt zum Ziel.

Die Regel ist also **pfadabhängig**: Kleine Unterschiede eines Abschnitts (±0,01 NA) wählen andere Stufen und ändern das Ergebnis um 0,25–0,34 NA. Die alten Werte von B2/B3 waren ein günstiger Pfad, kein systematischer Vorteil des Freigebens der Plattenspalte (auf B3 öffnete der Abschnitt ohne sie sogar weiter). Auch für die Regel lässt sich die Wirkung einer Änderung nur mit mehreren Läufen belastbar messen. **Referenz der Regel ab hier: V3RS neu** (B1 Ziel, B2 0,453, B3 0,600); die alten Werte stehen daneben.

### 12.5 Rangfolge und Folgerungen

- **B1:** alle drei Modelle 3/3; RMS-Median Opus 0,0097 < Haiku 0,0112 < Sonnet 0,0130 < Regel 0,0154. Die Spannen überlappen (Opus 0,0080–0,0135, Haiku 0,0110–0,0133, Sonnet 0,0118–0,0154).
- **B2:** kein Ziel. **Opus deutlich vorn** (Median 0,678, bester Lauf 0,783 ≈ alte Regel 0,790, mit 5–6 Elementen). Haiku und Sonnet liegen im Median auf der neuen Regel (0,453), Haiku einmal 0,538.
- **B3:** kein Ziel. Median Haiku 0,445 > Opus 0,375 = Sonnet 0,375, alle unter der neuen Regel (0,600). Einzelne Läufe reichen weiter (Sonnet 0,755, Opus 0,533); die Streuung ist groß.
- **Insgesamt:** Opus > Haiku ≈ Sonnet. Gegen die neue Regel sind die Modelle auf B1 besser (RMS), auf B2 ist Opus besser, auf B3 sind alle im Median schlechter. Haiku kostet etwa 1/30 von Opus. Kein Modell erreicht B2 oder B3; B3 erreichten bisher nur die Regel auf ihrem alten Pfad und Opus/Haiku in Einzelläufen von M10-T7c.

**Fragen an den Maintainer:**

1. **Operator A:** Soll Operator A bei zu kurzem Plattenspalt die untere Schranke der Teilstücke auf (T − Dicke)/2 senken (C++, `src/optim/insertion.cpp`)? Dann könnte `free_gaps` die Plattenspalte wieder mit Schranke = Plattenlage freigeben, ohne E14. Sinnvoll erst nach M6-T9 (gleicher Kern).
2. **E19:** `synth` lehnt einen Start knapp über F_max ab, wenn keine Geisterplatte Bedarf meldet. Ein kurzes `optimize` vor dem Abschnitt (in `synth_segment`) oder eine weichere Annahmeschwelle nach Strukturänderungen?
3. **Bewertung der Regel:** V3RS mit mehreren leicht gestörten Starts rechnen, damit Regeländerungen nicht an einem einzigen Pfad gemessen werden?
4. **Modellfehler beim Einfügen** (81 + 44 Antworten): Soll `insert_module front.shell_ca` die Dicke selbst an die Schranke des Arbeitsabstands anpassen und ohne `glasses` das hochbrechendste Katalogglas nehmen (wie die Regel)?

### 12.6 Reproduktion

```bash
python tools/m10t7_bewertung.py --out docs/ergebnisse/m10-t7d --varianten V3RS
KOSTEN=docs/ergebnisse/m10-t7d-arbeit/claude_cost.json
for m in claude-haiku-5-5 claude-sonnet-5-5 claude-opus-5-5; do   # liefen gleichzeitig
  python tools/m10t7_bewertung.py --out docs/ergebnisse/m10-t7d --benchmarks B1,B2,B3 --varianten LM \
      --decider claude --model $m --effort medium --max-usd 100 --cost-file $KOSTEN --wiederholung 1-3
done
python tools/m10t7_bewertung.py --out docs/ergebnisse/m10-t7d --varianten LM --decider ollama \
    --model qwen3:14b --no-think
python tools/m10t7_bewertung.py --out docs/ergebnisse/m10-t7d --archiv --tabelle --zusammenfassung
```

Jede Sitzung lässt sich ohne API wiederholen (`python -m mikroptim.design replay docs/ergebnisse/m10-t7d/<B>_<Variante> <ziel>`).

## 13. M10-T7e: Werkzeugverbesserungen und Nachmessung

Freigabe Maintainer 2026-11-20: die vier Fragen aus 12.5 umsetzen und kurz nachmessen. Einstellungen wie in Abschnitt 12 (`image_infinite`, `--effort medium`, ohne `strict`, 12 Entscheidungen, Rückfall auf die Regel, System-Prompt unverändert). Ergebnisse, Protokolle und beste Systeme unter [`m10-t7e/`](m10-t7e/), Kürzel `LMRS_<modell>_w<N>` und `V3RS_s<k>` (gestörter Start k). Bis zu acht Läufe liefen gleichzeitig auf der CPU; die Zeiten sind nur Orientierung. Code-Stand: `main` ef4829c plus dieser PR.

### 13.1 Umsetzung

| Nr. | Frage (12.5) | Umsetzung | Test |
|---|---|---|---|
| 1 | Operator A im kurzen Plattenspalt | Ist der Spalt T kürzer als 2 × untere Schranke + Dicke, setzt Operator A den Meniskus in die Mitte und gibt beiden Teilstücken eigene Dickeneinträge mit der Schranke max(min((T − d)/2, t_k), Plattenlage im Teilstück) – genau (T − d)/2, wenn der Optimierer den Spalt auf die Plattenlage geschoben hat; Warnung im Protokoll. Option `shrink_short_gap` (C++), `synth --shrink-short-gaps`; ohne Option bitgleich. `mikroptim.design` setzt sie immer, und `free_gaps` gibt Plattenspalte wieder frei (Schranke = Plattenlage, `include_ghost_plate_gaps: false` = Sperre aus M10-T7d) | `InsertionInit.MeniscusShortPlateGapLowersPieceBounds` (Merit-Invarianz ≤ 10⁻¹² relativ, Schranken, Platten), `test_free_gaps_releases_plate_gaps_without_e14` (B1: `free_gaps` + `optimize` + `synth_segment` → `goal_reached` mit Warnung; ohne Option kein Ziel), `test_optimize_widening_and_ghost_plate_gaps`, `test_free_gaps_skips_gap_thinner_than_plate` |
| 2 | E19 | `synth_segment` mit `start_rejected` und „kein Bedarf“: erst `optimize` (`start_retry_cycles`, Standard 20), dann zweiter Versuch; zählt nur, wenn der Start angenommen wird. Bericht `start_retry`, Verlauf | `test_synth_segment_retries_after_start_rejected_e19` |
| 3 | V3RS mit gestörten Starts | `tools/m10t7_bewertung.py --gestoerte-starts N --stoer-sigma σ --stoer-seed S`: Radien r(1 + σξ), Dicken und Arbeitsabstand t(1 + σξ) der Variablen der Datei, in den Schranken, nicht unter Plattenlagen, deterministisch; Zusammenfassung als Gruppe `<tag>_s*` | `test_perturbed_starts_within_bounds_and_summary` |
| 4 | Modellfehler beim Einfügen | `insert_module` ohne `glasses`: erstes Katalogglas der Glasregel des Moduls, sonst (Paarregel, keine Regel) das hochbrechendste Katalogglas. Dicke unter ihrer Schranke (Arbeitsabstand nach der Schale): Einfügen ohne untere Dickenschranken, danach auf die Schranke anheben. Beides als `warnings` in Bericht und Verlauf | `test_insert_module_default_glass`, `test_insert_module_raises_thickness_to_bound` |

**Während der Nachmessung gefunden und behoben:** (a) Die Dicken-Homotopie von `insert_module` ließ eine Geisterplatte im verkürzten Luftspalt stehen, `optimize` lehnte das System ab („ghost_plates[k].distance … zwischen 0 und der Dicke des Luftspalts“). Seit Plattenspalte wieder frei sind, trat das in der Regel auf zwei gestörten Starts von B2 auf (Null-Meniskus-Stufe). Jetzt rückt die Platte mit wie bei `set_stage` in `synth`; die beiden Läufe sind neu gerechnet (vorher 0,448 und 0,795, nachher 0,448 und 0,805). Test `test_thickness_homotopy_keeps_ghost_plates_in_gap`. (b) Die Bewertung brach mit `TypeError` ab, wenn die Gegenprobe (`eval`) für ein Tupel keinen RMS lieferte (Haiku B2 w3); RMS AP ist dann leer. Test `test_assess_tolerates_eval_tuple_without_value`. Haiku B2 w3 ist danach neu gerechnet (ein früherer Versuch kollidierte außerdem mit einem parallelen Lauf derselben Wiederholung und ist verworfen).

### 13.2 Ergebnisse

Ziel = NA ≥ Soll-NA, RMS ≤ 1/14 Welle, keine versagten Strahlen. Median über die Läufe, in Klammern die Spanne. „vorher“ = Abschnitt 12.2 (#112).

| Bench. | Variante | Ziel vorher | Ziel jetzt | NA Median vorher | NA Median jetzt (Spanne) | RMS Median jetzt (Spanne) | Elemente jetzt | Kosten/Lauf [$] |
|---|---|---|---|---|---|---|---|---|
| B1 | Regel V3RS | ja | ja | 0,600 | 0,600 | 0,0154 | 4 | – |
| B1 | Regel V3RS, 5 gestörte Starts | – | **5/5** | – | 0,600 (0,600–0,600) | 0,0155 (0,0152–0,0168) | 4 | – |
| B1 | Haiku 5.5 | 3/3 | **3/3** | 0,600 | 0,600 (0,600–0,600) | 0,0110 (0,0061–0,0138) | 4 (3–5) | 0,01 |
| B2 | Regel V3RS | nein | nein | 0,453 | **0,790** | 0,0907 | 4 | – |
| B2 | Regel V3RS, 5 gestörte Starts | – | 0/5 | – | 0,458 (0,448–0,828) | 0,0808 | 5 (4–5) | – |
| B2 | Haiku 5.5 | 0/3 | **1/3** | 0,453 | **0,798** (0,353–0,950) | 0,0370 (0,0356–0,0441) | 6 (5–7) | 0,02 |
| B2 | Opus 5.5 | 0/3 | 0/3 | 0,678 | **0,945** (0,628–0,950) | 0,0830 (0,0755–0,0894) | 6 | 0,64 |
| B3 | Regel V3RS | nein (0,600) | **ja** | 0,600 | 0,850 | 0,0595 | 4 | – |
| B3 | Regel V3RS, 5 gestörte Starts | – | 0/5 | – | 0,410 (0,405–0,668) | 0,0822 | 7 (5–7) | – |
| B3 | Haiku 5.5 | 0/3 | **1/3** | 0,445 | **0,850** (0,850–0,850) | 0,0764 (0,0504–0,0794) | 7 (7–10) | 0,02 |
| B3 | Opus 5.5 | 0/3 | **2/3** | 0,375 | **0,850** (0,355–0,850) | 0,0445 (0,0439–0,0456) | 6 (3–6) | 0,64 |

Einzelläufe (NA / RMS / Elemente / erster Schritt):

| Bench. | Variante | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| B1 | Haiku w | **0,600** / 0,0138 / 3 / öffnen | **0,600** / 0,0061 / 4 / öffnen | **0,600** / 0,0110 / 5 / `optimize` | | |
| B2 | Haiku w | 0,798 / 0,0356 / 5 / `free_gaps` | **0,950** / 0,0370 / 7 / `free_gaps` | 0,353 / 0,0441 / 6 / öffnen | | |
| B2 | Opus w | 0,950 / 0,0755 / 6 / öffnen | 0,628 / 0,0894 / 6 / öffnen | 0,945 / 0,0830 / 6 / öffnen | | |
| B3 | Haiku w | **0,850** / 0,0504 / 7 / `insert_module` | 0,850 / 0,0794 / 7 / öffnen | 0,850 / 0,0764 / 10 / öffnen | | |
| B3 | Opus w | 0,355 / 0,0439 / 6 / `free_gaps` | **0,850** / 0,0445 / 3 / `set_constraint` | **0,850** / 0,0456 / 6 / öffnen | | |
| B2 | V3RS s | 0,448 / 0,0799 / 5 | 0,453 / 0,0808 / 4 | 0,458 / 0,0789 / 4 | 0,805 / 0,0908 / 5 | 0,828 / 0,0918 / 5 |
| B3 | V3RS s | 0,405 / 0,0822 / 5 | 0,665 / 0,0910 / 7 | 0,410 / 0,0727 / 7 | 0,668 / 0,0917 / 6 | 0,410 / 0,0734 / 7 |

B1 mit gestörten Starts: alle fünf am Ziel, RMS 0,0152–0,0168, 4 Elemente.

**Kosten dieser Runde** (Kostendatei `m10-t7e-arbeit/claude_cost.json`, Obergrenze 50 $): **4,02 $** in 223 Anfragen – Opus 3,84 $ (93), Haiku 0,18 $ (130), einschließlich der verworfenen Versuche von Haiku B2 w3. **Kumuliert: 23,82 $** von 200 $ Gesamtfreigabe.

### 13.3 Wirkung der Änderungen

- **Fehler der Modelle beim Einfügen fast verschwunden.** In den 15 vergleichbaren Claude-Läufen (Haiku B1–B3, Opus B2–B3) scheiterten vorher 108 von 272 Aktionen (51 `insert_module` unter der Schranke, 39 ohne `glasses`), jetzt 11 von 400 (6 `set_constraint`, 3 Schalen mit negativem Arbeitsabstand, 1 leerer Glasfilter, 1 `start_from_skeleton`). Statt der Fehler stehen 29 angehobene Dicken und 13 Standardgläser als Warnungen in den Berichten. Die Modelle nutzen ihre 12 Entscheidungen damit für den Entwurf: auf B2/B3 entstehen Systeme mit 6–10 Elementen (Median 6–7), vorher 3–6 (Median 3–5).
- **Kurzer Plattenspalt** (Frage 1): 12 Warnungen in 7 Claude-Läufen, also hätte Operator A dort vorher abgelehnt (E14 in anderer Form). In der Regel trat er nicht auf (sie öffnet vor `free_gaps`).
- **E19** (Frage 2): Der zweite Versuch lief achtmal in vier Läufen; einmal nahm `synth` den Start danach an (Haiku B3 w2), siebenmal nicht (fünfmal in Haiku B2 w3, der bei NA 0,353 blieb). 20 LM-Zyklen auf den Variablen der Datei reichen meist nicht; der Startpunkt liegt dann nicht knapp, sondern strukturell über F_max. Die Option bleibt (sie kostet nur Rechenzeit und ist abschaltbar).
- **Regel V3RS**: Mit wieder freien Plattenspalten nimmt die Regel ihren alten Pfad (B2 0,790, B3 Ziel – dieselben Zahlen wie vor M10-T7d, Abschnitt 11.1). **Mit gestörten Starts (σ = 2 %) zerfällt das:** B1 bleibt 5/5, aber B2 0/5 mit Median 0,458 (zwei Starts bei 0,81–0,83, drei bei 0,45), B3 0/5 mit Median 0,410 (keiner über 0,67). Der ungestörte Start ist für B2/B3 ein günstiger Pfad, nicht typisch. Das bestätigt 12.4: die Regel ist stark pfadabhängig, und ihre Referenz sollte der Median über gestörte Starts sein, nicht ein Einzellauf.

### 13.4 Rangfolge und Folgerungen

- **B1:** Regel 5/5 (gestört) und Haiku 3/3; RMS-Median Haiku 0,0110 gegen Regel 0,0155.
- **B2:** Erstmals ein Ziel-Treffer auf B2: **Haiku w2 mit NA 0,950 und RMS 0,0370** (7 Elemente). Opus erreicht in zwei von drei Läufen NA 0,945–0,950, aber mit RMS 0,076–0,083 knapp über 1/14 Welle. Median NA: Opus 0,945 > Haiku 0,798 > Regel 0,458 (gestört).
- **B3:** Opus 2/3, Haiku 1/3 am Ziel (vorher 0/9 Modellläufe in Abschnitt 12), alle sechs Läufe bis auf einen bei NA 0,850. Regel gestört 0/5.
- **Insgesamt:** Gegen die Regel mit gestörten Starts sind beide Modelle auf B2 und B3 deutlich besser (Median NA 0,80–0,95 gegen 0,41–0,46). Die größte Einzelwirkung hatten nach den Zahlen die robusten Einfügeoperationen (Frage 4) und die freien Plattenspalte (Frage 1); E19 half wenig. Haiku kostet weiter etwa 1/30 von Opus.

**Offen und Fragen an den Maintainer:**

1. **Referenz der Regel:** künftig V3RS als Median über fünf gestörte Starts (σ = 2 %) statt des Einzellaufs? Der Einzellauf liegt auf B2/B3 am oberen Rand der Streuung.
2. **E19:** Der zweite Versuch nach 20 Zyklen wirkt selten. Mehr Zyklen, erweiterte Variablen (`curvatures_and_gaps`) im zweiten Versuch, oder eine weichere Annahmeschwelle nach Strukturänderungen?
3. Die drei Schalen mit negativem Arbeitsabstand nach dem Nachfokussieren (`insert_aplanatic_shell`) sind kein Schrankenfall; dafür müsste `insert_module` die Dicke der Schale verkleinern. Gewünscht?
4. Weiter offen in M10-T7: V4 (Mensch, mindestens B2) und B4 (private Patentsysteme, nur lokal).

### 13.5 Reproduktion

```bash
python tools/m10t7_bewertung.py --out docs/ergebnisse/m10-t7e --varianten V3RS
python tools/m10t7_bewertung.py --out docs/ergebnisse/m10-t7e --varianten V3RS \
    --gestoerte-starts 5 --stoer-sigma 0.02
KOSTEN=docs/ergebnisse/m10-t7e-arbeit/claude_cost.json
python tools/m10t7_bewertung.py --out docs/ergebnisse/m10-t7e --benchmarks B1,B2,B3 --varianten LM \
    --decider claude --model claude-haiku-5-5 --effort medium --max-usd 50 --cost-file $KOSTEN \
    --wiederholung 1-3
python tools/m10t7_bewertung.py --out docs/ergebnisse/m10-t7e --benchmarks B2,B3 --varianten LM \
    --decider claude --model claude-opus-5-5 --effort medium --max-usd 50 --cost-file $KOSTEN \
    --wiederholung 1-3
python tools/m10t7_bewertung.py --out docs/ergebnisse/m10-t7e --archiv --zusammenfassung
```

Jede Sitzung lässt sich ohne API wiederholen (`python -m mikroptim.design replay docs/ergebnisse/m10-t7e/<B>_<Variante> <ziel>`). Eine Wiederholung nicht in zwei Prozessen gleichzeitig rechnen: Das Skript legt die Sitzung im Arbeitsverzeichnis neu an und überschreibt die des anderen Prozesses.

## 14. M6-T11b: Referenz der Regel als Median gestörter Starts

Entscheidung Maintainer 2026-11-22 (Frage 1 aus 13.4): Die Referenz der Regel ist ab hier der **Median von V3RS über fünf gestörte Starts** (σ = 2 %, `--stoer-seed 0`), nicht der Einzellauf. Gleichzeitig ruht **B4** (die Patentsysteme sind Platzhalter, keine fertig entwickelten Systeme, als Referenz unbrauchbar; ebenso der Patentvergleich aus M6-T6). Ergebnisse, Protokolle und beste Systeme unter [`m6-t11b/`](m6-t11b/). Nur die Regel, keine Claude- oder Ollama-Aufrufe. 13 Prozesse liefen gleichzeitig auf der CPU (B1 in einem, B2 und B3 je ein Prozess pro Start); die Zeiten sind nur Orientierung. Code-Stand: `main` a79fa49 plus dieser PR (auch mit den übrigen Punkten von M6-T11b: Standard `--split-spectrum`/`--split-member-start`, `--insertion-improve` in der Startphase, Verkleinern der Schale in `insert_module`).

### 14.1 Umsetzung im Skript

`tools/m10t7_bewertung.py` rechnet ohne `--gestoerte-starts` für die Regel-Varianten (V3, V3AL) den ungestörten Einzellauf **und** fünf gestörte Starts (`RULE_REFERENCE_STARTS = 5`, Kürzel `V3RS_s1` … `_s5`); andere Varianten nur den Einzellauf. `--regel-einzellauf` rechnet nur den Einzellauf (Verhalten bis M10-T7e), `--gestoerte-starts N` gilt wie bisher für alle Varianten (0: nur Einzelläufe), `--nur-starts 0,3-5` verteilt die Starts auf mehrere Prozesse (0 = Einzellauf). `--tabelle` beginnt mit einer Tabelle der Referenz (Ziel x/5, Median und Spanne der NA, Median von RMS und Elementzahl, Einzellauf daneben); `--zusammenfassung` führt die gestörten Starts der Regel als Zeile „V3RS Referenz (Median 5 gestörte Starts)“ vor dem „V3RS Einzellauf“. Test `test_rule_reference_is_median_of_perturbed_starts` (Ablauf und Tabellen mit ersetzter Rechnung, Werte wie B2 in 13.2).

### 14.2 Ergebnisse

| Bench. | Referenz V3RS (5 gestörte Starts): Ziel | NA Median (Spanne) | RMS Median | Elemente Median | Einzellauf: Ziel | NA | RMS |
|---|---|---|---|---|---|---|---|
| B1 | **5/5** | 0,600 (0,600–0,600) | 0,0155 | 4 | ja | 0,600 | 0,0154 |
| B2 | 0/5 | **0,458** (0,448–0,828) | 0,0808 | 5 | nein | 0,790 | 0,0907 |
| B3 | 0/5 | **0,410** (0,405–0,668) | 0,0822 | 7 | ja | 0,850 | 0,0595 |

Einzelläufe (NA / RMS / Elemente / Entscheidungen / Zeit):

| Bench. | Einzellauf | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|
| B1 | **0,600** / 0,0154 / 4 / 2 / 16 s | **0,600** / 0,0157 / 4 / 2 | **0,600** / 0,0152 / 4 / 2 | **0,600** / 0,0154 / 4 / 2 | **0,600** / 0,0155 / 4 / 2 | **0,600** / 0,0168 / 4 / 2 |
| B2 | 0,790 / 0,0907 / 4 / 11 / 1.382 s | 0,448 / 0,0799 / 5 / 7 | 0,453 / 0,0808 / 4 / 5 | 0,458 / 0,0789 / 4 / 4 | 0,805 / 0,0908 / 5 / 9 | 0,828 / 0,0918 / 5 / 9 |
| B3 | **0,850** / 0,0595 / 4 / 5 / 1.153 s | 0,405 / 0,0822 / 5 / 12 | 0,665 / 0,0910 / 7 / 12 | 0,410 / 0,0727 / 7 / 12 | 0,668 / 0,0917 / 6 / 11 | 0,410 / 0,0734 / 7 / 12 |

Alle 18 Läufe ergeben dieselben NA, RMS und Elementzahlen wie in M10-T7e (13.2): Die Regel benutzt weder Farbregel noch Glasvorauswahl noch `--insertion-improve`, und ihre Schale ist so bemessen, dass der Arbeitsabstand positiv bleibt (`RuleDecider._shell`, Zielwert 1,05 × Schranke); das Verkleinern der Schale in `insert_module` griff in keinem Lauf. Die Zahlen von M10-T7e sind damit bestätigt und mit dem neuen Stand reproduzierbar.

### 14.3 Einordnung

- **Neue Referenz der Regel: B1 5/5 (RMS-Median 0,0155), B2 NA 0,458, B3 NA 0,410** (jeweils Median, kein Ziel). Der Einzellauf (B2 0,790, B3 Ziel 0,850) liegt am oberen Rand bzw. außerhalb der Streuung: Auf B2 enden drei Starts bei 0,45–0,46 und zwei bei 0,81–0,83, auf B3 drei bei 0,41 und zwei bei 0,67.
- **Gegen die Modelle (13.2, Median NA):** B2 Opus 0,945 und Haiku 0,798 gegen Regel 0,458; B3 Opus und Haiku 0,850 gegen 0,410. Beide Modelle sind auf B2/B3 klar besser als die Referenz der Regel, auf B1 gleich im Ziel und besser im RMS (Haiku 0,0110 gegen 0,0155). Gegen den Einzellauf der Regel sah das auf B3 umgekehrt aus; der Median ist die fairere Messlatte, weil auch die Modelle mehrfach gerechnet wurden.
- Der Erfolg der Regel hängt auf B2/B3 an einem Pfad durch die Stufen (12.4): Starts, die im zweiten Abschnitt die Wiederhol-Schwelle (+0,01 NA) knapp verfehlen, landen bei 0,41–0,46.

### 14.4 Reproduktion

```bash
python tools/m10t7_bewertung.py --out docs/ergebnisse/m6-t11b --varianten V3RS   # Einzellauf + 5 gestörte Starts
# verteilt auf Prozesse (eigenes Arbeitsverzeichnis je Prozess):
python tools/m10t7_bewertung.py --out docs/ergebnisse/m6-t11b --work <arbeit_B2_3> --benchmarks B2 \
    --varianten V3RS --nur-starts 3
python tools/m10t7_bewertung.py --out docs/ergebnisse/m6-t11b --archiv --tabelle --zusammenfassung
```
