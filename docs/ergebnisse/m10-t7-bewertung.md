# M10-T7: Bewertung der Entwurfssteuerung

Bericht zur Aufgabe M10-T7 ([Meilensteinplan](../plan/meilensteine.md#m10-t7--bewertung), [entwurfssteuerung.md, Abschnitt 8](../entwurfssteuerung.md#8-bewertung)). Verglichen werden die Synthese ohne Entscheider (V0, V1), das M6-T6-Skript (V2), die regelbasierte Strategie ohne Modell (V3) und zwei lokale Sprachmodelle über Ollama auf denselben Werkzeugen (`mikroptim.design`). Läufe mit der Claude-API (M10-T6c) und mit einem Menschen (V4) stehen aus (Abschnitt 6).

**Stand: Teilergebnis.** V0, V1 und V3 auf B1–B3 sowie zwei Ollama-Modelle (Instruktion und Thinking) auf B1–B3 sind gerechnet; V2 ist aus dem Bericht M6-T6 übernommen. **Ausstehend:** Claude-Läufe (kein `ANTHROPIC_API_KEY` in der Umgebung, `ant` nicht installiert), V4 (Mensch), B4 (private Patentsysteme).

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

**Offen (nicht in dieser Aufgabe):**

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

Unter PowerShell die Listen in Anführungszeichen setzen (`--benchmarks 'B1,B2,B3'`). Erwartete Kosten (Schätzung aus den Ollama-Läufen: 50.000–500.000 Prompt-Tokens und 20.000–40.000 Ausgabe-Tokens pro Lauf, mit Caching meist Cache-Lesen): Sonnet 5.5 etwa 0,3–1,5 $ pro Lauf, Opus 5.5 etwa das Doppelte; alle sechs Läufe zusammen voraussichtlich unter 10 $. Die Kosten pro Lauf stehen danach in der Tabelle (Spalte „Kosten“) und in `claude_cost.json`. Bricht ein Lauf an der API ab, steht die Meldung im Ergebnis unter `aborted` (Ergebnisdatei löschen und neu starten). Lehnt die API die abgeleiteten strict-Schemas ab (Fehler 400 mit Hinweis auf das Schema), mit `--no-strict` wiederholen; die Antworten werden auch dann gegen das Aktionsschema geprüft.

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

Vorhandene Ergebnisse werden übersprungen; zum Neurechnen die Datei `docs/ergebnisse/m10-t7/<B>_<Variante>.json` löschen. Das Arbeitsverzeichnis (`--work`, Standard `docs/ergebnisse/m10-t7-arbeit`) gehört nicht ins Repository.
