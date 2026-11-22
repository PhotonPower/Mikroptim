# Entwurfssteuerung: Mensch oder Sprachmodell als Entscheider (Entwurf, M10)

- **Status:** Entwurf; Grundsatzentscheidung in [ADR-0012](adr/0012-entwurfswerkzeuge-und-entscheider.md)
- **Verwandt:** [optimierung-erweiterungen.md](optimierung-erweiterungen.md) (vorgeschlagene Erweiterungen des Optimierers, M11; mehrere davon liefern Diagnosen und Aktionen für diesen Entwurf)
- **Grundlage:** Befunde aus [m6-t6-zielspezifikation.md](ergebnisse/m6-t6-zielspezifikation.md); Y. Zhang, *Systematic Design of Microscope Objectives*, Dissertation FSU Jena 2019 (im Folgenden „Zhang“, mit Kapitel- bzw. Tabellennummer)

## 1. Ziel und Abgrenzung

Mikroptim bekommt eine Schicht von **Entwurfswerkzeugen**: Diagnosen, die einen Kandidaten beschreiben, und Aktionen, die ihn verändern. Wer die Entscheidungen trifft, ist austauschbar:

- ein **Mensch** (Optikdesigner) über die GUI (GUI-Mikroptim), die Kommandozeile oder Python,
- ein **lokales Sprachmodell** über Ollama, wahlweise ein Instruktions- oder ein Thinking-Modell,
- eine **feste Regel** (Vergleichsstrategie ohne Modell, Abschnitt 8),
- ein **Wiedereinspieler**, der gespeicherte Entscheidungen wiederholt.

Ziel ist, dass das Sprachmodell den Optikdesigner ersetzen *kann*, ohne dass der Mensch Werkzeuge verliert. Es gibt keine Diagnose und keine Aktion nur für das Modell und keine nur für den Menschen. Beide arbeiten mit demselben Zustand, denselben Aktionen und denselben Prüfungen. Gemischte Betriebsarten (Modell schlägt vor, Mensch bestätigt) ergeben sich daraus von selbst (Abschnitt 2.2).

Die **Numerik** bleibt unverändert: Trace, Merit, LM, Homotopie, Strahlsuche und Einrasten.

Die Werkzeuge setzen an den Entscheidungen an, die heute Heuristiken oder das Skript `tools/m6t6_zielspezifikation.py` treffen, vor allem an den Stellen, an denen M6-T6 gescheitert ist:

| Befund | Heutige Entscheidung | Was der Entscheider entscheiden soll |
|---|---|---|
| F2 | Spalte hinter neuen Elementen bleiben fest | wann welche Spalte und Glasdicken frei werden |
| F3 | „kein Bedarf“, obwohl F_B > F_max | wo und welcher Operator eingefügt wird, wenn keine Platte aktiv ist |
| F4, F5 | feste Stufenfolge | Reihenfolge und Kopplung von Apertur, Feld, Petzval und Spektrum |
| F9, F13 | Diagnose empfiehlt nie Operator B | Kittglieder und Glaswahl nach Farbregeln, Farbe früh einbeziehen |
| – | Synthese startet immer bei einer Einzellinse | Start aus einem Gerüst nach Brechkraftverteilung und Linsenmodulen (Zhang Kap. 6, 7) |
| – | keine Tolerierung im Regelkreis | Abwägung zwischen Korrekturgüte, Empfindlichkeit und Herstellbarkeit |

**Leitregeln**

1. **Gleichberechtigung:** Jede Diagnose und jede Aktion ist über dieselbe öffentliche Python-API erreichbar. Die GUI nutzt nur diese API (GUI-Mikroptim, ADR-0002 dort), das Sprachmodell ebenso.
2. Ein Entscheider sieht nur Größen, die Mikroptim berechnet hat. Das Sprachmodell nimmt Glasdaten, Aberrationen und Empfindlichkeiten nie aus dem eigenen Gedächtnis.
3. Jede Aktion ist eine deterministische Operation mit geprüften Vorbedingungen. Kein Entscheider schreibt Systemdateien an den Werkzeugen vorbei. (Ein Mensch kann eine Systemdatei weiterhin von Hand bearbeiten; sie geht dann als neuer Wurzelkandidat in den Baum ein.)
4. Die Rangfolge der Kandidaten bleibt deterministisch (`beam_rank_before`, optimierung.md §8). Entscheider schlagen Kandidaten vor oder wählen zwischen ihnen; die Merit und die Zielprüfung bleiben der Maßstab.
5. Reproduzierbar sind Läufe über das Protokoll: Jede Entscheidung, gleich von wem, wird gespeichert und ist wieder einspielbar (Abschnitt 7).

## 2. Architektur

**Stand der Umsetzung (M10-T6a):** `mikroptim.design` gibt es als Grundgerüst über der Kommandozeile (`run_cli` mit JSON): Zustand, die in `mikroptim/design/actions.py` aufgeführten Aktionen, Kandidatenbaum, Protokoll, `HumanDecider` (Kommandozeile `python -m mikroptim.design new|state|act|tree|replay`) und `ReplayDecider`. Nicht umgesetzte Aktionen des Schemas melden `not_implemented`. Seit M10-T3 gibt es die Leseaktion `sensitivity` (Empfindlichkeiten und Herstellbarkeit über `mikroptim sensitivity`) und das Feld `sensitivity` des Zustands. Seit M11-T5 gibt es die Leseaktion `glass_preselect` (Glaspaare bzw. -tripel für Farbkorrektur, `element` hält dessen Glas fest; auch als Werkzeug für das Sprachmodell). Seit M11-T5g nimmt sie den Glasfilter von `glass_query` (`filter`, ohne `partner_of_element`; Kandidaten = Treffer der Abfrage, das feste Glas bleibt), sortiert mit `spectrum: true` nach dem spektralen Rest über das Spektrum des Systems und liefert Startkrümmungen des dünnen Glieds (`curvatures`, Biegung `bending`: `spherical` oder `coma`; [numerik.md, Abschnitt 15](numerik.md#15-algebraische-glasvorauswahl-m11-t5)). Seit M10-T6b gibt es außerdem `OllamaDecider` (`mikroptim/design/ollama.py`: Lese-Werkzeuge als Tool-Calls, danach Antwort mit `format` = eingebettetes Antwortschema `mikroptim/design/schema.py`, bis zu zwei Nachbesserungen, Protokoll von Modell, Digest, Optionen, Prüfsumme des Prompts, Denktext und Tool-Calls), `RuleDecider` (einfache Vergleichsstrategie V3), `ConfirmDecider` (Vorschlag annehmen, ändern, verwerfen) und einen Rückfall-Entscheider bei ungültigen Modellantworten. Seit M10-T7: `homotopy_point.aperture` im Zustand ist die aktuelle NA (Pupillenradius / Soll-Brennweite wie in `synth`), `spec.na` die Soll-NA; `OllamaDecider` protokolliert Tokens pro Anfrage (`trace.usage`) und kann das Modell entladen (`unload`, `run --keep-alive`); Parameter falschen Typs in Aktionen und Werkzeugaufrufen werden als strukturierte Fehler gemeldet statt die Sitzung abzubrechen. Bewertung: [m10-t7-bewertung.md](ergebnisse/m10-t7-bewertung.md). Seit M10-T6c gibt es `ClaudeDecider` (`mikroptim/design/claude.py`, [ADR-0016](adr/0016-claude-als-entscheider.md), Abschnitt 2.4). Seit M10-T4 gibt es die strukturellen Aktionen `place_ghost_dictionary`, `insert_module`, `remove_element`, `replace_glass`, `replace_group` und `start_from_skeleton` (`mikroptim/design/operators.py`, Modultabelle `mikroptim/design/modules.py`, Geometrie im Kern über `mikroptim insert`; Abschnitte 4.3–4.5, Verfahren in [optimierung.md, Abschnitt 13](optimierung.md#13-operatoren-der-entwurfssteuerung-m10-t4)). Seit M10-T5 gibt es die Wissensbasis unter `data/design/knowledge/` (`mikroptim/design/knowledge.py`, Abschnitt 5) mit den Lese-Aktionen `glass_query` und `module_lookup` (auch als Werkzeuge für Ollama und Claude), Glasfilter in allen Glasangaben der Aktionen, die Vorlagen `strong_doublet` und `petzval_group` und die Verteilung `double_gauss`; der System-Prompt enthält eine Kurzfassung der Wissensbasis, das Protokoll ihre Prüfsumme (`knowledge_sha256`). Seit M10-T7b: Zustandsfelder `glass_check` (ob `snap_glasses` möglich ist, Elemente mit Glasmodell, Namen der Kataloggläser, höchstens 60) und `spec.monochromatic` (bei einer Wellenlänge ist `spec.target_color_class` `null`); `snap_glasses` ohne Glasvariablen meldet `precondition`; Glasnamen, die nicht in den Katalogen stehen, sind vor jedem Aufruf ein Fehler `unknown_glass`; neue Kandidaten, deren Zustand sich nicht bilden lässt (geprüft ohne Merit und Empfindlichkeiten; z. B. Brechzahl ≤ 0 einer Glasvariablen nach `optimize`), sind ein Fehler `invalid_result` statt eines Sitzungsabbruchs; `set_bounds` (Teilumfang: `wd` und `t[i]`, Startwert wird auf neue Schranken geklemmt) und `set_constraint` (Teilumfang: `weight`, `enabled: false`, `paraxial_focus` neu anlegen) sind umgesetzt; `synth_segment` mit `augmented_lagrangian` (K) und `sine_condition` (w); `RuleDecider` mit Stufen bei Stillstand (Tabelle in 2.1). Die GUI-Seite folgt in GUI-Mikroptim.

```bash
python -m mikroptim.design new start.json sitzung --spec spec.json
python -m mikroptim.design run sitzung --decider ollama --model <modell> [--think] --steps 10 --fallback rule
python -m mikroptim.design run sitzung --decider ollama --model <modell> --confirm      # assistiert
python -m mikroptim.design run sitzung --decider rule --steps 10                          # Vergleich V3
python -m mikroptim.design run sitzung --decider claude [--model claude-sonnet-5-5] --max-usd 20   # Claude-API
python -m mikroptim.design replay sitzung sitzung_wieder                                  # ohne Ollama
```

```
 Mensch                       Sprachmodell (Ollama)          Regel, Wiedereinspieler
 GUI-Mikroptim, CLI, Python   Instruktions- oder Thinking-   (ohne Modell)
        │                     Modell, lokal
        │                            │                                │
        ▼                            ▼                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Entscheider-Schnittstelle  Decider.decide(state) → actions                   │
│   HumanDecider · OllamaDecider · RuleDecider · ReplayDecider · ConfirmDecider│
└───────────────▲──────────────────────────────────────────────┬───────────────┘
                │ Zustand  mikroptim.design.state/1             │ Aktionen  mikroptim.design.action/1
┌───────────────┴──────────────────────────────────────────────▼───────────────┐
│ Python-Paket mikroptim.design (neu, öffentliche API)                         │
│  state.py     Zustand aus den Diagnosen verdichten                           │
│  tools.py     Lese-Werkzeuge (Diagnosen), einzeln aufrufbar                  │
│  executor.py  Aktionen prüfen und ausführen, Fehler als JSON                 │
│  knowledge/   Linsenmodule, Brechkraftverteilungen, Glasregeln               │
│  session.py   Kandidatenbaum, Budget, Protokoll mikroptim.design.log/1       │
└───────────────▲──────────────────────────────────────────────┬───────────────┘
                │ Python-Bindings (M9-T1)                        │
┌───────────────┴──────────────────────────────────────────────▼───────────────┐
│ C++/CUDA-Kern (FP64-Trace auf der P100)                                      │
│  + neu: aberration_report, Gruppen, Klassen, Empfindlichkeiten               │
│  + neu: Operatoren insert_module, remove_element, replace_glass, …           │
└──────────────────────────────────────────────────────────────────────────────┘
```

Die Kommandozeile bekommt dieselben Werkzeuge als Unterbefehle, etwa `mikroptim design state`, `mikroptim design tool get_surface_table …` und `mikroptim design act actions.json`. Damit sind sie auch in Skripten und Jobs der GUI nutzbar.

### 2.1 Entscheider

`Decider` ist eine einzige Schnittstelle: `decide(state, tools) → Antwort im Format mikroptim.design.action/1`. Über `tools` kann ein Entscheider vor der Antwort Lese-Werkzeuge aufrufen (Diagnosen, Glasabfrage, Modulsuche). Ein Mensch tut das in der GUI durch Ansehen der Seiten, das Sprachmodell über Tool-Calls.

| Entscheider | Verhalten |
|---|---|
| `HumanDecider` | Zustand wird angezeigt (GUI-Seite oder Text auf der Kommandozeile), der Mensch wählt Aktionen aus Formularen bzw. schreibt sie als JSON; dieselbe Prüfung wie beim Modell |
| `OllamaDecider` | Abschnitt 2.3 |
| `RuleDecider` | feste Heuristik auf derselben Wissensbasis (Vergleichsstrategie V3); seit M10-T7: liegt die aktuelle NA unter der Soll-NA, `synth_segment` zur Soll-NA (höchstens 4 neue Elemente), danach `free_gaps` + `optimize` und erneut, höchstens vier Abschnitte, ein gescheiterter Abschnitt wird nicht wiederholt; dann Farb- und Sphärik-Regeln. Seit M10-T7b: bei Stillstand eines Abschnitts (Soll-NA nicht erreicht, meist `step_too_small`) oder Fehler je einmal die nächste Stufe aus `escalation`, dann ein neuer Abschnitt (höchstens fünf): (1) `free_gaps` + `optimize`; (2) aplanatische Schale `front.shell_ca` im Arbeitsabstand (`after_surface` = letzte Fläche, Lage p = 0,05 mm) aus dem Katalogglas mit dem größten n₀ (`shell_glass` oder `mikroptim glasses`); die Schale verkürzt den Arbeitsabstand etwa auf (w − p)/n′ − t, die Dicke t wird so gewählt, dass 1,05 w_min bleibt (höchstens 2 mm); bliebe t < 0,1 mm, ist die Stufe nicht anwendbar (Platz schaffen über eine gehobene Schranke des Arbeitsabstands scheiterte im Vorlauf an der Falle F12, siehe Bewertung), sonst `free_gaps` + `optimize`; (3) `place_ghost_dictionary` in der Mitte des Luftspalts mit der größten Randstrahlhöhe (≥ 0,4 mm, nicht der Arbeitsabstand; Gläser: hochbrechendes Glas und die Katalognamen der vorhandenen Platten), sofort gefolgt vom Abschnitt; (4) `insert_module` `generic.null_meniscus` im selben Spalt mit Dicke min(1 mm; 0,4 T) über die Dicken-Homotopie (4 Stufen à 50 Zyklen), dann `free_gaps` + `optimize`. Sind alle Stufen benutzt, folgen `free_gaps` + `optimize` und ein weiterer Abschnitt, solange der letzte Abschnitt die NA um mindestens 0,01 geöffnet hat. Nicht anwendbare Stufen werden übersprungen, gescheiterte gelten als benutzt; die benutzten erkennt die Regel an der Kennung `Stillstand: <Stufe>` in der Begründung (Protokoll). `synth_options` gibt jedem Abschnitt zusätzliche Parameter (z. B. `augmented_lagrangian`, `sine_condition`). Die Farbregel greift nur bei mehreren Wellenlängen (`spec.monochromatic`). Keine Linsendaten realer Objektive: alle Geometrien entstehen aus den Vorlagen und den Strahldaten des Kandidaten. Ergebnis: [m10-t7-bewertung.md](ergebnisse/m10-t7-bewertung.md), Abschnitt 9 |
| `ReplayDecider` | liefert die gespeicherten Antworten eines Protokolls |
| `ConfirmDecider(vorschlag, bestätigung)` | ein Entscheider schlägt vor, ein zweiter nimmt an, ändert oder verwirft; typisch Modell vorschlagend, Mensch bestätigend |

### 2.2 Betriebsarten

- **manuell:** `HumanDecider`. Der Mensch nutzt die Werkzeuge wie ein Entwurfsprogramm, mit Protokoll und Kandidatenbaum.
- **assistiert:** `ConfirmDecider(OllamaDecider, HumanDecider)`. Das Modell schlägt vor und begründet, der Mensch entscheidet.
- **autonom:** `OllamaDecider` mit Budget. Am Ende sieht der Mensch den Kandidatenbaum und das Protokoll.
- **Wechsel im Lauf:** Die Betriebsart ist pro Entscheidung wählbar. Ein Mensch kann einen autonomen Lauf an jedem Kandidaten übernehmen und umgekehrt, weil der Zustand vollständig im Kandidatenbaum liegt.

### 2.3 Sprachmodell über Ollama

- **Modellwahl:** Name und Art (Instruktion oder Thinking) stehen in einer Konfigurationsdatei und sind pro Lauf wählbar. Getestet werden mehrere Modelle gegen dieselben Benchmarks (Abschnitt 8).
- **Ausgabe:** Lese-Werkzeuge als Tool-Calls der Chat-API von Ollama; die Entscheidung am Ende als strukturierte Ausgabe, beschränkt durch das JSON-Schema `design_action.schema.json` (Parameter `format`). Bei Thinking-Modellen wird der Denktext getrennt geliefert und mitprotokolliert, aber nie ausgeführt.
- **Einstellungen:** Temperatur 0, fester `seed`, Kontextlänge `num_ctx` ausdrücklich setzen (der Standard von Ollama ist für Zustand plus Werkzeugergebnisse zu klein), `keep_alive` passend zur Laufdauer.
- **GPU:** Ollama nur auf der RTX 3090 (`CUDA_VISIBLE_DEVICES`), der FP64-Trace auf der P100. Läuft der Trace ebenfalls auf der 3090, belegt das geladene Modell Speicher; dann `EngineOptions::memory_limit_bytes` setzen.
- **Fehler:** Ungültige oder abgelehnte Aktionen gehen mit Fehlercode an das Modell zurück, höchstens zwei Nachbesserungen; danach entscheidet der Rückfall-Entscheider (Konfiguration: Regel, Mensch oder Abbruch).
- **Prüfsummen** von Modell (Ollama-Digest), System-Prompt und Wissensbasis stehen im Protokoll.

Die genauen Parameternamen der Ollama-API sind bei der Umsetzung gegen die aktuelle Ollama-Dokumentation zu prüfen.

### 2.4 Sprachmodell über die Claude-API (M10-T6c, ADR-0016)

Zusätzlicher Entscheider für Vergleiche; nur für Daten ohne Schutzbedarf (Kandidaten mit Katalogen oder Startsystem unter einem Verzeichnis `private` werden verweigert, `PrivacyError`). Das SDK `anthropic` ist ein optionales Extra (`pip install mikroptim[claude]`), die Zugangsdaten kommen nur aus der Umgebung (`ANTHROPIC_API_KEY` oder `ant auth login`).

| Punkt | Umsetzung |
|---|---|
| Modell | `--model`, Standard `claude-opus-5-5`; `claude-sonnet-5-5`, `claude-haiku-5-5` |
| Lese-Werkzeuge | wie bei Ollama als Tool-Calls, Ergebnisse als `tool_result`; höchstens `max_tool_calls` (6) |
| Antwort | Werkzeug `submit_decision` (Eingabe `assessment`, `actions`); `tool_choice` `auto`, weil erzwungene Werkzeugwahl bei Opus 5.5 und Sonnet 5.5 abgelehnt wird. Die Eingabe wird mit `validate_response` und gegen die Parameter von `schema.ACTION_DEFS` geprüft (Pflichtparameter, keine unbekannten Parameter, seit M10-T7c auch die Aufzählungen der obersten Ebene, z. B. `optimize.variables`; Typen und Schranken prüft die Ausführung mit strukturierten Fehlern); Fehler gehen als `tool_result` mit `is_error` zurück (höchstens zwei Nachbesserungen, danach `DecisionError` → Rückfall-Entscheider) |
| Werkzeugschemas, `strict` | aus `ACTION_DEFS` abgeleitet (`claude.strict_schema`): `oneOf` → `anyOf`, `prefixItems` → `items`, ohne Zahl- und Längenschranken (die prüft die Ausführung), jedes Objekt mit `additionalProperties: false`. **Seit M10-T7c Standard ohne `strict`:** Die API lehnt die strict-Schemas ab (400: „too many optional parameters (185) … limit: 24“; `submit_decision` allein hat 133, die sieben Lese-Werkzeuge zusammen 52; strict nur für einen Teil der Werkzeuge hilft also nicht, kleinere Schemas hießen, Aktionen oder Parameter wegzulassen). `strict` bleibt wählbar (`ClaudeDecider(strict=True)`, `--strict`), falls die Grenze fällt; `--no-strict` ist der Standard |
| Thinking, Aufwand | `thinking` adaptiv mit Zusammenfassung (protokolliert, nie ausgeführt); `output_config.effort` ausdrücklich, Standard `medium` (`--effort`) |
| Ablehnung | `stop_reason` wird vor dem Lesen geprüft; server-seitige Rückfälle `fallbacks="default"` (Beta `server-side-fallback-2026-07-01`) außer bei Haiku; abgelehnte Anfrage → `DecisionError`; `max_tokens` → Nachbesserung |
| Verlauf | `response.content` wird unverändert (mit Thinking-Blöcken) angehängt, der Verlauf nur verlängert |
| Caching | fester System-Prompt mit `cache_control`, nach Namen sortierte Werkzeugliste, automatisches Caching des Verlaufs; Treffer stehen pro Anfrage in `usage.cache_read_input_tokens` |
| Fehler | Ratenlimit, Serverfehler, Verbindung (nach den Wiederholungen des SDK) → `DecisionError`; fehlende oder ungültige Zugangsdaten, unbekanntes Modell, abgelehnte Anfrage (400) → `ClaudeUnavailableError` (Abbruch) |
| Kosten | pro Anfrage aus `usage` (Input, Output, Cache-Lesen, Cache-Schreiben = 1,25 × Input) mit den Preisen in `claude.PRICES_USD_PER_MTOK` (unbekanntes Rückfallmodell konservativ); Summe in `<sitzung>/claude_cost.json` bzw. `--cost-file` (`mikroptim.design.cost/1`), Obergrenze `--max-usd` (Standard 20 $). Vor jeder Anfrage: bisherige Summe + Schranke der Anfrage (Prompt-Zeichen/2 als Tokens zum Cache-Schreibpreis, volle `max_tokens`) ≤ Obergrenze, sonst `BudgetExceeded` (Abbruch, kein Rückfall) |
| Protokoll | `decider`: Modell, Aufwand, Rückfälle, SDK-Version, Prüfsummen von Prompt und Werkzeugliste, pro Anfrage Request-ID, bedienendes Modell, `stop_reason`, Usage und Kosten; Summen, Denktext, Tool-Calls, Rohantworten (ohne Thinking-Signaturen). Scheitert der Entscheider und springt der Rückfall ein, steht seine Spur unter `failed_decider` |

## 3. Diagnose: was der Entscheider sieht

### 3.1 Vorhandene Größen

| Quelle | Größen |
|---|---|
| `ParaxialData` | f, Schnittweiten, Hauptebenen, Petzvalsumme und -radius, Lage und Größe der Austrittspupille |
| `mikroptim trace --json`, `eval` | RMS pro Feld × Wellenlänge, versagte Strahlen, Strehl, PSF |
| Extrema (M2-T3) | größte Strahlhöhe und kleinster cos I pro Fläche |
| `ghost_diagnosis_json` | aktive Platten, Ordnungen, Glas, Merit-Gewinn, Empfehlung A/B |
| Homotopie-Protokoll (`mikroptim.synth/1`) | Punkt, F, F_B, F_G, LM-Status, Entscheidungen |
| mechanische Nebenbedingungen | Restverletzungen pro Eintrag |

### 3.2 Neu: Aberrationsbericht pro Fläche (M10-T1, Seidel-Teil umgesetzt; Vollordnung und OSC in M10-T1b)

Die folgenden Größen braucht man, um Zhangs Modulwissen anzuwenden, als Mensch wie als Modell (Zhang §2.2 und Tab. 6.1).

**Paraxiale Strahlen.** Rand- und Hauptstrahl bei λ_ref mit h, u, i bzw. h̄, ū, ī pro Fläche. `ynu_trace` liefert sie. Mikroptim tract vom kollimierten Raum (Blende) zum Objekt, also von kleiner zu großer NA. In dieser Richtung sind die paraxialen Daten nach Zhang §2.2.3 verlässlich; das iterative Verfahren aus seinem Anhang I ist deshalb **nicht** nötig.

**Seidel-Beiträge.** S_I … S_V sowie C_I und C_II (Farblängs- und Farbquerfehler) nach Welford. Ausgegeben werden sie auch als Wellenfrontkoeffizienten in Wellen bei λ_ref, damit sie mit der Merit vergleichbar sind:

| Koeffizient | aus |
|---|---|
| W₀₄₀ | S_I/8 |
| W₁₃₁ | S_II/2 |
| W₂₂₂ | S_III/2 |
| W₂₂₀ | S_IV/4 |
| W₃₁₁ | S_V/2 |
| ΔW₀₂₀ | C_I/2 |
| ΔW₁₁₁ | C_II |

Bezugsgröße ist das volle Feld (h̄ für θ_max).

**Sphärische Aberration aller Ordnungen.** Pro Fläche der Beitrag S_FSPH nach Delano (Zhang Gl. 2.6). Dafür wird ein realer meridionaler Randstrahl bei ρ = 1 getract, mit U, U′, I, I′ und der Sehnenlänge H pro Fläche. Der Anteil höherer Ordnung ist S_HSPH = S_FSPH − S_I (Gl. 2.7). Dasselbe bei ρ = 0,707 ergibt den Zonenfehler pro Fläche.

**Sphärochromasie 3. Ordnung** pro Fläche nach Berner et al., JOSA A 35(8), 2018 (Zhang Gl. 2.10), zwischen der kürzesten und der längsten gewichteten Wellenlänge.

**Verletzung der Sinusbedingung (OSC)** aus dem realen und dem paraxialen Randstrahl, als Maß für lineare Koma ohne Feldpunkte (Zhang §2.2.3). Dieselbe Größe dient als Merit-Term in M11-T4 ([optimierung-erweiterungen.md](optimierung-erweiterungen.md), Abschnitt 4).

**Orakel für die Abnahme:**
- (a) Σ_j S_FSPH,j stimmt mit dem Längs- bzw. Querfehler des getracten Randstrahls überein.
- (b) Für NA ≤ 0,05 stimmt Σ S_I mit W₀₄₀ aus der OPD überein (wie M1-T7, < 1 %).
- (c) Σ der Sphärochromasie-Beiträge stimmt mit der Differenz von Σ S_FSPH zwischen den beiden Wellenlängen überein, bis auf Terme höherer Ordnung, die mit NA² kleiner werden.

### 3.3 Neu: Gruppen, Kennzahlen, Klassen (M10-T2)

**Gruppen.** Eine Fläche gehört zur Front-, Mittel- oder Hintergruppe. Standard ist automatisch nach der Randstrahlhöhe:
- Mittelgruppe: die zusammenhängenden Glaselemente mit h ≥ 0,8 h_max.
- Frontgruppe: alles auf der Objektseite davon.
- Hintergruppe: alles auf der Blendenseite.

Ein Feld `"group"` an der Dateifläche überschreibt die Zuordnung. Module setzen es beim Einfügen (Abschnitt 4).

**Stand (Entscheidung Maintainer 2026-11-19):** umgesetzt als optionales Feld `group` (`front`, `middle`, `rear`) an den Flächen von `mikroptim.system/1` ([dateiformat.md](dateiformat.md#flächen)). Der Aberrationsbericht (`mikroptim aberrations`, JSON `groups`, Text „Gruppensummen“) summiert die Beiträge pro ausdrücklicher Gruppe (`AberrationReport::groups`); der Zustand von `mikroptim.design` (Gruppenbilanzen, Gruppe der Gläser, `replace_group`) nimmt die Gruppe einer Elementfläche statt der automatischen Zuordnung (`element_groups` in `mikroptim/design/state.py`), und `insert_module` schreibt den Parameter `group` (sonst die Gruppe des Moduls) an die neuen Flächen. Ohne Feld bleibt die automatische Zuordnung.

**Kennzahlen.**

| Kennzahl | Definition | Bezug |
|---|---|---|
| Gruppensummen | aller Beiträge aus 3.2 | Zhang Tab. 6.1 |
| Gruppenbrechkraft und Petzvalbeitrag | – | – |
| Retrofokusfaktor | r = h_exit/h_max | Zhang Gl. 6.1 |
| relativer Arbeitsabstand | κ = W.D./f | Zhang Gl. 4.1 |
| Etendue | G = π (y_obj · NA)² | Zhang Gl. 2.3; Werte von Zhangs Grenzkurven damit nachgerechnet |
| Etendue-Zone | 1 … 6 | Zhang Abb. 4.4 |

Für das Zielsystem (FZ 25, 50×, NA 0,95: y_obj = 0,25 mm) ist G ≈ 0,18 mm². Das ist **Zone 3**, also ein dreigruppiges System mit besonderen Korrekturmodulen, kein Zweigruppen-System.

**Klassen** des aktuellen Kandidaten und Zielklassen aus der Spezifikation (umgesetzt in M10-T2, `mikroptim/design/classes.py`; Fluorit über g … C in 2,5 DoF, Feldkrümmung in Seidel-Näherung, Vignettierung über versagte Strahlen am Feldrand):
- Farbklasse (a)–(g) aus dem chromatischen Fokusverlauf gegen die Schärfentiefe DoF = λ/(2 NA²) im trockenen Objektraum (Zhang Tab. 4.1, Abb. 4.2).
- Feldklasse 1–7 aus Bildfeldwölbung (Grenze 2,5 DoF), Koma und Vignettierung (Zhang Tab. 4.2).

Das Ziel der Spezifikation ist Farbklasse (c), Feldklasse 4 oder 5.

### 3.4 Neu: Empfindlichkeiten und Herstellbarkeit (M10-T3)

**Stand (M10-T3):** umgesetzt als `mikroptim sensitivity` ([cli.md](cli.md#mikroptim-sensitivity)) über den Bausteinen der Tolerierung (M8-T4); Verfahren, Formeln und Grenzwerte in [optimierung.md, Abschnitt 12](optimierung.md#12-empfindlichkeiten-und-herstellbarkeit-m10-t3). Abweichungen vom Entwurf unten: pro Komponente vier Varianten (±δ direkt, ±h·δ für die Jacobi-Spalte) statt zwei; Maß ist Q = Σ ŵ RMS² nach Abzug von Kolben und Kippung (ohne Kompensator); die Jacobi-Matrix wird nach den Toleranzparametern im selben Batch gebildet, nicht aus dem LM am Optimum übernommen (dessen Residuen enthalten die Kippung, und Brechzahlen und Dezentrierungen sind dort keine Variablen). Toleranzsatz der Datei oder Mikroptim-Standardsatz (Element 10 µm und 1′, Radius 0,1 %, Dicke 20 µm, Brechzahl 5·10⁻⁴). Herstellbarkeit: Halbkugelgrad, Mitten- und Randdicke, Luftrand, Durchmesser/Mittendicke und Zentrierbarkeit (Meniskusform); größter Einfallswinkel, Autofluoreszenz und Elementzahl folgen. Für Entscheider: Lese-Aktion `sensitivity` (Filter `kinds`, `elements`; Ausgabe sortiert nach Wirkung mit der RMS-Zunahme bei ±δ in Wellen) und im Zustand `sensitivity` (`set`, `rms_nominal_waves`, `decenter_worst` und `tilt_worst` als [[erste, letzte Fläche], RMS-Zunahme], `parameter_worst` als [[Art, Fläche], RMS-Zunahme], je die drei größten) und `mechanics.manufacturability` (`warnings`, `errors`, `findings` als [Prüfung, Fläche, Wert, Schwere, Hinweistext]). Die Befunde der Herstellbarkeit sind nur Hinweise, nie Nebenbedingungen (Entscheidung Maintainer 2026-11-19).

Ursprünglicher Entwurf:

- **Dezentrierung und Verkippung** pro Element (ADR-0011): zwei Varianten pro Element im Batch, berichtet ∂(RMS²)/∂δ² bei δ = 10 µm bzw. 1′.
- **Radien, Dicken und Brechzahlen:** aus der Jacobi-Matrix am Optimum, Spalte mal Standardtoleranz (Tabelle in der Systemdatei, sonst Vorgabe). Kostet keinen weiteren Trace.
- **Einfache Prüfungen:**
  - Halbkugelgrad h/R pro Fläche, Warnung bei > 0,95 (Hyperhemisphäre nicht fassbar, Zhang §3.2)
  - Rand- und Mittendicken, größter Einfallswinkel
  - Glas in der Autofluoreszenz-Sperrzone (Zhang Anh. VII)
  - Elementzahl gegen den Richtwert für die Abgleichlänge (Zhang Abb. 4.6: 13–15 Elemente bei 45 mm)

### 3.5 Zustandsformat `mikroptim.design.state/1`

Schema: `data/design/schemas/design_state.schema.json`.

Derselbe Zustand ist Grundlage der GUI-Ansicht eines Kandidaten und der Eingabe für das Sprachmodell. Für das Modell gilt ein Budget von etwa 4.000 Tokens. Pro Fläche werden nur die acht größten Beiträge jeder Aberrationsart gemeldet; die vollständige Tabelle liefert die Lese-Aktion `get_surface_table`. Das Feld `reference_sphere` (`kind`, `explicit`) nennt die Konvention der Referenzsphäre, mit der Merit und RMS gerechnet sind (ADR-0017; ohne Feld in der Datei `last_surface` mit `explicit` = false, Entscheidung Maintainer 2026-11-19). Beispiel (gekürzt, Zahlen erfunden):

```json
{
  "format": "mikroptim.design.state/1",
  "candidate": 17, "parent": 12, "elements": 9, "surfaces": 21,
  "spec": {"magnification": 50, "na": 0.95, "field_number": 25, "tube_focal_length": 164.5,
           "parfocal_max": 45.0, "spectrum_nm": [435.83, 656.27], "target_color_class": "c",
           "target_field_class": [4, 5], "etendue": 0.177, "zone": 3},
  "homotopy_point": {"aperture": 0.50, "field": 1.0, "spectrum_stage": 0},
  "merit": {"F": 2.1e-3, "F_B": 3.1e-3, "F_G": 1.0e-3, "rms_max_waves": 0.068,
            "rms_by_tuple": [[0, 546.07, 0.034], [0.7, 546.07, 0.043], [1.0, 546.07, 0.057]],
            "failed_rays": 0, "lm_status": "max_iterations"},
  "classes": {"color": "none", "field": 4, "vignetting": 0.0},
  "paraxial": {"f": 3.29, "working_distance": 0.25, "petzval_radius": -130.0,
               "retrofocus_factor": 0.62, "kappa": 0.076},
  "groups": {
    "front":  {"surfaces": [15, 20], "power": 0.21, "W040": -3.1, "W040_higher": 1.9,
               "W131": 0.4, "W220": -0.6, "dW020": -2.2, "dW111": 0.05, "sphchrom": -0.3},
    "middle": {"surfaces": [5, 14], "power": 0.12, "W040": 3.4, "W040_higher": -2.0,
               "W131": -0.9, "W220": -0.4, "dW020": 1.9, "dW111": -0.02, "sphchrom": 0.25},
    "rear":   {"surfaces": [1, 4], "power": -0.05, "W040": -0.2, "W040_higher": 0.1,
               "W131": 0.5, "W220": 1.0, "dW020": 0.1, "dW111": -0.03, "sphchrom": 0.02}
  },
  "top_contributors": {"W040_higher": [[18, 1.4], [16, 0.6], [9, -1.1]],
                       "dW020": [[19, -1.8], [11, 1.2]]},
  "ghosts": {"recommendation": {"operator": "A", "after_surface": 4, "order": 4, "glass": "SYN-36"},
             "active": [{"plate": 3, "sum_abs": 2.1e-3, "dominant_orders": [4, 2], "merit_gain": 0.8}]},
  "mechanics": {"violations": [], "hemisphere_max": [20, 0.81], "max_incidence_deg": [18, 57.3]},
  "sensitivity": {"set": "default", "rms_nominal_waves": 0.068,
                  "decenter_worst": [[[16, 17], 0.42], [[9, 10], 0.18]], "tilt_worst": [[[16, 17], 0.30]],
                  "parameter_worst": [[["radius", 16], 0.21]]},
  "glasses": [{"element": 1, "surface": 19, "name": "SYN-36", "nd": 1.807, "vd": 48.6, "dPgF": -0.004}],
  "budget": {"actions_left": 12, "elements_max": 15, "surfaces_max": 64},
  "history": ["#12: insert_null_meniscus nach 14 (Glas SYN-36) → F_B 4.6e-3 → 3.1e-3"]
}
```

**Umgesetzter Verlauf (`history`):** die letzten zehn verändernden Aktionen als `#<Kandidat>: <Aktion> → [neue Kandidaten]`; seit M10-T7c bei `synth_segment` mit Ausgang des Controllers und erreichter NA (`(step_too_small, NA 0.4475)`), bei `start_rejected` mit dessen Meldung (`(start_rejected, Apertur nicht geöffnet: Startpunkt nicht annehmbar …)`). Vorher sah ein abgelehnter Start wie ein Erfolg aus (Bewertung, Abschnitt 11). Seit M10-T7e mit dem zweiten Versuch nach E19 (`…, nach start_rejected optimize (20 Zyklen) und zweiter Versuch)`) und den Warnungen der Berichte (`[Warnung: …]`, z. B. Standardglas, angehobene Dicke, gesenkte Schranke im Plattenspalt), damit der Entscheider sie sieht.

## 4. Aktionen `mikroptim.design.action/1`

Schema: `data/design/schemas/design_action.schema.json`.

Eine Antwort enthält 1–4 Aktionen und zu jeder eine kurze Begründung. Das gilt für jeden Entscheider; in der GUI füllt der Mensch dieselben Felder über Formulare aus, die Begründung ist dort optional. Der Executor prüft die Vorbedingungen, führt aus und meldet pro Aktion zurück: `ok` oder `error` mit maschinenlesbarem Grund, neue Kandidatennummer und Änderungen der Kennzahlen. Bei einem Fehler darf das Modell höchstens zweimal nachbessern; ein Mensch sieht die Fehlermeldung im Formular.

„Merit unverändert“ heißt: Die Aktion ist ein Einfüge-Operator im Sinn von optimierung.md §6 und lässt F bis auf den Rundungsboden gleich. Alle anderen strukturellen Aktionen erzeugen einen neuen Kandidaten, der vor dem Vergleich nachoptimiert wird.

### 4.1 Lesen (ohne Kandidatenänderung)

Diese Werkzeuge stehen dem Sprachmodell als Tool-Calls zur Verfügung und dem Menschen als GUI-Ansichten bzw. als `mikroptim design tool …`.

| Aktion | Parameter | Umsetzung |
|---|---|---|
| `get_surface_table` | Felder, Flächenbereich | Aberrationsbericht 3.2 |
| `get_ghost_diagnosis` | – | `ghost_diagnosis_json` |
| `evaluate` | Raster, PSF ja/nein | `mikroptim eval` |
| `sensitivity` | Elemente, Arten | 3.4, `mikroptim sensitivity` (umgesetzt, M10-T3) |
| `glass_query` | Filter: Glasregeln (`rule`, `rules`), n_d, ν_d, P_g,F, ΔP_g,F, Reintransmission, Bezugsglas `partner` bzw. `partner_of_element`, `exclude`; `limit` | Regelwerk 5.3 auf die geladenen Kataloge, algebraische Vorauswahl (M11-T5); umgesetzt (M10-T5) |
| `module_lookup` | Gruppe, Aberrationsart (`corrects`), Etendue-Zone, NA, Immersion, `insertable_only` | Wissensbasis 5.1; umgesetzt (M10-T5) |

### 4.2 Läufe

| Aktion | Parameter | Umsetzung |
|---|---|---|
| `optimize` | Variablenauswahl, Zyklen, Jacobi-Verfahren | `mikroptim optimize`; `variables`: `current` (Variablen der Datei, Standard), `all_curvatures` (zusätzlich jede sphärische Fläche ohne Krümmungsvariable, ohne Schranken), `curvatures_and_gaps` (dazu die Luftspalte wie `free_gaps`), `everything` (dazu die Glasdicken wie `free_gaps` mit `include_glass_thickness`); Gläser bleiben fest. Spalte mit Geisterplatten wie bei `free_gaps` (M10-T7d fest, seit M10-T7e wieder frei, Befund E14). Die erweiterten Variablen stehen im neuen Kandidaten, der Bericht nennt sie unter `variables` (seit M10-T7c; vorher `not_implemented`) |
| `synth_segment` | Zielpunkt (Apertur, Feld, Spektrumstufe), Strahlbreite, Varianten, Elementbudget, seit M10-T7b `augmented_lagrangian` und `sine_condition`, seit M10-T7e `start_retry_cycles` | `mikroptim synth` (`--beam`, `--augmented-lagrangian`, `--sine-condition`, seit M10-T7e immer `--shrink-short-gaps`: Operator A senkt in einem Plattenspalt unter 2 × Schranke die Schranke der Teilstücke, Warnung unter `warnings` im Bericht). **E19 (M10-T7e):** endet der Abschnitt mit `start_rejected`, ohne dass eine Geisterplatte Bedarf meldet (F_B knapp über F_max, nichts eingefügt), läuft erst `optimize` mit `start_retry_cycles` LM-Zyklen (Standard 20, `actions.START_RETRY_CYCLES`; 0 schaltet ab) auf den Variablen der Datei und dann ein zweiter Versuch; wird der Start dann angenommen, zählt er, sonst bleibt das erste Ergebnis. Bericht `start_retry` (Zyklen, erste Meldung, Ausgang), im Verlauf „nach start_rejected optimize (20 Zyklen) und zweiter Versuch“ bzw. „… ebenfalls abgelehnt“ |
| `penalty_continuation` | Stufen, Faktor | Homotopie §7 |
| `snap_glasses` | k, max. Kombinationen | `mikroptim snap` |

### 4.3 Variablen, Nebenbedingungen, Homotopie

| Aktion | Wirkung | Bezug |
|---|---|---|
| `free_gaps` | Luftspalte und Glasdicken als Variablen, mit Schranken | F2; Schranken [min(0,1 mm; t) bzw. min(0,5 mm; t) für Glas, max(`max_gap`, t)]; in einem Spalt mit Geisterplatten ist die untere Schranke mindestens die größte Plattenlage (ein Spalt dünner als seine Platte bleibt fest). **Plattenspalte (Befund E14):** Operator A fügt dort ein, und beide Teilstücke erbten die untere Schranke der Spaltdicke; der Optimierer schob den Spalt nach der Blende auf die Plattenlage (B1: 2 → 1 mm), danach lehnte `synth` den Start ab (`start_rejected`, NA blieb 0,30; Bewertung Abschnitte 11 und 12). M10-T7d hielt Plattenspalte deshalb fest. Seit M10-T7e senkt Operator A in einem zu kurzen Spalt die Schranke der Teilstücke auf (T − Dicke)/2 (`synth --shrink-short-gaps`, [optimierung.md, Abschnitt 6](optimierung.md)), und `free_gaps` gibt Plattenspalte wieder frei (`include_ghost_plate_gaps`, Standard `true`; `false` wie M10-T7d; `actions.FREE_GHOST_PLATE_GAPS`). Gibt die Aktion nichts Neues frei, entsteht kein Kandidat |
| `freeze` | Variablen festsetzen | – |
| `set_bounds` | Schranken einzelner Variablen | seit M10-T7b im Teilumfang: `variable` `wd` (Arbeitsabstand) oder `t[i]`; der Startwert wird auf die neuen Schranken geklemmt |
| `set_constraint` | Petzval, Arbeitsabstand, Abgleichlänge, Mechanik: Grenze und Gewicht | F5, F7; seit M10-T7b im Teilumfang: `weight`, `enabled: false` vorhandener Einträge, `paraxial_focus` neu anlegen (gegen F12); `limit`, `mode` folgen |
| `set_homotopy_plan` | Reihenfolge und Kopplung von Apertur, Feld und Spektrum, Startfeldanteil | F4, F5, F13 |
| `place_ghost_dictionary` | Hilfsebene mit Glas-Wörterbuch an einer Lage einfügen (Merit unverändert) | F4: Lagen mit großem h̄/h für Feldfehler |
| `scale_ghost_penalty` | μ einer Platte ändern | F3 |

### 4.4 Topologie, Merit unverändert

| Aktion | Parameter | Umsetzung |
|---|---|---|
| `insert_null_meniscus` | Spalt, Lage, Glas, c₀ (Wert oder „von Fläche k“), Sattelzweige | Operator A, `insert_meniscus` und `meniscus_saddle_starts` |
| `split_lens` | Linse, Anteil, c_s (`achromat`, `mean` oder Wert), freier Teil, Zielglas oder Glasfilter | Operator B, `split_lens` und `free_glass` |

### 4.5 Topologie, mit Nachoptimierung (neue Operatoren)

**Stand (M10-T4):** umgesetzt ([optimierung.md, Abschnitt 13](optimierung.md#13-operatoren-der-entwurfssteuerung-m10-t4)); `place_ghost_dictionary` (4.3) ebenso, mit optionalem Strafgewicht `weight`. Abweichungen und Ergänzungen zum Entwurf:

- **Kompatible Erweiterung des Aktionsschemas** (`mikroptim.design.action/1` bleibt): `insert_module` mit `thickness`, `stages`, `cycles`; `remove_element` mit `stages`, `cycles`; `replace_glass` mit `cycles`; `replace_group` mit `glasses`, `stages`, `cycles`; `start_from_skeleton` mit `glasses`; `place_ghost_dictionary` mit `weight`.
- **Module:** seit M10-T5 aus der Wissensbasis (`data/design/knowledge/modules.json`, Abschnitt 5.1); `mikroptim/design/modules.py` ist nur noch die Sicht der Operatoren darauf. Einfügbar sind Module mit den Vorlagen `aplanatic_shell`, `null_meniscus`, `null_doublet`, `strong_doublet`, `gauss_pair` und `petzval_group`; `lens`, `thick_meniscus` und `amici_front` dienen dem Gerüst, `cemented_triplet` nur dem Nachschlagen. Vorlagen mit Dicke 0 lassen die Merit unverändert und wachsen über die Dicken-Homotopie (`thickness`); die aplanatische Schale hat sofort ihre Dicke und ändert die sphärische Aberration aller Ordnungen ihrer Gruppe nicht (Abnahme). **Starkes Kittglied** (`strong_doublet`, M10-T5): Null-Kittglied, dessen drei Flächen so gekrümmt sind, dass der paraxiale Randstrahl (Höhe und Steigung an der Einfügestelle, Schnitt mit der Kugel exakt) die Kittfläche unter `incidence_deg` trifft (Standard 40°, Bereich 35–50°); von beiden Biegungen die mit kleinerem |c|, Halbkugelgrad höchstens 0,9, sonst Fehler `precondition`. Ohne zweites Glas wählt es den Partner nach `merte_pair` mit dem größten |Δν_d|. Merit unverändert. **Petzval-Gruppe** (`petzval_group`, M10-T5): zwei Null-Menisken wie beim Gauss-Paar, dann Mittendicke `thickness` (Standard ¼ des Abstands) und Brechkräfte φ₁ = `power`/f (Standard 0,5) für das erste Glied in Tracerichtung und φ₂ = −φ₁/(1 − d φ₁) für das zweite (Abstand d; in Dünnlinsennäherung afokal, das zerstreuende Glied bei kleinerer Randstrahlhöhe); Vorderflächen behalten die Krümmung der Fläche vor dem Spalt, die Rückflächen folgen aus der Formel der dicken Linse. Ändert die Merit, Nachoptimierung mit `cycles` (Standard 50). Der Parameter `group` (sonst die Gruppe des Moduls) steht seit der Entscheidung des Maintainers vom 2026-11-19 im Feld `group` der neuen Flächen (Abschnitt 3.3). Seit M10-T7b bleibt bei der Dicken-Homotopie von `insert_module` der Luftspalt hinter dem wachsenden Modul mindestens auf der unteren Schranke seiner Dickenvariable; die folgenden Scheitel rücken dann.
- **`remove_element`:** entfernt bei Kittgliedern den ganzen Block; Null-Elemente sofort (Ergebnis nach `insert_null_meniscus` = Ausgangssystem), sonst Rückbau-Homotopie in `stages` Stufen (Standard 4) mit `cycles` LM-Zyklen (Standard 50).
- **`replace_glass`:** mit Namen ohne Nachoptimierung (außer `cycles` > 0); mit Filter (seit M10-T5 alle Filter der Glasabfrage, 5.3) die ersten `max_results` (Standard 4) Treffer ohne das bisherige Glas als getrennte Kandidaten, nachoptimiert und nach Merit geordnet. Ein leerer Filter ist ein Fehler.
- **Glasangaben mit Filter** (`{"filter": …}` in `insert_null_meniscus`, `split_lens.target_glass`, `insert_module`, `replace_group`, `start_from_skeleton`; M10-T5): das erste Glas der Glasabfrage (Rang der Glasvorauswahl mit `partner_of_element`, sonst Katalogreihenfolge); leer → Fehler `precondition`.
- **`replace_group`:** Gruppe wie im Zustand (Randstrahlhöhen), Module gleichmäßig verteilt im vereinigten Spalt; im Arbeitsabstand (Frontgruppe bis zum Objekt) nur die aplanatische Schale.
- **`start_from_skeleton`:** Gerüst aus dünnen Brechkräften nach den Zeichen der Verteilung (seit M10-T5 aus `power_distributions.json`, mit `double_gauss`), skaliert auf die Brennweite des Kandidaten, als neuer Wurzelkandidat (ohne Eltern). Erreicht das Gerüst die Brennweite mit positivem Faktor nicht (zu viele dicke Glieder, z. B. `double_gauss` mit fünf Linsen), werden Dicken und Abstände halbiert, höchstens zweimal (`length_scale` im Bericht). Die Startwerte (Brechkraftgewichte der Zeichen, Mittendicke 0,3 f, Luftspalte 0,15 f) sind Mikroptim-Annahmen ([optimierung.md, Abschnitt 13](optimierung.md#13-operatoren-der-entwurfssteuerung-m10-t4)).

| Aktion | Parameter | Umsetzung (neu) |
|---|---|---|
| `insert_module` | Modul-ID aus der Wissensbasis, Lage, Gläser oder Glasfilter, Gruppe | Vorlagengeometrie aus den lokalen Strahldaten (5.2), danach Homotopie in der Dicke wie Operator A. Seit M10-T7e robust gegen die zwei häufigsten Fehler der Modelle (Bewertung, Abschnitt 12.3): fehlt `glasses`, nimmt es das erste Katalogglas der Glasregel des Moduls (Regeln ohne Bezugsglas), sonst (Paarregel, keine Regel, kein Treffer) das hochbrechendste Katalogglas wie die Regel; liegt eine Dicke nach dem Einfügen unter der Schranke ihrer Variable (z. B. der Arbeitsabstand nach dem Nachfokussieren der Schale), wird ohne die unteren Dickenschranken eingefügt und die Dicke danach auf die Schranke angehoben (`merit_after` gilt vor dem Anheben). Beides steht als Warnung unter `warnings` im Bericht und im Verlauf. Seit M6-T11b: Würde der Arbeitsabstand nach dem Nachfokussieren der aplanatischen Schale negativ (kein Schrankenfall), verkleinert `insert_module` die Schale schrittweise (Dicke × 0,5, höchstens vier Schritte; die Radien folgen aus Lage und Dicke, der Arbeitsabstand danach ist etwa R₂ = Q n/n′ − t) mit Warnung; reicht das nicht, wird das Modul mit Fehler `precondition` („aplanatic_shell verworfen …“) abgelehnt, statt ein ungültiges System zu bauen |
| `remove_element` | Element | Rückbau: Brechkraft und Dicke über eine Homotopie gegen 0, dann Flächen entfernen (Umkehrung von A) |
| `replace_glass` | Element, Katalogname oder Filter | Glas tauschen, Nachoptimierung; mit Filter mehrere Varianten wie beim Einrasten |
| `replace_group` | Gruppe, Modulfolge | Gruppe ersetzen, z. B. Hintergruppe „Einzelmeniskus“ → „Gauss-Paar“ (Zhang Abb. 6.13) |
| `start_from_skeleton` | Brechkraftverteilung (Zhang Tab. 6.2), Modulfolge pro Gruppe, Gläser | neuer Wurzelkandidat statt Einzellinse (Abschnitt 6) |

### 4.6 Steuerung des Kandidatenbaums

| Aktion | Wirkung |
|---|---|
| `branch` | die nächsten Aktionen als getrennte Kandidaten ausführen (gemeinsamer LM-Aufruf, wenn die Topologie gleich ist) |
| `abandon` | Kandidat verwerfen, mit Grund (Protokoll). Seit M10-T7c beendet es nicht die Sitzung: `Session.run` macht mit einem offenen Kandidaten weiter (`resume`; Standard der nächste offene Vorfahre, im Bewertungsskript der beste offene Kandidat nach der Rangfolge der Bewertung) und nennt das in `last_errors`; ohne offenen Kandidaten endet der Lauf. Steht `abandon` nach einer verändernden Aktion derselben Antwort, betrifft es den neuen Kandidaten |
| `finish` | Kandidat als Ergebnis vorschlagen und die Sitzung beenden; Zielprüfung und Rangfolge entscheiden |

### 4.7 Prüfungen des Executors (nicht verhandelbar)

- **Topologie:** höchstens `kMaxSurfaces` Tabellenflächen, Elementbudget, Spalt breit genug für Lage und Mindestspalt.
- **Glas:** nur Gläser aus den geladenen Katalogen. Filter, die leer ausgehen, werden als Fehler gemeldet und nicht still aufgeweicht.
- **Daten:** AGENTS.md §8. Keine Linsendaten realer Objektive im Repository; Patentsysteme nur aus `data/glass/private/` bzw. einem privaten Systemverzeichnis.
- **Kosten:** Jede Aktion verbraucht Budget (Zahl der Aktionen, LM-Zyklen, Laufzeit). Ist das Budget leer, folgt `finish` mit dem besten Kandidaten. Für den Menschen ist das Budget eine Anzeige, keine Grenze.

## 5. Wissensbasis

Sie liegt unter `data/design/knowledge/` als JSON. Formuliert ist sie **in eigenen Worten** als Regeln mit Quellenangabe, nicht als Textauszug der Dissertation. Im Prompt des Sprachmodells steht davon nur eine kurze Zusammenfassung (etwa 1.500 Tokens). Den Rest holt das Modell gezielt mit `module_lookup` und `glass_query`. Dieselben Einträge zeigt die GUI dem Menschen als nachschlagbare Modultabelle.

**Stand (M10-T5):** umgesetzt.

- **Dateien:** `modules.json` (17 Module), `power_distributions.json` (9 Verteilungen), `glass_rules.json` (9 Regeln), alle im Format `mikroptim.design.knowledge/1` mit Schema `data/design/schemas/design_knowledge.schema.json`. Lader `mikroptim/design/knowledge.py`: Suche über `MIKROPTIM_KNOWLEDGE_DIR`, die Paketkopie `mikroptim/design/knowledge/` (CMake-Build und Wheel kopieren die Dateien dorthin) oder `data/design/knowledge` im Quellbaum; geprüft werden eindeutige IDs, nicht leere Quellen, bekannte Glasregeln der Module, Verweise und Zyklen der Regelbedingungen. `knowledge.digest()` (SHA-256 der drei Dateien) steht im Protokoll der Sprachmodell-Entscheider (`knowledge_sha256`), `knowledge.prompt_summary()` am Ende des System-Prompts.
- **Quellen:** Jeder Eintrag hat `source` (Kurzzeichen, aufgelöst unter `references` der Datei: Zhang 2019, Welford 1986, Kingslake & Johnson 2010, Smith 2008, Schott TIE-29, Bociort, Mikroptim-Dokumente). Was die Quelle nicht belegt – vor allem Zahlenschwellen für qualitative Aussagen („großer Abbe-Unterschied“) und die Wirkungsvektoren eigener Module –, steht im Feld `assumption` und beginnt mit „Mikroptim-Annahme“. Keine Linsendaten realer Objektive.
- **Regelbedingungen:** geschlossene Bereiche `[min, max]` (`null` offen) über `nd`, `vd`, `pgf`, `dpgf` und bei Paarregeln über `abs_delta_nd`, `abs_delta_vd`, `abs_delta_pgf`, `abs_delta_dpgf` gegenüber dem Bezugsglas, verknüpft mit `all`, `any`, `not` und `rule` (Verweis). Kennzahlen aus dem Kern (`mikroptim glasses --list`, [numerik.md, Abschnitt 17](numerik.md#17-glaskennzahlen-für-die-glasabfrage-m10-t5)).
- **`glass_query`** (`mikroptim/design/glass_query.py`): alle Angaben des Filters müssen gelten; `partner_of_element` beschränkt die Kandidaten auf die achromatischen Partner des Elementglases aus der Glasvorauswahl (M11-T5) in deren Rangfolge (mit `secondary_spectrum`, `power_sum`), sonst Katalogreihenfolge; `transmission` (`nm`, `min`, optional `thickness_mm`) wertet das Katalogfeld `transmission` aus (linear interpoliert, Umrechnung τ^(d/d₀)), Gläser ohne Daten stehen unter `no_transmission_data`. Ergebnis: `count`, `glasses` (höchstens `limit`, mit allen erfüllten Regeln), `reference`, `rules` (Titel, Erläuterung, Quelle, Annahme). Dichte, Härte und Preis enthalten die Kataloge (`mikroptim.glass/1`) nicht; Autofluoreszenz nur als Regel im n_d-ν_d-Diagramm.
- **Abnahme** (`python/tests/test_design_knowledge.py`): jede Regel, jedes Modul, jede Verteilung mit Quelle; alle Einzelregeln, beide Paarregeln (mit jedem der 39 Gläser als Bezug), 40 zufällige Filterkombinationen, `partner_of_element` und der Transmissionsfilter gegen Bruteforce auf den Testkatalogen `synthetisch.json` und `basis.json` (Kennzahlen im Test unabhängig aus den Katalogformeln, Regeln aus der Tabelle 5.3 ausgeschrieben): gleiche Listen in gleicher Reihenfolge; `module_lookup` gegen Bruteforce über `modules.json`. Bereichsfilter des Kerns gegen Bruteforce in `tests/test_glass_properties.cpp`.

### 5.1 Linsenmodule (`modules.json`)

Zhang beschreibt 29 Module: 10 für die Frontgruppe, 9 für die Mittelgruppe, 10 für die Hintergruppe (Tab. 6.4, 6.6, 6.8). Jeder Eintrag enthält:
- Wirkungsvektor über SA, ZSA, Koma, AST, FC, ACA, LCA und SPHCRM in −2 … +2
- Vorbedingungen: trocken oder Immersion, NA-Bereich, Etendue-Zone
- Nebenwirkungen: Empfindlichkeit, Kosten, Autofluoreszenz
- Vorlage für `insert_module`

Beispiele:

```json
[
  {"id": "front.shell_ca", "group": "front", "source": "Zhang Tab. 6.4 Nr. 3, Abb. 6.7",
   "effect": {"SA": 2, "coma": 2, "ACA": -1, "FC": -1},
   "requires": {"na_min": 0.6},
   "notes": "konzentrisch-aplanatische Schalenlinse hinter der Frontlinse; verteilt die Brechkraft, senkt die Dezentrierempfindlichkeit; Axialverschiebung ist SA-empfindlich (für CORR nutzbar)",
   "template": {"kind": "aplanatic_shell", "front": "concentric", "back": "aplanatic"},
   "glass_rule": "low_n_low_dispersion"},
  {"id": "front.thick_meniscus_dry", "group": "front", "source": "Zhang Tab. 6.4 Nr. 4, Anh. VII",
   "effect": {"SA": -2, "FC": 2},
   "requires": {"immersion": false, "na_min": 0.9},
   "notes": "konkave Frontfläche für Arbeitsabstand, nicht aplanatisch: große SA höherer Ordnung, die die Mittelgruppe kompensieren muss; nahezu frei von Petzvalkrümmung, entlastet die Hintergruppe",
   "template": {"kind": "thick_meniscus", "concave_to": "object"},
   "glass_rule": "front_medium_n_medium_vd"},
  {"id": "middle.merte_doublet", "group": "middle", "source": "Zhang Tab. 6.6 Nr. 3, Tab. 6.7",
   "effect": {"SA": 1, "ZSA": 2, "ACA": 1, "SPHCRM": 2},
   "requires": {},
   "notes": "stark gekrümmte Kittfläche mit kleinem Brechzahlsprung: viel SA höherer Ordnung bei geringer Brechkraftwirkung",
   "template": {"kind": "cemented_doublet", "cement": "strong"},
   "glass_rule": "merte_pair"},
  {"id": "rear.gauss_pair", "group": "rear", "source": "Zhang Tab. 6.8 Nr. 2, Abb. 6.13b",
   "effect": {"coma": 2, "AST": 2, "FC": 2, "LCA": 1},
   "requires": {},
   "notes": "zwei dicke Menisken mit stark gekrümmten Außenflächen, quasisymmetrisch; dezentrierempfindlich",
   "template": {"kind": "gauss_pair"},
   "glass_rule": "dense_flint_for_lca"}
]
```

### 5.2 Vorlagen

`insert_module` baut die Startgeometrie aus den **lokalen** paraxialen und realen Strahldaten an der Einfügestelle. So bringt das Modul zunächst wenig Störung, und der LM beginnt nahe an einem sinnvollen Punkt.

| Vorlage | Konstruktion |
|---|---|
| aplanatische Schalenlinse | Vorderfläche konzentrisch zum lokalen Bildpunkt des Randstrahls (I = 0); Rückfläche aplanatisch (Zhang §3.2, Gl. 3.1) für das gewählte n |
| Kittglied | Brechkraftaufteilung als dünnes Achromat (`achromat_split_curvature`); „stark“ heißt: Kittfläche so gekrümmt, dass der Randstrahl einen Einfallswinkel von 35–50° hat |
| Gauss-Paar und Petzval-Gruppe | Startradien aus Brechkraft und Abstand so, dass die Gruppe ungefähr afokal ist (Zhang Tab. 6.2, Typ d) |

Danach folgen Dicken-Homotopie und Penalty-Fortsetzung wie in optimierung.md §7.

### 5.3 Glasregeln (`glass_rules.json`)

Ausgewertet werden sie von `glass_query` auf den geladenen Katalogen.

| Regel | Bedingung | Quelle |
|---|---|---|
| `selectable_pgf` | P_g,F zwischen 0,535 und 0,58 (Optimum bis 0,56) | Zhang §6.3 |
| `merte_pair` | Δn_d 0,05–0,15, großer Abbe-Unterschied, kleines ΔP_g,F (umgesetzt: \|Δν_d\| ≥ 15, \|ΔP_g,F\| ≤ 0,01, Mikroptim-Annahme) | Zhang Tab. 6.7 |
| `front_medium_n_medium_vd` | n_d 1,60–1,90, ν_d 40–60; hochbrechende LASF nur, wenn die NA es erzwingt | Zhang §6.3, Anh. VII |
| `low_n_low_dispersion` | Fluorkron oder Phosphatkron für Schalenlinsen (umgesetzt: n_d ≤ 1,60, ν_d ≥ 66, Mikroptim-Annahme) | Zhang Anh. VII |
| `af_forbidden` | ν_d < 35, oder ν_d < 50 und n_d < 1,62; Ausnahme: 25 ≤ ν_d ≤ 41 und 1,65 ≤ n_d ≤ 1,80 | Zhang Abb. 2.4, Anh. VII |
| `not_af_forbidden` | Gegenteil von `af_forbidden` | wie dort |
| `dense_flint_for_lca` | positive dichte Flintgläser hinten, mittlere Dispersion bevorzugt (umgesetzt: n_d ≥ 1,65, 25 ≤ ν_d ≤ 40, Mikroptim-Annahme) | Zhang §6.4.3, Anh. VIII |
| `buried_surface` | gleiches n_d (Δn < 0,01), unterschiedliche ν_d (umgesetzt: \|Δν_d\| ≥ 10, Mikroptim-Annahme) | Zhang Anh. VIII |
| `anomalous_pgf` | anomale Teildispersion für Apochromate (umgesetzt: \|ΔP_g,F\| ≥ 0,01, Mikroptim-Annahme) | Schott TIE-29 |

Die Bereichsgrenzen sind geschlossen; „<“ der Tabelle wird als „≤“ ausgewertet (auf den Katalogen ohne Unterschied).

Für Kittglieder und Gruppen mit Farbziel greift `glass_query` zusätzlich auf die algebraische Glasvorauswahl zurück (M11-T5, [optimierung-erweiterungen.md](optimierung-erweiterungen.md), Abschnitt 5): Sie liefert Paare und Tripel mit kleinsten Einzelbrechkräften für Achromasie bzw. Apochromasie. Umgekehrt filtert `glass_preselect` seine Kandidaten mit demselben Filter wie `glass_query` (M11-T5g); beide Wege liefern damit dieselben Gläser.

Mit den frei erfundenen Testgläsern liefern viele Filter wenig oder nichts. Das ist ein Grund mehr, lizenzierte Kataloge unter `data/glass/private/` bereitzustellen (M6-T6, Punkt 8.4).

## 6. Vorgehen beim Entwurf

1. **Einordnen:** Etendue-Zone, Ziel-Farb- und Feldklasse, Richtwert für die Elementzahl.
2. **Gerüste vorschlagen:** 3–6 Gerüste nach Brechkraftverteilung (Zhang Tab. 6.2) und Modulfolge (`start_from_skeleton`, mit `branch`). Für 50×/0,95 trocken liegt nahe:
   - Frontgruppe: dicker Meniskus oder Amici-artige Frontlinse plus eine Schalenlinse
   - Mittelgruppe: zwei bis drei Kittglieder, davon eines mit Merte-Fläche
   - Hintergruppe: Gauss-Paar oder Petzval-Gruppe
   - Brechkraftverteilung PPN bzw. P(PN)0
3. **Homotopie mit Farbe von Anfang an:** zwei Wellenlängen ab NA 0,3, kleines Feld ab Start (F4, F5, F13). Apertur und Feld schrittweise gemeinsam.
4. **Bei Stillstand entscheiden:**
   - Gruppenbilanz (Tab. 6.1) und größte Beiträge lesen
   - Module nachschlagen, die die betroffene Aberration in der passenden Gruppe korrigieren
   - einfügen, ersetzen oder Spalte freigeben
   - Wörterbücher an Feldlagen setzen, wenn der Bedarf unsichtbar ist (F3, F4)
5. **Abschluss:** volles Spektrum, Einrasten, Empfindlichkeiten. Bei auffälliger Empfindlichkeit `replace_group` (etwa Gauss → Petzval, Zhang §6.4.3) und Vergleich.

Dieses Vorgehen steht im System-Prompt des Modells und ist zugleich die Grundlage des `RuleDecider`. Für den Menschen ist es eine Empfehlung.

Gegen Überkonstruktion (Zhang Kap. 8) gilt: Jede strukturelle Aktion des Modells muss in ihrer Begründung angeben, welche Gruppenbilanz sie ändern soll. Die Rangfolge bevorzugt weniger Elemente (optimierung.md §8).

## 7. Reproduzierbarkeit und Protokoll

- **Protokoll** `mikroptim.design.log/1`: pro Entscheidung der Zustand (Prüfsumme und Datei), der Entscheider (Art, bei Ollama Modellname, Digest und Optionen), aufgerufene Lese-Werkzeuge, Rohantwort und Denktext (Thinking-Modelle), geprüfte Aktionen, Ergebnisse. Verweise auf die Einträge von `mikroptim.synth/1`.
- **Wiedereinspielen** (`ReplayDecider`): Gespeicherte Antworten ersetzen den Entscheider, auch einen Menschen. Ergebnis ist derselbe Kandidatenbaum, im Rahmen der Reproduzierbarkeit der Numerik: bitgleich auf derselben GPU-Konfiguration (ADR-0007), innerhalb der Toleranzen bei anderer Verteilung auf GPUs (ADR-0013).
- **CI:** Wiedereinspiel-Läufe brauchen weder Ollama noch einen Menschen und laufen deshalb in der CI (CPU).
- **Modell selbst:** Temperatur 0 und fester `seed` machen die Antworten meist, aber nicht garantiert wiederholbar (Ollama-Version, Treiber, GPU). Verbindlich ist deshalb das Protokoll, nicht der erneute Modellaufruf.

## 8. Bewertung

| Nr. | Aufgabe | Start |
|---|---|---|
| B1 | NA 0,3 → 0,6 auf der Achse, monochromatisch | Abnahme M6-T3 |
| B2 | Zielspezifikation 50×/0,95, 436–656 nm, volles Feld | M6-T6 |
| B3 | 40×/0,85 Plan-Apo trocken, in Anlehnung an Zhang §7.2 (Schritte I–V) | Einzellinse oder Gerüst |
| B4 | Rekonstruktion: Patentsysteme aus dem privaten Verzeichnis, vereinfacht (Gruppe entfernt, Gläser vergröbert) | nur lokal; **ruht** (Entscheidung Maintainer 2026-11-22: die Patentsysteme sind Platzhalter, keine fertig entwickelten Systeme, als Referenz unbrauchbar) |

**Vergleichsverfahren**
- V0: `run_homotopy`
- V1: Strahlsuche mit Varianten
- V2: M6-T6-Skript
- V3: **regelbasierte Strategie ohne Modell** (`RuleDecider`). Sie verwendet dieselbe Aktions-API und Wissensbasis, die Entscheidungen trifft eine feste Heuristik.
- V4: **Mensch mit denselben Werkzeugen** (`HumanDecider`), soweit Zeit dafür ist; mindestens an B2.

V3 ist der entscheidende Vergleich. Er trennt den Nutzen der neuen Diagnose und der Operatoren vom Nutzen des Sprachmodells. V4 zeigt, wie weit das Modell vom Designer entfernt ist, der dieselben Werkzeuge hat. Verglichen werden mehrere Ollama-Modelle, jeweils Instruktions- und Thinking-Variante.

**Referenzsphäre der Benchmarks (Entscheidung Maintainer 2026-11-20, M10-T7c).** Alle Benchmarks rechnen mit `reference_sphere: image_infinite` ([ADR-0017](adr/0017-referenzsphaere-bildpunkt.md)). Das Skript `tools/m10t7_bewertung.py` setzt das Feld in der Startkopie im Arbeitsverzeichnis; die Startsysteme unter `data/systems/` und die Ergebnisse von M10-T7/M10-T7b bleiben unverändert und reproduzierbar. Die Konvention steht im Ergebnis-Kürzel und im Feld `reference_sphere` jeder Ergebnisdatei:

| Variante (Kommandozeile) | Referenzsphäre | Ergebnis-Kürzel |
|---|---|---|
| `V0`, `V1`, `V3`, `V3AL`, `V4`, `LM` (Standard) | `image_infinite` (`--referenzsphaere` wählt eine andere Art) | `V0RS`, `V1RS`, **`V3RS`**, `V3ALRS`, `V4RS`, `LMRS_<modell>` |
| `V0-alt`, `V1-alt`, `V3-alt`, `V3AL-alt`, `LM-alt` | `last_surface` (Startdatei ohne Feld, bitgleich zu M10-T7b) | `V0`, `V1`, `V3`, `V3AL`, `LM_<modell>` |
| `V0RS`, `V3RS` (Namen aus M10-T7b) | immer `image_infinite` | `V0RS`, `V3RS` |

Mit `--referenzsphaere image_exit_pupil` lautet das Kürzel `…RSAP`. **Referenz der Regel ist V3RS** (V3 mit `image_infinite`), **seit M6-T11b als Median über fünf gestörte Starts** (`V3RS_s1` … `_s5`, σ = 2 %; Entscheidung Maintainer 2026-11-22): Ohne `--gestoerte-starts` rechnen die Regel-Varianten (V3, V3AL) den Einzellauf und fünf gestörte Starts, `--regel-einzellauf` nur den Einzellauf; Tabelle (`--tabelle`) und Zusammenfassung führen die Referenz als eigene Zeile, den Einzellauf als „V3RS Einzellauf“ daneben ([Bericht, Abschnitt 14](ergebnisse/m10-t7-bewertung.md#14-m6-t11b-referenz-der-regel-als-median-gestörter-starts)). In M10-T7d (Plattenspalte fest) erreichte sie B1, B2 bis NA 0,453 und B3 bis 0,600; seit M10-T7e (Plattenspalte wieder frei, Operator A senkt im kurzen Spalt die Schranke) ungestört wieder B1, B2 bis 0,790 und B3, mit fünf gestörten Starts aber nur B1 (B2 Median 0,458, B3 Median 0,410; [Bericht, Abschnitt 13](ergebnisse/m10-t7-bewertung.md#13-m10-t7e-werkzeugverbesserungen-und-nachmessung)). Standard-Ergebnisverzeichnis ist `docs/ergebnisse/m10-t7c` (M10-T7d: `--out docs/ergebnisse/m10-t7d`); `--vergleich docs/ergebnisse/m10-t7b` stellt die Läufe gleicher Kürzel gegenüber. `--wiederholung 1-3` rechnet Wiederholungen unter `<kürzel>_w<N>`, `--zusammenfassung` fasst sie zusammen (Ziel-Treffer x/n, beste und mediane NA und RMS, Elemente, Kosten pro Lauf). Seit M10-T7e rechnet `--gestoerte-starts N` (mit `--stoer-sigma σ`, Standard 0,02, und `--stoer-seed`) jede Variante zusätzlich auf N leicht gestörten Starts unter `<kürzel>_s<k>`: Startwerte der Krümmungs- und Dickenvariablen und des Arbeitsabstands relativ um σ·ξ (Radien r(1 + σξ), Dicken t(1 + σξ), ξ standardnormal, deterministisch aus Seed, Benchmark und k), begrenzt auf die Schranken der Variable und nicht unter die Lage einer Geisterplatte; `--zusammenfassung` führt sie als Gruppe `<kürzel>_s*`. Für Ollama setzt `--num-ctx` die Kontextlänge (Standard 65.536, begrenzt auf das Maximum des Modells aus `/api/show`, 0 = Maximum; installiert: gpt-oss:20b 131.072, qwen3:14b 40.960).

**Messgrößen:** Ziel erreicht, Elementzahl, RMS bzw. Strehl, Empfindlichkeit, LM-Aufrufe, Laufzeit, Tokens (bei Thinking-Modellen getrennt nach Denken und Antwort), Anteil abgelehnter Aktionen.

## 9. Aufgaben (Vorschlag M10)

### M10-T1 – Aberrationsbericht pro Fläche
- **Inhalt:** Abschnitt 3.2, C++ mit Python-Binding und `mikroptim trace --aberrations`
- **Abnahme:** die drei Orakel aus 3.2

### M10-T2 – Gruppen, Kennzahlen, Klassen
- **Inhalt:** Abschnitt 3.3
- **Abnahme:** Klassen und Kennzahlen an konstruierten Testsystemen mit bekannter Antwort (frei erfundene Systeme, AGENTS.md §8)

### M10-T3 – Empfindlichkeiten
- **Stand:** umgesetzt (Abschnitt 3.4, optimierung.md Abschnitt 12)
- **Inhalt:** Abschnitt 3.4
- **Abnahme:** Dezentrierempfindlichkeit gegen Einzelrechnung mit ADR-0011; Jacobi-Empfindlichkeit gegen FD

### M10-T4 – Executor und neue Operatoren
- **Stand:** umgesetzt (Abschnitt 4.5, optimierung.md Abschnitt 13)
- **Inhalt:** Schemas, Vorbedingungen, `insert_module` (Vorlagen 5.2), `remove_element`, `replace_glass`, `replace_group`, `start_from_skeleton`, `place_ghost_dictionary`
- **Abnahme:**
  - Merit-Invarianz für `place_ghost_dictionary` wie optimierung.md §6
  - Aplanatische Vorlage erzeugt bei der Einfügung |ΔS_FSPH| < 10⁻³ der Gruppensumme
  - `remove_element` ∘ `insert_null_meniscus` ergibt das Ausgangssystem

### M10-T5 – Wissensbasis und `glass_query`
- **Stand:** umgesetzt (Abschnitt 5, numerik.md Abschnitt 17)
- **Inhalt:** Abschnitt 5
- **Abnahme:** Filter gegen Bruteforce auf den Testkatalogen; jede Regel mit Quelle

### M10-T6 – Entscheider-Schnittstelle, Ollama, Wiedereinspielen
- **Inhalt:** Abschnitte 2 und 7: `Decider` mit `HumanDecider` (Kommandozeile), `OllamaDecider` (Instruktion und Thinking, Tool-Calls, schema-beschränkte Antwort), `RuleDecider`, `ReplayDecider`, `ConfirmDecider`; Unterbefehle `mikroptim design …`
- **Abnahme:** Wiedereinspielen eines Laufs mit Modell und eines Laufs mit Mensch ergibt denselben Kandidatenbaum (gleiche GPU-Konfiguration bitgleich); ungültige Ausgaben werden abgefangen; derselbe Lauf mit zwei verschiedenen Ollama-Modellen ist allein über die Konfiguration umschaltbar

### M10-T7 – Bewertung
- **Inhalt:** Abschnitt 8
- **Abnahme:** Bericht unter `docs/ergebnisse/` mit V0–V3 auf B1–B3, mehrere Modelle, V4 an B2

**Folgeaufgabe in GUI-Mikroptim** (eigener Meilenstein dort): Seite „Entwurf“ mit Kandidatenbaum, Zustandsansicht, Formularen für alle Aktionen aus `design_action.schema.json`, Lese-Werkzeugen und der Betriebsart assistiert (Vorschlag des Modells mit Begründung annehmen, ändern, verwerfen). Die GUI nutzt dafür nur `mikroptim.design`.

**Abhängigkeiten zu M11.** M10-T4 profitiert von M11-T1 bis M11-T3 (Operatoren mit Nachoptimierung konvergieren dann verlässlicher), M10-T5 nutzt M11-T5, M10-T3 und M11-T6 teilen sich die Empfindlichkeitsrechnung.

## 10. Fragen an den Maintainer

Entschieden (ADR-0012, ADR-0013): Python-Schicht über die Bindings; Ollama mit wählbarem Modell; Mensch und Modell gleichberechtigt; Modell auf der RTX 3090, FP64-Trace auf der P100; glattes Maximum im Kernel mit Bibliotheksfunktionen, Bitgleichheit bei Verteilung auf mehrere GPUs entfällt.

Offen:

1. **Farbquerfehler-Ziel für Zeiss ICS:** Nach Zhang §6.1 lassen Leica und Zeiss einen festen Farbquerfehler im Objektiv, den die Tubuslinse ausgleicht. Mit welchem Sollwert soll die Merit rechnen?
2. ~~**Gruppenzuordnung:** automatisch nach h ≥ 0,8 h_max oder ausdrücklich in der Systemdatei?~~ Entschieden 2026-11-19: optionales Feld `group` an den Flächen, sonst automatisch (Abschnitt 3.3).
3. ~~**Private Benchmarks (B4):** welches Verzeichnis, welches Format?~~ Entschieden 2026-11-19: `data/systems/private/` (in `.gitignore`), Format `mikroptim.system/1` mit der Quelle im Feld `description`, Benchmark-Liste `b4.json`; Läufe nur lokal mit Ollama ([data/systems/README.md](../data/systems/README.md#private-patentsysteme-benchmark-b4)).
4. **Rückfall-Entscheider** bei wiederholt ungültigen Modellantworten: Regel, Mensch oder Abbruch als Standard?
5. **Erste Modelle** für die Bewertung (Abschnitt 8): welche Ollama-Modelle sollen verglichen werden?
