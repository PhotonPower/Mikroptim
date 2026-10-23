# Steuerung der Systemsynthese durch ein lokales Sprachmodell (Entwurf, M10)

- **Status:** Entwurf, braucht eine ADR (Vorschlag ADR-0012) und Entscheidungen des Maintainers (Abschnitt 10)
- **Verwandt:** [optimierung-erweiterungen.md](optimierung-erweiterungen.md) (vorgeschlagene Erweiterungen des Optimierers, M11; mehrere davon liefern Diagnosen und Aktionen für diesen Entwurf)
- **Grundlage:** Befunde aus [m6-t6-zielspezifikation.md](ergebnisse/m6-t6-zielspezifikation.md); Y. Zhang, *Systematic Design of Microscope Objectives*, Dissertation FSU Jena 2019 (im Folgenden „Zhang“, mit Kapitel- bzw. Tabellennummer)

## 1. Ziel und Abgrenzung

Ein lokales Sprachmodell übernimmt die **Entscheidungen**, die heute Heuristiken oder das Skript `tools/m6t6_zielspezifikation.py` treffen. Die **Numerik** bleibt unverändert: Trace, Merit, LM, Homotopie, Strahlsuche und Einrasten.

Gemeint sind konkret die Stellen, an denen M6-T6 gescheitert ist:

| Befund | Heutige Entscheidung | Was das Modell entscheiden soll |
|---|---|---|
| F2 | Spalte hinter neuen Elementen bleiben fest | wann welche Spalte und Glasdicken frei werden |
| F3 | „kein Bedarf“, obwohl F_B > F_max | wo und welcher Operator eingefügt wird, wenn keine Platte aktiv ist |
| F4, F5 | feste Stufenfolge | Reihenfolge und Kopplung von Apertur, Feld, Petzval und Spektrum |
| F9, F13 | Diagnose empfiehlt nie Operator B | Kittglieder und Glaswahl nach Farbregeln, Farbe früh einbeziehen |
| – | Synthese startet immer bei einer Einzellinse | Start aus einem Gerüst nach Brechkraftverteilung und Linsenmodulen (Zhang Kap. 6, 7) |
| – | keine Tolerierung im Regelkreis | Abwägung zwischen Korrekturgüte, Empfindlichkeit und Herstellbarkeit |

**Leitregeln**

1. Das Modell sieht nur Größen, die Mikroptim berechnet hat. Glasdaten, Aberrationen und Empfindlichkeiten nimmt es nie aus dem eigenen Gedächtnis.
2. Jede Aktion ist eine deterministische Operation mit geprüften Vorbedingungen. Das Modell schreibt nie Systemdateien.
3. Die Rangfolge der Kandidaten bleibt deterministisch (`beam_rank_before`, optimierung.md §8). Das Modell schlägt Kandidaten vor, bewertet sie aber nicht.
4. Bitgleichheit der Numerik (ADR-0007) bleibt unberührt. Nichtdeterministisch ist nur die Modellantwort. Sie wird protokolliert und ist wieder einspielbar (Abschnitt 7).

## 2. Architektur

```
┌──────────────────────────────────────────────────────────────────┐
│ Modellserver (RTX 3090): llama.cpp-Server oder vLLM,             │
│ OpenAI-kompatible API, Ausgabe durch JSON-Schema beschränkt      │
└───────────────▲──────────────────────────────┬───────────────────┘
                │ Zustand (agent_state/1)       │ Aktionen (agent_action/1)
┌───────────────┴──────────────────────────────▼───────────────────┐
│ Python-Paket mikroptim.agent (neu)                               │
│  state.py     Zustand aus Diagnose verdichten (Token-Budget)     │
│  executor.py  Aktionen prüfen und ausführen, Fehler als JSON     │
│  knowledge/   Linsenmodule, Brechkraftverteilungen, Glasregeln   │
│  policy.py    Schleife, Budget, Wiederholen, Protokoll           │
│  replay.py    gespeicherte Modellantworten statt Modellaufruf    │
└───────────────▲──────────────────────────────┬───────────────────┘
                │ Python-Bindings (M9-T1)       │
┌───────────────┴──────────────────────────────▼───────────────────┐
│ C++/CUDA-Kern (P100 für FP64, unverändert)                       │
│  + neu: aberration_report, Gruppen, Klassen, Empfindlichkeiten   │
│  + neu: Operatoren insert_module, remove_element, replace_glass  │
└──────────────────────────────────────────────────────────────────┘
```

**Hardware.** Das Modell läuft auf der RTX 3090: Tensor Cores, 24 GB, gute Unterstützung in den Inferenz-Frameworks. Der FP64-Trace läuft auf der P100, die ohnehin die FP64-Rolle hat (hardware.md: rund 4,7 gegen 0,55 TFLOPS FP64). Das setzt die noch offene Abnahme auf der P100 voraus. Bis dahin teilen sich Modell und Tracer die 3090; dann ist `EngineOptions::memory_limit_bytes` zu setzen.

**Zwei Ebenen der Einbindung**

- **Makroebene (zuerst umsetzen):** Das Modell entscheidet zwischen Synthese-Abschnitten, so wie heute `synth_loop` im M6-T6-Skript. Am C++-Kern ändert sich dafür nur die Diagnose (M10-T1 bis T3).
- **Mikroebene (später):** Ein Rückruf in `run_homotopy` bzw. `run_beam_search` an der Stelle „Δ < Δ_min, Diagnose“. Er ersetzt `insertion_options` und das Abbruchkriterium „kein Bedarf“. Umsetzung als `PolicyHomotopyModel`, das `OpticalHomotopyModel` umhüllt und nur diese zwei Methoden überschreibt.

## 3. Diagnose: was das Modell sieht

### 3.1 Vorhandene Größen

| Quelle | Größen |
|---|---|
| `ParaxialData` | f, Schnittweiten, Hauptebenen, Petzvalsumme und -radius, Lage und Größe der Austrittspupille |
| `mikroptim trace --json`, `eval` | RMS pro Feld × Wellenlänge, versagte Strahlen, Strehl, PSF |
| Extrema (M2-T3) | größte Strahlhöhe und kleinster cos I pro Fläche |
| `ghost_diagnosis_json` | aktive Platten, Ordnungen, Glas, Merit-Gewinn, Empfehlung A/B |
| Homotopie-Protokoll (`mikroptim.synth/1`) | Punkt, F, F_B, F_G, LM-Status, Entscheidungen |
| mechanische Nebenbedingungen | Restverletzungen pro Eintrag |

### 3.2 Neu: Aberrationsbericht pro Fläche (M10-T1)

Die folgenden Größen braucht man, um Zhangs Modulwissen anzuwenden (Zhang §2.2 und Tab. 6.1).

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

**Klassen** des aktuellen Kandidaten und Zielklassen aus der Spezifikation:
- Farbklasse (a)–(g) aus dem chromatischen Fokusverlauf gegen die Schärfentiefe DoF = λ/(2 NA²) im trockenen Objektraum (Zhang Tab. 4.1, Abb. 4.2).
- Feldklasse 1–7 aus Bildfeldwölbung (Grenze 2,5 DoF), Koma und Vignettierung (Zhang Tab. 4.2).

Das Ziel der Spezifikation ist Farbklasse (c), Feldklasse 4 oder 5.

### 3.4 Neu: Empfindlichkeiten und Herstellbarkeit (M10-T3, Vorgriff auf M8-T4)

- **Dezentrierung und Verkippung** pro Element (ADR-0011): zwei Varianten pro Element im Batch, berichtet ∂(RMS²)/∂δ² bei δ = 10 µm bzw. 1′.
- **Radien, Dicken und Brechzahlen:** aus der Jacobi-Matrix am Optimum, Spalte mal Standardtoleranz (Tabelle in der Systemdatei, sonst Vorgabe). Kostet keinen weiteren Trace.
- **Einfache Prüfungen:**
  - Halbkugelgrad h/R pro Fläche, Warnung bei > 0,95 (Hyperhemisphäre nicht fassbar, Zhang §3.2)
  - Rand- und Mittendicken, größter Einfallswinkel
  - Glas in der Autofluoreszenz-Sperrzone (Zhang Anh. VII)
  - Elementzahl gegen den Richtwert für die Abgleichlänge (Zhang Abb. 4.6: 13–15 Elemente bei 45 mm)

### 3.5 Zustandsformat `mikroptim.agent.state/1`

Schema: `data/agent/schemas/agent_state.schema.json`.

Das Budget liegt bei etwa 4.000 Tokens. Pro Fläche werden nur die acht größten Beiträge jeder Aberrationsart gemeldet; die vollständige Tabelle liefert die Lese-Aktion `get_surface_table`. Beispiel (gekürzt, Zahlen erfunden):

```json
{
  "format": "mikroptim.agent.state/1",
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
  "sensitivity": {"decenter_worst": [[16, 0.42], [9, 0.18]], "tilt_worst": [[16, 0.30]]},
  "glasses": [{"element": 1, "surface": 19, "name": "SYN-36", "nd": 1.807, "vd": 48.6, "dPgF": -0.004}],
  "budget": {"actions_left": 12, "elements_max": 15, "surfaces_max": 64},
  "history": ["#12: insert_null_meniscus nach 14 (Glas SYN-36) → F_B 4.6e-3 → 3.1e-3"]
}
```

## 4. Aktionen `mikroptim.agent.action/1`

Schema: `data/agent/schemas/agent_action.schema.json`.

Eine Antwort enthält 1–4 Aktionen und zu jeder eine kurze Begründung. Der Executor prüft die Vorbedingungen, führt aus und meldet pro Aktion zurück: `ok` oder `error` mit maschinenlesbarem Grund, neue Kandidatennummer und Änderungen der Kennzahlen. Bei einem Fehler darf das Modell höchstens zweimal nachbessern.

„Merit unverändert“ heißt: Die Aktion ist ein Einfüge-Operator im Sinn von optimierung.md §6 und lässt F bis auf den Rundungsboden gleich. Alle anderen strukturellen Aktionen erzeugen einen neuen Kandidaten, der vor dem Vergleich nachoptimiert wird.

### 4.1 Lesen (ohne Kandidatenänderung)

| Aktion | Parameter | Umsetzung |
|---|---|---|
| `get_surface_table` | Felder, Flächenbereich | Aberrationsbericht 3.2 |
| `get_ghost_diagnosis` | – | `ghost_diagnosis_json` |
| `evaluate` | Raster, PSF ja/nein | `mikroptim eval` |
| `sensitivity` | Elemente, Arten | 3.4 |
| `glass_query` | Filter: n_d, ν_d, ΔP_g,F, Autofluoreszenz, Transmission, Partner für Merte-Fläche | Regelwerk 5.3 auf die geladenen Kataloge, algebraische Vorauswahl (M11-T5) |
| `module_lookup` | Gruppe, Aberrationsart, Etendue-Zone | Wissensbasis 5.1 |

### 4.2 Läufe

| Aktion | Parameter | Umsetzung |
|---|---|---|
| `optimize` | Variablenauswahl, Zyklen, Jacobi-Verfahren | `mikroptim optimize` |
| `synth_segment` | Zielpunkt (Apertur, Feld, Spektrumstufe), Strahlbreite, Varianten, Elementbudget | `mikroptim synth` (`--beam`) |
| `penalty_continuation` | Stufen, Faktor | Homotopie §7 |
| `snap_glasses` | k, max. Kombinationen | `mikroptim snap` |

### 4.3 Variablen, Nebenbedingungen, Homotopie

| Aktion | Wirkung | Bezug |
|---|---|---|
| `free_gaps` | Luftspalte und Glasdicken als Variablen, mit Schranken | F2 |
| `freeze` | Variablen festsetzen | – |
| `set_bounds` | Schranken einzelner Variablen | – |
| `set_constraint` | Petzval, Arbeitsabstand, Abgleichlänge, Mechanik: Grenze und Gewicht | F5, F7 |
| `set_homotopy_plan` | Reihenfolge und Kopplung von Apertur, Feld und Spektrum, Startfeldanteil | F4, F5, F13 |
| `place_ghost_dictionary` | Hilfsebene mit Glas-Wörterbuch an einer Lage einfügen (Merit unverändert) | F4: Lagen mit großem h̄/h für Feldfehler |
| `scale_ghost_penalty` | μ einer Platte ändern | F3 |

### 4.4 Topologie, Merit unverändert

| Aktion | Parameter | Umsetzung |
|---|---|---|
| `insert_null_meniscus` | Spalt, Lage, Glas, c₀ (Wert oder „von Fläche k“), Sattelzweige | Operator A, `insert_meniscus` und `meniscus_saddle_starts` |
| `split_lens` | Linse, Anteil, c_s (`achromat`, `mean` oder Wert), freier Teil, Zielglas oder Glasfilter | Operator B, `split_lens` und `free_glass` |

### 4.5 Topologie, mit Nachoptimierung (neue Operatoren)

| Aktion | Parameter | Umsetzung (neu) |
|---|---|---|
| `insert_module` | Modul-ID aus der Wissensbasis, Lage, Gläser oder Glasfilter, Gruppe | Vorlagengeometrie aus den lokalen Strahldaten (5.2), danach Homotopie in der Dicke wie Operator A |
| `remove_element` | Element | Rückbau: Brechkraft und Dicke über eine Homotopie gegen 0, dann Flächen entfernen (Umkehrung von A) |
| `replace_glass` | Element, Katalogname oder Filter | Glas tauschen, Nachoptimierung; mit Filter mehrere Varianten wie beim Einrasten |
| `replace_group` | Gruppe, Modulfolge | Gruppe ersetzen, z. B. Hintergruppe „Einzelmeniskus“ → „Gauss-Paar“ (Zhang Abb. 6.13) |
| `start_from_skeleton` | Brechkraftverteilung (Zhang Tab. 6.2), Modulfolge pro Gruppe, Gläser | neuer Wurzelkandidat statt Einzellinse (Abschnitt 6) |

### 4.6 Steuerung des Kandidatenbaums

| Aktion | Wirkung |
|---|---|
| `branch` | die nächsten Aktionen als getrennte Kandidaten ausführen (gemeinsamer LM-Aufruf, wenn die Topologie gleich ist) |
| `abandon` | Kandidat verwerfen, mit Grund (Protokoll) |
| `finish` | Kandidat als Ergebnis vorschlagen; Zielprüfung und Rangfolge entscheiden |

### 4.7 Prüfungen des Executors (nicht verhandelbar)

- **Topologie:** höchstens `kMaxSurfaces` Tabellenflächen, Elementbudget, Spalt breit genug für Lage und Mindestspalt.
- **Glas:** nur Gläser aus den geladenen Katalogen. Filter, die leer ausgehen, werden als Fehler gemeldet und nicht still aufgeweicht.
- **Daten:** AGENTS.md §8. Keine Linsendaten realer Objektive im Repository; Patentsysteme nur aus `data/glass/private/` bzw. einem privaten Systemverzeichnis.
- **Kosten:** Jede Aktion verbraucht Budget (Zahl der Aktionen, LM-Zyklen, Laufzeit). Ist das Budget leer, folgt `finish` mit dem besten Kandidaten.

## 5. Wissensbasis

Sie liegt unter `data/agent/knowledge/` als JSON. Formuliert ist sie **in eigenen Worten** als Regeln mit Quellenangabe, nicht als Textauszug der Dissertation. Im Prompt steht davon nur eine kurze Zusammenfassung (etwa 1.500 Tokens). Den Rest holt das Modell gezielt mit `module_lookup` und `glass_query`.

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
| `merte_pair` | Δn_d 0,05–0,15, großer Abbe-Unterschied, kleines ΔP_g,F | Zhang Tab. 6.7 |
| `front_medium_n_medium_vd` | n_d 1,60–1,90, ν_d 40–60; hochbrechende LASF nur, wenn die NA es erzwingt | Zhang §6.3, Anh. VII |
| `low_n_low_dispersion` | Fluorkron oder Phosphatkron für Schalenlinsen | Zhang Anh. VII |
| `af_forbidden` | ν_d < 35, oder ν_d < 50 und n_d < 1,62; Ausnahme: 25 ≤ ν_d ≤ 41 und 1,65 ≤ n_d ≤ 1,80 | Zhang Abb. 2.4, Anh. VII |
| `dense_flint_for_lca` | positive dichte Flintgläser hinten, mittlere Dispersion bevorzugt | Zhang §6.4.3, Anh. VIII |
| `buried_surface` | gleiches n_d (Δn < 0,01), unterschiedliche ν_d | Zhang Anh. VIII |

Für Kittglieder und Gruppen mit Farbziel greift `glass_query` zusätzlich auf die algebraische Glasvorauswahl zurück (M11-T5, [optimierung-erweiterungen.md](optimierung-erweiterungen.md), Abschnitt 5): Sie liefert Paare und Tripel mit kleinsten Einzelbrechkräften für Achromasie bzw. Apochromasie.

Mit den frei erfundenen Testgläsern liefern viele Filter wenig oder nichts. Das ist ein Grund mehr, lizenzierte Kataloge unter `data/glass/private/` bereitzustellen (M6-T6, Punkt 8.4).

## 6. Vorgehen der Strategie

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

Gegen Überkonstruktion (Zhang Kap. 8) gilt: Jede strukturelle Aktion muss in ihrer Begründung angeben, welche Gruppenbilanz sie ändern soll. Die Rangfolge bevorzugt weniger Elemente (optimierung.md §8).

## 7. Determinismus und Protokoll

- **Modellaufruf:** Temperatur 0, fester Startwert, beschränkte Ausgabe (JSON-Schema). Prüfsumme von Modelldatei, System-Prompt und Wissensbasis stehen im Protokoll.
- **Protokoll:** pro Aufruf Zustand (Prüfsumme und Datei), Rohantwort, geprüfte Aktionen, Ergebnisse. Format `mikroptim.agent.log/1`, verweist auf die Einträge von `mikroptim.synth/1`.
- **Wiedereinspielen** (`replay.py`): Gespeicherte Antworten ersetzen den Modellaufruf. Ergebnis ist derselbe Kandidatenbaum bis aufs Bit. Damit sind Läufe ohne Modellserver prüfbar (CI) und Fehler reproduzierbar.

## 8. Bewertung

| Nr. | Aufgabe | Start |
|---|---|---|
| B1 | NA 0,3 → 0,6 auf der Achse, monochromatisch | Abnahme M6-T3 |
| B2 | Zielspezifikation 50×/0,95, 436–656 nm, volles Feld | M6-T6 |
| B3 | 40×/0,85 Plan-Apo trocken, in Anlehnung an Zhang §7.2 (Schritte I–V) | Einzellinse oder Gerüst |
| B4 | Rekonstruktion: Patentsysteme aus dem privaten Verzeichnis, vereinfacht (Gruppe entfernt, Gläser vergröbert) | nur lokal |

**Vergleichsverfahren**
- V0: `run_homotopy`
- V1: Strahlsuche mit Varianten
- V2: M6-T6-Skript
- V3: **regelbasierte Strategie ohne Modell**. Sie verwendet dieselbe Aktions-API und Wissensbasis, die Entscheidungen trifft eine feste Heuristik.

V3 ist der entscheidende Vergleich. Er trennt den Nutzen der neuen Diagnose und der Operatoren vom Nutzen des Sprachmodells.

**Messgrößen:** Ziel erreicht, Elementzahl, RMS bzw. Strehl, Empfindlichkeit, LM-Aufrufe, Laufzeit, Tokens, Anteil abgelehnter Aktionen.

## 9. Aufgaben (Vorschlag M10)

### M10-T1 – Aberrationsbericht pro Fläche
- **Inhalt:** Abschnitt 3.2, C++ mit Python-Binding und `mikroptim trace --aberrations`
- **Abnahme:** die drei Orakel aus 3.2

### M10-T2 – Gruppen, Kennzahlen, Klassen
- **Inhalt:** Abschnitt 3.3
- **Abnahme:** Klassen und Kennzahlen an konstruierten Testsystemen mit bekannter Antwort (frei erfundene Systeme, AGENTS.md §8)

### M10-T3 – Empfindlichkeiten
- **Inhalt:** Abschnitt 3.4
- **Abnahme:** Dezentrierempfindlichkeit gegen Einzelrechnung mit ADR-0011; Jacobi-Empfindlichkeit gegen FD

### M10-T4 – Executor und neue Operatoren
- **Inhalt:** Schemas, Vorbedingungen, `insert_module` (Vorlagen 5.2), `remove_element`, `replace_glass`, `replace_group`, `start_from_skeleton`, `place_ghost_dictionary`
- **Abnahme:**
  - Merit-Invarianz für `place_ghost_dictionary` wie optimierung.md §6
  - Aplanatische Vorlage erzeugt bei der Einfügung |ΔS_FSPH| < 10⁻³ der Gruppensumme
  - `remove_element` ∘ `insert_null_meniscus` ergibt das Ausgangssystem

### M10-T5 – Wissensbasis und `glass_query`
- **Inhalt:** Abschnitt 5
- **Abnahme:** Filter gegen Bruteforce auf den Testkatalogen; jede Regel mit Quelle

### M10-T6 – Strategie-Schleife, Modelladapter, Wiedereinspielen
- **Inhalt:** Abschnitte 2 und 7
- **Abnahme:** Wiedereinspielen bitgleich; ungültige Ausgaben werden abgefangen

### M10-T7 – Bewertung
- **Inhalt:** Abschnitt 8
- **Abnahme:** Bericht unter `docs/ergebnisse/` mit V0–V3 auf B1–B3

**Abhängigkeiten zu M11.** M10-T4 profitiert von M11-T1 bis M11-T3 (Operatoren mit Nachoptimierung konvergieren dann verlässlicher), M10-T5 nutzt M11-T5, M10-T3 und M11-T6 teilen sich die Empfindlichkeitsrechnung.

## 10. Fragen an den Maintainer

1. **Ort der Strategie-Schicht:** Python über die Bindings (Vorschlag) oder C++?
2. **Rolle der GPUs:** Modell auf der RTX 3090, FP64-Trace auf der P100?
3. **Farbquerfehler-Ziel für Zeiss ICS:** Nach Zhang §6.1 lassen Leica und Zeiss einen festen Farbquerfehler im Objektiv, den die Tubuslinse ausgleicht. Mit welchem Sollwert soll die Merit rechnen?
4. **Gruppenzuordnung:** automatisch nach h ≥ 0,8 h_max oder ausdrücklich in der Systemdatei?
5. **Private Benchmarks (B4):** welches Verzeichnis, welches Format?
6. **Wissensbasis im Repository:** Regeln in eigenen Worten mit Quellenangabe sind unbedenklich. Bitte bestätigen, dass das so gewünscht ist.
