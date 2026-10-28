# Dateiformate

Alle Dateien sind JSON (UTF-8). Jede Datei trägt ein Feld `format` mit Name und Hauptversion. Leser lehnen unbekannte Hauptversionen ab und ignorieren unbekannte Felder mit Warnung. Inkompatible Änderungen erhöhen die Hauptversion (Entscheidung des Maintainers, siehe AGENTS.md).

## Systembeschreibung: `mikroptim.system/1`

Beispiel: [data/systems/einzellinse_bk7.json](../data/systems/einzellinse_bk7.json).

| Feld | Typ | Pflicht | Bedeutung |
|---|---|---|---|
| `format` | string | ja | `"mikroptim.system/1"` |
| `name` | string | ja | Kurzname |
| `description` | string | nein | Beschreibung |
| `units` | string | ja | nur `"mm"` |
| `glass_catalogs` | string[] | ja | Pfade relativ zum Repository-Wurzelverzeichnis oder zur Datei |
| `spectrum.wavelengths_nm` | number[] | ja | Wellenlängen |
| `spectrum.weights` | number[] | ja | Gewichte w_λ, gleiche Länge; in der Merit auf Summe 1 normiert (nur Verhältnisse zählen), 0 schaltet eine Wellenlänge in der Merit ab ([numerik.md, Abschnitt 7](numerik.md#residuen-pro-strahl-m2-gewichte-m4-t1)) |
| `spectrum.reference_nm` | number | ja | Referenzwellenlänge (OPD in Wellen, Paraxialdaten) |
| `source.type` | string | ja | `"collimated"` (Standard) oder `"point"` (Tests) |
| `source.field_angles_deg` | number[] | bei collimated | Feldwinkel θ in der y-z-Ebene |
| `source.field_weights` | number[] | bei collimated | Gewichte w_f, gleiche Länge; normiert wie `spectrum.weights` |
| `source.point` | number[3] | bei point | Quellpunkt in lokalen Koordinaten der ersten Fläche |
| `source.numerical_aperture` | number | bei point | NA im Medium der Quelle |
| `target` | object | nein | Sollwerte `focal_length` (Brennweite f bei λ_ref), `parfocal_length` (Abgleichlänge), `working_distance` (Arbeitsabstand) [mm]; einzige Quelle der Sollwerte für Feldziele und [Nebenbedingungen](#nebenbedingungen-ab-m3-t3) |
| `surfaces` | object[] | ja | Flächen in Tracerichtung, siehe unten |
| `ghost_plates` | object[] | nein | Geisterplatten (M4-T5), siehe unten |
| `variables` | object | nein | Optimierungsvariablen (ab M3-T1), siehe [Variablen](#variablen-ab-m3-t1) |
| `constraints` | object | nein | Nebenbedingungen (ab M3-T3), siehe [Nebenbedingungen](#nebenbedingungen-ab-m3-t3) |

### Flächen

| Feld | Typ | Bedeutung |
|---|---|---|
| `type` | string | `"stop"` (nur erste Fläche, plan), `"plane"`, `"sphere"` |
| `curvature` | number | 1/mm, nur bei `sphere`; Vorzeichen nach numerik.md |
| `thickness` | number | Abstand zum nächsten Scheitel; bei der letzten Fläche Abstand zur Bildebene |
| `material` | string oder object | Medium **nach** der Fläche: `"AIR"`, Glasname aus einem Katalog, `{"model": "constant", "n": 1.5}` oder `{"model": "buchdahl", "n0": …, "nu": [ν₁, ν₂, ν₃]}` |
| `semi_diameter` | number | Halbdurchmesser (mechanisch); bei `stop` der Pupillenradius |
| `decenter` | number[2] | optional (ab M8-T3): Dezentrierung [Δx, Δy] in mm relativ zur nominellen Lage des Scheitels, Standard [0, 0] |
| `tilt_deg` | number[2] | optional (ab M8-T3): Verkippung [α, β] in Grad um die lokale x- bzw. y-Achse, Drehpunkt ist der Scheitel, R = R_y(β)·R_x(α) (aktiv, rechtshändig), Standard [0, 0] |
| `comment` | string | optional |

Krümmungen statt Radien, damit Planflächen ohne Sonderwert `inf` auskommen.

**Dezentrierung und Verkippung** (M8-T3, ADR-0011; Konventionen und Rechnung: [numerik.md, Abschnitt 2](numerik.md#dezentrierung-und-verkippung-m8-t3-adr-0011)): Die Transformation betrifft nur die Fläche selbst („dezentrieren und zurück“); `thickness` bleibt der Abstand der nominellen Scheitel entlang +z, Folgeflächen sind nicht betroffen. Elementbewegungen (eine Linse als Ganzes um einen Punkt P) werden aus Verkippung und Dezentrierung der Einzelflächen zusammengesetzt; der Axialanteil geht in die Dicken. Sobald eine Fläche transformiert ist, rechnen Wellenfront, Optimierung und EVAL mit der vollen Pupille für alle Felder; Paraxialdaten bleiben koaxial. Die Felder sind eine kompatible Erweiterung von `mikroptim.system/1`: ältere Dateien bleiben gültig, koaxiale Flächen schreiben die Felder nicht. Beispiel: [data/systems/testsystem_mehrlinser_dezentriert.json](../data/systems/testsystem_mehrlinser_dezentriert.json):

```json
{ "type": "sphere", "curvature": 0.05, "thickness": 4.0, "material": { "model": "constant", "n": 1.72 },
  "semi_diameter": 5.5, "decenter": [-0.015, 0.01], "tilt_deg": [0.0, 0.08] }
```

### Geisterplatten

Seit M4-T5 (Beispiel: [data/systems/einzellinse_geisterplatte.json](../data/systems/einzellinse_geisterplatte.json)):

```json
{
  "after_surface": 7,
  "distance": 0.5,
  "glass": "CAF2",
  "norm_radius": 4.0,
  "order": 8,
  "coefficients": [0, 0, 0, 0, 0, 0, 0, 0]
}
```

| Feld | Typ | Bedeutung |
|---|---|---|
| `after_surface` | ganze Zahl | Fläche i (0-basiert wie `surfaces`), nach der der Luftspalt liegt; 0 ≤ i ≤ S − 2 |
| `distance` | number | Abstand vom Scheitel der Fläche i [mm], 0 ≤ `distance` ≤ `surfaces[i].thickness` |
| `glass` | string oder object | virtuelles Glas: Glasname aus einem Katalog oder ein Materialobjekt (`constant`, `buchdahl`) wie bei `material`; nicht `"AIR"` |
| `norm_radius` | number | Normierungsradius r_g des Dickenprofils [mm], > 0 |
| `order` | ganze Zahl | Ordnung K = Zahl der Koeffizienten, 1 ≤ K ≤ 32 (`kMaxGhostTerms`) |
| `coefficients` | number[] | a_1 … a_K [mm], Länge `order` |

Die Platte liegt im Luftspalt nach `after_surface`, im Abstand `distance` von deren Scheitel. Dickenprofil t(r) = Σ a_k P_k(2(r/r_g)² − 1) und Physik nach [numerik.md, Abschnitt 6](numerik.md#6-geisterplatte). Mehrere Platten im selben Spalt sind erlaubt (auch am selben Ort); die Liste ist nach (`after_surface`, `distance`) aufsteigend sortiert. Im Modell: `mko::GhostPlate` in `SystemDescription::ghost_plates` (die Ordnung ist die Länge von `coefficients`).

#### Abbildung Dateiflächen ↔ Tabellenflächen

Tracer, Flächentabelle (`HostSurfaceTable`), Batch und Kernel arbeiten mit **Tabellenflächen**: den Dateiflächen, zwischen die die Geisterplatten als zusätzliche Flächen vom Typ `GhostPlate` eingefügt sind. Nach Dateifläche i folgen die Platten mit `after_surface` = i in der Reihenfolge der Datei. Der Luftspalt (Dicke T) wird geteilt: Dateifläche i erhält die Dicke d_1, die Platten d_2 − d_1, …, T − d_last; das Medium nach jeder Platte ist das des Spalts (Luft). Geisterplatte g der Datei ist die g-te Tabellenfläche vom Typ `GhostPlate` (`ghost_surface[g]` im Batch).

Beispiel `einzellinse_geisterplatte.json` (Platten bei 1,0 und 3,5 mm im 5-mm-Spalt nach der Blende):

| Tabellenfläche j | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Typ | Stop | GhostPlate (g = 0) | GhostPlate (g = 1) | Sphere | Sphere |
| Dateifläche | 0 | – | – | 1 | 2 |
| Dicke | 1,0 | 2,5 | 1,5 | 6,0 | 98,23 |

Die Abbildung liefert `mko::make_surface_map(system)` (`include/mikroptim/model/system.hpp`) als `SurfaceMap` mit `table_of_file`, `file_of_table` (−1 bei Geisterplatten), `ghost_of_table`, `table_of_ghost` und den Funktionen `file_surface(j)` und `table_surface(i)`, die zusätzlich die Referenzsphäre abbilden (j = S_Tabelle ↔ i = S_Datei). **Ausgaben mit Flächennummern beziehen sich auf Tabellenflächen**, insbesondere `failed_surface` in `CpuTraceResult`, `WavefrontMap` und der Batch-Trace; für Meldungen in Dateiflächen ist `file_surface` anzuwenden. `table_surface_count(system)` = Zahl der Tabellenflächen; zusammen höchstens `kMaxSurfaces` = 64. Die Paraxialoptik rechnet über dieselben Tabellenflächen. Optimierungsvariablen (M3-T1) beziehen sich auf Dateiflächen; die Dicke eines geteilten Luftspalts ist eine Variable mit dem Wert der Dateidicke (siehe [Variablen](#variablen-ab-m3-t1)).

### Variablen (ab M3-T1)

Beispiel: [data/systems/einzellinse_bk7_variablen.json](../data/systems/einzellinse_bk7_variablen.json).

```json
"variables": {
  "items": [
    { "type": "curvature", "surfaces": "all", "min": -0.05, "max": 0.05 },
    { "type": "thickness", "surface": 1, "min": 2.0, "max": 12.0, "comment": "Mittendicke" },
    { "type": "thickness", "surfaces": [0, 3], "min": 0.1, "typical": 2.0 },
    { "type": "working_distance", "min": 50.0, "max": 150.0, "typical": 10.0 }
  ]
}
```

`variables` ist ein Objekt (Platz für spätere Einstellungen) mit der Pflichtliste `items`. Jeder Eintrag beschreibt eine oder mehrere Variablen eines Typs:

| Feld | Typ | Bedeutung |
|---|---|---|
| `type` | string | `"curvature"` (Krümmung c [1/mm]), `"thickness"` (Dicke bzw. Luftabstand t [mm]), `"working_distance"` (Arbeitsabstand w [mm]), `"ghost_coefficient"` (Koeffizient a_k einer Geisterplatte [mm], ab M4-T6), `"glass"` (Glas des Mediums nach der Fläche: Buchdahl-Koordinaten n₀, ν₁, ν₂, ab M6-T1) |
| `surface` | integer | eine Fläche (Index in `surfaces`, ab 0) |
| `surfaces` | integer[] oder `"all"` | mehrere Flächen in dieser Reihenfolge; `"all"`: bei `curvature` alle Flächen vom Typ `sphere`, bei `thickness` alle Flächen außer der letzten, bei `glass` alle Flächen, deren Material nicht `"AIR"` ist |
| `plate` | integer | nur `ghost_coefficient`: eine Geisterplatte (Index in `ghost_plates`, ab 0) |
| `plates` | integer[] oder `"all"` | nur `ghost_coefficient`: mehrere Geisterplatten in dieser Reihenfolge bzw. alle |
| `order` | integer | nur `ghost_coefficient`: eine Ordnung k (1 … K der Platte, wie a_k) |
| `orders` | integer[] oder `"all"` | nur `ghost_coefficient`: mehrere Ordnungen bzw. alle; fehlen `order` und `orders`, gelten alle Ordnungen jeder gewählten Platte |
| `min`, `max` | number | Schranken (docs/optimierung.md, Abschnitt 1: \|c\| ≤ c_max als `min` = −c_max, `max` = c_max; t_min ≤ t ≤ t_max; w ≥ w_min; Geisterkoeffizienten frei, Schranken optional; bei `glass` verboten); fehlt ein Feld, ist die Schranke offen |
| `typical` | number | typische Größe x_typ > 0 für die Schrittweite der finiten Differenzen h = ε^{1/3}·max(\|x\|, x_typ); Standard: 0,01 1/mm (Krümmung), 1 mm (Dicke, Arbeitsabstand), 10⁻⁴ mm (Geisterkoeffizient, klein gegen die Glättung der L1-Strafe, docs/optimierung.md Abschnitt 1); bei `glass` verboten (fest 1, 0,01, 0,001 für n₀, ν₁, ν₂) |
| `comment` | string | optional |

Der Startwert jeder Variable ist der Wert in `surfaces` (`curvature` bzw. `thickness`) bzw. in `ghost_plates[g].coefficients[k − 1]`; Variablen werden nicht doppelt gespeichert. Die Schranken gelten für alle Flächen bzw. Koeffizienten eines Eintrags.

Glas (ab M6-T1, Beispiel [data/systems/einzellinse_glasvariable.json](../data/systems/einzellinse_glasvariable.json)):

```json
{ "type": "glass", "surface": 1 }
```

Jede gewählte Fläche ergibt drei Variablen n₀, ν₁, ν₂ (in dieser Reihenfolge; im Modell `ResolvedVariable::component` 0, 1, 2) der Buchdahl-Koordinaten des Mediums danach (docs/numerik.md, Abschnitt 10). Das Material darf in der Datei ein Katalogglas, ein konstantes Material oder ein Buchdahl-Material sein: Vor der Optimierung wird es in ein Buchdahl-Material umgewandelt (Katalogglas: 3-Term-Fit über 400–700 nm; konstant: n₀ = n, ν = 0); ν₃ bleibt dabei fest. Der Startwert ist der des umgewandelten Materials, das optimierte Glas steht in der Ergebnisdatei als `{"model": "buchdahl", "n0": …, "nu": [ν₁, ν₂, ν₃]}` (docs/optimierung.md, Abschnitt 1). Schranken und `typical` sind nicht erlaubt; die Hülle der Katalog-Gläser wirkt als Strafterm (Nebenbedingung `glass_hull`).

Geisterkoeffizienten (ab M4-T6, Beispiel [data/systems/einzellinse_geisterplatten_exakt.json](../data/systems/einzellinse_geisterplatten_exakt.json)):

```json
{ "type": "ghost_coefficient", "plates": "all" },
{ "type": "ghost_coefficient", "plate": 1, "orders": [2, 3], "min": -0.05, "max": 0.05 }
```

Reihenfolge der aufgelösten Variablen: Platten wie angegeben (bei `"all"` aufsteigend), je Platte die Ordnungen wie angegeben (fehlend oder `"all"`: 1 … K aufsteigend). Im Modell: `VariableSpec::plate_selection`/`plates` und `order_selection`/`orders` (Form der Auswahl wie bei den Flächen, `SurfaceSelection`), aufgelöst `ResolvedVariable::plate`/`order` (sonst −1 bzw. 0; `surface` ist dann `after_surface` der Platte).

Regeln (Fehler als `FormatError` mit JSON-Pfad, z. B. `variables.items[2].surfaces[1]`):

- `type` ist Pflicht; unbekannte Typen werden abgelehnt (nicht ignoriert), damit ein Tippfehler keine Variable verschwinden lässt.
- `curvature` und `thickness` verlangen genau eines der Felder `surface` und `surfaces`; Indizes sind nichtnegative ganze Zahlen (`3` und `3.0` sind erlaubt, wie bei `ghost_plates`), die Fläche muss existieren, eine Liste darf nicht leer sein.
- `ghost_coefficient` verlangt genau eines der Felder `plate` und `plates` (keine Flächen); die Platte muss existieren, `"all"` mindestens eine Platte finden; `order`/`orders` höchstens eines, jede Ordnung in 1 … K der jeweiligen Platte, eine Liste nicht leer. `plate`, `plates`, `order`, `orders` sind bei den anderen Typen verboten. Kein Koeffizient doppelt (auch nicht über `"all"`).
- `glass` verlangt genau eines der Felder `surface` und `surfaces`; das Material nach der Fläche ist nicht `"AIR"`; `min`, `max`, `typical` sind verboten (Fehler unter dem jeweiligen Feld). Keine Glasfläche doppelt.
- `curvature` nur bei Flächen vom Typ `sphere`. Die Blende ist eben und nie variabel. Eine Planfläche, die gekrümmt werden darf, steht als `"type": "sphere", "curvature": 0` in der Datei: Ein Wechsel plane → sphere würde den Flächentyp und damit die Topologie des Batches ändern (ADR-0004).
- `thickness` nicht für die letzte Fläche: Deren Dicke ist der Arbeitsabstand, dafür gibt es genau den Typ `working_distance` (eine Quelle der Wahrheit, wie `thickness[v][S−1]` im Batch). `working_distance` hat kein `surface`/`surfaces`.
- Keine Variable doppelt, auch nicht über Kurzformen (`"all"` und ein Einzeleintrag derselben Fläche), `working_distance` höchstens einmal.
- `min` < `max` (beide endlich), `typical` > 0; der Startwert liegt in [`min`, `max`] (sonst Fehler unter `min` bzw. `max` mit dem Pfad des Werts, z. B. `surfaces[0].thickness`). Ein unzulässiger Start würde sonst durch die Projektion des Optimierers stillschweigend verändert.
- Unbekannte Felder in `variables` und in den Einträgen werden wie überall mit Warnung ignoriert.

**Flächenindizes** beziehen sich immer auf die Dateiflächen `surfaces[i]`, auch wenn Geisterplatten als zusätzliche Tabellenflächen eingefügt sind; die Abbildung auf den Batch übernimmt die `SurfaceMap` (`checked_surface_map` in `include/mikroptim/optim/variable_set.hpp`). Die Dicke einer Dateifläche, in deren Luftspalt Geisterplatten liegen, ist als Variable die ganze Dateidicke T; im Batch ändert sie nur das letzte Teilstück nach der letzten Platte, die Abstände `distance` der Platten bleiben fest (docs/optimierung.md). Ihre untere Schranke ist mindestens der `distance` der letzten Platte im Spalt (die Platte muss im Spalt bleiben).

Modell: `mko::Variables` mit Einträgen `mko::VariableSpec` (`include/mikroptim/model/variables.hpp`), wie sie in der Datei stehen (inklusive Kurzform, damit das Schreiben die Form erhält). `resolve_variables(system)` löst sie in einzelne Variablen auf (`ResolvedVariable`: Typ, Dateifläche, Schranken ±∞ wenn offen, x_typ, Eintrag) und prüft dabei die Regeln; `validate_system` ruft es auf. Reihenfolge: Einträge in Dateireihenfolge, innerhalb eines Eintrags die Flächen wie angegeben (bei `"all"` aufsteigend); sie ist die Reihenfolge der Komponenten des Variablenvektors (docs/optimierung.md).

### Nebenbedingungen (ab M3-T3)

Beispiel: [data/systems/einzellinse_bk7_nebenbedingungen.json](../data/systems/einzellinse_bk7_nebenbedingungen.json).

```json
"target": { "focal_length": 100.0, "parfocal_length": 115.0, "working_distance": 95.0 },
"constraints": {
  "items": [
    { "type": "focal_length", "weight": 100.0 },
    { "type": "parfocal_length", "reference_surface": 0, "offset": 5.0 },
    { "type": "working_distance", "mode": "min", "weight": 10.0 }
  ]
}
```

Petzval-Bedingung (ab M4-T4, Beispiel [data/systems/zweilinser_petzval.json](../data/systems/zweilinser_petzval.json)):

```json
{ "type": "petzval", "max_curvature": 0.01, "weight": 1000.0, "comment": "Petzvalradius mindestens 100 mm" }
```

L1-Strafe der Geisterkoeffizienten (ab M4-T6, Beispiele [data/systems/einzellinse_na05_geisterplatten.json](../data/systems/einzellinse_na05_geisterplatten.json), [data/systems/einzellinse_geisterplatten_exakt.json](../data/systems/einzellinse_geisterplatten_exakt.json)):

```json
{ "type": "ghost_penalty", "weight": 13.6, "order_weight": 1.0, "order_factor": 2.0, "smoothing": 1e-06 },
{ "type": "ghost_penalty", "plates": [4, 5], "weight": 100.0 }
```

Strafe außerhalb der Glas-Hülle (ab M6-T1, Beispiel [data/systems/einzellinse_glasvariable.json](../data/systems/einzellinse_glasvariable.json)):

```json
{ "type": "glass_hull", "weight": 10000.0, "margin": 0.0 }
```

Mechanische Bedingungen (ab M4-T3, Beispiel [data/systems/zweilinser_mechanik.json](../data/systems/zweilinser_mechanik.json)):

```json
{ "type": "max_semi_diameter", "surfaces": [3, 4], "semi_diameter": 2.65, "weight": 1e6 },
{ "type": "min_center_thickness", "medium": "glass", "thickness": 1.0 },
{ "type": "max_center_thickness", "medium": "air", "thickness": 8.0 },
{ "type": "min_edge_thickness", "medium": "glass", "thickness": 1.35, "height": "rays", "clearance": 0.3 },
{ "type": "max_incidence_angle", "angle": 55.0 }
```

`constraints` ist ein Objekt (Platz für spätere Einstellungen) mit der Pflichtliste `items`. Jeder Eintrag ist eine Nebenbedingung, die als Residuum √w · q in den Residuenvektor eingeht (Definitionen von q: [optimierung.md, Abschnitte 2.2 und 2.3](optimierung.md#22-paraxiale-nebenbedingungen-ab-m3-t3)); `ghost_penalty` liefert ein Residuum pro Geisterkoeffizient ([optimierung.md, Abschnitt 3](optimierung.md#3-warum-l1-für-geisterplatten)), `glass_hull` drei pro Glasvariable ([optimierung.md, Abschnitt 2.4](optimierung.md#24-glas-hülle-ab-m6-t1)), die mechanischen Typen eines pro gewählter Fläche bzw. gewähltem Spalt, √w · max(0, g) ([optimierung.md, Abschnitt 2.5](optimierung.md#25-mechanische-nebenbedingungen-ab-m4-t3)):

| Feld | Typ | Bedeutung |
|---|---|---|
| `type` | string | `"focal_length"` (Brennweite bei λ_ref), `"parfocal_length"` (Abgleichlänge), `"working_distance"` (Arbeitsabstand = Dicke der letzten Fläche), `"paraxial_focus"` (Bildebene im paraxialen Brennpunkt F′ bei λ_ref), `"petzval"` (\|κ_P\| ≤ κ_max bei λ_ref, κ_P = −n′_L P einschließlich Geisterplatten; ab M4-T4), `"ghost_penalty"` (geglättete L1-Strafe der Geisterkoeffizienten, ab M4-T6), `"glass_hull"` (Strafe außerhalb der konvexen Hülle der Katalog-Gläser für alle Glasvariablen, ab M6-T1); mechanisch (ab M4-T3): `"max_semi_diameter"` (größte Strahlhöhe + `clearance` ≤ Halbdurchmesser), `"min_center_thickness"`, `"max_center_thickness"` (Mittendicke ≥ bzw. ≤ `thickness`), `"min_edge_thickness"` (Randdicke ≥ `thickness`), `"max_incidence_angle"` (größter Einfallswinkel ≤ `angle`) |
| `mode` | string | `"equal"` (Standard): Ist = Soll; `"min"`: Ist ≥ Soll (Soll ist Untergrenze); `"max"`: Ist ≤ Soll (Obergrenze). Einseitig nur bei `parfocal_length` und `working_distance`; `petzval` nur `"max"` (Standard bei `petzval`); bei `ghost_penalty`, `glass_hull` und den mechanischen Typen verboten (bei diesen steckt die Richtung im Typ) |
| `weight` | number | Gewicht w ≥ 0 (Standard 1; 0 schaltet die Bedingung ab). Einheit 1/mm², weil die q Längen in mm sind, bei `petzval` mm², weil q eine Krümmung in 1/mm ist; √w·q ist dann dimensionslos wie die Wellenfrontresiduen (Wellen bei λ_ref), Wahl nach der Faustregel in [optimierung.md, Abschnitt 2.2](optimierung.md#22-paraxiale-nebenbedingungen-ab-m3-t3) bzw. 2.3. Bei `ghost_penalty` das Strafgewicht μ [Wellen²/mm] (Wahl aus einem Beugungsziel: optimierung.md, Abschnitt 3). Bei `glass_hull` das Gewicht des Abstands in normierten Glaskoordinaten [Wellen²] (Standard 10⁴: 1 % der Katalogspanne kostet eine Welle²; optimierung.md, Abschnitt 2.4). Mechanisch: 1/mm² (Längen), bei `max_incidence_angle` dimensionslos (g ist eine Differenz von cos I) |
| `reference_surface` | integer | nur `parfocal_length`: Fläche (Index in `surfaces`), ab deren Scheitel die Abgleichlänge zählt (Standard 0 = Blende) |
| `offset` | number | nur `parfocal_length`: Abstand [mm] der Anlagefläche vor dem Scheitel der Referenzfläche (in Gegenrichtung von +z; Standard 0) |
| `max_curvature` | number | nur `petzval`, dort Pflicht: größter zulässiger Betrag κ_max der Petzvalkrümmung [1/mm], endlich und ≥ 0 (0 = ebenes Feld). Ein Mindest-Petzvalradius R_min entspricht κ_max = 1/R_min (z. B. 130 mm → 0,0076923) |
| `plate`, `plates` | integer bzw. integer[] oder `"all"` | nur `ghost_penalty`: Geisterplatten (Index in `ghost_plates`), für die der Eintrag gilt; fehlt beides, alle Platten. Jede Platte steht in höchstens einem Eintrag (verschiedene μ pro Platte, z. B. für Operator A, optimierung.md Abschnitt 6) |
| `order_weight` | number | nur `ghost_penalty`: w₀ > 0 der Ordnungsgewichte w_k = w₀ κ^k (Standard 1) |
| `order_factor` | number | nur `ghost_penalty`: κ > 0 (Standard 2) |
| `smoothing` | number | nur `ghost_penalty`: Glättung ε > 0 [mm, Einheit des Dickenprofils] (Standard 10⁻⁶); die Restkoeffizienten nicht benötigter Platten liegen bei ε\|g_k\|/(μ w_k) (optimierung.md, Abschnitt 3) |
| `margin` | number | nur `glass_hull`: Sicherheitsabstand δ ≥ 0 nach innen in normierten Glaskoordinaten (Standard 0); die Strafe beginnt δ vor dem Rand. δ > 0 verlangt eine dreidimensionale Hülle (mindestens vier nicht koplanare Gläser) |
| `surface`, `surfaces` | integer bzw. integer[] oder `"all"` | nur mechanisch: Dateiflächen (Index in `surfaces`), höchstens eines der Felder, keine doppelt. Bei `max_semi_diameter` und `max_incidence_angle` die Fläche selbst (0 … S − 1; fehlt beides oder `"all"`: Flächen 1 … S − 1, ohne Blende); bei den Dicken der Spalt von Fläche i bis i + 1 (0 … S − 2, der Arbeitsabstand hat eigene Bedingungen; fehlt beides oder `"all"`: alle Spalte mit passendem `medium`) |
| `medium` | string | nur Mitten- und Randdicke: `"glass"` (Medium nach Fläche i ist nicht `"AIR"`) oder `"air"`; fehlt: Glas und Luft. Explizit gewählte Spalte müssen dazu passen |
| `semi_diameter` | number | nur `max_semi_diameter`: Grenze D > 0 [mm] für alle gewählten Flächen; fehlt: `semi_diameter` jeder Fläche |
| `thickness` | number | nur Mitten- und Randdicke, dort Pflicht: Grenze [mm], endlich; ≥ 0 bei `min_*`, > 0 bei `max_center_thickness` |
| `angle` | number | nur `max_incidence_angle`, dort Pflicht: größter Einfallswinkel θ_max in Grad, 0 < θ_max < 90 |
| `clearance` | number | nur `max_semi_diameter` und `min_edge_thickness`: Fassungszugabe δ ≥ 0 [mm] zur Strahlhöhe bzw. zum Halbdurchmesser der Randdicke (Standard 0) |
| `height` | string | nur `min_edge_thickness`: Halbdurchmesser h der Randdicke: `"semi_diameter"` (Standard: max der `semi_diameter` beider Flächen, unabhängig vom Trace) oder `"rays"` (max der größten Strahlhöhen beider Flächen, freie Öffnung), jeweils plus `clearance` |
| `comment` | string | optional |

**Sollwerte stehen nur in `target`** (eine Quelle der Wahrheit): `focal_length` nimmt `target.focal_length` (dieselbe Größe bestimmt die Feldziele h_f = f_soll · tan θ_f), `parfocal_length` nimmt `target.parfocal_length`, `working_distance` nimmt `target.working_distance`; bei `mode` `min`/`max` ist der Sollwert die Grenze (z. B. Mindestarbeitsabstand). `paraxial_focus` hat den festen Sollwert 0 und braucht keinen Eintrag in `target`. Ausnahme sind reine Grenzen ohne Zielgröße in `target`: κ_max der Petzval-Bedingung (`max_curvature`) und die Grenzen der mechanischen Bedingungen (`semi_diameter`, `thickness`, `angle`) stehen nur im Eintrag; `ghost_penalty` und `glass_hull` haben keine Sollgröße. Ein `target` ohne Nebenbedingung wird (außer `focal_length` für die Feldziele) nicht ausgewertet.

**Abgleichlänge.** Die Anlagefläche (Schulter des Objektivs) ist kein Teil des Linsenmodells. Die Abgleichlänge wird daher ab einer konfigurierbaren Referenz gemessen: L = `offset` + Σ der Dicken ab Fläche `reference_surface` bis zur Bildebene (einschließlich Arbeitsabstand; die Bildebene ist die Objektebene des Mikroskops, weil vom Unendlichraum zum Objekt getract wird). Mit den Standardwerten ist das der Abstand Blende – Bildebene. Die Lage der Anlagefläche legt der Maintainer fest (offene Frage im PR zu M3-T3).

Regeln (Fehler als `FormatError` mit JSON-Pfad, z. B. `constraints.items[1].mode`):

- `type` ist Pflicht; unbekannte Typen werden abgelehnt (nicht ignoriert), damit ein Tippfehler keine Bedingung verschwinden lässt.
- Der Sollwert des Typs muss in `target` stehen (sonst Fehler unter `constraints.items[i].type`); `petzval` braucht stattdessen `max_curvature` (sonst Fehler unter `constraints.items[i].max_curvature`), die Dicken `thickness` und `max_incidence_angle` `angle` (Fehler unter dem Pfad des Felds); `ghost_penalty` und `max_semi_diameter` brauchen keinen.
- Mechanische Typen: kein `mode`; `surface`/`surfaces`, `medium`, `semi_diameter`, `thickness`, `angle`, `clearance`, `height` nur bei den Typen laut Tabelle (sonst Fehler unter dem Pfad des Felds, auch bei den nicht mechanischen Typen); gewählte Flächen existieren und liegen im erlaubten Bereich, passen zu `medium`, keine doppelt; eine Auswahl, die keine Fläche ergibt (z. B. `"medium": "glass"` ohne Glas), ist ein Fehler unter `.type`.
- `focal_length` und `paraxial_focus` nur mit `mode` `equal` (oder ohne `mode`), `petzval` nur mit `max` (oder ohne `mode`).
- `weight` endlich und ≥ 0; `offset` endlich; `max_curvature` endlich und ≥ 0 und nur bei `petzval`.
- `ghost_penalty`: kein `mode`; `order_weight`, `order_factor`, `smoothing` endlich und > 0; `plate`/`plates` höchstens eines, Platten existieren, eine Liste ist nicht leer. Diese Felder sind bei den anderen Typen verboten.
- `glass_hull`: kein `mode`, höchstens ein Eintrag (er gilt für alle Glasvariablen); `margin` endlich und ≥ 0, bei anderen Typen verboten. Die Hülle entsteht aus allen Gläsern der `glass_catalogs`, deren `range_nm` 400–700 nm abdeckt ([data/glass/README.md](../data/glass/README.md): eigene Kataloge unter `data/glass/private/`). Ohne Glasvariablen hat der Eintrag keine Wirkung.
- `reference_surface` und `offset` nur bei `parfocal_length`; `reference_surface` ist eine nichtnegative ganze Zahl (`3` und `3.0` erlaubt) und die Fläche muss existieren (auch die letzte: dann L = `offset` + Arbeitsabstand).
- Jeder Typ höchstens einmal (Fehler beim zweiten Eintrag unter `.type`); `ghost_penalty` darf mehrfach stehen, jede Geisterplatte aber nur in einem Eintrag (Fehler unter dem Pfad der Platte bzw. `.type`, wenn der Eintrag alle Platten meint). Mechanische Typen dürfen beliebig oft stehen (etwa Glas und Luft mit verschiedenen Grenzen); überlappende Einträge liefern je ein eigenes Residuum.
- `paraxial_focus` nur bei `source.type` `collimated` (bei einer Punktquelle ist der Brennpunkt F′ nicht der Bildpunkt).
- Unbekannte Felder in `constraints` und in den Einträgen werden wie überall mit Warnung ignoriert.

**Flächenindizes** (`reference_surface`, `surface`/`surfaces`) beziehen sich auf die Dateiflächen; die Abbildung auf die Tabellenflächen des Batches (Geisterplatten) übernehmen `make_paraxial_constraints` bzw. `make_mechanical_constraints` mit der `SurfaceMap`. Ein Luftspalt mit Geisterplatten zählt mit seiner ganzen Dicke.

Modell: `mko::Constraints` mit Einträgen `mko::ConstraintSpec` (`include/mikroptim/model/constraints.hpp`), wie sie in der Datei stehen (fehlende Felder bleiben leer, damit das Schreiben die Form erhält; Standardwerte über `*_or_default()`), `check_constraints(system)` (von `validate_system` aufgerufen), `constraint_target(system, type)` (Sollwert aus `target`, bei `petzval` leer) und `constraint_bound(system, spec)` (Sollwert bzw. bei `petzval` `max_curvature`). Die aufgelösten Bedingungen und Residuen liefert `include/mikroptim/optim/paraxial_residuals.hpp` ([optimierung.md](optimierung.md#22-paraxiale-nebenbedingungen-ab-m3-t3)); die paraxialen Typen kennzeichnet `is_paraxial` (falsch für `ghost_penalty`). `ghost_penalty`: Felder `plate_selection`/`plates`, `order_weight`, `order_factor`, `smoothing` (Standardwerte `kDefaultGhostOrderWeight` = 1, `kDefaultGhostOrderFactor` = 2, `kDefaultGhostSmoothing` = 10⁻⁶ mm), aufgelöst in Strafterme pro Koeffizient von `make_ghost_penalty` (`include/mikroptim/optim/ghost_penalty.hpp`). `glass_hull`: Feld `margin` (`margin_or_default()`, Standard `kDefaultGlassHullMargin` = 0), `weight_or_default()` liefert bei diesem Typ `kDefaultGlassHullWeight` = 10⁴; Strafe aus `make_glass_hull_penalty` (`include/mikroptim/optim/glass_variables.hpp`). Mechanische Typen (`is_mechanical`): Felder `surface_selection`/`surfaces`, `medium` (`ConstraintMedium`), `semi_diameter`, `thickness`, `angle`, `clearance` (Standard `kDefaultMechanicalClearance` = 0), `height` (`EdgeHeight`, Standard `SemiDiameter`); `constraint_surfaces(system, spec)` liefert die gewählten Dateiflächen, `constraint_bound` die Grenze; aufgelöst pro Fläche von `make_mechanical_constraints` (`include/mikroptim/optim/mechanical_constraints.hpp`).

### Einlesen und Validierung

API in `include/mikroptim/io/system_json.hpp` (Bibliothek `Mikroptim::model`), Datenmodell `mko::SystemDescription` in `include/mikroptim/model/system.hpp`:

```cpp
SystemDescription read_system_json(std::string_view text, std::vector<std::string>* warnings = nullptr);
SystemDescription read_system_file(const std::filesystem::path& file, std::vector<std::string>* warnings = nullptr);
std::string write_system_json(const SystemDescription& system);
void write_system_file(const SystemDescription& system, const std::filesystem::path& file);
void validate_system(const SystemDescription& system);
void validate_materials(const SystemDescription& system, const std::function<bool(std::string_view)>& glass_exists);
std::vector<std::filesystem::path> resolve_glass_catalogs(const SystemDescription& system, const std::filesystem::path& base_dir);
```

Fehler im Inhalt werfen `mko::FormatError`. `json_path()` liefert den JSON-Pfad der beanstandeten Stelle (`surfaces[1].material`, `spectrum.weights[2]`), `what()` die Meldung in der Form `<datei>: <pfad>: <meldung>`. Syntaxfehler tragen keinen Pfad, aber Zeile und Spalte. Nicht lesbare Dateien werfen `std::runtime_error`.

Regeln beim Einlesen (erst die Struktur, danach `validate_system`):

- `format` wird zuerst geprüft: fremde Formatnamen und andere Hauptversionen als 1 werden abgelehnt.
- Pflichtfelder laut Tabelle; Typen werden streng geprüft (eine Zahl als Zeichenkette ist ein Fehler, ganze Zahlen sind als Zahlen erlaubt).
- `units` ist `"mm"`, `name` nicht leer, Einträge von `glass_catalogs` nicht leer.
- `spectrum`: mindestens eine Wellenlänge, alle > 0; `weights` gleich lang, jedes ≥ 0, Summe > 0; `reference_nm` > 0.
- `source`: bei `collimated` mindestens ein Feldwinkel mit |θ| < 90°, `field_weights` gleich lang (Regeln wie bei Gewichten); bei `point` genau drei Koordinaten und `numerical_aperture` > 0. Felder des jeweils anderen Typs werden mit Warnung ignoriert.
- `target` (optional): `focal_length` ≠ 0, `parfocal_length` und `working_distance` > 0.
- `surfaces`: mindestens zwei Flächen; `stop` genau an Position 0 und nur dort; die Blende liegt in Luft (Material `"AIR"`, ADR-0006); `curvature` bei `sphere` Pflicht, bei anderen Typen verboten; `thickness`, `material`, `semi_diameter` Pflicht, `semi_diameter` > 0. Dicken dürfen negativ sein (keine Plausibilitätsprüfung). `decenter` und `tilt_deg` (optional) sind Listen aus genau zwei endlichen Zahlen (Fehler unter `surfaces[i].decenter` bzw. `surfaces[i].tilt_deg`), Kippwinkel betragsmäßig < 90° (Fehler unter `surfaces[i].tilt_deg[k]`); die Blende (Fläche 0) ist das Bezugssystem und darf nicht transformiert sein (Fehler unter `surfaces[0]`).
- `material`: `"AIR"`, sonst Glasname (nicht leer); Objekte mit `model` `constant` (`n` > 0) oder `buchdahl` (`n0` > 0, `nu` mit 2 oder 3 Koeffizienten, fehlende = 0).
- `ghost_plates` (Liste von Objekten, Pfade `ghost_plates[g].<feld>`): Pflichtfelder laut Tabelle; `after_surface` und `order` nichtnegative ganze Zahlen (`7` und `7.0` sind erlaubt); `order` gleich der Länge von `coefficients`, 1 ≤ `order` ≤ 32, Koeffizienten endlich; `after_surface` ≤ S − 2 (nach der letzten Fläche liegt der Arbeitsabstand) und das Medium nach `after_surface` ist `"AIR"` (Luftspalt); 0 ≤ `distance` ≤ Dicke des Spalts; `norm_radius` > 0; `glass` wie `material`, aber nicht `"AIR"`; die Liste ist nach (`after_surface`, `distance`) aufsteigend sortiert (Fehler unter `ghost_plates[g]`). Eine leere Liste wird nicht geschrieben.
- Ob Glasnamen in den Katalogen existieren, prüft `validate_materials` mit einer vom Aufrufer gelieferten Abfrage (Fehler unter `surfaces[i].material` bzw. `ghost_plates[g].glass`), weil der Leser den Glaskatalog nicht kennt.
- `variables`: Regeln im Abschnitt [Variablen](#variablen-ab-m3-t1).
- `constraints`: Regeln im Abschnitt [Nebenbedingungen](#nebenbedingungen-ab-m3-t3).
- Unbekannte Felder werden ignoriert; als Warnung `"<pfad>: unbekanntes Feld wird ignoriert"` an `warnings` angehängt (die Bibliothek schreibt nichts auf stderr).

Glaskataloge (`resolve_glass_catalogs`): Relative Einträge werden zuerst relativ zu `base_dir` (Verzeichnis der Systemdatei) gesucht, danach in den Elternverzeichnissen aufwärts; so findet `data/systems/x.json` den Eintrag `data/glass/basis.json` im Repository-Wurzelverzeichnis. Nicht gefundene Einträge werfen `FormatError` unter `glass_catalogs[i]`.

Schreiben: Der Schreiber validiert das Modell (keine unlesbaren Dateien), gibt die Felder in der Reihenfolge der Tabellen aus, Zahlen verlustfrei in kürzester Darstellung (`10.0`, `0.00967`), zwei Leerzeichen Einrückung, Listen und Objekte einzeilig, solange die Zeile höchstens 120 Zeichen hat. `curvature` wird nur bei `sphere` geschrieben, `nu` ohne verschwindenden dritten Koeffizienten, `decenter` und `tilt_deg` nur, wenn einer ihrer Werte ≠ 0 ist (nach `semi_diameter`). Variableneinträge in der Reihenfolge `type`, `surface`/`surfaces`, `plate`/`plates`, `order`/`orders`, `min`, `max`, `typical`, `comment`, nur vorhandene Felder; Nebenbedingungen in der Reihenfolge `type`, `mode`, `weight`, `reference_surface`, `offset`, `max_curvature`, `plate`/`plates`, `order_weight`, `order_factor`, `smoothing`, `margin`, `surface`/`surfaces`, `medium`, `semi_diameter`, `thickness`, `angle`, `clearance`, `height`, `comment`, nur vorhandene Felder (Standardwerte werden nicht ergänzt). Lesen → Schreiben → Lesen ergibt dasselbe Modell, Schreiben → Lesen → Schreiben denselben Text.

## Glasdaten: `mikroptim.glass/1` {#glasdaten}

Beispiel: [data/glass/basis.json](../data/glass/basis.json).

| Feld | Typ | Bedeutung |
|---|---|---|
| `glasses[].name` | string | eindeutiger Name (Großschreibung) |
| `glasses[].model` | string | `"sellmeier"` (weitere Modelle bei Bedarf per ADR) |
| `glasses[].B`, `glasses[].C` | number[3] | Sellmeier-Koeffizienten, λ in µm, C in µm² |
| `glasses[].range_nm` | number[2] | Gültigkeitsbereich |
| `glasses[].reference.n_d` | number | Kontrollwert bei 587,56 nm (Test) |
| `glasses[].transmission` | object | optional: Reintransmission, z. B. `{"thickness_mm": 10, "nm": [...], "tau": [...]}` |
| `glasses[].source` | string | Herkunft der Daten |

Brechzahlen sind relativ zu Luft.

### Einlesen (`mikroptim/glass/catalog.hpp`, ab M1-T2)

```cpp
mko::GlassCatalog cat = mko::GlassCatalog::from_file("data/glass/basis.json");
cat.merge(mko::GlassCatalog::from_file("data/glass/private/schott.json")); // mehrere Kataloge
const mko::Glass& g = cat.at("N-BK7");   // find() liefert nullptr statt Ausnahme
double n = g.index(587.56);              // nm; Ausnahme außerhalb von range_nm
```

Regeln des Lesers (Fehler als `mko::GlassError`, Meldung mit Datei und JSON-Pfad, z. B. `basis.json: glasses[2].B: Feld mit genau 3 Zahlen erwartet`):

- `format` muss genau `"mikroptim.glass/1"` sein; `mikroptim.glass/<n>` mit n ≠ 1 wird als unbekannte Hauptversion abgelehnt.
- Pflichtfelder je Glas: `name`, `model` (nur `"sellmeier"`), `B`, `C` (je 3 endliche Zahlen), `range_nm` (2 Zahlen, 0 < min < max).
- Namen sind eindeutig, nicht leer und nicht `AIR` (Schlüsselwort der Systemdatei). Der Vergleich beachtet Groß-/Kleinschreibung. Doppelte Namen werden in einer Datei und beim Zusammenführen (`merge`) abgelehnt.
- Ein Sellmeier-Pol (λ² = C_i) innerhalb von `range_nm` wird abgelehnt.
- Optional: `reference.n_d` (wird gespeichert), `source` (wird gespeichert), `transmission` und `comment` (werden ignoriert). Unbekannte Felder werden ignoriert und in `GlassCatalog::warnings()` gemeldet.

Die Brechzahltabelle für ein System beschreibt [architektur.md](architektur.md#brechzahltabelle).

### Erzeugen aus Herstellerkatalogen (ab M9-T3)

AGF-Dateien und YAML-Dateien von refractiveindex.info wandelt `python -m mikroptim.glass_import` um (Regeln, Fit und Kontrollen: [data/glass/README.md](../data/glass/README.md#glasimport-ab-m9-t3)). Das Feld `transmission` schreibt der Import als `{"thickness_mm": d, "nm": [...], "tau": [...]}` (Reintransmission bei der Probendicke d, aufsteigende Wellenlängen).
