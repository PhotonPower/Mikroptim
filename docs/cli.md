# Kommandozeilenprogramm `mikroptim`

```
mikroptim [--version] [--devices]
mikroptim trace <system.json> [--json] [--rings N] [--spokes M] [--gpu [--device N] [--precision fp64|df64]]
mikroptim optimize <system.json> [-o <ergebnis.json>] [--json] [--quiet] [--max-iterations N]
                   [--rings N] [--spokes M] [--runs K [--perturbation s] [--seed n]] [--gpu [--device N]]
mikroptim eval <system.json> [--grid N] [--half] [--reference exit-pupil|vertex] [--json]
               [--map F,L --map-file DATEI.csv [--remove piston|tilt|defocus]] [--gpu [--device N]]
mikroptim snap <system.json> [-o <ergebnis.json>] [--json] [--neighbors k] [--max-combinations N]
               [--max-iterations N] [--rings N] [--spokes M] [--gpu [--device N]]
mikroptim insert <system.json> -o <neu.json> (--meniscus I,POS,C0,GLAS [--thickness T] |
                 --split I,C_S[,ANTEIL] [--free-glass]) [--no-variables]
mikroptim synth <start.json> --target-na NA [-o <ergebnis.json>] [--json] [--log <protokoll.json>]
                [--quiet] [--max-elements N] [--max-iterations N] [--gpu [--device N]] ...
```

| Aufruf | Wirkung |
|---|---|
| `mikroptim --version` | Version ausgeben |
| `mikroptim --devices` | CUDA-Geräte auflisten (nur mit `MIKROPTIM_ENABLE_CUDA=ON`) |
| `mikroptim trace …` | System auswerten (M1-T8, mit `--gpu` M2-T5; [Abschnitt `trace`](#mikroptim-trace)) |
| `mikroptim optimize …` | Variablen eines Systems optimieren, Ergebnis als neue Systemdatei (M3-T5; [Abschnitt `optimize`](#mikroptim-optimize)) |
| `mikroptim eval …` | Auswertung auf dem dichten Pupillenraster (Modus EVAL, M8-T1; Abschnitt [`mikroptim eval`](#mikroptim-eval)) |
| `mikroptim snap …` | Glasvariablen auf Katalog-Gläser einrasten, übrige Variablen reoptimieren (M6-T5; Abschnitt [`mikroptim snap`](#mikroptim-snap)) |
| `mikroptim insert …` | Einfüge-Operator A oder B anwenden, neues System schreiben (M6-T2; Abschnitt [`mikroptim insert`](#mikroptim-insert)) |
| `mikroptim synth …` | Systemsynthese mit dem Homotopie-Controller: Apertur (und Feld) schrittweise öffnen, Elemente bei Bedarf einfügen, Beugungsziel prüfen (M6-T3; Abschnitt [`mikroptim synth`](#mikroptim-synth)) |

Das Programm liegt nach dem Bauen unter `build/<preset>/mikroptim` (Ninja, Makefiles) bzw. `build/<preset>/<Konfiguration>/mikroptim.exe` (Visual Studio). Mit dem Python-Paket (M9-T1, [python/README.md](../python/README.md)) sind dieselben Unterbefehle auch als `python -m mikroptim <befehl> …` bzw. `mikroptim.run_cli([...])` erreichbar (Ausgabe erst am Ende des Befehls).

## `mikroptim trace`

Lädt eine Systembeschreibung `mikroptim.system/1` ([dateiformat.md](dateiformat.md)) samt Glaskatalogen und gibt aus:

- **Paraxialdaten** bei der Referenzwellenlänge `spectrum.reference_nm` (`compute_paraxial`, Definitionen und Vorzeichen: [numerik.md, Abschnitt 9](numerik.md#9-paraxialoptik)),
- **RMS-Wellenfront** pro Feld und Wellenlänge, in Wellen bei λ_ref und in mm, sowie die **Anzahl versagter Strahlen** (`evaluate_wavefront` mit den idealen Bildpunkten, [numerik.md, Abschnitt 7](numerik.md#7-referenzsphäre-und-opd); Bibliothek `Mikroptim::merit`, [architektur.md](architektur.md#wellenfront-m1-t7)).

Gerechnet wird in FP64, die Wellenfront standardmäßig auf der CPU, mit `--gpu` auf einem CUDA-Gerät (siehe [GPU-Backend](#gpu-backend---gpu)).

### Optionen

| Option | Bedeutung | Standard |
|---|---|---|
| `<system.json>` | Systemdatei; Glaskataloge werden relativ zu ihr aufgelöst | – |
| `--json` | maschinenlesbare Ausgabe `mikroptim.trace/1` statt Text | aus |
| `--rings N` | radiale Ringe der Pupillenquadratur, 1 ≤ N ≤ 16 | 10 |
| `--spokes M` | Speichen der Pupillenquadratur, M ≥ 1 (Achsfeld immer 1) | 8 |
| `--gpu` | Wellenfront auf der GPU rechnen (Kernel `RESIDUALS` über `mko::cuda::TraceEngine`) | aus (CPU) |
| `--device N` | CUDA-Gerät für `--gpu`, N ≥ 0 (Liste: `mikroptim --devices`); nur zusammen mit `--gpu` | 0 |
| `--precision P` | Rechentyp der Wellenfront: `fp64` oder `df64` (Paar aus zwei float, M7-T1; auf GeForce-Karten wie der RTX 3090 im Kernel etwa 4× schneller, Abweichung zu FP64 ≤ 10⁻⁹ mm). `df64` nur zusammen mit `--gpu`, die CPU rechnet immer in FP64 | `fp64` |
| `--help`, `-h` | Hilfe | – |

Die Pupillenquadratur folgt [numerik.md, Abschnitt 8](numerik.md#8-pupillenquadratur): Auf der Achse ein meridionaler Fächer mit N Stützstellen, außerhalb der Achse die halbe Pupille mit N·M Stützstellen (Punktquelle außerhalb der y-z-Ebene: volle Pupille).

### Exit-Codes und Fehler

| Code | Bedeutung |
|---|---|
| 0 | Auswertung ausgegeben, auch wenn die Wellenfront nicht auswertbar ist (siehe unten) |
| 1 | Fehler beim Laden oder Auswerten: Datei fehlt oder ist nicht lesbar, JSON-Syntax- oder Formatfehler, unbekanntes Glas, Brechzahl außerhalb des Gültigkeitsbereichs, … Mit `--gpu` auch: Gerät `--device N` existiert nicht oder ist ungeeignet (unter sm_60), CUDA-Fehler, Speichermangel |
| 2 | ungültige Kommandozeile (unbekannte Option, fehlende Datei, ungültige Zahl, `--device` ohne `--gpu`, `--precision df64` ohne `--gpu`, unbekannter Rechentyp) |
| 3 | `--gpu`, aber das GPU-Backend ist nicht verfügbar: ohne CUDA gebaut (`MIKROPTIM_ENABLE_CUDA=OFF`) oder kein CUDA-Gerät sichtbar. Geprüft vor dem Laden der Systemdatei (`kExitNoGpu`) |

Fehlermeldungen gehen auf stderr, die Standardausgabe bleibt dann leer. Formatfehler nennen Datei und JSON-Pfad, z. B.

```
mikroptim trace: Fehler: /pfad/system.json: surfaces[1].material: unbekanntes Glas "N-GIBTSNICHT" (in keinem der Glaskataloge enthalten)
```

Warnungen des Laders (unbekannte, ignorierte Felder in System- und Glasdateien) erscheinen als `mikroptim trace: Warnung: …` auf stderr und zusätzlich im JSON-Feld `warnings`.

**Afokale Systeme** (Brechkraft exakt 0, z. B. `planplatte_bk7.json`): Die Paraxialdaten werden ausgegeben, Brennweite, Schnittweiten, Hauptebenen und Defokus sind nicht endlich. Ohne `target.focal_length` gibt es keine Sollbrennweite für die Bildpunkte der Feldwinkel; die Wellenfront ist dann nicht auswertbar. Die Textausgabe meldet das, im JSON sind `wavefront`, `rays`, `failed_rays` und `target_focal_length` `null` und `wavefront_error` enthält den Grund. Der Exit-Code bleibt 0.

### Textausgabe

Zahlen werden printf-artig mit fester Stellenzahl in der C-Locale (Dezimalpunkt) geschrieben: Längen in mm mit 6 Nachkommastellen (`%.6f`), Größen in 1/mm und RMS in mm mit 6 Stellen in Exponentialschreibweise (`%.6e`), Wellenlängen und Feldwinkel mit 3 Nachkommastellen, RMS in Wellen mit 6 Nachkommastellen. Nicht endliche Werte erscheinen als `unendlich` bzw. `-` (nicht definiert, NaN), ein Minuszeichen vor einer zu null gerundeten Zahl entfällt. So liefern Linux und Windows dieselbe Ausgabe. Lagen beziehen sich wie in `ParaxialData` auf den Scheitel von Fläche 0 bzw. der letzten Fläche, positiv in Tracerichtung (+z).

Beispiel (`mikroptim trace data/systems/einzellinse_bk7.json`, aus dem Repository-Wurzelverzeichnis):

```
System: Einzellinse N-BK7
  Datei                                       data/systems/einzellinse_bk7.json
  Quelle                                      kollimiert, Feldwinkel 0.000, 1.000, 2.000 deg
  Wellenlängen                                486.130, 587.560, 656.270 nm
  Referenzwellenlänge                         587.560 nm
  Flächen                                     3

Paraxialdaten bei 587.560 nm
  Brennweite f'                               101.049681 mm
  Brechkraft                                  9.896122e-03 1/mm
  Hintere Schnittweite (ab letzter Fläche)    99.052088 mm
  Vordere Schnittweite (ab Fläche 0)          -94.052088 mm
  Hauptebene H (ab Fläche 0)                  6.997593 mm
  Hauptebene H' (ab letzter Fläche)           -1.997593 mm
  Defokus der Bildebene                       0.002088 mm
  Petzvalsumme                                6.589473e-03 1/mm
  Petzvalkrümmung                             -6.589473e-03 1/mm
  Petzvalradius                               -151.757193 mm
  Austrittspupille Lage (ab letzter Fläche)   -9.515815 mm
  Austrittspupille Radius                     10.744012 mm
  Pupillenvergrößerung                        1.074401

Wellenfront (Quadratur 10 Ringe x 8 Speichen, RMS in Wellen bei 587.560 nm)
  Backend                                     CPU, FP64
  Sollbrennweite für die Bildpunkte           101.049681 mm
  Feld  Winkel [deg]  Wellenlänge [nm]  Strahlen  versagt  RMS [Wellen]       RMS [mm]
     0         0.000           486.130        10        0      4.502093   2.645250e-03
     0         0.000           587.560        10        0      1.936014   1.137524e-03
     0         0.000           656.270        10        0      0.866035   5.088477e-04
     1         1.000           486.130        80        0      4.602484   2.704235e-03
     1         1.000           587.560        80        0      2.082975   1.223873e-03
     1         1.000           656.270        80        0      1.098614   6.455015e-04
     2         2.000           486.130        80        0      4.896489   2.876981e-03
     2         2.000           587.560        80        0      2.478862   1.456480e-03
     2         2.000           656.270        80        0      1.613832   9.482230e-04
  Versagte Strahlen: 0 von 510
```

Die Tabelle hat eine Zeile pro Feld und Wellenlänge in der Reihenfolge der Ablage `maps[f · L + l]`. `Strahlen` ist die Zahl der Pupillenstützstellen, `versagt` die Zahl der versagten Strahlen (an einer Fläche oder an der Referenzsphäre); RMS über die gültigen Strahlen, `-`, wenn kein Strahl gültig ist. Die Bildebene der Testlinse liegt im paraxialen Fokus; die RMS-Werte von einigen Wellen sind die sphärische Aberration und der Farblängsfehler der unkorrigierten Einzellinse bei f/5.

### JSON-Ausgabe (`--json`), Format `mikroptim.trace/1`

Ein JSON-Objekt (eingerückt mit zwei Leerzeichen, abschließender Zeilenumbruch). Zahlen werden verlustfrei geschrieben (kürzeste Darstellung, die beim Einlesen denselben FP64-Wert ergibt), sind also bitgleich zu den Werten der Bibliotheksfunktionen. **Nicht endliche Werte (±∞, NaN) sind in JSON nicht darstellbar und werden `null`**; ob ein `null` „unendlich“ oder „nicht definiert“ bedeutet, ergibt sich aus den Flags (`afocal`, `exit_pupil.at_infinity`) und den Definitionen in `ParaxialData`. Längen in mm, Wellenlängen in nm.

| Feld | Typ | Inhalt |
|---|---|---|
| `format` | string | `"mikroptim.trace/1"` |
| `mikroptim_version` | string | Programmversion (`mko::kVersion`) |
| `system` | string | `name` der Systembeschreibung |
| `file` | string | Pfad der Systemdatei wie auf der Kommandozeile angegeben |
| `reference_nm` | number | Referenzwellenlänge λ_ref |
| `wavelengths_nm` | number[] | Wellenlängen des Spektrums, Index `l` |
| `source` | object | `{"type": "collimated", "field_angles_deg": [...]}` oder `{"type": "point", "point": [x, y, z], "numerical_aperture": NA}` |
| `paraxial` | object | Paraxialdaten bei λ_ref, siehe unten |
| `quadrature` | object | `{"rings": N, "spokes": M}` wie angefordert |
| `backend` | string | Rechenwerk der Wellenfront: `"cpu"` oder `"gpu"` (seit M2-T5; die Paraxialdaten entstehen immer auf dem Host) |
| `device` | object \| null | bei `"gpu"`: `{"index": N, "name": "<Gerätename>"}`, sonst `null` (seit M2-T5) |
| `precision` | string | Rechentyp der Wellenfront: `"fp64"` oder `"df64"` (nur mit `--gpu`); kompatible Erweiterung nach M7-T1 |
| `target_focal_length` | number \| null | verwendete Sollbrennweite f_soll für die Bildpunkte (`target.focal_length` oder paraxial); `null` bei der Punktquelle oder ohne Wellenfront |
| `wavefront` | object[] \| null | ein Eintrag pro Feld f und Wellenlänge l, Reihenfolge `f · L + l`; `null`, wenn nicht auswertbar |
| `wavefront_error` | string \| null | Grund, warum `wavefront` `null` ist |
| `rays` | integer \| null | Summe der Strahlen über alle Einträge |
| `failed_rays` | integer \| null | Summe der versagten Strahlen über alle Einträge |
| `warnings` | string[] | Warnungen des Laders |

`paraxial` (Bedeutung wie in `mko::ParaxialData`, [architektur.md](architektur.md#paraxialoptik)):

| Feld | Einheit | Inhalt |
|---|---|---|
| `wavelength_nm` | nm | Wellenlänge der Paraxialrechnung (= `reference_nm`) |
| `afocal` | bool | Brechkraft exakt 0 |
| `power` | 1/mm | Brechkraft φ |
| `focal_length` | mm | Brennweite 1/φ (`null` bei afokal) |
| `back_focal_distance` | mm | letzte Fläche → F′ |
| `front_focal_distance` | mm | Fläche 0 → F |
| `front_principal_plane` | mm | H relativ zu Fläche 0 |
| `rear_principal_plane` | mm | H′ relativ zur letzten Fläche |
| `image_defocus` | mm | Schnittweite − Dicke der letzten Fläche (> 0: Fokus hinter der Bildebene) |
| `petzval_sum` | 1/mm | Petzvalsumme P |
| `petzval_curvature` | 1/mm | κ_P = −n′ P |
| `petzval_radius` | mm | 1/κ_P (`null` bei κ_P = 0) |
| `exit_pupil.at_infinity` | bool | Austrittspupille im Unendlichen (bildseitig telezentrisch) |
| `exit_pupil.position` | mm | Lage relativ zur letzten Fläche |
| `exit_pupil.magnification` | – | Pupillenvergrößerung 1/D |
| `exit_pupil.semi_diameter` | mm | Radius der Austrittspupille |
| `abcd` | – | Systemmatrix `[A, B, C, D]` in reduzierten Winkeln |

Einträge von `wavefront`:

| Feld | Typ | Inhalt |
|---|---|---|
| `field` | integer | Feldindex f |
| `field_angle_deg` | number \| null | Feldwinkel θ_f (kollimiert), `null` bei der Punktquelle |
| `wavelength` | integer | Wellenlängenindex l |
| `wavelength_nm` | number | Wellenlänge |
| `image_point` | number[3] | Mittelpunkt **M**_f der Referenzsphäre, lokale Koordinaten der letzten Fläche [mm] |
| `pupil_mode` | string | `"axis"`, `"half"` oder `"full"` |
| `rays` | integer | Pupillenstützstellen |
| `failed_rays` | integer | versagte Strahlen |
| `rms_waves` | number \| null | RMS-Wellenfront in Wellen bei λ_ref (`null` ohne gültigen Strahl) |
| `rms_mm` | number \| null | RMS-Wellenfront [mm] |
| `piston_mm` | number \| null | Kolben Σ W·OPL / Σ W [mm]; mit `--gpu` immer `null` (der Kernel `RESIDUALS` gibt den Kolben nicht aus) |

**Stabilität:** Neue Felder können in `mikroptim.trace/1` hinzukommen; Leser sollen unbekannte Felder ignorieren. Umbenennen, Entfernen oder Bedeutungsänderungen erhöhen die Version auf `mikroptim.trace/2`.

Beispiel (`mikroptim trace data/systems/einzellinse_bk7.json --json`):

<details>
<summary>Vollständige Ausgabe</summary>

```json
{
  "format": "mikroptim.trace/1",
  "mikroptim_version": "0.1.0",
  "system": "Einzellinse N-BK7",
  "file": "data/systems/einzellinse_bk7.json",
  "reference_nm": 587.56,
  "wavelengths_nm": [
    486.13,
    587.56,
    656.27
  ],
  "source": {
    "type": "collimated",
    "field_angles_deg": [
      0.0,
      1.0,
      2.0
    ]
  },
  "paraxial": {
    "wavelength_nm": 587.56,
    "afocal": false,
    "power": 0.009896122289808292,
    "focal_length": 101.04968094724019,
    "back_focal_distance": 99.05208836399525,
    "front_focal_distance": -94.05208836399525,
    "front_principal_plane": 6.9975925832449395,
    "rear_principal_plane": -1.9975925832449377,
    "image_defocus": 0.002088363995255804,
    "petzval_sum": 0.006589473496335327,
    "petzval_curvature": -0.006589473496335327,
    "petzval_radius": -151.75719282521442,
    "exit_pupil": {
      "at_infinity": false,
      "position": -9.515814775176269,
      "magnification": 1.074401246213303,
      "semi_diameter": 10.74401246213303
    },
    "abcd": [
      0.980231579510994,
      8.856853813893544,
      -0.009896122289808292,
      0.9307509680619526
    ]
  },
  "quadrature": {
    "rings": 10,
    "spokes": 8
  },
  "backend": "cpu",
  "device": null,
  "precision": "fp64",
  "target_focal_length": 101.04968094724019,
  "wavefront": [
    {
      "field": 0,
      "field_angle_deg": 0.0,
      "wavelength": 0,
      "wavelength_nm": 486.13,
      "image_point": [
        0.0,
        0.0,
        99.05
      ],
      "pupil_mode": "axis",
      "rays": 10,
      "failed_rays": 0,
      "rms_waves": 4.502092950462526,
      "rms_mm": 0.0026452497339737612,
      "piston_mm": 14.130347293916047
    },
    {
      "field": 0,
      "field_angle_deg": 0.0,
      "wavelength": 1,
      "wavelength_nm": 587.56,
      "image_point": [
        0.0,
        0.0,
        99.05
      ],
      "pupil_mode": "axis",
      "rays": 10,
      "failed_rays": 0,
      "rms_waves": 1.9360135431732859,
      "rms_mm": 0.0011375241174268957,
      "piston_mm": 14.099531981210557
    },
    {
      "field": 0,
      "field_angle_deg": 0.0,
      "wavelength": 2,
      "wavelength_nm": 656.27,
      "image_point": [
        0.0,
        0.0,
        99.05
      ],
      "pupil_mode": "axis",
      "rays": 10,
      "failed_rays": 0,
      "rms_waves": 0.8660353587622437,
      "rms_mm": 0.0005088477353943438,
      "piston_mm": 14.085840661889346
    },
    {
      "field": 1,
      "field_angle_deg": 1.0,
      "wavelength": 0,
      "wavelength_nm": 486.13,
      "image_point": [
        0.0,
        1.763828741909749,
        99.05
      ],
      "pupil_mode": "half",
      "rays": 80,
      "failed_rays": 0,
      "rms_waves": 4.602483961671361,
      "rms_mm": 0.0027042354765196243,
      "piston_mm": 14.128888270713023
    },
    {
      "field": 1,
      "field_angle_deg": 1.0,
      "wavelength": 1,
      "wavelength_nm": 587.56,
      "image_point": [
        0.0,
        1.763828741909749,
        99.05
      ],
      "pupil_mode": "half",
      "rays": 80,
      "failed_rays": 0,
      "rms_waves": 2.082974772167356,
      "rms_mm": 0.0012238726571346514,
      "piston_mm": 14.098071862450222
    },
    {
      "field": 1,
      "field_angle_deg": 1.0,
      "wavelength": 2,
      "wavelength_nm": 656.27,
      "image_point": [
        0.0,
        1.763828741909749,
        99.05
      ],
      "pupil_mode": "half",
      "rays": 80,
      "failed_rays": 0,
      "rms_waves": 1.0986136789180152,
      "rms_mm": 0.0006455014531850688,
      "piston_mm": 14.08438005235029
    },
    {
      "field": 2,
      "field_angle_deg": 2.0,
      "wavelength": 0,
      "wavelength_nm": 486.13,
      "image_point": [
        0.0,
        3.5287326155732273,
        99.05
      ],
      "pupil_mode": "half",
      "rays": 80,
      "failed_rays": 0,
      "rms_waves": 4.896489155733337,
      "rms_mm": 0.002876981168342679,
      "piston_mm": 14.124511478829488
    },
    {
      "field": 2,
      "field_angle_deg": 2.0,
      "wavelength": 1,
      "wavelength_nm": 587.56,
      "image_point": [
        0.0,
        3.5287326155732273,
        99.05
      ],
      "pupil_mode": "half",
      "rays": 80,
      "failed_rays": 0,
      "rms_waves": 2.478862452445385,
      "rms_mm": 0.0014564804225588102,
      "piston_mm": 14.093691785779615
    },
    {
      "field": 2,
      "field_angle_deg": 2.0,
      "wavelength": 2,
      "wavelength_nm": 656.27,
      "image_point": [
        0.0,
        3.5287326155732273,
        99.05
      ],
      "pupil_mode": "half",
      "rays": 80,
      "failed_rays": 0,
      "rms_waves": 1.6138317561135485,
      "rms_mm": 0.0009482229866220763,
      "piston_mm": 14.079998504173323
    }
  ],
  "wavefront_error": null,
  "rays": 510,
  "failed_rays": 0,
  "warnings": []
}
```

</details>

Für `planplatte_bk7.json` (afokal) sind in `paraxial` die Felder `focal_length`, `back_focal_distance`, `front_focal_distance`, `front_principal_plane`, `rear_principal_plane`, `image_defocus` und `petzval_radius` `null`, `afocal` ist `true`, und es gilt `"wavefront": null` mit `"wavefront_error": "Wellenfront: Sollbrennweite ist nicht endlich (afokales System?); target.focal_length setzen"`.

### GPU-Backend (`--gpu`)

Mit `--gpu` entsteht die Wellenfront über die Host-API `mko::cuda::TraceEngine` ([architektur.md](architektur.md), Abschnitt „Host-API TraceEngine“) auf dem Gerät `--device N` (Standard 0):

1. Sollbrennweite f_soll wie im CPU-Pfad mit `target_focal_length` (nur kollimierte Quelle), dann `make_system_batch(system, index, 1, f_soll)` – eine Variante mit den Feldzielen h_f = f_soll · tan θ_f, also denselben idealen Bildpunkten wie `evaluate_wavefront`.
2. `TraceEngine::run(batch, TraceMode::Residuals, {quadrature = {N, M, FailureMode::Hard}, opd = true, status = true})` – hartes Versagen wie `evaluate_wavefront` (das weiche Versagen der Optimierung, M4-T2, würde versagte Strahlen fortsetzen und den Kolben ändern).
3. `tuple_wavefronts(out.residuals, out.layout, 0)` liefert RMS [mm] und versagte Strahlen pro Feld und Wellenlänge; RMS in Wellen = RMS / λ_ref wie im CPU-Pfad.

Paraxialdaten, Sollbrennweite und Bildpunkte entstehen auf dem Host und sind bitgleich zur CPU-Variante; Text und JSON werden von denselben Ausgabefunktionen geschrieben. Unterschiede zur CPU-Variante:

- `rms_waves`, `rms_mm`: Abweichung höchstens max(`kCpuGpuFp64`, `kCpuGpuFp64Relative` · |RMS|) (`tests/tolerances.hpp`; FMA-Kontraktion und CUDA-Mathematikbibliothek). Gemessen auf der RTX 3090: höchstens 7,7·10⁻¹⁵ mm (relativ ≤ 4,8·10⁻¹²). Die Textausgabe ist bei 6 Stellen deshalb praktisch immer identisch; sie unterscheidet sich nur in der Zeile `Backend`.
- `failed_rays` und `rays` sind identisch (gleiches Versagenskriterium im header-only Tracer).
- `piston_mm` ist `null`: Der Kernel `RESIDUALS` zieht den Kolben ab, gibt ihn aber nicht aus.
- `backend` ist `"gpu"`, `device` nennt Index und Namen des Geräts.

**df64** (`--gpu --precision df64`): Die Kernel rechnen im Zahlentyp df64 (numerik.md, Abschnitt 12), Batch und Ausgaben bleiben FP64. RMS-Werte weichen von der CPU höchstens um max(`kDf64VsFp64`, `kDf64VsFp64Relative` · |RMS|) ab (10⁻⁹ mm bzw. 10⁻¹⁰ relativ); Strahlzahlen und versagte Strahlen sind in den Testsystemen identisch (an der Versagensgrenze kann der Status in Ausnahmefällen kippen). Im JSON ist `precision` `"df64"`, die Zeile `Backend` endet auf `df64`. Test: `CliTraceGpu.Df64MatchesCpuWithinDf64Tolerance`.

Afokale Systeme ohne `target.focal_length` verhalten sich wie im CPU-Pfad (`wavefront: null`, gleiche Meldung in `wavefront_error`, Exit-Code 0); die Punktquelle (Achse, y-z-Ebene, allgemein) wird unterstützt. Ohne CUDA gebaut oder ohne sichtbares Gerät endet `--gpu` mit einer Meldung auf stderr und Exit-Code 3, z. B.

```
mikroptim trace: --gpu: ohne CUDA gebaut (MIKROPTIM_ENABLE_CUDA=OFF), GPU-Backend nicht verfügbar
mikroptim trace: --gpu: kein CUDA-Gerät gefunden (GPU-Backend nicht verfügbar)
```

Beispiel (`mikroptim trace data/systems/einzellinse_bk7.json --gpu`, RTX 3090): Ausgabe wie oben, nur die Zeile `Backend` lautet

```
Wellenfront (Quadratur 10 Ringe x 8 Speichen, RMS in Wellen bei 587.560 nm)
  Backend                                     GPU 0 (NVIDIA GeForce RTX 3090), FP64
  Sollbrennweite für die Bildpunkte           101.049681 mm
```

Mit `--json` ändern sich gegenüber dem Beispiel oben nur

```json
  "backend": "gpu",
  "device": {
    "index": 0,
    "name": "NVIDIA GeForce RTX 3090"
  },
  "precision": "fp64",
```

sowie in den Einträgen von `wavefront` die RMS-Werte in den letzten Stellen und der Kolben, z. B. für Feld 0 bei 486,13 nm `"rms_waves": 4.5020929504633`, `"rms_mm": 0.002645249733974216`, `"piston_mm": null` (CPU: `4.502092950462526`, `0.0026452497339737612`, `14.130347293916047`).

### Tests

`tests/test_cli_trace.cpp` ruft `mko::cli::run_trace_command` direkt auf und prüft die JSON-Ausgabe gegen `compute_paraxial` und `evaluate_wavefront` (bitgleich), das afokale System, die Punktquelle, Warnungen, Fehlerfälle und die Textausgabe. Die ctest-Kommandos `cli_trace_json`, `cli_trace_text` und `cli_trace_missing_file` starten das gebaute Programm.

`--gpu`: Im CPU-Build prüfen `CliTrace.GpuUnavailableWithoutCuda` und das ctest-Kommando `cli_trace_gpu_unavailable` Meldung und Exit-Code 3 (läuft in der CI). Im CUDA-Build vergleichen die Tests `CliTraceGpu.*` (übersprungen ohne Gerät) GPU- gegen CPU-Ausgabe für `einzellinse_bk7.json`, `testsystem_mehrlinser.json` (auch mit 16 × 12 Stützstellen), ein System mit versagten Strahlen, die Punktquelle und das afokale System; dazu die ctest-Kommandos `cli_trace_gpu_json` (Exit-Code 3 ohne Gerät = übersprungen, `SKIP_RETURN_CODE`) und `cli_trace_gpu_no_device` (`CUDA_VISIBLE_DEVICES=-1`).

Implementierung: `src/cli/trace_command.cpp` (Bibliothek `mikroptim_cli`, interner Header `src/cli/trace_command.hpp`; im CUDA-Build gelinkt gegen `mikroptim_cuda`, GPU-Pfad unter `#ifdef MIKROPTIM_HAS_CUDA`), `src/cli/main.cpp` verteilt nur die Unterbefehle.

## `mikroptim optimize`

Optimiert die Variablen einer Systemdatei (Abschnitt `variables`, [dateiformat.md](dateiformat.md)) mit dem gebündelten Levenberg-Marquardt ([optimierung.md, Abschnitt 4](optimierung.md#4-gebündelter-levenberg-marquardt)) und schreibt das Ergebnis als neue Systemdatei. Ausgegeben werden ein Iterationsprotokoll und eine Zusammenfassung: Status, Zyklen, Merit vorher/nachher, Variablen vorher/nachher mit Schranken, Nebenbedingungen Ist/Soll, bei Systemen mit Geisterplatten der Diagnosebericht (ab M4-T6, siehe [Geisterplatten](#geisterplatten-diagnosebericht-ab-m4-t6)) und RMS-Wellenfront pro Feld und Wellenlänge vorher/nachher.

Die Merit ist dieselbe wie in der Bibliothek (`optimize_lm_cpu` bzw. `cuda::optimize_lm_gpu`): F = ‖r‖² aus den Strahlresiduen der Wellenfront (Pupillenquadratur `--rings` × `--spokes`), beim weichen Strahlversagen den Strafresiduen der versagten Strahlen (ab M4-T2, siehe [Strahlversagen](#strahlversagen-ab-m4-t2)) und den Zusatzresiduen: paraxiale Nebenbedingungen (Abschnitt `constraints`, `make_paraxial_residuals(make_paraxial_constraints(system))`), dahinter die L1-Strafe der Geisterkoeffizienten (Einträge `ghost_penalty`, `make_ghost_penalty_residuals(make_ghost_penalty(system))`, ab M4-T6), die Strafe außerhalb der Glas-Hülle (Eintrag `glass_hull`, `make_glass_hull_residuals(make_glass_hull_penalty(system, set, make_glass_hull(catalog)))`, ab M6-T1) und die mechanischen Bedingungen ohne Strahlbezug (ab M4-T3, `make_mechanical_residuals(make_mechanical_constraints(system)).geometric`), verkettet mit `concat_extra_residuals`, ganz hinten die mechanischen Bedingungen aus den Strahlhöhen und Einfallswinkeln (trace-abhängige Zusatzresiduen, `LmOptions::jacobian.trace_extra`) ([optimierung.md, Abschnitte 2 und 3](optimierung.md#2-residuenvektor)). Die Einheit ist die der Bibliothek: seit M4-T1 **Wellen² bei λ_ref** (Strahlresiduen √(ŵ_f ŵ_λ W_i V_j)·OPD/λ_ref, paraxiale Zusatzresiduen √w·q mit w in 1/mm²; Wahl der Gewichte: [optimierung.md, Abschnitt 2.2](optimierung.md#22-paraxiale-nebenbedingungen-ab-m3-t3)). Text und JSON nennen sie (`[Wellen²]`, `merit_unit`). Gerechnet wird in FP64.

### Optionen

| Option | Bedeutung | Standard |
|---|---|---|
| `<system.json>` | Systemdatei mit Abschnitt `variables` (optional `constraints`, Geisterkoeffizienten als Variablen `ghost_coefficient`) | – |
| `-o DATEI`, `--output DATEI` | Ergebnis als neue Systemdatei schreiben (siehe [Ergebnisdatei](#ergebnisdatei)); ohne `-o` nur Protokoll | keine Datei |
| `--json` | maschinenlesbare Ausgabe `mikroptim.optimize/1` statt Text | aus |
| `--quiet`, `-q` | ohne Iterationsprotokoll (Text: nur Kopf und Zusammenfassung, JSON: `"iterations": null`) | aus |
| `--max-iterations N` | Zyklen pro Lauf (Versuchsschritte einschließlich verworfener, `LmOptions::max_iterations`), N ≥ 0; 0 wertet nur den Startpunkt aus | 100 |
| `--stall W[,T]` | Abbruch bei Stagnation (`converged_stall`): F sank in W Zyklen um höchstens T·F (T Standard 10⁻³; M11-T1c) | 0 (aus) |
| `--rings N`, `--spokes M` | Pupillenquadratur der Strahlresiduen wie bei `trace` | 10, 8 |
| `--runs K` | Multistart mit K ≥ 1 Läufen (siehe unten) | 1 |
| `--perturbation s` | relative Störung der Startpunkte, s ≥ 0; nur mit `--runs` ≥ 2 | 0,05 |
| `--seed n` | Startwert der Störungen (0 … 2⁶⁴ − 1); nur mit `--runs` ≥ 2 | 0 |
| `--fail-weight w` | Start-Strafgewicht w_fail des weichen Strahlversagens in Wellen², endlich und > 0 (`ResidualSettings::fail_weight`, siehe [Strahlversagen](#strahlversagen-ab-m4-t2)); nicht mit `--hard-failure` | 10⁶ (`kDefaultFailWeight`) |
| `--hard-failure` | hartes Strahlversagen wie vor M4-T2: versagte Strahlen maskiert (r = 0), keine Strafresiduen, kein Strafverfahren (`FailureMode::Hard`) | aus (weich) |
| `--jacobian V` | Verfahren der Jacobi-Matrix (ab M5-T3, `LmOptions::jacobian_method`): `fd` zentrale finite Differenzen, `adjoint` adjungiert (Strahl- und Strafzeilen aus dem Rückwärtsdurchlauf; Zusatzzeilen weiter per zentraler FD), [optimierung.md, „Jacobi-Verfahren im LM“](optimierung.md#jacobi-verfahren-im-lm-ab-m5-t3) | `fd` |
| `--gpu` | auf der GPU rechnen (`cuda::optimize_lm_gpu`: Jacobi-Matrix und Normalgleichungen auf dem Gerät, Merit im Modus MERIT der TraceEngine) | aus (CPU, `optimize_lm_cpu`) |
| `--device N` | CUDA-Gerät für `--gpu`; nur zusammen mit `--gpu` | 0 |
| `--help`, `-h` | Hilfe | – |

Die übrigen Einstellungen des Verfahrens (Toleranzen, Dämpfung, zentrales FD-Schema, Ablehnung neuer hart versagter Strahlen, Faktor 10 und Obergrenze 10¹⁴ des Strafverfahrens) sind die Standardwerte von `LmOptions`. Die Zahl der Host-Threads wählt die Bibliothek; das Ergebnis ist davon unabhängig bitgleich.

**Strahlversagen (ab M4-T2).** <a id="strahlversagen-ab-m4-t2"></a>Standard ist weiches Versagen ([numerik.md, Abschnitt 11](numerik.md#11-weiches-versagen), [optimierung.md, Abschnitt 2](optimierung.md#2-residuenvektor)): Verfehlte Flächen und Totalreflexion werden fortgesetzt, jeder Strahl trägt ein Strafresiduum p = √(w_fail ŵ_f ŵ_λ W)·π bei, F enthält Σp². Endet ein Lauf mit versagten Strahlen, startet die Bibliothek ihn an seinem Endpunkt mit zehnfachem w_fail neu (äußeres Strafverfahren, `optimize_lm_rounds`), bis kein Strahl mehr versagt oder w_fail 10¹⁴ überschritte; `--max-iterations` gilt pro Runde, Zyklen und Protokoll laufen über die Runden fort. Für Systeme ohne versagte Strahlen sind alle Strafresiduen null und die Ergebnisse bitgleich zu `--hard-failure`. Die Spalte `versagt` zählt weich und hart versagte Strahlen, `hart` nur die hart versagten (Strahl läuft rückwärts, Fläche verfehlt bei D ≥ 0, nicht unterstützt); nur diese lehnt LM an einem Versuchspunkt als neu versagt ab. Die Wellenfronttabelle rechnet immer hart wie `mikroptim trace`.

**Multistart.** Mit `--runs K` laufen K Läufe gemeinsam: Lauf 0 startet im Punkt x₀ der Datei, Lauf j ≥ 1 im gestörten Punkt `perturbed_vector(set, x₀, options, j)` mit `relative = s` für alle Variablentypen (Geisterkoeffizienten a = 0 bleiben ungestört), `seed = n` und `first_is_base` (gleichverteilt, plattformunabhängig, auf die Schranken projiziert; [optimierung.md, Variantengenerator](optimierung.md#variantengenerator-ab-m3-t1)). Lauf 0 ist bitgleich zum Aufruf ohne `--runs` (die Läufe sind voneinander unabhängig), der Multistart ist also nie schlechter. **Bester Lauf:** endliche Merit, dann die wenigsten versagten Strahlen (hart versagte Strahlen haben r = 0, beim weichen Versagen enthält F Strafen mit dem w_fail der letzten Runde des Laufs; in beiden Fällen ist eine Merit mit mehr versagten Strahlen nicht vergleichbar), dann die kleinste Merit, bei Gleichstand der kleinste Index. Die Auswahl ist deterministisch; zwei Aufrufe mit gleichen Argumenten liefern dieselbe Ausgabe.

**Glasvariablen** (ab M6-T1, [optimierung.md, Abschnitte 1 und 2.4](optimierung.md#24-glas-hülle-ab-m6-t1)): Hat das System Variablen `glass`, wandelt der Befehl nach dem Laden die Materialien dieser Flächen in Buchdahl-Materialien um (`convert_glass_variables`: Katalogglas → 3-Term-Fit über 400–700 nm, ν₃ fest) und baut Brechzahltabelle und Batch neu; Startpunkt, Wellenfront „vorher“ und Merit beziehen sich auf das umgewandelte System (beim Fit von N-BK7 ändern sich die Brechzahlen um höchstens 4,3·10⁻⁶). Mit einem Eintrag `glass_hull` entsteht die Hülle aus allen Gläsern aller `glass_catalogs` der Systemdatei, deren Gültigkeitsbereich 400–700 nm abdeckt (eigene Kataloge: [data/glass/README.md](../data/glass/README.md)). Im Multistart werden die Glaskoordinaten wie die übrigen Variablen relativ gestört (`--perturbation`). Die Strafe erreicht den Rand nur asymptotisch (Abstand ∝ 1/w). Mit großem Gewicht direkt vom Start kommt LM zwar an den Rand, bleibt dort aber leicht vor dem beschränkten Optimum stehen (Beispiel: `mikroptim optimize data/systems/einzellinse_glasvariable.json --max-iterations 200 --rings 6 --spokes 4` mit w = 10⁸: Überschreitung am Ende 1,5·10⁻⁸ normiert, Status `max_iterations` bei n₀ = 1,5205, ν₁ = −0,0439; die zentralen Differenzen überspannen den Knick der Strafe am Rand wie bei Petzval). Das Optimum auf dem Rand (dort die Ecke CaF₂) findet die Penalty-Fortsetzung mit stufenweise wachsendem Gewicht (optimierung.md, Abschnitt 2.4; später der Homotopie-Controller).

### Ergebnisdatei

Die Ergebnisdatei ist die Eingabe mit den Variablenwerten des besten Laufs (`VariableSet::apply_to_description`, danach `write_system_file`: Zahlen verlustfrei; Name, Beschreibung, Variablen, Nebenbedingungen und Kommentare unverändert). Sie lässt sich mit `mikroptim trace` und `mikroptim optimize` laden und liefert dort **bitgleich dieselbe Merit** wie die Endmerit der Optimierung, sofern dasselbe Backend rechnet: Die Datei enthält x exakt (kürzeste Darstellung, die beim Einlesen denselben FP64-Wert ergibt), der Lader baut daraus denselben Batch wie `lm_point_batch` (auch die Dicke eines Luftspalts mit Geisterplatten, letztes Teilstück fl(x − d)), und die Merit ist eine deterministische Funktion des Batches. Prüfen lässt sich das mit `mikroptim optimize <ergebnis.json> --max-iterations 0`. CPU und GPU unterscheiden sich um die Rundung der Strahlresiduen: Weicht die OPD jedes Strahls um höchstens δ = `kCpuGpuFp64` ab, gilt ‖Δr‖₂ ≤ N·δ mit N = √(Σ_{f,l} s_{f,l}²) (`residual_scale_norm`) und damit |ΔF| ≤ 2√F·Nδ + (Nδ)².

- **Glaskataloge:** Ein Eintrag von `glass_catalogs` bleibt unverändert, wenn er vom Verzeichnis der Ergebnisdatei aus (mit der Suche in den Elternverzeichnissen, [dateiformat.md](dateiformat.md)) dieselbe Datei findet; sonst wird er durch den relativen Pfad dorthin ersetzt (bzw. den absoluten, wenn es keinen relativen gibt, z. B. anderes Laufwerk).
- **Eingabedatei:** Ist `-o` die Eingabedatei (auch über einen anderen Pfad oder eine Verknüpfung), bricht der Befehl vor dem Laden mit Exit-Code 2 ab; die Eingabe wird nie überschrieben, eine Option zum Erzwingen gibt es bewusst nicht. Eine andere vorhandene Datei wird überschrieben.
- **Glas:** Das Material jeder Glasvariable steht als `{"model": "buchdahl", "n0": …, "nu": [ν₁, ν₂, ν₃]}` in der Ergebnisdatei (auch wenn die Eingabe ein Katalogglas nannte); ν₃ ist der des Startglases. Den Namen eines Katalogglases ordnet erst das Glas-Einrasten wieder zu ([`mikroptim snap`](#mikroptim-snap), M6-T5).
- **Feldziele:** Die idealen Bildpunkte h_f = f_soll · tan θ_f bleiben während der Optimierung fest (Batch des Startsystems). Ohne `target.focal_length` ist f_soll die paraxiale Brennweite des *Startsystems*; beim Laden des Ergebnisses ergeben sich dann andere Feldziele und, bei Feldwinkeln ≠ 0, eine andere Merit. Der Befehl warnt in diesem Fall (`mikroptim optimize: Warnung: Feldziele aus der paraxialen Brennweite des Startsystems …`, auch im JSON-Feld `warnings`); für reproduzierbare Ergebnisse `target.focal_length` setzen.

### Exit-Codes und Fehler

Wie bei `trace`:

| Code | Bedeutung |
|---|---|
| 0 | Optimierung abgeschlossen, auch ohne Konvergenz (Status `max_iterations`, `damping_limit`; siehe `result.status`) |
| 1 | Fehler beim Laden (wie `trace`), **keine Variablen** (Abschnitt `variables` fehlt oder ist leer), Nebenbedingung passt nicht zum System, Merit in keinem Lauf endlich, Fehler beim Schreiben der Ergebnisdatei (z. B. Verzeichnis fehlt); mit `--gpu` auch: Gerät `--device N` existiert nicht, CUDA-Fehler |
| 2 | ungültige Kommandozeile (wie `trace`; zusätzlich `-o` ohne Datei oder gleich der Eingabedatei, `--runs` < 1, `--max-iterations` < 0, `--perturbation` < 0, `--perturbation`/`--seed` ohne `--runs` ≥ 2, `--fail-weight` nicht endlich oder ≤ 0 oder zusammen mit `--hard-failure`) |
| 3 | `--gpu`, aber ohne CUDA gebaut oder kein CUDA-Gerät (vor dem Laden geprüft) |

Fehlermeldungen beginnen mit `mikroptim optimize: Fehler: `, z. B.

```
mikroptim optimize: Fehler: data/systems/einzellinse_bk7.json: keine Variablen (Abschnitt variables fehlt oder ist leer)
```

Bei Exit-Code 1 bleibt die Standardausgabe leer, außer wenn der Fehler erst nach dem Start der Optimierung auftritt (Textausgabe: Kopf und Protokoll stehen dann schon dort). Die Ergebnisdatei wird nur nach erfolgreicher Optimierung geschrieben.

### Textausgabe

Zahlenformat wie bei `trace` (C-Locale, plattformunabhängig; Linux/GCC und Windows/MSVC liefern dieselbe Ausgabe): Merit mit 9 Stellen in Exponentialschreibweise, Krümmungen in 1/mm und Geisterkoeffizienten in mm ebenso, Längen in mm mit 6 Nachkommastellen, nicht endliche Werte als `unendlich`/`-unendlich` bzw. `-`. Das Protokoll wird während der Rechnung geschrieben, eine Zeile pro Zyklus und Lauf (in Laufreihenfolge, `LmOptions::on_iteration`):

| Spalte | Inhalt (`LmIterationInfo`) |
|---|---|
| `Lauf`, `Zyklus` | Index des Laufs, Zyklen des Laufs nach diesem |
| `Schritt` | `angenommen`, `verworfen` oder `gescheitert` (Cholesky-Zerlegung gescheitert, kein Versuchspunkt) |
| `Merit` | F nach dem Zyklus (bei Ablehnung unverändert) |
| `Testmerit` | F am Versuchspunkt |
| `Gewinn` | Gewinnverhältnis ρ = (F − F_neu)/(L(0) − L(Δx)) |
| `Dämpfung` | μ, mit dem der Schritt gerechnet wurde |
| `Schrittnorm` | max_k \|Δx_k\|/s_k des Versuchsschritts |
| `versagt` | versagte Strahlen am Versuchspunkt (weich und hart) |
| `hart` | davon hart versagte Strahlen (`LmIterationInfo::hard_failed_rays`, ab M4-T2) |
| `Status` | Endstatus, wenn der Lauf in diesem Zyklus endet; das meldet die Bibliothek nur für `damping_limit` und `converged_merit` bei F = 0. Die übrigen Endzustände (`converged_*`, `max_iterations`) stellt sie erst vor dem nächsten Zyklus fest, ohne Rückruf; sie stehen in der Zusammenfassung |

Der Kopf nennt die Versagensart (`Strahlversagen  weich, w_fail ab …` bzw. `hart (versagte Strahlen maskiert)`), bei Systemen mit Geisterplatten zusätzlich die Zeile `Geisterplatten  G (L1-Strafe: n Koeffizienten)`. Die Zusammenfassung enthält neben den versagten Strahlen vorher/nachher die hart versagten (`davon hart vorher/nachher`) und im weichen Modus die Runden des Strafverfahrens und das w_fail der letzten Runde; die Tabelle der Läufe (Multistart) hat dafür die Spalten `hart` und `Runden`.

Mit Glasvariablen (ab M6-T1) haben n₀, ν₁, ν₂ die Einheit `1` und stehen in Exponentialschreibweise mit 9 Stellen. Nach den Nebenbedingungen folgen die Abschnitte `Glasvariablen` (pro Glasfläche das Startglas, bei Katalogglas der maximale Fitfehler, und das feste ν₃) und, mit Eintrag `glass_hull`, `Glas-Hülle` (Zahl der Gläser, Dimension, Ecken, Gewicht, Sicherheitsabstand; pro Glasfläche die Überschreitung max(0, φ) vorher und nachher in normierten Koordinaten):

```text
Glasvariablen (Startglas als Buchdahl-Material, n0 und nu1, nu2 variabel)
  Fläche 1: N-BK7 (Fit, max. Fehler 5.51e-06), nu3 = -5.754123e-04 fest

Glas-Hülle (39 Gläser, Dimension 3, 17 Ecken; Abstand normiert, Gewicht 1.000e+08, Sicherheitsabstand 0.000e+00)
  Fläche 1: Überschreitung vorher 0.000e+00, nachher 1.509e-08
```

Die Wellenfronttabelle der Zusammenfassung rechnet immer die CPU (`evaluate_wavefront` mit den idealen Bildpunkten, wie `mikroptim trace` ohne `--gpu`) für das Start- und das Ergebnissystem; „nachher“ ist also genau das, was `mikroptim trace <ergebnis.json>` ausgibt.

### Mechanische Nebenbedingungen (ab M4-T3)

Enthält die Systemdatei mechanische Bedingungen ([dateiformat.md](dateiformat.md#nebenbedingungen-ab-m3-t3), [optimierung.md, Abschnitt 2.5](optimierung.md#25-mechanische-nebenbedingungen-ab-m4-t3)), nennt der Kopf ihre Zahl (`Nebenbedingungen (mechanisch)`, eine pro Fläche bzw. Spalt), und die Zusammenfassung zeigt nach den paraxialen Bedingungen eine Tabelle Ist/Grenze vorher und nachher: Eintrag (`Nr` in `constraints.items`), Typ, Fläche (bei Dicken der Spalt `i-(i+1)`), Gewicht, Grenze, Ist vorher/nachher (mm; beim Einfallswinkel Grad), Residuum r = √w·max(0, g) vorher/nachher und `verletzt`, wenn g > 0 am Ergebnis. Die Extrema (Strahlhöhen, Einfallswinkel) rechnet die CPU mit `surface_extrema_cpu` und derselben Quadratur wie die Optimierung, auch mit `--gpu`. Eine Strafe erreicht die Grenze nur asymptotisch; eine Restverletzung im Bereich von 10⁻⁵ mm ist mit der Penalty-Fortsetzung der Normalfall (optimierung.md, Abschnitt 2.5). Beispiel (`mikroptim optimize data/systems/zweilinser_mechanik.json --quiet --max-iterations 300`, gekürzt; direkt mit den Gewichten der Datei, ohne Fortsetzung, endet dieser Lauf mit 4 Runden des Strafverfahrens in einem anderen lokalen Minimum als die Abnahme):

```text
  Nebenbedingungen (mechanisch)               13
…
Mechanische Nebenbedingungen (Grenze und Ist in mm, Einfallswinkel in Grad; r = Wurzel(w) max(0, g))
   Nr  Typ                  Fläche     Gewicht      Grenze  Ist vorher Ist nachher     r vorher    r nachher
    1  max_semi_diameter    1        1.000e+06      4.0200      4.0250      3.9887    4.979e+00    0.000e+00
    2  max_semi_diameter    3        1.000e+06      2.6500      3.5729      2.5450    9.229e+02    0.000e+00
    3  max_center_thickness 2-3      1.000e+06      8.0000      1.0000      8.0000    0.000e+00    5.730e-03  verletzt
    4  min_edge_thickness   1-2      1.000e+06      1.3500      0.5521      1.3499    7.979e+02    5.432e-02  verletzt
    5  min_edge_thickness   0-1      1.000e+06      0.1000      1.7677      0.1000    0.000e+00    1.062e-02  verletzt
    6  max_incidence_angle  2        1.000e+06     26.0000     15.2035     25.7796    0.000e+00    0.000e+00
…
```

(`verletzt` mit r = 5,4·10⁻² heißt hier g = r/√w = 5,4·10⁻⁸ mm.)

### Geisterplatten: Diagnosebericht (ab M4-T6)

Hat das System Geisterplatten, folgt nach den Nebenbedingungen der Bericht von `diagnose_ghost_plates` am Ergebnis des besten Laufs ([optimierung.md, Abschnitt 5](optimierung.md#diagnosebericht-ab-m4-t6)), Text aus `write_ghost_diagnosis_text`. Die Merit-Werte (F, F mit a = 0 je Platte) rechnet immer die CPU mit derselben Quadratur, derselben Versagensart und denselben Zusatzresiduen wie die Optimierung (`make_cpu_merit`), beim weichen Versagen mit dem w_fail der letzten Runde des besten Laufs (ab M4-T2), auch mit `--gpu`. Inhalt: Merit und Merit-Wellenlängen, eine Zeile pro Platte (Platte, Dateifläche `after_surface`, Abstand, K, Σ|a| [mm], dominante Ordnungen, φ_G [1/mm], paraxiale Randstrahlhöhe y_R [mm], Hauptstrahlhöhe y_H für 1 rad, Strafe Σ r², Gewinn F(a = 0 für diese Platte) − F in Wellen², aktiv, n_ref, ν_d, ΔP_g,F, Glas), die Koeffizienten der aktiven Platten, pro Lage (gleiches `after_surface` und `distance`) Platten, aktive Platten, Operator (`none`, `A`, `B`, `B_anomalous`), Befund und Begründung, zuletzt die Empfehlung. Beispiel (`mikroptim optimize data/systems/einzellinse_na05_geisterplatten.json --quiet`, gekürzt):

```
Geisterplatten (Diagnose, docs/optimierung.md Abschnitt 5)
  Merit F [Wellen²]: 4.321694751e-01, Merit-Wellenlängen: 1
   Pl Fläche  Abstand  K  Summe|a| [mm]  dominant  phi_G [1/mm]  y_R [mm]   y_H/rad     Strafe     Gewinn  aktiv   n_ref   nu_d   dP_gF  Glas
    0      0    0.050  6      1.267e-06         -     1.067e-05     1.000     0.050   2.02e-05   1.91e-05   nein  1.4338   95.0  0.0546  CAF2
…
    8      2    0.200  6      7.770e-03         2     9.728e-02     0.772     0.976   4.23e-01   2.40e+01     ja  1.7534   26.9  0.0001  buchdahl(n0=1.76, nu=-0.18/0.112)
  (Strafe = Σ r² der Platte, Gewinn = F(a = 0 für diese Platte) − F in Wellen²; y_R Randstrahl-, y_H Hauptstrahlhöhe paraxial)
  Platte 8 a_k [mm]: -3.0601e-07 7.7691e-03 -2.0891e-07 2.2752e-07 -4.1881e-07 2.3464e-07
  Lagen:
    nach Fläche 0 im Abstand 0.05 mm: Platten 0, 1, 2, aktiv - -> none
      Befund: keine aktive Geisterplatte
      Begründung: alle |a_k| unter der Aktivitätsschwelle
…
    nach Fläche 2 im Abstand 0.2 mm: Platten 6, 7, 8, aktiv 8 -> A
      Befund: monochromatische Korrektur (sphärische Aberration höherer Ordnung, Aplanasie, Petzval)
      Begründung: nur eine Merit-Wellenlänge: die Platten korrigieren das Muster dieser Wellenlänge
  Empfehlung: Operator A (Meniskus ohne Brechkraft) nach Fläche 2 im Abstand 0.2 mm, Glas wie Platte 8 (buchdahl(n0=1.76, nu=-0.18/0.112)), dominante Ordnung 2; Biegung aus dem Vorzeichen der Korrektur
```

### Beispiel

Die Einzellinse `data/systems/einzellinse_bk7_biegung.json` (wie `einzellinse_bk7.json` mit Blendenradius 2,5 mm, also NA ≈ 0,025, eine Wellenlänge, nur Achsfeld) hat als Variablen beide Krümmungen in [−0,05; 0,05] und den Arbeitsabstand in [50; 150] mm; die Mittendicke 6 mm ist fest. Die Brennweite 100 mm ist eine schwach gewichtete Gleichung (Gewicht 0,029/mm², das Gewicht der Abnahmetests von M3-T4 nach der Faustregel in [optimierung.md, Abschnitt 2.2](optimierung.md#22-paraxiale-nebenbedingungen-ab-m3-t3); ein viel größeres Gewicht verdeckt die Biegung in der Rundung der Normalgleichungen, [optimierung.md, Abschnitt 4](optimierung.md#verfahren-ab-m3-t4)). Das Optimum ist die Biegung minimaler sphärischer Aberration bei fester Brennweite.

Aufruf aus dem Repository-Wurzelverzeichnis (CPU; gekürzt, … markiert ausgelassene Zeilen):

```
mikroptim optimize data/systems/einzellinse_bk7_biegung.json -o einzellinse_bk7_biegung_opt.json
```

```
Optimierung: Einzellinse N-BK7, Biegung
  Datei                                       data/systems/einzellinse_bk7_biegung.json
  Ergebnisdatei                               einzellinse_bk7_biegung_opt.json
  Backend                                     CPU, FP64
  Quadratur                                   10 Ringe x 8 Speichen
  Strahlversagen                              weich, w_fail ab 1.000e+06 Wellen² (Strafverfahren x10 bis 1e14)
  Jacobi-Matrix                               finite Differenzen (zentral)
  Variablen                                   3
  Nebenbedingungen (paraxial)                 1
  Läufe                                       1
  Zyklen höchstens                            100

Iterationsprotokoll (Merit F = Summe r² in Wellen² bei 587.560 nm, docs/optimierung.md)
  Lauf  Zyklus     Schritt            Merit        Testmerit     Gewinn   Dämpfung Schrittnorm versagt  hart  Status
     0       1  angenommen  6.747402936e-06  6.747402936e-06  9.999e-01  1.000e-03   1.116e-02       0     0
     0       2  angenommen  3.797180076e-06  3.797180076e-06  1.000e+00  3.333e-04   9.135e-05       0     0
     0       3  angenommen  3.796146522e-06  3.796146522e-06  9.999e-01  1.111e-04   1.312e-04       0     0
…
     0      42  angenommen  1.557334269e-06  1.557334269e-06  6.601e-01  1.914e-08   4.349e-04       0     0
     0      43  angenommen  1.557334158e-06  1.557334158e-06  8.857e-01  1.851e-08   1.135e-04       0     0
     0      44  angenommen  1.557334153e-06  1.557334153e-06  4.909e-02  1.001e-08   1.489e-04       0     0
     0      45  angenommen  1.557334135e-06  1.557334135e-06  5.096e-01  1.735e-08   6.196e-05       0     0
…
     0      53   verworfen  1.557334134e-06  1.557334138e-06 -1.405e+04  4.738e-05   3.283e-09       0     0
     0      54   verworfen  1.557334134e-06  1.557334137e-06 -1.006e+05  3.790e-04   4.109e-10       0     0

Ergebnis
  Status                                      converged_step
  Zyklen                                      54
  Angenommene Schritte                        46
  Jacobi-Matrizen                             47
  Merit vorher [Wellen²]                      3.134515536e-02
  Merit nachher [Wellen²]                     1.557334134e-06
  Merit nachher/vorher                        4.968341e-05
  Versagte Strahlen vorher/nachher            0 / 0
  davon hart vorher/nachher                   0 / 0
  Runden des Strafverfahrens                  1
  w_fail am Ende [Wellen²]                    1.000e+06

Variablen
  Name    Einheit   untere Schranke           vorher          nachher   obere Schranke
  c[1]    1/mm     -5.000000000e-02  9.670000000e-03  1.672853973e-02  5.000000000e-02
  c[2]    1/mm     -5.000000000e-02 -9.670000000e-03 -2.714119101e-03  5.000000000e-02
  w       mm              50.000000        99.050000        96.548727       150.000000

Nebenbedingungen (Istwerte in mm, Residuum r = Wurzel(w) q)
   Nr  Typ              Modus     Gewicht         Soll   Ist vorher  Ist nachher      r vorher     r nachher
    0  focal_length     equal   2.900e-02   100.000000   101.049681   100.000002  1.768974e-01  2.730606e-07

Wellenfront (Quadratur 10 Ringe x 8 Speichen, CPU, FP64, RMS in Wellen bei 587.560 nm)
  Feld  Winkel [deg]  Wellenlänge [nm]  Strahlen    RMS vorher   RMS nachher  versagt vorher/nachher
     0         0.000           587.560        10      0.007244      0.001248                   0 / 0

Ergebnis geschrieben: einzellinse_bk7_biegung_opt.json
```

Die Startmerit 3,1·10⁻² Wellen² ist fast ganz die Brennweitenbedingung (r = 0,177, Wellenfront 0,0072 Wellen RMS). Im ersten Zyklus bringt LM die Brennweite fast auf 100 mm und fokussiert über den Arbeitsabstand; danach wandert die Linse mit Gewinnverhältnissen um 0,7 zur besten Biegung (die Gauß-Newton-Krümmung in Biegerichtung verschwindet am Optimum, [optimierung.md, Abschnitt 4](optimierung.md#verfahren-ab-m3-t4)). Am Ende liegen die Merit-Änderungen im Rundungsrauschen, die Schritte werden verworfen, bis die Schrittnorm unter `xtol` fällt (`converged_step`). Der Formfaktor q = (c₁ + c₂)/(c₁ − c₂) = 0,7208 ist der Wert der Bibliothekstests (Dünnlinsentheorie 2(n² − 1)/(n + 2) = 0,7397; der Unterschied ist der Dickeneffekt).

**Ergebnisdatei** (Auszug; der Glaskatalog `data/glass/basis.json` bleibt unverändert, weil die Datei im Wurzelverzeichnis liegt):

```json
  "surfaces": [
    { "type": "stop", "thickness": 5.0, "material": "AIR", "semi_diameter": 2.5 },
    {
      "type": "sphere",
      "curvature": 0.01672853972858829,
      "thickness": 6.0,
      "material": "N-BK7",
      "semi_diameter": 12.5
    },
    {
      "type": "sphere",
      "curvature": -0.002714119101320638,
      "thickness": 96.54872747022759,
      "material": "AIR",
      "semi_diameter": 12.5
    }
  ],
```

**Vorher/nachher mit `mikroptim trace`** (`data/systems/einzellinse_bk7_biegung.json` gegen `einzellinse_bk7_biegung_opt.json`, alle Zeilen, die sich unterscheiden):

| Größe | vorher | nachher |
|---|---|---|
| Brennweite f′ | 101,049681 mm | 100,000002 mm |
| Brechkraft | 9,896122·10⁻³ 1/mm | 1,000000·10⁻² 1/mm |
| Hintere Schnittweite (ab letzter Fläche) | 99,052088 mm | 96,580179 mm |
| Vordere Schnittweite (ab Fläche 0) | −94,052088 mm | −94,445153 mm |
| Hauptebene H (ab Fläche 0) | 6,997593 mm | 5,554848 mm |
| Hauptebene H′ (ab letzter Fläche) | −1,997593 mm | −3,419822 mm |
| Defokus der Bildebene | 0,002088 mm | 0,031452 mm |
| Petzvalsumme | 6,589473·10⁻³ 1/mm | 6,624451·10⁻³ 1/mm |
| Austrittspupille Lage / Radius | −9,515815 mm / 2,686003 mm | −9,301383 mm / 2,647039 mm |
| RMS-Wellenfront (Achse, 587,56 nm) | 0,007244 Wellen (4,256352·10⁻⁶ mm) | 0,001248 Wellen (7,332349·10⁻⁷ mm) |

Die RMS-Wellenfront sinkt auf ein Sechstel; die Endmerit 1,557·10⁻⁶ Wellen² ist praktisch ihr Quadrat (0,001248² Wellen², eine Wellenlänge, ein Feld), der Beitrag der Brennweite ist (2,7·10⁻⁷)². Die Bildebene liegt jetzt 0,031 mm vor dem paraxialen Fokus: Bei sphärischer Aberration liegt der Fokus kleinster RMS-Wellenfront nicht im paraxialen Fokus, die Wellenfront-Merit findet ihn über den Arbeitsabstand.

**Multistart** (`--runs 8 --perturbation 0.2 --seed 1 --quiet`, Auszug): Alle Läufe enden im selben Optimum; die Endmerits unterscheiden sich nur im Rundungsrauschen (≈ 10⁻⁹ relativ), hier gewinnt Lauf 4.

```
Läufe
  Lauf  Status               Zyklen      Merit Start       Merit Ende  versagt  hart  Runden
     0  converged_step           54  3.134515536e-02  1.557334134e-06        0     0       1
     1  converged_step           55  1.544867972e+00  1.557334136e-06        0     0       1
     2  converged_step           52  5.204384846e+00  1.557334136e-06        0     0       1
     3  converged_step           59  4.067522266e+00  1.557334134e-06        0     0       1
     4  converged_step           51  1.154977668e+00  1.557334134e-06        0     0       1  bester
     5  converged_step           51  4.200103488e+00  1.557334136e-06        0     0       1
     6  converged_step           63  7.160670488e+00  1.557334135e-06        0     0       1
     7  converged_step           47  2.021973972e+00  1.557334135e-06        0     0       1
```

**GPU** (`--gpu`, RTX 3090): Kopfzeile `Backend  GPU 0 (NVIDIA GeForce RTX 3090), FP64`; konvergiert nach 56 Zyklen (`converged_step`) mit q = 0,720768 und Endmerit 1,557334138·10⁻⁶ Wellen² (CPU: 54 Zyklen, q = 0,720808, 1,557334134·10⁻⁶; die Rundungsunterschiede der Strahlresiduen ändern den Weg im flachen Tal, nicht das Optimum). Die GPU-Ergebnisdatei ergibt auf der CPU eine um 1,1·10⁻¹⁵ Wellen² andere Merit (Schranke oben: 4,2·10⁻¹²).

### JSON-Ausgabe (`--json`), Format `mikroptim.optimize/1`

Ein JSON-Objekt am Ende der Rechnung (eingerückt mit zwei Leerzeichen, abschließender Zeilenumbruch); Zahlen verlustfrei wie bei `trace`, nicht endliche Werte `null`. Längen in mm, Krümmungen in 1/mm, Wellenlängen in nm.

| Feld | Typ | Inhalt |
|---|---|---|
| `format` | string | `"mikroptim.optimize/1"` |
| `mikroptim_version` | string | Programmversion |
| `system`, `file` | string | `name` der Systembeschreibung, Pfad wie angegeben |
| `output` | string \| null | Ergebnisdatei wie angegeben, `null` ohne `-o` |
| `backend`, `device`, `precision` | | wie bei `trace` (`precision` immer `"fp64"`) |
| `merit_unit` | string | Einheit aller Merit-Werte (`merit`, `trial_merit`, `start_merit`): `"waves^2"` (Wellen² bei λ_ref, seit M4-T1) |
| `quadrature` | object | `{"rings": N, "spokes": M}` |
| `settings` | object | `max_iterations`, `runs`, `perturbation` und `seed` (beide `null` bei `runs` = 1), `failure` (`"soft"` oder `"hard"`, ab M4-T2), `fail_weight` (Start-w_fail in Wellen², `null` bei `"hard"`), `jacobian` (`"fd"` oder `"adjoint"`, ab M5-T3) |
| `variables` | object[] | pro Variable k: `name` (`"c[1]"`, `"t[2]"`, `"w"`, `"a[g,k]"`, `"n0[1]"`, `"nu1[1]"`, `"nu2[1]"`), `type`, `surface` (Dateifläche, bei Glas die Fläche vor dem Medium; `null` bei Geisterkoeffizienten), `plate` und `order` (nur Geisterkoeffizienten, sonst `null`; ab M4-T6), `lower`, `upper` (`null` = offen), `typical`, `start` (x₀), `result` (x des besten Laufs) |
| `constraints` | object[] | pro paraxialer Nebenbedingung: `item` (Eintrag in `constraints.items`), `type`, `mode`, `weight`, `target` (Soll), `value_start`, `value_result` (Ist: Brennweite 1/φ, Arbeitsabstand, Abgleichlänge; bei `paraxial_focus` die Randstrahlhöhe q in der Bildebene), `residual_start`, `residual_result` (r = √w·q̃ wie im Residuenvektor) |
| `glass_hull` | object \| null | Glas-Hülle (ab M6-T1; `null` ohne Glasvariablen oder ohne Eintrag `glass_hull`): `glasses` (Zahl der Gläser der Hülle), `dimension` (0–3), `vertices` (Namen der Gläser auf den Ecken), `weight`, `margin`, `elements` (pro Glasfläche `surface`, `excess_start`, `excess_result`: max(0, φ) in normierten Koordinaten am Start und am Ergebnis) |
| `mechanical_constraints` | object[] | pro mechanischer Bedingung und Fläche (ab M4-T3, leer ohne solche Einträge): `item`, `type`, `surface` (Dateifläche), `next_surface` (bei Dicken i + 1, sonst `null`), `weight`, `unit` (`"mm"` bzw. `"deg"` beim Einfallswinkel), `limit` (Grenze in `unit`), `value_start`, `value_result` (Ist in `unit`), `violation_start`, `violation_result` (g wie in der Residuenformel: mm bzw. Differenz von cos I; > 0 verletzt), `residual_start`, `residual_result` (r = √w·max(0, g)), `height_start`, `height_result` (Halbdurchmesser h der Randdicke einschließlich `clearance`, sonst `null`), `satisfied` (g ≤ 0 am Ergebnis) |
| `ghost_diagnosis` | object \| null | Diagnosebericht der Geisterplatten (ab M4-T6; `null` ohne Geisterplatten), Objekt aus `ghost_diagnosis_json`: `merit`, `reference_nm`, `merit_wavelengths`, `short_nm`, `long_nm`; `plates` (pro Platte `plate`, `after_surface`, `distance`, `table_surface`, `norm_radius`, `order`, `glass` {`name`, `n_ref`, `dispersion`, `n_d`, `abbe`, `partial_gf`, `delta_partial_gf`}, `coefficients`, `sum_abs`, `max_abs`, `activity_threshold`, `active`, `dominant_orders`, `power`, `petzval`, `marginal_height`, `chief_height_per_rad`, `penalized`, `penalty`, `merit_without`, `merit_gain`); `positions` (pro Lage `after_surface`, `distance`, `plates`, `active_plates`, `chromatic_ratio`, `single_glass_ratio`, `merit_gain`, `sum_abs`, `recommendation`, `finding`, `reason`); `recommended_position` (Index in `positions` oder `null`), `recommendation` (`"none"`, `"A"`, `"B"`, `"B_anomalous"`), `recommendation_text` |
| `iterations` | object[] \| null | Protokoll in der Reihenfolge der Rückrufe: `run`, `iteration`, `status`, `evaluated`, `accepted`, `merit`, `trial_merit`, `gain_ratio`, `damping`, `step_norm`, `failed_rays`, `hard_failed_rays` (wie `LmIterationInfo`; `iteration` läuft über die Runden des Strafverfahrens fort); `null` mit `--quiet` |
| `runs` | object[] | pro Lauf: `run`, `status`, `iterations`, `accepted`, `jacobians`, `start_merit` (F am Startpunkt), `merit`, `failed_rays`, `hard_failed_rays`, `penalty_rounds` (Runden des Strafverfahrens, 1 ohne Neustart), `fail_weight` (w_fail der letzten Runde, `null` bei hartem Versagen), `start` (Startpunkt), `x` (Endpunkt) |
| `best_run` | integer | Index des besten Laufs |
| `result` | object | bester Lauf: `status`, `converged` (einer der drei Converged-Zustände), `iterations`, `accepted`, `jacobians`, `start_merit` (F(x₀) der Datei, = `runs[0].start_merit`), `merit`, `start_failed_rays`, `failed_rays`, `start_hard_failed_rays`, `hard_failed_rays`, `penalty_rounds`, `fail_weight`, `damping`, `gradient_norm`, `model_decrease` (wie `LmRunResult`) |
| `reference_nm` | number | λ_ref für `rms_waves` |
| `wavefront` | object \| null | `{"start": [...], "result": [...]}`, Einträge pro Feld und Wellenlänge (Reihenfolge f · L + l): `field`, `field_angle_deg`, `wavelength`, `wavelength_nm`, `rays`, `failed_rays`, `rms_waves`, `rms_mm` (wie bei `trace`); `null`, wenn nicht auswertbar |
| `wavefront_error` | string \| null | Grund, warum `wavefront` `null` ist (afokales System ohne `target.focal_length`) |
| `warnings` | string[] | Warnungen des Laders und des Befehls (Feldziele) |

Status-Namen: `converged_merit`, `converged_step`, `converged_gradient`, `max_iterations`, `damping_limit`, `numerical_error` (`to_string(LmStatus)`), im Protokoll zusätzlich `running`. **Stabilität** wie bei `mikroptim.trace/1`: Neue Felder können hinzukommen; Umbenennen, Entfernen oder Bedeutungsänderungen erhöhen die Version auf `mikroptim.optimize/2`.

Beispiel (Auszug aus `mikroptim optimize data/systems/einzellinse_bk7_biegung.json --json`, zur Kürze teilweise in einer Zeile; die echte Ausgabe schreibt jedes Feld in eine eigene Zeile):

```json
{
  "format": "mikroptim.optimize/1",
  "mikroptim_version": "0.1.0",
  "system": "Einzellinse N-BK7, Biegung",
  "file": "data/systems/einzellinse_bk7_biegung.json",
  "output": null,
  "backend": "cpu",
  "device": null,
  "precision": "fp64",
  "merit_unit": "waves^2",
  "quadrature": { "rings": 10, "spokes": 8 },
  "settings": { "max_iterations": 100, "runs": 1, "perturbation": null, "seed": null,
                "failure": "soft", "fail_weight": 1000000.0 },
  "variables": [
    { "name": "c[1]", "type": "curvature", "surface": 1, "plate": null, "order": null,
      "lower": -0.05, "upper": 0.05, "typical": 0.01, "start": 0.00967,
      "result": 0.01672853972858829 },
    …
  ],
  "constraints": [
    { "item": 0, "type": "focal_length", "mode": "equal", "weight": 0.029, "target": 100.0,
      "value_start": 101.04968094724019, "value_result": 100.00000160346728,
      "residual_start": 0.17689736616623206, "residual_result": 2.730606328905618e-07 }
  ],
  "ghost_diagnosis": null,
  "iterations": [
    { "run": 0, "iteration": 1, "status": "running", "evaluated": true, "accepted": true,
      "merit": 6.7474029362767126e-06, "trial_merit": 6.7474029362767126e-06,
      "gain_ratio": 0.9999097450120137, "damping": 0.001, … },
    …
  ],
  "runs": [ { "run": 0, "status": "converged_step", "iterations": 54, … } ],
  "best_run": 0,
  "result": {
    "status": "converged_step",
    "converged": true,
    "iterations": 54,
    "accepted": 46,
    "jacobians": 47,
    "start_merit": 0.031345155357940434,
    "merit": 1.557334133980554e-06,
    "start_failed_rays": 0,
    "failed_rays": 0,
    "start_hard_failed_rays": 0,
    "hard_failed_rays": 0,
    "penalty_rounds": 1,
    "fail_weight": 1000000.0,
    "damping": 0.0060647987285953695,
    "gradient_norm": 5.76557505989509e-09,
    "model_decrease": 8.035301977234237e-06
  },
  "reference_nm": 587.56,
  "wavefront": {
    "start": [ { "field": 0, …, "rms_waves": 0.007244114948732903, "rms_mm": 4.256352179277504e-06 } ],
    "result": [ { "field": 0, …, "rms_waves": 0.0012479319129737986, "rms_mm": 7.33234874786885e-07 } ]
  },
  "wavefront_error": null,
  "warnings": []
}
```

### Tests

`tests/test_cli_optimize.cpp` ruft `mko::cli::run_optimize_command` direkt auf:

- `CliOptimize.MeritDecreasesAndResultReproducesMerit`: Biegung der Einzellinse, Merit 3,13·10⁻² → 1,56·10⁻⁶ Wellen², x und Merit bitgleich zu `optimize_lm_cpu`, q = 0,7208; die Ergebnisdatei liefert beim Neuladen denselben Variablenvektor und **bitgleich dieselbe Merit** (über `CpuLmBackend` und über `optimize --max-iterations 0`); `mikroptim trace` lädt sie (Glaskatalog relativ zum neuen Verzeichnis) und gibt dieselbe Wellenfront aus wie „nachher“.
- `CliOptimize.GhostPlateResultReproducesMerit`: Einzellinse mit Geisterplatten, Dicke des geteilten Spalts als Variable, bitgleiche Merit nach dem Neuladen; absoluter Glaskatalog bleibt unverändert.
- `CliOptimize.GhostPenaltyAndDiagnosis` (M4-T6): `einzellinse_geisterplatten_exakt.json` mit Geisterkoeffizienten und L1-Strafe: alle \|a\| < 10⁻⁸ mm, `plate`/`order` im JSON, `ghost_diagnosis` ohne aktive Platte (Empfehlung `none`), Ergebnisdatei reproduziert die Merit einschließlich Strafe bitgleich; Text bei `einzellinse_na05_geisterplatten.json` mit Diagnoseabschnitt und Empfehlung Operator A; ohne Geisterplatten `ghost_diagnosis: null` und kein Abschnitt.
- `CliOptimize.GlassVariableAndHull` (M6-T1): `einzellinse_glasvariable.json` mit Glasvariable und Glas-Hülle: Variablennamen `n0[1]`, `nu1[1]`, `nu2[1]`, Abschnitt `glass_hull` (39 Gläser, Dimension 3, Überschreitung am Start 0), Ergebnisdatei mit Buchdahl-Material (Koordinaten des Ergebnisses, ν₃ des N-BK7-Fits) reproduziert die Merit einschließlich Strafe bitgleich; Textabschnitte; ohne Glasvariablen `glass_hull: null`.
- `MechanicalOptimization.CliReportsMechanicalConstraints` (M4-T3, in `tests/test_mechanical_optimization.cpp`): Kopfzeile und Tabelle im Text, `mechanical_constraints` im JSON (13 Einträge für `zweilinser_mechanik.json`, Ist, g und r gegen `mechanical_value` mit `surface_extrema_cpu`).
- `CliOptimize.JsonIterationsMatchLibraryHistory`: ein Protokolleintrag pro Zyklus, Merit bitgleich zu `LmRunResult::history`; `--quiet` ohne Protokoll.
- `CliOptimize.ConstraintsReportedBeforeAndAfter`: Istwerte und Residuen gegen `compute_paraxial` und die Dicken (`einzellinse_bk7_nebenbedingungen.json`).
- `CliOptimize.MultistartIsDeterministic`: zwei Aufrufe gleiche Ausgabe, Startpunkte bitgleich zu `perturbed_vector`, bester Lauf nach der Regel oben, Lauf 0 gleich dem Einzellauf.
- `CliOptimize.SoftFailureStartBecomesFailureFree` (M4-T2): `weiches_versagen_start.json` mit 27 von 90 versagenden Strahlen wird versagensfrei (F 2,160648·10⁶ → 1,878483·10³ Wellen², 146 Zyklen, 4 Runden bis w_fail = 10⁹), x und Merit bitgleich zu `optimize_lm_cpu`, Protokoll mit fortlaufenden Zyklen; `--hard-failure` behält 27 hart versagte Strahlen (eine Runde, `fail_weight: null`), `--fail-weight 1e9` setzt das Startgewicht; Text mit Versagensart, Runden und w_fail.
- `CliOptimize.AdjointJacobian` (M5-T3): `--jacobian adjoint` bitgleich zu `optimize_lm_cpu` mit `JacobianMethod::Adjoint`, Endmerit relativ ≤ 10⁻⁶ und Formfaktor ≤ 10⁻³ wie mit `fd` (Standard), `settings.jacobian` im JSON, Zeile „Jacobi-Matrix“ im Text.
- `CliOptimize.WarnsAboutFieldTargetsFromStartFocalLength`, `CliOptimize.TextOutput`, `CliOptimize.Errors` (keine Variablen, fehlende Datei, `-o` gleich der Eingabe, fehlendes Ausgabeverzeichnis, Kommandozeilenfehler), `CliOptimize.GpuUnavailableWithoutCuda` (CPU-Build).
- Im CUDA-Build `CliOptimizeGpu.MatchesCpuAndReproducesMerit` (übersprungen ohne Gerät): GPU-Optimierung konvergiert mit demselben Formfaktor; Neuladen auf der GPU bitgleich, auf der CPU innerhalb der Schranke oben; Multistart auf der GPU deterministisch. `CliOptimizeGpu.AdjointJacobianMatchesCpu` (M5-T3): `--gpu --jacobian adjoint` am Optimum der CPU, Ausgabe deterministisch. `CliOptimizeGpu.SoftFailureStartBecomesFailureFree` (M4-T2): versagensfrei auf der GPU, Ausgabe deterministisch. `CliOptimizeGpu.InvalidDevice`.

ctest-Kommandos mit dem gebauten Programm: `cli_optimize_text` (schreibt `einzellinse_bk7_biegung_opt.json` ins Build-Verzeichnis), danach `cli_optimize_result_trace` (`mikroptim trace` auf dieser Ergebnisdatei, f′ = 100,0000 mm) und `cli_optimize_no_variables`; im CPU-Build `cli_optimize_gpu_unavailable` (Exit-Code 3), im CUDA-Build `cli_optimize_gpu_json` (übersprungen ohne Gerät) und `cli_optimize_gpu_no_device`.

Implementierung: `src/cli/optimize_command.cpp` (interner Header `src/cli/optimize_command.hpp`, Bibliothek `mikroptim_cli`, gelinkt gegen `mikroptim_optim`, im CUDA-Build `mikroptim_cuda`); Exit-Codes, Zahlen aus Argumenten und Textformatierung teilen sich `trace` und `optimize` in `src/cli/cli_util.hpp`.

## `mikroptim eval`

Wertet ein System im Modus EVAL aus (M8-T1): dichtes kartesisches N×N-Raster über die Pupille, Referenzsphäre standardmäßig durch die paraxiale Austrittspupille, jedes Versagen hart; Definitionen: [numerik.md, „Auswertung auf dem Raster“](numerik.md#auswertung-auf-dem-raster-m8-t1), Bausteine: [architektur.md, „Modus EVAL“](architektur.md#modus-eval-m8-t1). Das System wird als Batch mit einer Variante aufgebaut (`make_system_batch`, ideale Bildhöhen h_f = f_soll tan θ_f wie bei `trace`; afokal ohne `target.focal_length`: h_f = 0, Verzeichnung nicht definiert), dann `eval_cpu` bzw. mit `--gpu` `TraceEngine::evaluate`, Kennzahlen mit `eval_statistics`. Gerechnet wird immer in FP64.

| Option | Bedeutung | Standard |
|---|---|---|
| `<system.json>` | Systemdatei | – |
| `--grid N` | Raster N×N über das Quadrat der normierten Pupille, 1 ≤ N ≤ 2048 | 64 |
| `--half` | nur die Hälfte x ≥ 0 rechnen (Symmetrie x → −x; nicht bei einer Punktquelle mit x ≠ 0) | aus |
| `--reference R` | Referenzsphäre durch die Austrittspupille (`exit-pupil`) oder durch den Scheitel der letzten Fläche (`vertex`, wie `trace`) | `exit-pupil` |
| `--json` | maschinenlesbare Ausgabe `mikroptim.eval/1` | aus |
| `--map F,L` | OPD-Karte von Feld F und Wellenlänge L (ab 0) schreiben; verlangt `--map-file` | – |
| `--map-file D` | Zieldatei der Karte (CSV) | – |
| `--remove X` | vor der Karte abziehen: `piston`, `tilt` (Kolben und Kippung) oder `defocus` (Kolben, Kippung, Defokus) | `piston` |
| `--psf` | skalare PSF, MTF und Strehl über die FFT für alle Tupel (M8-T2) | aus |
| `--padding Q` | Zero-Padding der FFT: M = kleinste Zweierpotenz ≥ max(2N, Q·N), Q ≥ 2, M ≤ 8192 (setzt `--psf`) | 4 |
| `--na X` | NA der Pupillenabbildung vorgeben, beide Richtungen (setzt `--psf`; nötig z. B. bei afokalen Systemen) | aus den Feldstrahlen |
| `--psf-tuple F,L` | Tupel für `--psf-file` und `--mtf-file` (setzt `--psf`) | 0,0 |
| `--psf-file D` | PSF des Tupels als CSV (setzt `--psf`) | – |
| `--mtf-file D` | MTF-Schnitte des Tupels als CSV (setzt `--psf`) | – |
| `--vector` | vektorielle PSF nach Richards-Wolf statt der skalaren (M8-T5; setzt `--psf`) | aus |
| `--polarization P` | Eingangsfeld der vektoriellen PSF: `x`, `y`, `circular` oder `unpolarized` (setzt `--vector`) | `x` |
| `--defocus Z` | Auswerteebene Z mm von der Bildebene entlang der Achse, > 0 in +z (Tracerichtung, hinter der Bildebene; setzt `--vector`) | 0 |
| `--component-files P` | \|E_x\|², \|E_y\|², \|E_z\|² des Tupels als CSV nach `P_ex.csv`, `P_ey.csv`, `P_ez.csv` (setzt `--vector`) | – |
| `--gpu`, `--device N` | auf der GPU rechnen (Kernel EVAL, mit `--psf` auch die FFT über cuFFT), wie bei `trace` | aus, 0 |
| `--help`, `-h` | Hilfe | – |

Exit-Codes wie `mikroptim trace` (0 Erfolg, 1 Fehler beim Laden, Auswerten oder Schreiben der Karte, 2 ungültige Kommandozeile, 3 GPU-Backend nicht verfügbar).

**Textausgabe:** Kopf (Raster, Lage der Referenzsphäre, Backend), dann pro Feld und Wellenlänge eine Zeile: versagte Strahlen, RMS-OPD und RMS nach Abzug der Kippung in Wellen bei λ_ref, Strehl nach Maréchal exp(−(2π σ/λ)²) mit dem RMS σ bei der Wellenlänge des Tupels (grobe Angabe für kleine Fehler; Strehl über die FFT mit `--psf`, unten), RMS-Spotradius und Schwerpunkt ȳ relativ zum idealen Bildpunkt [mm], Verzeichnung des Hauptstrahls [%], Brennlinien z_t, z_s relativ zur Bildebene [mm]. Beispiel (`mikroptim eval data/systems/einzellinse_bk7.json --grid 128`):

```
System: Einzellinse N-BK7
  Datei: data/systems/einzellinse_bk7.json
  Raster: 128 x 128, 12892 Punkte in der Pupille
  Referenzsphäre: durch die Austrittspupille (z = -9.515815 mm ab letzter Fläche)
  Backend: CPU, FP64
  OPD in Wellen bei 587.560 nm, Strehl nach Maréchal bei der Wellenlänge des Tupels, Längen in mm

  Feld  Winkel   λ [nm] versagt    RMS-OPD ohne Kipp.  Strehl    RMS-Spot   Schwerp. y  Verz. [%]         z_t         z_s
     0   0.000   486.13       0    4.52913    4.52913  0.0000   1.535e-01   2.2768e-18          - -1.0624e+00 -1.0624e+00
     0   0.000   587.56       0    1.94831    1.94831  0.0000   7.874e-02  -1.8364e-18          -  2.0868e-03  2.0868e-03
     0   0.000   656.27       0    0.87331    0.87331  0.0000   4.783e-02   8.1485e-19          -  4.8247e-01  4.8247e-01
     1   1.000   486.13       0    4.63046    4.61519  0.0000   1.562e-01   7.8784e-03   -0.08631 -1.1084e+00 -1.0844e+00
     1   1.000   587.56       0    2.09618    2.03658  0.0000   8.145e-02   9.1372e-03   -0.00553 -4.4518e-02 -2.0281e-02
     1   1.000   656.27       0    1.10662    0.96035  0.0000   5.050e-02   9.6982e-03    0.03044  4.3559e-01  4.5997e-01
     2   2.000   486.13       0    4.92728    4.87474  0.0000   1.642e-01   1.5401e-02   -0.09721 -1.2463e+00 -1.1506e+00
     2   2.000   587.56       0    2.49476    2.30317  0.0000   8.956e-02   1.7922e-02   -0.01634 -1.8420e-01 -8.7355e-02
     2   2.000   656.27       0    1.62438    1.22574  0.0000   5.850e-02   1.9045e-02    0.01966  2.9511e-01  3.9249e-01
  Versagte Strahlen: 0 von 116028
```

Die Einzellinse ist bei voller Öffnung (a = 10 mm) weit von der Beugungsgrenze entfernt (RMS 1,9 Wellen, Strehl ≈ 0); der Farblängsfehler zeigt sich in den Brennlinien des Achsfeldes (−1,06 mm bei 486 nm, +0,48 mm bei 656 nm), die Feldkrümmung in z_t, z_s der Feldpunkte.

**JSON** (`mikroptim.eval/1`): `format`, `mikroptim_version`, `system`, `file`, `reference_nm`, `backend`, `device`, `grid`, `half_pupil`, `grid_points` (Rasterpunkte n in der Pupille), `reference_sphere` (`"exit-pupil"`/`"vertex"`), `exit_pupil_position` (z_AP ab letzter Fläche, `null` bei `vertex` oder Austrittspupille im Unendlichen), `tuples` (pro Feld und Wellenlänge in der Reihenfolge f, l: `field`, `field_angle_deg`, `wavelength`, `wavelength_nm`, `image_height_mm`, `rays`, `failed_rays`, `rms_opd_mm`, `rms_opd_waves`, `pv_opd_waves`, `rms_opd_tilt_waves`, `rms_opd_defocus_waves`, `strehl_marechal`, `tilt_mm` [t_x, t_y], `defocus_mm` (ρ²-Koeffizient), `centroid_mm` [x̄, ȳ], `rms_spot_mm`, `chief_ray_mm` [x, y], `distortion_chief_percent`, `distortion_centroid_percent`, `focus_tangential_mm`, `focus_sagittal_mm`, `astigmatism_mm`), `rays`, `failed_rays`, `warnings`. Wellen immer bei λ_ref; nicht endliche Werte (z. B. die Verzeichnung auf der Achse) sind `null`.

**OPD-Karte** (`--map F,L --map-file D`): CSV mit N Zeilen zu je N Werten, OPD in Wellen bei λ_ref nach dem gewählten Abzug, verlustfrei mit 17 Stellen; erste Zeile = größtes y (wie ein Bild der Pupille, x nach rechts), leere Felder außerhalb der Pupille und bei versagten Strahlen; bei `--half` gespiegelt (`eval_opd_map`).

**PSF, MTF, Strehl** (`--psf`, M8-T2; Formeln, Abtastung und Grenzen der skalaren Näherung: [numerik.md, „Beugung: PSF, MTF, Strehl“](numerik.md#beugung-psf-mtf-strehl-m8-t2)): Pro Tupel wird aus der OPD-Karte die Pupillenfunktion gebildet (`make_psf_pupils`, NA aus den Feldstrahlen oder `--na`) und mit `psf_cpu` bzw. mit `--gpu` über `PsfEngine` (cuFFT) gerechnet; die Tupel laufen nacheinander, die PSF selbst bleibt nur für das Tupel von `--psf-tuple` erhalten. Die Textausgabe bekommt eine zweite Tabelle mit NA (tangential), Airy-Radius j₁,₁/(2π)·λ/NA ≈ 0,61 λ/NA [mm], Strehl (nach Abzug der Kippung), Strehl im idealen Bildpunkt (FFT), Maximum der abgetasteten PSF und seiner Lage y [mm] sowie der MTF bei f_c/2 sagittal und tangential. Beispiel (`mikroptim eval data/systems/einzellinse_bk7.json --grid 512 --psf --gpu`, Ausschnitt; RTX 3090, 4 s mit 1,9·10⁶ Strahlen und neun FFT-Paaren 2048 × 2048):

```
  PSF skalar (FFT 2048 x 2048, Zero-Padding 4.00), NA aus den Feldstrahlen; Strehl auf die ungestörte Pupille normiert, MTF bei f_c/2 mit f_c = 2 NA/λ
  Feld   λ [nm]       NA   Airy [mm]   Strehl  im Bildp.  Maximum  Max. y [mm]   MTF_s   MTF_t
     0   486.13  0.10002  2.9639e-03  0.00019    0.00019  0.00025  -1.8226e-03  0.0004  0.0004
     0   587.56  0.09896  3.6208e-03  0.02189    0.02189  0.02189   0.0000e+00  0.0008  0.0008
     0   656.27  0.09849  4.0635e-03  0.09198    0.09198  0.09198   0.0000e+00  0.0778  0.0778
     1   486.13  0.10003  2.9637e-03  0.00018    0.00019  0.00025   7.2898e-03  0.0000  0.0004
     1   587.56  0.09897  3.6204e-03  0.00611    0.01723  0.01742  -7.4210e-04  0.0005  0.0023
     1   656.27  0.09850  4.0632e-03  0.02156    0.02884  0.10856   3.3314e-03  0.0774  0.0634
     2   486.13  0.10005  2.9629e-03  0.00020    0.00015  0.00020   8.5027e-03  0.0003  0.0002
     2   587.56  0.09899  3.6195e-03  0.00177    0.00964  0.01026  -2.9677e-03  0.0007  0.0011
     2   656.27  0.09852  4.0621e-03  0.00907    0.01013  0.10820   4.9959e-03  0.0512  0.0207
```

Bei voller Öffnung (RMS bis 4,9 Wellen) ist die Einzellinse weit von der Beugungsgrenze entfernt; außerhalb der Achse liegt das Maximum dann nicht im Bezugspunkt der Kippung (`Strehl` < `Maximum`), siehe [numerik.md](numerik.md#beugung-psf-mtf-strehl-m8-t2). Mit `--grid 64` warnt das Programm für alle Tupel vor einem zu kleinen Bildfenster (unten).

**Bildfenster.** Die PSF deckt M·Δx = λN/(2 NA) ab (N/2 Airy-Durchmesser, unabhängig von Q); was darüber hinausgeht, faltet die periodische FFT zurück. Reicht der geometrische Spot eines Tupels (Schwerpunkt ± 3 RMS-Radien) über das halbe Fenster hinaus, warnt `mikroptim eval` (stderr und `warnings` in JSON); Abhilfe ist ein größeres Raster `--grid`. Die Einzellinse ist bei voller Öffnung (RMS-Spot bis 0,16 mm) weit davon entfernt, beugungsbegrenzt zu sein, und braucht dafür N ≥ 256.

In JSON steht bei `--psf` oben `psf` (`fft_size`, `padding`, `numerical_aperture` oder `null`) und pro Tupel `psf` mit `na` [NA_x, NA_y], `pixel_mm` [Δx, Δy], `airy_radius_mm`, `strehl`, `strehl_reference_mm` (Bezugspunkt nach Abzug der Kippung), `strehl_origin` (direkt im idealen Bildpunkt), `strehl_fft`, `peak`, `peak_mm`, `energy` (transmittierter Anteil), `cutoff_lpmm` [f_c,x, f_c,y], `mtf_half_cutoff` [sagittal, tangential]; ohne `--psf` sind beide `null`.

Das Objekt `psf` oben enthält zusätzlich `model` (`"scalar"` oder `"vector"`), `polarization` und `defocus_mm` (beide `null` bei skalarer PSF); jedes Tupel `psf` hat zusätzlich `vector` (`null` bei skalarer PSF).

**Vektorielle PSF** (`--vector`, M8-T5; Modell, Apodisation 1/√cos θ, Polarisationsdrehung und Gültigkeit: [numerik.md, „Vektorielle PSF“](numerik.md#vektorielle-psf-m8-t5)): Statt der skalaren Pupillenfunktion werden pro Eingangsfeld die drei Feldkomponenten in der Pupille transformiert (`make_vector_psf_pupils`, `vector_psf_cpu` bzw. mit `--gpu` `VectorPsfEngine`); Eingangsfeld gleichmäßig beleuchtet, linear (`x`, `y`), zirkular oder unpolarisiert (inkohärente Summe aus x und y). Die PSF ist |**E**|² normiert auf das Maximum der perfekten vektoriellen PSF derselben NA, Abtastung, Bildfenster und Warnung wie oben. `--defocus Z` rechnet die Ebene im Abstand Z von der Bildebene (Fokusschnitte: mehrere Aufrufe). Die Textausgabe ersetzt die PSF-Tabelle durch: NA (tangential), Strehl (im Bezugspunkt der Kippung), Strehl im idealen Bildpunkt, Maximum, Halbwertsbreiten in x und y durch das Maximum [mm] (bei linearer Polarisation in Polarisationsrichtung gestreckt), Anteil von |E_z|² an der Energie der Ebene, größter Wert von |E_z|² und MTF bei f_c/2. Beispiel (`mikroptim eval data/systems/einzellinse_geisterplatten_exakt.json --grid 256 --vector --gpu`, Ausschnitt; RTX 3090):

```
  PSF vektoriell (Richards-Wolf, Polarisation x, Ebene z = 0.0000e+00 mm; FFT 1024 x 1024, Zero-Padding 4.00), NA aus den Feldstrahlen; Strehl auf die perfekte vektorielle PSF normiert, Halbwertsbreiten durch das Maximum, Anteil von |E_z|² an der Energie der Ebene
  Feld   λ [nm]       NA   Strehl  im Bildp.  Maximum  FWHM x [mm]  FWHM y [mm]  Anteil z  max |E_z|²   MTF_s   MTF_t
     0   587.56  0.02478  0.99693    0.99693  0.99693   1.2197e-02   1.2195e-02   0.00015     0.00008  0.3893  0.3895
```

Bei NA 0,025 ist die vektorielle PSF praktisch die skalare (Strehl gleich auf fünf Stellen, FWHM_x/FWHM_y − 1 = 1,6·10⁻⁴, Anteil von |E_z|² ≈ NA²/4); bei NA 0,95 (Zielobjektiv) ist der Fokus bei linearer Polarisation in Polarisationsrichtung um 45 % gestreckt (FWHM 0,730 λ gegen 0,503 λ) und |E_z|² trägt 26,5 % der Energie ([numerik.md](numerik.md#vektorielle-psf-m8-t5)).

In JSON steht pro Tupel `psf.vector` mit `fwhm_mm` [x, y], `energy_fraction` [x, y, z] (Σ|E_c|²/Σ|**E**|² in der Ebene), `component_peak` [x, y, z] (größte Abtastwerte in Strehl-Einheiten), `reference_intensity` (I_0) und `evanescent_points` (Rasterpunkte mit |s_⊥| ≥ 1). Die Felder von `psf` haben die vektorielle Bedeutung (Strehl relativ zur perfekten vektoriellen PSF, `energy` = Σ_trans W²/Σ_voll W²); `airy_radius_mm` bleibt der skalare Wert. **Komponentendateien** (`--component-files P`): `P_ex.csv`, `P_ey.csv`, `P_ez.csv` im Format der PSF-Datei; ihre Summe ist die PSF.

**PSF-Datei** (`--psf-file D`): CSV mit M Zeilen zu je M Werten, normiert auf das Maximum der ungestörten PSF (Strehl-Einheiten), 17 Stellen; erste Zeile = größtes y, Spalte i bei x = (i − M/2)·Δx, Zeile von unten j bei y = (j − M/2)·Δy relativ zum idealen Bildpunkt. **MTF-Datei** (`--mtf-file D`): Kopfzeile `k,nu,f_sagittal_lpmm,mtf_sagittal,f_tangential_lpmm,mtf_tangential`, dann N + 1 Zeilen für ν = k/N.

**Tests:** `tests/test_cli_eval.cpp` (JSON bitgleich zu `eval_cpu`/`eval_statistics`, Karte gegen `eval_opd_map`, mit `--psf` Kennzahlen bitgleich zu `psf_cpu`, PSF- und MTF-Datei, mit `--vector` Kennzahlen bitgleich zu `vector_psf_cpu` und Komponentendateien, Text, Fehlerfälle; `CliEvalGpu.MatchesCpu`, `CliEvalGpu.PsfMatchesCpu`, `CliEvalGpu.VectorPsfMatchesCpu` im CUDA-Build, sonst `CliEval.GpuUnavailableWithoutCuda`), ctest `cli_eval_json`, `cli_eval_psf_json`, `cli_eval_vector_json` und `cli_eval_gpu_json` (Exit-Code 3 ohne Gerät = übersprungen). Implementierung: `src/cli/eval_command.cpp` (Bibliothek `mikroptim_cli`, gemeinsame Bausteine aus `src/cli/cli_util.hpp`).

### Dezentrierte und verkippte Systeme (ab M8-T3)

`mikroptim trace` (CPU und `--gpu`) und `mikroptim eval` rechnen Systeme mit `decenter`/`tilt_deg` an Flächen ([dateiformat.md](dateiformat.md#flächen), ADR-0011) ohne weitere Optionen. Unterschiede zu koaxialen Systemen:

- **Volle Pupille für alle Felder**, auch auf der Achse (`trace`: Spalte „Strahlen“ N·M statt N, in JSON `pupil_mode` = `"full"`); `eval --half` wird mit Exit-Code 1 abgelehnt („halbe Pupille nur bei Symmetrie x → −x (System nicht koaxial, ADR-0011)“).
- **Referenzsphäre und ideale Bildpunkte** bleiben nominell (Paraxialdaten koaxial, numerik.md, Abschnitt 2): Der Bildversatz durch die Toleranzen erscheint in `trace` als Kippung im RMS, in `eval` im Schwerpunkt und in der Verzeichnung; „RMS-OPD ohne Kipp.“ zeigt die Bildfehler ohne den Versatz.
- Die **sagittale Brennlinie** z_s in `eval` nimmt die Spiegelsymmetrie an x = 0 an und ist bei Dezentrierung in x ohne Bedeutung.

Beispiel ([data/systems/testsystem_mehrlinser_dezentriert.json](../data/systems/testsystem_mehrlinser_dezentriert.json), frei erfunden: das Testsystem Mehrlinser mit Dezentrierungen bis 0,03 mm und Verkippungen bis 0,1° an fünf Flächen; ohne sie RMS 0,60 / 0,84 / 1,26 Wellen bei 546 nm):

```text
$ mikroptim trace data/systems/testsystem_mehrlinser_dezentriert.json
…
Wellenfront (Quadratur 10 Ringe x 8 Speichen, RMS in Wellen bei 546.070 nm)
  Backend                                     CPU, FP64
  Sollbrennweite für die Bildpunkte           20.551558 mm
  Feld  Winkel [deg]  Wellenlänge [nm]  Strahlen  versagt  RMS [Wellen]       RMS [mm]
     0         0.000           486.130        80        0      6.402784   3.496368e-03
     0         0.000           546.070        80        0      6.360018   3.473015e-03
…
     2         3.000           656.270        80        0      6.887277   3.760935e-03
  Versagte Strahlen: 0 von 960

$ mikroptim eval data/systems/testsystem_mehrlinser_dezentriert.json --grid 64
…
  Feld  Winkel   λ [nm] versagt    RMS-OPD ohne Kipp.  Strehl    RMS-Spot   Schwerp. y  Verz. [%]         z_t         z_s
     0   0.000   486.13       0    6.43425    0.61374  0.0000   1.142e-02   2.6517e-02          - -5.7333e-03 -2.2526e+01
…
     2   3.000   656.27       0    6.94890    1.12497  0.0000   1.941e-02   3.2872e-02    2.26816 -6.5355e-02 -2.2550e+01
```

Der Bildpunkt auf der Achse liegt um 0,026 mm (Schwerpunkt y) neben dem nominellen; ohne die Kippung bleiben 0,61 Wellen (koaxial 0,60). Tests: `DecenterCli.TraceAndEvalWorkWithDecenteredSystem` (JSON gegen `evaluate_wavefront`, volle Pupille, Ablehnung von `--half`).

## `mikroptim snap`

Glas-Einrasten (M6-T5, [optimierung.md, Abschnitt 9](optimierung.md#9-glas-einrasten)): Ersetzt die Glasvariablen (Typ `glass`, kontinuierliche Buchdahl-Koordinaten n₀, ν₁, ν₂) einer Systemdatei durch Katalog-Gläser. Für jedes Glaselement werden die k nächsten Gläser im normierten (n₀, ν₁, ν₂)-Raum der Glas-Hülle gesucht, jede Kombination (kartesisches Produkt, höchstens N mit der kleinsten Summe der quadrierten Abstände) wird ein Lauf des gebündelten Levenberg-Marquardt: Glas fest auf das Katalogglas (Brechzahlen aus der Sellmeier-Formel des Katalogs), übrige Variablen ab den Werten der Datei reoptimiert. Die beste Kombination (Regel des Multistarts von `optimize`) wird das Ergebnis; die Ergebnisdatei nennt Katalognamen und hat keine Glasvariablen mehr.

Eingabe ist typischerweise die Ergebnisdatei von `mikroptim optimize` mit Glasvariablen (Materialien als `{"model": "buchdahl", …}`, Werte = kontinuierliches Optimum). Eine Datei mit Katalog- oder konstanten Startgläsern wird wie in `optimize` zuerst in Buchdahl-Materialien umgewandelt; „kontinuierlich“ ist dann der Startpunkt. Als eigener Befehl (statt einer Option von `optimize`) lässt sich das Einrasten mit anderen Einstellungen wiederholen, ohne die kontinuierliche Optimierung neu zu rechnen, und das kontinuierliche Ergebnis bleibt als Datei erhalten.

**Merit:** wie bei `optimize` (Wellenfront, Strafresiduen beim weichen Versagen, paraxiale Nebenbedingungen, L1-Strafe der Geisterkoeffizienten, mechanische Nebenbedingungen ab M4-T3) ohne die Glas-Hülle; dieselben Einstellungen für das kontinuierliche und die eingerasteten Systeme. Einheit Wellen² bei λ_ref, FP64.

### Optionen

| Option | Bedeutung | Standard |
|---|---|---|
| `<system.json>` | Systemdatei mit mindestens einer Variable `glass` und mindestens einer weiteren Variable | – |
| `-o DATEI`, `--output DATEI` | Ergebnis als neue Systemdatei (Glaskataloge wie bei `optimize`, nie die Eingabedatei) | keine Datei |
| `--json` | maschinenlesbare Ausgabe `mikroptim.snap/1` statt Text | aus |
| `--neighbors k` | Katalog-Gläser pro Glaselement, k ≥ 1 | 3 |
| `--max-combinations N` | höchstens N Kombinationen = Läufe, N ≥ 1; bei mehr die N mit der kleinsten Summe d² | 64 |
| `--max-iterations N` | Zyklen der Reoptimierung pro Lauf, N ≥ 0 (0 = nur einrasten und auswerten) | 100 |
| `--rings N`, `--spokes M` | Pupillenquadratur wie bei `trace` | 10, 8 |
| `--fail-weight w`, `--hard-failure` | Strahlversagen wie bei `optimize` | weich, 10⁶ |
| `--jacobian V` | Jacobi-Matrix der Reoptimierung wie bei `optimize` (`fd` oder `adjoint`, ab M5-T3) | `fd` |
| `--gpu`, `--device N` | Reoptimierung auf der GPU (`cuda::optimize_lm_gpu`); Kandidaten und Kombinationen entstehen immer auf dem Host | CPU |
| `--help`, `-h` | Hilfe | – |

Kandidaten sind die Gläser der Glas-Hülle aller `glass_catalogs` (Gültigkeitsbereich deckt 400–700 nm ab), deren Gültigkeitsbereich zusätzlich alle Wellenlängen des Spektrums und λ_ref abdeckt. Der Abstand ist euklidisch in den normierten Koordinaten u = (x − m)/s der Hülle (Gewichte 1; die Bibliothek erlaubt Gewichte pro Koordinate, `GlassSnapOptions::distance_weights`).

### Exit-Codes und Fehler

Wie bei `optimize`: 0 Erfolg; 1 Fehler beim Laden, System ohne Glasvariablen (`keine Glasvariablen`) oder ohne weitere Variablen, kein Katalogglas deckt das Spektrum ab, Rechenfehler, Schreiben der Ergebnisdatei; 2 ungültige Kommandozeile (auch `-o` gleich der Eingabedatei); 3 GPU-Backend nicht verfügbar.

### Textausgabe und Beispiel

Kopf (Datei, Backend, Quadratur, Strahlversagen, Glaselemente, Hülle, k, Kombinationen „gerechnet von insgesamt“), pro Element die kontinuierlichen Koordinaten, der Abstand zur Hülle und die Kandidaten mit Abstand, die Tabelle der Läufe (Gläser, Summe d², Merit eingerastet ohne Reoptimierung, Merit reoptimiert, Zyklen, Status, versagte Strahlen, Markierung `bester`), das Ergebnis (gewählte Gläser, Merit kontinuierlich / eingerastet / reoptimiert, Verschlechterung absolut und relativ), die Variablen ohne Glas vorher/nachher und die RMS-Wellenfront pro Feld und Wellenlänge vorher (kontinuierliches System) und nachher (Ergebnis). Zahlenformat wie bei `optimize`.

Beispiel: Achromat mit zwei freien Gläsern (`data/systems/achromat_glasvariablen.json`, frei erfunden, Testgläser `data/glass/synthetisch.json`), erst kontinuierlich optimieren, dann einrasten (aus dem Repository-Wurzelverzeichnis):

```
mikroptim optimize data/systems/achromat_glasvariablen.json -o build/achromat_opt.json --max-iterations 5000 --quiet
mikroptim snap build/achromat_opt.json -o build/achromat_snap.json
```

```
Glas-Einrasten: Achromat mit zwei Glasvariablen
  Datei                                       build/achromat_opt.json
  Ergebnisdatei                               build/achromat_snap.json
  Backend                                     CPU, FP64
  Quadratur                                   10 Ringe x 8 Speichen
  Strahlversagen                              weich, w_fail ab 1.000e+06 Wellen² (Strafverfahren x10 bis 1e14)
  Jacobi-Matrix                               finite Differenzen (zentral)
  Glaselemente                                2
  Glas-Hülle                                  39 Gläser, Dimension 3
  Nachbarn je Element                         3
  Kombinationen (Läufe)                       9 von 9 (alle)
  Zyklen höchstens                            100

Kandidaten (nächste Katalog-Gläser, Abstand im normierten (n0, nu1, nu2)-Raum der Glas-Hülle)
  Fläche 1: kontinuierlich n0 = 1.5186688, nu1 = -4.991144e-02, nu2 = 1.872511e-02, Abstand zur Hülle 0.000e+00
       0  SYN-10        3.685541e-02
       1  N-BK7         6.339631e-02
       2  SYN-07        6.632414e-02
  Fläche 2: kontinuierlich n0 = 1.6617284, nu1 = -1.223474e-01, nu2 = 6.229057e-02, Abstand zur Hülle 0.000e+00
       0  SYN-21        2.905891e-03
       1  SYN-19        3.503752e-02
       2  SYN-24        4.447560e-02

Kombinationen (sortiert nach Summe d²; Merit in Wellen² bei 546.070 nm)
  Lauf  Gläser                          Summe d²      eingerastet      reoptimiert  Zyklen  Status             versagt
     0  SYN-10, SYN-21                 1.367e-03  2.277735998e+03  4.108101971e-05     100  max_iterations           0
     1  SYN-10, SYN-19                 2.586e-03  4.155345530e+03  3.795836885e-05      85  converged_merit          0  bester
     2  SYN-10, SYN-24                 3.336e-03  2.989592464e+04  4.219076322e-05     100  max_iterations           0
     3  N-BK7, SYN-21                  4.028e-03  5.152928918e+00  6.545068082e-05      50  converged_merit          0
     4  SYN-07, SYN-21                 4.407e-03  3.045461661e+01  6.417879792e-05     100  max_iterations           0
     5  N-BK7, SYN-19                  5.247e-03  3.613318686e+02  6.524554915e-05     100  max_iterations           0
     6  SYN-07, SYN-19                 5.627e-03  1.259033855e+02  6.625337978e-05     100  max_iterations           0
     7  N-BK7, SYN-24                  5.997e-03  1.624782387e+04  7.033398599e-05     100  max_iterations           0
     8  SYN-07, SYN-24                 6.377e-03  1.432368438e+04  7.108641123e-05     100  max_iterations           0

Ergebnis
  Gewählte Gläser                             Fläche 1: SYN-10, Fläche 2: SYN-19
  Bester Lauf                                 1
  Status                                      converged_merit
  Merit kontinuierlich [Wellen²]              9.169341762e-06
  Merit eingerastet, ohne Reopt. [Wellen²]    4.155345530e+03
  Merit eingerastet, reoptimiert [Wellen²]    3.795836885e-05
  Verschlechterung [Wellen²]                  2.878903e-05
  Verschlechterung relativ                    3.139705e+00
  Versagte Strahlen kontinuierlich/Ergebnis   0 / 0

Variablen (ohne Glas)
  Name    Einheit            vorher          nachher
  c[1]    1/mm      4.179137718e-02  4.271858629e-02
  c[2]    1/mm     -4.184124394e-02 -4.472039306e-02
  c[3]    1/mm     -6.492629399e-03 -5.889697692e-03
  w       mm              46.175926        46.097040

Wellenfront (Quadratur 10 Ringe x 8 Speichen, CPU, FP64, RMS in Wellen bei 546.070 nm; vorher = kontinuierlich, nachher = eingerastet)
  Feld  Winkel [deg]  Wellenlänge [nm]  Strahlen    RMS vorher   RMS nachher  versagt vorher/nachher
     0         0.000           486.130        10      0.002110      0.004522                   0 / 0
     0         0.000           546.070        10      0.003893      0.008682                   0 / 0
     0         0.000           656.270        10      0.002810      0.004248                   0 / 0

Ergebnis geschrieben: build/achromat_snap.json
```

Die Merit „eingerastet, ohne Reoptimierung“ ist groß, weil die Brechzahlen der Katalog-Gläser um bis zu 10⁻² von den kontinuierlichen abweichen: Brennweite (Nebenbedingung mit Gewicht 10⁴/mm²) und Fokus verschieben sich. Die nächste Kombination (Lauf 0) ist nicht die beste; mit `--max-iterations 500` konvergieren mehr Läufe (die Abnahme verwendet 500), die Auswahl bleibt SYN-10/SYN-19. Die Ergebnisdatei lädt mit `mikroptim trace` (f′ = 50,000 mm, RMS wie „nachher“) und liefert mit `mikroptim optimize build/achromat_snap.json --max-iterations 0` die berichtete Merit bitgleich (dasselbe Backend vorausgesetzt; Feldziele ohne `target.focal_length` wie bei `optimize`, der Befehl warnt dann).

### JSON-Ausgabe (`--json`), Format `mikroptim.snap/1`

Zahlen verlustfrei, nicht endliche Werte `null` (wie `mikroptim.optimize/1`).

| Feld | Inhalt |
|---|---|
| `format`, `mikroptim_version`, `system`, `file`, `output`, `backend`, `device`, `precision`, `merit_unit`, `quadrature` | wie bei `optimize` |
| `settings` | `neighbors`, `max_combinations`, `max_iterations`, `failure` (`"soft"`/`"hard"`), `fail_weight`, `jacobian` (`"fd"`/`"adjoint"`, ab M5-T3), `distance_weights` |
| `hull` | `glasses` (Zahl der Gläser der Hülle), `dimension` |
| `elements[]` | pro Glaselement: `surface` (Dateifläche), `coordinates` {`n0`, `nu1`, `nu2`} des kontinuierlichen Ergebnisses, `hull_distance`, `candidates[]` mit `name`, `distance`, `coordinates` (aufsteigend nach Abstand) |
| `total_combinations`, `truncated` | Größe des kartesischen Produkts, ob begrenzt wurde |
| `runs[]` | pro Lauf: `run`, `glasses` (pro Element), `choice` (Kandidatenindex pro Element), `cost` (Σ d²), `snapped_merit`, `snapped_failed_rays` (eingerastet ohne Reoptimierung), `merit`, `failed_rays`, `hard_failed_rays`, `status`, `iterations`, `penalty_rounds`, `fail_weight`, `x` (Variablen ohne Glas) |
| `best_run` | Index des besten Laufs |
| `variables[]` | Variablen ohne Glas: `name`, `type`, `surface`, `start` (kontinuierliches Ergebnis), `result` |
| `result` | `glasses[]` ({`surface`, `glass`}), `status`, `converged`, `continuous_merit`, `snapped_merit`, `merit`, `degradation` (= `merit` − `continuous_merit`), `relative_degradation`, `continuous_failed_rays`, `failed_rays`, `hard_failed_rays` |
| `reference_nm`, `wavefront` {`start`, `result`} oder `null` mit `wavefront_error`, `warnings` | wie bei `optimize` (`start` = kontinuierliches System) |

### Tests

`tests/test_cli_snap.cpp`: `CliSnap.OptimizeThenSnapReproducesMerit` (Achromat über `optimize` kontinuierlich optimiert, dann eingerastet: 9 Läufe, nur Katalog-Gläser und keine Glasvariablen in der Ergebnisdatei, Verschlechterung = Merit − kontinuierlich, bester Lauf mit der kleinsten Merit, Ergebnis bitgleich zu `snap_glasses`, Merit der neu geladenen Datei und RMS von `mikroptim trace` bitgleich; erneutes Einrasten scheitert mit „keine Glasvariablen“), `CliSnap.TextOutputAndLimit` (Abschnitte, „3 von 4 (kleinste Summe d²)“), `CliSnap.AdjointJacobian` (`--jacobian adjoint` bitgleich zu `snap_glasses` mit `JacobianMethod::Adjoint`, M5-T3), `CliSnap.Errors`, `CliSnap.GpuUnavailableWithoutCuda` bzw. im CUDA-Build `CliSnapGpu.MatchesCpuSelection` (gleiche Auswahl, Merit relativ ≤ 10⁻⁶). ctest-Kommandos: `cli_snap_optimize` → `cli_snap_text` → `cli_snap_result_trace` (Beispiel oben). Implementierung: `src/cli/snap_command.cpp`; `same_file`, `utf8` und `catalogs_for` teilen sich `optimize` und `snap` in `src/cli/cli_util.hpp`.

## `mikroptim insert`

Wendet einen Einfüge-Operator der Systemsynthese auf eine Systemdatei an und schreibt das neue System ([optimierung.md, Abschnitt 6](optimierung.md#6-einfüge-operatoren)). Gibt die Merit vorher und nachher aus (CPU, Pupillenquadratur 10 × 8; Strahl- und Strafresiduen, paraxiale Nebenbedingungen und L1-Strafe der Geisterplatten, ohne Glas-Hülle) sowie die paraxiale Brennweite bei λ_ref; beide bleiben bis auf Rundung gleich.

| Option | Bedeutung |
|---|---|
| `--meniscus I,POS,C0,GLAS` | Operator A (`insert_meniscus`): zwei Flächen der Krümmung C0 [1/mm] im Luftspalt nach Dateifläche I, Vorderfläche im Abstand POS [mm] vom Scheitel von I; GLAS ist ein Katalogname oder `n=<Brechzahl>` (konstant) |
| `--thickness T` | Mittendicke des Meniskus [mm] (nur mit `--meniscus`; Standard 0, dann ändert sich die Merit nicht) |
| `--split I,C_S[,ANTEIL]` | Operator B (`split_lens`): Linse aus Fläche I und I + 1 an einer Kittfläche der Krümmung C_S teilen, ANTEIL der Mittendicke vor der Kittfläche (Standard 0,5) |
| `--free-glass` | nur mit `--split`: Glas des hinteren Teils als Variable `glass` freigeben (`free_glass`; `mikroptim optimize` wandelt es beim Laden in Buchdahl um) |
| `--no-variables` | keine Krümmungsvariablen für die neuen Flächen (Standard: eigener Eintrag `curvature` ohne Schranken) |
| `-o DATEI`, `--output DATEI` | neues System (Pflicht); Glaskataloge wie bei `optimize` relativ zum Verzeichnis der Ergebnisdatei; die Eingabedatei wird nie überschrieben (Exit-Code 2) |

Genau einer von `--meniscus` und `--split` ist nötig. Exit-Codes wie bei `trace`: 0 Ergebnis geschrieben, 1 Datei- oder Formatfehler (z. B. Luftspalt ohne Luft, Glas unbekannt, Teilstück unter der Schranke einer Dickenvariable), 2 ungültige Kommandozeile.

Beispiel (Operator B auf die frei erfundene Einzellinse, c_s für ein dünnes Achromat mit SYN-30):

```
$ mikroptim insert data/systems/einzellinse_achromat_start.json --split 1,-0.03076,0.5 --free-glass -o kitt.json
mikroptim insert: Operator B (Kittglied)
  Eingefügte Flächen                          2
  Flächen vorher / nachher                    3 / 4
  Merit vorher [Wellen²]                      3.639291160270993e+03
  Merit nachher [Wellen²]                     3.639291160270992e+03
  relative Änderung                           1.25e-16
  Brennweite vorher / nachher [mm]            50.610619298 / 50.610619298
Ergebnis geschrieben: kitt.json
```

**Tests:** `tests/test_cli_insert.cpp` (Ergebnisdatei gleich der Bibliothek, Merit unverändert, Kataloge aus einem anderen Verzeichnis auffindbar, konstantes Glas und Dicke, Fehlerfälle). Implementierung: `src/cli/insert_command.cpp`.

## `mikroptim synth`

Systemsynthese mit dem Homotopie-Controller ([optimierung.md, Abschnitt 7](optimierung.md#festlegungen-der-implementierung-m6-t3)): Ausgehend von der Startdatei (Variablen, Nebenbedingungen, Geisterplatten-Wörterbuch mit L1-Strafe) wird die numerische Apertur in Schritten mit Schrittweitensteuerung bis `--target-na` geöffnet (danach optional der Feldanteil). Ein Schritt ist eine LM-Optimierung; abgelehnte Schritte werden halbiert, unter Δ_min wird bei Bedarf (Diagnose der Geisterplatten) ein Einfüge-Operator angewandt (A: Meniskus mit innerer Homotopie über die Dicke und zwei Starts am Sattelpunkt, B: Kittglied mit freiem Glas). Am Ende ε-Fortsetzung und Prüfung des Beugungsziels am realen System (Geisterkoeffizienten 0). Kollimierte Quellen brauchen `target.focal_length` (Pupillenradius = f · NA).

| Option | Bedeutung |
|---|---|
| `--target-na NA` | numerische Apertur am Ende (Standard: die des Starts, a/f) |
| `--start-field F`, `--target-field F` | Feldanteil (Faktor auf die Feldwinkel der Datei) am Start (Standard 1) bzw. am Ende (Standard: wie am Start) |
| `--rms-target W` | Beugungsziel [Wellen] pro Feld und Wellenlänge (Standard 1/14) |
| `--merit-factor K` | Schwelle F_max = (K · Ziel)² für die Bewertungsmerit F_B = Wellenfront + L1-Strafe (Standard 1) |
| `--stall W[,T]` | LM-Abbruch bei Stagnation (`converged_stall`): F sank in W Zyklen um höchstens T·F (M11-T1c; T Standard 10⁻³). **Standard aus** – auf der Zielspezifikation verschlechterte der Abbruch die Synthese (`20,1e-3`: Aperturstufe bei NA 0,37 abgebrochen; `20,1e-5`: unkontrolliertes Einfügen bis 24 Elemente). Nur für schnelle Erkundungen an kleinen Systemen. Steht im JSON unter `options.stall` |
| `--color-rule` | Farbregel (M11-T5c): Operator B mit Glas aus der Vorauswahl, wenn die Farbe dominiert (RMS einer Wellenlänge ≥ Faktor × RMS bei λ_ref, oder Seidel \|ΔW020\| ≥ 0,25 und ≥ \|W040\|); Standard aus |
| `--color-ratio F` | Faktor der Farbregel (Standard 2; setzt `--color-rule`) |
| `--split-preselect` | Operator B: Zielglas aus der Glasvorauswahl (bester Partner des Linsenglases), c_s aus dem dünnen Achromat in η (M11-T5b) |
| `--split-achromat` | Operator B mit Vorauswahl und Achromat-Start: Kitt- und hintere Außenkrümmung aus den Brechkraftanteilen des dünnen Achromats, freigegebenes Glas startet beim Partnerglas (M11-T5e) |
| `--spectrum-stages N` | Spektrum-Homotopie: von monochromatisch bei λ_ref in N Stufen zum Spektrum der Datei (M11-T5d) |
| `--split-start-target` | wie `--split-preselect`, und das freigegebene Glas startet beim Zielglas |
| `--predictor` | Start jedes Apertur-/Feldschritts aus der Sekante der beiden letzten angenommenen Zustände (M11-T3; nicht mit `--beam`, Standard aus) |
| `--extrema-beta B` | geglättete Strahl-Extrema mit β = B [1/mm] (für die Höhe und dimensionslos für cos I) ab dem Start; am Endpunkt `--extrema-stages` Stufen (2) mit Faktor `--extrema-factor` (10), danach hart, vor der ε-Fortsetzung (M11-T1b; nur CPU, Standard 0 = hart) |
| `--step D0,DMIN,DMAX` | Schrittweite der Apertur (Standard 0.02,0.002,0.1) |
| `--max-elements N` | höchstens N Glaselemente (Standard 6) |
| `--max-steps N` | höchstens N Optimierungen (Standard 400) |
| `--max-iterations N` | LM-Zyklen pro Schritt (Standard 500) |
| `--rings N`, `--spokes M` | Pupillenquadratur (Standard 10 × 8) |
| `--element-thickness T`, `--element-max-thickness T`, `--element-stages J` | Operator A: Mittendicke am Ende der inneren Homotopie (0,5 mm), danach Variable bis T (5 mm; ≤ Dicke: fest), Zahl der Dickenstufen (5) |
| `--min-air-gap G` | kleinster Luftspalt hinter einem eingefügten Meniskus (0,1 mm) |
| `--saddle-delta D` | zwei Starts am Sattelpunkt mit c₀ ± D (0,002/mm; 0: ein Zweig) |
| `--curvature-bound C` | Schranken ±C [1/mm] der neuen Krümmungsvariablen (Standard offen) |
| `-o DATEI` | Ergebnis am Endpunkt als Systemdatei (Pupille, Feldwinkel, Spektrum und Gewichte des Endpunkts, ohne Vorhalt; Kataloge relativ zum Verzeichnis der Datei) |
| `--log DATEI` | Protokoll als JSON `mikroptim.synth/1` in eine Datei (zusätzlich zur Ausgabe) |
| `--json` | Ausgabe als JSON statt Text |
| `--quiet` | Text ohne Schrittprotokoll |
| `--gpu [--device N]` | LM-Läufe auf einem CUDA-Gerät (`optimize_lm_gpu`); Diagnose und Zielprüfung bleiben auf der CPU |
| `--jacobian V` | Jacobi-Matrix der LM-Läufe wie bei `optimize`: `fd` (Standard) oder `adjoint` (ab M5-T3, `LmOptions::jacobian_method`) |

Exit-Codes: 0 Lauf beendet (auch wenn das Ziel verfehlt wurde; der Status steht in der Ausgabe), 1 Datei-, Format- oder Rechenfehler, 2 ungültige Kommandozeile (auch: `-o` oder `--log` gleich der Eingabedatei), 3 GPU-Backend nicht verfügbar.

**Textausgabe.** Kopf (System, Start, Ziel, F_max, Backend, Jacobi-Matrix), danach eine Zeile pro Schritt (`write_homotopy_step_line`): Nummer, Art, Punkt (NA, Feldanteil, Spektrumstufe), Δ, LM-Status und Zyklen, F, F_B, größter RMS mit Geisterplatten [Wellen], Σ|a|, Glaselemente, Entscheidung (`grow`/`keep`/`shrink`/`insert`/`stop` mit dem Δ danach, sonst `ok`) und Bemerkung (Operator, Zweig und Stufe, Gewichtsstufe, Ziel). Am Ende Status, Endpunkt, Elemente, RMS des realen Systems pro Feld und Wellenlänge, Laufzeit.

**Beispiel** (Abnahme von M6-T3, CPU, Release-Build auf dem Rechner des Maintainers):

```
$ mikroptim synth data/systems/einzellinse_homotopie_start.json --target-na 0.6 --curvature-bound 1 -o ergebnis.json --log protokoll.json
mikroptim synth 0.1.0: data/systems/einzellinse_homotopie_start.json
  System                                      Einzellinse NA 0,3 mit Geisterplatten, Start der Homotopie
  Start                                       NA 0.3000, Feldanteil 1.000, 1 Glaselement(e)
  Ziel                                        NA 0.6000, RMS <= 0.07143 Wellen
  Schwelle F_max [Wellen²]                    5.102041e-03
  Backend                                     CPU
  Jacobi-Matrix                               finite Differenzen (zentral)

Schritt  Art              NA      Feld  Sp  Delta     Status            Zyklen  F [W^2]       F_B [W^2]     RMS [W]   Sum|a| [mm]  El  Entscheidung
      0  penalty          0.3000  1.000   0  -        converged_merit       22  1.096582e-01  1.096552e-01  0.05158   2.274e-03   1  ok: Gewichte der einseitigen Bedingungen ×0.01
      1  penalty          0.3000  1.000   0  -        converged_merit        0  1.096582e-01  1.096552e-01  0.05158   2.274e-03   1  ok: Gewichte der einseitigen Bedingungen ×0.1
      2  penalty          0.3000  1.000   0  -        converged_merit        0  1.096582e-01  1.096552e-01  0.05158   2.274e-03   1  ok: Gewichte der einseitigen Bedingungen ×1
      3  insertion        0.3000  1.000   0  -        -                      -  -             -             -         -             1  ok: Operator A: Meniskus nach Fläche 0 bei 1 mm, c0 = 0.124736/mm, Glas buchdahl(n0=1.80668, nu=-0.106159/0.0448234/0.0236138) (Platte 1, mu x10), Dicke bis 0.5 mm, 2 Zweige
      4  insertion_stage  0.3000  1.000   0  -        converged_step       217  2.098751e-02  2.098739e-02  0.05743   3.803e-04   1  ok, Zweig 0, Stufe 0.03125
      5  insertion_stage  0.3000  1.000   0  -        converged_step        67  2.098507e-02  2.098494e-02  0.05740   3.803e-04   1  ok, Zweig 0, Stufe 0.0625
      6  insertion_stage  0.3000  1.000   0  -        converged_step        75  2.098019e-02  2.098007e-02  0.05735   3.803e-04   1  ok, Zweig 0, Stufe 0.125
      7  insertion_stage  0.3000  1.000   0  -        converged_step        85  2.097046e-02  2.097033e-02  0.05725   3.804e-04   1  ok, Zweig 0, Stufe 0.25
      8  insertion_stage  0.3000  1.000   0  -        converged_step        89  2.095078e-02  2.095065e-02  0.05702   3.805e-04   1  ok, Zweig 0, Stufe 0.5
      9  insertion_stage  0.3000  1.000   0  -        converged_step       186  2.098751e-02  2.098739e-02  0.05743   3.803e-04   1  ok, Zweig 1, Stufe 0.03125
     10  insertion_stage  0.3000  1.000   0  -        converged_step        82  2.098507e-02  2.098494e-02  0.05740   3.803e-04   1  ok, Zweig 1, Stufe 0.0625
     11  insertion_stage  0.3000  1.000   0  -        converged_step        81  2.098019e-02  2.098007e-02  0.05735   3.803e-04   1  ok, Zweig 1, Stufe 0.125
     12  insertion_stage  0.3000  1.000   0  -        converged_step        86  2.097046e-02  2.097033e-02  0.05725   3.804e-04   1  ok, Zweig 1, Stufe 0.25
     13  insertion_stage  0.3000  1.000   0  -        converged_step        95  2.095078e-02  2.095065e-02  0.05702   3.805e-04   1  ok, Zweig 1, Stufe 0.5
     14  penalty          0.3000  1.000   0  -        converged_step       222  1.894521e-02  1.894511e-02  0.05688   3.387e-04   2  ok: Zweig 1 von 2 übernommen; Gewichte der einseitigen Bedingungen ×0.01
     15  penalty          0.3000  1.000   0  -        converged_step         7  1.894521e-02  1.894511e-02  0.05688   3.387e-04   2  ok: Gewichte der einseitigen Bedingungen ×0.1
     16  penalty          0.3000  1.000   0  -        converged_step         7  1.894521e-02  1.894511e-02  0.05688   3.387e-04   2  ok: Gewichte der einseitigen Bedingungen ×1
     17  insertion        0.3000  1.000   0  -        -                      -  -             -             -         -             2  ok: Operator A: Meniskus nach Fläche 0 bei 1 mm, c0 = 0.0633467/mm, Glas N-BK7 (Platte 0, mu x10), Dicke bis 0.5 mm, 2 Zweige
     18  insertion_stage  0.3000  1.000   0  -        max_iterations       500  2.962288e-04  2.712286e-04  0.01622   1.612e-06   2  ok, Zweig 0, Stufe 0.03125
     19  insertion_stage  0.3000  1.000   0  -        converged_step       380  2.964099e-04  2.714098e-04  0.01623   1.613e-06   2  ok, Zweig 0, Stufe 0.0625
     20  insertion_stage  0.3000  1.000   0  -        converged_step       150  2.969464e-04  2.719463e-04  0.01624   1.617e-06   2  ok, Zweig 0, Stufe 0.125
     21  insertion_stage  0.3000  1.000   0  -        converged_step       314  2.980150e-04  2.730149e-04  0.01627   1.626e-06   2  ok, Zweig 0, Stufe 0.25
     22  insertion_stage  0.3000  1.000   0  -        converged_step       488  3.001348e-04  2.751348e-04  0.01634   1.642e-06   2  ok, Zweig 0, Stufe 0.5
     23  insertion_stage  0.3000  1.000   0  -        converged_step       136  1.979290e-02  1.976770e-02  0.07950   2.897e-04   2  ok, Zweig 1, Stufe 0.03125
     24  insertion_stage  0.3000  1.000   0  -        converged_step        61  1.976504e-02  1.973984e-02  0.07933   2.897e-04   2  ok, Zweig 1, Stufe 0.0625
     25  insertion_stage  0.3000  1.000   0  -        converged_step        57  1.971051e-02  1.968530e-02  0.07898   2.897e-04   2  ok, Zweig 1, Stufe 0.125
     26  insertion_stage  0.3000  1.000   0  -        converged_step        58  1.960612e-02  1.958092e-02  0.07832   2.897e-04   2  ok, Zweig 1, Stufe 0.25
     27  insertion_stage  0.3000  1.000   0  -        converged_step        97  1.941527e-02  1.939007e-02  0.07708   2.897e-04   2  ok, Zweig 1, Stufe 0.5
     28  penalty          0.3000  1.000   0  -        converged_step       188  2.759722e-04  2.757220e-04  0.01635   1.646e-06   3  ok: Zweig 0 von 2 übernommen; Gewichte der einseitigen Bedingungen ×0.01
     29  penalty          0.3000  1.000   0  -        converged_step       233  2.782030e-04  2.757029e-04  0.01635   1.646e-06   3  ok: Gewichte der einseitigen Bedingungen ×0.1
     30  penalty          0.3000  1.000   0  -        converged_step        55  3.007025e-04  2.757025e-04  0.01635   1.646e-06   3  ok: Gewichte der einseitigen Bedingungen ×1
     31  aperture         0.3200  1.000   0  0.02000  max_iterations       500  4.796765e-04  4.546763e-04  0.02099   2.156e-06   3  keep (Delta 0.02000)
     32  aperture         0.3400  1.000   0  0.02000  max_iterations       500  7.583469e-04  7.333463e-04  0.02663   2.825e-06   3  keep (Delta 0.02000)
     33  aperture         0.3600  1.000   0  0.02000  max_iterations       500  1.176571e-03  1.151569e-03  0.03331   3.732e-06   3  keep (Delta 0.02000)
     34  aperture         0.3800  1.000   0  0.02000  max_iterations       500  1.788313e-03  1.763309e-03  0.04106   5.096e-06   3  keep (Delta 0.02000)
     35  aperture         0.4000  1.000   0  0.02000  max_iterations       500  2.623513e-03  2.598500e-03  0.04911   8.091e-06   3  keep (Delta 0.02000)
     36  aperture         0.4200  1.000   0  0.02000  max_iterations       500  3.691754e-03  3.666726e-03  0.05154   2.579e-05   3  keep (Delta 0.02000)
     37  aperture         0.4400  1.000   0  0.02000  max_iterations       500  4.924733e-03  4.899673e-03  0.05159   5.176e-05   3  keep (Delta 0.02000)
     38  aperture         0.4600  1.000   0  0.02000  max_iterations       500  6.355247e-03  6.329754e-03  0.05166   8.187e-05   3  shrink (Delta 0.01000)
     39  aperture         0.4500  1.000   0  0.01000  max_iterations       500  5.579714e-03  5.554647e-03  0.05161   6.558e-05   3  shrink (Delta 0.00500)
     40  aperture         0.4450  1.000   0  0.00500  max_iterations       500  5.206054e-03  5.180997e-03  0.05160   5.770e-05   3  shrink (Delta 0.00250)
     41  aperture         0.4425  1.000   0  0.00250  max_iterations       500  5.032468e-03  5.007416e-03  0.05159   5.405e-05   3  keep (Delta 0.00250)
     42  aperture         0.4450  1.000   0  0.00250  max_iterations       500  5.151708e-03  5.126649e-03  0.05160   5.655e-05   3  insert (Delta 0.00125): Operator A (Meniskus ohne Brechkraft) nach Fläche 6 im Abstand 0.5 mm, Glas wie Platte 3 (buchdahl(n0=1.80668, nu=-0.106159/0.0448234/0.0236138)), dominante Ordnung 4; Biegung aus dem Vorzeichen der Korrektur
     43  insertion        0.4425  1.000   0  -        -                      -  -             -             -         -             3  ok: Operator A: Meniskus nach Fläche 6 bei 0.5 mm, c0 = 0.151146/mm, Glas buchdahl(n0=1.80668, nu=-0.106159/0.0448234/0.0236138) (Platte 3, mu x10), Dicke bis 0.5 mm, 2 Zweige
     44  insertion_stage  0.4425  1.000   0  -        max_iterations       500  5.438332e-03  5.412719e-03  0.07249   6.075e-06   3  ok, Zweig 0, Stufe 0.03125
     45  insertion_stage  0.4425  1.000   0  -        max_iterations       500  4.784563e-03  4.755219e-03  0.06813   4.985e-06   3  ok, Zweig 0, Stufe 0.0625
     46  insertion_stage  0.4425  1.000   0  -        max_iterations       500  4.302526e-03  4.276961e-03  0.06469   4.447e-06   3  ok, Zweig 0, Stufe 0.125
     47  insertion_stage  0.4425  1.000   0  -        max_iterations       500  4.208372e-03  4.182769e-03  0.06398   4.366e-06   3  ok, Zweig 0, Stufe 0.25
     48  insertion_stage  0.4425  1.000   0  -        max_iterations       500  2.584377e-03  2.558942e-03  0.05014   3.005e-06   3  ok, Zweig 0, Stufe 0.5
     49  insertion_stage  0.4425  1.000   0  -        max_iterations       500  4.067704e-03  4.042176e-03  0.06292   4.209e-06   3  ok, Zweig 1, Stufe 0.03125
     50  insertion_stage  0.4425  1.000   0  -        max_iterations       500  3.704260e-03  3.678773e-03  0.06005   3.878e-06   3  ok, Zweig 1, Stufe 0.0625
     51  insertion_stage  0.4425  1.000   0  -        max_iterations       500  3.426491e-03  3.401011e-03  0.05776   3.650e-06   3  ok, Zweig 1, Stufe 0.125
     52  insertion_stage  0.4425  1.000   0  -        max_iterations       500  3.237148e-03  3.211665e-03  0.05614   3.503e-06   3  ok, Zweig 1, Stufe 0.25
     53  insertion_stage  0.4425  1.000   0  -        max_iterations       500  3.666648e-03  3.641028e-03  0.05974   3.892e-06   3  ok, Zweig 1, Stufe 0.5
     54  penalty          0.4425  1.000   0  -        max_iterations       500  3.331250e-06  3.081250e-06  0.00175   4.554e-08   4  ok: Zweig 0 von 2 übernommen; Gewichte der einseitigen Bedingungen ×0.01
     55  penalty          0.4425  1.000   0  -        max_iterations       500  5.387893e-06  2.887892e-06  0.00169   4.404e-08   4  ok: Gewichte der einseitigen Bedingungen ×0.1
     56  penalty          0.4425  1.000   0  -        max_iterations       500  2.770950e-05  2.709497e-06  0.00164   4.259e-08   4  ok: Gewichte der einseitigen Bedingungen ×1
     57  aperture         0.4625  1.000   0  0.02000  max_iterations       500  2.997929e-05  4.979294e-06  0.00222   5.844e-08   4  keep (Delta 0.02000)
     58  aperture         0.4825  1.000   0  0.02000  max_iterations       500  3.509171e-05  1.009171e-05  0.00317   8.424e-08   4  keep (Delta 0.02000)
     59  aperture         0.5025  1.000   0  0.02000  max_iterations       500  4.423192e-05  1.923191e-05  0.00437   1.176e-07   4  keep (Delta 0.02000)
     60  aperture         0.5225  1.000   0  0.02000  max_iterations       500  6.026233e-05  3.526230e-05  0.00592   1.612e-07   4  keep (Delta 0.02000)
     61  aperture         0.5425  1.000   0  0.02000  max_iterations       500  8.709887e-05  6.209880e-05  0.00785   2.167e-07   4  keep (Delta 0.02000)
     62  aperture         0.5625  1.000   0  0.02000  max_iterations       500  1.320080e-04  1.070078e-04  0.01031   2.889e-07   4  keep (Delta 0.02000)
     63  aperture         0.5825  1.000   0  0.02000  max_iterations       500  2.044409e-04  1.794404e-04  0.01335   3.817e-07   4  keep (Delta 0.02000)
     64  aperture         0.6000  1.000   0  0.02000  max_iterations       500  2.998484e-04  2.748472e-04  0.01652   4.814e-07   4  keep (Delta 0.02000)
     65  smoothing        0.6000  1.000   0  -        max_iterations       500  2.915273e-04  2.665261e-04  0.01632   4.750e-08   4  ok: ε ×0.1
     66  goal             0.6000  1.000   0  -        -                      -  -             -             -         -             4  ok: Ziel erreicht: größter RMS 0.01633 Wellen (Ziel 0.07143)

Status: goal_reached (Ziel erreicht)
Endpunkt: NA 0.6000, Feldanteil 1.000, Spektrumstufe 0, mu x1, eps x0.1 (Ziel NA 0.6000)
Glaselemente: 1 -> 4 (3 Einfügungen), Optimierungen: 63, F_max = 5.102041e-03 Wellen^2
Reales System (Geisterkoeffizienten 0), RMS-Wellenfront pro Feld und Wellenlänge (Ziel 0.07143 Wellen):
  Feld 0 (0.0000 deg), 546.07 nm: RMS 0.01633 Wellen, versagt 0
  Laufzeit [s]                                19.2
Ergebnis geschrieben: ergebnis.json
Protokoll geschrieben: protokoll.json
```

Mit `--gpu` (RTX 3090) dieselben Entscheidungen, RMS am Ende 0,01566 Wellen, 35 s. `mikroptim trace ergebnis.json` liefert 0,016320 Wellen (die Ergebnisdatei enthält die Restkoeffizienten der Geisterplatten von ~10⁻⁸ mm, die Zielprüfung setzt sie auf 0).

**JSON** (`--json` bzw. `--log`), Format `mikroptim.synth/1`: `format`, `version`, `input`, `backend` (`gpu`, mit `--gpu` `device` und `device_name`), `options` (Ziel-NA, Startpunkt, Ziel-RMS, Faktor, Elemente, Zyklen, Quadratur, `jacobian` (`"fd"`/`"adjoint"`, ab M5-T3), Einstellungen des Operators A), `result` und `output` (Ergebnisdatei oder `null`). `result`: `status` (`goal_reached`, `goal_not_reached`, `step_too_small`, `max_elements`, `max_steps`, `start_rejected`), `message`, `start_point`, `target`, `point` (je `aperture`, `field`, `spectrum`, `ghost_weight`, `ghost_smoothing`, `penalty_scale`), `merit_threshold`, `start_elements`, `elements`, `insertions`, `solves`, `goal` (`reached`, `target_waves`, `max_rms_waves`, `tuples` mit Feld, Wellenlänge, Winkel, λ, `rms_waves`, `failed_rays`), `seconds` und `steps`. Ein Schritt (hier der Aperturschritt 42, der zum Einfügen führte):

```json
{
  "index": 42,
  "kind": "aperture",
  "point": {
    "aperture": 0.4450000000000001,
    "field": 1.0,
    "spectrum": 0,
    "ghost_weight": 1.0,
    "ghost_smoothing": 1.0,
    "penalty_scale": 1.0
  },
  "delta": 0.0025,
  "branch": null,
  "stage": null,
  "lm": {
    "status": "max_iterations",
    "iterations": 500,
    "start_merit": 0.6153761907875889,
    "merit": 0.005151708173780439,
    "wavefront": 0.0026628403204233793,
    "ghost_penalty": 0.0024638085090079253,
    "rms_waves": 0.051602716211681915,
    "ghost_sum_mm": 5.65538305661399e-05,
    "failed_rays": 0
  },
  "accepted": false,
  "action": "insert",
  "next_delta": 0.00125,
  "recommendation": "A",
  "operation": "Operator A (Meniskus ohne Brechkraft) nach Fläche 6 im Abstand 0.5 mm, Glas wie Platte 3 (buchdahl(n0=1.80668, nu=-0.106159/0.0448234/0.0236138)), dominante Ordnung 4; Biegung aus dem Vorzeichen der Korrektur",
  "elements": 3,
  "goal": null,
  "seconds": 0.4285371
}
```

Nicht endliche Zahlen stehen als `null`; `seconds` ist der einzige nicht deterministische Wert.

### Strahlsuche (`--beam`, ab M6-T4)

Mit `--beam B` läuft statt des einzweigigen Controllers die Strahlsuche ([optimierung.md, Abschnitt 8](optimierung.md#festlegungen-der-implementierung-m6-t4)): bis zu B Kandidaten im Gleichschritt am selben Homotopie-Punkt, Kandidaten gleicher Topologie in einem gemeinsamen LM-Aufruf; beim Einfügen entstehen Kinder (Varianten × Sattelzweige × gestörte Starts), behalten werden die besten B nach Rang (annehmbar, weniger Elemente, kleineres F), Duplikate fallen heraus. `--beam 1` trifft dieselben Entscheidungen wie der Lauf ohne `--beam`. Die übrigen Optionen gelten wie oben; `--max-steps` zählt die Optimierungen aller Kandidaten.

| Option | Bedeutung |
|---|---|
| `--beam B` | Breite des Strahls (≥ 1) |
| `--alternatives K` | Varianten pro Einfügen aus der Diagnose: Empfehlung, weitere Lagen, andere Gläser (Standard 1) |
| `--starts N` | gestörte Starts pro Sattelzweig (Krümmungen der neuen Flächen ± S, Standard 0) |
| `--start-perturbation S` | Störung S [1/mm] (Standard 0,01) |
| `--seed N` | Startwert der Zufallszahlen der gestörten Starts (Standard 0); gleiche Eingabe und gleicher Startwert ergeben dasselbe Protokoll |
| `--prefilter` | FP32-Vorfilter der Starts eines Einfügens (CPU `prefilter_cpu`, mit `--gpu` auf dem Gerät) |
| `--dedup T` | Duplikat bei größter relativer Variablenabweichung ≤ T (Standard 10⁻⁴; negativ: aus) |
| `--dot DATEI` | Stammbaum als Graphviz-DOT (`dot -Tsvg DATEI -o stammbaum.svg`) |

Diese Optionen ohne `--beam` sind ein Fehler der Kommandozeile (Exit-Code 2).

**Textausgabe.** Kopf mit der Zeile `Strahlsuche`, Schrittprotokoll mit zwei Spalten vorweg: Kandidat (`#n`) und Nummer des gemeinsamen LM-Aufrufs (`Gr`; gleiche Nummer = derselbe Aufruf, `-` ohne Optimierung). Am Ende Status, Kennzahlen der Strahlsuche (Kandidaten, Einfügungen, Optimierungen in Aufrufen), Ergebnis-Kandidat und Strahl, Endpunkt, RMS des realen Systems und der **Stammbaum**: ein Kandidat pro Zeile, eingerückt unter seinem Elternkandidaten, mit Herkunft (Operator, Variante, Zweig, Start), Punkt der Entstehung, letzter Merit F und F_B mit ihrem Punkt, Elementzahl, Zahl der Optimierungen und Status (`behalten`, `fertig`, `verworfen`, `expandiert`) mit Grund.

**Beispiel** (Kurzlauf wie im Test, B = 2, ein gestörter Start pro Zweig, Vorfilter, Startwert 7; CPU, Release):

```
$ mikroptim synth data/systems/einzellinse_homotopie_start.json --target-na 0.31 --merit-factor 10 --max-elements 2 --max-iterations 40 --element-stages 2 --beam 2 --starts 1 --prefilter --seed 7 --dot stammbaum.dot
mikroptim synth 0.1.0: data/systems/einzellinse_homotopie_start.json
  System                                      Einzellinse NA 0,3 mit Geisterplatten, Start der Homotopie
  Start                                       NA 0.3000, Feldanteil 1.000, 1 Glaselement(e)
  Ziel                                        NA 0.3100, RMS <= 0.07143 Wellen
  Schwelle F_max [Wellen²]                    5.102041e-01
  Backend                                     CPU
  Strahlsuche                                 B = 2, Seed 7, Varianten 1, gestörte Starts 1, Vorfilter

Kand  Gr    Schritt  Art              NA      Feld  Sp  Delta     Status            Zyklen  F [W^2]       F_B [W^2]     RMS [W]   Sum|a| [mm]  El  Entscheidung
  #0     0        0  penalty          0.3000  1.000   0  -        converged_merit       22  1.096582e-01  1.096552e-01  0.05158   2.274e-03   1  ok: Gewichte der einseitigen Bedingungen ×0.01
  #0     1        1  penalty          0.3000  1.000   0  -        converged_merit        0  1.096582e-01  1.096552e-01  0.05158   2.274e-03   1  ok: Gewichte der einseitigen Bedingungen ×0.1
  #0     2        2  penalty          0.3000  1.000   0  -        converged_merit        0  1.096582e-01  1.096552e-01  0.05158   2.274e-03   1  ok: Gewichte der einseitigen Bedingungen ×1
  #0     3        3  aperture         0.3100  1.000   0  0.02000  converged_merit       27  1.262922e-01  1.262881e-01  0.05772   2.613e-03   1  grow (Delta 0.04000)
  #0     4        4  smoothing        0.3100  1.000   0  -        max_iterations        40  1.264322e-01  1.264281e-01  0.05902   2.606e-03   1  ok: ε ×0.1
  #0     -        5  goal             0.3100  1.000   0  -        -                      -  -             -             -         -             1  -: Ziel verfehlt: größter RMS 1.65734 Wellen (Ziel 0.07143)
  #0     5        6  ghost_weight     0.3100  1.000   0  -        converged_merit       15  1.139859e+00  1.139485e+00  0.34368   2.165e-03   1  ok: μ ×10
  #0     -        7  goal             0.3100  1.000   0  -        -                      -  -             -             -         -             1  -: Ziel verfehlt: größter RMS 1.67530 Wellen (Ziel 0.07143)
  #0     6        8  ghost_weight     0.3100  1.000   0  -        converged_merit       14  3.124067e+00  3.115783e+00  1.76509   3.832e-07   1  ok: μ ×100
  #0     -        9  goal             0.3100  1.000   0  -        -                      -  -             -             -         -             1  -: Ziel verfehlt: größter RMS 1.76519 Wellen (Ziel 0.07143)
  #0     -       10  insertion        0.3100  1.000   0  -        -                      -  -             -             -         -             1  ok: Operator A: Meniskus nach Fläche 0 bei 1 mm, c0 = 0.124016/mm, Glas buchdahl(n0=1.80668, nu=-0.106159/0.0448234/0.0236138) (Platte 1, mu x10), Dicke bis 0.5 mm, 2 Zweige
  #1     7       11  insertion_stage  0.3100  1.000   0  -        max_iterations        40  1.702664e-01  1.701149e-01  0.08005   3.475e-03   2  ok, Zweig 0, Stufe 0.25
  #2     7       12  insertion_stage  0.3100  1.000   0  -        max_iterations        40  1.867512e-01  1.864512e-01  0.06943   3.855e-03   2  ok, Zweig 0, Stufe 0.25
  #3     7       13  insertion_stage  0.3100  1.000   0  -        max_iterations        40  1.849765e-01  1.814322e-01  0.10296   3.626e-03   2  ok, Zweig 1, Stufe 0.25
  #4     7       14  insertion_stage  0.3100  1.000   0  -        max_iterations        40  1.234804e-01  1.088021e-01  0.07241   2.201e-03   2  ok, Zweig 1, Stufe 0.25
  #1     8       15  insertion_stage  0.3100  1.000   0  -        max_iterations        40  2.447021e-02  2.447005e-02  0.05528   4.592e-04   2  ok, Zweig 0, Stufe 0.5
  #2     8       16  insertion_stage  0.3100  1.000   0  -        max_iterations        40  1.330009e-02  1.330009e-02  0.06637   1.938e-04   2  ok, Zweig 0, Stufe 0.5
  #3     8       17  insertion_stage  0.3100  1.000   0  -        max_iterations        40  2.444814e-02  2.444799e-02  0.05538   4.585e-04   2  ok, Zweig 1, Stufe 0.5
  #4     8       18  insertion_stage  0.3100  1.000   0  -        max_iterations        40  1.249154e-02  1.249142e-02  0.06554   1.789e-04   2  ok, Zweig 1, Stufe 0.5
  #4     9       19  penalty          0.3100  1.000   0  -        max_iterations        40  1.243282e-02  1.243264e-02  0.06573   1.770e-04   2  ok: Kandidat #4 (Operator A, Zweig 1, Start 1) übernommen; Gewichte der einseitigen Bedingungen ×0.01
  #2     9       20  penalty          0.3100  1.000   0  -        max_iterations        40  1.243282e-02  1.243264e-02  0.06573   1.770e-04   2  ok: Kandidat #2 (Operator A, Zweig 0, Start 1) übernommen; Gewichte der einseitigen Bedingungen ×0.01
  #4    10       21  penalty          0.3100  1.000   0  -        max_iterations        40  1.243282e-02  1.243263e-02  0.06573   1.770e-04   2  ok: Gewichte der einseitigen Bedingungen ×0.1
  #2    10       22  penalty          0.3100  1.000   0  -        max_iterations        40  1.243282e-02  1.243263e-02  0.06573   1.770e-04   2  ok: Gewichte der einseitigen Bedingungen ×0.1
  #4    11       23  penalty          0.3100  1.000   0  -        converged_merit       13  1.243282e-02  1.243263e-02  0.06573   1.770e-04   2  ok: Gewichte der einseitigen Bedingungen ×1
  #2    11       24  penalty          0.3100  1.000   0  -        converged_step        37  1.243282e-02  1.243263e-02  0.06573   1.770e-04   2  ok: Gewichte der einseitigen Bedingungen ×1
  #2    12       25  smoothing        0.3100  1.000   0  -        max_iterations        40  1.254910e-02  1.254893e-02  0.06587   1.745e-04   2  ok: ε ×0.1
  #2     -       26  goal             0.3100  1.000   0  -        -                      -  -             -             -         -             2  -: Ziel verfehlt: größter RMS 0.13790 Wellen (Ziel 0.07143)
  #2    13       27  ghost_weight     0.3100  1.000   0  -        max_iterations        40  1.272578e-02  1.272523e-02  0.11279   1.033e-07   2  ok: μ ×10
  #2     -       28  goal             0.3100  1.000   0  -        -                      -  -             -             -         -             2  -: Ziel verfehlt: größter RMS 0.11282 Wellen (Ziel 0.07143)
  #2    14       29  ghost_weight     0.3100  1.000   0  -        max_iterations        40  9.259787e-03  9.259518e-03  0.09623   6.455e-09   2  ok: μ ×100
  #2     -       30  goal             0.3100  1.000   0  -        -                      -  -             -             -         -             2  -: Ziel verfehlt: größter RMS 0.09623 Wellen (Ziel 0.07143)

Status: max_elements (Ziel am Endpunkt verfehlt (größter RMS 0.09623 Wellen); Bedarf (Operator A), aber höchste Elementzahl 2 erreicht)
Strahlsuche: B = 2, Seed 7, 5 Kandidaten, 1 Einfügungen, 24 Optimierungen in 15 Aufrufen
Ergebnis: Kandidat #2, Strahl am Ende: #2
Endpunkt: NA 0.3100, Feldanteil 1.000, Spektrumstufe 0, mu x100, eps x0.1 (Ziel NA 0.3100)
Glaselemente: 1 -> 2, F_max = 5.102041e-01 Wellen^2
Reales System (Geisterkoeffizienten 0), RMS-Wellenfront pro Feld und Wellenlänge (Ziel 0.07143 Wellen):
  Feld 0 (0.0000 deg), 546.07 nm: RMS 0.09623 Wellen, versagt 0

Stammbaum (Kandidat, Herkunft @ Punkt der Entstehung: letzte Merit @ Punkt, Elemente, Optimierungen [Status]):
#0 start @ NA 0.3000, Feld 1.000, Sp 0: F 3.1241e+00, F_B 3.1158e+00 @ NA 0.3100, Feld 1.000, Sp 0, El 1, 7 Opt. [expandiert: 4 Kinder]
|-- #1 Operator A, Zweig 0 @ NA 0.3100, Feld 1.000, Sp 0: F 2.4470e-02, F_B 2.4470e-02 @ NA 0.3100, Feld 1.000, Sp 0, El 2, 2 Opt. [verworfen: Rang hinter den besten 2]
|-- #2 Operator A, Zweig 0, Start 1 @ NA 0.3100, Feld 1.000, Sp 0: F 9.2598e-03, F_B 9.2595e-03 @ NA 0.3100, Feld 1.000, Sp 0, El 2, 8 Opt. [behalten: Ziel am Endpunkt verfehlt (größter RMS 0.09623 Wellen); Bedarf (Operator A), aber höchste Elementzahl 2 erreicht; Ergebnis]
|-- #3 Operator A, Zweig 1 @ NA 0.3100, Feld 1.000, Sp 0: F 2.4448e-02, F_B 2.4448e-02 @ NA 0.3100, Feld 1.000, Sp 0, El 2, 2 Opt. [verworfen: Rang hinter den besten 2]
`-- #4 Operator A, Zweig 1, Start 1 @ NA 0.3100, Feld 1.000, Sp 0: F 1.2433e-02, F_B 1.2433e-02 @ NA 0.3100, Feld 1.000, Sp 0, El 2, 5 Opt. [verworfen: Duplikat von #2 (Abstand 1.67e-05)]

  Laufzeit [s]                                1.0
Stammbaum geschrieben: stammbaum.dot
```

Die vier Kinder des Einfügens (zwei Sattelzweige mit je einem gestörten Start) laufen in jeder Stufe der inneren Homotopie in **einem** LM-Aufruf (Gruppe 7, dann 8); die gestörten Starts #2 und #4 finden ein besseres Minimum als die Sattelzweige (F 1,2·10⁻² gegen 2,4·10⁻²) und werden behalten, nach der Penalty-Fortsetzung liegen beide im selben Minimum, #4 fällt als Duplikat heraus. Ohne gestörte Starts endet derselbe Kurzlauf mit RMS 0,227 Wellen. Die DOT-Datei:

```
digraph stammbaum {
  rankdir=TB;
  node [shape=box, fontname="Helvetica"];
  n0 [label="#0 start\nNA 0.3000, Feld 1.000, Sp 0\nEl 1, F 3.124e+00\nexpandiert", style=filled, fillcolor=lightblue];
  n1 [label="#1 Operator A, Zweig 0\nNA 0.3100, Feld 1.000, Sp 0\nEl 2, F 2.447e-02\nverworfen", style=filled, fillcolor=lightgray];
  n2 [label="#2 Operator A, Zweig 0, Start 1\nNA 0.3100, Feld 1.000, Sp 0\nEl 2, F 9.260e-03\nbehalten", style=filled, fillcolor=khaki, penwidth=3];
  n3 [label="#3 Operator A, Zweig 1\nNA 0.3100, Feld 1.000, Sp 0\nEl 2, F 2.445e-02\nverworfen", style=filled, fillcolor=lightgray];
  n4 [label="#4 Operator A, Zweig 1, Start 1\nNA 0.3100, Feld 1.000, Sp 0\nEl 2, F 1.243e-02\nverworfen", style=filled, fillcolor=lightgray];
  n0 -> n1;
  n0 -> n2;
  n0 -> n3;
  n0 -> n4;
}
```

Stammbaum des längeren Laufs NA 0,3 → 0,6 mit `--beam 2 --alternatives 2 --curvature-bound 1` (Startsystem der Abnahme von M6-T3; Ergebnis mit drei statt vier Elementen, Einordnung in [optimierung.md, Abschnitt 8](optimierung.md#festlegungen-der-implementierung-m6-t4)):

```
#0 start @ NA 0.3000, Feld 1.000, Sp 0: F 1.0966e-01, F_B 1.0966e-01 @ NA 0.3000, Feld 1.000, Sp 0, El 1, 3 Opt. [expandiert: 2 Kinder]
|-- #1 Operator A, Zweig 0 @ NA 0.3000, Feld 1.000, Sp 0: F 2.0951e-02, F_B 2.0951e-02 @ NA 0.3000, Feld 1.000, Sp 0, El 2, 5 Opt. [verworfen: Duplikat von #2 (Abstand 1.27e-05)]
`-- #2 Operator A, Zweig 1 @ NA 0.3000, Feld 1.000, Sp 0: F 1.8945e-02, F_B 1.8945e-02 @ NA 0.3000, Feld 1.000, Sp 0, El 2, 8 Opt. [expandiert: 4 Kinder]
    |-- #3 Operator A, Variante 0, Zweig 0 @ NA 0.3000, Feld 1.000, Sp 0: F 3.0013e-04, F_B 2.7513e-04 @ NA 0.3000, Feld 1.000, Sp 0, El 3, 5 Opt. [verworfen: Rang hinter den besten 2]
    |-- #4 Operator A, Variante 0, Zweig 1 @ NA 0.3000, Feld 1.000, Sp 0: F 1.9415e-02, F_B 1.9390e-02 @ NA 0.3000, Feld 1.000, Sp 0, El 3, 5 Opt. [verworfen: Rang hinter den besten 2]
    |-- #5 Operator A, Variante 1, Zweig 0 @ NA 0.3000, Feld 1.000, Sp 0: F 1.4047e-03, F_B 1.4045e-03 @ NA 0.6000, Feld 1.000, Sp 0, El 3, 24 Opt. [fertig: Ziel erreicht]
    `-- #6 Operator A, Variante 1, Zweig 1 @ NA 0.3000, Feld 1.000, Sp 0: F 1.3520e-03, F_B 1.3518e-03 @ NA 0.6000, Feld 1.000, Sp 0, El 3, 24 Opt. [fertig: Ziel erreicht; Ergebnis]
```

**JSON** (`--json`, `--log`): Format `mikroptim.synth/1`, kompatibel erweitert. `options.beam` (`width`, `seed`, `alternatives`, `perturbed_starts`, `start_perturbation`, `prefilter`, `dedup_tolerance`); `result` wie oben und zusätzlich `beam_search` mit `beam_width`, `seed`, `best` (Kandidat des Ergebnisses), `beam` (Kandidaten am Ende nach Rang), `groups` (LM-Aufrufe) und `candidates` (Stammbaum): pro Kandidat `id`, `parent` (`null` bei der Wurzel), `children`, `operation`, `operator`, `option`, `branch`, `start`, `created` (Punkt), `created_step`, `point`, `merit`, `evaluation` (F_B), `rms_waves`, `failed_rays`, `acceptable`, `elements`, `solves`, `goal`, `status` (`kept`, `finished`, `discarded`, `expanded`), `reason`, `final_step`. Jeder Schritt hat zusätzlich `candidate` und `group` (`null` ohne Optimierung). Ohne die Laufzeiten (`seconds`) ist die Ausgabe zweier Läufe mit gleicher Eingabe und gleichem Startwert bitgleich.

**Tests:** `tests/test_cli_synth.cpp` (Hilfe und Fehler der Kommandozeile, Kurzlauf mit JSON, derselbe Kurzlauf mit `--jacobian adjoint` mit gleichem Status, gleichen Einfügungen und gleichem Endpunkt (M5-T3), Protokolldatei gleich der Ausgabe, Ergebnisdatei am Endpunkt, Textausgabe mit und ohne `--quiet`; Strahlsuche: Optionen nur mit `--beam`, Kurzlauf mit JSON, Stammbaum, Protokoll- und DOT-Datei, Textausgabe mit Baum). Implementierung: `src/cli/synth_command.cpp`.

## `mikroptim aberrations`

Aberrationsbericht pro Fläche (M10-T1, Vollordnung und OSC M10-T1b, Formeln und Prüfungen: [numerik.md, Abschnitt 14](numerik.md#14-aberrationsbericht-m10-t1)): paraxiale Rand- und Hauptstrahlen, Seidel-Beiträge S_I … S_V, chromatische Beiträge C_I, C_II und Sphärochromasie, pro Dateifläche und als Summe, zusätzlich als Wellenfrontkoeffizienten in Wellen bei λ_ref. Nur Systeme mit kollimierter Quelle; Geisterplatten werden nicht berücksichtigt.

```
mikroptim aberrations <system.json> [--json] [--field-deg θ] [--reference-nm λ] [--short-nm λ] [--long-nm λ]
```

| Option | Bedeutung |
|---|---|
| `--json` | Ausgabe im Format `mikroptim.aberrations/1` |
| `--field-deg θ` | Feldwinkel des Hauptstrahls (Standard: größter Feldwinkel der Datei) |
| `--reference-nm λ` | Referenzwellenlänge (Standard: `spectrum.reference_nm`) |
| `--short-nm λ`, `--long-nm λ` | Wellenlängen für C_I, C_II und Sphärochromasie (Standard: kürzeste und längste der Datei) |

Exit-Codes: 0 Erfolg, 1 Fehler beim Laden oder Rechnen (z. B. Punktquelle), 2 ungültige Kommandozeile.

**JSON** (`mikroptim.aberrations/1`): `reference_nm`, `short_nm`, `long_nm`, `field_angle_deg`, `stop_semi_diameter`, `lagrange`, `ghost_plates_ignored`, `sign_convention`, `real_rays_valid`, `zone_rho`, `focal_shifts` (pro Linie `line`, `nm`, `shift_mm`, M10-T2), `totals` (`S` [5], `C` [2], `spherochromatism`, `full_order_spherical`, `higher_order_spherical`, `full_order_spherical_zone`, `osc`, `osc_zone`, `waves`) und `surfaces[]` mit `surface`, `curvature`, `n_before`, `n_after`, `y`, `u_before`, `u_after`, `ybar`, `ubar_before`, `ubar_after`, `A`, `Abar`, `S`, `C`, `spherochromatism`, `full_order_spherical`, `higher_order_spherical`, `full_order_spherical_zone`, `waves` (`W040`, `W131`, `W222`, `W220`, `W311`, `dW020`, `dW111`, `dW040`, `W040_full`, `W040_higher`, `W040_zone`). Nicht definierte Verzeichnung (A = 0) steht als `null`.

**Python:** `mikroptim.aberrations(pfad, field_angle_deg=None, reference_nm=None, short_nm=None, long_nm=None)` liefert dasselbe als `dict`.

**Text** (Ausschnitt, `data/systems/testsystem_mehrlinser.json`):

```
Beiträge in Wellen bei λ_ref (Welford, W = −OPD):
 Fl.           y        W040        W131        W222        W220        W311       ΔW020       ΔW111       ΔW040
   1      4.0000   4.583e-01   7.927e-01   3.428e-01   2.033e-01   3.240e-01   1.554e+00   1.344e+00   1.897e-03
   2      3.8586  -1.169e+00   5.704e-01  -6.958e-02  -6.407e-02   2.412e-02  -8.303e+00   2.025e+00  -7.766e-02
   …
   Σ               1.907e+00  -2.805e+00   9.189e-01   6.879e-01   6.297e-01   1.145e-01   2.819e-01  -6.797e-02
```

## `mikroptim glasses`

Algebraische Glasvorauswahl (M11-T5, [numerik.md, Abschnitt 15](numerik.md#15-algebraische-glasvorauswahl-m11-t5)): Paare (Achromat) bzw. Tripel (Apochromat) dünner Elemente aus den Gläsern der Kataloge, deren gewichtete Brechkraft in den Buchdahl-Termen bis zur Ordnung k farbfrei ist, mit Brechkraftanteilen.

```
mikroptim glasses <katalog.json>... [--system <system.json>] [--order 1|2] [--fixed GLAS]...
                  [--weights w1,w2[,w3]] [--power P] [--max-power-sum S] [--top N] [--json]
```

| Option | Bedeutung | Standard |
|---|---|---|
| `--system DATEI` | Kataloge aus `glass_catalogs` der Systemdatei (zusätzlich zu den Katalogen auf der Kommandozeile) | – |
| `--order K` | 1 Achromat (Paare), 2 Apochromat (Tripel) | 1 |
| `--fixed GLAS` | Glas, das jede Kombination enthält, an erster Stelle (mehrfach möglich) | – |
| `--weights W` | Gewichte w_i = h_i² (Randstrahlhöhen, normiert), k + 1 Werte | alle 1 |
| `--power P` | Gesamtbrechkraft φ [1/mm]; 1 liefert Anteile | 1 |
| `--max-power-sum S` | verwerfen bei Σ\|wφ\|/\|φ\| > S | 10 |
| `--top N` | höchstens N Ergebnisse | 20 |
| `--json` | Format `mikroptim.glasses/1`: `order`, `power`, `glasses` (Zahl mit Fit), `skipped`, `results[]` mit `glasses`, `powers`, `power_sum`, `residual`, `dispersion` (n₀, η) | – |

Beispiel (synthetischer Katalog und Basis, Paare):

```
$ mikroptim glasses data/glass/synthetisch.json data/glass/basis.json --top 3
Glasvorauswahl Achromat (Paare): 39 Gläser, 2 ohne Buchdahl-Fit übersprungen
 Rang  Gläser                                      Rest  Σ|wφ|/|φ|  Brechkräfte
    1  SYN-04 + SYN-07                        2.992e-05      6.669  3.835e+00, -2.835e+00
    2  SYN-04 + SYN-12                        2.228e-04      5.709  3.354e+00, -2.354e+00
    3  SYN-04 + N-BK7                         3.179e-03      6.300  3.650e+00, -2.650e+00
```

Exit-Codes: 0 Erfolg, 1 Fehler beim Laden (auch unbekanntes festes Glas), 2 ungültige Kommandozeile.

