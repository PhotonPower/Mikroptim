# Kommandozeilenprogramm `mikroptim`

```
mikroptim [--version] [--devices]
mikroptim trace <system.json> [--json] [--rings N] [--spokes M] [--gpu [--device N]]
```

| Aufruf | Wirkung |
|---|---|
| `mikroptim --version` | Version ausgeben |
| `mikroptim --devices` | CUDA-Geräte auflisten (nur mit `MIKROPTIM_ENABLE_CUDA=ON`) |
| `mikroptim trace …` | System auswerten (M1-T8, mit `--gpu` M2-T5; dieser Abschnitt) |

Das Programm liegt nach dem Bauen unter `build/<preset>/mikroptim` (Ninja, Makefiles) bzw. `build/<preset>/<Konfiguration>/mikroptim.exe` (Visual Studio).

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
| `--help`, `-h` | Hilfe | – |

Die Pupillenquadratur folgt [numerik.md, Abschnitt 8](numerik.md#8-pupillenquadratur): Auf der Achse ein meridionaler Fächer mit N Stützstellen, außerhalb der Achse die halbe Pupille mit N·M Stützstellen (Punktquelle außerhalb der y-z-Ebene: volle Pupille).

### Exit-Codes und Fehler

| Code | Bedeutung |
|---|---|
| 0 | Auswertung ausgegeben, auch wenn die Wellenfront nicht auswertbar ist (siehe unten) |
| 1 | Fehler beim Laden oder Auswerten: Datei fehlt oder ist nicht lesbar, JSON-Syntax- oder Formatfehler, unbekanntes Glas, Brechzahl außerhalb des Gültigkeitsbereichs, … Mit `--gpu` auch: Gerät `--device N` existiert nicht oder ist ungeeignet (unter sm_60), CUDA-Fehler, Speichermangel |
| 2 | ungültige Kommandozeile (unbekannte Option, fehlende Datei, ungültige Zahl, `--device` ohne `--gpu`) |
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
2. `TraceEngine::run(batch, TraceMode::Residuals, {quadrature = {N, M}, opd = true, status = true})`.
3. `tuple_wavefronts(out.residuals, out.layout, 0)` liefert RMS [mm] und versagte Strahlen pro Feld und Wellenlänge; RMS in Wellen = RMS / λ_ref wie im CPU-Pfad.

Paraxialdaten, Sollbrennweite und Bildpunkte entstehen auf dem Host und sind bitgleich zur CPU-Variante; Text und JSON werden von denselben Ausgabefunktionen geschrieben. Unterschiede zur CPU-Variante:

- `rms_waves`, `rms_mm`: Abweichung höchstens max(`kCpuGpuFp64`, `kCpuGpuFp64Relative` · |RMS|) (`tests/tolerances.hpp`; FMA-Kontraktion und CUDA-Mathematikbibliothek). Gemessen auf der RTX 3090: höchstens 7,7·10⁻¹⁵ mm (relativ ≤ 4,8·10⁻¹²). Die Textausgabe ist bei 6 Stellen deshalb praktisch immer identisch; sie unterscheidet sich nur in der Zeile `Backend`.
- `failed_rays` und `rays` sind identisch (gleiches Versagenskriterium im header-only Tracer).
- `piston_mm` ist `null`: Der Kernel `RESIDUALS` zieht den Kolben ab, gibt ihn aber nicht aus.
- `backend` ist `"gpu"`, `device` nennt Index und Namen des Geräts.

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
```

sowie in den Einträgen von `wavefront` die RMS-Werte in den letzten Stellen und der Kolben, z. B. für Feld 0 bei 486,13 nm `"rms_waves": 4.5020929504633`, `"rms_mm": 0.002645249733974216`, `"piston_mm": null` (CPU: `4.502092950462526`, `0.0026452497339737612`, `14.130347293916047`).

### Tests

`tests/test_cli_trace.cpp` ruft `mko::cli::run_trace_command` direkt auf und prüft die JSON-Ausgabe gegen `compute_paraxial` und `evaluate_wavefront` (bitgleich), das afokale System, die Punktquelle, Warnungen, Fehlerfälle und die Textausgabe. Die ctest-Kommandos `cli_trace_json`, `cli_trace_text` und `cli_trace_missing_file` starten das gebaute Programm.

`--gpu`: Im CPU-Build prüfen `CliTrace.GpuUnavailableWithoutCuda` und das ctest-Kommando `cli_trace_gpu_unavailable` Meldung und Exit-Code 3 (läuft in der CI). Im CUDA-Build vergleichen die Tests `CliTraceGpu.*` (übersprungen ohne Gerät) GPU- gegen CPU-Ausgabe für `einzellinse_bk7.json`, `testsystem_mehrlinser.json` (auch mit 16 × 12 Stützstellen), ein System mit versagten Strahlen, die Punktquelle und das afokale System; dazu die ctest-Kommandos `cli_trace_gpu_json` (Exit-Code 3 ohne Gerät = übersprungen, `SKIP_RETURN_CODE`) und `cli_trace_gpu_no_device` (`CUDA_VISIBLE_DEVICES=-1`).

Implementierung: `src/cli/trace_command.cpp` (Bibliothek `mikroptim_cli`, interner Header `src/cli/trace_command.hpp`; im CUDA-Build gelinkt gegen `mikroptim_cuda`, GPU-Pfad unter `#ifdef MIKROPTIM_HAS_CUDA`), `src/cli/main.cpp` verteilt nur die Unterbefehle.
