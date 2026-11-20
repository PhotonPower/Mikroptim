# Vorgeschlagene Erweiterungen des Optimierers (Entwurf, M11)

- **Status:** Entwurf. Jede Erweiterung ist eine eigene Aufgabe (Abschnitt 9). Für das glatte Maximum ist die Grundsatzentscheidung getroffen ([ADR-0013](adr/0013-glattes-maximum-und-mehrere-gpus.md)); weitere grundsätzliche Punkte (neue Merit-Terme) brauchen bei der Umsetzung eine eigene ADR. Augmented Lagrangian (M11-T2) und toleranzbewusste Merit (M11-T6) sind in [ADR-0018](adr/0018-optionale-merit-terme.md) festgehalten.
- **Anlass:** Befunde F3–F5, F9, F10 und F13 aus [m6-t6-zielspezifikation.md](ergebnisse/m6-t6-zielspezifikation.md)
- **Bezug:** [optimierung.md](optimierung.md) (vorhandener Stand); [entwurfssteuerung.md](entwurfssteuerung.md) (M10) nutzt mehrere der hier vorgeschlagenen Größen als Diagnose oder Aktion.

Dieses Dokument ändert nichts am vorhandenen Verhalten. Alle Erweiterungen sind abschaltbar und mit dem heutigen Verfahren als Standard einzuführen, bis eine Messung den Wechsel begründet.

## Übersicht

| Nr. | Erweiterung | Zielt auf | Aufwand | Abschnitt |
|---|---|---|---|---|
| M11-T1 | glattes Maximum für Strahlhöhen und Einfallswinkel | F10 (LM endet mit `max_iterations`) | mittel (Kernel-Reduktion) | 1 |
| M11-T2 | Augmented Lagrangian für einseitige Bedingungen | Penalty-Stufen, Kondition, Restverletzung | klein | 2 |
| M11-T3 | Tangentenprädiktor in der Homotopie (Sekante umgesetzt, ohne Zyklengewinn) | LM-Zyklen pro Schritt, Δ < Δ_min | klein | 3 |
| M11-T4 | Verletzung der Sinusbedingung (OSC) als Residuum | F4, F5 (Koma bei kleinem Feld unsichtbar) | klein | 4 |
| M11-T5 | algebraische Glasvorauswahl für Achromasie und Apochromasie | F9, F13 (Farbstufen scheitern) | mittel | 5 |
| M11-T6 | toleranzbewusste Optimierung (umgesetzt, `optimize --tolerance-aware`) | Baubarkeit, Vorgriff auf M8-T4 | mittel | 6 |
| M11-T7 | memetische globale Suche (CMA-ES mit LM) | Multistart in fester Topologie | mittel | 7.1 |
| M11-T8 | Netz der Minima (Sattelsuche in fester Topologie), Escape-Funktion | F3 (Stillstand ohne Bedarf) | groß, Forschung | 7.2, 7.3 |

Empfohlene Reihenfolge: T1, T2, T3 (Konvergenz und Laufzeit, am Abnahmesystem von M6-T3 messbar), dann T4 und T5 (die beiden Lücken aus M6-T6), dann T6, zuletzt T7 und T8.

## 1. Glattes Maximum über Strahlen (M11-T1)

**Problem.** Die trace-abhängigen Bedingungen (`max_semi_diameter`, `min_edge_thickness` mit `height = "rays"`, `max_incidence_angle`) verwenden Maxima über die Strahlen, bei der Randdicke zusätzlich über zwei Flächen. Sie sind nur stückweise glatt (optimierung.md §2.1, §2.5). Wechselt der Träger des Maximums, springt die Ableitung. M6-T3 und M6-T6 (F10) führen das Kriechen des LM auf diese Knicke zurück; fast alle Läufe enden mit `max_iterations`.

**Vorschlag.** Ein glattes Maximum ersetzt das harte:

$$\widetilde{\max}_\beta(h_1,\dots,h_N) = m + \frac{1}{\beta}\ln\sum_i e^{\beta (h_i - m)},\qquad m = \max_i h_i,$$

mit $\max_i h_i \le \widetilde{\max}_\beta \le \max_i h_i + \ln N/\beta$. Die Überschätzung ist **konservativ**: Die Bedingung wird eher zu früh aktiv. Für den kleinsten cos I gilt dasselbe mit −cos I. Bei der Randdicke werden beide Flächen in dieselbe Summe genommen.

**Umsetzung.**
- Zweite Reduktion im Beobachter von `trace_ray` (`surface_extrema.hpp`): pro Fläche das harte Maximum m und die Summe Σ e^{β(h−m)}. Mit verschobenem Exponenten ist das stabil. Die Rechnung läuft in zwei Durchgängen oder mit der üblichen Online-Fortschreibung (m und Summe gemeinsam aktualisieren).
- Deterministische Baumreduktion wie die vorhandenen Max-Reduktionen (ADR-0007), direkt im Kernel mit `exp` und `log` der jeweiligen Mathematikbibliothek (ADR-0013, Variante A). Diese Funktionen sind nicht korrekt gerundet: Wiederholbar ist das Ergebnis auf demselben Gerät, zwischen CPU und GPU sowie zwischen sm_60 und sm_86 nur innerhalb einer Toleranz (`kSmoothMaxCpuGpuRelative` in `tests/tolerances.hpp`, bei der Umsetzung zu messen).
- Im Zahlentyp df64 wird die glatte Reduktion in FP64 gerechnet (Eingaben in FP64 umgewandelt). Die Reduktion ist nicht OPD-kritisch, die Genauigkeit von df64 ist hier nicht nötig.
- β ist Homotopie-Parameter (Fortsetzung in β wie in ε, optimierung.md §7). Am Ende wird β so gewählt, dass ln N/β kleiner als der Vorhalt ist (Standard 5·10⁻⁵ mm). Der Vorhalt bleibt dann gültig.
- β in Einheiten der Größe: 1/mm für Höhen, dimensionslos für cos I.

**Stand M11-T1a (CPU):** umgesetzt als Option `constraints.extrema_smoothing` (Standard hart, [dateiformat.md](dateiformat.md#glättung-der-strahl-extrema-ab-m11-t1)). Gemeinsame Formeln `smooth_max_add`, `smooth_max_merge`, `smooth_max_value` (`surface_extrema.hpp`, `MKO_HD`, für den Kernel in M11-T1b), Reduktion in `surface_extrema_cpu` und `merit_cpu`, Glättung über `TraceExtraResiduals::smoothing` an LM, FD- und adjungierte Jacobi sowie Geisterdiagnose; die GPU-Pfade lehnen sie bis M11-T1b ab (`require_hard_extrema`). Die β-Fortsetzung im Homotopie-Controller folgt mit M11-T1b.

**Befund am Zweilinser mit mechanischen Bedingungen** (`zweilinser_mechanik.json`, CLI `optimize`, CPU): Hart konvergiert er (`converged_step`, 483 Zyklen, F = 18,53). Mit festem β ab dem stark verletzten Start (F₀ = 2,3·10⁶) landet der LM in anderen Minima (β = 10³: F = 673, β = 10⁴: F = 3,2·10⁴; nachbewertet mit harten Extrema gleich, also kein Bias der Glättung, sondern ein anderer Weg); β = 10⁵ ist bitgleich zum harten Lauf. Ab dem harten Optimum als Fortsetzung in β (3·10³ → 10⁶) bleibt F bei 18,53 und jede Stufe konvergiert in 7–21 Zyklen. Folgerung: Die Glättung ist kein Ersatz für einen guten Start; sinnvoll ist sie als Fortsetzung (großes β am Ende, kleines nur nahe der Zulässigkeit). Ob sie das Kriechen aus F10 behebt, ist am System von M6-T3 mit der β-Fortsetzung des Controllers zu messen (M11-T1b, auf herkules3).

**Stand M11-T1b (β-Fortsetzung, CPU) und Befund zur Ursache von F10.** Der Homotopie-Controller und die Strahlsuche setzen `HomotopyPoint::extrema_beta` (Start `--extrema-beta`, am Ende Stufen × 10, zuletzt hart, vor der ε-Fortsetzung). Gemessen am Abnahmesystem von M6-T3 (`einzellinse_homotopie_start.json`, NA 0,3 → 0,6, CPU):

| Lauf | Lösungen | davon `max_iterations` | LM-Zyklen | Zeit | Ziel |
|---|---|---|---|---|---|
| hart | 63 | 36 | 21.216 | 14 s | erreicht, 4 Elemente |
| β = 300 … 10⁴ | 66 | 39–40 | 22.716–22.959 | 17 s | erreicht, 4 Elemente |
| ohne mechanische Bedingungen | 39 | 2 | 4.513 | 1,7 s | erreicht, 3 Elemente |
| ohne Randdicken (`min_edge_thickness`) | 45 | 2 | 4.611 | 1,7 s | erreicht, 3 Elemente |
| Randdicken mit `height: semi_diameter` (ohne Strahl-Extrema) | 55 | 28 | 15.545 | 9 s | Start abgelehnt |

Weitere Eingrenzung (hart): nur der Glasrand 4 von 40 Läufen mit `max_iterations` (5.547 Zyklen), nur der Luftrand 7 von 63 (7.834), beide 36 von 63 (21.216); beide mit Gewicht 10³ statt 10⁴ 28 von 63 (18.484), mit 10² 21 von 63 (14.950). Das Kriechen entsteht vor allem, wenn Glas- und Luftrand **gleichzeitig** aktiv sind (sie hängen über dieselben Pfeilhöhen zusammen), und nimmt mit dem Gewicht ab.

**M11-T1c: Ursache gefunden, Stagnationskriterium.** Die Läufe mit `max_iterations` liegen bei sehr kleinen Merits (F ≈ 10⁻⁵ … 10⁻³ Wellen²) mit aktiven Randdicken: Die Modellkriterien des LM (`ftol = xtol = 10⁻¹⁰` auf das Gauß-Newton-Modell) greifen am Knick von max(0, g) nicht, F sinkt nur noch unmerklich. Mit `--max-iterations 50` statt 500 erreicht derselbe Lauf das Ziel mit derselben Topologie in 2,1 s statt 14,7 s (RMS 0,033 statt 0,015 Wellen). Das neue Stagnationskriterium (`LmOptions::stall_window`, `stall_tolerance`, Status `converged_stall`; CLI `--stall W[,T]` in `optimize` und `synth`) bricht ab, wenn F in W Zyklen um höchstens T·F gesunken ist:

| `--stall` | Ziel | RMS [Wellen] | Elemente | LM-Zyklen | Zeit | Status |
|---|---|---|---|---|---|---|
| aus | erreicht | 0,0154 | 4 | 21.216 | 14,6 s | 36 × `max_iterations` |
| 20, 10⁻³ | erreicht | 0,0166 | 4 | 6.436 | 4,8 s | 56 × `converged_stall`, 8 × `max_iterations` |
| 20, 10⁻² | erreicht | 0,0323 | 4 | 3.021 | 2,0 s | 66 × `converged_stall` |
| 10, 10⁻³ | erreicht | 0,0610 | 4 | 2.051 | 1,4 s | zu knapp am Ziel 0,0714 |

`--stall 20,1e-3` war kurz Standard von `mikroptim synth` (#65) und ist seit dem Lauf auf der Zielspezifikation (29.10.2026, herkules3) **wieder aus**:

| Lauf (Zielspezifikation, `tools/m6t6_zielspezifikation.py --gpu --farbregel`) | Aperturstufe 0,3 → 0,5 | Feldstufe 10 → 100 % | Aperturstufe → 0,95 |
|---|---|---|---|
| `--stall 0` | NA 0,5, 4 Elemente | Ziel, 5 Elemente | NA 0,605, 7 Elemente |
| `--stall 20,1e-3` | **bei NA 0,37 abgebrochen** (zweite Runde `start_rejected`) | – | – |
| `--stall 20,1e-5` | NA 0,5, 4 Elemente | **20 Einfügungen in einem Lauf** (24 Elemente, F bis 10¹¹), danach Absturz | – |

Der frühere Abbruch lässt die Zwischenstände weniger auskonvergiert; im Controller mit Annahmeschwelle und Einfügungen wirkt das nicht nur auf die Laufzeit, sondern auf die Entscheidungen. Am kleinen Abnahmesystem war davon nichts zu sehen. Der Abbruch bleibt eine Option für schnelle Erkundungen; ein brauchbarer Standard bräuchte mindestens einen strengen Abschluss jedes Zustands, an dem der Controller entscheidet. Am Zweilinser mit mechanischen Bedingungen (`optimize --max-iterations 1000`) endet der Lauf ohne Fenster nach 6.053 Zyklen bei F = 31.914, mit `--stall 20,1e-3` nach 159 Zyklen bei F = 18,53 (wie mit dem Standardlimit 100).

Die Glättung ändert am Kriechen nichts. Die Ursache von F10 sind die **Randdicken-Bedingungen selbst**, auch ganz ohne Maximum über Strahlen (Variante mit Halbdurchmesser). Naheliegend sind das Ein- und Ausschalten der einseitigen Strafe max(0, g) (wechselnde aktive Menge, Knick der Jacobi-Zeile bei g = 0) und die steife Strafe bei großen Gewichten. Deshalb: **Kernel für das glatte Maximum zurückgestellt**; als Nächstes M11-T2 (Augmented Lagrangian, kleine feste Gewichte) und eine Glättung des Knicks max(0, g) selbst (z. B. Softplus mit Fortsetzung) am selben System messen.

**Abnahme.**
- Grenzwert: Für β → ∞ stimmt der Wert mit dem harten Maximum überein (Toleranz ln N/β).
- Glattheit: zentrale Differenzen über einen Trägerwechsel hinweg stimmen mit dem analytischen Gradienten der Glättung überein.
- Nutzen: Das Abnahmesystem von M6-T3 und der Zweilinser von M4-T3 konvergieren mit `converged_*` statt `max_iterations`. Messgröße sind Zyklen bis zur Konvergenz gegen den heutigen Stand.

## 2. Augmented Lagrangian für einseitige Bedingungen (M11-T2)

**Problem.** Eine Strafe w·max(0, g)² erreicht die Grenze nur asymptotisch: g* ≈ λ/(2w) (optimierung.md §2.3–2.5). Deshalb braucht es Gewichtsstufen bis zu großen w. Große w verschlechtern die Kondition von JᵀJ und machen die Knicke der Extrema schärfer.

**Vorschlag (Multiplikatormethode, Form mit Verschiebung).** Jede einseitige Bedingung bekommt eine Verschiebung θ ≥ 0 in der Einheit von g:

$$r = \sqrt{w}\,\max(0,\ g + \theta),\qquad \theta \leftarrow \max(0,\ \theta + g^*)$$

nach jedem LM-Lauf (g* am Endpunkt). Am Fixpunkt gilt g* = 0 bei aktiver Bedingung, also exakte Einhaltung bei **festem, moderatem** w. Der Multiplikator ist λ = 2wθ. Die heute gemessene Größe w·g* ≈ λ/2 liefert damit einen guten Startwert θ₀ = g* der vorigen Stufe.

**Umsetzung.**
- θ pro Residuum der einseitigen Bedingungen im Host-Code: `ExtraResiduals`, `TraceExtraResiduals`, Petzval, Glas-Hülle.
- Die äußere Schleife sitzt im Homotopie-Controller anstelle der Penalty-Fortsetzung (Option, Standard aus).
- Abbruch, wenn |Δθ| unter einer Toleranz bleibt.
- θ gehört zum Zustand eines Kandidaten (Protokoll, Strahlsuche: gleiche Gruppe nur bei gleichem θ, optimierung.md §8).

**Abnahme.**
- Petzval-Abnahme (M4-T4) und Mechanik-Abnahme (M4-T3) mit festem w (z. B. das kleinste Gewicht der heutigen Fortsetzung) erreichen Restverletzungen unter den heutigen Toleranzen.
- Die Zahl der LM-Zyklen insgesamt ist nicht größer als mit der Fortsetzung.

**Stand M11-T2:** umgesetzt ([optimierung.md, Abschnitt 2.6](optimierung.md#26-augmented-lagrangian-für-einseitige-bedingungen-m11-t2), [ADR-0018](adr/0018-optionale-merit-terme.md)): θ in `ConstraintSpec::shift` (Zustand, nicht im Dateiformat) für paraxiale Bedingungen mit Modus `min`/`max`, Petzval und alle mechanischen (Glas-Hülle offen), Fortschreibung θ ← min(max(0, θ + g*), θ_max/√w), Abbruch bei max √w|Δθ| ≤ T; `optimize --augmented-lagrangian K[,T]`, im Controller `synth --augmented-lagrangian K[,T[,S]]` statt der Penalty-Fortsetzung mit Fortschreibung nach jeder Optimierung (Standard aus, nicht mit der Strahlsuche). Abnahme erreicht, aber nicht mit dem kleinsten Gewicht der Fortsetzung (dort konvergieren die Multiplikatoren zu langsam), sondern mit Datei · 10⁻⁴ (Petzval) bzw. · 10⁻¹ (Mechanik): Petzval 391 statt 546 LM-Zyklen, Restverletzung 1,5·10⁻⁹ relativ; Mechanik 56 statt 104, alle g ≤ 2,1·10⁻⁷ mm. Am System von M6-T3 kein Gewinn allein (28.637 statt 21.038 Zyklen), mit `--stall 20,1e-3` 1.785 statt 4.931 Zyklen bei besserem RMS.

## 3. Tangentenprädiktor in der Homotopie (M11-T3)

**Problem.** Jeder stetige Homotopie-Schritt startet beim Ergebnis des vorigen Punkts, also mit einem Prädiktor nullter Ordnung. Bei großer NA verlängert das die LM-Läufe. Es führt auch früher zu abgelehnten Schritten und Halbierungen bis Δ_min.

**Vorschlag.** Am Optimum x*(p) gilt die Stationarität Jᵀr = 0. Ableiten nach dem Homotopie-Parameter p (Apertur, Feldanteil) gibt in Gauß-Newton-Näherung

$$\frac{dx}{dp} \approx -\,(J^\top J + \mu D)^{-1} J^\top \frac{\partial r}{\partial p}.$$

**Umsetzung.**
- ∂r/∂p als zusätzliche zentrale FD-Spalte über p. Das sind zwei weitere Varianten pro Lauf, realisiert mit `realize` bei p ± h_p. Die Residuenzeilen bleiben einander zugeordnet: Quadratur und Felder sind dieselben, nur Pupillenradius bzw. Feldwinkel ändern sich.
- A = JᵀJ liegt nach dem letzten LM-Zyklus ohnehin vor. Zerlegung mit aktiver Menge wie im LM-Schritt, kleine Dämpfung μ.
- Startpunkt des Versuchs: x₀ = clamp(x* + Δp · dx/dp).
- Sicherung: F an x₀ und an x* gemeinsam im Modus MERIT auswerten. Der kleinere Wert gewinnt; bei Gleichstand x*. Damit kann der Prädiktor nie schaden. Die Entscheidung ist deterministisch.
- Spektrumstufen sind diskret und bekommen keinen Prädiktor.

**Abnahme.**
- Abnahme von M6-T3 (Einzellinse NA 0,3 → 0,6): gleiches Ergebnis (`goal_reached`) mit weniger LM-Zyklen insgesamt, Ziel −30 %.
- Determinismus wie `HomotopySmoke.DeterministicCpu`.

**Stand M11-T3 (erste Stufe: Sekante) und Befund.** Umgesetzt ist der günstigere Sekantenprädiktor statt der Tangente: x₀ = x_k + (Δp_neu/Δp_alt)·(x_k − x_{k−1}) aus den beiden letzten angenommenen Zuständen derselben Topologie (`HomotopyModel::predict`, im optischen Modell über die `VariableSet` am Punkt, auf die Schranken projiziert), Option `HomotopyOptions::predictor`, CLI `synth --predictor` (nicht mit `--beam`). Nach einem Einfügen wird die Sekante neu aufgebaut. Gemessen am Abnahmesystem (CPU):

| Lauf | Ziel | RMS [Wellen] | Lösungen | LM-Zyklen |
|---|---|---|---|---|
| Standard (`--stall 20,1e-3`) | erreicht, 4 Elemente | 0,0166 | 67 | 6.436 |
| `--predictor` | erreicht, 4 Elemente | 0,0166 | 64 | 6.919 |
| `--stall 0` | erreicht, 4 Elemente | 0,0154 | 63 | 21.216 |
| `--stall 0 --predictor` | erreicht, 4 Elemente | **0,0047** | 65 | 22.216 |

Der Prädiktor senkt die Merit am Start eines Schritts um zwei bis drei Größenordnungen (z. B. NA 0,36: 0,38 statt ≈ 500), spart aber **keine LM-Zyklen**: Die Läufe enden durch das Stagnationsfenster (mindestens etwa 20 Zyklen) oder am Iterationslimit, nicht weil der Start schlecht war. Abgelehnte Schritte bei NA ≈ 0,44 liegen am fehlenden Element (F_B über der Schwelle), nicht am Prädiktor; eine Wiederholung ohne Prädiktor brachte dort nichts und wurde wieder entfernt. Ohne Stagnationsabbruch führt der Prädiktor zu einem besseren Endsystem (RMS 0,0047 statt 0,0154 Wellen). Das Abnahmeziel −30 % Zyklen ist damit nicht erreicht; der Prädiktor bleibt eine Option (Standard aus). Die Tangente lohnt sich erst, wenn die Zahl der Zyklen pro Schritt von der Startgüte abhängt (z. B. mit einem kürzeren Stagnationsfenster nach gutem Start).

**Nachtrag M6-T8:** Der Prädiktor arbeitet auch in der gemeinsamen Apertur-/Feld-Phase (`--joint-aperture-field`, Sekante in t). Dazu kamen die Feldregel (Bedarf aus Feldfehlern statt aus den Geisterplatten) und „Einfügen nur mit Verbesserung“; Festlegungen in [optimierung.md](optimierung.md#gemeinsame-apertur-feld-homotopie-und-feldregel-m6-t8), Messung im [Bericht M6-T8](ergebnisse/m6-t8-apertur-feld.md).

## 4. Sinusbedingung als Residuum (M11-T4)

**Problem.** Lineare Koma wächst mit dem Feld. Bei reiner Achsoptimierung ist sie in der Merit unsichtbar, und die rotationssymmetrischen Geisterplatten zeigen keinen Bedarf (F4). Ein kleines Feld von Anfang an hilft, kostet aber Feldpunkte und verschiebt das Problem in die Kopplung von Apertur und Feld (F5).

**Vorschlag.** Die Verletzung der Sinusbedingung (Offence against Sine Condition, Conrady) aus dem Achsbündel allein. Für das kollimierte Bündel im Unendlichraum, also Mikroptims Tracerichtung, ist sie bei der Pupillenzone ρ_k

$$\mathrm{OSC}(\rho_k) = \frac{H_k}{f\,\sin U'_k} - 1 - \frac{L'_k - l'}{L'_k - l'_p},$$

mit Einfallshöhe H_k, realem Winkel U′_k und Schnittweite L′_k des Zonenstrahls, paraxialer Schnittweite l′ und Lage l′_p der Austrittspupille (in Tracerichtung). Formel und Vorzeichen sind vor der Umsetzung gegen Welford, *Aberrations of Optical Systems*, Kap. 9, und gegen numerik.md zu prüfen.

Das Residuum ist √w · OSC(ρ_k) für zwei bis drei Zonen (etwa die äußeren Gauß-Ringe). Der zweite Term macht die Größe unabhängig von der Restsphärik; aplanatisch ist das System, wenn zusätzlich die sphärische Aberration klein ist, und dafür sorgt die Merit ohnehin.

**Umsetzung.**
- Die Zonenstrahlen liegen im Trace der Achse schon vor (Speichen nahe dem Meridian). Gebraucht wird nur der Austrittswinkel und die Schnittweite des meridionalen Strahls jedes Rings.
- Als trace-abhängiges Zusatzresiduum (optimierung.md §2.1) oder als zusätzliche Ausgabe des Kernels.
- Gleichzeitig Diagnosegröße für M10 (entwurfssteuerung.md, Abschnitt 3.2).

**Abnahme.**
- Orakel: aplanatische Fläche bzw. aplanatisches Linsenpaar (Zhang §3.2) ergibt OSC = 0 bis auf Rundung; eine Einzellinse ergibt das Vorzeichen und die Größenordnung aus der Seidel-Koma (S_II ∝ OSC bei kleiner NA).
- Nutzen: Wiederholung des M6-T6-Versuchs „Apertur zuerst auf der Achse“ (F4) mit OSC-Term. RMS bei 7 % und 10 % Feld deutlich unter dem damaligen Wert von 0,28 Wellen.

**Stand M11-T4 (umgesetzt).** Formel nach Prüfung gegen numerik.md §14: die Conrady-Form des Aberrationsberichts mit Pupillenfaktor, OSC = (f_real/f′)·(l′ − l′_P)/(L′ − l′_P) − 1, und zwar mit der bildseitigen Brennweite f′ = n′_L/φ (statt der oben skizzierten additiven Form, damit Bericht und Residuum dieselbe Größe sind). Umsetzung als Nebenbedingung `sine_condition` im Block der paraxialen Residuen mit einem eigenen meridionalen Zonenstrahl pro Zone auf dem Host statt über den Kernel (die Gauß-Ringe liegen nicht bei ρ_k, die Speichen neben dem Meridian); Definition, Gewichtswahl und Prüfungen: [optimierung.md, Abschnitt 2.7](optimierung.md#27-sinusbedingung-ab-m11-t4). Orakel: zur Blende konzentrische Systeme ergeben |OSC| ≤ 4,4·10⁻¹⁶ (die aplanatischen Punkte einer Kugelfläche gehören zu endlichen Konjugierten und passen nicht zur kollimierten Quelle). Versuch ([Bericht](ergebnisse/m11-t4-sinusbedingung.md)): Am Endpunkt der Synthese sinkt der RMS bei 7/10 % Feld von 14,6/20,9 auf 0,80/0,80 Wellen; die hohe NA erreicht der Versuch aber nur über die Falle F12, mit `paraxial_focus` dagegen bleibt er mit und ohne Residuum bei NA 0,37 stehen, und nachoptimiert ohne Geisterplatten ist der Unterschied dort klein. Der Vergleich mit 0,28 Wellen bei NA 0,95 ist damit offen.

## 5. Algebraische Glasvorauswahl (M11-T5)

**Problem.** Die Farbstufen scheitern (F9, F13). Die Diagnose empfiehlt keinen Operator B, und die freien Glasvariablen finden aus einem monochromatisch aufgebauten System mit Gläsern ähnlicher Dispersion nicht heraus. Was fehlt, ist eine gezielte Glaswahl.

**Vorschlag.** Vor dem Trace werden Gläser auf dünnen Elementen bewertet. Für m Elemente mit Brechkräften φ_i, paraxialen Randstrahlhöhen h_i (aus dem aktuellen System, normiert auf die erste) und den Dispersionskoeffizienten η_ij = ν_ij/(n₀,i − 1) des Materials i (aus den Buchdahl-Koeffizienten ν_ij, numerik.md §10; bei der Umsetzung korrigiert: für ein dünnes Element gilt φ ∝ n − 1, deshalb η statt ν) lautet die Bedingung für Farbkorrektur in den Buchdahl-Termen j = 1 … k:

$$\sum_i h_i^2\,\varphi_i = \varphi,\qquad \sum_i h_i^2\,\varphi_i\,\eta_{ij} = 0\quad (j = 1,\dots,k).$$

- k = 1 entspricht Achromasie (zwei Gläser), k = 2 Apochromasie (drei Gläser bzw. zwei mit gleicher Teildispersion).
- Für m = k + 1 ist das ein lineares Gleichungssystem in φ_i.
- Bewertet wird nach Σ|φ_i|/|φ| (kleine Einzelbrechkräfte, Zhang §3.2) und nach dem Rest in den nicht korrigierten Termen (sekundäres bzw. tertiäres Spektrum).
- Fast singuläre Systeme werden verworfen.

Grundlage sind die Arbeiten zur Glaswahl mit Buchdahl-Koeffizienten (Robb, Mercado; Rayces und Rosete-Aguilar). Vor der Umsetzung sind die genauen Formulierungen dort nachzulesen.

**Umsetzung.**
- Alle Paare (N²/2) und Tripel (N³/6) der Gläser der Glas-Hülle, vektorisiert auf dem Host. Bei einigen hundert Gläsern sind das bis 10⁷ kleine Systeme, also Sekunden.
- Filter: Regeln aus entwurfssteuerung.md, Abschnitt 5.3 (Autofluoreszenz, Transmission, Merte-Paar).
- Ergebnis: Rangliste mit Startbrechkräften. Sie fließt in `split_from_diagnosis` (Zielglas und c_s statt nur nach Abbezahl), in Operator B und in `glass_query` (M10).

**Stand M11-T5:** Kern, Kommandozeile `mikroptim glasses` und Aktion `glass_preselect` in `mikroptim.design` umgesetzt ([numerik.md, Abschnitt 15](numerik.md#15-algebraische-glasvorauswahl-m11-t5)). Paraxiale Probe mit Katalog-Gläsern: Spanne der Schnittweiten über g … C bei f = 100 mm für die Einzellinse 2,36 mm, für das beste Paar 0,016 mm, für das beste Tripel 0,004 mm. Anbindung an Operator B (M11-T5b): `SplitInitOptions::preselect` (Zielglas = bester Partner des Linsenglases, c_s aus den Anteilen in η) und `start_at_target` (freigegebenes Glas startet beim Zielglas), CLI `synth --split-preselect` bzw. `--split-start-target`, `insert --start-glass GLAS`, in `mikroptim.design` `split_lens` mit `cement_curvature: "achromat"` und `target_glass`.

**Befund M11-T5b.** (1) In der Synthese greift die Vorauswahl bisher nicht: An der polychromatischen Einzellinse mit Geisterplatten (NA 0,5, 486/588/656 nm) empfiehlt die Diagnose nur Operator A, Operator B kommt nicht vor (das ist F9). Ohne eine Entscheidung für B, durch Mensch oder Sprachmodell in `mikroptim.design` oder eine künftige Regel im Controller, bleibt `--split-preselect` wirkungslos. (2) Direkt angewandt am Achromat aus `einzellinse_achromat_start.json` (N-BK7, 480/644 nm, Katalog Basis + synthetisch, danach 300 LM-Zyklen):

| Operator B | Partner | RMS max [Wellen] | Farbklasse |
|---|---|---|---|
| c_s = Mittel, Start beim Linsenglas | – | 0,0771 | b |
| c_s = dünnes Achromat aus der Vorauswahl, Start beim Linsenglas | SYN-04 | 0,0753 | b |
| ebenso, Start beim Zielglas | SYN-04 | 0,1021 | b |

Die Kittkrümmung aus der Vorauswahl hilft leicht; der Start beim Zielglas schadet, weil die Außenradien noch zur Einzellinse gehören (Startmerit 2,5·10⁶ statt 3,6·10³). Sinnvoll wäre er erst zusammen mit Startbrechkräften für alle Flächen des Glieds; deshalb Standard aus.

**M11-T5c: Farbregel.** Weil die Geisterdiagnose B nicht empfiehlt, entscheidet eine Regel im optischen Modell vor der Diagnose (`OpticalHomotopyOptions::color_rule`, CLI `synth --color-rule [--color-ratio F]`): Ist der größte RMS (getracte Wellenfront am Punkt) mindestens 0,05 Wellen und dominiert die Farbe – (a) der RMS einer Wellenlänge ist mindestens F-mal (Standard 2) der RMS bei λ_ref desselben Feldes, oder (b) Seidel |ΔW020| ≥ 0,25 Wellen und ≥ |W040| –, dann Operator B an der Linse mit dem größten Beitrag zu ΔW020 (Katalogglas), Partner und Kittkrümmung aus der Vorauswahl (`split_for_color`), ohne Platte. Kriterium (b) allein versagt bei hoher NA (Seidel-Sphärik groß, im System durch höhere Ordnungen ausgeglichen), (a) allein bei kleiner NA (Farbe und Sphärik gleich groß); deshalb beide.

| System (CLI `synth`, CPU) | ohne Regel | mit `--color-rule` |
|---|---|---|
| `einzellinse_achromat_start.json` (N-BK7, 480/644 nm) | `start_rejected` (keine Platte, kein Bedarf), 1 Element | **`goal_reached`**: B mit CaF₂, RMS 0,0155 Wellen, 2 Elemente, 8 Optimierungen |
| Einzellinse NA 0,5 mit Geisterplatten, 486/588/656 nm, höchstens 5 Elemente | `start_rejected` nach drei Menisken (A, A, A), 4 Elemente | **`goal_reached`**: B, B (CaF₂), A, RMS 0,025 Wellen, 4 Elemente |
| `einzellinse_homotopie_start.json` (monochromatisch) | Ziel, 4 Elemente | unverändert (Regel braucht ≥ 2 Wellenlängen) |

**Lauf auf der Zielspezifikation (29.10.2026, herkules3):** Die Farbstufen liefen mit #74 ohne Absturz, die Farbregel griff aber nicht: Alle Linsen der Zielspezifikation haben Glasvariablen (Buchdahl), die Regel berücksichtigte nur Katalogglas. Seitdem nimmt die Vorauswahl die Dispersion einer Glasvariablen direkt (gleiche Buchdahl-Koordinate wie der Katalog-Fit). Erneuter Lauf steht aus.

**M11-T5g: Filter, Startradien, Bruteforce-Abnahme** ([numerik.md, Abschnitt 15](numerik.md#15-algebraische-glasvorauswahl-m11-t5)).
- *Filter:* dieselbe Filterbeschreibung wie die Glasabfrage (M10-T5): Bereiche n_d, ν_d, P_g,F, ΔP_g,F im Kern (`GlassCandidateFilter`, `filter_dispersions`, CLI `--nd/--vd/--pgf/--dpgf`, `--only`, `--exclude`). Glasregeln (u. a. `af_forbidden` gegen Autofluoreszenz) und Transmission wertet `glass_preselect` in `mikroptim.design` über `glass_query` aus. Vorauswahl mit Filter = Vorauswahl ohne Filter, nachgefiltert.
- *Startradien des ganzen Glieds:* `thin_member_shape`. Die Krümmungen des dünnen Kittglieds bzw. -tripels folgen aus den Brechkräften der Vorauswahl. Die Biegung macht die Seidel-Sphärik S_I = 0 (flachere der beiden Lösungen) oder die Koma S_II = 0. Geprüft gegen die klassische Biegung der Einzellinse und gegen den Aberrationsbericht.
- *Spektraler Rest:* Rang nach der Streuung des Brechkraftrests über das Spektrum, mit allen nicht korrigierten Buchdahl-Termen (Option, `--wavelengths`, `--spectrum`). Der Koeffizient des sekundären Spektrums allein übersieht den ω³-Anteil anomaler Gläser (CaF₂).
- *Bruteforce-Abnahme:* 563 lösbare Paare der Testkataloge als Kittglied (f = 100 mm, NA 0,05, g … C) mit beiden Startformen optimiert. Der beste Vorschlag nach spektralem Rest (SYN-11 + CaF₂) ist **Rang 1** nach Merit, bei NA 0,025 Rang 3, bei NA 0,1 Rang 6. Nach dem Koeffizienten allein wäre es SYN-04 + SYN-07 mit den Rängen 6, 2 und 54. Bei größerer NA überwiegen Monochromatfehler stark gekrümmter Glieder, und die Vorauswahl ist dort nur ein grober Filter.

Noch offen: Elementbudget in den Farbstufen der Zielspezifikation. Startradien und spektraler Rest werden in Operator B und in der Farbregel noch nicht genutzt (dort Rang nach dem Koeffizienten, Achromat-Start M11-T5e mit fester vorderer Außenkrümmung).

**Abnahme.**
- Orakel: Für zwei Gläser entspricht das Ergebnis dem dünnen Achromat (`achromat_split_curvature`).
- Ein konstruiertes Glastripel mit bekannter Lösung wird gefunden.
- Rangliste gegen Bruteforce-Raytrace an einem Kittglied (frei erfunden, NA 0,05): Der beste Kandidat der Vorauswahl liegt unter den besten fünf nach Merit. Erfüllt (M11-T5g): Rang 1 von 563.

## 6. Toleranzbewusste Optimierung (M11-T6)

**Problem.** Die Merit bewertet das nominale System. Systeme mit stark gekrümmten Kittflächen, Gauss-Hintergruppen und dünnen Luftspalten können beugungsbegrenzt und trotzdem nicht baubar sein. M8-T4 plant Monte-Carlo-Tolerierung nur als Auswertung.

**Vorschlag.** Erwartete Merit über S feste Störungen (gemeinsame Zufallszahlen, fester Startwert):

$$F_{\text{tol}}(x) = (1 - \alpha)\,F(x) + \frac{\alpha}{S}\sum_{s=1}^{S} F(x \oplus \delta_s).$$

x ⊕ δ_s ist das System mit der Störung s: Dezentrierung und Verkippung nach ADR-0011, Radien, Dicken und Brechzahlen.

**Umsetzung.**
- Die gestörten Systeme sind gewöhnliche Varianten im Batch (ADR-0004). Ihre Residuen werden mit √(α/S) gewichtet an den Residuenvektor gehängt.
- Die Jacobi-Matrix entsteht damit ohne Sonderfall über FD bzw. adjungiert, mit Kosten etwa (1 + S)-fach.
- Kompensatoren (Arbeitsabstand) pro Störung wären eigene Variablen. Sie sind für eine erste Version nicht nötig.
- Einsatz nur in späten Stufen (nach dem Spektrum) und für die Rangfolge der Strahlsuche (optimierung.md §8: „ab M8 Empfindlichkeit“).

**Abnahme.**
- Ein frei erfundenes System mit zwei bekannten Lösungen ähnlicher nominaler Merit und unterschiedlicher Dezentrierempfindlichkeit: Die toleranzbewusste Optimierung wählt die unempfindlichere.
- Die Monte-Carlo-Auswertung (M8-T4, sobald vorhanden) bestätigt die bessere Verteilung.

**Stand M11-T6:** umgesetzt ([optimierung.md, Abschnitt 14](optimierung.md#14-toleranzbewusste-optimierung-m11-t6), [ADR-0018](adr/0018-optionale-merit-terme.md)): `ToleranceAwareBackend` als LM-Backend über dem CPU- bzw. GPU-Backend, pro Lauf 1 + S innere Läufe in einem Batch, Zusammensetzung A = Σ w_s D_s A_s D_s, g = Σ w_s D_s g_s, F = Σ w_s F_s in fester Reihenfolge; Störungen ±δ je Zufallsgröße (S = 2D, Standard) oder S feste Stichproben mit Startwert; Störungen von Variablen über x′_s = extract(apply_s(x)) und D_s (relative Radien). `optimize --tolerance-aware α [--tolerance-samples S] [--tolerance-seed n] [--tolerance-set file|default]`, Standard aus. Abnahme am Kittblock (Kittfläche zwischen gleichen Gläsern, Brechzahltoleranz): nominal gleichwertige Lösungen A (c = 0,05/mm) und B (c = 0), die toleranzbewusste Optimierung läuft von A nach B, die gewöhnliche bleibt; die Monte-Carlo-Tolerierung bestätigt (RMS Mittel 0,0958 → 0,0728 Wellen). Abweichend vom Vorschlag oben ist das Orakel eine Brechzahl- statt einer Dezentrierempfindlichkeit: Nur so sind beide Lösungen analytisch exakt gleichwertig (eine dezentrierte Fläche zwischen gleichen Gläsern wirkt nicht), und die Bewertung der Tolerierung nach Abzug der Kippung stimmt mit der Merit überein. Offen: Kompensatoren pro Störung, Bewertung ohne Bildverschiebung, `synth` und Strahlsuche (Folgeaufgabe M11-T6b).

**Stand M11-T6b:** umgesetzt ([optimierung.md, „Kippungsabzug“](optimierung.md#kippungsabzug-m11-t6b) und „Synthese und Strahlsuche“). Teiloption `--tolerance-tilt remove` (zunächst Standard `keep`, bitgleich zu M11-T6; seit der Entscheidung des Maintainers vom 2026-11-19 Standard `remove`, ADR-0018, Nachtrag): Die gestörten Teilsysteme werden nach Abzug von Kolben und Kippung pro Tupel bewertet, mit derselben Anpassung wie Tolerierung und Empfindlichkeiten (`PistonTiltFit`, `TiltProjection`); die Projektion wirkt auf r_s und auf jede Spalte von J_s (Normalgleichungen auf dem Host aus J, das die inneren Backends über `LmBackend::host_jacobians` liefern). Gewählt wurde der Kippungsabzug statt eines Bildlage-Kompensators, weil die Abnahme die Übereinstimmung mit der Bewertung der Monte-Carlo-Tolerierung verlangt (die die Kippung abzieht) und ein Kompensator in linearer Näherung dieselbe Projektion ist, aber eigene Variablen pro Störung braucht. Abnahme: Die Dezentrierung (10 µm) der Sammellinse des Kittblocks ändert F_tol um 6·10⁻⁴ relativ statt um den Faktor 84; jedes gestörte Teilsystem gleich Q der Tolerierung (ohne Kompensator) bzw. Q± der Empfindlichkeiten bis 1,5·10⁻¹⁵ relativ. `synth --tolerance-aware α [--tolerance-samples S] [--tolerance-seed n] [--tolerance-tilt …]` minimiert F_tol in jedem Schritt (Standardsatz der realisierten Beschreibung), die Strahlsuche ordnet danach. Weiter offen: Arbeitsabstand-Kompensator pro Störung, `--global` mit `--tolerance-aware`.

## 7. Globale Suche in fester Topologie (M11-T7, M11-T8)

### 7.1 Memetische Suche: CMA-ES mit LM

Eine CMA-ES-Population über die Variablen einer festen Topologie, in normierten Koordinaten x/x_typ. Jedes Individuum wird kurz mit dem gebündelten LM verfeinert, etwa 20–50 Zyklen, alle Individuen in einem Aufruf.

Die Verteilung wird aus den verfeinerten Punkten aktualisiert (Lamarck'sche Variante). Gegenüber zufälligen Multistarts lernt die Startverteilung die Lage der guten Becken. Der FP32-Vorfilter (M7-T2) sortiert hoffnungslose Starts vorab aus. Zufallszahlen kommen aus dem vorhandenen, plattformunabhängigen Generator (optimierung.md §4); der Lauf ist mit Startwert reproduzierbar.

**Einsatz:** Feinoptimierung großer Systeme, nach dem Glas-Einrasten, auf den Zwischenständen der Strahlsuche.

**Abnahme:** Auf dem Mehrlinser-Testsystem und einem M6-T6-Zwischenstand wird bei gleicher Zahl von LM-Zyklen eine kleinere beste Merit gefunden als mit `perturbed_starts`.

**Stand M11-T7:** umgesetzt ([optimierung.md, Abschnitt 15](optimierung.md#15-memetische-globale-suche-m11-t7), `optimize --global`). Abweichungen vom Vorschlag: Normierung mit max(|x₀|, x_typ) statt x_typ (σ₀ relativ wie `--perturbation`), Elitismus (das beste Individuum läuft in jeder Generation weiter), gekürzte eingespeiste Punkte nach Hansen. Mehrlinser: beste Merit zwei Größenordnungen kleiner als `perturbed_starts` bei gleichen LM-Zyklen (CPU und GPU); M6-T6-Zwischenstand a2: im Mittel über vier Startwerte kleiner, aber nicht bei jedem (Kriechen des LM, F10).

### 7.2 Netz der Minima

Die Minima einer festen Topologie sind über Sattelpunkte vom Morse-Index 1 verbunden (Bociort und Mitarbeiter). Mikroptim nutzt Sattelpunkte bisher nur beim Einfügen (Operator A).

Vorschlag: Sattelsuche von einem Minimum aus, zunächst mit der Dimer-Methode. Die Hesse-Vektor-Produkte kommen aus Gradientendifferenzen (Jᵀr an zwei nahen Punkten); Startrichtungen sind die Eigenvektoren von JᵀJ mit kleinsten Eigenwerten. Vom gefundenen Sattel führt der Abstieg in beide Richtungen zu einem neuen Minimum. Die Richtungen kommen als neue Kandidaten derselben Elementzahl in die Strahlsuche.

Damit gibt es einen Ausweg bei Stillstand ohne Geisterbedarf (F3), der kein Element kostet.

**Abnahme (Forschung):** An einem Zweilinser mit bekannten zwei Minima wird das jeweils andere über einen Sattel gefunden. Ein Bericht klärt, wie oft das an M6-T6-Zwischenständen ein besseres Minimum liefert.

### 7.3 Escape-Funktion (Isshiki)

Gefundene Minima x_k erhalten ein zusätzliches Residuum √H · exp(−‖x − x_k‖²/(2W²)) in normierten Koordinaten (Schnittstelle `ExtraResiduals`). Der LM wird damit aus dem bekannten Becken gedrängt. Das ist einfach umzusetzen, überlappt aber mit 7.1 und 7.2. Es lohnt sich nur, falls diese zu teuer sind.

## 8. Bewusst nicht vorgeschlagen

- **Genetische Algorithmen, Simulated Annealing direkt auf Radien und Dicken:** Gegen gradientenbasierten Multistart auf der GPU fast immer unterlegen. Der memetische Ansatz (7.1) nimmt das Nützliche davon auf.
- **Adam und ähnliche Verfahren erster Ordnung:** ohne Gauß-Newton-Struktur bei diesen schlecht skalierten Problemen deutlich langsamer als LM.
- **Bayes'sche Optimierung:** nur auf einer sehr kleinen, teuren Entscheidungsebene sinnvoll (Wahl von Gerüsten). Diese Rolle ist in M10 dem Sprachmodell zugedacht.
- **Quasi-Newton-Korrektur für große Residuen** (strukturierte Sekantenverfahren, etwa NL2SOL-artig): kann die Endphase bei großem Restfehler beschleunigen (optimierung.md §4, Einzellinse). Zurückgestellt, bis T1–T3 gemessen sind.

## 9. Aufgaben

Die Aufgaben stehen auch im [Meilensteinplan](plan/meilensteine.md#m11--erweiterungen-des-optimierers-vorschlag).

| Aufgabe | Abhängig von | Inhalt | Abnahme |
|---|---|---|---|
| M11-T1 | M4-T3 | Abschnitt 1, mit ADR zur glatten Reduktion | Abschnitt 1 |
| M11-T2 | M6-T3 | Abschnitt 2 | Abschnitt 2 |
| M11-T3 | M6-T3 | Abschnitt 3 | Abschnitt 3 |
| M11-T4 | M4-T3 | Abschnitt 4 | Abschnitt 4 |
| M11-T5 | M6-T2 | Abschnitt 5 | Abschnitt 5 |
| M11-T6 | M8-T3 | Abschnitt 6 | Abschnitt 6 |
| M11-T7 | M7-T2 | Abschnitt 7.1 | Abschnitt 7.1 |
| M11-T8 | M5-T3 | Abschnitte 7.2, 7.3 (Forschung) | Abschnitt 7.2 |

## 10. Literatur

- J. Nocedal, S. J. Wright: *Numerical Optimization*, Kap. 17 (Augmented Lagrangian), Kap. 11 (Homotopie, Prädiktor-Korrektor).
- W. T. Welford: *Aberrations of Optical Systems* (Sinusbedingung, OSC).
- P. N. Robb, R. I. Mercado; M. Rosete-Aguilar, J. L. Rayces: Arbeiten zur Glaswahl für Achromate und Apochromate mit Buchdahl-Dispersionskoeffizienten (Applied Optics, ab 1980er-Jahren).
- N. Hansen: The CMA Evolution Strategy: A Tutorial.
- G. Henkelman, H. Jónsson: A dimer method for finding saddle points on high dimensional potential surfaces, J. Chem. Phys. 111 (1999).
- F. Bociort et al.: Netze lokaler Minima und Sattelpunktkonstruktion (TU Delft, ab 2007).
- M. Isshiki et al.: Lens design with escape function.
- Y. Zhang: *Systematic Design of Microscope Objectives*, Dissertation FSU Jena 2019.
