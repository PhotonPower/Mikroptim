# Vorgeschlagene Erweiterungen des Optimierers (Entwurf, M11)

- **Status:** Entwurf. Jede Erweiterung ist eine eigene Aufgabe (Abschnitt 9). Grundsätzliche Punkte (neue Reduktion im Kernel, Merit-Terme) brauchen eine ADR.
- **Anlass:** Befunde F3–F5, F9, F10 und F13 aus [m6-t6-zielspezifikation.md](ergebnisse/m6-t6-zielspezifikation.md)
- **Bezug:** [optimierung.md](optimierung.md) (vorhandener Stand); [sprachmodell-steuerung.md](sprachmodell-steuerung.md) (M10) nutzt mehrere der hier vorgeschlagenen Größen als Diagnose oder Aktion.

Dieses Dokument ändert nichts am vorhandenen Verhalten. Alle Erweiterungen sind abschaltbar und mit dem heutigen Verfahren als Standard einzuführen, bis eine Messung den Wechsel begründet.

## Übersicht

| Nr. | Erweiterung | Zielt auf | Aufwand | Abschnitt |
|---|---|---|---|---|
| M11-T1 | glattes Maximum für Strahlhöhen und Einfallswinkel | F10 (LM endet mit `max_iterations`) | mittel (Kernel-Reduktion) | 1 |
| M11-T2 | Augmented Lagrangian für einseitige Bedingungen | Penalty-Stufen, Kondition, Restverletzung | klein | 2 |
| M11-T3 | Tangentenprädiktor in der Homotopie | LM-Zyklen pro Schritt, Δ < Δ_min | klein | 3 |
| M11-T4 | Verletzung der Sinusbedingung (OSC) als Residuum | F4, F5 (Koma bei kleinem Feld unsichtbar) | klein | 4 |
| M11-T5 | algebraische Glasvorauswahl für Achromasie und Apochromasie | F9, F13 (Farbstufen scheitern) | mittel | 5 |
| M11-T6 | toleranzbewusste Optimierung | Baubarkeit, Vorgriff auf M8-T4 | mittel | 6 |
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
- Deterministische Baumreduktion wie die vorhandenen Max-Reduktionen (ADR-0007). Exp und Log sind keine korrekt gerundeten Funktionen; Bitgleichheit CPU/GPU ist deshalb nicht zu erwarten, wohl aber Wiederholbarkeit auf demselben Gerät. Das gehört in eine ADR (Vorschlag ADR-0013), ebenso eine CPU/GPU-Toleranz.
- β ist Homotopie-Parameter (Fortsetzung in β wie in ε, optimierung.md §7). Am Ende wird β so gewählt, dass ln N/β kleiner als der Vorhalt ist (Standard 5·10⁻⁵ mm). Der Vorhalt bleibt dann gültig.
- β in Einheiten der Größe: 1/mm für Höhen, dimensionslos für cos I.

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

## 4. Sinusbedingung als Residuum (M11-T4)

**Problem.** Lineare Koma wächst mit dem Feld. Bei reiner Achsoptimierung ist sie in der Merit unsichtbar, und die rotationssymmetrischen Geisterplatten zeigen keinen Bedarf (F4). Ein kleines Feld von Anfang an hilft, kostet aber Feldpunkte und verschiebt das Problem in die Kopplung von Apertur und Feld (F5).

**Vorschlag.** Die Verletzung der Sinusbedingung (Offence against Sine Condition, Conrady) aus dem Achsbündel allein. Für das kollimierte Bündel im Unendlichraum, also Mikroptims Tracerichtung, ist sie bei der Pupillenzone ρ_k

$$\mathrm{OSC}(\rho_k) = \frac{H_k}{f\,\sin U'_k} - 1 - \frac{L'_k - l'}{L'_k - l'_p},$$

mit Einfallshöhe H_k, realem Winkel U′_k und Schnittweite L′_k des Zonenstrahls, paraxialer Schnittweite l′ und Lage l′_p der Austrittspupille (in Tracerichtung). Formel und Vorzeichen sind vor der Umsetzung gegen Welford, *Aberrations of Optical Systems*, Kap. 9, und gegen numerik.md zu prüfen.

Das Residuum ist √w · OSC(ρ_k) für zwei bis drei Zonen (etwa die äußeren Gauß-Ringe). Der zweite Term macht die Größe unabhängig von der Restsphärik; aplanatisch ist das System, wenn zusätzlich die sphärische Aberration klein ist, und dafür sorgt die Merit ohnehin.

**Umsetzung.**
- Die Zonenstrahlen liegen im Trace der Achse schon vor (Speichen nahe dem Meridian). Gebraucht wird nur der Austrittswinkel und die Schnittweite des meridionalen Strahls jedes Rings.
- Als trace-abhängiges Zusatzresiduum (optimierung.md §2.1) oder als zusätzliche Ausgabe des Kernels.
- Gleichzeitig Diagnosegröße für M10 (sprachmodell-steuerung.md, Abschnitt 3.2).

**Abnahme.**
- Orakel: aplanatische Fläche bzw. aplanatisches Linsenpaar (Zhang §3.2) ergibt OSC = 0 bis auf Rundung; eine Einzellinse ergibt das Vorzeichen und die Größenordnung aus der Seidel-Koma (S_II ∝ OSC bei kleiner NA).
- Nutzen: Wiederholung des M6-T6-Versuchs „Apertur zuerst auf der Achse“ (F4) mit OSC-Term. RMS bei 7 % und 10 % Feld deutlich unter dem damaligen Wert von 0,28 Wellen.

## 5. Algebraische Glasvorauswahl (M11-T5)

**Problem.** Die Farbstufen scheitern (F9, F13). Die Diagnose empfiehlt keinen Operator B, und die freien Glasvariablen finden aus einem monochromatisch aufgebauten System mit Gläsern ähnlicher Dispersion nicht heraus. Was fehlt, ist eine gezielte Glaswahl.

**Vorschlag.** Vor dem Trace werden Gläser auf dünnen Elementen bewertet. Für m Elemente mit Brechkräften φ_i, paraxialen Randstrahlhöhen h_i (aus dem aktuellen System, normiert auf die erste) und Buchdahl-Dispersionskoeffizienten ν_ij des Materials i (vorhanden, numerik.md §10) lautet die Bedingung für Farbkorrektur in den Buchdahl-Termen j = 1 … k:

$$\sum_i h_i^2\,\varphi_i = \varphi,\qquad \sum_i h_i^2\,\varphi_i\,\nu_{ij} = 0\quad (j = 1,\dots,k).$$

- k = 1 entspricht Achromasie (zwei Gläser), k = 2 Apochromasie (drei Gläser bzw. zwei mit gleicher Teildispersion).
- Für m = k + 1 ist das ein lineares Gleichungssystem in φ_i.
- Bewertet wird nach Σ|φ_i|/|φ| (kleine Einzelbrechkräfte, Zhang §3.2) und nach dem Rest in den nicht korrigierten Termen (sekundäres bzw. tertiäres Spektrum).
- Fast singuläre Systeme werden verworfen.

Grundlage sind die Arbeiten zur Glaswahl mit Buchdahl-Koeffizienten (Robb, Mercado; Rayces und Rosete-Aguilar). Vor der Umsetzung sind die genauen Formulierungen dort nachzulesen.

**Umsetzung.**
- Alle Paare (N²/2) und Tripel (N³/6) der Gläser der Glas-Hülle, vektorisiert auf dem Host. Bei einigen hundert Gläsern sind das bis 10⁷ kleine Systeme, also Sekunden.
- Filter: Regeln aus sprachmodell-steuerung.md, Abschnitt 5.3 (Autofluoreszenz, Transmission, Merte-Paar).
- Ergebnis: Rangliste mit Startbrechkräften. Sie fließt in `split_from_diagnosis` (Zielglas und c_s statt nur nach Abbezahl), in Operator B und in `glass_query` (M10).

**Abnahme.**
- Orakel: Für zwei Gläser entspricht das Ergebnis dem dünnen Achromat (`achromat_split_curvature`).
- Ein konstruiertes Glastripel mit bekannter Lösung wird gefunden.
- Rangliste gegen Bruteforce-Raytrace an einem Kittglied (frei erfunden, NA 0,05): Der beste Kandidat der Vorauswahl liegt unter den besten fünf nach Merit.

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

## 7. Globale Suche in fester Topologie (M11-T7, M11-T8)

### 7.1 Memetische Suche: CMA-ES mit LM

Eine CMA-ES-Population über die Variablen einer festen Topologie, in normierten Koordinaten x/x_typ. Jedes Individuum wird kurz mit dem gebündelten LM verfeinert, etwa 20–50 Zyklen, alle Individuen in einem Aufruf.

Die Verteilung wird aus den verfeinerten Punkten aktualisiert (Lamarck'sche Variante). Gegenüber zufälligen Multistarts lernt die Startverteilung die Lage der guten Becken. Der FP32-Vorfilter (M7-T2) sortiert hoffnungslose Starts vorab aus. Zufallszahlen kommen aus dem vorhandenen, plattformunabhängigen Generator (optimierung.md §4); der Lauf ist mit Startwert reproduzierbar.

**Einsatz:** Feinoptimierung großer Systeme, nach dem Glas-Einrasten, auf den Zwischenständen der Strahlsuche.

**Abnahme:** Auf dem Mehrlinser-Testsystem und einem M6-T6-Zwischenstand wird bei gleicher Zahl von LM-Zyklen eine kleinere beste Merit gefunden als mit `perturbed_starts`.

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
