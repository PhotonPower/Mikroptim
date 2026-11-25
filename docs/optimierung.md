# Optimierung und Systemsynthese

## 1. Variablen

| Variable | Pro | Schranken |
|---|---|---|
| Krümmung c | Fläche | |c| ≤ c_max |
| Dicke bzw. Luftabstand t | Fläche | t_min ≤ t ≤ t_max (Mitten- und Randdicke separat als Nebenbedingung) |
| Glas (n₀, ν₁, ν₂) | Glaselement | konvexe Hülle der Katalog-Gläser (Strafterm) |
| Arbeitsabstand w | System | w ≥ w_min |
| Geisterkoeffizienten a_{g,k} | Geisterplatte, Ordnung | frei, L1-bestraft |

Typische Größenordnung beim Zielsystem: P ≈ 60–100 Variablen.

### Variablenabbildung (ab M3-T1)

Die Variablen stehen im Abschnitt `variables` der Systemdatei ([dateiformat.md](dateiformat.md#variablen-ab-m3-t1)). Implementiert sind Krümmung, Dicke, Arbeitsabstand, (ab M4-T6) Geisterkoeffizienten und (ab M6-T1) Glas.

`mko::VariableSet` (`include/mikroptim/optim/variable_set.hpp`, Bibliothek `Mikroptim::optim`) wird aus Systembeschreibung und Batch-Topologie gebaut und legt die Komponenten des Variablenvektors x ∈ ℝ^P fest (Reihenfolge von `resolve_variables`):

| Typ | Komponente x_k | Element im Batch |
|---|---|---|
| Krümmung c der Dateifläche i | `surfaces[i].curvature` | `curvature[v][m(i)]` |
| Dicke t der Dateifläche i (i < S_Datei − 1) | `surfaces[i].thickness` | `thickness[v][m(i+1) − 1]`, bei Geisterplatten im Spalt x − d_last (siehe unten) |
| Arbeitsabstand w | `surfaces[S_Datei − 1].thickness` | `thickness[v][S − 1]` (einzige Quelle, `image_distance`) |
| Geisterkoeffizient a_k der Platte g (ab M4-T6) | `ghost_plates[g].coefficients[k − 1]` | `ghost_coeff[v][g][k − 1]` |
| Glas: n₀, ν₁, ν₂ des Mediums nach Dateifläche i (ab M6-T1, drei Komponenten) | `surfaces[i].material`: `n0`, `nu[0]`, `nu[1]` des Buchdahl-Materials | `buchdahl_coeff[v][m(i)][c]`, c = 0, 1, 2; `apply` rechnet danach `index[v][m(i)][l]` und `paraxial_index[v][m(i)]` neu |

Geisterkoeffizienten heißen in Berichten `a[g,k]`; `Variable::ghost_plate`/`ghost_order` nennen Platte und Ordnung, `batch_surface` ist die Tabellenfläche der Platte. Standard x_typ = 10⁻⁴ mm (`kDefaultTypicalGhostCoefficient`): Die FD-Schrittweite bei a = 0 ist dann h = 6·10⁻¹⁰ mm, klein gegen die Glättung ε der L1-Strafe (Abschnitt 3), sodass die zentrale Differenz den glatten Bereich der Strafe auflöst ((h/ε)² ≈ 4·10⁻⁷ bei ε = 10⁻⁶ mm); die Wellenfrontspalten bleiben dabei auf etwa 10⁻⁴ relativ genau (OPD-Änderung (n_G − 1)·h gegen die Rundung ε_m·L der optischen Länge). Schranken sind optional (frei). Störung im Variantengenerator relativ wie bei den übrigen Typen (`PerturbationOptions::ghost_coefficient`, Standard 1 %); a = 0 bleibt ungestört.

**Glas** (ab M6-T1): Ein Eintrag `{"type": "glass", "surface": i}` ergibt drei Variablen n₀, ν₁, ν₂ (Namen `n0[i]`, `nu1[i]`, `nu2[i]`, `Variable::glass_component` 0, 1, 2) der Buchdahl-Koordinaten (numerik.md §10) des Mediums nach Dateifläche i. ν₃ bleibt fest. Vor dem Aufbau der Variablenmenge wird das Material in ein Buchdahl-Material umgewandelt (`convert_glass_variables`, `include/mikroptim/optim/glass_variables.hpp`: Katalogglas → 3-Term-Fit über 400–700 nm, konstantes n → n₀ = n, ν = 0), danach Brechzahltabelle und Batch neu (`prepare_system`, `make_system_batch`); `VariableSet` verlangt ein Buchdahl-Material (sonst `std::invalid_argument`). Die Koordinaten pro Variante stehen im Host-Feld `buchdahl_coeff[v][s][0 … 3]` des Batches (docs/architektur.md): Aus den Brechzahlen allein sind sie nicht immer eindeutig zurückzugewinnen (eine Wellenlänge, λ_ref unter den Wellenlängen), und das Feld macht `extract` exakt. `apply` schreibt die drei Koeffizienten und rechnet die Brechzahlen bei allen Wellenlängen und bei λ_ref mit `buchdahl_index` (derselbe Ausdruck wie `material_index`): Der Batch ist danach bitgleich zu dem aus der mit `apply_to_description` geänderten Datei (Test `GlassVariables.VariableSetRoundtripAndIndices`, auch mit Geisterplatten im Luftspalt davor). `apply_to_description` schreibt `n0`, `nu[0]`, `nu[1]`; die Datei enthält danach `{"model": "buchdahl", "n0": …, "nu": [ν₁, ν₂, ν₃]}`. Schranken gibt es nicht (die Hülle der Katalog-Gläser wirkt als Strafterm, Abschnitt 2.4); x_typ ist pro Komponente fest: 1 (n₀), 0,01 (ν₁), 0,001 (ν₂), `kDefaultTypicalGlass*`. Geisterplatten liegen nur in Luftspalten, ein Glasmedium ist im Batch also nie geteilt. Störung im Variantengenerator relativ wie bei den übrigen Typen (`PerturbationOptions::glass`, Standard 1 %).

m = `table_of_file` der `SurfaceMap` (`make_surface_map`, geprüft gegen die Topologie mit `checked_surface_map`: Flächenzahl, Zahl der Geisterplatten, Typ jeder Tabellenfläche); ohne Geisterplatten die Identität. Das ist die einzige Stelle, an der `VariableSet` den Indexversatz durch Geisterplatten berücksichtigt.

**Spalt mit Geisterplatten.** `make_surface_table` teilt den Luftspalt T nach Dateifläche i an den Plattenabständen d_1 ≤ … ≤ d_m in die Dicken d_1, d_2 − d_1, …, T − d_m. Die Variable t_i behält die Bedeutung der Dateidicke T (Schranken, Startwert und `apply_to_description` wie ohne Platten). Festlegung:

- `apply`: nur das letzte Teilstück (Tabellenfläche m(i+1) − 1, die letzte Platte) wird fl(x − d_m), mit genau dem Ausdruck von `make_surface_table`; die Plattenabstände d_g sind feste Daten der Datei (Lage relativ zum Scheitel der vorderen Fläche) und bleiben. Damit ist ein Batch nach `apply` bitgleich zu dem, der aus der mit `apply_to_description` geänderten Datei neu gebaut wird (reproduzierbares Schreiben und Wiederladen optimierter Systeme).
- `extract`: x = fl(d_m + letztes Teilstück).
- Roundtrip Vektor → Batch → Vektor: Für x ≥ d_m ist fl(d_m + fl(x − d_m)) = x, außer wenn x − d_m genau in der Mitte zweier Gleitkommazahlen derselben Binade wie x liegt (dann ±1 ulp; z. B. d = 0,11 mit x in [0,235; 0,25)). Für Plattenabstände mit wenigen Mantissenbits (0,5, 1,0, 3,5 …) tritt der Fall nicht auf (Tests `VariableSet.GhostGapRoundtrip`, `VariableSet.GhostGapRoundtripTieCase`).
- Die untere Schranke wird auf max(`min`, d_m) angehoben, damit die letzte Platte im Spalt bleibt (validate_system verlangt d ≤ T).
- Der Arbeitsabstand ist nie geteilt (nach der letzten Fläche liegen keine Geisterplatten).

- `extract(batch, v)` → x und `apply(batch, v, x)` lesen bzw. schreiben genau diese Elemente; alles andere (Brechzahlen außer denen der Glasvariablen, Feldwinkel, Feldziele, Geisterkoeffizienten) bleibt unverändert. Der Roundtrip Vektor → Batch → Vektor ist bitgleich (Test `VariableSet.RoundtripVectorBatchVector`; einzige Ausnahme: Rundungsfall „genau halb“ bei einem Spalt mit Geisterplatten, siehe unten). `apply` prüft Länge, Endlichkeit und Topologie, bevor es schreibt.
- `extract(system)` und `apply_to_description(system, x)` tun dasselbe auf den Dateiflächen (Schreiben optimierter Systeme, M3-T5).
- Schranken als Vektoren `lower()`, `upper()` (±∞ für offene Schranken), `typical()` für x_typ; `feasible(x)` und die Projektion `clamp_to_bounds(x)`: x_k ← min(max(x_k, lower_k), upper_k).
- Skalierte Koordinaten gibt es bewusst nicht: Die Marquardt-Skalierung mit diag(JᵀJ) (Abschnitt 4) macht LM invariant gegen die Skalierung der Variablen; x_typ dient nur der Schrittweite.

## 2. Residuenvektor

Die Merit ist F = ‖r‖². Der Residuenvektor einer Variante besteht aus:

1. **Wellenfront:** √(ŵ_f · ŵ_λ · W_i · V_j) · OPD_ij / λ_ref für alle Felder f, Wellenlängen λ, Pupillenstützstellen (i, j), mit den auf Summe 1 normierten Gewichten ŵ_f = w_f/Σw_f, ŵ_λ = w_λ/Σw_λ (ab M4-T1, Definition und Begründung: [numerik.md, „Residuen pro Strahl“](numerik.md#residuen-pro-strahl-m2-gewichte-m4-t1)). Die Residuen sind dimensionslos (Wellen bei λ_ref); ohne Versagen ist ihr Anteil an F der gewichtete mittlere RMS² Σ ŵ_f ŵ_λ (RMS_fλ/λ_ref)² in Wellen².
2. **Paraxiale Nebenbedingungen:** √w · (Ist − Soll) für Brennweite (3,29 mm, über die Brechkraft formuliert), Abgleichlänge (45 mm), ggf. Arbeitsabstand (Gleichung oder Untergrenze); Definitionen in Abschnitt 2.2.
3. **Petzval:** √w · max(0, |κ_P| − κ_max), inklusive Beiträgen der Geisterplatten; Definition und Gewichtswahl in Abschnitt 2.3.
4. **Mechanik (einseitig):** √w · max(0, g) für Durchmesser, Mitten- und Randdicken, maximalen Einfallswinkel (Vorgabe ca. 55–60°); ab M4-T3, Definitionen in Abschnitt 2.5. Mit dem Augmented Lagrangian (ab M11-T2, Option) tragen die einseitigen Bedingungen 2–4 eine Verschiebung θ: √w · max(0, g + θ), Abschnitt 2.6. Strahlhöhen und Einfallswinkel kommen aus den Max-Reduktionen des Traces (trace-abhängige Zusatzresiduen, Abschnitt 2.1).
5. **Weiches Versagen** (ab M4-T2): ein Strafresiduum pro Strahl, p_k = √(w_fail · ŵ_f · ŵ_λ · W_i · V_j) · π_k, mit der Summe π_k der dimensionslosen Strafgrößen seiner Ereignisse (−D, −k, −Radikand, −disc/R_ref²; hart versagt zusätzlich 1). Die Strafresiduen stehen als eigener Block gleicher Länge direkt hinter den Wellenfrontresiduen (vor den Zusatzresiduen 2–7) und entstehen im Kernel; weich versagte Strahlen bleiben mit der OPL ihrer Fortsetzung in Kolben und Wellenfrontresiduen. Gewicht w_fail in Wellen² (`ResidualSettings::fail_weight`, Standard 10⁶), im LM mit äußerem Strafverfahren (Abschnitt 4). Regeln, Grenzfälle und Messungen: [numerik.md, Abschnitt 11](numerik.md#11-weiches-versagen).
6. **Geisterplatten (L1, geglättet):** im Plan √(μ·w_k) · (a_{g,k}² + ε²)^{1/4}, sodass das Quadrat μ·w_k·√(a² + ε²) ≈ μ·w_k·|a| ist. Implementiert (ab M4-T6) ist die gleichwertige Form √(μ·w_k) · a/√(ε + √(a² + ε²)) mit dem Quadrat μ·w_k·(√(a² + ε²) − ε): gleicher Gradient, also gleiches Minimum, aber r(0) = 0 und eine Gauß-Newton-Krümmung, die bei a = 0 exakt stimmt (Begründung in Abschnitt 3).
7. **Glas-Hülle:** √w · Abstand außerhalb der Hülle der Katalog-Gläser in normierten Koordinaten, als Vektor √w (u − P(u)) mit drei Residuen pro Glaselement (ab M6-T1, Abschnitt 2.4).
8. **Sinusbedingung** (ab M11-T4, optional): √w · OSC(ρ_k) für die Pupillenzonen ρ_k des Achsbündels, aus je einem realen meridionalen Strahl; steht als paraxiale Bedingung im Block 2 (Abschnitt 2.7).

Gewichte w_f, w_λ stammen aus der Systemdatei (`source.field_weights`, `spectrum.weights`); Gewicht 0 schaltet ein Feld bzw. eine Wellenlänge in der Merit ab (Residuen exakt 0, verfolgt wird es weiter). **Einheit der Merit:** Wellen² bei λ_ref. Alle Zusatzresiduen (Punkte 2–8) sind √w · (Größe in ihrer eigenen Einheit) und gehen mit ihrem Quadrat direkt in dieselbe Summe ein; ihre Gewichte haben daher die Einheit 1/(Einheit der Größe)² und sind relativ zu „ein Residuum von 1 entspricht einer Welle RMS“ zu wählen (Faustregel in Abschnitt 2.2). Geisterplatten-Gewichte steigen mit der Ordnung: w_k = w₀ · κ^k (κ ≈ 2).

### 2.1 Zusatzresiduen (Schnittstelle, ab M3-T3)

Residuen, die nicht pro Strahl im Kernel entstehen (Punkte 2–4, 6, 7), liefert der Host pro Variante über `mko::ExtraResiduals` (`include/mikroptim/optim/extra_residuals.hpp`): `count` Werte pro Variante, `evaluate(batch, v, out)` schreibt sie nach `out` (Länge `count`). Im Residuenvektor einer Variante stehen sie hinter den Strahlresiduen (Platz `OutputRequest::extra_residuals`; die Ablage im Jacobi-Aufbau legen M3-T2/M3-T4 fest). `evaluate_extra_residuals(extra, batch, out, first, count, threads)` (`include/mikroptim/optim/paraxial_residuals.hpp`) wertet die Varianten [first, first + count) auf Host-Threads aus, `out[(v − first)·extra.count + i]`; `evaluate` muss dafür threadsicher sein. Mehrere Quellen verkettet `concat_extra_residuals({…})` (ab M4-T6, `extra_residuals.hpp`): count = Σ count_i, die Teile in der angegebenen Reihenfolge, Teile mit count = 0 entfallen. `mikroptim optimize` verwendet paraxiale Nebenbedingungen, dahinter die L1-Strafe der Geisterkoeffizienten, die Glas-Hülle (M6-T1) und die mechanischen Bedingungen ohne Strahlbezug. Strafblöcke im Kernel (weiches Versagen, M4-T2) gehören zu den Strahlresiduen und sind davon unabhängig.

**Trace-abhängige Zusatzresiduen (ab M4-T3).** Residuen aus Strahlhöhen und Einfallswinkeln hängen vom Trace ab, den `ExtraResiduals` nicht sieht. Sie werden aus den Max-Reduktionen (`SurfaceExtremaResult`, größte Strahlhöhe und kleinster cos I pro Tabellenfläche über die gültigen Strahlen, M2-T3) **derselben** Auswertung berechnet, die auch die Strahlresiduen liefert: `mko::TraceExtraResiduals` (`include/mikroptim/optim/trace_extra_residuals.hpp`) mit `count` und `evaluate(batch, v, extrema, out)`, `extrema` als `VariantExtrema` (2·S Werte der Variante). Im Residuenvektor stehen sie hinter allen übrigen Zusatzresiduen, R = R_v + E + E_t (`FdJacobianOptions::trace_extra`). Jede FD-Variante bekommt die Residuen aus den Extrema ihres eigenen Traces (CPU `surface_extrema_cpu`, GPU die Extrema jeder Tranche der Engine), und die Merit der Versuchspunkte enthält Σ t² aus den Extrema des MERIT-Laufs (CPU `merit_cpu` mit Extrema, GPU Modus MERIT mit `extrema = true`, die Summe Σ t² addiert der Host nach `extra_merit`). Damit ist F wieder die Quadratsumme des ganzen Residuenvektors. Wege und Kosten: [architektur.md](architektur.md#optimierer-trace-abhängige-zusatzresiduen-und-mechanische-nebenbedingungen-m4-t3). `evaluate_trace_extra_residuals(extra, batch, extrema, extrema_first, out, first, count, threads)` wertet Varianten wie `evaluate_extra_residuals` auf Host-Threads aus (Extrema der Variante v beim Eintrag v − `extrema_first`), `trace_extra_merit_values` liefert Σ t² pro Variante.

*Glattheit.* Ein Maximum über Strahlen ist stetig, aber nur stückweise glatt: Wechselt der Strahl, der das Maximum stellt (anderer Ring, andere Speiche, anderes Feld), springt die Ableitung. Zusammen mit dem Knick von max(0, g) ist das Residuum stückweise glatt; für LM ist das akzeptabel (dieselbe Lage wie bei allen einseitigen Strafen). Eine FD-Spalte, deren Schritt einen Trägerwechsel überspannt, mittelt die beiden einseitigen Steigungen.

**Parallelisierung auf dem Host.** Feste, zusammenhängende Aufteilung der Varianten auf T Threads (`std::thread`; `std::execution::par` bräuchte unter GCC TBB, also eine neue Abhängigkeit): Thread c bearbeitet [first + ⌊count·c/T⌋, first + ⌊count·(c+1)/T⌋). Jede Variante wird unabhängig und von genau einem Thread gerechnet, die Ergebnisse sind daher für jede Threadzahl bitgleich (keine Reduktion über Varianten). T ≤ 0 heißt `hardware_concurrency`. Wirft eine Variante, wird nach dem Join die Ausnahme der kleinsten werfenden Variante weitergegeben (unabhängig von T).

### 2.2 Paraxiale Nebenbedingungen (ab M3-T3)

Schema in der Systemdatei: [dateiformat.md](dateiformat.md#nebenbedingungen-ab-m3-t3); Sollwerte stehen in `target`. Jede Bedingung liefert ein Residuum

$$r = \sqrt{w}\;\tilde q,\qquad \tilde q = \begin{cases} q & \text{Gleichung (equal)}\\ \min(q, 0) & \text{Untergrenze (min): aktiv nur für Ist < Soll}\\ \max(q, 0) & \text{Obergrenze (max)} \end{cases}$$

Einseitige Bedingungen sind stetig an der Grenze (r = 0 dort), inaktiv exakt 0; ihre Ableitung springt an der Grenze (wie bei allen Straftermen der Form max(0, g)). Alle q sind Längen in mm, das Gewicht w hat die Einheit 1/mm², sodass r wie die Wellenfrontresiduen dimensionslos ist (Merit in Wellen², Abschnitt 2).

**Wahl der Gewichte (Faustregel, ab M4-T1).** Eine Abweichung δq kostet w·δq², die Wellenfront ŵŵ-gemittelt RMS²/λ_ref². Soll eine Abweichung δq so viel kosten wie ein RMS-Wellenfrontfehler ε (in Wellen), dann ist

$$w \approx \Big(\frac{\varepsilon}{\delta q}\Big)^2,\qquad \text{z. B. } \delta q = 10^{-2}\ \text{mm} \leftrightarrow \varepsilon = 2\cdot 10^{-3}\ \text{Wellen}:\ w \approx 0{,}04\ /\text{mm}^2 .$$

Nach oben begrenzt die Kondition: Ist w·(∂q/∂x)² um viele Größenordnungen größer als die Wellenfrontanteile von JᵀJ, verschwinden die Richtungen, die q kaum ändern (z. B. die Biegung einer Linse bei fester Brennweite), in der Rundung der Normalgleichungen; LM wird langsam oder JᵀJ numerisch singulär (beobachtet in M3-T4). Gewichte so wählen, dass die Bedingung am Optimum einen Beitrag in der Größenordnung des Wellenfrontanteils hat, nicht mehr; eine harte Einhaltung leisten Variablenschranken. Die Abnahmetests von M3-T4 verwenden für die Brennweite 0,029 /mm² (Einzellinse f = 100 mm, NA 0,025, RMS ≈ 10⁻³ Wellen; f bleibt auf 1,6·10⁻⁶ mm) und 0,28 /mm² (Mehrlinser, 3 Felder, 4 Wellenlängen); das sind die vorher für Residuen in mm gewählten Gewichte 10⁻⁸ bzw. 10⁻⁶ /mm², exakt umgerechnet mit dem Faktor 1/(F·L·λ_ref²), um den die Wellenfrontmerit bei gleichen Gewichten gewachsen ist (`tests/lm_test_systems.hpp`). Wegen der Normierung (Abschnitt 2) bleibt dieses Verhältnis erhalten, wenn die Homotopie Felder oder Wellenlängen hinzufügt.

| Typ | q | Soll |
|---|---|---|
| `focal_length` | f_s (1 − f_s φ) = f_s² (φ_s − φ) | f_s = `target.focal_length`, φ_s = 1/f_s |
| `working_distance` | t_{S−1} − w_s | w_s = `target.working_distance` |
| `parfocal_length` | `offset` + Σ_{j = m(r)}^{S−1} t_j − L_s | L_s = `target.parfocal_length` |
| `paraxial_focus` | a (A + t_{S−1} C / n′_L) | 0 |

φ ist die Brechkraft bei λ_ref (−C der Systemmatrix, numerik.md §9), t_j die Dicken der Variante (Tabellenflächen), t_{S−1} der Arbeitsabstand, a der Pupillenradius, n′_L die Brechzahl des Bildraums.

**Brennweite über die Brechkraft.** f = 1/φ ist an φ = 0 unstetig (Sprung von +∞ nach −∞). Ein Residuum √w (f − f_s) würde bei schwach brechenden Startsystemen (Systemsynthese ab einer Linse, Homotopie) beliebig groß, seine Ableitung ∂f/∂x = −f² ∂φ/∂x ebenso; LM wäre schlecht konditioniert. Die Formulierung f_s² (φ_s − φ)

- ist linear in φ und damit für dünne Elemente nahezu linear in den Krümmungen (LM-freundlich),
- ist stetig und beschränkt durch φ = 0 (dort r = √w f_s),
- hat die Einheit mm wie die übrigen Bedingungen und ist nahe dem Soll gleich der Längenabweichung: Mit f = f_s (1 + ε) ist q = f_s ε/(1 + ε) = (f − f_s)(1 − ε + O(ε²)). Das Gewicht hat also dieselbe Bedeutung wie bei √w (f − f_s).

Vorzeichen: r > 0 ⇔ φ < φ_s (bei Sammellinsen f > f_s). Nur als Gleichung definiert, weil „f ≥ f_min“ über φ = 0 hinweg keine stetige einseitige Bedingung in φ ergibt.

**Arbeitsabstand** ist die Dicke der letzten Fläche (`thickness[v][S−1]`, einzige Quelle). Die Spezifikation lässt offen, ob es einen Zielwert oder eine Untergrenze gibt; beides geht (`mode` `equal` bzw. `min`). Die einseitige Bedingung ist eine weiche Strafe; die Variable `working_distance` mit `min` ist dagegen eine harte Schranke (Projektion, Abschnitt 4).

**Abgleichlänge** („Anlagefläche bis Objektebene“). Die Anlagefläche liegt nicht im Linsenmodell; gemessen wird ab dem Scheitel der Referenzfläche r (Dateifläche, Standard 0 = Blende, m(r) = zugehörige Tabellenfläche) bis zur Bildebene, plus dem Abstand `offset` der Anlagefläche vor diesem Scheitel. Die Summe läuft in fester Reihenfolge (`offset` zuerst, dann j aufsteigend). Ein Luftspalt mit Geisterplatten trägt die Summe seiner Teilstücke bei (bis auf Rundung die Dateidicke). Die tatsächliche Lage der Anlagefläche legt der Maintainer fest.

**Bildebene im paraxialen Fokus.** q ist die Höhe des paraxialen Randstrahls mit Einfallshöhe a in der Bildebene: Nach der letzten Fläche hat er die Höhe a A und den Winkel a C/n′_L. Es gilt q = −a C δ/n′_L mit dem Defokus δ = s′_F − t_{S−1} (numerik.md §9). Verwendet wird die Strahlhöhe statt δ, weil δ = −n′_L A/C − t bei C → 0 divergiert, die Strahlhöhe aber in allen Daten stetig (polynomiell) ist. Nur Gleichung, nur kollimierte Quelle (Ziel ist F′). Bei hoher NA liegt der beste Fokus nicht im paraxialen Fokus; die Bedingung ist für Startphasen mit kleiner NA gedacht (Homotopie-Schritt 1), danach den Arbeitsabstand über die Wellenfront bestimmen lassen.

**Paraxialtrace aller Varianten.** `batch_paraxial(batch, v)` rechnet `compute_paraxial` über die Tabellenflächen der Variante mit den Brechzahlen bei λ_ref (`paraxial_index[v][s]`, architektur.md) – λ_ref muss nicht unter den Wellenlängen des Batches sein (z. B. Homotopie-Schritt 4 mit 480 und 644 nm bei λ_ref = 546 nm). Geisterplatten gehen als dünne Elemente mit φ_G = −2 (n_G − n_m) t₂ aus `ghost_coeff`, `ghost_norm_radius` und `paraxial_ghost_index` ein (numerik.md §6). Für eine aus einem System gebaute Variante ist das Ergebnis bitgleich zu `compute_paraxial(system, catalog)`. `batch_paraxial_all(batch, threads)` rechnet alle Varianten parallel (Aufteilung wie oben). Der Trace läuft nur, wenn eine Bedingung ihn braucht (Brennweite, paraxialer Fokus); Arbeitsabstand und Abgleichlänge lesen nur Dicken.

API (`include/mikroptim/optim/paraxial_residuals.hpp`, Bibliothek `Mikroptim::optim`):

```cpp
ParaxialData batch_paraxial(const SystemBatchHost&, int v);
std::vector<ParaxialData> batch_paraxial_all(const SystemBatchHost&, int threads = 0);
struct ParaxialConstraint { ConstraintType type; ConstraintMode mode; double weight, target;
                            int reference_surface /* Tabellenfläche */; double offset; int item; };
struct ParaxialConstraints { std::vector<ParaxialConstraint> items; int size() const;
                             bool needs_trace() const; };
ParaxialConstraints make_paraxial_constraints(const SystemDescription&); // target + constraints
void check_paraxial_constraints(const ParaxialConstraints&, const BatchTopology&);
double paraxial_residual(const ParaxialConstraint&, const SystemBatchHost&, int v, const ParaxialData&);
void paraxial_residuals(const ParaxialConstraints&, const SystemBatchHost&, int v, std::span<double> out);
ExtraResiduals make_paraxial_residuals(ParaxialConstraints);   // count = size(), threadsicher
void evaluate_extra_residuals(const ExtraResiduals&, const SystemBatchHost&, std::span<double> out,
                              int first = 0, int count = -1, int threads = 0);
```

Fehler: `FormatError` beim Auflösen aus der Datei (wie `check_constraints`), `std::invalid_argument` bei Bedingungen, die nicht zur Topologie passen (Referenzfläche außerhalb oder Geisterplatte, `paraxial_focus` bei Punktquelle, Modus, Gewicht), bei falscher Ausgabelänge und aus `compute_paraxial` (z. B. `paraxial_index` ≤ 0 in einem von Hand gebauten Batch); `std::out_of_range` bei ungültiger Variante bzw. ungültigem Bereich.

### 2.3 Petzval-Nebenbedingung (ab M4-T4)

Schema: Typ `petzval` in [dateiformat.md](dateiformat.md#nebenbedingungen-ab-m3-t3). Residuum

$$r = \sqrt{w}\;\max\bigl(0,\ |\kappa_P| - \kappa_{\max}\bigr),\qquad \kappa_P = -\,n'_L\,P,\qquad P = \sum_i c_i\,\frac{n'_i - n_i}{n_i\,n'_i} + \sum_G \frac{\varphi_G}{n_m\,n_G}$$

mit der Petzvalsumme P bei λ_ref aus `batch_paraxial` (Brechzahlen `paraxial_index`, Geisterplatten mit φ_G = −2 (n_G − n_m) t₂ und `paraxial_ghost_index`, numerik.md §6 und §9), n′_L der Brechzahl des Bildraums und κ_max = `max_curvature` aus dem Eintrag.

- **Einheiten:** q = |κ_P| − κ_max in 1/mm, das Gewicht w daher in mm² (die übrigen paraxialen Bedingungen haben q in mm und w in 1/mm²); √w·q ist dimensionslos wie die Wellenfrontresiduen in Wellen. Nach der Faustregel aus Abschnitt 2.2 kostet eine Überschreitung δκ so viel wie ein RMS-Wellenfrontfehler ε bei w ≈ (ε/δκ)², z. B. δκ = 10⁻⁴/mm ↔ ε = 10⁻² Wellen: w ≈ 10⁴ mm².
- **Grenze im Eintrag:** κ_max steht als `max_curvature` [1/mm] im Eintrag und nur dort (kein Gegenstück in `target`). Ein Mindest-Petzvalradius R_min entspricht κ_max = 1/R_min; die Spezifikation (R ≥ ca. 130 mm, spezifikation.md) ergibt κ_max = 1/130 mm ≈ 0,00769/mm. Die Krümmung statt des Radius ist die Quelle, weil das Ziel (ebenes Feld) bei κ = 0 liegt, wo der Radius unendlich ist, und weil P für feste Brechzahlen linear in den Krümmungen und Geisterkoeffizienten und unabhängig von den Dicken ist: Das Residuum ist stückweise linear in den Variablen.
- **Betrag:** Eine Bildschale ist in beiden Richtungen schädlich; das Vorzeichen von κ_P hängt davon ab, ob sammelnde oder zerstreuende Glieder überwiegen. |κ_P| ist bei κ_P = 0 nicht differenzierbar, dort ist die Bedingung für κ_max > 0 aber inaktiv (r = 0 in einer Umgebung). Für κ_max = 0 ist r = √w |κ_P| und r² = w κ_P² glatt.
- **Einseitig und glatt im Quadrat:** Nur Modus `max` (Standard). Inaktiv ist r exakt 0, an der Grenze ist r stetig (r = 0). Der Beitrag r² = w · max(0, g)² mit g = |κ_P| − κ_max ist an der Grenze C¹ (Ableitung 2 w g ∂g/∂x → 0), erst die zweite Ableitung springt.

**Gewichtswahl.** Als Strafterm erreicht die Bedingung die Grenze nur asymptotisch. Ist sie im Optimum aktiv, gilt dort 2 w g* = λ mit dem Lagrange-Multiplikator λ (Abnahme des übrigen Merits pro Einheit |κ_P|), die Überschreitung ist also g* = λ/(2w) ∝ 1/w. Das Gewicht muss daher zur gewünschten Toleranz δ passen: w ≥ λ/(2δ). λ ist vorab unbekannt, deshalb wird w in Stufen erhöht (**Penalty-Fortsetzung**): Faktor 10 pro Stufe, jede Stufe startet am Ergebnis der vorigen, bis die Überschreitung klein genug ist. Das Produkt w · g* ist über die Stufen nahezu konstant und schätzt λ/2 (Abnahme: w · g* ≈ 0,13 bis 0,24 mm). Weil die Merit seit M4-T1 normiert in Wellen² gerechnet wird, bleibt dieses Verhältnis erhalten, wenn Felder oder Wellenlängen hinzukommen.

Direkt mit großem Gewicht zu beginnen, ist ungünstig: Liegt die Überschreitung unter der Änderung von |κ_P| über die FD-Schrittweite (h = ε^{1/3} · max(|x|, x_typ), im Abnahmesystem rund 10⁻⁷/mm), überspannt das zentrale FD-Muster den Knick von max(0, ·). Die Jacobi-Zeile ist dann nur etwa halb so groß wie die einseitige Ableitung, das Gauß-Newton-Modell passt nicht, und LM kommt nur in kleinen Schritten voran. Beobachtet im Abnahmesystem (während der Entwicklung mit Residuen in mm; w = 1 mm² entspricht heute 1,7·10⁶ mm²): Start mit diesem Gewicht direkt vom Startsystem endet nach 500 Zyklen mit `max_iterations` 20 % über dem beschränkten Optimum, dieselbe Bedingung als Gleichung konvergiert in 147 Zyklen. Mit der Fortsetzung liegt der Punkt bei kleinem w im glatten aktiven Bereich (Überschreitung ≫ FD-Änderung), läuft dort zum beschränkten Optimum, und die späteren Stufen mit kleiner Überschreitung brauchen nur noch 8–18 Zyklen bis zur Konvergenz. Die Gewichtsstufen der Systemsynthese übernimmt der Homotopie-Controller (Abschnitt 7, ab M6-T3: generische Penalty-Fortsetzung aller einseitigen Bedingungen).

**Abnahme** (`tests/test_petzval_optimization.cpp`, GPU `tests/test_cuda_petzval_optimization.cpp`): frei erfundener Zweilinser `data/systems/zweilinser_petzval.json` (Sammellinse n = 1,8, Luftspalt, Zerstreuungslinse CaF₂, f = 20 mm, NA ≈ 0,1, zwei Felder, eine Wellenlänge), Variablen: vier Krümmungen, Luftspalt, Arbeitsabstand (mit Schranken); Merit = Wellenfront (Wellen²) + Brennweite (Gewicht 16.767,7/mm²) + Petzval mit κ_max = 0,01/mm (Petzvalradius ≥ 100 mm, Gewicht 1,677·10⁹ mm²) bei Start-κ_P = −0,0268/mm. Die Gewichte sind die zuerst für Residuen in mm gewählten 0,01/mm² und 1.000 mm², exakt umgerechnet mit 1/(F·L·λ_ref²) = 1/(2·(546,07 nm)²); das Problem ist damit bis auf einen konstanten Faktor dasselbe, und die Zahlen unten stimmen bis auf Rundung mit dem Lauf vor M4-T1 überein. Fortsetzung über sieben Stufen w = 1,677·10³ … 1,677·10⁹ mm² (Faktor 10 pro Stufe, das letzte ist das Gewicht der Datei), LM auf der CPU:

| Kriterium | Ergebnis |
|---|---|
| alle Stufen konvergiert | ja (462, 18, 15, 15, 8, 18, 10 Zyklen; GPU (RTX 3090): 467, 18, 15, 17, 8, 14, 10 mit (\|κ_P\| − κ_max)/κ_max = 1,4·10⁻⁸ und F relativ 1,4·10⁻¹⁰ zur CPU) |
| \|κ_P\| ≤ κ_max nach der Fortsetzung | (\|κ_P\| − κ_max)/κ_max = 1,4·10⁻⁸ (Toleranz `kPetzvalPenaltyRelative` = 10⁻⁶) |
| Bedingung aktiv | Überschreitung > 0 in jeder Stufe, ∝ 1/w; ohne die Bedingung läuft LM vom Ergebnis weg zu κ_P = −0,01067/mm und kleinerem Merit |
| Brennweite eingehalten | f − f_s = 4,8·10⁻⁷ mm (Toleranz `kPetzvalFocalLength` = 10⁻⁵ mm) |

**API** (`include/mikroptim/model/constraints.hpp`, `include/mikroptim/optim/paraxial_residuals.hpp`): `ConstraintType::Petzval`, `ConstraintSpec::max_curvature`, `ConstraintSpec::mode_or_default()` liefert bei Petzval `Max`, `constraint_bound(system, spec)` (Grenze bzw. Sollwert eines Eintrags; `constraint_target` bleibt für Petzval leer). In `ParaxialConstraint` steht κ_max in `target`, der Modus ist `Max`; `check_paraxial_constraints` lehnt andere Modi und κ_max < 0 ab. Die Bedingung braucht den Paraxialtrace (`needs_trace()`) und läuft wie die übrigen über `make_paraxial_residuals` und `evaluate_extra_residuals`.

### 2.4 Glas-Hülle (ab M6-T1)

API: `include/mikroptim/glass/convex_hull.hpp` (Hülle im ℝ³), `include/mikroptim/glass/glass_hull.hpp` (Hülle der Gläser, Bibliothek `Mikroptim::glass`), `include/mikroptim/optim/glass_variables.hpp` (Umwandlung, Strafe, `Mikroptim::optim`); Schema: Variablentyp `glass` und Nebenbedingung `glass_hull` in [dateiformat.md](dateiformat.md#nebenbedingungen-ab-m3-t3).

**Koordinaten und Normierung.** Jedes Katalogglas, dessen `range_nm` den Fitbereich 400–700 nm abdeckt, wird mit dem 3-Term-Buchdahl-Fit (dieselben Einstellungen wie bei der Umwandlung des Startglases, numerik.md §10) zum Punkt x = (n₀, ν₁, ν₂); Gläser ohne Abdeckung stehen in `GlassHull::excluded()`. Normiert wird pro Koordinate mit Mittelwert m_j und Spannweite s_j = max − min über die Gläser der Hülle: u_j = (x_j − m_j)/s_j (`GlassHull::normalize`, Konstanten `center()`, `scale()`; Spannweite 0 → s_j = 1). Eine normierte Einheit ist die Ausdehnung des Katalogs in dieser Koordinate; n₀, ν₁, ν₂ zählen damit gleich, unabhängig von ihren Größenordnungen (n₀ − 1 ≈ 0,5, ν₁ ≈ 0,1, ν₂ ≈ 0,01). Für die frei erfundenen Testgläser (`data/glass/synthetisch.json`) ist m = (1,6272; −0,09844; 0,05024), s = (0,3806; 0,1949; 0,2137).

**Hülle.** Konvexe Hülle der normierten Punkte, eigener Code (`convex_hull`): Starttetraeder aus den extremen Punkten, danach inkrementell nach fallendem Abstand vom Schwerpunkt; ein Punkt erweitert die Hülle, wenn er mehr als ε = 10⁻¹⁰ · Skala vor einem Dreieck liegt (Skala = max(Diagonale der Hüllbox, größter Koordinatenbetrag)); der Horizont sind die gerichteten Kanten sichtbarer Dreiecke ohne sichtbares Gegenstück, die neuen Dreiecke übernehmen ihre Richtung (Orientierung nach außen). Ergebnis: Randdreiecke mit äußeren Einheitsnormalen (koplanare Dreiecke werden nicht zusammengefasst). Entartete Kataloge ergeben eine Hülle niedrigerer Dimension, entschieden mit derselben Toleranz: alle Punkte gleich → Punkt, kollinear → Strecke (Endpunkte), koplanar → konvexes Polygon (monotone Kette nach Andrew in der Ebene, Fächertriangulierung). `basis.json` allein (drei Gläser) ergibt ein Dreieck, die Testgläser einen Polyeder mit 15 Ecken und 26 Dreiecken.

**Abstand.** d(u) = min über die Randdreiecke des Abstands zum Dreieck (Fallunterscheidung über die Voronoi-Gebiete von Ecken, Kanten und Fläche), bei einem Polyeder 0, wenn u hinter allen Ebenen liegt; vorzeichenbehaftet (nur Polyeder) innen −min_i (b_i − n_i·u), der Abstand zum Rand. Bei niedrigerer Dimension ist alles außerhalb der Punktmenge selbst „außen“ (etwa ν₂ abseits der Ebene der drei Basisgläser). Glattheit: P(u) (nächster Punkt) ist stetig, also ist d außerhalb C¹ mit ∇d = (u − P(u))/d, d² überall C¹ mit ∇d² = 2 (u − P(u)), aber nicht C² (die Hesse-Matrix springt zwischen den Gebieten von Ecken, Kanten und Facetten); d selbst hat am Rand einen Knick (innen 0, außen Steigung 1).

**Residuum.** Pro Glaselement drei Residuen, der Vektor

$$r = \sqrt{w}\,\max(0, \varphi(u))\,g(u),\qquad \varphi = \begin{cases} d(u) & \text{außen}\\ \text{signed\_distance}(u) + \delta & \text{Polyeder mit Sicherheitsabstand } \delta\end{cases},\qquad g = \nabla\varphi,$$

ohne δ also r = √w (u − P(u)) außen und 0 innen; |r|² = w max(0, φ)² ist der Strafbeitrag (die Form „√w · Abstand“ des Plans) und C¹. Mit δ > 0 (nur bei einer dreidimensionalen Hülle) beginnt die Strafe δ vor dem Rand: innen ist φ = δ − (Abstand zur nächsten Randebene), also auf den Facetten genau der Abstand zur um δ nach innen versetzten Fläche; g ist dort die Normale dieser Ebene.

*Warum ein Vektor:* Mit dem skalaren Residuum √w · d fehlt im Gauß-Newton-Modell die Krümmung der Strafe quer zum Gradienten: Liegt der nächste Punkt auf einer Ecke oder Kante, ist ∇²(w d²/2) = w (I bzw. I − e eᵀ), das Modell JᵀJ = w g gᵀ hat davon nur den Anteil in Richtung g. Beobachtet (Abnahmesystem, w = 100): LM im Zickzack in ν₂, Dämpfung ≈ 350, nach 1000 Zyklen `max_iterations` 1,5·10⁻³ außerhalb, und mit wachsendem w blieb der Abstand bei 8,8·10⁻⁴ stehen. u − P(u) ist dagegen stückweise linear (Abstand zu einem affinen Unterraum: Ecke, Kante, Facettenebene), das Gauß-Newton-Modell ist in jedem Stück exakt; dieselbe Fortsetzung konvergiert damit in jeder Stufe (Tabelle unten).

**Gewicht.** w in Wellen² pro normierte Einheit² (Standard `kDefaultGlassHullWeight` = 10⁴: d = 0,01, also 1 % der Katalogspanne, kostet eine Welle²). Wie bei Petzval (Abschnitt 2.3) erreicht die Strafe den Rand nur asymptotisch, d* ≈ λ/(2w) mit dem Lagrange-Multiplikator λ; die Tests nutzen die Penalty-Fortsetzung (Faktor 10 pro Stufe, jede Stufe ab dem Ergebnis der vorigen). w·d ist über die Stufen nahezu konstant (Abnahmesystem 0,13–0,25).

**Ablauf** (`mikroptim optimize` macht ihn automatisch, [cli.md](cli.md#mikroptim-optimize)): `convert_glass_variables` (Startgläser → Buchdahl), `prepare_system`, `make_system_batch`, `VariableSet`; `make_glass_hull(loaded.catalog)` (alle Gläser aller `glass_catalogs`); `make_glass_hull_penalty(system, set, hull)` (Gewicht und δ aus dem Eintrag `glass_hull`, ein Element pro Glasfläche; δ > 0 mit ebener Hülle → `std::invalid_argument`); `make_glass_hull_residuals` über `concat_extra_residuals` hinter die übrigen Zusatzresiduen. Hilfsfunktionen für Berichte und Tests: `glass_hull_excess` (max(0, φ)), `glass_hull_residual` (Vektor), `glass_hull_penalty` (|r|²) und `glass_hull_penalty_gradient` (2 w max(0, φ) g_j/s_j in Rohkoordinaten).

**Hüllentests** (`tests/test_convex_hull.cpp`, `tests/test_glass_hull.cpp`, `tests/test_glass_variables.cpp`): Würfel mit inneren, Flächen- und Kantenpunkten → genau die 8 Ecken, 12 Dreiecke in 6 Ebenen, Volumen 1, Abstand und vorzeichenbehafteter Abstand gegen die analytische Formel (2000 Punkte, Abweichung 0); Tetraeder mit bekannten Facetten und Projektionen auf Fläche, Kante, Ecke; 12 zufällige Punktwolken (8–52 Punkte, ein Drittel auf einer Kugel) mit Facetten gleich den Brute-Force-Facetten, geschlossene orientierte Fläche (Euler V − E + F = 2), Abstand gegen eine unabhängige Projektion auf alle Facetten, Kanten und Ecken (< 10⁻¹²); entartete Fälle Punkt, Strecke, Dreieck, ebenes Gitter (4 Ecken im Umlaufsinn), fast koplanar (10⁻¹³ → eben, 10⁻⁶ → räumlich); Gradient gegen zentrale Differenzen in allen Dimensionen (< 10⁻⁷); Testgläser gegen eine unabhängige Python-Referenz (numpy-Fit, scipy/Qhull): dieselben 15 Ecken, Volumen und Oberfläche auf 2·10⁻¹³ relativ, Normierung; Abstand entlang der Facettennormalen; Strafe innen 0, außen √w (u − P(u)), Gradient des Strafbeitrags gegen zentrale Differenzen außen, im Band des Sicherheitsabstands und für die ebene Hülle.

**Abnahme „Optimierung bleibt innerhalb der Hülle“** (`tests/test_glass_optimization.cpp`, GPU `tests/test_cuda_glass_optimization.cpp`): frei erfundene Einzellinse `data/systems/einzellinse_glasvariable.json` (f = 50 mm, NA 0,05 auf der Achse, Wellenlängen 486,13/546,07/656,27 nm, Start N-BK7), Variablen c₁, c₂, Glas (n₀, ν₁, ν₂) und Arbeitsabstand, Brennweite als Gleichung, Hülle aus `basis.json` und den Testgläsern (39 Gläser, 17 Ecken). Der Farblängsfehler verschwindet nur ohne Dispersion, also weit außerhalb jeder Glas-Hülle. Ergebnisse (CPU):

| Kriterium | Ergebnis |
|---|---|
| Kontrollfall ohne Strafe (300 Zyklen) | Glas läuft nach ν₁ = 4,5·10⁻⁶, ν₂ = −5,4·10⁻⁵ (normierter Abstand 0,27), F = 1,27·10⁻⁴ Wellen² |
| Penalty-Fortsetzung w = 10² … 10⁸ | alle Stufen konvergiert (315, 23, 17, 14, 14, 15, 14 Zyklen; GPU (RTX 3090): 315, 22, 17, 17, 15, 15, 15 mit Abstand 2,2·10⁻⁹ und F relativ 1,0·10⁻¹⁰ zur CPU) |
| Abstand am Ende | 2,2·10⁻⁹ normiert (Toleranz `kGlassHullPenaltyDistance` = 10⁻⁶), ∝ 1/w; das Glas sitzt auf der Ecke CaF₂ (kleinste Dispersion der Hülle) |
| Merit schlechter als ohne Strafe | 1,968·10⁻² gegen 1,27·10⁻⁴ Wellen² (ohne Strafterm gerechnet) |

### 2.5 Mechanische Nebenbedingungen (ab M4-T3)

API: `include/mikroptim/optim/mechanical_constraints.hpp` (`Mikroptim::optim`); Schema: Typen `max_semi_diameter`, `min_center_thickness`, `max_center_thickness`, `min_edge_thickness`, `max_incidence_angle` in [dateiformat.md](dateiformat.md#nebenbedingungen-ab-m3-t3). Jeder Eintrag gilt für eine Auswahl von Dateiflächen (Dicken: für den Spalt von Fläche i bis i + 1) und liefert pro Fläche ein Residuum

$$r = \sqrt{w}\,\max(0,\ g),$$

inaktiv exakt 0, an der Grenze stetig (r = 0); r² = w·max(0, g)² ist an der Grenze C¹ (wie Petzval, Abschnitt 2.3). Die Grenzen stehen im Eintrag (keine Gegenstücke in `target`), mehrere Einträge eines Typs sind erlaubt (z. B. Glas und Luft getrennt).

| Typ | Istgröße | g | Gewicht w | Quelle |
|---|---|---|---|---|
| `max_semi_diameter` | h_max(i) + δ: größte Strahlhöhe auf Fläche i plus Fassungszugabe δ = `clearance` | h_max + δ − D_i, D_i = `semi_diameter` des Eintrags oder der Fläche | 1/mm² | Extrema |
| `min_center_thickness` | Mittendicke t_i des Spalts i → i + 1 | t_min − t_i | 1/mm² | Batch |
| `max_center_thickness` | t_i | t_i − t_max | 1/mm² | Batch |
| `min_edge_thickness` | Randdicke e_i = t_i − sag(c_i, h) + sag(c_{i+1}, h) | e_min − e_i | 1/mm² | Batch bzw. Extrema |
| `max_incidence_angle` | min cos I auf Fläche i | cos θ_max − min cos I | dimensionslos | Extrema |

„Batch“: gewöhnliches Zusatzresiduum (`MechanicalResiduals::geometric`, steht in `mikroptim optimize` hinter Paraxial, L1-Strafe und Glas-Hülle), „Extrema“: trace-abhängiges Zusatzresiduum (`MechanicalResiduals::traced`, Abschnitt 2.1). Die Krümmungen c sind die wirksamen der Variante (Ebene und Blende: 0), t_i die Summe der Tabellendicken des Spalts (ein Luftspalt mit Geisterplatten zählt ganz). Glas oder Luft (`medium`) entscheidet das Material nach Fläche i in der Datei (`"AIR"` ist Luft, alles andere Glas); ohne `medium` gilt ein Dicken-Eintrag für beide.

**Pfeilhöhe** (Konventionen aus numerik.md §2: z in Tracerichtung, c = 1/R > 0 mit dem Krümmungsmittelpunkt auf der +z-Seite):

$$\mathrm{sag}(h) = \frac{c\,h^2}{1 + \sqrt{1 - c^2 h^2}},\qquad 1 - c^2h^2 = (1 - |c|h)(1 + |c|h).$$

Der Punkt der Fläche im Abstand h von der Achse liegt bei z = sag(h); für c > 0 ist sag > 0 (die Fläche weicht am Rand in +z zurück). Die Form ist für alle |c| h < 1 ohne Auslöschung, exakt 0 für c = 0, ungerade in c und gerade in h (Test `MechanicalSag.OnSphereAndSymmetric` prüft c (h² + sag²) − 2 sag = 0, also den Punkt auf der Sphäre, und die Reihe c h²/2 + c³h⁴/8 + …). Für |c| h ≥ 1 (Halbdurchmesser über dem Krümmungsradius, nicht herstellbar) gilt die Fortsetzung sag = c h²: stetig bei |c| h = 1 (Halbkugel, sag = 1/c) und weiter wachsend, sodass die Randdicke eine endliche, mit h fallende Strafe ergibt statt NaN; die Ableitung ist dort (von unten) unbeschränkt.

**Randdicke.** Zwischen Fläche i (Scheitel bei z = 0) und Fläche i + 1 (Scheitel bei z = t_i) ist der Abstand bei der Höhe h

$$e_i(h) = t_i - \mathrm{sag}(c_i, h) + \mathrm{sag}(c_{i+1}, h).$$

Bikonvexe Linse (c_i > 0 > c_{i+1}): e = t − |sag_i| − |sag_{i+1}|; Meniskus (gleiche Vorzeichen): t − sag_i + sag_{i+1}; konvexe Vorderfläche nach der Blende (c_1 > 0): der Luftrand e_0 = t_0 + sag(c_1, h) ist dicker als die Mitte, eine Meniskus-Vorderfläche (c_1 < 0) macht ihn dünner (Test `MechanicalSag.EdgeThicknessBiconvexMeniscusPlano`). Die Höhe h ist mit `height`:

- `"semi_diameter"` (Standard): h = max(D_i, D_{i+1}) + δ aus den mechanischen Halbdurchmessern der Datei. Unabhängig vom Trace, e ist eine glatte Funktion von c und t.
- `"rays"`: h = max(h_max(i), h_max(i+1)) + δ aus den größten Strahlhöhen beider Flächen (Randdicke bei der freien Öffnung plus Fassungszugabe), trace-abhängig. Mit den Variablen wächst und schrumpft die freie Öffnung mit, die Bedingung folgt ihr.

**Einfallswinkel.** Die Extrema speichern cos I (exakte Reduktion, kein `acos` im Kernel); I ≥ θ_max ⇔ cos I ≤ cos θ_max, also g = cos θ_max − min cos I. Das Residuum ist glatt in cos I und vermeidet die Singularität von arccos bei I = 0; um θ_max = 55–60° entspricht 1° rund 0,014–0,015 in cos. Die Grenze steht in Grad im Eintrag (`angle`, 0 < θ < 90°), aufgelöst als `limit` = cos(θ π/180).

**Was die Extrema sehen.** Maxima über die Stützstellen der Pupillenquadratur und alle Felder und Wellenlängen, nur Strahlen mit Status Ok (weich versagte zählen nicht; sie werden durch ihre Strafresiduen ohnehin in den gültigen Bereich gedrängt; ohne gültigen Strahl Höhe 0 und cos I = 1, die Bedingung ist dann inaktiv). Der Randstrahl ρ = 1 ist nicht dabei: Der äußerste Gauß-Ring liegt bei ρ_N = √u_N (6 Ringe: 0,9830, 10 Ringe: 0,9935, 16 Ringe: 0,9973), die Speichen bei (j + ½)π/M neben dem Meridian. Strahlhöhen werden deshalb um etwa (1 − ρ_N) des Bündelradius unterschätzt; die Fassungszugabe δ sollte das abdecken. Flächenindizes der Extrema sind Tabellenflächen; `make_mechanical_constraints` bildet die Dateiflächen der Einträge mit der `SurfaceMap` ab.

**Glattheit.** Mittendicke und Randdicke mit festem h sind glatt in den Variablen (bis auf den Knick von max(0, ·)). Strahlhöhe, Einfallswinkel und Randdicke mit h aus den Strahlen sind Maxima über Strahlen und damit stückweise glatt (Abschnitt 2.1); bei der Randdicke kommt der Wechsel zwischen den beiden Flächen in max(h_max(i), h_max(i+1)) hinzu.

**Gewichte und Penalty-Fortsetzung.** Wie bei Petzval erreicht die Strafe die Grenze nur asymptotisch, g* ≈ λ/(2w) mit dem Lagrange-Multiplikator λ; das Gewicht wird in Stufen erhöht (Faktor 10, jede Stufe ab dem Ergebnis der vorigen). Faustregel aus Abschnitt 2.2: w = 10⁶/mm² lässt eine Verletzung von 1 µm eine Welle² kosten. Untergrenze der erreichbaren Verletzung: Sobald g* kleiner wird als die Änderung von g über die FD-Schrittweite (h_k |∂g/∂x_k| ≈ 10⁻⁵ mm bei Krümmungen um 0,2/mm und Strahlhöhen um 3 mm, ≈ 10⁻⁶ in cos), überspannt das zentrale Muster den Knick von max(0, ·), die Jacobi-Zeile stimmt nicht mehr, LM verwirft Schritte und endet über `converged_step`; höhere Gewichte verkleinern g dann kaum noch (beobachtet im Abnahmesystem: w = 10⁷ … 10⁹ lassen g bei 1–3·10⁻⁵ mm). Anders als bei Petzval (P linear in den Krümmungen, das Modell auf der aktiven Seite exakt) sind diese Bedingungen nichtlinear. Wer die Grenze strikt einhalten muss, setzt einen Vorhalt: die Grenze um einige 10⁻⁵ mm (bzw. über `clearance`) enger als die mechanische Grenze. Gewichtsstufen und Vorhalt übernimmt der Homotopie-Controller (Abschnitt 7, umgesetzt in M6-T3).

**Abnahme** (`tests/test_mechanical_optimization.cpp`, GPU `tests/test_cuda_mechanical.cpp`): frei erfundener Zweilinser `data/systems/zweilinser_mechanik.json` (zwei Sammellinsen n = 1,6, f = 10 mm mit Gewicht 10⁴/mm², Pupillenradius 4 mm, Felder 0° und 2°, eine Wellenlänge), Variablen: vier Krümmungen, zwei Mittendicken in [0,5; 2] mm, Luftspalt in [0,1; 10] mm, Arbeitsabstand. Bedingungen: Strahlhöhe auf Fläche 1 ≤ 4,02 mm und auf den Flächen 3, 4 ≤ 2,65 mm, Luftabstände ≤ 8 mm, Glasränder ≥ 1,35 mm bei der freien Öffnung plus 0,3 mm, Luftränder ≥ 0,1 mm bei den mechanischen Halbdurchmessern, Einfallswinkel ≤ 26°. Fortsetzung w = 10² … 10⁶ (alle mechanischen Gewichte gleich, das letzte ist das der Datei), LM auf der CPU:

| Kriterium | Ergebnis |
|---|---|
| Kontrolle: freies Optimum (mechanische Gewichte 0) verletzt jede Art | Strahlhöhe Fläche 1 4,0279 mm (> 4,02), Luftspalt 10 mm (> 8, Variablenschranke), Glasrand 1,2752 mm (< 1,35), Einfallswinkel Fläche 3 28,38° (> 26); F = 1,114 Wellen² |
| alle Stufen konvergiert | ja (CPU 28, 8, 37, 18, 13 Zyklen; GPU (RTX 3090) 28, 8, 37, 33, 13) |
| alle Grenzen eingehalten (Restverletzung der Strafe) | g ≤ 9,4·10⁻⁵ mm (Glasrand Linse 1), 2,9·10⁻⁵ mm (Strahlhöhe Fläche 3), 8,2·10⁻⁶ mm (Luftspalt), 2,4·10⁻⁵ in cos (Einfallswinkel 26,0031°); Toleranzen `kMechanicalPenaltyLength` = 2·10⁻⁴ mm, `kMechanicalPenaltyCos` = 10⁻⁴; GPU (RTX 3090) dieselben Werte auf drei Stellen (F am Ende relativ 5·10⁻¹¹ zur CPU) |
| Bedingungen aktiv | Luftspalt, Glasrand Linse 1, Strahlhöhe Fläche 3 und Einfallswinkel Fläche 3 liegen auf ihrer Grenze; F steigt von 1,11 (frei) auf 18,5 Wellen² |
| Brennweite | f − f_s = −3,5·10⁻⁴ mm (Toleranz `kMechanicalFocalLength` = 2·10⁻³ mm) |

### 2.6 Augmented Lagrangian für einseitige Bedingungen (M11-T2)

API: `include/mikroptim/optim/augmented_lagrangian.hpp` (`Mikroptim::optim`); Vorschlag und Begründung: [optimierung-erweiterungen.md, Abschnitt 2](optimierung-erweiterungen.md#2-augmented-lagrangian-für-einseitige-bedingungen-m11-t2); Grundsatzentscheidung [ADR-0018](adr/0018-optionale-merit-terme.md). Option, Standard aus: Ohne Verschiebungen rechnen alle Residuen bitgleich wie bisher.

**Problem.** Die Strafe w · max(0, g)² erreicht die Grenze nur asymptotisch, g* ≈ λ/(2w) (Abschnitte 2.3–2.5). Deshalb die Gewichtsstufen bis zu großen w; große w verschlechtern die Kondition von JᵀJ, und sobald g* unter die Änderung von g über die FD-Schrittweite fällt, liegt der Punkt am Knick von max(0, ·) (Abschnitt 2.5, Befund F10).

**Residuum mit Verschiebung.** Jedes einseitige Residuum bekommt eine Verschiebung θ ≥ 0 in der Einheit von g:

$$r = \sqrt{w}\,\max(0,\ g + \theta),\qquad r^2 = w\,\max(0, g + \theta)^2 .$$

Paraxial mit Modus `max` (auch Petzval, g = q bzw. |κ_P| − κ_max) heißt das √w · max(q + θ, 0), mit Modus `min` (g = −q) √w · min(q − θ, 0), das Vorzeichen des Modus bleibt (`paraxial_deviation`, `paraxial_violation`). Mechanisch √w · max(0, g + θ) mit g aus `mechanical_violation`. Für θ = 0 wird nichts addiert (bitgleich zur reinen Strafe). Nicht verschoben werden Gleichungen, die L1-Strafe und die Glas-Hülle: Deren Residuum ist ein Vektor √w (u − P(u)), dessen Richtung innerhalb der Hülle nicht definiert ist (offen, siehe unten).

**Fortschreibung** nach jedem LM-Lauf, mit g* am Endpunkt (Multiplikatormethode nach Hestenes und Powell, Form mit Verschiebung; Nocedal/Wright, Kap. 17):

$$\theta \leftarrow \min\Bigl(\max(0,\ \theta + g^*),\ \frac{\theta_{\max}}{\sqrt{w}}\Bigr),\qquad \lambda = 2 w \theta .$$

Am Fixpunkt ist g* = 0 bei aktiver Bedingung (θ > 0), bei inaktiver θ = 0; die Grenze wird also bei **festem** w exakt erreicht. Der Punkt liegt dabei im glatten Teil (g + θ = θ > 0, weit weg vom Knick bei g = −θ), anders als die Strafe mit großem w. Bei w = 0 ist θ = 0.

- **Schranke** θ_max in Residueneinheiten (`AugmentedLagrangianOptions::max_shift`, Standard `kDefaultMaxShift` = 10³): √w θ ≤ θ_max, der Beitrag der Verschiebung zu F ist höchstens θ_max² Wellen². Sie begrenzt das lineare Wachsen von θ bei einer unerfüllbaren Bedingung.
- **Abbruch** der äußeren Schleife, wenn max √w |Δθ| ≤ T (`tolerance`, Standard 10⁻⁶ Wellen), oder nach K Läufen (`rounds`). Ist eine Bedingung am Endpunkt verletzt (g* > 0), gilt Δθ = g* (ohne Schranke); der Abbruch garantiert also g* ≤ T/√w für alle Bedingungen. Die Toleranz T = √w · δ folgt damit aus der zulässigen Verletzung δ.
- **Wahl von w.** Die Multiplikatoren konvergieren linear mit einem Faktor etwa H/(H + 2w) (H: Krümmung der übrigen Merit entlang der Bedingung). Zu kleines w heißt viele Läufe, zu großes die Steifheit der Strafe; moderat ist ein Gewicht einige Dekaden unter dem der Penalty-Fortsetzung am Ende (Messung unten).
- **Determinismus.** Die Fortschreibung ist elementweise ohne Reduktion; g* kommt aus denselben Rechnungen wie die Residuen (Paraxialtrace, Extrema auf der CPU mit derselben Quadratur und Glättung wie im LM). Zwei Läufe mit gleichen Eingaben liefern bitgleiche θ.

**Zustand.** θ steht pro Eintrag in `ConstraintSpec::shift` (Länge 1 bei paraxialen Bedingungen, sonst die Zahl der Flächen aus `constraint_surfaces`; leer = 0), ist Zustand des Optimierers und **nicht** Teil des Dateiformats (weder gelesen noch geschrieben). Für einen Neustart steht θ im Ergebnis- bzw. Neustartzustand eines Laufs (Entscheidung Maintainer 2026-11-19): JSON von `mikroptim optimize` und Protokoll von `mikroptim synth`, Feld `augmented_lagrangian.shifts` (`item`, `index`, `type`, `shift`), eingelesen mit `--shifts-from` (`shift_values`, `set_shift_values`, `ShiftValue`). Zwei Hälften mit Neustart ergeben bitgleich einen Lauf (`CliOptimize.AugmentedLagrangianRestartEqualsSingleRun`). `make_paraxial_constraints` und `make_mechanical_constraints` übernehmen θ nach `ParaxialConstraint::shift` bzw. `MechanicalConstraint::shift`; passt die Länge nicht zur Zahl der Residuen (etwa nachdem ein Einfügen die Flächen eines Eintrags erweitert hat), gelten alle θ des Eintrags als 0. `check_constraints` lehnt θ < 0, nicht endliche θ und θ an Einträgen ohne `supports_shift` ab. Weil θ zur Beschreibung gehört, gehört es zum Zustand eines Kandidaten; gleiche Topologie im Sinn der Strahlsuche verlangt gleiche θ (die realisierten Beschreibungen werden verglichen).

```cpp
double update_shift(double theta, double violation, double weight, double max_shift);
struct ShiftEntry { int item, index; ConstraintType type; int file_surface;
                    double weight, violation, shift, new_shift; };
struct ShiftUpdate { std::vector<ShiftEntry> entries; double max_change, max_violation; };
ShiftUpdate shift_update(const SystemDescription&, const SystemBatchHost&, int v,
                         const ResidualSettings& quadrature, double max_shift = kDefaultMaxShift);
void apply_shift_update(SystemDescription&, const ShiftUpdate&);   // θ nach ConstraintSpec::shift
bool has_shiftable_constraints(const SystemDescription&);
void clear_shifts(SystemDescription&);
```

**Wo die Schleife sitzt.** `mikroptim optimize --augmented-lagrangian K[,T]` ([cli.md](cli.md#mikroptim-optimize)): K LM-Läufe mit den Gewichten der Datei, Zusatzresiduen pro Lauf mit den aktuellen θ neu gebaut. Homotopie-Controller (`HomotopyOptions::augmented_lagrangian`, `al_penalty_scale`; `synth --augmented-lagrangian K[,T[,S]]`): Die Penalty-Fortsetzung am Start und nach jedem Einfügen wird durch höchstens K Optimierungen mit dem festen Faktor S ersetzt (Schrittart `penalty`, Abbruch über T), und **nach jeder Optimierung** schreibt das Modell θ fort (`HomotopyModel::update_shifts`; optisch an der realisierten Beschreibung mit Gewichten × S und Vorhalt, θ zurück in die Basisbeschreibung). Die stetigen Schritte starten so mit den Multiplikatoren des vorigen Punkts (θ₀ = θ der vorigen Stufe). Die Strahlsuche unterstützt die Option nicht.

**Abnahme** (`tests/test_augmented_lagrangian.cpp`, GPU `tests/test_cuda_augmented_lagrangian.cpp`): dieselben Systeme und Toleranzen wie die Abnahmen von M4-T4 (Petzval) und M4-T3 (Mechanik), eine äußere Schleife mit festem Gewicht (`al_loop` in `tests/augmented_lagrangian_test_systems.hpp`, LM höchstens 1000 Zyklen pro Lauf), T = √w · δ aus der Toleranz δ der Abnahme; Vergleich mit der Penalty-Fortsetzung im selben Test.

| Kriterium | Petzval (w = 1,677·10⁵ mm², Datei · 10⁻⁴) | Mechanik (w = 10⁵/mm², Datei · 10⁻¹) |
|---|---|---|
| Läufe | 2 (378 + 13 Zyklen) | 2 (46 + 10 Zyklen) |
| LM-Zyklen AL / Fortsetzung, CPU | **391 / 546** | **56 / 104** |
| LM-Zyklen AL / Fortsetzung, GPU (RTX 3090) | 390 / 540 | 56 / 119 |
| Restverletzung | (\|κ_P\| − κ_max)/κ_max = 1,5·10⁻⁹ (GPU 1,6·10⁻¹⁰), Toleranz `kPetzvalPenaltyRelative` = 10⁻⁶ | alle g ≤ 2,1·10⁻⁷ mm (GPU ≤ 0), Toleranzen `kMechanicalPenaltyLength` = 2·10⁻⁴ mm, `kMechanicalPenaltyCos` = 10⁻⁴ |
| Brennweite | f − f_s = 4,8·10⁻⁷ mm | f − 10 mm = −3,2·10⁻⁵ mm |
| aktiv am Ende | Petzval (θ = 7,8·10⁻⁷/mm) | Luftspalt, Glasrand Linse 1, Luftrand vor Linse 1 |

Der Mechaniklauf endet in einem anderen lokalen Minimum als die Fortsetzung (F = 18,27 statt 18,5 Wellen²; Einfallswinkel und Strahlhöhe dort nicht aktiv, dafür der Luftrand vor der ersten Linse). Messung der festen Gewichte (CPU, gleiche Schleife):

| Gewicht | Petzval | Mechanik |
|---|---|---|
| kleinstes der Fortsetzung (Datei · 10⁻⁶ bzw. 10⁻⁴) | 3 Läufe, 492 Zyklen; auf der GPU bleibt g bei 1,1·10⁻⁶ relativ stehen (der innere LM endet mit `converged_step`, bevor er die winzige Änderung von θ nachvollzieht), 28 Läufe | nach 30 Läufen noch g = 0,024 mm, 525 Zyklen (Multiplikatoren zu langsam) |
| Datei · 10⁻⁵ bzw. 10⁻³ | 2 Läufe, 414 Zyklen (GPU 425) | nach 30 Läufen g = 1,5·10⁻⁴ mm, 617 Zyklen |
| **Datei · 10⁻⁴ bzw. 10⁻²** | **2 Läufe, 391 Zyklen** (GPU 390) | 6 Läufe, 178 Zyklen |
| Datei · 10⁻³ bzw. **10⁻¹** | erster Lauf vom Start 1000 Zyklen `max_iterations` | **2 Läufe, 56 Zyklen** |

Folgerung: Mit einem Gewicht drei bzw. eine Dekade unter dem der Datei erreicht der Augmented Lagrangian die Grenze genauer als die Fortsetzung (Petzval um zwei Größenordnungen) und mit 28 bzw. 46 % weniger LM-Zyklen; mit dem kleinsten Gewicht der Fortsetzung konvergieren die Multiplikatoren zu langsam.

**Am Abnahmesystem von M6-T3** (`synth einzellinse_homotopie_start.json --target-na 0.6 --max-iterations 500 --curvature-bound 1`, CPU; mechanische Gewichte der Datei 10⁴/mm², Penalty-Fortsetzung 10², 10³, 10⁴; mit AL `--augmented-lagrangian 10,1e-3,S`):

| Lauf | Ziel | RMS [Wellen] | Optimierungen | LM-Zyklen | davon `max_iterations` | Zeit |
|---|---|---|---|---|---|---|
| Penalty-Fortsetzung (Standard) | erreicht, 4 Elemente | 0,0163 | 63 | 21.038 | 35 | 17,9 s |
| AL, S = 1 (10⁴/mm²) | erreicht, 4 Elemente | 0,0141 | 72 | 28.637 | 53 | 22,6 s |
| AL, S = 0,1 (10³/mm²) | erreicht, 4 Elemente | **0,0032** | 73 | 23.461 | 37 | 20,3 s |
| Fortsetzung mit `--stall 20,1e-3` | erreicht, 4 Elemente | 0,0179 | 64 | 4.931 | 4 | 4,8 s |
| AL, S = 1, mit `--stall 20,1e-3` | erreicht, 4 Elemente | 0,0119 | 58 | **1.785** | 0 | 1,9 s |

Allein behebt der Augmented Lagrangian das Kriechen aus F10 nicht: Die LM-Läufe der stetigen Schritte enden weiter mit `max_iterations`. In diesen Schritten wechseln die aktiven Randdicken (Träger des Maximums, Glas- und Luftrand gleichzeitig), und der innere LM muss jeder Änderung von θ folgen. Zusammen mit dem Stagnationsfenster braucht er aber 2,8-mal weniger LM-Zyklen als die Fortsetzung mit demselben Fenster, bei kleinerem RMS und ohne einen Lauf am Iterationslimit; mit S = 0,1 wird das Endsystem deutlich besser (RMS 0,0032 statt 0,0163 Wellen). Die Option bleibt Standard aus; ein Lauf auf der Zielspezifikation (M6-T6) steht aus.

**Offen.** Glas-Hülle (Vektorresiduum, Richtung innen undefiniert; eine Verschiebung auf φ mit Sicherheitsabstand wäre möglich, braucht aber ein inneres g); Strahlsuche (θ pro Kandidat und gemeinsame Gruppen nur bei gleichem θ sind vorbereitet, die Fortschreibung pro Kandidat fehlt).
### 2.7 Sinusbedingung (ab M11-T4)

API: `batch_sine_condition` und Typ `ConstraintType::SineCondition` in `include/mikroptim/optim/paraxial_residuals.hpp`, Formel `conrady_osc` in `include/mikroptim/paraxial/sine_condition.hpp`; Schema: Typ `sine_condition` in [dateiformat.md](dateiformat.md#nebenbedingungen-ab-m3-t3); CLI `--sine-condition w` in `optimize` und `synth` ([cli.md](cli.md)). Begründung (Befund F4 von M6-T6: lineare Koma ist für eine reine Achsoptimierung und für die rotationssymmetrischen Geisterplatten unsichtbar): [optimierung-erweiterungen.md, Abschnitt 4](optimierung-erweiterungen.md#4-sinusbedingung-als-residuum-m11-t4).

Pro Pupillenzone ρ_k (Anteil des Pupillenradius a, Standard ρ = 1/√2 und 1 wie im Aberrationsbericht) ein Residuum

$$r_k = \sqrt{w}\;\mathrm{OSC}(\rho_k),\qquad \mathrm{OSC} = \frac{f_{\mathrm{real}}}{f'}\cdot\frac{l' - l'_P}{L' - l'_P} - 1,\quad f_{\mathrm{real}} = -\frac{\rho_k a}{\sin U'},\quad f' = \frac{n'_L}{\varphi},$$

die Verletzung der Sinusbedingung nach Conrady (Staeble–Lihotzky-Form mit dem Pupillenfaktor, der die Restsphärik herausrechnet) mit derselben Formel wie der Aberrationsbericht ([numerik.md, Abschnitt 14](numerik.md#14-aberrationsbericht-m10-t1)). Für die Variante v verfolgt `batch_sine_condition` einen realen Strahl parallel zur Achse in der Höhe ρ_k a (meridional, φ = 0) bei λ_ref durch die Tabellenflächen der Variante: Brechzahlen `paraxial_index`, Geisterplatten mit `paraxial_ghost_index` und ihrem Dickenprofil, ohne Dezentrierung und Verkippung (wie `batch_paraxial`). Aus Schnittpunkt und Richtung nach der letzten Fläche folgen sin U′ = d_y und L′ = z − y d_z/d_y; f′, l′ und l′_P kommen aus `batch_paraxial` (Geisterplatten als dünne Elemente). Bildseitig telezentrisch entfällt der Pupillenfaktor.

- **Einordnung.** Die Bedingung ist ein gewöhnliches Zusatzresiduum (Block 2, `ParaxialConstraints`, ein Eintrag pro Zone mit derselben Nummer `item`); sie wirkt damit ohne weitere Verdrahtung in `optimize`, `synth`, `insert`, `snap`, in FD- und adjungierter Jacobi (Zusatzzeilen per FD) und auf CPU und GPU (die Zusatzresiduen rechnet immer der Host). Der eigene Strahl kostet pro Variante und Zone einen Trace auf dem Host (gegen den Batch-Trace vernachlässigbar). Der Weg über die Max-Reduktionen des Kernels (trace-abhängige Zusatzresiduen, Abschnitt 2.1) wurde nicht gewählt: Die Gauß-Ringe liegen nicht bei ρ_k, und die Speichen liegen neben dem Meridian.
- **Nicht bestimmbar** (Zonenstrahl versagt hart, auch TIR; Strahl schneidet die Achse nicht; afokales System): Residuum 0. Das Versagen bestrafen die Strahlresiduen.
- **Normierung.** f′ = n′_L/φ ist die bildseitige Brennweite (= −a/u′_L des paraxialen Randstrahls), sodass OSC → 0 für ρ → 0 auch bei einem Bildraum in Glas (vom Maintainer bestätigt, Entscheidung 2026-11-19). Bei Bildraum in Luft ist f′ die Brennweite 1/φ des Berichts von M10-T1b (seit M11-T4 rechnet auch der Bericht mit f′).
- **Gewicht.** OSC ist dimensionslos, w in Wellen². Bei kleiner Apertur ist OSC = −S_II/(2H), die lineare Koma bei der Bildhöhe η′ also W131 = OSC · NA · η′/λ (H = NA η′) mit dem RMS ohne Kippung W131/√72. Soll das Residuum in K Zonen so viel kosten wie die Koma bei η′: w = (NA η′/λ)²/(72 K). Für die Zielspezifikation (NA 0,95, f = 3,29 mm, 546 nm) und η′ bei 10 % Feld (0,025 mm): w ≈ 13 (`osc_weight_for_field` in `tools/m6t6_zielspezifikation.py`).
- **Glattheit.** OSC ist glatt in den Variablen, solange der Zonenstrahl nicht versagt; keine Max-Reduktion, kein Knick.

**Abnahme** (`tests/test_sine_condition.cpp`, `CliOptimize.SineConditionOption`, `CliSynth.SineConditionOption`):

| Prüfung | Ergebnis |
|---|---|
| Analytisches Orakel: zur Blende konzentrische Systeme (Krümmungsmittelpunkte in der Blendenmitte; n p invariant für den Abstand p vom Mittelpunkt, Austrittspupille im Mittelpunkt, daher OSC = (u′/sin U′)(sin U′/u′) − 1 = 0 in jeder Zone und bei jeder Apertur): Meniskus in Luft, große Apertur (H = 0,7 |R₁|), Bildraum in Glas, zwei Gläser; Zonen 0,25, 0,5, 1/√2, 1 | \|OSC\| ≤ 4,4·10⁻¹⁶ (Toleranz `kSineConditionExact` = 10⁻¹⁴), Bericht ebenso; Gegenprobe mit verschobener Blende OSC ≠ 0 |
| gegen den Aberrationsbericht (Rand, Zone): Einzellinse, Mehrlinser, Zweilinser | relativ ≤ 10⁻¹² (`kSineConditionConsistency`) |
| kleine Apertur (a · 0,02) gegen −S_II/(2H), Zone gegen die Hälfte | ≤ 10⁻² relativ (`kOscThirdOrder`) |
| Wirkung: Einzellinse `einzellinse_bk7_biegung.json` nur auf der Achse, w = 10⁶, 50 Zyklen | OSC am Rand 3,19·10⁻⁴ → 1,6·10⁻⁸ (Biegung ohne Koma) |
| Nutzen: Versuch „Apertur zuerst auf der Achse“ von M6-T6 (F4) mit und ohne Residuum | [docs/ergebnisse/m11-t4-sinusbedingung.md](ergebnisse/m11-t4-sinusbedingung.md) |

### 2.8 Verzeichnung (ab M6-T14, ADR-0019)

API: Typ `ConstraintType::Distortion` und `batch_chief_ray_height`; Schema: Typ `distortion` mit Pflichtfeld `max_distortion` in [dateiformat.md](dateiformat.md#nebenbedingungen-ab-m3-t3); CLI `--distortion D_MAX[,W]` in `optimize` und `synth` ([cli.md](cli.md)). Begründung: [ADR-0019](adr/0019-verzeichnung-hauptstrahl.md) (Befund F20 aus M6-T13: am Feldrand begrenzt die Kippung aus Verzeichnung und Querfarbe).

Pro Feld f der Quelle ein Residuum

$$r_f = \sqrt{w}\;\max\big(0,\ |D_f| - D_{\max}\big),\qquad D_f = \frac{\bar y_f - h_f}{h_f},$$

mit der Höhe ȳ_f des realen Hauptstrahls bei λ_ref in der Bildebene (numerik.md §7, „Bildpunkt aus realen Strahlen“; immer der Hauptstrahl, auch bei `reference_point` = `centroid`) und der idealen Höhe h_f = f_soll tan θ_f (`field_target`). Auf der Achse (h_f = 0), bei versagendem Hauptstrahl oder für ein Feld, das der Batch nicht hat, ist das Residuum 0 (q = −D_max; das Versagen bestrafen die Strahlresiduen).

- **Einordnung.** Gewöhnliches Zusatzresiduum im Block der paraxialen Bedingungen (`ParaxialConstraints`, ein Eintrag pro Feld mit derselben Nummer `item`, `ParaxialConstraint::field`), also ohne weitere Verdrahtung in `optimize`, `synth`, `insert`, `snap`, FD- und adjungierter Jacobi (Zusatzzeilen per FD) und auf CPU und GPU. Ein Hauptstrahl pro Feld und Variante auf dem Host.
- **Nur Obergrenze** (`mode` fehlt oder `max`), D_max ≥ 0 (Anteil, 0,01 = 1 %); D_max = 0 ergibt √w |D_f|, im Quadrat glatt. Höchstens ein Eintrag, nur kollimierte Quelle. Keine Verschiebung des Augmented Lagrangian (`supports_shift` falsch: ein Residuum pro Feld).
- **Gewicht.** D ist dimensionslos, w in Wellen²; Standard der CLI 10⁴ (1 % über der Grenze kostet eine Welle², 0,1 % noch 0,01 Wellen² – mehr als die Merit eines beugungsbegrenzten Systems, F ≤ 5,1·10⁻³).
- **Maßstab.** D_f enthält den Maßstabsfehler f/f_soll − 1 (wie die Kippung der Merit mit idealem Bildpunkt); mit der Bedingung `focal_length` bleibt die eigentliche Verzeichnung.
- **Mit Bildpunkt aus realen Strahlen** (`reference_point` = `centroid` oder `chief_ray`) ist das die einzige Stelle, an der die Merit die Verzeichnung sieht; mit idealem Bildpunkt bestraft die Wellenfront sie zusätzlich. Die Querfarbe bleibt in beiden Fällen in der Wellenfront (Kippung relativ zu λ_ref).
- **Glattheit.** Glatt in den Variablen außer am Knick |D_f| = D_max (wie die übrigen einseitigen Bedingungen) und beim Versagen des Hauptstrahls.

**Abnahme** (`tests/test_reference_point.cpp`, `CliOptimize.ReferencePointAndDistortionOptions`): siehe [Bericht M6-T14](ergebnisse/m6-t14-verzeichnung-querfarbe.md#5-tests).

## 3. Warum L1 für Geisterplatten

Mit einer quadratischen Strafe wäre der Grenznutzen bei a = 0 null; der Optimierer würde immer ein wenig Geisterkorrektur verwenden. Mit L1 gilt die Theorie der exakten Penalty: Ist μ größer als der zugehörige Lagrange-Multiplikator, bleiben die Koeffizienten exakt null, solange das reale System das Ziel alleine erreicht. Zusätzlich entsteht Sparsity: wenige, gezielte Koeffizienten statt vieler kleiner. Die Glättung mit ε (Startwert 10⁻⁶ in Einheiten des Dickenprofils) hält LM differenzierbar.

### Festlegungen der Implementierung (M4-T6)

API: `include/mikroptim/optim/ghost_penalty.hpp` (`Mikroptim::optim`), Einstellungen in der Systemdatei als Nebenbedingung `ghost_penalty` ([dateiformat.md](dateiformat.md#nebenbedingungen-ab-m3-t3)).

**Residuum.** Pro Koeffizient a = a_{g,k} jeder bestraften Platte ein Zusatzresiduum

$$r = \sqrt{\mu w_k}\;\frac{a}{\sqrt{\varepsilon + \sqrt{a^2+\varepsilon^2}}},\qquad r^2 = \mu w_k\Bigl(\sqrt{a^2+\varepsilon^2}-\varepsilon\Bigr) = \mu w_k\,\frac{a^2}{\varepsilon+\sqrt{a^2+\varepsilon^2}},\qquad w_k = w_0\,\kappa^k,$$

μ = `weight`, w₀ = `order_weight` (Standard 1), κ = `order_factor` (Standard 2), ε = `smoothing` (Standard 10⁻⁶ mm). Die Einheit von μ w_k ist Wellen²/mm (Merit pro mm Profil); r ist wie alle Residuen dimensionslos. w_k wird als Produkt w₀·κ·κ·… gebildet (plattformunabhängig), r² ohne Auslöschung als μ w_k a²/(ε + s) mit s = hypot(a, ε).

**Warum diese Form statt √(μ w_k)(a² + ε²)^{1/4}.** Beide Quadrate unterscheiden sich nur um die Konstante μ w_k ε, die Gradienten sind gleich (μ w_k a/√(a² + ε²)), also auch die Minima. Für LM zählt aber das Residuum selbst, weil das Gauß-Newton-Modell die Krümmung von F = Σ r² durch 2 JᵀJ ersetzt:

- Plan-Form: r(0) = √(μ w_k ε) ≠ 0 und dr/da(0) = 0. Das Modell sieht bei a = 0 keine Krümmung der Strafe (wahr ist μ w_k/ε), der Gauß-Newton-Schritt in a schießt um den Faktor μ w_k/(2 ε H) über das Minimum hinaus (H = Krümmung der Wellenfront in a, im Abnahmesystem ≈ 10⁶ Wellen²/mm²; Faktor ≈ 10²). LM müsste die Dämpfung für alle Variablen hochsetzen. Außerdem trägt jeder Koeffizient die Konstante μ w_k ε zu F bei.
- Implementierte Form: r ist analytisch und ungerade, r(0) = 0, dr/da(0) = √(μ w_k/(2ε)); 2 (dr/da)² = μ w_k/ε ist bei a = 0 genau die Krümmung von r². Die Strafe ist in a nahezu diagonal und sehr steif; die Normalgleichungen bleiben gut konditioniert (Diagonaldominanz der skalierten Matrix). Für |a| ≫ ε ist r ≈ √(μ w_k |a|) sign a, das Modell überschätzt dort die Krümmung (μ w_k/(4|a|) statt ≈ 0), die Schritte werden nur kürzer.

**Abschätzung der Restkoeffizienten („exakt 0“ mit Glättung).** Am Minimum von F_rest(x, a) + Σ μ w_k(√(a² + ε²) − ε) gilt für jeden Koeffizienten g_k + μ w_k a_k/√(a_k² + ε²) = 0 mit dem Gradienten g_k = ∂F_rest/∂a_k der übrigen Merit (Wellenfront, Nebenbedingungen) am Endpunkt. Mit s_k = g_k/(μ w_k):

$$a_k = -\,\varepsilon\,\frac{s_k}{\sqrt{1-s_k^2}} \;\approx\; -\,\frac{\varepsilon\,g_k}{\mu\,w_k}\quad (|s_k| \ll 1).$$

- **Schwelle:** |s_k| < 1, also μ w_k > |g_k|, heißt „μ oberhalb der Schwelle“; dann bleibt |a_k| im Bereich ε |s_k|, sonst wird der Koeffizient aktiv (|a| ≫ ε, dort gilt die Stationarität mit dem L1-Subgradienten). Exakt 0 erreicht die geglättete Strafe nie, aber |a_k| ≈ ε|g_k|/(μ w_k) ist beliebig klein: Für |a| < 10⁻⁸ mm genügt ε = 10⁻⁷ mm bei μ wenig über der Schwelle oder ε = 10⁻⁶ mm bei μ ≥ 100-fach über der Schwelle. Eine Fortsetzung in ε (ε am Ende verkleinern) leistet dasselbe; sie gehört zum Homotopie-Controller (umgesetzt in M6-T3, Abschnitt 7). Kleinere ε als etwa 10 h (h = FD-Schrittweite bei a = 0, Abschnitt 1) sollte man nicht wählen, sonst überspannt die zentrale Differenz den glatten Bereich.
- **Merit:** Gegenüber dem realen System allein senken die Restkoeffizienten F um Σ_k μ w_k ε (1 − √(1 − s_k²)) ≈ Σ_k ε g_k²/(2 μ w_k) (Minimum von g a + μ w(√(a² + ε²) − ε) über a); bei exakter L1-Strafe wäre dieser Gewinn 0.
- **Kopplung:** g_k ist der Gradient am Endpunkt, nicht bei a = 0 mit festem x: Alle Ordnungen tragen zu t₂ und damit zur Brechkraft der Platte bei (Abschnitt 6 von numerik.md); im Abnahmesystem korrigieren Restkoeffizienten von ~10⁻⁹ mm zusätzlich die Brennweite (Residuum bei a = 0: 2·10⁻⁵), und g_k der Ordnungen 1, 3, 4 sinkt dadurch um bis zu zwei Größenordnungen.

**Beugungsziel als Gewicht.** Ein Restfehler mit RMS σ (Wellen bei λ_ref), dessen Pupillenverlauf genau dem Modus P_k(2u − 1) entspricht (Bündel füllt r_g, gleichmäßig gewichtete Pupille, Merit-Gewichte auf Summe 1), erzeugt bei a = 0 den Gradienten

$$g_k = \frac{2\,\sigma\,(n_G - n_m)}{\lambda_{\mathrm{ref}}\,\sqrt{2k+1}}$$

(`ghost_activation_gradient`; aus g_k = 2 Σ r_i ∂r_i/∂a_k mit ∂OPD/∂a_k = (n_G − n_m) P_k und dem Mittel 1/(2k + 1) von P_k² über ξ ∈ [−1, 1]). Wählt man μ w_k = g_k(σ_Ziel), wird Ordnung k genau dann aktiv, wenn der Restfehler des realen Systems in diesem Modus σ_Ziel übersteigt; an einem aktiven Koeffizienten bleibt im Optimum ein Modusrest der Größe σ_Ziel (0,5/(n_G − 1)) übrig (Stationarität mit |g_k| = μ w_k). Die Testsysteme verwenden σ_Ziel = λ/14 (Maréchal), Δn = 0,5 und k = 2 (sphärische Aberration nach Fokussierung ∝ P_2): μ w_2 = 54,37 Wellen²/mm, μ = 13,6. Der Gesamtfehler nach der Optimierung liegt etwas darüber, weil auch höhere, nicht aktivierte Ordnungen Rest tragen (Einzellinse NA 0,5: 0,0875 Wellen RMS gegen λ/14 = 0,0714).

**Abnahme** (`tests/test_ghost_optimization.cpp`, GPU `tests/test_cuda_ghost_optimization.cpp`, LM mit FD-Jacobi, Quadratur 10 Ringe):

| Kriterium | System | Ergebnis |
|---|---|---|
| Exaktheit: alle \|a\| < 10⁻⁸ mm | `einzellinse_geisterplatten_exakt.json`: Einzellinse f = 100 mm, NA 0,025, Biegung und Arbeitsabstand frei, Brennweite 0,029/mm²; vier Platten (CaF₂ und Flint bei 1 mm, N-BK7 bei 4 mm hinter der Blende, N-BK7 hinter der Linse), K = 4, Start a ≠ 0 (bis 2·10⁻⁵ mm); μ = 13,6, ε = 10⁻⁷ mm | max \|a\| = 2,63·10⁻⁹ mm, 83 Zyklen (converged_step); max \|g_k\|/(μ w_k) = 0,026 (μ etwa 38-fach über der Schwelle); a_k gegen −ε s/√(1 − s²) auf 4,8·10⁻⁷ ε; F_allein − F = 4,034·10⁻⁹ gegen vorhergesagt 4,024·10⁻⁹ Wellen² (F = 1,553·10⁻⁶) |
| Skalierung mit ε und μ | dasselbe mit ε = 10⁻⁶ mm (μ = 13,6) bzw. μ = 100 (ε = 10⁻⁶ mm) | max \|a\| = 2,57·10⁻⁸ bzw. 3,57·10⁻⁹ mm, wie ε/μ vorhersagt |
| Bedarf: Koeffizienten werden aktiv, Bericht nennt Lage, Ordnung, Glas | `einzellinse_na05_geisterplatten.json`: Einzellinse N-BK7 f = 2 mm, NA 0,5 (paraxial a/f), Achse, eine Wellenlänge; neun Platten (CaF₂, N-BK7, Flint an drei Lagen: 0,05 und 0,4 mm hinter der Blende, 0,2 mm hinter der Linse), K = 6; μ aus dem Beugungsziel | RMS allein 2,553 Wellen, mit Geisterplatten 0,0875 Wellen (73 Zyklen); aktiv nur Platte 8 (Flint, nach Fläche 2 bei 0,2 mm), dominante Ordnung 2 (a_2 = 7,77·10⁻³ mm), Merit-Gewinn 24,0 gegen Strafe 0,42 Wellen²; Empfehlung Operator A |

Dass die L1-Strafe bei monochromatischer Korrektur das Glas mit dem größten n − 1 wählt, folgt aus Φ = (n_G − 1) t: Dieselbe Phase kostet mit dem Flint (n − 1 = 0,753) das kleinste Σ|a|. Welche Lage gewinnt, ist bei einem einzigen Achsfeld nahezu entartet (alle Lagen sehen dasselbe Modusprofil); die Tests prüfen deshalb nur, dass der Bericht die Lage der aktiven Platte nennt.

## 4. Gebündelter Levenberg-Marquardt

- Viele unabhängige Läufe (Multistart, Homotopie-Zweige, beide Abstiegsrichtungen an Sattelpunkten) werden **gleichzeitig** iteriert; jeder Lauf hat eigenen Zustand (x, Dämpfung, Status).
- Jacobi-Matrix in Version 1 per **zentralen finiten Differenzen**: Für P Variablen werden 2P (oder P vorwärts) gestörte Varianten und x₀ selbst in denselben Batch gepackt. Schrittweite h_k = ε_m^{1/3} · max(|x_k|, x_typ,k), ε_m^{1/3} ≈ 6·10⁻⁶. Schablonen an Schranken, Ablage und Genauigkeit: [Finite-Differenzen-Jacobi](#finite-differenzen-jacobi-ab-m3-t2).
- JᵀJ und Jᵀr auf dem Gerät mit einem eigenen deterministischen Kernel, Lösung per Cholesky auf dem Host (Plan: `cublasDgemmStridedBatched` und `cusolverDnDpotrfBatched`/`potrsBatched`; Abweichung begründet in [ADR-0010](adr/0010-normalgleichungen-ohne-cublas.md), Status „vorgeschlagen“).
- Marquardt-Skalierung mit diag(JᵀJ), Dämpfungssteuerung nach Nielsen (Gewinnverhältnis).
- Schranken per Projektion auf den zulässigen Bereich mit aktiver Menge.
- Abbruch: relative Abnahme von F < 10⁻¹⁰ (vom ungedämpften Modell vorhergesagt) oder Schrittnorm < Toleranz oder Gradient < Toleranz oder Iterationslimit.
- Version 2 (M5, seit M5-T3 wählbar über `LmOptions::jacobian_method`): Jacobi-Zeilen aus dem adjungierten Kernel; die finite-Differenzen-Variante bleibt als Test-Orakel erhalten ([Jacobi-Verfahren im LM](#jacobi-verfahren-im-lm-ab-m5-t3)).

### Verfahren (ab M3-T4)

Host-API `include/mikroptim/optim/levenberg_marquardt.hpp` (`Mikroptim::optim`), GPU-Backend `include/mikroptim/cuda/levenberg_marquardt.hpp` und Normalgleichungen `include/mikroptim/cuda/normal_equations.hpp` (`Mikroptim::cuda`).

**Zustand pro Lauf:** x, F = ‖r(x)‖² (Strahl-, Straf- und Zusatzresiduen, Abschnitt 2), Zahl versagter und hart versagter Strahlen, Dämpfung μ (Start `initial_damping` = 10⁻³) und ν (Start 2), Status, Zähler. Die Steuerlogik `levenberg_marquardt(backend, P, bounds, x0, options)` ist einmal implementiert; ein **Backend** (`LmBackend`) liefert für eine Liste von Läufen A = JᵀJ und g = Jᵀr an ihren Punkten (`normal_equations`) sowie F und versagte Strahlen (`merit`). Backends: `CpuLmBackend` (CPU, CI), `cuda::GpuLmBackend` (GPU) und in den Tests analytische Residuenfunktionen.

**Zyklus** (alle aktiven Läufe gemeinsam):

1. Für Läufe nach einem angenommenen Schritt (und am Anfang): J und r an x per finiten Differenzen (alle diese Läufe in einem FD-Batch) bzw. adjungiert (M5-T3), A = JᵀJ, g = Jᵀr. Danach Skalierung, Gradientenmaß und Modell-Kriterien (unten).
2. Für jeden aktiven Lauf der Schritt mit aktiver Menge (unten), unabhängig pro Lauf auf Host-Threads.
3. Alle Versuchspunkte gemeinsam bewerten: Modus MERIT der Engine bzw. `merit_cpu`, Zusatzresiduen als `extra_merit` = Σ e² (aufsteigend), trace-abhängige Zusatzresiduen (M4-T3) als Σ t² aus den Extrema desselben Laufs danach, sodass F genau die Quadratsumme des Residuenvektors ist (bis auf die Summationsreihenfolge).
4. Gewinnverhältnis ρ = (F − F_neu)/(L(0) − L(Δx)) mit L(0) − L(Δx) = −(2Δxᵀg + ΔxᵀAΔx) für den tatsächlichen (projizierten) Schritt Δx. Angenommen, wenn F_neu endlich ist, F_neu < F, die Vorhersage positiv ist und (Standard `reject_new_failures`) am Versuchspunkt nicht mehr Strahlen **hart** versagen als bei x – hart versagte Strahlen sind maskiert (r = 0) und könnten F sonst senken. Weich versagte Strahlen (M4-T2) zählen nicht: Ihre Wellenfront- und Strafresiduen sind stetig und stehen in F; eine Ablehnung jedes Schritts, der einen Strahl knapp über eine Grenze schiebt, würde den LM an Versagensgrenzen festhalten. Im harten Modus sind beide Zahlen gleich (Verhalten wie M3-T4).
5. Nielsen: angenommen μ ← μ·max(1/3, 1 − (2ρ − 1)³), ν ← 2; verworfen (auch bei gescheiterter Cholesky-Zerlegung) μ ← μ·ν, ν ← 2ν.

Fertige Läufe werden eingefroren und gehen in keinen weiteren Batch ein.

**Äußeres Strafverfahren (ab M4-T2).** Die optischen Treiber `optimize_lm_cpu`, `cuda::optimize_lm_gpu` und `cuda::optimize_lm_multi_gpu` (und damit `mikroptim optimize`, Optionen `--fail-weight` und `--hard-failure`, [cli.md](cli.md#strahlversagen-ab-m4-t2)) rufen die Steuerlogik über `optimize_lm_rounds` auf. Endet ein Lauf im weichen Modus mit versagten Strahlen (Status außer `NumericalError`), startet er an seinem Endpunkt neu, mit w_fail · `fail_weight_factor` (Standard 10) und frischem Zustand (μ₀, ν = 2, neue Jacobi-Matrix); alle betroffenen Läufe gemeinsam in einem Aufruf, mit ihrer Vorlage (`lm_template_batch`). Das wiederholt sich, bis kein Strahl mehr versagt oder w_fail `max_fail_weight` (10¹⁴) übersteigen würde. Begründung und Messungen: [numerik.md, Abschnitt 11](numerik.md#festlegungen-der-implementierung-m4-t2) (mit festem w_fail endet der LM je nach System in Minima mit versagten Strahlen oder hängt an der Kante zum harten Versagen; mit dem Verfahren enden 108 von 108 bzw. 36 von 36 Starts der Testfamilien versagensfrei). Ergebnis eines Laufs: x, F, Zahlen versagter Strahlen, Status, μ, Gradientenmaß und Modellabnahme der letzten Runde; Zyklen, angenommene Schritte und Jacobi-Auswertungen summiert; `history` aneinandergehängt (jede Runde beginnt mit F am Endpunkt unter dem neuen Gewicht, deshalb springt F dort nach oben, solange Strahlen versagen); `fail_weight` und `penalty_rounds`. `on_iteration` erhält den ursprünglichen Laufindex und fortlaufende Zyklen. Die Entscheidung über einen Neustart hängt nur vom Lauf selbst ab; Gruppe und Einzellauf bleiben bitgleich. `fail_weight_factor` ≤ 1, w_fail = 0 oder hartes Versagen: ein Durchgang wie in M3-T4.

**Marquardt-Skalierung:** (A + μD)δ = −g mit D_kk = max(A_kk, `diag_floor`·max_j(A_jj s_j²)/s_k²), s_k = max(|x_k|, x_typ,k), `diag_floor` = 10⁻¹². Die Untergrenze hält das System regulär, wenn eine Spalte von J verschwindet (Variable ohne Einfluss, alle Strahlen versagt); sie ist in Einheiten der Variablen formuliert (A_jj s_j² ist die Empfindlichkeit pro relativer Änderung) und damit unabhängig von deren Einheiten. Sind alle A_jj = 0, ist D_kk = 1/s_k².

**Schranken mit aktiver Menge:** Variablen auf einer Schranke, deren Abstiegsrichtung −g_k nach außen zeigt (sowie Variablen mit lower = upper), werden für den Schritt festgehalten; das reduzierte System über die freien Variablen wird per Cholesky gelöst. Zeigt danach der Schritt einer freien Variable auf ihrer Schranke nach außen, kommt sie in die aktive Menge und das System wird neu gelöst (höchstens P Runden). Der Versuchspunkt ist clamp(x + δ) (`clamp_to_bounds`), also immer zulässig.

**Abbruch** (Status `LmStatus`):

| Status | Kriterium |
|---|---|
| `ConvergedGradient` | nach neuer Jacobi-Matrix: max_k |g_k|/√(A_kk·F) ≤ `gtol` = 10⁻¹⁰ über die nicht festgehaltenen Variablen (Kosinus zwischen r und Spalte k) |
| `ConvergedMerit` | nach neuer Jacobi-Matrix: das ungedämpfte Modell (Gauß-Newton-Schritt mit aktiver Menge, μ = 10⁻¹² relativ zu D) sagt eine relative Abnahme 0 ≤ (L(0) − L(Δx_GN))/F ≤ `ftol` = 10⁻¹⁰ voraus; oder F = 0 |
| `ConvergedStep` | max_k |Δx_k|/s_k ≤ `xtol` = 10⁻¹⁰ für Δx_GN, oder für den gedämpften Schritt nach mindestens einer Ablehnung seit dem letzten angenommenen Schritt |
| `MaxIterations` | `max_iterations` Zyklen (Versuchsschritte einschließlich verworfener) |
| `DampingLimit` | μ > `max_damping` = 10¹⁶ |
| `NumericalError` | F am Startpunkt oder A, g nicht endlich |

Die Kriterien prüfen bewusst das *ungedämpfte* Modell: Ein schwer gewichteter Residuenanteil (z. B. eine Nebenbedingung) dominiert diag(JᵀJ); in Richtungen, die er nicht sieht, ist der gedämpfte Schritt dann um Größenordnungen kleiner als der Gauß-Newton-Schritt, solange μ nicht klein ist. Ein Kriterium „gedämpfter Schritt klein“ oder „Abnahme des angenommenen Schritts klein“ würde dort vorzeitig abbrechen (beobachtet an der Einzellinse: Abbruch nach 3 Zyklen bei q = 0). Eine negative Modellvorhersage (Rundung bei sehr schlecht konditioniertem JᵀJ) zählt nicht als Konvergenz. Am Optimum endet ein Lauf meist über `ConvergedMerit` oder – wenn die Merit-Änderungen in ihrem Rundungsrauschen liegen (≈ 10⁻⁹ relativ bei OPD um 10⁻⁶ mm, unabhängig von der Skalierung der Residuen) – über verworfene Schritte und `ConvergedStep`.

**Kondition und Gewichte.** Die Normalgleichungen quadrieren die Kondition von J. Werden Nebenbedingungen so hoch gewichtet, dass ihre Spalten die Wellenfrontspalten um mehr als etwa 10⁶ übertreffen, verschwinden die Richtungen, die nur die Wellenfront sieht, in der Rundung von JᵀJ (Einzellinse mit Brennweiten-Gewicht 100/mm² gegen OPD-Residuen von 10⁻⁶ mm: Verhältnis der Diagonalen ~10¹⁸, die Biegung bleibt bei q = 0). Mit Gewicht 10⁻⁸/mm² hält die Bedingung f trotzdem auf 1,6·10⁻⁶ mm; die Konvergenz wird mit wachsendem Gewicht langsamer (10⁻⁸: 54 Zyklen, 10⁻⁶: 132, 10⁻⁴: > 400). Diese Zahlen gelten für die Residuen in mm von M3; seit M4-T1 sind die Residuen in Wellen, dieselben Verhältnisse ergeben sich mit den Gewichten × 1/(F·L·λ_ref²) (10⁻⁸/mm² ↔ 0,029/mm² bei der Einzellinse). Faustregel für die Gewichte: Abschnitt 2.2.

**Ergebnis** `LmRunResult`: x, F, versagte und hart versagte Strahlen, Status, Zyklen (`iterations`), angenommene Schritte, Jacobi-Auswertungen, μ am Ende, Gradientenmaß und Modellabnahme der letzten Jacobi-Matrix, Verlauf `history` (F(x₀) und F nach jedem Zyklus), w_fail und Runden des Strafverfahrens (M4-T2). Der Rückruf `on_iteration` erhält nach jedem Zyklus pro Lauf (in Laufreihenfolge) Zyklus, Annahme, F, F am Versuchspunkt, ρ, μ, Schrittnorm, versagte und hart versagte Strahlen und x (Iterationsprotokoll von `mikroptim optimize`, M3-T5, [cli.md](cli.md#mikroptim-optimize)).

**Vorlagen:** `base` hat eine Variante für alle Läufe oder eine pro Lauf (z. B. Homotopie-Zweige mit verschiedener Apertur); `lm_template_batch` und `lm_point_batch` wählen die Vorlagen der Läufe eines Aufrufs.

**Determinismus und Bitgleichheit** (ADR-0007, Abnahme 3). Das Ergebnis eines Laufs hängt nur von seinem Startpunkt und seiner Vorlage ab, nicht von den anderen Läufen, ihrer Zahl, Reihenfolge oder der Aufteilung in Tranchen und Teilstücke:

- FD-Jacobi: Varianten unabhängig, Differenzkernel ohne Reduktion (M3-T2); Merit-Kernel pro Variante unabhängig von der Tranche (M2-T3); Zusatzresiduen pro Variante.
- Normalgleichungen: jedes Element von G = [J r]ᵀ[J r] ist eine FMA-Kette acc ← fma(u_r, v_r, acc) über die Zeilen in aufsteigender Reihenfolge ab acc = 0, auf dem Gerät (`gram_kernel`) wie auf dem Host (`normal_equations_host`, `std::fma`) – beide bitgleich (Test `CudaNormalEquations.BitwiseEqualToHost`).
- Schritt, Nielsen-Update und Kriterien: pro Lauf auf dem Host in fester Reihenfolge; Host-Threads arbeiten nur auf verschiedenen Läufen bzw. Varianten.

`cublasDgemmStridedBatched` erfüllt das nicht: Auf der RTX 3090 (CUDA 12.9, `CUBLAS_ATOMICS_NOT_ALLOWED`) unterscheidet sich JᵀJ einer Matrix im Aufruf mit batchCount = 1 von derselben Matrix im Aufruf mit batchCount ≥ 2 in fast allen Elementen (R = 5.000, P = 14: 194 von 196; R = 20.000, P = 101: 10.069 von 10.201; nur R = 683, P = 14 zufällig gleich); bei batchCount 2 … 1.000 war das Ergebnis untereinander gleich. Da die Zahl der aktiven Läufe sich während der Optimierung ändert (bis hinunter zu einem), wäre ein Lauf dann nicht bitgleich zum Einzellauf. Details: ADR-0010.

**GPU-Backend.** `normal_equations` teilt die Läufe in Teilstücke, deren Jacobi-Matrizen gemeinsam höchstens die Hälfte des nutzbaren Gerätespeichers belegen (`cudaMemGetInfo` plus vorhandener J-Puffer, höchstens `EngineOptions::memory_limit_bytes`, abzüglich der Reserve der Engine); pro Teilstück `DeviceFdJacobian::compute`, `DeviceNormalEquations::compute` (Kernel: ein Block mit 16 × 16 Threads pro Lauf und Kachelpaar tp ≤ tq der (P + 1) × (P + 1)-Gram-Matrix, Zeilen in Kacheln von 16 im Shared Memory, 2 · 16 · 17 · 8 Byte) und Download von A und g ((P + 1)² Werte pro Lauf). Die Cholesky-Zerlegung (≈ P³/6 FMA pro Lauf und Runde) läuft auf dem Host, gemeinsam mit dem CPU-Pfad. `merit` baut einen Batch mit einer Variante pro Lauf und ruft die Engine im Modus MERIT. Statistik in `cuda::LmGpuStats`. Host-Threads (`threads`, `extra_threads` ≤ 0) werden nach der Arbeit des Aufrufs gewählt (`lm_host_threads`): Ein Thread kostet unter Windows beim Start bis zu einige Millisekunden, kleine Aufrufe laufen daher im aufrufenden Thread (für die Einzellinse sank die Zeit pro Zyklus dadurch von ≈ 30–200 ms auf ≈ 0,6 ms).

**Gemessen** (RTX 3090, FP64; Tests in `tests/test_levenberg_marquardt.cpp` und `tests/test_cuda_levenberg_marquardt.cpp`):

| Aufgabe | Ergebnis |
|---|---|
| Einzellinse N-BK7, f = 100 mm, NA ≈ 0,025, Variablen c₁, c₂, w, Brennweite als Gleichung (Gewicht 10⁻⁸/mm² mit Residuen in mm, M3-T4) | q = 0,7208 (CPU 0,720803, GPU 0,720821; Dünnlinse 2(n² − 1)/(n + 2) = 0,7397; der Unterschied ist der Dickeneffekt der 6 mm dicken Linse), 54 bzw. 53 Zyklen |
| dieselbe Aufgabe mit Residuen in Wellen (M4-T1), Gewicht 0,029/mm² | q = 0,7208 (CPU 0,720808, GPU 0,720789), 55 bzw. 56 Zyklen, f − 100 mm = 1,6·10⁻⁶ mm |
| Mehrlinser (13 Variablen, 7 am Ende auf Schranken), Rückkehr aus 1 % Störung (4 Vorzeichenmuster) | F_end/F_opt − 1 = −4,1·10⁻¹¹ … −2,0·10⁻¹¹ (CPU und GPU), 41–73 Zyklen; seit M4-T1 (Gewicht 0,28/mm²) −1,1·10⁻¹² … +1,7·10⁻¹¹ (CPU) bzw. −5,5·10⁻¹² … +6,7·10⁻¹¹ (GPU), 41–73 Zyklen |
| 1.000 Läufe Einzellinse (±5 %) gemeinsam gegen einzeln | bitgleich (x, F, Verlauf, Zähler, Status), auch mit 64 KiB Speichergrenze (5.898 Teilstücke, Merit in Tranchen); gemeinsam 0,3–0,5 s, einzeln 37–57 s |
| Durchsatz 1.000 Läufe Mehrlinser (27 FD-Varianten pro Lauf, 681 Zeilen) | ≈ 18 Zyklen der Schleife/s, ≈ 1,8·10⁴ Lauf-Zyklen/s |

Die Einzellinse ist für Gauß-Newton schwierig: Der Residuenvektor ist nahezu W₀₄₀(q)·φ(ρ) mit einem festen Pupillenprofil φ, sein Minimum über q ist nicht null, und am Optimum verschwindet ∂r/∂q – die Gauß-Newton-Krümmung in Biegerichtung geht gegen null (großer Restfehler). LM läuft dort mit ρ ≈ 0,7 und kleinen Schritten, konvergiert aber verlässlich.

API (Auszug):

```cpp
enum class LmStatus { Running, ConvergedMerit, ConvergedStep, ConvergedGradient, MaxIterations,
                      DampingLimit, NumericalError };
struct LmOptions { int max_iterations = 100; double ftol = 1e-10, xtol = 1e-10, gtol = 1e-10;
                   double initial_damping = 1e-3, max_damping = 1e16, diag_floor = 1e-12;
                   bool reject_new_failures = true; int threads = 0;   // vergleicht hart versagte
                   FdJacobianOptions jacobian;          // Quadratur, Strahlversagen, Schema, Zusatzresiduen
                   JacobianMethod jacobian_method;      // FiniteDifferences (Standard) | Adjoint, M5-T3
                   double fail_weight_factor = 10.0, max_fail_weight = 1e14;   // Strafverfahren (M4-T2)
                   LmIterationFn on_iteration; };
struct LmRunResult { std::vector<double> x; double merit; int64_t failed_rays, hard_failed_rays;
                     LmStatus status; int iterations, accepted, jacobians; double damping,
                     gradient_norm, model_decrease; std::vector<double> history;
                     double fail_weight; int penalty_rounds; };              // M4-T2
struct LmResult { std::vector<LmRunResult> runs; int cycles; };
class LmBackend { virtual void normal_equations(span<const int> runs, span<const double> x,
                                                span<double> a, span<double> g) = 0;
                  virtual void merit(span<const int> runs, span<const double> x,
                                     span<double> merit, span<int64_t> failed,
                                     span<int64_t> hard_failed) = 0; };     // hard_failed: M4-T2
LmResult levenberg_marquardt(LmBackend&, int P, const LmBounds&, span<const double> x0, const LmOptions&);
using LmBackendFactory = std::function<std::unique_ptr<LmBackend>(const LmOptions&, const SystemBatchHost&)>;
LmResult optimize_lm_rounds(const LmBackendFactory&, const VariableSet&, const SystemBatchHost& base,
                            span<const double> x0, const LmOptions&);   // Strafverfahren (M4-T2)
LmResult optimize_lm_cpu(const VariableSet&, const SystemBatchHost& base, span<const double> x0,
                         const LmOptions& = {});
// cuda/levenberg_marquardt.hpp
LmResult cuda::optimize_lm_gpu(TraceEngine&, const VariableSet&, const SystemBatchHost& base,
                               span<const double> x0, const LmOptions& = {}, LmGpuStats* = nullptr);
// optim/normal_equations.hpp, cuda/normal_equations.hpp
void normal_equations_host(span<const double> J, span<const double> r, int P, span<double> a, span<double> g);
class cuda::DeviceNormalEquations { void compute(const DeviceFdJacobian&);
                                    void compute(const DeviceAdjointJacobian&);  // M5-T3
                                    void compute(const DeviceJacobianRef&);       // M5-T3
                                    void download(std::vector<double>& a, std::vector<double>& g) const; };
```

Rechentyp: FP64 (die Abnahme). df64 für Jacobi und Merit ist im LM nicht angebunden (`FdJacobianOptions` hat keinen Rechentyp); `DeviceFdJacobian::compute` (seit M5-T3) und `DeviceAdjointJacobian::compute` nehmen einen `Precision`-Parameter (Benchmark `jacobian_throughput`).

### Variantengenerator (ab M3-T1)

`include/mikroptim/optim/variant_generator.hpp`. Alle Varianten sind Kopien einer Vorlagenvariante (`base`, `base_variant`), nur die Variablen unterscheiden sich. `write_*` schreibt an eine beliebige Stelle `first` eines vorhandenen Batches (mehrere Läufe in einem Batch, gebündelter LM; Vorlage und Ziel dürfen derselbe Batch sein), `make_*` baut einen neuen Batch mit den gemeinsamen Feldern der Vorlage. x₀ muss endlich und zulässig sein.

**Muster der finiten Differenzen** (`fd_pattern`, `write_fd_variants`, `make_fd_batch`):

- Schrittweite h_k = ε_m^{1/3}·max(|x₀_k|, x_typ,k) mit ε_m^{1/3} = 2^{−52/3} = `kFdStepFactor` = 6,0554544523933395·10⁻⁶ (Konstante, nicht `cbrt` zur Laufzeit).
- Zentral (`FdScheme::Central`, 2P + 1 Varianten): Variante 0 = x₀, Variante 2k + 1 = x₀ + d⁺_k e_k, Variante 2k + 2 = x₀ + d⁻_k e_k. Vorwärts (`FdScheme::Forward`, P + 1 Varianten): Variante 0 = x₀, Variante k + 1 = x₀ + d⁺_k e_k.
- Schranken (endgültig seit M3-T2, Abschnitt [Finite-Differenzen-Jacobi](#finite-differenzen-jacobi-ab-m3-t2)): zentral x₀ ± h, wenn beide zulässig sind; sonst einseitig x₀ + σh und x₀ + 2σh zur Seite σ mit mehr Platz; nur wenn auch das nicht passt (Bereich kürzer als etwa 3h), gekürzt bis an die Schranken. Vorwärts: x₀ + h, sonst der Rückwärtsschritt x₀ − h, sonst die Schranke mit dem größeren Abstand.
- Rückgabe `FdPattern`: nominelles h_k, die geschriebenen Punkte x⁺_k, x⁻_k (Varianten 2k + 1 und 2k + 2), die tatsächlichen Schritte d = fl(x − x₀) (exakt für |x₀| ≥ x_typ und x₀ = 0, Sterbenz-Lemma; sonst auf ½ ulp) und die Schablone `stencil` pro Variable (`FdStencil`, `fd_formula.hpp`). Bei der einseitigen Schablone liegen beide Punkte auf derselben Seite; „plus“ und „minus“ benennen dann nur die erste und zweite Variante.

**Zufällig gestörte Varianten** (`perturbed_vector`, `write_perturbed_variants`, `make_perturbed_batch`), deterministisch und plattformunabhängig:

- Variante j (Zählung im Aufruf ab 0) hat einen eigenen `std::mt19937_64` (im Standard bitgenau festgelegt) mit Startwert splitmix64(seed + j·0x9E3779B97F4A7C15 mod 2⁶⁴) (SplitMix64-Mischfunktion). Variante j hängt also nicht von der Zahl der übrigen Varianten ab.
- Die Verteilungen der Standardbibliothek sind nicht plattformgleich festgelegt; Box-Muller bräuchte log, cos und sin, deren Rundung zwischen Bibliotheken um ein ulp abweichen darf. Daher eigene Umwandlungen nur mit exakten bzw. korrekt gerundeten Operationen: u = (bits ≫ 11)·2⁻⁵³ ∈ [0, 1); gleichverteilt ξ = 2u − 1 ∈ [−1, 1) (`Distribution::Uniform`, eine Ziehung pro Variable) oder Irwin-Hall ξ = Σ_{i=1}^{12} u_i − 6 (`Distribution::IrwinHall`: Mittel 0, Varianz 1, begrenzt auf (−6, 6), nahe der Normalverteilung; 12 Ziehungen pro Variable). Ziehungen in der Reihenfolge k = 0 … P − 1.
- Störung pro Variablentyp: s_k = `absolute` + `relative`·|x₀_k| (`PerturbationScale`, Standard 1 % relativ), y_k = x₀_k + s_k·ξ, beide Ausdrücke mit `std::fma` (unabhängig von der Kontraktion des Compilers), danach Projektion auf die Schranken. `first_is_base`: Variante 0 = x₀.
- Erwartungswerte für Startwert 42 stehen im Test `VariantGenerator.PerturbedValuesArePlatformIndependent` (berechnet mit einer unabhängigen Python-Implementierung); sie gelten unverändert unter MSVC und GCC.
- Glasvariablen (ab M6-T1) werden wie die übrigen relativ gestört (`PerturbationOptions::glass`); `apply` rechnet danach die Brechzahlen der Variante neu. Alle übrigen Brechzahlen bleiben unverändert.

### Finite-Differenzen-Jacobi (ab M3-T2)

Host-API `include/mikroptim/optim/fd_jacobian.hpp` (`Mikroptim::optim`), Differenzenformeln header-only in `include/mikroptim/optim/fd_formula.hpp` (`fd_derivative`, `MKO_HD`, dieselbe Funktion auf CPU und GPU), GPU-Pfad `include/mikroptim/cuda/fd_jacobian.hpp` (`Mikroptim::cuda`).

**Residuenvektor einer Variante.** R = R_v + E Zeilen: die R_v = `ResidualLayout::trace_rows` Zeilen des Traces (Strahlresiduen, im weichen Modus seit M4-T2 dahinter ebenso viele Strafresiduen), danach die E Zusatzresiduen aus `ExtraResiduals` (Abschnitt [2.1](#21-zusatzresiduen-schnittstelle-ab-m3-t3), z. B. die paraxialen Nebenbedingungen `make_paraxial_residuals`, später Strafterme). Das ist dieselbe Ablage wie im Ausgabepuffer der TraceEngine mit `TraceSettings::extra_residuals = E`. Die Zusatzresiduen werden auf dem Host für jede Variante des FD-Batches ausgewertet (`evaluate_extra_residuals` aus M3-T3 auf `FdJacobianOptions::extra_threads` Host-Threads, bitgleich für jede Threadzahl, Ablage [V·E]). Trace-abhängige Zusatzresiduen (`FdJacobianOptions::trace_extra`, ab M4-T3, Abschnitt 2.1) folgen dahinter (E = `extra_count()`); jede FD-Variante bekommt sie aus den Extrema ihres eigenen Traces (`fd_extra_residuals`).

**Schablonen.** Die Schrittweite bleibt h_k = ε_m^{1/3}·max(|x₀_k|, x_typ,k). Zu jeder Variable gehören zwei gestörte Punkte a und b (Varianten 2k + 1 und 2k + 2 des Laufs) mit den tatsächlichen Schritten d_a = fl(x_a − x₀), d_b = fl(x_b − x₀). `fd_pattern` wählt die Schablone (`FdStencil`) nach den Schranken:

| Schablone | Bedingung | Punkte a, b | Ableitung J = ∂r/∂x_k | Abbruchfehler |
|---|---|---|---|---|
| `Central` | x₀ − h ≥ lower und x₀ + h ≤ upper | x₀ + h, x₀ − h | (r_a − r_b)/(d_a − d_b) | h²/6·r‴ |
| `OneSided` | sonst, wenn x₀ + 2σh zulässig ist; σ = +1, wenn upper − x₀ ≥ x₀ − lower, sonst −1 | x₀ + σh, x₀ + 2σh | (d_b²(r_a − r₀) − d_a²(r_b − r₀)) / (d_a·d_b·(d_b − d_a)) | −d_a·d_b/6·r‴ = −h²/3·r‴ |
| `Truncated` | sonst (Bereich der Variable kürzer als etwa 3h) | min(x₀ + h, upper), max(x₀ − h, lower) | (r_a − r_b)/(d_a − d_b) | (d_a + d_b)/2·r″ |
| `Forward` | Vorwärtsschema (P + 1 Varianten) | x₀ + d⁺ (siehe oben) | (r_a − r₀)/d_a | d_a/2·r″ |

Begründung: Eine Variable auf einer aktiven Schranke ist im LM mit aktiver Menge der Normalfall (z. B. Mindestdicke). Die einseitige Dreipunktformel ist dort wie die zentrale von zweiter Ordnung und kostet keine zusätzliche Variante: Der Punkt, der bei gekürzter zentraler Differenz auf der Schranke (also auf x₀) läge, wird für x₀ + 2σh verwendet. Gekürzte unsymmetrische Schablonen (d_a ≪ |d_b|) wären schlecht konditioniert; sie bleiben nur für fast feste Variablen (Bereich < 3h), dort mit erster Ordnung. Liegt x₀ dabei auf einer Schranke, ist ein Schritt 0, der Punkt gleich x₀ (bitgleiche Residuen) und die Formel die einseitige Differenz (r_a − r₀)/d_a. Im zentralen Fall ist d_a + d_b wegen der Rundung von x₀ ± h nicht genau 0; der dadurch entstehende Term (d_a + d_b)/2·r″ liegt bei ulp(x₀)·|r″| und ist vernachlässigbar.

**Rundung.** Jedes Strahlresiduum r = √W·OPD trägt einen Rundungsfehler der Größe ε·L·√W, wobei L die optische Länge ist (die OPL vor dem Kolbenabzug bestimmt die Rundung, nicht |OPD|). Die Gewichte der Formeln sind 1/h (zentral, zweimal 1/(2h)), 2/h, 1/(2h) und 3/(2h) (einseitig, zusammen 4/h) bzw. 2/h (vorwärts). Der Rundungsanteil des relativen Spaltenfehlers ist damit etwa √S·ε·L·√(F·L_λ)·Σ|c|/‖J_p‖₂ (Test-Schätzung `column_rounding_estimate`, S Flächen, F·L_λ Tupel). Spalten mit schwacher Empfindlichkeit sind deshalb rundungsbegrenzt: Die Dicke einer Linse im kollimierten Strahl ändert die OPL fast gleichmäßig, nach dem Kolbenabzug bleibt wenig übrig (Einzellinse: ‖J_t‖ = 1,5·10⁻⁵ gegen ‖J_c‖ = 0,47). Die ε^{1/3}-Regel unterstellt einen Rundungsfehler ≈ ε·|r|; für OPD-Residuen ist er größer, und ein größerer Faktor würde solche Spalten genauer machen (gemessen: mit dem Faktor 10⁻⁴ statt 6·10⁻⁶ liegen alle Spalten beider Testsysteme bei höchstens 1,5·10⁻⁷). Die Schrittweite bleibt bis zu einer Entscheidung wie oben festgelegt.

**Genauigkeit** (Tests `FdJacobian.MatchesRichardson*`, `FdJacobian.BoundStencils`, Referenz: Richardson-Extrapolation zweistufig aus zentralen Differenzen mit H = 10⁻³·max(|x|, x_typ), H/2, H/4, R₂ = (64 D(H/4) − 20 D(H/2) + D(H))/45, Fehler O(H⁶), Konvergenz R₁ gegen R₂ ≤ 4,9·10⁻⁹). Maß der Abnahme ist die spaltenskalierte Frobeniusnorm ‖(J − D)S‖_F/‖DS‖_F mit S = diag(max(|x₀_k|, x_typ,k)): Jede Spalte wird in die Einheit des Residuums pro relativer Änderung der Variable gebracht, die Norm ist unabhängig von den Einheiten der Variablen (die unskalierte Frobeniusnorm wird von den Krümmungsspalten in mm² dominiert) und gewichtet schwach empfindliche, rundungsbegrenzte Spalten nach ihrem Beitrag zum linearen Modell r(x₀ + Sδ) ≈ r₀ + JSδ. Der relative Fehler pro Spalte ‖J_p − D_p‖₂/‖D_p‖₂ wird zusätzlich gegen 10⁻⁶ plus den Rundungsanteil geprüft.

| System | skaliert | unskaliert | größter Spaltenfehler |
|---|---|---|---|
| Mehrlinser, 13 Variablen, 680 Residuen, zentral | 2,1·10⁻⁸ | 3,5·10⁻⁸ | 1,7·10⁻⁶ (t[2], Rundungsschätzung 9,3·10⁻⁶); Krümmungen ≤ 5,6·10⁻⁸ |
| Einzellinse, 4 Variablen, zentral | 3,6·10⁻⁸ | 3,9·10⁻⁸ | 2,4·10⁻⁶ (t[1], Rundungsschätzung 8,1·10⁻⁵); c, w ≤ 4,6·10⁻⁸ |
| Einzellinse an Schranken: einseitig c[1], t[1], gekürzt c[2], zentral w | 9,2·10⁻⁸ | 1,4·10⁻⁷ | c[1] 1,9·10⁻⁷, c[2] 2,8·10⁻⁸, t[1] 4,4·10⁻⁶, w 2,5·10⁻⁸ |
| Mehrlinser, vorwärts | – | – | 6,0·10⁻⁶ (w), erste Ordnung |
| Einzellinse + paraxiale Zusatzresiduen (`einzellinse_bk7_nebenbedingungen.json`, 10 + 3 Zeilen) | 2,5·10⁻¹¹ | 2,5·10⁻¹¹ | (Spaltennormen von der Brennweitenzeile mit Gewicht 100 bestimmt) |
| Mehrlinser + Brennweite, Arbeitsabstand, paraxialer Fokus (680 + 3 Zeilen) | 3,8·10⁻¹⁰ | 5,0·10⁻¹⁰ | paraxiale Zeilen allein ≤ 9,6·10⁻¹¹ (skaliert, je Zeile) |

**Ablage** (`FdJacobian`, ebenso im Gerätespeicher): pro Lauf i eine R × P-Matrix spaltenweise (Spalte p zusammenhängend), Lauf-Index außen:

J_i(r, p) = `jacobian`[(i·P + p)·R + r],  r₀_i(r) = `residuals`[i·R + r],  Status-Wechsel `status_changes`[i·P + p].

Das ist das Spaltenformat mit führender Dimension R und Abstand R·P zwischen den Läufen, wie `cublasDgemmStridedBatched` es für JᵀJ (P × P, Abstand P²) und Jᵀr (Abstand R für r₀) in M3-T4 braucht. Mehrere Läufe liegen in einem FD-Batch hintereinander: Lauf i belegt die Varianten [i·n, (i + 1)·n), n = 2P + 1 (vorwärts P + 1), Variante i·n ist sein x₀ (`make_fd_jacobian_batch`; die Vorlage hat eine Variante für alle Läufe oder eine pro Lauf).

**Versagte Strahlen.** Im harten Modus haben versagte Strahlen r = 0 (M2); an einer Versagensgrenze springt das Residuum, die Jacobi wird dort unbrauchbar. Im weichen Modus (Standard seit M4-T2) sind Wellenfront- und Strafresiduen an den Grenzen stetig, nur ihre Ableitung springt (Knick bzw. Wurzelsingularität auf der gültigen Seite, numerik.md, Abschnitt 11); FD-Spalten über eine Grenze hinweg mitteln dann die einseitigen Steigungen. `status_changes` zählt pro Lauf und Spalte die Strahlen, deren Gültigkeit (Status `Ok` oder nicht) sich zwischen x₀ und einem der beiden gestörten Punkte ändert (Diagnose).

**Pfade.**

- CPU (`fd_jacobian_cpu`, CI ohne GPU und Referenz): `make_fd_jacobian_batch` → `residuals_cpu` für alle Varianten → `evaluate_extra_residuals` → `fd_jacobian_from_residuals` (elementweise `fd_derivative`).
- GPU (`mko::cuda::DeviceFdJacobian::compute`, Bequemlichkeit `fd_jacobian_gpu`): Batch, Zusatzresiduen und Schritte auf dem Host; ein `cudaMalloc` für J, r₀, Status-Wechsel, Zusatzresiduen [V·E], Schablonen und Schritte, angelegt vor dem Lauf der Engine (die ihn so als belegt sieht); dann `TraceEngine::run_on_device` mit Tranchen aus ganzen Läufen (Granularität n) und pro Tranche der Kernel `fd_difference_kernel` direkt auf dem Ausgabepuffer der Engine. J und r₀ bleiben auf dem Gerät (`device_jacobian`, `device_residuals`) und werden nur auf Anforderung heruntergeladen. Die Zusatzresiduen gehen nicht durch den Ausgabepuffer der Engine (`extra_residuals = 0` dort), sondern liegen in einem eigenen Gerätepuffer; die Zeilenablage von J und r₀ ist trotzdem die oben beschriebene.
- Differenzen auf dem Host aus einem Download der Engine (`engine.run` mit `extra_residuals = E`, dann `fd_jacobian_from_residuals`): bitgleich zum Gerätepfad für zentrale und gekürzte Spalten ((r_a − r_b)/(d_a − d_b) lässt keine FMA-Kontraktion zu), einseitige innerhalb der Rundung der Formel (Test `CudaFdJacobian.DevicePathMatchesHostDifferences`).

GPU gegen CPU (Test `CudaFdJacobian.MatchesCpu`): Die Formel ist linear in den Residuen, J = Σ_x c_x·r_x, also |J_GPU − J_CPU| ≤ Σ_x |c_x|·δ(r_x) mit der Residuen-Toleranz δ(r) = max(u·`kCpuGpuFp64`, `kCpuGpuFp64Relative`·|r|) aus M2-T2 (die Toleranz der OPD, seit M4-T1 mit u = s_{f,l}√W der Zeile in Wellen umgerechnet; 0 für Zusatzresiduen) plus 8ε·Σ|c_x·r_x| für die Rundung der Formel. Gemessen höchstens 2,9·10⁻³ dieser Schranke, relativ pro Spalte 5,9·10⁻⁸. Mehrere Läufe in einem Batch, Tranchen und wiederholte Rechnungen sind bitgleich (ADR-0007; keine Gleitkomma-Reduktion im Differenzkernel, die Status-Wechsel sind eine ganzzahlige Baumreduktion). Laufzeit (RTX 3090, Test `CudaFdJacobian.LargeBatch`, kein Benchmark): 1.000 Läufe des Mehrlinsers (27.000 Varianten, 680 Strahlen, J = 70,7 MB) in 85–125 ms für `compute` einschließlich Batchaufbau auf dem Host.

### Adjungierte Jacobi (ab M5-T2)

Dieselbe Ablage und Form (`JacobianShape`) wie die FD-Jacobi, aber die Zeilen des Traces (Strahl- und Strafresiduen) aus dem Rückwärtsdurchlauf pro Strahl mit dem Kolbenabzug als Projektion: `adjoint_jacobian_cpu` (`include/mikroptim/optim/adjoint_jacobian.hpp`), GPU `cuda::DeviceAdjointJacobian` / `adjoint_jacobian_gpu` (`include/mikroptim/cuda/adjoint_jacobian.hpp`, gleiche Signatur von `compute` wie `DeviceFdJacobian`). Die Zusatzzeilen bleiben zentrale finite Differenzen mit denselben Mustern (hybrider Ansatz, bitgleich zu den Zeilen der FD-Jacobi); trace-abhängige Zusatzresiduen kosten dafür weiter 2P + 1 Traces pro Lauf. Es gibt keine Schrittweite und keine Schablonen an Schranken: Die Ableitung ist an jedem Punkt exakt bis auf Rundung, auch auf einer aktiven Schranke. Formeln und Messungen: [numerik.md, „Rückwärtsdurchlauf (M5-T2)“](numerik.md#rückwärtsdurchlauf-m5-t2); Einbindung in den LM: [unten](#jacobi-verfahren-im-lm-ab-m5-t3).

### Jacobi-Verfahren im LM (ab M5-T3)

`LmOptions::jacobian_method` wählt die Jacobi-Matrix der optischen Backends: `JacobianMethod::FiniteDifferences` (Standard, `"fd"`) oder `JacobianMethod::Adjoint` (`"adjoint"`); `to_string`, `jacobian_method_from_string`, CLI `--jacobian fd|adjoint` bei `optimize`, `snap` und `synth` ([cli.md](cli.md#mikroptim-optimize)). Die finiten Differenzen bleiben dauerhaft als Orakel (ADR-0009). Mit `Adjoint` gelten aus `LmOptions::jacobian` die Quadratur, das Strahlversagen, die Zusatzresiduen und `extra_threads` (`adjoint_options_of`); `scheme` gilt nur für die FD-Jacobi, die Zusatzzeilen sind immer zentral. Ein unbekannter Wert wirft `std::invalid_argument` („unbekanntes Verfahren der Jacobi-Matrix“); einen stillen Rückfall auf FD gibt es nicht. Konfigurationen, die der adjungierte Pfad nicht rechnen kann (mit Glasvariablen eine Wellenlänge ≤ 146 nm, Geisterdaten über dem Shared Memory), werfen wie der FD-Pfad mit klarer Meldung; alle Variablentypen (Krümmung, Dicke auch im geteilten Spalt, Arbeitsabstand, Geisterkoeffizient, Glas) und alle Flächenzahlen bis `kMaxSurfaces` sind unterstützt.

**Backends.**

- `CpuLmBackend`: `adjoint_jacobian_cpu` für alle Läufe des Aufrufs, Trace-Zeilen auf Host-Threads (Paare aus Lauf und Feld, `AdjointJacobianOptions::threads`, nach `lm_host_threads` mit `kLmRaySurfacesPerThread`), dann `normal_equations_host` pro Lauf.
- `cuda::GpuLmBackend`: Teilstücke aus ganzen Läufen wie beim FD-Pfad (J höchstens die Hälfte des nutzbaren Speichers); Bedarf pro Lauf `DeviceAdjointJacobian::buffer_bytes` (J, r₀, Status) plus eine Variante des Batches, höchstens `kMaxGridBlocks`/(F·L) Läufe. Pro Teilstück ein Kernelstart (`DeviceAdjointJacobian::compute`, FP64), `DeviceNormalEquations::compute(const DeviceAdjointJacobian&)` (derselbe Gram-Kernel; allgemein über `DeviceJacobianRef`), Download von A und g. `LmGpuStats::jacobian_tranches` zählt einen Kernelstart pro Teilstück. Die Merit der Versuchspunkte ist unverändert (MERIT).
- `cuda::MultiGpuLmBackend` und `optimize_lm_rounds` (weiches Versagen) übernehmen das Verfahren aus `LmOptions` ohne Änderung.

**Determinismus.** Der adjungierte Kernel rechnet einen Block pro (Lauf, Feld, Wellenlänge) ohne Abhängigkeit von anderen Läufen und ohne Atomics (Kolbenprojektion mit fester Reihenfolge, ein Warp pro Variable), der Gram-Kernel ist eine FMA-Kette pro Element: Ein Lauf ist gemeinsam mit anderen, in Teilstücken von einem Lauf und auf mehreren (logischen) GPUs bitgleich zu seinem Einzellauf (Test `CudaAdjointLm.RunsBitwiseEqualToSingleRunsChunksAndSlots`; CPU `AdjointLm.RunsTogetherBitwiseEqualToSingleRuns`, auch für 1 und 4 Threads).

**Zusatzzeilen (hybrid).** Gewöhnliche Zusatzresiduen (paraxial, L1-Strafe, Glas-Hülle, mechanisch ohne Strahlbezug) rechnet `extra_rows_untraced`: pro Lauf `fd_pattern` und eine Variante als Arbeitsfläche, nacheinander x₀ und die 2P FD-Punkte (`VariableSet::apply` wie `write_fd_variants`), `ExtraResiduals::evaluate`, `fd_derivative` – bitgleich zu den Zeilen der FD-Jacobi, ohne den FD-Batch aufzubauen, Läufe auf `extra_threads` Host-Threads. Trace-abhängige Zusatzresiduen (mechanisch aus Strahlhöhen und Einfallswinkeln) brauchen weiter die Extrema aller 2P + 1 FD-Varianten (ein MERIT-Lauf über den FD-Batch, auf der CPU `surface_extrema_cpu`).

**Abnahme „gleiche Optimierungsergebnisse wie mit finiten Differenzen“** (Endmerit relativ ≤ `kLmReturnRelative` = 10⁻⁶; die Wege dürfen abweichen, weil die adjungierte Jacobi bis auf Rundung exakt und die FD-Jacobi um 10⁻⁸ … 10⁻⁶ ungenau ist):

| Aufgabe | adjungiert gegen FD |
|---|---|
| Einzellinse, Biegung (M3-T4, Abnahme 1) | q = 0,720798 (FD 0,720808) auf der CPU, 0,720797 (FD 0,720789) auf der GPU; F relativ 8,4·10⁻¹⁰ bzw. 3,2·10⁻¹⁰; 54 statt 55 bzw. 56 Zyklen |
| Mehrlinser, 13 Variablen (M3-T4, Abnahme 2) | F_opt relativ 3,3·10⁻¹¹ (CPU) bzw. 9,2·10⁻¹² (GPU), Endpunkte max \|Δx\|/s = 5,9·10⁻⁶ bzw. 2,9·10⁻⁶; Rückkehr aus 1 % Störung −4,2·10⁻¹¹ … −1,1·10⁻¹¹ relativ zu F_opt |
| Läufe gemeinsam gegen einzeln (M3-T4, Abnahme 3) | bitgleich: 8 Läufe CPU (1 und 4 Threads), 120 Läufe GPU, auch mit 64 KiB Speichergrenze (5.647 Teilstücke zu je einem Lauf) und auf zwei logischen GPUs (Anteile 1 : 3) |
| Weiches Versagen (M4-T2), 27 von 90 Strahlen versagt | versagensfrei nach 4 Runden wie FD, F relativ 2,6·10⁻¹⁴ (CPU) bzw. 3,2·10⁻¹⁴ (GPU), 144 statt 146 Zyklen |
| Geisterplatten (M4-T6): Exaktheit, Bedarf | max \|a\| = 2,6·10⁻⁹ < 10⁻⁸, Stationarität 5·10⁻⁷ (CPU) bzw. 3,3·10⁻⁷ (GPU), F relativ 8,7·10⁻¹⁰ bzw. 4,4·10⁻⁹; Bedarf: RMS 0,0875 Wellen wie FD, F relativ 2,4·10⁻¹⁴ |
| Glas als Variable (M6-T1), Fortsetzung 10² … 10⁸ | alle Stufen konvergiert, Abstand zur Hülle am Ende 2,2·10⁻⁹, F relativ 1,5·10⁻¹⁰ |
| Mechanische Bedingungen (M4-T3), Fortsetzung 10² … 10⁶ | alle Grenzen eingehalten, Endmerit 18,525512538 wie FD. In einer Vorversion des Kernels (andere Summationsreihenfolge der Projektion) endete die adjungierte Fortsetzung ab w = 10⁵ in einem anderen, tieferen lokalen Minimum (F = 18,23): Die Strafen √w·max(0, g) aus Extrema sind nicht glatt. Geprüft wird deshalb, dass jedes Verfahren am Endpunkt des anderen stehen bleibt (relativ ≤ 10⁻⁶). |
| CLI `optimize`, `snap`, `synth` mit `--jacobian adjoint` | Ergebnis bitgleich zur Bibliothek mit `JacobianMethod::Adjoint`; `optimize`: F relativ zu `fd` ≤ 10⁻⁶; `synth`-Kurzlauf mit gleichem Status, gleichen Einfügungen und gleichem Endpunkt |

Normalgleichungen adjungiert gegen FD (Fehler der FD-Jacobi, `kAdjointLmNormalEquations` = 10⁻⁶): skaliert 2,7·10⁻¹⁰ (Mehrlinser), 2,6·10⁻⁸ (Einzellinse an Schranken, einseitige Schablonen), 5·10⁻¹² (Zweilinser mit mechanischen Bedingungen); GPU gegen CPU adjungiert ≤ 3,9·10⁻¹¹.

**Abnahme „Jacobi mindestens 10× schneller für P ≥ 60“** ([benchmarks.md](benchmarks.md#adjungierte-jacobi-im-lm-m5-t3), RTX 3090): ohne Zusatzresiduen 20–37-mal ab 100 Läufen pro Aufruf (P = 61 und 81, FP64 und df64), 11–18-mal einschließlich der Normalgleichungen; mit 10 Läufen 4,6–12-mal, mit einem Lauf 1,2–2-mal (Latenz eines Strahls, die GPU ist nicht ausgelastet). Mit paraxialen Zusatzresiduen begrenzen die 2P + 1 Auswertungen pro Lauf auf dem Host (1,0–9,2-mal), mit trace-abhängigen mechanischen Bedingungen ist die adjungierte Jacobi so teuer wie die FD-Jacobi (0,7–1,0-mal). Die Tesla P100 ist hier nicht verfügbar; Befehl in benchmarks.md.

**Standard.** `FiniteDifferences` bleibt vorerst Standard: (1) Die Abnahme auf der P100 steht aus. (2) Mit trace-abhängigen mechanischen Bedingungen und bei einzelnen Läufen bringt die adjungierte Jacobi nichts bzw. wenig; mit wenigen Läufen und mechanischen Bedingungen ist sie bis 30 % langsamer. (3) Ein Wechsel ändert die Iterationsverläufe aller dokumentierten Ergebnisse. Empfehlung: `--jacobian adjoint` für Multistart und Homotopie ohne trace-abhängige Bedingungen; Umstellung des Standards nach der P100-Messung (Frage an den Maintainer).

## 5. Geisterplatten

### Referenzsphäre für die Synthese mit Geisterplatten

**Empfehlung (Entscheidung Maintainer 2026-11-19, [ADR-0017](adr/0017-referenzsphaere-bildpunkt.md), Punkt 7 und Nachtrag):** Systeme, die mit Geisterplatten wachsen, rechnen mit `reference_sphere: "image_infinite"` ([numerik.md, Abschnitt 7](numerik.md#referenzsphäre-um-den-bildpunkt-m6-t7-adr-0017)). Bei `image_exit_pupil` kann die paraxiale Austrittspupille eines Systems mit Geisterplatten nahe am Bild liegen (M6-T7: 0,43 mm hinter dem Bild, Radius R ≈ 0,43 mm); die Sphäre ist dann winzig, und ihre Falle im Punkt **E** liegt in der Nähe der Bildebene. Bei unendlichem Radius gibt es weder Falle noch Verfehlen der Sphäre. Werkzeuge, die neue Systeme anlegen (`start_from_skeleton` in `mikroptim.design`), setzen das Feld deshalb auf `image_infinite`; bestehende Dateien ohne Feld rechnen unverändert mit `last_surface` (die CLI warnt, [cli.md](cli.md#warnung-ohne-reference_sphere-entscheidung-maintainer-2026-11-19)).

### Platzierung

Mehrere Kandidatenpositionen in Luftspalten: mindestens eine nahe der Blende (feldunabhängige Fehler: sphärische Aberration, Farblängsfehler), eine in der Mitte und eine im Frontbereich (großes Verhältnis Hauptstrahl- zu Randstrahlhöhe: Koma, Astigmatismus, Petzval, Farbquerfehler).

### Glas-Wörterbuch

An jeder Position liegen mehrere Geisterplatten mit festen, repräsentativen Gläsern (z. B. Fluorit-artig, hochbrechendes Kronglas, Flintglas, KZFS-artig). Die L1-Strafe wählt aus, welche gebraucht werden. Freie Glasparameter in Geisterplatten wären nutzlos, weil ihr Gradient bei a = 0 verschwindet.

### Erweitertes Wörterbuch (M6-T12)

Anlass: Das Startsystem der Zielspezifikation und die Rückkehr zu Geisterplatten (M6-T10) benutzen das Wörterbuch SYN-36, CaF₂, SYN-30 (n_d/ν_d/ΔP_g,F = 1,803/48,6/+0,008, 1,434/95,0/+0,055, 1,784/22,3/+0,038). Die monochromatischen Syntheseläufe bauen ihre Elemente nur aus diesen drei Gläsern und sind danach nicht mehr farbkorrigierbar ([Bericht M6-T11](ergebnisse/m6-t11-farbe.md), F13). Messung: [Bericht M6-T12](ergebnisse/m6-t12-glaswoerterbuch.md).

```cpp
struct GhostDictionaryGlassOptions { int count; std::vector<std::string> names;
                                     std::vector<double> wavelengths_nm;  // g, F, e, d, C
                                     double max_power_sum = 6, anomalous_dpgf = 0.01,
                                            duplicate_nd = 0.005, duplicate_vd = 1; };
std::vector<GhostDictionaryGlass> select_ghost_dictionary_glasses(const SystemDescription&, const GlassCatalog&,
                                                                  const GhostDictionaryGlassOptions&);
GhostDictionaryExtension extend_ghost_dictionaries(const SystemDescription&, const GlassCatalog&,
                                                   const GhostDictionaryGlassOptions&);
```

API: `include/mikroptim/synth/ghost_dictionary.hpp` (Bibliothek `Mikroptim::synth`), Kommandozeile `mikroptim synth --ghost-dictionary-extra N` und `--ghost-dictionary-glass GLAS` ([cli.md](cli.md#mikroptim-synth)), Skript `tools/m6t6_zielspezifikation.py --woerterbuch-extra [N]`, `--woerterbuch-glas GLAS`, Aktion `synth_segment` mit `ghost_dictionary_extra` in `mikroptim.design`. Standard aus; ohne Option bitgleich (Test `CliSynth.GhostDictionaryExtra`: auch mit N = 0, also ohne ausgewähltes Glas, Protokoll und Ergebnisdatei bitgleich bis auf die Laufzeiten).

1. **Rangfolge** (Glasvorauswahl M11-T5g, [numerik.md, Abschnitt 15](numerik.md#15-algebraische-glasvorauswahl-m11-t5)): Jedes Katalogglas mit Buchdahl-Fit über 400–700 nm wird mit jedem anderen als dünner Achromat gelöst (Paare, Σ|w φ|/|φ| ≤ 6 wie die Vorauswahl von Operator B) und erhält den kleinsten spektralen Rest über die Linien g, F, e, d, C (436–656 nm, Zielspektrum der Spezifikation; unabhängig vom Spektrum der Datei, damit eine monochromatische Stufe dasselbe Wörterbuch erhält wie die Farbstufe). Die Rangfolge hängt nicht vom System ab.
2. **Klassen** (Kennzahlen wie die Glasabfrage, numerik.md §17): *anomal* bei |ΔP_g,F| ≥ 0,01 (Glasregel `anomalous_pgf` der Wissensbasis), sonst *Flint* bei ν_d < 55 für n_d < 1,60 bzw. ν_d < 50 für n_d ≥ 1,60 (Kron-Flint-Grenze der üblichen Glaskarten; Mikroptim-Annahme). Kron mit normaler Teildispersion gehört zu keiner Klasse.
3. **Auswahl:** ausdrücklich genannte Gläser, dann N Gläser abwechselnd das beste Flint- und das beste anomale Glas (Flint zuerst; ist eine Klasse erschöpft, die andere); ein Glas, das einem schon gewählten gleicht (|Δn_d| < 0,005 und |Δν_d| < 1), wird übersprungen. Ein gewähltes Glas, das schon im Wörterbuch steht (Katalogname einer Platte oder ein Modell mit gleichen Kennzahlen, etwa der Buchdahl-Fit von SYN-36 im Startsystem), zählt mit und bekommt keine Platte: Das Wörterbuch enthält danach die N besten Gläser der beiden Klassen. Deshalb ist die Erweiterung wiederholbar ohne Wirkung – `synth` wendet sie bei jedem Aufruf an, das Skript reicht die Option an jeden Aufruf durch, auch nach der Rückkehr zu Geisterplatten. Mit den Katalogen `basis.json` + `synthetisch.json` ergibt das SYN-32 (Flint, 1,769/48,4), CaF₂ (anomal, schon vorhanden), SYN-13 (Flint, 1,573/47,8), SYN-04 (anomal, 1,474/88,5, FK51-artig): N = 2 setzt eine, N = 4 drei neue Platten je Lage. Die besten Flintgläser der Rangfolge sind leichte Flinte (Lanthan-Flint-artig) als Partner von CaF₂, nicht die Schwerflinte; dichte Flinte mit normaler Teildispersion (etwa SYN-21, ν_d 34,5) lassen sich mit `--ghost-dictionary-glass` nennen.
4. **Platten:** je Lage (Platten mit gleichem `after_surface` und `distance`) eine Platte je neuem Glas über `place_ghost_dictionary` (Koeffizienten 0, Ordnung K und r_g der ersten Platte der Lage, hinter den vorhandenen; Variablen und Strafe wie die vorhandenen Platten, z. B. der Eintrag `ghost_penalty` mit `"all"`). Platten mit a = 0 brechen nicht und haben keine Strafe: Die Merit bleibt bis auf Rundung gleich (Test, auch mit aktiven Platten: bitgleich). Jede Platte kostet K Variablen (im Startsystem 10 je Lage, also 20 je Glas). Gemessen (12 El., 7 Wellenlängen, zwei Lagen): LM-Zeit je Zyklus +6 % (N = 2, 8 Platten), +38 % (N = 4, 12 Platten), +80 % (N = 6, 16 Platten).
5. **Standardgröße N = 4** (Skript `--woerterbuch-extra` ohne Zahl) für die monochromatischen Stufen: Nur mit SYN-13 und SYN-04 wurde der monochromatische Endstand farbfähig (Farbstufe danach beugungsbegrenzt), N = 2 verlief wie ohne Erweiterung; in polychromatischen Stufen brachte die Erweiterung nichts Messbares ([Bericht](ergebnisse/m6-t12-glaswoerterbuch.md)). Die Option bleibt in `synth` standardmäßig aus.

**Diagnose und Operatoren mit den Zusatzgläsern.** Die Diagnose (Abschnitt 5) wertet jede Platte mit ihrem Glas aus; die Zusatzplatten sind Katalogglas, ihr Name erscheint im Bericht. Operator A nimmt das Glas der aktivsten Platte der Lage (ein Flint-Meniskus, wenn die Flint-Platte die Korrektur trägt), Operator B das Glas der aktiven Platte mit der größten Abweichung |1/ν_Platte − 1/ν_Linse| als Zielglas (bei einer Kronlinse und aktiver Flint-Platte das Flint; mit `--split-start-target` startet das freigegebene Glas dort). Einrasten (Abschnitt 9) betrifft nur Glasvariablen; Menisken aus Zusatzgläsern tragen schon den Katalognamen (keine Umbenennung wie bei den Buchdahl-Fits des Startsystems, `buchdahl_to_catalog` im Skript). Orakel (`tests/test_ghost_dictionary.cpp`): eine Flint-Platte (SYN-21, ausdrücklich genannt) mit δφ_G = −δφ_L vor einer N-BK7-Einzellinse senkt die Farbe der Brechkraft von 3,06·10⁻⁴ auf 1,4·10⁻⁵ /mm (dünner Achromat, Platte zerstreuend); allein aktiv → Operator A mit Flint-Meniskus; CaF₂ sammelnd und Flint zerstreuend bei Brechkraft 0 → Operator B an der Linse mit dem Flint als Zielglas, Kittkrümmung −0,0641/mm (dünner Achromat aus den relativen Dispersionen −0,0638/mm), mit den Krümmungen des dünnen Achromats sinkt die Längsfarbe 0,778 → 0,005 mm.

Tests: `GhostDictionary.SelectionFollowsPreselectionRank` (Orakel: jedes Paar einzeln in der Vorauswahl, Klassen aus glass_properties; vorhandene Gläser markiert; genannte Gläser; Fehler), `…ExtensionKeepsMeritAndIsIdempotent`, `…FlintPlateCorrectsColorOfSinglet`, `…FlintPlateLeadsToFlintElementOrDoublet`; `CliSynth.GhostDictionaryExtra`; Python `test_tool_m6t6.py::test_synth_options_ghost_dictionary`, `test_design.py::test_synth_segment_ghost_dictionary_extra`.

### Diagnose

Nach jeder Konvergenz wird pro Geisterplatte berichtet: Summe |a_k|, dominante Ordnungen, Glas, paraxiale Brechkraft. Daraus folgt die Empfehlung:

| Befund | Bedeutung | Operator |
|---|---|---|
| hauptsächlich monochromatische Korrektur (gleiches Muster über alle Wellenlängen) | sphärische Aberration höherer Ordnung, Aplanasie, Petzval | A: Meniskus ohne Brechkraft |
| Korrektur über Gläser mit unterschiedlicher Dispersion | Farbfehler 1. Ordnung | B: Aufspalten in Kittglied (normale Gläser) |
| Bedarf an anomaler Teildispersion (Fluorit-Platte aktiv) | sekundäres Spektrum | B mit Fluorit- oder FK51-artigem Glas |

### Diagnosebericht (ab M4-T6)

API `include/mikroptim/optim/ghost_diagnosis.hpp`: `diagnose_ghost_plates(system, catalog, batch, v, penalty, merit, options)` liefert `GhostDiagnosis` (Struktur), `write_ghost_diagnosis_text` den Textbericht und `ghost_diagnosis_json` ein JSON-Objekt (nicht endliche Zahlen als null). `merit` ist eine Funktion, die F für alle Varianten eines Batches liefert (`make_cpu_merit(quadratur, zusatzresiduen)`: wie `CpuLmBackend::merit`); leer heißt ohne Merit-Werte. `mikroptim optimize` gibt den Bericht als Abschnitt der Zusammenfassung aus, wenn das System Geisterplatten hat ([cli.md](cli.md#mikroptim-optimize)).

**Pro Platte** (`GhostPlateReport`): Lage (`after_surface`, `distance`, Tabellenfläche), r_g, K, Koeffizienten, Σ|a_k|, max |a_k|, Aktivität (max |a_k| > Schwelle, Standard 10 ε der Strafe der Platte; inaktive Koeffizienten liegen bei |a| ≲ ε, Abschnitt 3), dominante Ordnungen (aktive Ordnungen mit |a_k| ≥ 0,1 · max |a|, absteigend), Glas (Name, n_G bei λ_ref, n_G(λ_kurz) − n_G(λ_lang) der Merit-Wellenlängen, n_d, ν_d = (n_d − 1)/(n_F − n_C), P_g,F = (n_g − n_F)/(n_F − n_C) und ΔP_g,F = P_g,F − (0,6438 − 0,001682 ν_d) gegen die Normalgerade, aus `material_index` bei g, F, d, C; NaN, wenn nicht auswertbar), φ_G und φ_G/(n_m n_G) bei λ_ref (numerik.md §6), paraxiale Randstrahlhöhe (volle Pupille) und Hauptstrahlhöhe für 1 rad Feldwinkel an der Platte (Blendennähe: Hauptstrahlhöhe klein gegen Randstrahlhöhe; im kollimierten Strahl vor der ersten Linse ist sie gleich dem Abstand von der Blende), Strafbeitrag Σ r² der Platte, F mit a = 0 für diese Platte (ein zusätzlicher Trace pro Platte, alle in einem Batch mit G + 1 Varianten) und der Merit-Gewinn F(a_g = 0) − F (> 0: die Platte senkt die Merit netto, Strafe eingerechnet).

**Pro Lage** (`GhostPositionReport`, Platten mit gleichem `after_surface` und `distance`, also ein Glas-Wörterbuch): aktive Platten, Klassifikation und Begründung nach der Tabelle oben mit folgender Heuristik:

1. Keine aktive Platte: keine Empfehlung.
2. Nur eine Merit-Wellenlänge (Gewicht > 0): **A** – die Platten korrigieren das Muster dieser Wellenlänge (monochromatisch).
3. Sonst wird die gemeinsame Korrektur der aktiven Platten in einen monochromatischen und einen Farbanteil zerlegt: T_k = Σ_g (n_g,ref − n_m) a_{g,k} (Phase bei λ_ref) und ΔT_k = Σ_g Δn_g a_{g,k} mit Δn_g = n_g(λ_kurz) − n_g(λ_lang) (Unterschied der Phase zwischen den äußeren Merit-Wellenlängen), beide in der Norm Σ_k x_k²/(2k + 1) (RMS des Profils, Orthogonalität der P_k). Der Farbanteil χ = ‖ΔT‖/‖T‖ einer einzelnen Platte ist |Δn_g|/(n_g,ref − n_m), also höchstens das größte χ_g der aktiven Gläser („gleiches Muster über alle Wellenlängen“ bis auf die Dispersion eines Glases). Ist χ > 2 · max χ_g, unterscheiden sich die Muster der Wellenlängen stärker, als es eine einzelne Platte könnte – das geht nur durch Kombination von Gläsern verschiedener Dispersion (etwa Kron positiv, Flint negativ bei kompensierter Brechkraft): **B**. Sonst **A**.
4. **B mit anomaler Teildispersion**, wenn bei B mindestens drei Merit-Wellenlängen vorliegen und eine aktive Platte mit ΔP_g,F ≥ 0,02 (CaF₂: +0,055, FK51-artig ≈ +0,03; normale Gläser |ΔP| < 0,01) mindestens 25 % von Σ|a| der Lage trägt. Bei zwei Wellenlängen genügt ein großer Abbe-Unterschied für die Farbkorrektur; eine aktive Fluorit-Platte ist dort kein Nachweis eines sekundären Spektrums (Hinweis in der Begründung, Klassifikation B).

Alle Schwellen stehen in `GhostDiagnosisOptions`. **Empfehlung** des Berichts: die Lage mit dem größten Merit-Gewinn ihrer aktiven Platten (ohne Merit: größte Σ|a|), mit Operator, Glas der aktivsten Platte und dominanter Ordnung; Initialisierung des Operators daraus: Abschnitt 6 (M6-T2).

Tests: `tests/test_ghost_diagnosis.cpp` (Felder gegen unabhängige Rechnungen: t₂ und φ_G, Rand- und Hauptstrahlhöhe im kollimierten Strahl, Merit mit Orakel 1 + Σ a², Abbezahlen und ΔP_g,F von CaF₂, N-BK7 und dem Flint; Klassifikation A/B/B anomal an vorgegebenen Koeffizienten; Text und JSON), Bedarf-Abnahme in Abschnitt 3.

## 6. Einfüge-Operatoren

Beide Operatoren ändern die Merit **exakt nicht** (Plan: |ΔF| ≤ 10⁻¹² relativ; in FP64 bis auf den Rundungsboden des Traces, siehe „Festlegungen der Implementierung“).

**A – Meniskus ohne Brechkraft (nach Bociort):** In einem Luftspalt werden zwei Flächen gleicher Krümmung c₀ im Abstand 0 mit einem Glas aus dem Wörterbuch eingefügt. Die Dicke wird anschließend über eine Homotopie auf die Mindestdicke gebracht. c₀ und Glas werden aus der aktivsten Geisterplatte an dieser Position initialisiert (Vorzeichen der Biegung, Glastyp). Danach wird das Strafgewicht dieser Geisterplatte erhöht, sodass die Korrektur in das reale Element wandert. Dass eingefügte Elemente Sattelpunkte erzeugen können, wird genutzt: Beide Abstiegsrichtungen werden als getrennte Läufe weiterverfolgt.

**B – Aufspalten in ein Kittglied:** Eine Glaslinse wird an einer inneren Fläche (Krümmung c_s, z. B. aus der Geisterdiagnose oder als Mittel der Außenflächen) in zwei Teile aus **demselben** Glas geteilt. Danach werden die Glasparameter eines Teils freigegeben (Variable `glass`, Abschnitt 1; Hülle der Katalog-Gläser als Strafterm, Abschnitt 2.4) und wandern stetig, z. B. von einem Kronglas Richtung Fluorit.

### Festlegungen der Implementierung (M6-T2)

API: `include/mikroptim/optim/insertion.hpp` (Bibliothek `Mikroptim::optim`), Kommandozeile `mikroptim insert` ([cli.md](cli.md#mikroptim-insert)). Die Operatoren arbeiten auf der `SystemDescription` und geben eine neue zurück (`InsertionResult`: System, Abbildung alte → neue Dateiflächen, erste eingefügte Fläche, Zahl der eingefügten Flächen); die Topologie ändert sich, Brechzahltabelle, Batch und `VariableSet` werden neu gebaut (`prepare_system`, `make_system_batch`). Das neue System wird mit `validate_system` geprüft.

```cpp
SystemDescription renumber_surfaces(const SystemDescription&, std::span<const int> new_of_old,
                                    std::vector<Surface> new_surfaces);
InsertionResult insert_meniscus(const SystemDescription&, const MeniscusInsertion&);   // A
InsertionResult split_lens(const SystemDescription&, const LensSplit&);                // B
std::array<SystemDescription, 2> meniscus_saddle_starts(const SystemDescription&, int first,
                                                        double delta_front, double delta_back);
void free_glass(SystemDescription&, int surface, const std::optional<MaterialSpec>& start = {});
void scale_ghost_penalty(SystemDescription&, int plate, double factor);
MeniscusPlan meniscus_from_diagnosis(const SystemDescription&, const GhostDiagnosis&,
                                     const MeniscusInitOptions& = {});
SplitPlan split_from_diagnosis(const SystemDescription&, const GlassCatalog&,
                               const GhostDiagnosis&, const SplitInitOptions& = {});
double achromat_split_curvature(double c1, double c2, double abbe_front, double abbe_back);
double ghost_profile_t4(const GhostPlate&);   double abbe_number(const MaterialSpec&, const GlassCatalog&);
```

**Umnummerierung** (`renumber_surfaces`, einzige Stelle mit Flächenbezügen; Test `RenumberSurfaces.*`). Die Abbildung i → new_of_old[i] ist streng monoton, die Blende bleibt Fläche 0, die Frontfläche die letzte. Angepasst werden: Flächenlisten der Variablen `curvature`, `thickness`, `glass` (Einträge mit `"all"` werden in die explizite Liste der alten Flächen umgewandelt, damit die Variablenmenge erhalten bleibt und eingefügte Flächen nicht unbemerkt Variablen werden; die Reihenfolge der Variablen bleibt), `after_surface` der Geisterplatten, `reference_surface` der Abgleichlänge und die Flächen der mechanischen Nebenbedingungen (M4-T3; Standardauswahl bzw. `"all"` wird wie bei den Variablen zur expliziten Liste der alten Flächen aus `constraint_surfaces`, sodass eingefügte Flächen nicht unbemerkt eingeschränkt werden; bei den Dicken-Bedingungen ist die Fläche i der Spalt i → i + 1). Arbeitsabstand, Geisterkoeffizienten und `ghost_penalty` beziehen sich auf die letzte Fläche bzw. auf Plattenindizes und bleiben; die Operatoren fügen keine Platten ein und ändern ihre Reihenfolge nicht. Weitere Nebenbedingungen mit Flächenbezug gehören in diese Funktion (Test `RenumberSurfaces.MechanicalConstraints`: Einfügen vor, zwischen und hinter den referenzierten Flächen).

**Operator A** (`insert_meniscus`): Im Luftspalt T nach Dateifläche i entstehen die Flächen i + 1 (Typ `sphere`, c₀, Glas, Dicke t_m, Standard 0) und i + 2 (Typ `sphere`, c₀, Luft, Dicke fl(T − p − t_m)); Fläche i hat danach die Dicke p (`position`). Geisterplatten des Spalts mit distance ≤ p bleiben vor dem Meniskus, die übrigen (distance ≥ p + t_m) liegen danach nach Fläche i + 2 mit distance − p − t_m; eine Platte im Glas ist ein Fehler. Eine Dickenvariable des Spalts gilt danach für beide Luftspalte (gleicher Eintrag, gleiche Schranken; ein Teilstück unter der Schranke ergibt `FormatError`), ebenso mechanische Mitten- und Randdicken-Bedingungen des Spalts (dort kein Fehler: Verletzt ein Teilstück die Grenze, wird die Strafe aktiv und die Merit ändert sich). Optional (`curvature_variables`, Standard an) ein neuer Eintrag `curvature` für beide Meniskusflächen mit eigenen Schranken. Halbdurchmesser: Vorgabe, sonst der größere der beiden Spaltflächen (er geht nicht in den Trace ein). Die Mitteldicke t_m ist der Homotopie-Parameter (M6-T3); sie ist keine Variable (der Homotopie-Controller macht sie nach der inneren Homotopie optional zur Variable, Abschnitt 7).

**Operator B** (`split_lens`): Die Linse aus Fläche i (Glas danach) und i + 1 wird an der neuen Fläche i + 1 (Typ `sphere`, c_s, dasselbe Material) geteilt, Dicken t₁ = fl(f·t) und t₂ = fl(t − t₁) mit dem Anteil f (`fraction`, 0 … 1). Dicken- und Glasvariablen sowie mechanische Mitten- und Randdicken-Bedingungen der Linse gelten danach für beide Teile (eine Grenze für die ganze Linse, etwa die Randdicke, ist für jeden Teil meist zu streng; dann wird die Strafe aktiv); optional ein neuer Eintrag `curvature` für die Kittfläche. Halbdurchmesser: Vorgabe, sonst der kleinere der beiden Linsenflächen. `free_glass(system, s, start)` gibt das Glas nach Fläche s als Variable frei (Eintrag `glass`, nichts, wenn es schon frei ist), optional mit anderem Startglas; danach wie in M6-T1 `convert_glass_variables` und die Nebenbedingung `glass_hull`.

**Merit-Invarianz.** Mit t_m = 0 brechen die Strahlen an zwei identischen Flächen ein und wieder aus, an der Kittfläche ist μ = n/n = 1; mathematisch ändert sich nichts. Bitgleich ist das Ergebnis nicht: Spalt bzw. Glasdicke werden in Teilstücke zerlegt (zusätzliche Transfers und Schnittpunkte), die OPL-Summe erhält mehr Summanden, und die Brechungsformel (numerik.md, Abschnitt 5) stellt die Richtung nur bis auf Rundung her (k = 1 − μ²(1 − cos² I) ist nicht bitgleich cos² I). Die Änderung liegt am Rundungsboden des sequenziellen FP64-Traces, der relativ zu F umso größer ist, je besser das System korrigiert ist (OPD-RMS ≈ 10⁻⁴ mm gegen optische Längen ≈ 100 mm): Schon zwei Flächen mit n = 1 im Luftspalt (optisch nichts, nur Transfer in Teilstücken) ändern F der Einzellinse (F = 3·10⁻² Wellen²) um bis zu 1,1·10⁻¹² relativ (Test `InsertionMerit.RoundingFloorControl`). Gemessen über 26 Fälle (`InsertionMerit.UnchangedOnCpu`, GPU `CudaInsertion.MeritUnchangedOnGpu`: A an verschiedenen Lagen, Krümmungen −0,4 … 0,6/mm und Gläsern, auch zwischen, an und hinter Geisterplatten mit Koeffizienten ≠ 0 und mit paraxialen Nebenbedingungen und L1-Strafe; B mit Anteil 0 … 1 und verschiedenen Kittflächen; Einzellinsen und Mehrlinser mit drei Feldern und vier Wellenlängen; zwei Fälle von A im Zweilinser mit mechanischen Nebenbedingungen, M4-T3, dort ≤ 1,9·10⁻¹⁵):

| | CPU | GPU (RTX 3090) |
|---|---|---|
| größte relative Änderung, Operator A | 2,1·10⁻¹² | 2,0·10⁻¹² |
| größte relative Änderung, Operator B | 3,0·10⁻¹³ | 5,7·10⁻¹³ |
| Fälle mit ≤ 10⁻¹² | 24 von 26 | 23 von 26 |
| F GPU gegen CPU (eingefügte Systeme) | – | höchstens 2,3·10⁻¹² relativ |
| Rundungsboden (Flächen mit n = 1) | 1,1·10⁻¹² | – |

Die Planvorgabe 10⁻¹² halten B in diesen Fällen immer und A in der Regel; im Abnahmesystem von B unten (Einzellinse am Optimum, zwei Wellenlängen) ändert der Split F um 4,7·10⁻¹² (GPU 1,3·10⁻¹²). Die Tests prüfen gegen `kInsertionMeritRelative` = 10⁻¹¹ (Begründung in `tests/tolerances.hpp`, Frage an den Maintainer im PR). Paraxial ändert sich die Brennweite höchstens um `kParaxialRelative`. Gegenprobe: derselbe Meniskus mit t_m = 0,5 mm ändert F um mehr als 10⁻³ relativ.

**Initialisierung aus dem Diagnosebericht** (Abschnitt 5):

- *A* (`meniscus_from_diagnosis`): Lage = empfohlene Lage des Berichts (`after_surface`, `distance` → `position`); liegt sie näher als die größte untere Schranke b der Dickenvariablen des Spalts an einem Spaltende (beide Teilstücke erben diese Variablen, `renumber_surfaces`), rückt sie auf [b, T − t_m − b] in den Spalt (bei t_m = 0 merit-neutral, die Platten an der alten Lage liegen danach im passenden Teilstück), und ist T − t_m < 2b, wirft die Funktion `std::invalid_argument` (der Controller endet dann geordnet mit Grund statt mit einem Ladefehler; M10-T7b, Befund E6: eine frühere Dicken-Homotopie hatte Platten ans Spaltende geschoben, `insert_meniscus` lehnte das Teilstück der Dicke 0 ab, `synth` brach ab); **kurzer Plattenspalt (M10-T7e, Befund E14):** mit `MeniscusInitOptions::shrink_short_gap` (CLI `synth --shrink-short-gaps`, in `mikroptim.design` immer gesetzt) liegt der Meniskus bei T − t_m < 2b stattdessen in der Mitte, Lage (T − t_m)/2, und beide Teilstücke verlassen den Dickeneintrag des Spalts und erhalten je einen eigenen (Kopie mit max/typical) mit der unteren Schranke b_k = max(min((T − t_m)/2, t_k), größte Plattenlage im Teilstück k) (`MeniscusInsertion::piece_lower_bound`; t_k Startdicke des Teilstücks, damit der Start in den Schranken liegt und keine Schranke eine Dicke unter einer Platte zulässt); der Plan trägt eine Warnung (`MeniscusPlan::warning`), die im Protokoll von `synth` als „Warnung: …“ an der Einfüge-Entscheidung steht. Die Merit bleibt beim Einfügen gleich (Dicke 0; Test `InsertionInit.MeniscusShortPlateGapLowersPieceBounds`). Begründung der Hälfte: Die Schranke b ist hier die Plattenlage, auf die der Optimierer den Spalt geschoben hat (T = b); mit (T − t_m)/2 passen beide Teilstücke genau in den Spalt, ohne die Lage der übrigen Scheitel zu ändern. Ohne Option (Standard) wirft die Funktion wie bisher; Quellplatte = aktive Platte der Lage mit der größten Σ|a|; Glas = Glas dieser Platte. |c₀|: Vorgabe, sonst die Krümmung der dem Meniskus näheren Spaltfläche (wie Bociorts Konstruktion, die das Nullelement an eine vorhandene Fläche legt), sonst die der anderen, sonst 0 (Planplatte). Vorzeichen von c₀ = Vorzeichen von (n_G − n_m)·t₄ mit dem r⁴-Koeffizienten des Dickenprofils t₄ = Σ_k a_k q_k/r_g⁴, q_k = (−1)^k C(k, 2) C(k + 2, 2) (q₂ = 6, q₃ = −30, q₄ = 90: der Anteil der sphärischen Korrektur, bei dominanter Ordnung 2 das Vorzeichen von a₂); bei t₄ = 0 das von φ_G. Die Regel ist eine Heuristik (in einem kollimierten Bündel ist die Wirkung eines dicken Meniskus gleicher Krümmungen in c₀ gerade, das Vorzeichen zählt erst im konvergenten Bündel); die Gegenrichtung liefert ein zweiter Lauf. `meniscus_saddle_starts(system, first, δ_a, δ_b)` erzeugt die beiden Startvarianten c_a = c₀ ± δ_a, c_b = c₀ ± δ_b (Standard δ_b = −δ_a: dünne Brechkraft ±2 (n − 1) δ); die Verwaltung der Zweige: Abschnitt 8 (Strahlsuche, M6-T4).
- *Strafgewicht* (`scale_ghost_penalty(system, g, factor)`): μ der Platte g wird mit `factor` multipliziert. Deckt ihr Eintrag `ghost_penalty` weitere Platten ab, wird die Platte daraus gelöst (`"all"` → Liste der übrigen) und erhält einen eigenen Eintrag mit μ·factor und denselben w₀, κ, ε; die übrigen Terme bleiben unverändert (Test `GhostPenaltyScale.SplitsSharedEntry`). `ghost_penalty_weight` liest μ einer Platte.
- *B* (`split_from_diagnosis`): Linse = die dem Spalt der Lage benachbarte Glaslinse (vor oder hinter dem Spalt, bei beiden die der Plattenlage nähere); Zielglas = Glas der aktiven Platte mit der größten Abweichung |1/ν_d − 1/ν_d,Linse| (bei „B anomal“ die mit dem größten ΔP_g,F); freigegeben wird Teil `free_part` (Standard 1, der hintere), sein Glas startet beim Linsenglas (Merit unverändert), das Zielglas bestimmt nur c_s. c_s: dünnes Achromat aus Linse (c₁, c₂) und Zielglas, φ₁ = φ ν₁/(ν₁ − ν₂), φ₂ = −φ ν₂/(ν₁ − ν₂), also c_s = c₁ − (c₁ − c₂) ν₁/(ν₁ − ν₂) (`achromat_split_curvature`, unabhängig von n); alternativ das Mittel der Außenkrümmungen. Verletzt c_s die Randdicke `min_edge_thickness` (Standard 0) eines Teils bei der Höhe h = kleinerer Halbdurchmesser (höchstens 0,99/|c| jeder Fläche, und nur so weit, wie die ungeteilte Linse dort Randdicke hat), wird c_s auf der Strecke zum Mittel zurückgenommen (Bisektion). Die Abbezahlen kommen aus `abbe_number` (Linien d, F, C; konstantes n: ∞).

**Abnahme B** (`tests/test_insertion_optimization.cpp`, GPU `tests/test_cuda_insertion.cpp`): frei erfundene Einzellinse `data/systems/einzellinse_achromat_start.json` (N-BK7, f = 50 mm, NA 0,05, Achse, 480 und 644 nm, λ_ref = 546,07 nm), Variablen c₁, c₂, w, Brennweite als Gleichung (10⁴/mm²). Ablauf: Einzellinse optimieren; Operator B mit c_s aus `achromat_split_curvature` und dem Glas kleinster Abbezahl der Hülle (SYN-30, ν_d = 22,3), Anteil 0,5, Kittfläche als Variable; Glas des hinteren Teils freigeben (Start N-BK7, Buchdahl-Fit); Penalty-Fortsetzung der Glas-Hülle (basis.json und Testgläser) über 10² … 10⁸ wie in M6-T1 mit höchstens 300 Zyklen pro Stufe. Farblängsfehler = paraxiale hintere Schnittweite bei 480 nm minus der bei 644 nm.

| Kriterium | CPU | GPU (RTX 3090) |
|---|---|---|
| Einzellinse: Farblängsfehler | −0,7664 mm (F = 6,45·10⁻² Wellen², 141 Zyklen) | −0,7664 mm (148 Zyklen) |
| Merit nach dem Operator B | relativ 4,7·10⁻¹² | relativ 1,3·10⁻¹² |
| Kittglied: Farblängsfehler | −0,0433 mm, **Faktor 17,7 kleiner** (F = 1,05·10⁻⁴) | −0,0433 mm, **Faktor 17,7 kleiner** (F = 9,86·10⁻⁵) |
| Brennweite | 50,00000001 mm (Einzellinse 50,00000013 mm) | 50,00000000 mm |
| Glas am Ende | n₀ = 1,519, ν_d = 52,2 (im Inneren der Hülle) | n₀ = 1,519, ν_d = 52,2 |

Verlauf: Nach 50 Zyklen der ersten Stufe ist der Farblängsfehler schon −0,046 mm; danach kriecht LM durch das flache Tal der Achromate (Farbe nahezu 0, Rest sphärische Aberration und Sphärochromasie bei NA 0,05) und erreicht in den Stufen bis 10⁶ die Zyklengrenze (`max_iterations`), die letzten Stufen konvergieren (`converged_step`). Mit dem vollen Gewicht der Hülle von Anfang an bleibt das Glas an der Ecke N-BK7 der Hülle hängen (Farblängsfehler nur Faktor 1,4 kleiner), die Fortsetzung ist also nötig. Das Glas wandert nicht bis zum Zielglas: LM findet ein Achromat mit einem schwächer dispergierenden Partner und stärker gekrümmter Kittfläche (c_s = −0,166/mm statt −0,031/mm).

**Ablauf A** (`InsertionOptimization.OperatorAFromDiagnosisCpu`, GPU `CudaInsertion.OperatorAFromDiagnosisGpu`): Einzellinse NA 0,5 mit Geisterplatten-Wörterbuch (`einzellinse_na05_geisterplatten.json`, Abnahme „Bedarf“ aus M4-T6): LM mit Geisterplatten, Diagnose (Empfehlung A, Flint-Platte 8 hinter der Linse, Ordnung 2), Plan aus `meniscus_from_diagnosis` (c₀ = +0,1417/mm = |c| der hinteren Linsenfläche, Glas Flint), Einfügen bei t_m = 0, Strafgewicht der Platte ×10, dann t_m = 0,01; 0,02; 0,05; 0,1; 0,15 mm (Luftspalt dahinter entsprechend kürzer), jede Stufe LM ab der vorigen (Krümmungen der Linse und des Meniskus, Arbeitsabstand, alle Geisterkoeffizienten):

| | CPU | GPU (RTX 3090) |
|---|---|---|
| mit Geisterplatten | RMS 0,0875 Wellen, Platte 8 Σ\|a\| = 7,77·10⁻³ mm | gleich (73 Zyklen) |
| Merit nach dem Einfügen | relativ 3,1·10⁻¹⁴ | relativ 3,9·10⁻¹⁴ |
| nach t_m = 0,15 mm | RMS 0,0473 Wellen, Platte 8 Σ\|a\| = 2,7·10⁻⁷ mm, übrige Platten 8,6·10⁻⁴ mm | gleich auf die gezeigten Stellen (256, 60, 75, 88, 89 Zyklen) |

Die Korrektur wandert vollständig aus der Platte in das reale Element; schon bei t_m = 0,01 mm ist sie übernommen (LM-Läufe 245, 62, 74, 92, 81 Zyklen, alle konvergiert). Der „Meniskus“ wird dabei zu einer sammelnden Linse (c = +0,788/+0,401 mm⁻¹): Die Meniskusflächen sind frei, und das Teilen der Brechkraft senkt die sphärische Aberration stärker als ein brechkraftloser Meniskus. Ohne Mechanik-Nebenbedingungen (M4-T3) hat sie bei r = 0,8 mm keine Randdicke mehr; das Beispiel zeigt den Ablauf, kein fertigbares System. Mit −c₀ (zweiter Zweig) endet der Lauf im selben Minimum.

## 7. Homotopie-Controller

Homotopie-Parameter: NA, Feld, Spektrum, Strafgewicht μ der Geisterplatten. Standardablauf:

1. monochromatisch (546 nm), auf der Achse, eine Linse, NA 0,3
2. NA in Schritten erhöhen bis 0,95; bei Geisterbedarf über Schwelle Operator A
3. Feld bis 100 % zuschalten, Petzval-Nebenbedingung scharf schalten
4. zwei Wellenlängen (480 und 644 nm): Achromasie, Operator B
5. 436–656 nm: Apochromasie, Operator B mit anomaler Teildispersion
6. 405 und 700 nm mit kleinem Gewicht dazu, Feinoptimierung

Schrittweitensteuerung: Konvergiert LM nach einem Schritt Δ schnell und bleibt die Merit unter der Schwelle, wird Δ vergrößert, sonst halbiert. Fällt Δ unter Δ_min und ist Geisterbedarf vorhanden, wird ein Einfüge-Operator angewandt.

### Festlegungen der Implementierung (M6-T3)

API: `include/mikroptim/synth/homotopy.hpp` (Steuerlogik, Protokoll) und `include/mikroptim/synth/optical_homotopy.hpp` (optisches Modell), Bibliothek `Mikroptim::synth` ([architektur.md](architektur.md#systemsynthese-homotopie-controller-m6-t3)); Kommandozeile `mikroptim synth` ([cli.md](cli.md#mikroptim-synth)).

```cpp
struct HomotopyPoint { double aperture, field; int spectrum;
                       double ghost_weight, ghost_smoothing, penalty_scale; };
class HomotopyModel {   // optisch: OpticalHomotopyModel; Tests: synthetische Modelle
    virtual HomotopySolve solve(SystemDescription&, const HomotopyPoint&) = 0;
    virtual HomotopyNeed diagnose(const SystemDescription&, const HomotopyPoint&) = 0;
    virtual HomotopyInsertion insert(const SystemDescription&, const HomotopyPoint&,
                                     const HomotopyNeed&) = 0;
    virtual void set_stage(SystemDescription&, const HomotopyInsertion&, double) = 0;
    virtual void finish_insertion(SystemDescription&, const HomotopyInsertion&) = 0;
    virtual HomotopyGoal goal(const SystemDescription&, const HomotopyPoint&, double target) = 0;
    virtual int elements(const SystemDescription&) const = 0;
    virtual bool one_sided_constraints(const SystemDescription&) const = 0; };
HomotopyResult run_homotopy(HomotopyModel&, SystemDescription start, const HomotopyPoint&,
                            const HomotopyOptions&);
```

**Zustand.** Basisbeschreibung (Werte aller Variablen, Topologie, μ pro Geisterplatte nach `scale_ghost_penalty`) und ein Punkt der Homotopie: numerische Apertur NA, Feldanteil φ ∈ [0, 1], Spektrumstufe k, Faktor auf μ aller Geisterstrafen, Faktor auf ε, Faktor s auf die Gewichte der einseitigen Nebenbedingungen. Die Basisbeschreibung behält Pupille, Feldwinkel, Spektrum und Gewichte der Startdatei; erst das Modell macht daraus das System des Punkts (`realize`). Die Steuerlogik kennt keine Optik: Lösen, Diagnose, Einfügen und Zielprüfung liefert das Modell (`HomotopyModel`), die Tests prüfen die Steuerung mit synthetischen Modellen.

**Realisierung eines Punkts** (`OpticalHomotopyModel::realize`):

| Parameter | Wirkung auf die Beschreibung |
|---|---|
| NA | kollimiert: Pupillenradius a = f_s · NA (Blende, `semi_diameter`), f_s = `target.focal_length` (Pflicht; die paraxiale Nebenbedingung hält die Brennweite fest, a = f·NA ist dann die Sinusbedingung im Unendlichraum); Punktquelle: `numerical_aperture`. Normierungsradien r_g aller Geisterplatten im selben Verhältnis wie der Pupillenradius zur Basis (das Profil skaliert mit dem Bündel, numerik.md §6: r_g soll den Strahlbereich abdecken) |
| Feldanteil φ | Feldwinkel der Startdatei × φ (Feldziele f_s tan θ) |
| Spektrumstufe k | Spektrum = k-tes Element von `OpticalHomotopyOptions::spectra` (leer: nur Stufe 0 = Datei) |
| μ-, ε-Faktor | Gewicht und Glättung jedes Eintrags `ghost_penalty` × Faktor |
| s | Gewicht jeder einseitigen Bedingung × s (`is_one_sided`: paraxial mit Modus min/max, Petzval, Glas-Hülle, alle mechanischen); Gleichungen und die L1-Strafe bleiben |
| Vorhalt (nur im LM, nicht in der Ergebnisdatei) | Mindestdicken (Mitte, Rand) + 5·10⁻⁵ mm, Höchstdicken − 5·10⁻⁵ mm, `clearance` der Strahlhöhe + 5·10⁻⁵ mm, Einfallswinkel − 0,01°, κ_max × (1 − 10⁻³), Sicherheitsabstand der Glas-Hülle + 0 (einstellbar). Die Strafe erreicht die Grenze nur asymptotisch (g* ≈ λ/(2w), Abschnitte 2.3–2.5); mit dem Vorhalt hält das Ergebnis die eigentliche Grenze ein, solange g* kleiner als der Vorhalt ist |

**Ein Schritt** (`solve`): realisieren, laden (Kataloge des Starts, ohne Dateizugriff), Batch, `VariableSet`, Zusatzresiduen wie `mikroptim optimize` (paraxial, L1-Strafe, Glas-Hülle, mechanisch geometrisch; trace-abhängig mechanisch), ein LM-Lauf über den Optimierer (Standard `optimize_lm_cpu`; `HomotopyOptimizer` z. B. mit `cuda::optimize_lm_gpu`, mit äußerem Strafverfahren des weichen Versagens) ab den Werten der Basis, Ergebnis zurück in die Basis (`apply_to_description`). Bericht: Status und Zyklen des LM, F, die Anteile Strahlresiduen F_W und L1-Strafe F_G, größter RMS-Wellenfrontfehler über die Tupel mit Gewicht > 0 (mit Geisterplatten, in Wellen bei λ des Tupels), Σ|a|, versagte Strahlen.

**Bewertung.** Ein Schritt ist annehmbar, wenn F endlich ist, kein Strahl versagt, der Status nicht `NumericalError` ist und die Bewertungsmerit

$$F_B = F_W + F_G \le F_{\max} = (m\,\sigma_{\text{Ziel}})^2,\qquad \sigma_{\text{Ziel}} = 1/14\ \text{Welle},\ m = 1\ (\texttt{merit\_factor}).$$

F_W ist der gewichtete mittlere RMS² der Wellenfront (Wellen², Abschnitt 2), F_G die Kosten der Geisterkorrektur. Die Nebenbedingungen zählen nicht: Eine aktive Strafe trägt am Optimum λ²/(4w) bei (beobachtet: Randdicke mit w = 10⁶/mm² ≈ 2,5·10⁻³ Wellen², halb so viel wie (λ/14)²), das hängt von der Wahl der Gewichte ab und nicht von der Güte des Systems. Mit m = 1 heißt annehmbar: Das System erreicht mit Geisterplatten das Beugungsziel, und die Geisterkorrektur ist billig. Steigt F_G, weil das reale System das Ziel nicht mehr allein erreicht, wird F_B > F_max: Das ist der Geisterbedarf, der zum Einfügen führt.

**Schrittweitensteuerung** (stetige Parameter Apertur und Feld, `HomotopyStepControl`, Standard Apertur Δ₀ = 0,02, Δ_min = 0,002, Δ_max = 0,1, Feld Δ₀ = 1/8, Δ_min = 1/64, Δ_max = 1/2): Versuchspunkt = guter Punkt + Δ (auf das Ziel gekürzt; Reste unter 10⁻⁶ Δ_min gehen aufs Ziel), Optimierung ab dem guten Zustand.

| Ergebnis des Versuchs | Entscheidung (`HomotopyAction`) |
|---|---|
| annehmbar, konvergiert (`lm_converged`) nach ≤ `fast_iterations` (50) Zyklen | `grow`: übernehmen, Δ ← min(2Δ, Δ_max) |
| annehmbar, langsamer oder nicht konvergiert | `keep`: übernehmen, Δ bleibt |
| nicht annehmbar, Δ/2 ≥ Δ_min | `shrink`: verwerfen, Δ ← Δ/2, nächster Versuch ab dem letzten guten Zustand |
| nicht annehmbar, Δ/2 < Δ_min, Diagnose meldet Bedarf | `insert`: Einfüge-Operator am letzten guten Zustand, danach Δ ← Δ₀ |
| nicht annehmbar, Δ/2 < Δ_min, kein Bedarf bzw. höchste Elementzahl | `stop`: Abbruch `step_too_small` bzw. `max_elements` mit Bericht |

Die Stufe „annehmbar, aber langsam“ ergänzt den Plan (der nur vergrößern oder halbieren kennt): Ein konvergierter, annehmbarer Schritt ist kein Grund zum Halbieren. Spektrumstufen sind diskret: eine Stufe pro Schritt, abgelehnt → sofort Diagnose und Einfügen (wie Δ < Δ_min).

**Bedarf und Einfügen.** Diagnose (`diagnose_ghost_plates`, Abschnitt 5) zuerst am abgelehnten Versuch (dort zeigt sich der Bedarf des nächsten Schritts, beim Spektrum z. B. Farbe statt Monochromasie), ohne Empfehlung am letzten guten Zustand; ohne Empfehlung kein Ausweg. Der Operator wird immer auf den **letzten guten Zustand** angewandt (die Topologie beider ist dieselbe, der Plan aus dem Versuch passt): Der Controller bewegt sich nur zwischen annehmbaren Zuständen. Ablauf:

1. Operator A (`meniscus_from_diagnosis`, `insert_meniscus` mit Dicke 0, Abschnitt 6) bzw. B (`split_from_diagnosis`, `split_lens`, `free_glass`, Glas-Hülle mit Standardgewicht, falls sie fehlt, `convert_glass_variables`); Strafgewicht μ der Quellplatte × 10 (`scale_ghost_penalty`).
2. Zweige: bei A zwei Starts am Sattelpunkt (`meniscus_saddle_starts` mit (δ, −δ), Standard δ = 0,002/mm; δ = 0: ein Zweig mit c₀ aus dem Plan), bei B einer. Hat der neue Krümmungseintrag Schranken (`MeniscusInitOptions::curvature_variables`, CLI `--curvature-bound`), wird c₀ auf [min + δ, max − δ] begrenzt (bei einer Spanne unter 2δ auf [min, max]) und jeder Sattelstart zusätzlich auf [min, max] (ab M6-T6): c₀ kommt von einer vorhandenen Fläche, die auf ihrer eigenen, weiteren Schranke liegen kann, und ein Start außerhalb der Schranken brach den Lauf mit `FormatError` ab (beobachtet bei c₀ = −1,19999/mm, Schranke −1,2/mm). Bei Dicke 0 ändert c₀ die Merit nicht.
3. Innere Homotopie jedes Zweigs (nur A): Mittendicke t = t_E · 2^{−(J−1)}, …, t_E/2, t_E (Standard t_E = 0,5 mm, J = 5), jede Stufe eine Optimierung am selben Punkt; der Luftspalt hinter dem Meniskus wird um t kürzer, aber nicht kürzer als `min_air_gap` (Standard 0,1 mm) bzw. die untere Schranke seiner Dickenvariable – sonst wird das System länger; Geisterplatten dieses Spalts bleiben im Spalt (Abstand höchstens die Spaltdicke).
4. Bester Zweig: endliche Merit, dann die wenigsten versagten Strahlen, dann das kleinste F, bei Gleichstand der kleinere Index. Der Controller verfolgt **einen** Zweig; mehrere Zweige verfolgt die Strahlsuche ([Abschnitt 8](#festlegungen-der-implementierung-m6-t4), M6-T4).
5. Abschluss (`finish_insertion`): bei A wird die Mittendicke Variable in [t_E, `element_max_thickness`] (Standard 5 mm; ≤ t_E: bleibt fest), mechanische Einträge mit Standardauswahl (bzw. `"all"`) gelten wieder für alle Flächen einschließlich der neuen (`renumber_surfaces` hatte sie auf die alten Flächen festgelegt, damit das Einfügen die Merit nicht ändert; Option `constrain_new_elements`).
6. Penalty-Fortsetzung am Punkt (unten); der neue Zustand ist der gute Zustand.

**Generische Fortsetzungen.**

- *Penalty-Fortsetzung* der einseitigen Bedingungen (Abschnitte 2.3–2.5) am Start und nach jedem Einfügen: s = 10^{−(K−1)}, …, 1/10, 1 (K = `penalty_stages` = 3, Faktor `penalty_factor` = 10), jede Stufe ab dem Ergebnis der vorigen; ohne einseitige Bedingungen eine Optimierung mit s = 1. Während der stetigen Schritte bleibt s = 1: Ein kleiner Schritt verletzt die Grenzen nur wenig. Zusammen mit dem Vorhalt (oben) ist das die Lehre aus M4-T3, M4-T4 und M6-T1.
- *Augmented Lagrangian* statt Penalty-Fortsetzung (ab M11-T2, Option `augmented_lagrangian`, Standard aus): fester Faktor `al_penalty_scale` auf die Gewichte, am Start und nach jedem Einfügen bis zu `rounds` Optimierungen bis max √w|Δθ| ≤ `tolerance`, Fortschreibung der Verschiebungen nach jeder Optimierung (Abschnitt 2.6).
- *ε-Fortsetzung* der L1-Glättung am Endpunkt (Abschnitt 3): `smoothing_stages` = 1 Stufe mit ε × 0,1 (10⁻⁶ → 10⁻⁷ mm); die Restkoeffizienten |a| ≈ ε|g|/(μw) werden entsprechend kleiner.
- *μ-Fortsetzung* am Endpunkt, nur wenn das Ziel verfehlt ist: höchstens `ghost_weight_stages` = 2 Stufen mit μ × 10; danach übernimmt das reale System die Korrektur.

**Ablauf** (`run_homotopy`, Reihenfolge des Standardablaufs oben):

1. Start: Penalty-Fortsetzung; ist der Startpunkt nicht annehmbar, Einfügen nach Diagnose, bis er es ist (ohne Bedarf: `start_rejected`).
2. Apertur bis `target_aperture`, dann Feldanteil bis `target_field` (Schrittweitensteuerung), dann Spektrumstufen bis `target_spectrum`.
3. Abschluss: ε-Fortsetzung, **Zielprüfung am realen System** (alle Geisterkoeffizienten 0, ohne Vorhalt, hartes Versagen wie die Auswertung, gleiche Quadratur): RMS-Wellenfrontfehler ≤ σ_Ziel für jedes Tupel (Feld, Wellenlänge) mit Gewicht > 0 und kein versagter Strahl. Verfehlt: μ-Fortsetzung mit Prüfung nach jeder Stufe; weiter verfehlt: Diagnose am Zustand vor dem Abschluss, Einfügen dort und erneuter Abschluss, ohne Bedarf `goal_not_reached`.

Abbruch: Ziel erreicht (`goal_reached`), höchste Elementzahl (`max_elements`, Standard 6 Glaselemente, ein Kittglied zählt zwei), Δ_min ohne Ausweg (`step_too_small`), Zahl der Optimierungen (`max_steps` = 400). Die Meldung (`HomotopyResult::message`) nennt Parameter, letzte F_B und den Grund.

**Protokoll.** Ein Eintrag pro Optimierung, angewandtem Operator und Zielprüfung (`HomotopyStep`): Nummer, Art (`start`, `penalty`, `aperture`, `field`, `spectrum`, `insertion`, `insertion_stage`, `smoothing`, `ghost_weight`, `goal`), Punkt (alle Parameter), Δ und Δ danach, Zweig und Stufe der inneren Homotopie, LM-Status, Zyklen, F am Start und Ende, F_W, F_G, RMS, Σ|a|, versagte Strahlen, Annahme und Entscheidung, Empfehlung der Diagnose, ausgeführter Operator (Text), Elementzahl, Zielprüfung pro Tupel, Laufzeit. Ausgabe als Struktur, Text (`write_homotopy_step_line`, `write_homotopy_summary`) und JSON (`homotopy_result_json`); Rückruf `on_step` nach jedem Eintrag. **Determinismus:** Die Steuerlogik ist eine reine Funktion der Ergebnisse des Modells; das optische Modell ist mit dem CPU-Optimierer bitgleich reproduzierbar (Test `HomotopySmoke.DeterministicCpu`: zwei Läufe mit gleichem Protokoll und gleicher Endbeschreibung bis aufs Bit). Nicht deterministisch sind nur die Laufzeiten.

**Abnahme** „Ausgehend von einer Linse bei NA 0,3, monochromatisch, entsteht automatisch ein beugungsbegrenztes System bei NA 0,6 auf der Achse“ (`HomotopyOptimization.DISABLED_ApertureToNa06Cpu`, GPU `CudaHomotopy.DISABLED_ApertureToNa06Gpu`, beide nur lokal: Laufzeit ~20 s bzw. ~35 s in Release, Minuten im Debug-Build; in der CI läuft der Kurzlauf `HomotopySmoke.*`). Startsystem frei erfunden, `data/systems/einzellinse_homotopie_start.json`: Einzellinse aus einem hochbrechenden Glas (Buchdahl-Koordinaten des 3-Term-Fits des frei erfundenen Testglases SYN-36, n = 1,807; Buchdahl statt Katalogname, damit die Datei nur Gläser aus `basis.json` nennt), f = 10 mm, Pupillenradius 3 mm (NA 0,3), Achse, 546,07 nm; Krümmungen und Arbeitsabstand variabel, Brennweite als Gleichung; Geisterplatten N-BK7 und das hochbrechende Glas (K = 6) im kollimierten Bündel vor der Linse und im konvergenten Bündel dahinter, μ = 47,2 mit Ordnungsfaktor κ = 1 (Aktivierung bei σ_Ziel/2 pro Modus für das hochbrechende Glas; mit κ = 2 und μ aus σ_Ziel blieben die Restfehler höherer Ordnung – hier dominiert Ordnung 4 – unterhalb der Schwellen σ√((2k+1)/5)·2^{k−2}, der Bedarf zeigte sich nicht, in einem Vorversuch endete der Lauf bei NA 0,32 mit RMS 0,13 Wellen ohne aktive Platte); Glasrand ≥ 0,3 mm bei der freien Öffnung plus 0,2 mm, Luftrand ≥ 0,1 mm, Luftspalt ≥ 0,1 mm (w = 10⁴/mm²). Controller mit den Standardwerten, LM höchstens 500 Zyklen pro Schritt, Schranken ±1/mm der neuen Krümmungen.

| Kriterium | CPU | GPU (RTX 3090) |
|---|---|---|
| Ergebnis | `goal_reached` bei NA 0,6 | gleich |
| RMS des realen Systems (Geisterkoeffizienten 0) | 0,01633 Wellen (Ziel 1/14 = 0,0714); `mikroptim trace` der Ergebnisdatei (Restkoeffizienten der Geisterplatten ~10⁻⁸ mm): 0,016320 | 0,01566 Wellen |
| automatische Schritte | 63 Optimierungen: Penalty-Fortsetzung, zwei Menisken am Start (Einzellinse mit Geisterplatten F_B = 0,110 > F_max = 5,1·10⁻³), 20 Aperturschritte (3 Halbierungen), dritter Meniskus bei NA 0,4425 (Versuch 0,445: F_B = 5,13·10⁻³ mit F_G = 2,5·10⁻³, Empfehlung A hinter der Linse, dominante Ordnung 4), ε-Stufe, Zielprüfung | dieselben Entscheidungen in derselben Reihenfolge |
| Elemente | 1 → 4 (Meniskus N-BK7 1,34 mm, hochbrechender Meniskus 1,15 mm, Linse, hochbrechender Meniskus 0,5 mm) | gleich |
| Mechanik am Ende | alle Grenzen eingehalten (Glasränder der beiden vorderen Menisken auf der Grenze 0,3001 mm, Vorhalt) | – |
| Laufzeit | 19–23 s | 35 s |

Beobachtungen: (1) Die LM-Läufe enden nach dem ersten Einfügen meist mit `max_iterations`: Die aktive Randdicken-Bedingung mit h aus den Strahlhöhen ist ein Maximum über Strahlen und über die beiden Flächen eines Meniskus, die im fast kollimierten Bündel gleich hohe Strahlen sehen (Wechsel des Trägers, Abschnitt 2.5); vermutlich kriecht LM entlang dieses Knicks (Kontrolle: dasselbe System ohne die Glasrand-Bedingung konvergiert in 914 Zyklen). Deshalb ist kein Aperturschritt „schnell“, Δ wächst in der Abnahme nicht (die Vergrößerung zeigen die Tests mit synthetischem Modell). (2) Beim ersten Einfügen enden beide Sattelzweige im selben Minimum, beim zweiten (F 2,8·10⁻⁴ gegen 1,9·10⁻²) und dritten (2,6·10⁻³ gegen 3,7·10⁻³) liegen sie deutlich auseinander – Hinweis für M6-T4: Die Zweige lohnen; gemeinsam im gebündelten LM gerechnet, halbiert sich die Zahl der Aufrufe. (3) Die Annahme eines nicht konvergierten Schritts („keep“) ist bewusst: Der Fortschritt bemisst sich an F_B, nicht an der Konvergenz.

**Tests** (`tests/test_homotopy.cpp`, synthetisches Modell): Vergrößern nach schnellen Schritten und Kürzen auf das Ziel; Halbieren mit Neustart am letzten guten Zustand (nicht am abgelehnten Ergebnis); Einfügen bei Δ < Δ_min mit Bedarf (Operator, zwei Zweige × zwei Stufen, besserer Zweig, Penalty-Stufen, danach Δ₀); Abbruch ohne Bedarf, bei der höchsten Elementzahl, bei `max_steps`, Start nicht annehmbar; Penalty-, ε- und μ-Fortsetzung mit genauer Folge der Schritte; Einfügen am Ende bei verfehltem Ziel; Feld- und Spektrumphase; Optionen; Protokoll als Text und JSON. Optisches Modell: `realize` (Pupille, Normierungsradien, Feld, Spektrum, Gewichte, Vorhalt), Operator A aus einer Diagnose mit vorgegebenen Koeffizienten (Merit unverändert innerhalb `kInsertionMeritRelative`, Stufen, Spalt, Mindestspalt, Dickenvariable, Nebenbedingungen auf neuen Flächen, Sattelzweige), Zielprüfung am realen System. Kurzlauf `HomotopySmoke.ShortRunCpu` (NA 0,3 → 0,31, alle Phasen, ein Einfügen, Abbruch `max_elements`; < 1 s in Release), `HomotopySmoke.DeterministicCpu`, GPU `CudaHomotopy.ShortRunMatchesCpu` (gleiche Entscheidungen; F bis zum ersten Einfügen relativ ≤ `kHomotopyCpuGpuMerit` = 10⁻⁶, danach laufen die 40-Zyklen-Optimierungen des Kurzlaufs unkonvergiert und weichen um bis zu ~1 % ab).

### Gemeinsame Apertur-/Feld-Homotopie und Feldregel (M6-T8)

Anlass: Befund F14 aus M6-T6b ([Bericht](ergebnisse/m6-t6b-zielspezifikation-adr0017.md)) – ab der Einzellinse endet die Feldstufe immer bei 27 % Feld mit „keine Geisterplatte aktiv“. Diagnose und Messung: [Bericht M6-T8](ergebnisse/m6-t8-apertur-feld.md). Beide Optionen sind standardmäßig aus; ohne sie sind Protokoll und Ergebnis bitgleich zu vorher (Tests `HomotopyControl.JointApertureFieldOffOrSingleParameterUnchanged`, CLI `CliSynth.JointApertureFieldAndFieldRule`).

**Gemeinsamer Parameter** (`HomotopyOptions::joint_aperture_field`, CLI `--joint-aperture-field`): Unterscheiden sich Apertur und Feldanteil zwischen Start und Ziel, ersetzt eine Phase (Schrittart `aperture_field`) die Apertur- und die Feldphase. Ein Parameter t ∈ [0, 1] mit der Schrittweitensteuerung oben (`joint_step`, Standard Δ₀ = 1/32, Δ_min = 1/256, Δ_max = 1/4, CLI `--joint-step`) bestimmt den Punkt

$$\mathrm{NA}(t) = \mathrm{NA}_0 + t\,(\mathrm{NA}_1 - \mathrm{NA}_0),\qquad \varphi(t) = \varphi_0 + t^{p}\,(\varphi_1 - \varphi_0),$$

p = `joint_field_exponent` (Standard 1: Diagonale im (NA, φ)-Raum; p < 1: das Feld läuft voraus; CLI `--joint-field-exponent`); bei t = 1 genau der Endpunkt. Ablehnung, Halbieren, Einfügen und Prädiktor wie in den Einzelphasen, nur in t. Nicht mit der Strahlsuche. Ändert sich nur ein Parameter, laufen die bisherigen Phasen.

**Feldregel** (`OpticalHomotopyOptions::field_rule`, CLI `--field-rule`): Meldet die Diagnose der Geisterplatten keinen Bedarf (die Platten sehen Feldfehler kaum, siehe Bericht) und dominieren Feldfehler – kollimierte Quelle, mindestens zwei Felder mit Gewicht > 0, größter RMS (getracte Wellenfront am Punkt) ≥ `field_min_rms` (0,05 Wellen) und größter RMS am größten Feldwinkel ≥ `field_ratio` (1,5) × der am kleinsten –, empfiehlt `diagnose` Operator A in der Feldlinsenlage: im Luftspalt (Dicke ≥ 2·`min_air_gap`) mit dem größten Verhältnis |ȳ|/|y| von paraxialem Haupt- zu Randstrahl (Mittel der beiden Spaltflächen, Aberrationsbericht M10-T1), Vorderfläche in der Spaltmitte, c₀ = Krümmung der Fläche hinter dem Spalt, Glas `field_glass` (CLI `--field-glass`) bzw. das der Linse vor dem Spalt (sonst dahinter). Danach wie Operator A aus der Diagnose (innere Homotopie über die Dicke, Sattelzweige, Mittendicke als Variable); kein Strafgewicht wird erhöht (keine Quellplatte). Die Empfehlung nennt die Seidel-Summen W131, W222, W220 und das Verhältnis; das Protokoll markiert den Operator mit „(Feldregel)“. Die Mindestdicke des Spalts verhindert, dass die Regel unmittelbar hinter einem eben eingefügten Meniskus (Spalt `min_air_gap`) einfügt: Im Vorversuch kollidierten dort die Ränder (F = 25 Wellen² aus der Randdicken-Strafe).

**Einfügen nur mit Verbesserung** (`HomotopyOptions::insertion_must_improve`, `insertion_attempts`, CLI `--insertion-improve N`; Befund F16): In den stetigen Phasen (Apertur, Feld, gemeinsam, Spektrumbreite) wird ein Einfügen nur übernommen, wenn F_B nach der Penalty-Fortsetzung kleiner ist als F_B des guten Zustands davor (und kein Strahl versagt). Sonst wird der Zustand zurückgesetzt, ein Vermerk (Schrittart `insertion`, nicht angenommen, „Einfügen verworfen“) protokolliert und die nächste Variante aus `HomotopyModel::insertion_options` versucht (Geisterbedarf: andere Lagen bzw. Gläser wie in der Strahlsuche; Feldregel: die übrigen Spalte nach |ȳ/y|), höchstens N. Verbessert keine, endet der Lauf mit `step_too_small` („kein Einfügen verbessert F_B“). Anlass: In den ersten Messläufen machte das Einfügen den guten Zustand oft schlechter (F_B danach über F_max), der Controller lief dann von einem nicht annehmbaren Zustand aus in Einfüge-Serien. Spektrum- und Abschlussphase bleiben unverändert; nicht mit der Strahlsuche.

**Nachtrag M6-T11b – auch in der Startphase, mit Spielraum** (Freigabe Maintainer 2026-11-22): Mit derselben Option prüft der Controller auch die Einfügungen der Startphase (solange der Startpunkt nicht annehmbar ist). Weil F_B dort definitionsgemäß über F_max liegt (oder Strahlen versagen), gilt ein Spielraum: Ein Einfügen wird übernommen, wenn F_B danach endlich ist, kein Strahl versagt und F_B danach < F_B davor **oder** F_B danach ≤ F_max ist (der zweite Fall greift nach einem Start mit versagten Strahlen und kleinem F_B). Sonst Vermerk „Einfügen verworfen: F_B danach … ≥ … davor und > F_max = … (…, Startphase)“ und die nächste Variante; verbessert keine, endet der Lauf mit `start_rejected` („kein Einfügen verbessert F_B (… alle > F_max = …)“). Anlass: In M6-T11, Lauf F7, stieg F_B beim Einfügen in der Startphase von 0,011 auf 0,046, der Controller fügte weiter ein, bis das Budget erreicht war. Ohne `--insertion-improve` ist die Startphase unverändert (bitgleich). Tests: `HomotopyControl.InsertionMustImproveInStartPhase` (Orakel: Sprung F_B 2 → 4,5 F_max verworfen, Variante 1 mit 0,45 F_max übernommen; ohne Option wird der Sprung übernommen; verbessert keine, `start_rejected`), `…InsertionInStartPhaseAcceptedBelowThreshold` (Spielraum: versagter Strahl bei F_B 0,2 F_max, danach 0,45 F_max übernommen).

Tests (`tests/test_homotopy.cpp`): `HomotopyControl.InsertionMustImprove` (Variante 0 verworfen, Variante 1 übernommen; ohne Option Variante 0; keine verbessert: Abbruch im Zustand davor), `HomotopyControl.JointApertureFieldDiagonal` (Punkte auf der Diagonale, Schrittweite, Endpunkt genau), `…Exponent` (φ = √t), `…InsertsOnPath` (Halbieren und Einfügen am letzten guten Punkt der Bahn), `…OffOrSingleParameterUnchanged`, `…InvalidOptions`; `OpticalHomotopy.FieldRuleRecommendsMeniscusAtFieldPosition` (Mehrlinser ohne Geisterplatten: Lage gegen den Aberrationsbericht als Orakel, Glas, Merit nach dem Einfügen innerhalb `kInsertionMeritRelative`, kein Bedarf oberhalb des Verhältnisses).

### Rückbau und Elementbudget (M6-T9)

Anlass: Befund F16 aus M6-T6b und der 25-Linser der Feldregel aus M6-T8 ([Bericht](ergebnisse/m6-t8-apertur-feld.md)) – die Feldregel öffnet das Feld, fügt aber Elemente ein, die am Ende nichts mehr tragen. Messung: [Bericht M6-T9](ergebnisse/m6-t9-rueckbau.md). Beide Optionen sind standardmäßig aus; ohne sie sind Protokoll und Ergebnis bitgleich zu vorher (JSON-Felder `removals`, `removal_attempts` und `options.prune` nur mit Option).

```cpp
struct HomotopyRemoval { int first_surface; int elements; double score; std::string text; };
// HomotopyModel (Standards: keine Kandidaten, collapse/remove werfen)
virtual std::vector<HomotopyRemoval> removal_candidates(const SystemDescription&, const HomotopyPoint&);
virtual SystemDescription collapse(const SystemDescription& before, const SystemDescription& current,
                                   const HomotopyRemoval&, double fraction, std::vector<std::string>& warnings);
virtual SystemDescription remove(const SystemDescription& before, const SystemDescription& collapsed,
                                 const HomotopyRemoval&, std::vector<std::string>& warnings);
```

**Ein Rückbau-Versuch** am guten Zustand (Punkt bleibt, Faktor der einseitigen Bedingungen 1): Rückbau-Homotopie mit K = `prune_stages` Stufen (Standard 3, CLI `--prune-stages`), Anteile 1/K, 1/(K − 1), …, 1 des verbleibenden Wegs zum Null-Element (`collapse_element`, [Abschnitt 13](#13-operatoren-der-entwurfssteuerung-m10-t4)), nach jeder Stufe eine Optimierung (Schrittart `removal_stage`, nicht angenommen), dann `remove_element` des Null-Elements (Merit unverändert bis auf Rundung, keine weitere Optimierung). Übernommen wird das Ergebnis, wenn es annehmbar ist (endlich, kein versagter Strahl, nicht `NumericalError`) und

$$F_B \le \rho\,F_{\max},\qquad 0 < \rho = \texttt{prune\_merit\_factor} \le 1,$$

mit ε = `prune_tolerance` ≥ 0 zusätzlich F_B ≤ (1 + ε)·F_B vor dem Rückbau (Standard aus; CLI `--prune-factor`, `--prune-tolerance`). Sonst bleibt der Zustand unverändert. Ein Vermerk (Schrittart `removal`, ohne Optimierung) nennt Kandidat, F_B und Grenze bzw. den Grund der Ablehnung und angehobene Schranken (Warnungen von `collapse_element`/`remove_element`). ρ < 1 lässt Abstand zur Schwelle, die das System ohne Geisterplatten erst nach der Feinoptimierung einhalten muss.

**Runden.** Pro Runde fragt der Controller die Kandidaten des Modells ab und versucht höchstens `prune_attempts` (Standard 4, CLI `--prune-attempts`) in deren Reihenfolge; der erste Erfolg beendet die Runde, und eine neue beginnt am kleineren System (neue Rangfolge). Eine Runde ohne Erfolg beendet den Rückbau. Nie unter `prune_min_elements` Glaselemente (Standard 1; ein Kittglied mit k Gläsern zählt k).

**Wann** (`HomotopyOptions::prune`, CLI `--prune`): nach jeder stetigen Phase (gemeinsam, Apertur, Feld, Spektrumbreite) und der Spektrumphase, wenn in ihr eingefügt wurde, und vor dem Abschluss (ε-Fortsetzung, Ziel) sowie am letzten guten Zustand eines abgebrochenen Laufs (`step_too_small`, `max_elements`, …; Vermerk „am Ende“, der Status bleibt), jeweils wenn seit dem letzten Rückbau eingefügt wurde oder noch keiner lief. Ein Lauf ohne Phasen (Ziel = Start) baut also nur zurück und prüft dann das Ziel.

**Elementbudget** (`prune_at_limit`, CLI `--element-budget N` = `--max-elements N` mit Rückbau): Verlangt ein Einfügen mehr als `max_elements` Elemente, läuft zuerst eine Rückbau-Runde (Vermerk „Elementbudget N“), je Punkt der Homotopie höchstens einmal (sonst könnte Einfügen und Entfernen am selben Punkt wechseln). Entfernt sie etwas, wird nicht eingefügt; die Phase versucht ab dem kleineren System neu (Δ₀) und fügt bei erneutem Bedarf ein. Entfernt sie nichts, Abbruch `max_elements` wie bisher.

**Kandidaten des optischen Modells.** Alle Glasblöcke (Einzellinsen, Kittglieder: Luft davor, Glas, Luft dahinter), geordnet nach dem Betrag ihres Beitrags zu den Seidel-Summen am Punkt: Σ über W040, W131, W222, W220, W311, ΔW020, ΔW111 von |Summe der Flächenbeiträge des Blocks| in Wellen bei λ_ref (Aberrationsbericht M10-T1, paraxial, realisierte Beschreibung mit Pupille und Feld des Punkts), kleinster zuerst; ohne Bericht (Punktquelle) in Flächenreihenfolge. Ein Null-Element hat den Beitrag 0 (die Beiträge seiner Flächen heben sich auf). Die Rangfolge ist eine Heuristik (ein Element mit kleinem Nettobeitrag kann für höhere Ordnungen nötig sein); die Annahme entscheidet allein F_B nach der Nachoptimierung.

**Mechanische Bedingungen während der Rückbau-Homotopie.** Mit Glasdicke gegen 0 verletzte z. B. die Randdicke der schrumpfenden Linse ihre Grenze (w = 10⁴/mm²: 0,3 mm kosten 900 Wellen²), und mit Strahlhöhen hängt die Strafe von den übrigen Variablen ab. `collapse` nimmt deshalb die Flächen bzw. Spalte des Blocks aus allen mechanischen Einträgen (Liste ohne sie; bleibt nichts, Gewicht 0); `remove` stellt danach Gewichte und Standardauswahl der Einträge aus dem Zustand vor dem Rückbau wieder her (sonst schränkten sie später eingefügte Elemente nicht mehr ein, `finish_insertion`). Die Zuordnung der Einträge folgt der Abbildung von `remove_element` (mechanische Einträge ohne verbleibende Fläche entfallen).

Tests (`tests/test_homotopy.cpp`, `tests/test_homotopy_optimization.cpp`, `tests/test_cli_synth.cpp`): `HomotopyControl.PruneRemovesSuperfluousKeepsNecessary` (Mock: notwendiges Element verworfen und Zustand zurück, überflüssiges entfernt, Anteile 1/3, 1/2, 1, Versuche pro Runde, Mindestzahl, ρ, ε, Text und JSON), `…PruneAfterPhaseWithInsertions` (auch nach einem Abbruch), `…ElementBudgetPrunesBeforeInserting` (Rückbau statt Abbruch, danach Einfügen; ohne entfernbares Element einmal versucht, dann `max_elements`), `…PruneOffUnchangedAndInvalidOptions` (Protokoll bis auf die Laufzeit gleich, kein Aufruf des Modells); `OpticalHomotopy.RemovalOfNullElementRestoresSystem` (Orakel: Null-Meniskus hat Beitrag 0 und ist erster Kandidat, Merit nach jeder Stufe und nach dem Entfernen innerhalb `kInsertionMeritRelative`, Flächen, Platten und Nebenbedingungen wie vor dem Einfügen); `HomotopySmoke.PruneRemovesNullElementKeepsLens` (CPU-Lauf: Null-Meniskus entfernt, die Einzellinse versucht und verworfen); `CliSynth.PruneAndElementBudget`.

### Rückkehr zu Geisterplatten (M6-T10)

Anlass: Befund F17 aus M6-T9 ([Bericht](ergebnisse/m6-t9-rueckbau.md)) – der Rückbau wirkt am realen System (ohne Geisterplatten) am Ende einer Stufe, mit Geisterplatten kaum. Der Ablauf „Stufe → reales System nachoptimieren → Rückbau → nächste Stufe“ braucht deshalb den Weg zurück vom realen System zu einem Startsystem der Synthese mit Geisterplatten. Messung: [Bericht M6-T10](ergebnisse/m6-t10-zyklisch.md).

```cpp
enum class GhostReturnLocation { Stop, Aperture, Field, Front };
struct GhostReturnOptions { std::vector<MaterialSpec> glasses; std::vector<GhostReturnLocation> locations;
                            int order; double penalty_weight, order_weight, order_factor, smoothing;
                            double min_gap; bool fix_plate_gaps; double norm_margin, norm_field_scale;
                            int rings, spokes; };
GhostReturnResult return_to_ghost_plates(const LoadedSystem& real, const GhostReturnOptions&);
```

API: `include/mikroptim/synth/ghost_return.hpp` (Bibliothek `Mikroptim::synth`), Kommandozeile `mikroptim insert --ghost-return` und `mikroptim synth --start-from-real` ([cli.md](cli.md#mikroptim-synth)). Das System darf noch keine Geisterplatten haben.

1. **Lagen** (Luftspalte des realen Systems, Spaltmitte; paraxial bei λ_ref und dem größten Feldwinkel der Datei, `compute_aberrations` wie die Feldregel): `aperture` = innerer Luftspalt (nach Fläche 1 … S − 2, Dicke ≥ `min_gap` = 0,2 mm) mit der größten Randstrahlhöhe |y| (Mittel der beiden Spaltflächen; sphärische Aberration, Farblängsfehler), `field` = innerer Luftspalt mit dem größten |ȳ/y| (Feldlinsenlage, Koma, Astigmatismus, Farbquerfehler; dieselbe Kennzahl wie die Feldregel), `front` = letzter innerer Luftspalt (vor der Frontlinse am Objekt), `stop` = Blendenspalt (kollimiertes Bündel vor der ersten Linse). Standard `aperture,field` (Abschnitt 5, „Platzierung“: eine Lage nahe den feldunabhängigen Fehlern, eine bei großem ȳ/y). Fallen zwei Lagen auf denselben Spalt, gilt die erste (Hinweis im Bericht). Der Arbeitsabstand ist keine Lage (`place_ghost_dictionary` setzt keine Platten hinter die letzte Fläche).
2. **Strafe:** Hat das System keinen Eintrag `ghost_penalty`, kommt vor dem Einsetzen einer mit `"all"` dazu, mit den Werten des Startsystems der Zielspezifikation (μ = 47,2 Wellen²/mm, w₀ = 1, κ = 1, ε = 10⁻⁶ mm); ein vorhandener Eintrag deckt die Platten nach den Regeln von `place_ghost_dictionary` ab.
3. **Wörterbücher:** `place_ghost_dictionary` je Lage (eine Platte pro Glas, Ordnung K = 10, Koeffizienten 0, Variablen aller Ordnungen).
4. **Spalte fest** (`fix_plate_gaps`, Standard an): Dickenvariablen der Plattenspalte werden entfernt (Einträge mit `"all"` werden zur Liste der übrigen Spalte), wie im Startsystem und im Skript (`remove_gap_variables_in_plate_gaps`): Operator A teilt genau diese Spalte, und eine geerbte untere Schranke kann das Einfügen verhindern (Abschnitt 6, `meniscus_from_diagnosis`); außerdem bliebe der Abstand der Platten vom Scheitel bei kleinerem Spalt nicht im Spalt. Nach dem nächsten Nachoptimieren ohne Platten (Z2) gibt das Skript die Spalte wieder frei.
5. **Normierungsradius** jeder Lage: r_g = 1,02 × größte reale Strahlhöhe an den Platten über alle Felder (Feldwinkel × `norm_field_scale`), Wellenlängen und Stützstellen (`surface_extrema_cpu`, hartes Versagen, Quadratur 16 × 8; ohne gültigen Strahl der größere Halbdurchmesser der Spaltflächen). So deckt das Profil das Bündel ab, ohne es wie in M6-T8 (Befund F14, Normierungsradius 6× zu groß) zu verdünnen. `synth --start-from-real` rechnet mit dem größten Feldanteil des Laufs (max(Start-, Zielfeld)), das Skript mit vollem Feld.

**Merit-Invarianz.** Platten mit a = 0 brechen nicht und haben keine Strafe; Variablen entfernen und r_g setzen ändert F nicht. Gemessen (`GhostReturn.MeritUnchangedAndPlatesAtExpectedGaps`, Mehrlinser mit drei Feldern und vier Wellenlängen, zwei bzw. vier Lagen): größte relative Änderung 1,1·10⁻¹⁵ (Toleranz `kInsertionMeritRelative`). Die Platten mit a = 0 rechnen d′ = n d/n′ mit n = n′, das ist bis auf Rundung die Identität; an sehr gut korrigierten Systemen ist der Rundungsboden relativ zu F größer (Abschnitt 6): beim 9-Linser aus M6-T9 (F = 1,0·10⁻³ Wellen²) 1,1·10⁻¹⁰, beim 5-Linser 5·10⁻¹¹ relativ, also |ΔF| ≈ 10⁻¹³ Wellen² (Bericht der Funktion `merit_before`/`merit_after`, CPU wie `mikroptim insert`).

**Zyklischer Ablauf** (Skript `tools/m6t6_zielspezifikation.py --zyklus feld,farbe,apertur`): Für jede Stufe der Liste, deren Ziel noch nicht erreicht ist (Feld 100 %, Spektrum 436–656 nm, NA 0,95): reales System mit vollem Feld → `insert --ghost-return` → Stufe wie im Hauptlauf (`synth_loop` mit Neustarts, erzwungenem Einfügen; Farbe: zwei Wellenlängen, dann fünf, Operator B) → Z2 (ohne Platten nachoptimiert, 2.000 Zyklen, Spektrum der Stufe) → Rückbau am realen System (`synth --prune` am erreichten Punkt, ρ = 0,5, 30 Versuche, höchstens die Elementzahl von Z2) → Z2 nach dem Rückbau, bewertet; die nächste Stufe beginnt dort. Die Liste wird bis zu `--zyklus-wiederholen` Mal durchlaufen, solange ein Durchlauf Feld, NA oder Spektrum öffnet; eine Stufe ohne Fortschritt wird erst wieder versucht, wenn eine andere etwas geöffnet hat. Messung: Der Rückbau entfernt direkt nach einer Stufe nichts (0 von 195 Versuchen), weil die Stufe an der Schwelle F_B ≈ F_max endet; die zurückgekehrten Platten bleiben in monochromatischen Stufen passiv ([Bericht](ergebnisse/m6-t10-zyklisch.md)).

Tests: `tests/test_ghost_return.cpp` (`GhostReturn.MeritUnchangedAndPlatesAtExpectedGaps`: Lagen gegen den Aberrationsbericht als Orakel, Platten, Variablen, Strafe, Merit; `…NormRadiusFromRealRaysInCollimatedBeam`: im Blendenspalt a ρ_max ≤ h ≤ a bzw. ≤ a + d tan θ; `…ExistingPenaltyAndOrderOptions`; `…LocationsParseAndErrors`), `CliInsert.GhostReturnMatchesLibrary`, `CliSynth.StartFromReal` (auch: ohne Option kein Eintrag im Protokoll), Python `test_tool_m6t6.py::test_return_to_ghosts_command`, `::test_cycle_run_stops_at_goal_or_stall`, `::test_system_state_and_stage_done`.

### Farbstufe (M6-T11)

Anlass: In M6-T10 ([Bericht](ergebnisse/m6-t10-zyklisch.md)) scheiterten alle Farbstufen: Startpunkt nicht annehmbar, die Geisterplatten empfehlen Operator A, die Farbregel greift kaum. Messung und Diagnose: [Bericht M6-T11](ergebnisse/m6-t11-farbe.md). Alle Optionen waren standardmäßig aus. **Seit M6-T11b** (Freigabe Maintainer 2026-11-22; in keinem Lauf schlechter, an Achromaten deutlich besser, Bericht M6-T11, Abschnitt 2) sind der spektrale Rang und die Startradien des Glieds **Standard von `synth`**; `--no-split-spectrum` und `--no-split-member-start` schalten sie ab (dann bitgleich zum Stand vor M6-T11b). Sie wirken nur, wo die Glasvorauswahl benutzt wird: Operator B mit `--split-preselect` (bzw. `--split-achromat`, `--split-start-target`), die Farbregel (`split_for_color` wählt immer vor) und die Gläser des Kittglied-Moduls; Operator B aus der Geisterdiagnose ohne Vorauswahl (Glas der Platte) und Läufe ohne Farbregel bleiben unverändert. `--split-spectrum` und `--split-member-start` schalten wie bisher zusätzlich die Vorauswahl für B aus der Diagnose ein. Das Protokoll enthält `options.split.spectrum_rank` (Wellenlängen; `null` mit `--no-split-spectrum` oder monochromatischer Startdatei) und `options.split.member_start` seitdem immer; `options.color_rule_ext` nur mit Option. Die Bibliothek (`SplitInitOptions`) behält ihre Vorgaben (aus); die Kommandozeile setzt sie.

**Operator B mit spektralem Rang und Startradien** (`SplitInitOptions`, `include/mikroptim/optim/insertion.hpp`):

- `preselect_wavelengths_nm` (CLI `synth --split-spectrum`: Spektrum der Startdatei, also das Zielspektrum der Spektrum-Homotopie): Partner des Linsenglases nach dem spektralen Rest der Glasvorauswahl (M11-T5g, [numerik.md, Abschnitt 15](numerik.md#15-algebraische-glasvorauswahl-m11-t5)) statt nach dem Koeffizienten des sekundären Spektrums; gilt für die Farbregel (`split_for_color`) und mit `--split-preselect` für B aus der Diagnose. `SplitPlan::spectral_residual` und die Begründung nennen den Rest.
- `member_start` (CLI `--split-member-start`): weiterer Zweig des Einfügens mit den Startradien des ganzen Glieds (`apply_member_start`): dünne Brechkraft φ = (n_L − 1)(c₁ − c₂) der ungeteilten Linse bei λ_ref, Anteile aus der Vorauswahl, alle drei Krümmungen aus `thin_member_shape` (Biegung S_I = 0, flachere Lösung) mit der Randstrahlneigung u/y vor der Linse aus dem paraxialen Strahl des realisierten Systems (`incoming_slope`, neu: u_i = u₁ − Σ_{j<i} K_j statt kollimiert), freigegebenes Glas beim Partnerglas. Der Controller wählt zwischen den Zweigen (wie beim Achromat-Start M11-T5e). Die Formel nimmt die Linse dünn und in Luft an; in Systemen hoher NA ist der Zweig ein grober Start (Bericht).

**Farbbedarf erkennen (Kriterium d)** (`OpticalHomotopyOptions::color_chromatic_share` = S > 0, CLI `--color-chromatic S`, setzt `--color-rule`): Die Farbregel (M11-T5c) prüft zusätzlich zu (a)–(c), ob der chromatische Anteil aus dem Aberrationsbericht mindestens S × größter RMS (getracte Wellenfront am Punkt) beträgt:

$$\sigma_\mathrm{col}^2 = \Big(\frac{W_L}{4\sqrt3}\Big)^2 + \Big(\frac{\Delta W_{111}}{4}\Big)^2 + \Big(\frac{\Delta W_{040}}{2\sqrt{180}}\Big)^2,\qquad W_L = \frac{\Delta z\,\mathrm{NA}^2}{2\lambda_\mathrm{ref}}.$$

Δz ist die Spanne der paraxialen Fokuslagen (Fokusverlauf M10-T2) der Spektrallinien im Bereich des Spektrums einschließlich λ_ref, also Längsfarbe mit sekundärem Spektrum; jede Wellenlänge liegt höchstens W_L/2 von der besten gemeinsamen Ebene, und eine Defokussierung W ρ² hat ohne Kolben den RMS W/√12. ΔW111 (Querfarbe, Kippung mit RMS W/2) und ΔW040 (Sphärochromasie, ρ⁴ mit bester Defokussierung, RMS W/√180) sind die Seidel-Werte zwischen kürzester und längster Wellenlänge, jeweils halbiert für den Abstand zur Mitte. NA = Apertur des Punkts. Der Bericht kennt die Geisterplatten nicht: Farbe, die die Platten tragen (in M6-T10 der Normalfall, die Platten haben die Dispersion ihres Glases), zählt mit – sie ersetzt dort ein Kittglied. Kriterium (a) versagt genau dann, weil die Platten die Wellenlängen ausgleichen (RMS-Verhältnis ≈ 1), (b), weil |W040| bei hoher NA groß ist.

**Kittglied-Modul statt Meniskus** (`color_module` = `Doublet` bzw. `StrongDoublet`, CLI `--color-module doublet|strong`, `--color-module-always`, `--color-module-anomalous`, `--color-module-incidence W`): Dominiert die Farbe und lässt sich keine Linse teilen (in den Systemen der Zielspezifikation sind die meisten Linsen an der unteren Dickenschranke, Operator B braucht die doppelte Dicke) bzw. mit `always` immer, empfiehlt die Farbregel ein Null-Kittglied aus der Wissensbasis (M10-T5, `generic.null_doublet` bzw. `generic.strong_doublet`) – vor der Geisterdiagnose, also auch dann, wenn die Platten Operator A empfehlen. Sonst bleibt Operator B die Empfehlung und das Modul eine Variante (`insertion_options`, für `--insertion-improve`), umgekehrt ebenso.

1. *Lage:* innere Luftspalte (nicht der Arbeitsabstand) mit Dicke ≥ 2·`min_air_gap`, absteigend nach der mittleren Randstrahlhöhe |y| der Spaltflächen (Längsfarbe eines dünnen Elements ∝ y²φ/ν); Vorderfläche in der Spaltmitte.
2. *Gläser:* bestes Paar der Glasvorauswahl über den ganzen Katalog (Σ|φ|/φ ≤ `preselect_max_power_sum`, Rang wie oben nach spektralem Rest bzw. Koeffizient); mit `anomalous` mindestens ein Glas mit |ΔP_g,F| ≥ 0,01 (Glasregel `anomalous_pgf`), beim starken Kittglied die Paarregel `merte_pair` (|Δn_d| 0,05 … 0,15, |Δν_d| ≥ 15, |ΔP_g,F| ≤ 0,01; Schwellen aus `data/design/knowledge/glass_rules.json`, dort als Mikroptim-Annahme gekennzeichnet).
3. *Krümmung c₀ aller drei Flächen:* Null-Kittglied die der Fläche hinter dem Spalt (auf die Schranken der neuen Krümmungsvariablen begrenzt), starkes Kittglied so, dass der paraxiale Randstrahl (Höhe und Steigung in der Spaltmitte) die Fläche unter W (Standard 40°) trifft, exakter Schnittpunkt wie `incidence_angle_deg` in `mikroptim.design`, Halbkugelgrad ≤ 0,9, flachere Lösung; Spalte, in denen das nicht geht, entfallen.
4. *Einfügen:* `insert_null_doublet` (Merit unverändert bis auf Rundung, gemessen 3,7·10⁻¹⁴ relativ), beide Gläser fest aus der Vorauswahl (eine Glasvariable hätte bei Dicke 0 keinen Gradienten; im Lauf P schob der LM sie bis n ≤ 0), ein Zweig; innere Homotopie über die Dicke beider Teile (Stufen wie Operator A, `HomotopyInsertion::doublet`, der Spalt dahinter wird um 2t kürzer), danach Dickenvariable beider Teile in [`element_thickness`, `element_max_thickness`].

**Werfender Zweig** (Controller, ohne Option): Hat ein Einfügen mehrere Zweige und wirft die Optimierung eines Zweigs eine Ausnahme (Lauf F3: im Zweig „Startradien des Glieds“ lief die freie Glasvariable von Operator B auf n ≤ 0, der Paraxialtrace warf), wird nur dieser Zweig verworfen (Vermerk „Zweig verworfen: …“, Schrittart `insertion_stage`); werfen alle Zweige, geht die erste Ausnahme weiter. Ein einzelner Zweig wie bisher. Läufe ohne Ausnahme sind unverändert.

**Startpunkt der Farbstufe** (Skript, Diagnose im Bericht): Das Skript setzte in M6-T10 das Spektrum der Teilstufe (480/644 nm, dann 436–656 nm) direkt in die Startdatei; ein monochromatisch entworfenes System hat dort einen RMS von mehreren Wellen, also F_B ≫ F_max. Mit `--farb-stetig` (synth `--spectrum-continuous`) beginnt die Stufe monochromatisch bei λ_ref (F_B des monochromatischen Endstands, annehmbar) und öffnet die Spektrumbreite weich; `--farb-ziel voll` geht direkt zu 436–656 nm. Beim Neustart einer Runde setzt das Skript jetzt das Zielspektrum der Teilstufe wieder ein (synth schreibt das Spektrum des Endpunkts); Fortschritt der Runden = erreichte Spektrumbreite.

**Rückbau mit Abstand zur Schwelle** (Skript `--rueckbau-punkt Q`, Vorschlag aus M6-T10/F17): Der Rückbau im Zyklus läuft bei Q × Parameter der Stufe (Apertur: Pupillenradius Q·NA; Feld: Q·Feldanteil; Farbe: Spektrumbreite Q, Wellenlängen um λ_ref skaliert), wo F_B Abstand zu F_max hat. Hat er etwas entfernt, geht `synth` ohne Einfügen (höchstens die Elementzahl nach dem Rückbau) zurück zum erreichten Punkt (Farbe: direkt über Z2 am vollen Spektrum); erreicht er ihn nicht, bleibt der Stand vor dem Rückbau. Zusammenfassung `prune.point`, `prune.reopen`.

Tests (`tests/test_color_stage.cpp`, `tests/test_glass_preselection.cpp`, `tests/test_cli_synth.cpp`, Python `test_tool_m6t6.py`): `ColorStage.SplitForColorRanksBySpectralResidual` (Orakel Glasvorauswahl direkt), `…MemberStartIsThinAchromatAndCorrectsLongitudinalColor` (Krümmungen = `thin_member_shape`, dünne Brechkraft erhalten; Orakel: paraxiale Längsfarbe der N-BK7-Einzellinse 0,778 mm → 0,018 mm schon beim Einfügen, ohne Option ein Zweig wie bisher), `…ChromaticCriterionThreshold` (Schwelle exakt bei σ_col/RMS, Orakel aus Bericht und Wellenfront), `…NullDoubletModuleKeepsMeritAndGrows`, `…ModuleOnlyWhenNoLensCanBeSplit` (ohne Modul-Option bitgleiche Empfehlung), `…AnomalousModuleGlass`, `…StrongDoubletIncidence` (Orakel sin I = |y c|); `GlassPreselection.MemberShapeWithIncomingSlope` (Seidel-Summen des Glieds hinter einer dünnen Linse gegen den Aberrationsbericht); `CliSynth.ColorStageOptions`; `test_synth_options_color_stage`, `test_prune_point`, `test_reopen_after_prune`.

### Varianten der Farbregel (M6-T13)

Anlass: In den Öffnungsstufen ab den polychromatischen Systemen (Lauf F3 aus M6-T11) endeten die Läufe mit `--insertion-improve 4` mehrmals mit „kein Einfügen verbessert F_B (4 Varianten, F_B 0.00502288, 0.00502288, 0.00502288, 0.00502288 …)“: Die Varianten der Farbregel (Operator B an den übrigen Linsen der Kandidatenliste) ließen sich am Zustand nicht teilen (Linsen an der unteren Dickenschranke), und `insert` fiel auf die nächste teilbare Kandidatin zurück – die empfohlene Linse selbst. Viermal dieselbe Einfügung, je drei Optimierungen. Umgekehrt brach ein Lauf ab, wenn die Empfehlung am verworfenen Versuch (andere Dicken) entstanden war und sich am guten Zustand keine Linse teilen ließ („zu dünn für zwei Teile“), bevor eine andere Variante versucht wurde. Messung: [Bericht M6-T13](ergebnisse/m6-t13-farbe-frueh.md).

Option `OpticalHomotopyOptions::color_variants` (Bibliothek: Standard aus, ohne sie bitgleich; CLI `synth --color-variants` setzt `--color-rule`; Skript `--farb-varianten`). **Seit M6-T13b Standard von `synth`**, wirksam nur mit der Farbregel; `--no-color-variants` (Skript `--ohne-farb-varianten`) rechnet bitgleich wie vor M6-T13b. `insertion_options` liefert für eine Empfehlung der Farbregel:

1. die Empfehlung selbst, wenn sie am Zustand einfügbar ist (Modul, oder eine Linse der Kandidatenliste lässt sich teilen); sonst entfällt sie;
2. Operator B an jeder weiteren Linse der Kandidatenliste, die sich am Zustand teilen lässt und nicht die Linse der Empfehlung ist – genau diese Linse, ohne Rückfall auf andere Kandidaten (bei einem empfohlenen Modul zuerst die übrigen Spalte des Moduls, dann die Linsen);
3. die übrigen Spalte des Kittglied-Moduls;
4. die Empfehlung der Geisterdiagnose am Zustand (ohne Farbregel; „Variante (Geisterdiagnose)“), wenn eine Platte Bedarf meldet;
5. mit `--field-rule` die Empfehlung der Feldregel am Zustand („Variante (Feldregel)“),

höchstens `insertion_attempts` (bzw. `--alternatives` in der Strahlsuche). Jede Variante fügt an einer anderen Stelle oder mit einem anderen Operator ein. Im JSON `options.color_rule_ext.variants`, im Textkopf unter „Farbregel (M6-T11)“.

Tests: `ColorStage.ColorVariantsAreDistinctInsertions` (Dreilinser mit nicht teilbarer Mittellinse; Orakel: Fläche, an der `insert` tatsächlich einfügt – ohne Option mindestens eine Variante gleich der Empfehlung, mit Option alle verschieden, keine nicht teilbare Linse, Empfehlung einer nicht teilbaren Linse wirft bei `insert` und entfällt als Variante; mit aktiver Geisterplatte ist die letzte Variante gleich `diagnose` ohne Farbregel), `CliSynth.ColorStageOptions` (JSON und Text nur mit Option, Einzellinse gleiches Ergebnis).

### Querfarbe aus realen Hauptstrahlen (M6-T14)

Anlass (Bericht M6-T13, F20): `--color-module-lateral` griff in keinem Lauf. Ursache: Das Kriterium vergleicht Seidel-Summen am Punkt, |ΔW111|/4 > |ΔW020|/(2√3); bei den Systemen hoher NA ist die Seidel-Längsfarbe ΔW020 groß (≈ −0,5 Wellen bei F3Z), wird aber im realen System von Termen höherer Ordnung ausgeglichen (Achse ≤ 0,027 Wellen), während die reale Querfarbe am Feldrand (Spanne der Hauptstrahlhöhen 0,34 µm, ≈ 0,1 Wellen Kippung) die Seidel-Querfarbe übersteigt.

Option `OpticalHomotopyOptions::color_lateral_rays` (CLI `synth --color-lateral-rays`, verlangt `--color-module`; Skript `--farb-querfarbe-strahlen`; Standard aus, ohne sie bitgleich): Die Querfarbe gilt zusätzlich als überwiegend, wenn

$$\sigma_{\mathrm{lat}} = \frac{\Delta\bar y\;\mathrm{NA}}{4\,\lambda_{\mathrm{ref}}} > \max_\lambda \mathrm{RMS}(\text{kleinstes Feld}, \lambda)$$

mit der Spanne Δȳ = max_l ȳ_l − min_l ȳ_l der realen Hauptstrahlhöhen (`batch_chief_ray_height` bei den Wellenlängen des Spektrums mit Gewicht > 0) am größten Feldwinkel mit Gewicht > 0 und der NA des Punkts (σ_lat ist der RMS der Kippung der äußeren Wellenlängen gegen ihren mittleren Bildpunkt: Kippung mit Randwert a hat den RMS a/2, a = (Δȳ/2) NA/λ_ref). Dann wie `--color-module-lateral`: Kittglied-Modul zuerst, Spalt nach |y·ȳ|. Der Text der Empfehlung nennt Δȳ, σ_lat und den Achswert.

### Farbe früh im Hauptlauf (M6-T13)

Skript `tools/m6t6_zielspezifikation.py --farbe MODUS` (Ablauf der Farbe im Hauptlauf ab der Einzellinse):

| Modus | Ablauf | Werkzeuge |
|---|---|---|
| `spaet` | Stufen 1–3 monochromatisch (Apertur bei 10 % Feld, Feld, Apertur), Farbe (p4 480/644 nm, p5 436–656 nm) danach – Standard bis M6-T12, bitgleich | wie angegeben (`synth --stall 0` ohne `--farbregel`/`--aktuell`) |
| `frueh` | wie `--farbe-frueh`: Farbstufen nach der Feldstufe, dann Apertur polychromatisch | wie angegeben |
| `apertur` (**Standard**) | Stufe 1 monochromatisch (NA 0,3 → `--na-feld` bei 10 % Feld), dann **Farbstufe p1f** am erreichten Punkt bei 10 % Feld mit stetiger Spektrum-Homotopie ab λ_ref direkt zu 436–656 nm, dann Feld- und Aperturstufe am polychromatischen System; Z2 am Spektrum der Feldstufe | `COLOR_TOOLS` |
| `anfang` | 436–656 nm ab dem Start (wie `--farbe-von-anfang voll`), Z2 polychromatisch | `COLOR_TOOLS` |

`COLOR_TOOLS` (wo nicht anders angegeben): `--farbregel`, `--farb-stetig`, `--farb-ziel voll`, `--farb-modul doublet`, `--farb-chromatisch 0.5`, `--feldregel`, `--einfuegen-verbessern 4`, `--farb-budget 4` – die Einstellungen der Läufe F3 (M6-T11) und A1W4_farbe (M6-T12) – sowie aus M6-T13 `--farb-varianten`, `--wachsen-marge 0.8` und `--einfuegen-abschluss` (unten). Gemessen (Bericht): Ab der Einzellinse erreicht die Farbstufe p1f bei NA 0,5 und 10 % Feld mit 7 Elementen 436–656 nm beugungsbegrenzt (0,060/0,047/0,057 Wellen, Strehl ≥ 0,87); die polychromatische Feldstufe endet danach bei 19 % (F14). `anfang` lässt die Geisterplatten die Farbe tragen, das reale System verfehlt das Ziel pro Tupel, und der Abschluss zerstörte es durch Einfügungen (Lauf H2). Die Farbstufe p1f schreibt beim Neustart einer Runde wieder das Zielspektrum (wie der Zyklus), ihr Fortschritt ist die erreichte Spektrumbreite. Die alten Schalter `--farbe-frueh` und `--farbe-von-anfang` laufen allein unverändert; zusammen mit `--farbe` sind sie ein Fehler, ebenso `--farbe apertur` mit `--gemeinsam`. Läufe aus Berichten vor M6-T13 ohne diese Schalter rechnen mit `--farbe spaet` nach. Begründung der Wahl und Messung: [Bericht M6-T13](ergebnisse/m6-t13-farbe-frueh.md).

### Weitere Steuerung der Öffnungsstufen (M6-T13)

In der Bibliothek (`HomotopyOptions`, `OpticalHomotopyOptions`) standardmäßig aus, ohne sie bitgleich; Messung im [Bericht M6-T13](ergebnisse/m6-t13-farbe-frueh.md). **Seit M6-T13b (Freigabe Maintainer 2026-11-24) sind Vergrößern mit Abstand zur Schwelle (Q = 0,8), Einfügen nur mit Verbesserung im Abschluss und die Varianten der Farbregel Standard von `synth`** (CLI; die Aktion `synth_segment` ruft `synth` und übernimmt damit Q = 0,8, die beiden anderen wirken dort nicht, weil sie weder Farbregel noch `--insertion-improve` setzt). Abschalter, je bitgleich zum Stand vor M6-T13b: `--grow-margin 0` bzw. `--no-grow-margin`, `--no-insertion-improve-finish`, `--no-color-variants` (Skript: `--wachsen-marge 0`, `--ohne-einfuegen-abschluss`, `--ohne-farb-varianten`; das Skript übergibt ohne `--aktuell`/`--farbregel` wie bisher die Standards der dokumentierten Läufe, also auch `--grow-margin 0`). Referenzläufe mit `synth` ändern sich; Nachmessung im Bericht, Abschnitt „M6-T13b“. Querfarbe (`--color-module-lateral`), `--split-preselect` und `--ghost-dictionary-extra` bleiben aus (Entscheidung Maintainer 2026-11-24).

- **Vergrößern mit Abstand zur Schwelle** (`HomotopyOptions::grow_margin` = Q ∈ (0, 1], CLI `synth --grow-margin Q`, seit M6-T13b Standard Q = 0,8, `--grow-margin 0`: aus; Skript `--wachsen-marge Q`): In den stetigen Phasen (Apertur, Feld, gemeinsam, Spektrumbreite) wächst Δ auch nach einem annehmbaren, langsamen Schritt, wenn F_B ≤ Q · F_max. Anlass (F10): Fast jeder LM-Lauf endet mit `max_iterations`, kein Schritt gilt als schnell; die stetige Farbstufe kroch nach einigen Halbierungen mit Δ_min = 1/1024 weiter (109 Schritte bis zur Breite 0,16), obwohl F_B dabei auf 0,3 F_max sank. Mit Q = 0,8 erreichte derselbe Lauf die volle Breite. JSON `options.grow_margin`.
- **Einfügen nur mit Verbesserung auch im Abschluss** (`HomotopyOptions::insertion_improve_finish`, CLI `--insertion-improve-finish`, verlangt `--insertion-improve`; seit M6-T13b Standard mit `--insertion-improve`, `--no-insertion-improve-finish`: aus; Skript `--einfuegen-abschluss`, `--ohne-einfuegen-abschluss`): Ist das Ziel am Endpunkt verfehlt, wird ein Einfügen nur übernommen, wenn F_B kleiner wird als am Zustand vor dem Abschluss (Varianten wie in den stetigen Phasen); verbessert keine, endet der Lauf mit `goal_not_reached` an diesem Zustand. Anlass: Lauf H2 – das Ziel pro Tupel war verfehlt (größter RMS 0,126 Wellen bei F_B < F_max), jedes Einfügen im Abschluss verschlechterte (bis F_B = 305 Wellen², 6 → 15 Elemente). JSON `options.insertion_improve.finish`.
- **Werfende Variante** (mit `--insertion-improve`, ohne eigene Option): Wirft die Optimierung nach einem Einfügen (z. B. freie Glasvariable von Operator B mit n ≤ 0, „SystemBatch: index muss > 0 sein“), gilt die Variante als verworfen (Vermerk „Einfügen verworfen: Fehler der Optimierung (…)“), statt synth abzubrechen. Läufe ohne Ausnahme sind unverändert; vorher endete synth mit Fehler. Bleibt ohne eigene Option (Entscheidung Maintainer 2026-11-24).
- **Querfarbe** (`OpticalHomotopyOptions::color_module_lateral`, CLI `--color-module-lateral`, verlangt `--color-module`; Skript `--farb-querfarbe`): Überwiegt im Aberrationsbericht am Punkt die Querfarbe (|ΔW111|/4 > |ΔW020|/(2√3), RMS von Kippung bzw. Defokus), empfiehlt die Farbregel das Kittglied-Modul zuerst und ordnet die Spalte nach dem mittleren |y·ȳ| der Spaltflächen (Querfarbe eines dünnen Elements ∝ y ȳ φ/ν). In den Läufen ab F3 griff das Kriterium nicht (Seidel am Punkt zeigt Längsfarbe, die Kippung am Rand stammt aus höheren Ordnungen und der Verzeichnung). JSON `options.color_rule_ext.lateral`.

Tests: `HomotopyControl.GrowMarginGrowsAfterSlowStepsBelowThreshold`, `…InsertionMustImproveInFinishPhase`, `…ThrowingBranchIsDiscarded` (werfende Variante), `ColorStage.LateralColorModuleOrderedByChiefRayHeight` (Seidel-Orakel und Strahldaten des Aberrationsberichts), `CliSynth.ColorStageOptions`; M6-T13b: `CliSynth.M6T13bDefaultsAndSwitches` (Standards im Protokoll, Abschalter ergeben das Protokoll und die Ergebnisdatei wie vor M6-T13b, Standardlauf vergrößert Δ nach langsamen Schritten), Python `test_synth_options_m6t13b_switches`.

Tests (Python, `test_tool_m6t6.py`): `test_color_mode_default_is_color_early` (Folge p1 → p1f → p2 → p3 mit aufgezeichneten Aufrufen, Spektrum und Optionen der Farbstufe, Werkzeuge, Z2), `test_color_mode_late_and_legacy_unchanged` (`spaet`: p1, p2, p3, p4, p5 mit `--stall 0` wie bisher; `--farbe-frueh`; `anfang`), `test_apply_color_mode`, `test_synth_options_color_variants`.

## 8. Kandidatenverwaltung

Die Synthese ist eine Strahlsuche (beam search) über Zweige: Jede Einfügung und jeder Sattelpunkt-Abstieg erzeugt neue Kandidaten. Behalten werden die besten B Kandidaten nach Merit, Elementzahl und (ab M8) Empfindlichkeit. Alle Kandidaten einer Topologie laufen gemeinsam in einem Batch.

### FP32-Vorfilterung im Multistart (ab M7-T2)

Vor der Bewertung eines großen Multistart-Batches in FP64 (oder df64) sortiert `TraceEngine::prefilter_variants` hoffnungslose Varianten aus: auf der RTX 3090 im Kernel rund 19-mal so schnell wie MERIT in FP64 und 4,5-mal so schnell wie df64 ([benchmarks.md](benchmarks.md#fp32-vorfilterung-m7-t2)). Die Entscheidung ist konservativ: Eine Variante wird nur aussortiert, wenn sie das Kriterium auch in FP64 sicher erfüllt ([numerik.md, „FP32-Vorfilterung“](numerik.md#fp32-vorfilterung-m7-t2)); im Zweifel bleibt sie im Batch. Ablauf:

1. Startvarianten erzeugen (`make_perturbed_batch` o. ä.), weitere Residuen E_v (paraxiale Nebenbedingungen, `evaluate_extra_residuals`) in FP64 auf dem Host, Σ e² als `PrefilterSettings::extra_merit`.
2. `prefilter_variants` mit den Kriterien (`PrefilterCriteria`): (a) Anteil versagter Strahlen > `max_failed_fraction` (Standard 0,5), (b) Merit > `max_merit`, (c) Merit > `max_merit_factor` × beste Merit des Batches (b und c Standard aus). Die Margen (`PrefilterMargins`) bleiben auf den Standardwerten.
3. Nur die Varianten mit `keep(v)` in einen neuen Batch kopieren und in FP64/df64 bewerten bzw. optimieren. `merit_lower`/`merit_upper` dürfen für eine Rangfolge der Vorauswahl benutzt werden, **nie** als Merit, Startwert der Dämpfung, Abbruchkriterium oder in der Jacobi-Matrix.

Die Strahlsuche (M6-T4, unten) setzt den Vorfilter auf die Starts eines Einfügens an (`BeamOptions::prefilter`). Ein rangbasiertes Kriterium („nur die schlechtesten p % aussortieren“) lässt sich aus `merit_lower` und `merit_upper` bauen (aussortieren, wenn L_v über dem p-Quantil der U_u liegt), ist aber nicht umgesetzt.

### Festlegungen der Implementierung (M6-T4)

API: `include/mikroptim/synth/beam_search.hpp` (Bibliothek `Mikroptim::synth`, [architektur.md](architektur.md#systemsynthese-homotopie-controller-m6-t3)), optische Bausteine in `OpticalHomotopyModel`; Kommandozeile `mikroptim synth --beam B` ([cli.md](cli.md#strahlsuche---beam-ab-m6-t4)).

```cpp
struct BeamOptions { HomotopyOptions homotopy;        // Ziel, Schwelle, Schrittweiten, Abbruch
                     int beam_width = 2;              // B
                     int alternatives = 1;            // Varianten pro Einfügen
                     int perturbed_starts = 0;        // gestörte Starts pro Zweig
                     std::uint64_t seed = 0; bool prefilter = false;
                     double dedup_tolerance = 1e-4; };
BeamResult run_beam_search(HomotopyModel&, SystemDescription start, const HomotopyPoint&,
                           const BeamOptions&, const std::function<void(const BeamStep&)>& on_step);
// neu in HomotopyModel (Standards ohne Bündelung, bestehende Modelle bleiben gültig):
std::vector<HomotopySolve> solve_all(std::span<SystemDescription* const>, const HomotopyPoint&,
                                     std::vector<int>* groups);
std::vector<HomotopyNeed> insertion_options(const SystemDescription&, const HomotopyPoint&,
                                            const HomotopyNeed&, int max);
double distance(const SystemDescription&, const SystemDescription&, const HomotopyPoint&);
std::vector<SystemDescription> perturbed_starts(const SystemDescription&, const HomotopyInsertion&,
                                                const HomotopyPoint&, int count, std::uint64_t seed);
std::vector<bool> prefilter(std::span<const SystemDescription* const>, const HomotopyPoint&);
```

**Kandidat.** Systembeschreibung (Basis wie in Abschnitt 7), letzte Optimierung (F, F_B, RMS, versagte Strahlen, annehmbar), Elementzahl und Herkunft (Eltern, Operator, Variante, Zweig, Start, Punkt der Entstehung). Kandidaten entstehen nur an Verzweigungen (Wurzel = Startsystem, Kinder eines Einfügens); die Schritte dazwischen ändern den Zustand eines Kandidaten, nicht seine Nummer. Die Nummern werden in der Reihenfolge der Entstehung vergeben (deterministisch).

**Gleichschritt.** Alle Kandidaten des Strahls stehen an *einem* gemeinsamen Homotopie-Punkt; jeder Schritt (Apertur, Feld, Spektrum, Penalty-, ε-, μ-Stufe, innere Homotopie) wird für alle betroffenen Kandidaten am selben Punkt gerechnet. Begründung gegenüber asynchronen Zweigen mit Gruppierung pro Runde: (1) Die Rangfolge vergleicht Merits; F hängt stark vom Punkt ab (in der Abnahme von Abschnitt 7 wächst F_B zwischen NA 0,4625 und 0,60 um mehr als den Faktor 50), Kandidaten an verschiedenen Punkten wären nicht vergleichbar, eine Rangfolge bräuchte ein Fortschrittsmaß neben der Merit. (2) Nur Kandidaten am selben Punkt passen in einen gemeinsamen LM-Aufruf: Der Batch hat gemeinsame Felder, die vom Punkt abhängen (Normierungsradien der Geisterplatten, Spektrum, Gewichte), und die Zusatzresiduen (Gewichte der Nebenbedingungen) gelten für alle Läufe eines Aufrufs. (3) Mit B = 1 ergibt sich genau der einzweigige Controller. Der Preis: Die Schrittweite ist gemeinsam, ein schwacher Kandidat bremst die übrigen (unten).

**Ablauf** (`run_beam_search`, Phasen wie `run_homotopy`):

1. Start: Wurzel mit Penalty-Fortsetzung. Kandidaten, die nicht annehmbar sind, fügen ein (Diagnose am Startpunkt), die übrigen warten; bis alle annehmbar sind.
2. Stetige Parameter (Apertur, Feld) mit einer gemeinsamen Schrittweite Δ: Versuch aller Kandidaten bei p + Δ.
   - Alle annehmbar: alle übernehmen; Δ wächst nur, wenn alle schnell konvergiert sind, sonst bleibt es.
   - Mindestens einer abgelehnt und Δ/2 ≥ Δ_min: alle verwerfen (auch die annehmbaren), Δ/2, neuer Versuch aller ab dem guten Zustand.
   - Mindestens einer abgelehnt und Δ/2 < Δ_min: Die abgelehnten fügen am guten Zustand ein (Diagnose zuerst am Versuch, Abschnitt 7), die annehmbaren bleiben am guten Zustand stehen (ihr Versuch wird verworfen, Protokoll „bleibt am guten Zustand“); danach Auswahl, Δ ← Δ₀ für alle.
   Ein Kandidat wird also nur nach seinen eigenen Ergebnissen eingefügt oder aufgegeben, nie, weil ein anderer schneller war. Die verworfenen annehmbaren Versuche kosten je eine Optimierung pro Halbierung bzw. Einfügen.
3. Spektrumstufen: wie 2 ohne Halbieren (abgelehnt → Einfügen, die übrigen warten, dieselbe Stufe erneut).
4. Abschluss: ε-Stufen und Zielprüfung aller; sobald ein Kandidat das Ziel erreicht, endet der Lauf. Sonst μ-Stufen mit Prüfung nach jeder; verfehlt: alle fügen am Zustand vor dem Abschluss ein, danach erneut.

Abbruch wie in Abschnitt 7; ein Kandidat ohne Ausweg (kein Bedarf, höchste Elementzahl) wird verworfen, der Lauf endet erst, wenn kein Kandidat übrig bleibt – dann mit Status und Meldung des ersten (bestplatzierten) Kandidaten ohne Ausweg, der Strahl bleibt im Zustand vor dem Einfügen. `max_steps` zählt die Optimierungen aller Kandidaten; ein gemeinsamer Aufruf, der die Grenze überschreiten würde, beendet den Lauf (`max_steps`).

**Verzweigung** (Einfügen eines Kandidaten am Punkt p):

- *Varianten* (`insertion_options`, höchstens `alternatives`): die Empfehlung der Diagnose zuerst, dann die Standardwahl der übrigen Lagen mit Empfehlung, absteigend nach Merit-Gewinn (Abschnitt 5; ohne Merit-Werte Σ|a|), dann – nur für Operator A – die übrigen aktiven Gläser jeder Lage (empfohlene Lage zuerst), absteigend nach Σ|a|. Eine Glasvariante ist der Bericht mit dieser einen aktiven Platte an der Lage, sodass `meniscus_from_diagnosis` ihr Glas und das Vorzeichen ihres t₄ nimmt. Bei Operator B wählt die Dispersion das Zielglas (Abschnitt 6), dort gibt es keine Glasvarianten.
- *Zweige* jeder Variante wie in Abschnitt 7 (A: zwei Starts am Sattelpunkt, B: einer).
- *Gestörte Starts* (`perturbed_starts`, Standard 0) pro Zweig: Die Krümmungen der neuen Flächen (A: beide Meniskusflächen, B: Kittfläche) werden um s·ξ gestört, ξ = 2u − 1 gleichverteilt in [−1, 1) (u wie im Variantengenerator, Abschnitt 4), c ← clamp(fma(s, ξ, c)) auf die Schranken der Krümmungsvariable, s = `start_perturbation` (Standard 0,01/mm, das Fünffache der Sattelstörung). Start j = 1 … N eines Zweigs zieht aus einem eigenen `std::mt19937_64` mit dem Startwert `variant_seed(s_n, j)`, eine Ziehung pro neuer Fläche in aufsteigender Reihenfolge; s_n = `variant_seed(seed, n)` mit dem fortlaufenden Zähler n der Zweige des Laufs. Alles bitgenau festgelegt und plattformunabhängig wie der Variantengenerator.
- *Vorfilter* (`prefilter`, Standard aus): Die Starts eines Einfügens (alle Varianten, Zweige und Starts) werden vor der inneren Homotopie gemeinsam grob bewertet (FP32, M7-T2: `prefilter_cpu` bzw. mit `--gpu` `TraceEngine::prefilter_variants`), Kriterien: mehr als die Hälfte der Strahlen versagt sicher oder L_v > 10 × beste U_u der Starts (`homotopy_prefilter_defaults`); Zusatzresiduen E_v in FP64 ohne die trace-abhängigen (L_v bleibt eine untere Schranke). Aussortierte Starts gehen mit Status „verworfen: Vorfilter“ in den Stammbaum und werden nicht optimiert. Die Werte der Vorfilterung beeinflussen nur diese Entscheidung.

Danach die innere Homotopie aller Kinder gemeinsam (Stufe j aller Kinder in einem Aufruf von `solve_all`), dann die Auswahl, dann für die gewählten Kinder der Abschluss des Einfügens und die Penalty-Fortsetzung (gemeinsam). Wie beim einzweigigen Controller wird vor dem Abschluss gewählt; die übrigen Kandidaten des Strahls haben ihre Fortsetzung schon hinter sich.

**Rangfolge und Auswahl** (`beam_rank_before`, vollständige Ordnung): (1) endliche Merit vor nicht endlicher; (2) annehmbar (F_B ≤ F_max, kein versagter Strahl) vor nicht annehmbar; (3a) unter annehmbaren weniger Glaselemente zuerst, dann kleineres F; (3b) unter nicht annehmbaren weniger versagte Strahlen, dann kleineres F; (4) kleinere Nummer. F ist die Merit der letzten Optimierung am gemeinsamen Punkt (alle Residuen wie im LM, wie die Wahl des besten Zweigs in Abschnitt 7; F_B allein würde mechanisch unzulässige Kandidaten bevorzugen); mit `synth --tolerance-aware` ist das die erwartete Merit F_tol (Abschnitt 14, M11-T6b). Weniger Elemente gehen vor kleinerem F, weil jeder annehmbare Kandidat das Ziel am Punkt mit Geisterplatten erreicht: Ein zusätzliches Element lohnt erst, wenn das kleinere System nicht mehr weiterkommt. Behalten werden die ersten B nach Rang ohne Duplikate (unten), die übrigen werden verworfen („Rang hinter den besten B“).

**Dominanz.** Kandidat a dominiert b am selben Punkt, wenn a annehmbar ist, höchstens so viele Elemente und höchstens dasselbe F hat (eines echt kleiner). Dominierte Kandidaten werden **nicht** vorab entfernt, solange Platz im Strahl ist: Ein Kandidat mit mehr Elementen und größerem F kann bei höherer Apertur mehr Reserve haben (er hat den Freiheitsgrad schon). Die Dominanz wirkt nur über die Rangfolge.

**Deduplizierung** (`distance`, `dedup_tolerance`, Standard 10⁻⁴; < 0 aus): Zwei Kandidaten sind gleich, wenn sie dieselbe Topologie haben (Kriterium wie für gemeinsame Aufrufe, unten) und d = max_k |x_a,k − x_b,k| / max(|x_a,k|, |x_b,k|, x_typ,k) ≤ Toleranz (größte relative Abweichung der Variablen am Punkt). Geprüft wird bei jeder Auswahl gegen die schon behaltenen (besser platzierten) Kandidaten und nach jedem gemeinsamen Schritt im Strahl (Zweige, die erst später ins selbe Minimum laufen); der schlechter platzierte wird verworfen („Duplikat von #k“). In der Abnahme enden die beiden Sattelzweige des ersten Einfügens bei d = 1,3·10⁻⁵ (Abschnitt 7: „im selben Minimum“). Die Toleranz ist eine Festlegung oberhalb der Streuung nicht konvergierter LM-Läufe im selben Tal (gemessen 1,3·10⁻⁵ und 2,0·10⁻⁵ für Zweige im selben Minimum).

**Gemeinsame Aufrufe pro Topologie** (`OpticalHomotopyModel::solve_all`). Zwei Kandidaten a, b am selben Punkt gehören in denselben LM-Aufruf, wenn (1) die Batch-Topologie, die Flächentypen, die Ordnungen der Geisterplatten und die Variablen (Flächen, Schranken, typische Größen, Versätze der Spalte mit Geisterplatten) gleich sind, (2) die realisierten Beschreibungen gleich sind, nachdem in beide die Variablenwerte von b geschrieben wurden (also alles außerhalb der Variablen: Gläser, nicht variable Dicken wie die Stufe der inneren Homotopie, Nebenbedingungen und ihre Gewichte einschließlich μ pro Platte), und (3) `lm_point_batch` der Vorlage von a mit x_b bitgleich zu dem der Vorlage von b mit x_b ist. Jeder Kandidat kommt in die erste Gruppe, deren erster Kandidat (Vorlage) passt. Der LM-Aufruf einer Gruppe hat eine Vorlage für alle Läufe und die Zusatzresiduen der Vorlage; unter (1)–(3) sieht der LM für jeden Lauf dieselben Batches und Zusatzresiduen wie im Einzelaufruf, und das Ergebnis eines Laufs hängt nicht von den übrigen ab (Abschnitt 4, ADR-0007): **Die Gruppierung ändert kein Ergebnis**, auch nicht auf der GPU (Tests `BeamSmoke.GroupingDoesNotChangeResultsCpu`, `CudaBeamSearch.GroupingAndRepeatBitwiseGpu`: Protokoll ohne Gruppennummern und Laufzeiten, Stammbaum und Endsystem bitgleich; `OpticalBeam.SolveAllGroupsByTopologyBitwise`). Typische Gruppen: die Zweige und gestörten Starts einer Variante in jeder Stufe der inneren Homotopie, Kandidaten verschiedener Eltern nach dem Einfügen an derselben Lage, Schwester-Kandidaten in allen späteren Schritten, solange sie sich nur in den Variablen unterscheiden. Nicht gemeinsam: verschiedene Elementzahl oder Lage, verschiedene Gläser oder μ der Quellplatte, verschiedene Mittendicken vor dem Abschluss.

**Reproduzierbarkeit.** Die Strahlsuche ist eine reine Funktion der Ergebnisse des Modells: feste Reihenfolge der Kandidaten (Rang mit vollständigem Gleichstandsbruch über die Nummer), der Varianten, Zweige und Starts, Zufallszahlen nur aus dem Startwert (oben), keine Threads in der Steuerung; Parallelität nur innerhalb des LM (bitgleich für jede Threadzahl, Abschnitt 4). Zwei Läufe mit gleichen Eingaben und gleichem Startwert ergeben dasselbe Protokoll und denselben Stammbaum bis aufs Bit (ohne Laufzeiten) und dieselben Systeme (Tests `BeamSearch.ReproducibleWithSeed`, `BeamSmoke.DeterministicWithSeedCpu` mit gestörten Starts und Vorfilter, `CudaBeamSearch.GroupingAndRepeatBitwiseGpu`).

**Protokoll.** Ein Eintrag pro Optimierung, Einfügen und Zielprüfung wie in Abschnitt 7 (`HomotopyStep`) mit Kandidat und Nummer des gemeinsamen Aufrufs (`BeamStep`). Stammbaum (`BeamNode`): Nummer, Eltern, Kinder, Herkunft (Operator, Variante, Zweig, Start), Punkt der Entstehung, letzte Optimierung (Punkt, F, F_B, RMS, versagte Strahlen, annehmbar), Elementzahl, Zahl der Optimierungen, letzte Zielprüfung, Status mit Grund und Protokolleintrag der Statusänderung. Status: `kept` (im Strahl; am Ende behalten, Ziel verfehlt oder Lauf abgebrochen), `finished` (im Strahl, Ziel erreicht), `discarded` (Rang, Duplikat, Vorfilter, kein Ausweg), `expanded` (in Kinder aufgeteilt). Ergebnis ist der bestplatzierte Kandidat unter denen, die das Ziel erreicht haben, sonst der bestplatzierte des Strahls. Ausgabe: Text (`write_beam_step_line`, `write_beam_summary`, Baum `write_beam_tree`), JSON (`beam_result_json`: Felder wie in Abschnitt 7, dazu `beam_search` mit Breite, Startwert, bestem Kandidaten, Strahl, Zahl der Aufrufe und `candidates`; pro Schritt `candidate` und `group`) und Graphviz-DOT (`write_beam_dot`).

**Abnahme** „Läufe sind mit festem Seed reproduzierbar; Protokoll zeigt den Stammbaum der Kandidaten“: Reproduzierbarkeit siehe oben; Stammbaum in Text, JSON und DOT (Tests `BeamSearch.ProtocolTreeJsonDot`, `CliSynth.BeamShortRunJsonTreeAndDot`, Beispiel in [cli.md](cli.md#strahlsuche---beam-ab-m6-t4)).

**Messungen** (Startsystem und Einstellungen der Abnahme von Abschnitt 7: Einzellinse, NA 0,3 → 0,6 auf der Achse, LM höchstens 500 Zyklen pro Schritt, Schranken ±1/mm der neuen Krümmungen; `mikroptim synth … --curvature-bound 1 --beam B`, CPU, Release; Laufzeiten bei teils paralleler Last, nur zur Orientierung; längerer Lauf als Test `BeamSearchOptimization.DISABLED_ApertureToNa06BeamCpu`, GPU `CudaBeamSearch.DISABLED_ApertureToNa06BeamGpu`):

| Lauf | Ergebnis | Elemente | RMS am Ziel [Wellen] | Kandidaten | Einfügungen | Optimierungen / LM-Aufrufe | Laufzeit |
|---|---|---|---|---|---|---|---|
| einzweigig (`run_homotopy`) | `goal_reached` | 4 | 0,01633 | – | 3 | 63 / 63 | 19 s |
| B = 1 | `goal_reached` | 4 | 0,01633 | 7 | 3 | 63 / 48 | 24 s |
| B = 2 | `goal_reached` | 4 | **0,00224** | 9 | 4 | 100 / 68 | 39 s |
| B = 2, 2 Varianten | `goal_reached` | **3** | 0,03679 | 7 | 3 | 74 / 40 | 22–26 s |
| B = 4, 2 Varianten | `goal_reached` | **3** | 0,03538 | 11 | 5 | 158 / 97 | 74 s |
| B = 2, 2 gestörte Starts, Vorfilter, Startwert 1 | `goal_reached` | 4 | 0,01634 | 19 | 3 | 110 / 62 | 26 s |

GPU (RTX 3090, `optimize_lm_gpu`): einzweigig 4 Elemente, RMS 0,01566 Wellen, 41 s; B = 2 mit 2 Varianten dieselben Entscheidungen wie auf der CPU, 3 Elemente, RMS 0,03746 Wellen, 74 Optimierungen in 40 Aufrufen, 44 s. Gebündelt gegen einzeln (Kurzlauf B = 2, 23 Optimierungen in 15 bzw. 23 Aufrufen): 1,43 s gegen 1,46 s – bei diesen kleinen Systemen (1–3 Linsen, 10 × 8 Strahlen, 2P + 1 ≤ 40 Varianten pro Lauf) ist die GPU nicht ausgelastet, die Zeit steckt im Host (Zusatzresiduen, Cholesky, Diagnose); der Gewinn der Bündelung ist für große Strahlen und große Systeme (M6-T6) zu messen.

Beobachtungen: (1) B = 1 trifft dieselben 63 Entscheidungen wie der einzweigige Controller; die Sattelzweige jeder Stufe der inneren Homotopie laufen gemeinsam (48 statt 63 LM-Aufrufe). (2) Beim ersten Einfügen enden beide Sattelzweige im selben Minimum (d = 1,3·10⁻⁵, Duplikat). (3) Mit B = 2 bleibt beim zweiten Einfügen auch der Zweig im Strahl, den der einzweigige Controller verwirft (F_B = 1,8·10⁻² > F_max): Er fügt sofort ein weiteres Element ein und erreicht mit ebenfalls vier Elementen RMS 0,0022 Wellen, siebenmal kleiner als der einzweigige Weg (dessen Nachfahre endet bei 0,0163 Wellen und verliert in der Rangfolge nach F). (4) Mit zwei Varianten pro Einfügen ist beim zweiten Einfügen die zweite Lage der Diagnose besser als die empfohlene: statt eines N-BK7-Meniskus vor der Linse (Variante 0, Platte 0) ein hochbrechender Meniskus hinter der Linse (Variante 1, Platte 3). Mit ihm erreichen beide Sattelzweige NA 0,6 ohne weiteres Einfügen: **drei statt vier Elemente**, beugungsbegrenzt (0,037 Wellen ≤ 1/14). B = 4 findet dasselbe Ergebnis mit drei Elementen und behält die Vier-Element-Systeme (RMS bis 0,0022) als Nebenkandidaten, die in der Rangfolge wegen der Elementzahl zurückstehen. (5) Die Strahlsuche ist in allen Läufen mindestens so gut wie der einzweigige Controller (weniger Elemente, oder gleich viele und RMS nicht größer). Auf der CPU kostet ein gemeinsamer Aufruf etwa so viel wie die Einzelläufe nacheinander (gleiche Arbeit); die Laufzeit wächst mit der Zahl der Optimierungen. (6) Gestörte Starts helfen im Kurzlauf (RMS 0,096 statt 0,227 Wellen, [cli.md](cli.md#strahlsuche---beam-ab-m6-t4)), im längeren Lauf ändern sie das Ergebnis nicht.

**Tests** (`tests/test_beam_search.cpp`, synthetisches Modell `tests/homotopy_mock.hpp`): B = 1 gleich `run_homotopy` in elf Szenarien des einzweigigen Controllers (Status, Meldung, Endsystem, alle Schritte außer der Reihenfolge der inneren Homotopie), Verzweigung und Auswahl der besten B mit Stammbaum, Rangfolge, Deduplizierung (an und aus), Varianten × Zweige × gestörte Starts mit Startwerten, Reproduzierbarkeit und Einfluss des Startwerts, Gruppierung ohne Einfluss auf Ergebnisse, Gleichschritt mit gemischter Ablehnung, Vorfilter, Abbruch bei leerem Strahl, Optionen, Protokoll als Text, JSON und DOT. Optisch (`tests/test_beam_search_optimization.cpp`): Varianten aus einer Diagnose mit drei aktiven Platten, gemeinsame LM-Aufrufe bitgleich zu Einzelaufrufen, Abstand, gestörte Starts, Vorfilter; Kurzläufe `BeamSmoke.*` (NA 0,3 → 0,31 wie `HomotopySmoke`, B = 1 gleich `run_homotopy` bitgleich, Gruppierung ohne Einfluss, Determinismus mit Startwert); GPU `tests/test_cuda_beam_search.cpp`.

## 9. Glas-Einrasten

Am Ende: für jedes Glaselement die k nächsten Katalog-Gläser im normierten (n₀, ν₁, ν₂)-Raum (Normierung der Glas-Hülle, Abschnitt 2.4); Kombinationen werden als Varianten im Batch reoptimiert, das beste Ergebnis gewinnt. Weiche Nebenbedingung auf Transmission bei 420–450 nm (Farbstich), Kennzeichnung spannungsarmer Gläser für DIC.

### Festlegungen der Implementierung (M6-T5)

API `include/mikroptim/optim/glass_snap.hpp` (Bibliothek `Mikroptim::optim`), CLI `mikroptim snap` ([cli.md](cli.md#mikroptim-snap)). Eingabe ist ein System mit Glasvariablen, dessen Werte das kontinuierliche Ergebnis sind (typisch die Ergebnisdatei von `mikroptim optimize` mit Buchdahl-Materialien); Katalog- oder konstante Startgläser werden wie in der Optimierung zuerst umgewandelt (`convert_glass_variables`).

**Glaselemente** sind die Flächen mit Variable `glass` in der Reihenfolge der Variablen (wie die Elemente der Glas-Hülle), ihre kontinuierlichen Koordinaten x_e = (n₀, ν₁, ν₂) stehen in `buchdahl_coeff` des Batches.

**Nachbarn** (`nearest_glasses`): Kandidaten sind die Gläser der Glas-Hülle (`make_glass_hull(catalog)`, also alle Gläser der Kataloge, deren `range_nm` 400–700 nm abdeckt; ihre Koordinaten sind die 3-Term-Fits, dieselben wie die der Variablen), eingeschränkt auf die, deren Gültigkeitsbereich alle Wellenlängen des Spektrums und λ_ref abdeckt (sonst ließe sich die Brechzahltabelle des eingerasteten Systems nicht bilden). Abstand im normierten Raum der Hülle, euklidisch mit optionalen Gewichten pro Koordinate,

$$d(x, y) = \sqrt{\textstyle\sum_j w_j\,(u_j(x) - u_j(y))^2},\qquad u_j = (x_j - m_j)/s_j,$$

Standard w = (1, 1, 1) (`GlassSnapOptions::distance_weights`; die CLI verwendet den Standard). Gewählt werden die k kleinsten Abstände, bei Gleichstand der kleinere Index in der Hülle (Lade-Reihenfolge der Kataloge). Gibt es weniger als k Gläser, sind es alle; gibt es keines, bricht das Einrasten mit `std::invalid_argument` ab.

**Kombinationen** (`glass_combinations`): Kosten einer Kombination c = (j_0, …, j_{E−1}) sind Σ_e d_{e,j_e}² (aufsteigend in e summiert), also das Quadrat des Abstands im Produktraum aller Elemente. Ist das kartesische Produkt Π k_e höchstens N = `max_combinations` (Standard 64), werden alle Kombinationen gerechnet, sonst die N mit den kleinsten Kosten. Bestimmt werden sie mit einer Strahlsuche über die Elemente: Nach jedem Element bleiben die N besten Teilkombinationen nach (Teilkosten, Index-Tupel lexikographisch). Weil die Kosten eine Summe unabhängiger, nicht negativer Beiträge sind, ist das genau die Auswahl aus dem vollen Produkt: Gehörte eine Kombination mit einem Präfix p zu den N besten, obwohl p nicht unter den N besten Präfixen ist, ergäben die N besseren Präfixe mit demselben Rest N bessere Kombinationen. Ausnahme sind Gleichstände, die erst durch die Rundung der Summe entstehen. Aufwand O(E · N k log(N k)) statt k^E. Die Läufe stehen in der Reihenfolge (Kosten, Tupel); die Zahl aller Kombinationen (`total_combinations`, gesättigt bei 2⁶⁴ − 1) und `truncated` gehen in den Bericht.

**Läufe.** Jede Kombination ergibt die Systembeschreibung `snapped_description`: Material der Elemente = Katalogname, alle Einträge `glass` aus `variables` entfernt (der Abschnitt entfällt, wenn er leer wird), alles Übrige unverändert (auch der Eintrag `glass_hull`, der ohne Glasvariablen wirkungslos ist). Daraus entstehen Brechzahltabelle (Sellmeier des Katalogs, nicht der Buchdahl-Fit) und eine Batch-Variante wie beim Laden einer Datei (`rebuild_loaded_system`, `make_system_batch`); die Variante c der gemeinsamen Vorlage (`GlassSnapResult::base`) ist die der Kombination c. Topologie und gemeinsame Felder hängen nicht vom Glas ab, die Brechzahlen (alle Wellenlängen und λ_ref) und – ohne `target.focal_length` – die Feldziele aus der paraxialen Brennweite unterscheiden sich pro Variante. Alle Läufe starten bei den übrigen Variablen des kontinuierlichen Ergebnisses (Krümmungen, Dicken, Arbeitsabstand, Geisterkoeffizienten) und laufen gemeinsam im gebündelten LM mit einer Vorlage pro Lauf (Abschnitt 4: `optimize_lm_cpu`, auf der GPU `cuda::optimize_lm_gpu`, einstellbar über `GlassSnapOptimizer`). Ein System ohne weitere Variablen lehnt `snap_glasses` ab (nichts zu reoptimieren).

**Merit.** Dieselben Einstellungen (Quadratur, Strahlversagen, Zusatzresiduen) für das kontinuierliche und die eingerasteten Systeme; die Glas-Hülle gehört nicht dazu (ohne Glasvariablen gibt es sie nicht, und ein Katalogglas liegt auf oder in der Hülle). `mikroptim snap` verwendet wie `mikroptim optimize` die paraxialen Nebenbedingungen und die L1-Strafe der Geisterkoeffizienten. Berichtet werden pro Lauf F eingerastet ohne Reoptimierung (F am Startpunkt, Auswertung mit 0 Zyklen ohne Strafverfahren, dazu die versagten Strahlen) und F nach der Reoptimierung, dazu F_kont des kontinuierlichen Systems an seinem Punkt und die Verschlechterung ΔF = F_best − F_kont absolut und relativ. Beim weichen Versagen enthalten F_kont und F eingerastet Strafen mit dem Start-w_fail, F nach der Reoptimierung die mit dem w_fail der letzten Runde (gleich, solange kein Strahl versagt).

**Bester Lauf** (`select_best_run`, dieselbe Regel wie der Multistart von `mikroptim optimize`): endliche Merit, dann die wenigsten versagten Strahlen, dann die kleinste Merit, bei Gleichstand der kleinste Index (= kleinste Kosten). Das Ergebnis (`GlassSnapResult::result`) ist die Beschreibung der besten Kombination mit dessen x (`apply_to_description`); geschrieben und neu geladen liefert sie die berichtete Merit bitgleich (wie bei `mikroptim optimize`, gleiche Einschränkung bei Feldzielen ohne `target.focal_length`).

**Kann die Verschlechterung negativ werden?** Ja, ohne Fehler: (1) Die Merit ist nicht konvex. Das kontinuierliche Ergebnis ist ein lokales Optimum; das Ändern des Glases und die Reoptimierung können in ein anderes Becken mit kleinerer Merit führen. (2) Das kontinuierliche Modell ist nicht das Katalogglas: ν₃ bleibt beim Startglas, und der Buchdahl-Fit weicht um bis zu 10⁻⁵ von Sellmeier ab (numerik.md §10); ein Katalogglas kann daher in einem Teil des Spektrums besser passen als das Modell an derselben Stelle. (3) War das kontinuierliche Ergebnis nicht konvergiert, ist F_kont kein Optimum. Die Abnahme prüft ΔF ≥ −10⁻⁶ F_kont (`kGlassSnapMeritSlack`) an einem konvergierten kontinuierlichen Optimum; im Abnahmesystem ist ΔF deutlich positiv.

**Determinismus.** Kandidaten, Kombinationen und Auswahl sind reine Funktionen der Eingabe (feste Sortierschlüssel), die Läufe des LM unabhängig voneinander (Abschnitt 4); zwei Aufrufe liefern bitgleiche Ergebnisse, und die ersten N′ Läufe mit Begrenzung N′ sind bitgleich zu den ersten N′ Läufen ohne Begrenzung.

**Nicht umgesetzt (Erweiterungspunkte).** *Transmission 420–450 nm:* Der Glaskatalog liest das Feld `transmission` (dateiformat.md) bisher nicht ein, und weder `basis.json` noch die Testgläser enthalten Transmissionsdaten. Vorgesehen ist ein Zusatzresiduum pro eingerastetem Element, √w · max(0, τ_min − τ_i(λ)) über 420–450 nm bei der Elementdicke (τ_i aus der Reintransmission τ_ref bei d_ref: τ = τ_ref^{d/d_ref}), oder eine Vorauswahl der Kandidaten über `usable` von `nearest_glasses`. *Spannungsarme Gläser (DIC):* Es gibt kein Datenfeld (z. B. Spannungsdoppelbrechung K bzw. Spannungsoptischer Koeffizient); eine Kennzeichnung bräuchte eine kompatible Erweiterung von `mikroptim.glass/1` und würde im Bericht pro Kandidat ausgegeben. Beides entscheidet der Maintainer zusammen mit den Herstellerkatalogen (AGENTS.md §8).

**Abnahme** (`tests/test_glass_snap.cpp`, GPU `tests/test_cuda_glass_snap.cpp`, CLI `tests/test_cli_snap.cpp`): frei erfundener Achromat `data/systems/achromat_glasvariablen.json` (verkittetes Doppel f = 50 mm, NA 0,05, F/e/C; Start N-BK7 und ein Flint mit den Buchdahl-Koordinaten des Testglases SYN-21 (Buchdahl-Material, damit die Datei nur Gläser aus `basis.json` nennt); Variablen c₁, c₂, c₃, beide Gläser, w; Brennweite und Glas-Hülle mit w = 10⁶; Kataloge `basis.json` und `synthetisch.json`). Kontinuierlich (CPU): F 35,7 → 9,169·10⁻⁶ Wellen² in 419 Zyklen (`converged_step`), beide Gläser im Inneren der Hülle. Einrasten mit k = 3, N = 64 (9 Kombinationen), 500 Zyklen:

| Kriterium | Ergebnis |
|---|---|
| Kandidaten | Fläche 1: SYN-10 (d = 0,0369), N-BK7 (0,0634), SYN-07 (0,0663); Fläche 2: SYN-21 (0,0029), SYN-19 (0,0350), SYN-24 (0,0445) |
| nur Katalog-Gläser | Ergebnis: SYN-10/SYN-19, keine Buchdahl-Materialien, keine Glasvariablen; im Batch hat jede Variante die Sellmeier-Brechzahlen ihrer Gläser (bitgleich zu `Glass::index`) |
| Merit-Verschlechterung | kontinuierlich 9,169·10⁻⁶, eingerastet ohne Reoptimierung 4,16·10³ (die Brechzahländerung verschiebt Brennweite und Fokus), reoptimiert 3,796·10⁻⁵ Wellen²; ΔF = 2,88·10⁻⁵ (relativ 3,14) |
| bester Lauf | kleinste Merit der neun Läufe (3,796·10⁻⁵ gegen 3,810·10⁻⁵ … 6,92·10⁻⁵); nicht die Kombination mit dem kleinsten Abstand (SYN-10/SYN-21, Lauf 0) |
| Ergebnisdatei | geschrieben, neu geladen: Merit bitgleich; `mikroptim trace` liefert die berichtete RMS-Wellenfront bitgleich |
| Determinismus | zwei Läufe bitgleich; mit N = 4 die ersten vier Läufe der vollen Liste, bitgleich |
| Nachbarn, Kombinationen | 1.392 Kandidaten gegen Bruteforce (zwei Gewichtungen), 256 Fälle der Kombinationen gegen das volle Produkt (mit und ohne Gleichstände) |

## 10. Literatur

- F. Bociort, M. van Turnhout et al.: Arbeiten zur Saddle-Point Construction in optischen Systemen (TU Delft, ab ca. 2007).
- M. Isshiki et al.: Global optimization with escape function (Lens design).
- H. B. Nielsen: Damping parameter in Marquardt's method (1999).
- J. Nocedal, S. J. Wright: Numerical Optimization (exakte Penalty-Funktionen).

## 11. Monte-Carlo-Tolerierung (M8-T4)

Ziel: Verteilung von RMS-Wellenfront und Strehl eines fertigen Entwurfs unter Fertigungstoleranzen, mit Kompensator. Toleranzsatz im Systemformat ([dateiformat.md, „Toleranzen“](dateiformat.md#toleranzen-ab-m8-t4)), Bibliothek `Mikroptim::tolerance` (`include/mikroptim/tolerance/`, [architektur.md](architektur.md#monte-carlo-tolerierung-m8-t4)), Kommandozeile `mikroptim tolerance` ([cli.md](cli.md#mikroptim-tolerance)).

### Stichproben

- **Zufallsgrößen.** Jede aufgelöste Toleranz k (`resolve_tolerances`) hat eine (Radius, Dicke, Brechzahl, Abbe-Zahl) oder zwei (Dezentrierung Δx, Δy; Verkippung α, β) unabhängige Zufallsgrößen ξ ∈ [−1, 1]; der Wert ist δ_k·ξ. Stichprobe i zieht ihre ξ in der Reihenfolge der Toleranzen aus einem eigenen `std::mt19937_64` mit Startwert `variant_seed(seed, i)` (wie der Variantengenerator, Abschnitt 4). Damit hängt jede Stichprobe nur von Startwert und Index ab, nicht von Tranchen, Batchgröße, Threads oder Backend.
- **Verteilungen** nur mit exakten bzw. korrekt gerundeten Operationen (portabel bitgleich): gleichverteilt ξ = 2u − 1; „normal“ als Irwin-Hall-Näherung z = Σ₁¹² u_i − 6 (Mittel 0, Varianz 1, Exzess −0,1; vom Maintainer als Normalverteilung der Tolerierung bestätigt, Entscheidung 2026-11-19), neu gezogen, solange |z| > c, ξ = z/c, also σ = δ/c und abgeschnitten bei ±δ (c = `cutoff_sigma`, Standard 2, erlaubt 0,5 … 6). u = `unit_uniform` (53 Bit). Exakte Varianz von ξ aus der Irwin-Hall-Dichte: c = 2 → Var(ξ) = 0,1962 (abgeschnittene Normalverteilung 0,1934; die Irwin-Hall-Verteilung hat leichtere Ränder).
- **Anwendung** (`ToleranceModel::apply`, Batch mit voller Pupille, `make_non_coaxial`), ausgehend von der nominellen Variante:
  - Radius: c′ = c/(1 + ε) bzw. in Passen c′ = c + N·Δc₁ (Δc₁ aus der Pfeilhöhe am Rand, dateiformat.md).
  - Dicke: + δ auf das letzte Teilstück des Spalts nach der Fläche.
  - Brechzahl und Abbe-Zahl pro Glas: n′(λ) = n_ref + Δn + (n(λ) − n_ref)/(1 + e), n_ref = n(λ_ref) des Batches (`paraxial_index`); ohne Abbe-Toleranz n(λ) + Δn. Die Hauptdispersion und damit 1/ν skaliert mit 1/(1 + e); das Glasmodell selbst bleibt unverändert.
  - Dezentrierung und Verkippung einer Fläche: Werte der Flächentransformation (ADR-0011), zur Transformation der Datei addiert (Verkippung um den schon dezentrierten Scheitel, Drehungen zusammengesetzt).
  - Elemente danach in Toleranzreihenfolge als starrer Körper: x → P + R(x − P) + (Δx, Δy, 0), R = R_y(β)R_x(α), P = (0, 0, z_P) auf der Achse (nominelle Scheitellagen z_s = Σ Dicken). Pro Fläche neuer Scheitel und neue Drehung R·R_s; die Drehung wird in R_y(β′)R_x(α′)R_z(γ) zerlegt, R_z(γ) entfällt (wirkungslos bei Kugel und Ebene mit kreisförmigem Rand): α′ = atan2(−n_y, √(n_x² + n_z²)), β′ = atan2(n_x, n_z) mit der Normalen n = R R_s e_z. Die axiale Verschiebung δz des Scheitels geht in die Dicken (davor + δz, danach − δz; nach der letzten Fläche in den Arbeitsabstand, den der Kompensator ohnehin neu setzt). Ohne Drehung wird nur verschoben (bitgleich). Die Elementbewegung rechnet mit den nominellen Scheitellagen; Dickentoleranzen innerhalb des Elements verschieben den Drehpunkt also nicht (Fehler zweiter Ordnung).
  - Grenzen 0 und ξ = 0 lassen die Variante bitgleich (Test `ToleranceModel.ZeroDeviatesLeaveBatchBitIdentical`).

### Kompensator und Bewertung

- **Tranchen** von höchstens `chunk` Stichproben (Standard 4.096; nur Speicher und Laufzeit): Stichproben schreiben (`write_samples`), mit Kompensator den Arbeitsabstand aller Stichproben gemeinsam mit dem gebündelten LM optimieren (eine Variable `working_distance` ohne Schranken, Start beim Arbeitsabstand der Stichprobe, Merit wie in der Optimierung: Strahlresiduen in Wellen mit weichem Versagen, keine Zusatzresiduen; CPU `optimize_lm_cpu`, GPU `optimize_lm_gpu`), dann OPD und Status über RESIDUALS mit hartem Versagen. Die nominelle Bewertung läuft genauso mit ξ = 0, also ebenfalls kompensiert.
- **Pro Tupel** (Feld, Wellenlänge) aus OPD und Status der Pupillenquadratur (`tuple_quality`, [numerik.md, Abschnitt 16](numerik.md#16-tolerierung-rms-und-strehl-aus-der-quadratur-m8-t4)): RMS nach Abzug von Kolben und Kippung [Wellen bei λ_ref] und Strehl bei der Wellenlänge des Tupels.
- **Pro Stichprobe**: RMS gesamt √(Σ ŵ_f ŵ_λ RMS²) mit den normierten Gewichten der Merit, kleinster Strehl über die Tupel mit Gewicht > 0, Arbeitsabstand, Kompensation, Status und Zyklen des LM, Zahl hart versagter Strahlen.
- **Verteilungen** (`distribution_stats`): Mittel, Standardabweichung (n − 1), Minimum, Maximum über die endlichen Werte; Perzentile 50/90/95/99 % nach dem nächsten Rang ohne Interpolation in der schlechten Richtung (RMS: Wert, den p % nicht überschreiten, x_(⌈pn⌉); Strehl: Wert, den mindestens p % erreichen, x_(n+1−⌈pn⌉)); NaN zählt dort als schlechtester Wert. Ausbeute = Anteil mit kleinstem Strehl ≥ Schwelle (Standard 0,8).

### Determinismus

Stichprobe i hängt nur von (seed, i) ab; der LM bündelt Läufe ohne Kopplung (Abschnitt 4), RESIDUALS ist pro Variante unabhängig von der Batchgröße (ADR-0007), die Bewertung summiert in fester Reihenfolge. Ergebnisse sind daher für jede Tranchengröße und Threadzahl bitgleich, auf demselben Gerät auch auf der GPU. CPU und GPU stimmen innerhalb der Toleranzen überein (ohne Kompensator RMS und Strehl ≤ 3,3·10⁻¹², mit Kompensator Arbeitsabstand ≤ 1,6·10⁻¹⁰ mm, gemessen auf der RTX 3090).

### Abnahme (M8-T4)

| Prüfung | Orakel | Ergebnis (Test) |
|---|---|---|
| Verteilung über 10.000 Stichproben, `testsystem_toleranz.json` (alle acht Typen, 26 Zufallsgrößen), Kompensator Arbeitsabstand, Startwert 2026 | – | RMS nominal 0,0275, Mittel 0,0945, 95 % ≤ 0,214 Wellen; kleinster Strehl nominal 0,960, Mittel 0,615, ≥ 0,044 bei 95 %; Ausbeute (S ≥ 0,8) 39,1 %; keine versagten Strahlen. CPU 1,7 s, RTX 3090 0,5 s (`ToleranceAnalysis.TenThousandSamplesReproducible`, `CudaTolerance.TenThousandSamplesReproducible`) |
| Reproduzierbar mit festem Startwert | bitgleich | zweiter Lauf mit Tranchen zu 777 (GPU 1.234) und einem Thread: erste 2.500 (GPU 3.000) Stichproben bitgleich; anderer Startwert andere Werte |
| Kompensator | verschobene planparallele Platte: Kompensation = nominal − δ, RMS wie nominal | ≤ 1,08·10⁻⁶ mm, RMS relativ ≤ 2,0·10⁻⁵ (300 Stichproben, `ToleranceAnalysis.CompensatorRecoversShiftedPlate`) |
| Strehl der Quadratur | direkte Summe der PSF (M8-T2) auf dem Raster 256², Kippung abgezogen | ≤ 1,7·10⁻³ bei Strehl 0,14 … 0,98 (`ToleranceAnalysis.QuadratureStrehlMatchesDirectPsfSum`) |
| RMS und Strehl eines Tupels | Defokus A ρ²: RMS A/√12, Strehl sinc²(πA/λ); Kolben und Kippung; versagte Strahlen undurchlässig | ≤ 10⁻¹³ (`ToleranceQuality.*`) |
| Zufallsgrößen | Mittel und Varianz gegen Gleichverteilung und abgeschnittene Irwin-Hall-Dichte (c = 0,5 … 3) | innerhalb 5 Standardfehlern bei 200.000 Ziehungen (`ToleranceModel.*DeviateMoments`) |
| Anwendung | geschlossene Formeln pro Typ, Elementkippung um den Drehpunkt, Zusammensetzung zweier Drehungen | `ToleranceModel.*` |

## 12. Empfindlichkeiten und Herstellbarkeit (M10-T3)

Ziel: für einen Kandidaten sagen, welche Toleranz die Bildgüte am stärksten verschlechtert und welche Linsen schwer zu fertigen sind, als Diagnose für Mensch und Entscheider ([entwurfssteuerung.md, Abschnitt 3.4](entwurfssteuerung.md#34-neu-empfindlichkeiten-und-herstellbarkeit-m10-t3)). Bibliothek `Mikroptim::tolerance` (`include/mikroptim/tolerance/sensitivity.hpp`, `manufacturability.hpp`), Kommandozeile `mikroptim sensitivity` ([cli.md](cli.md#mikroptim-sensitivity)), Python-Aktion `sensitivity` und Feld `sensitivity` des Zustands (`mikroptim.design`). Die Bausteine der Monte-Carlo-Tolerierung (Abschnitt 11) werden wiederverwendet: Toleranzsatz und Auflösung (`resolve_tolerances`), Störungen (`ToleranceModel::apply`, ADR-0011), Trace (`tolerance_trace_cpu` bzw. RESIDUALS auf der GPU über `cuda::use_gpu`) und Bewertung (`tuple_quality`).

### Toleranzsatz

Der Abschnitt `tolerances` der Systemdatei, sonst (oder mit `--set default`) der **Mikroptim-Standardsatz** (`default_sensitivity_tolerances`): pro Element (`all_elements`, Linsen und Kittglieder als starrer Körper) Dezentrierung δ = 10 µm und Verkippung 1′ um die Elementmitte (Werte aus entwurfssteuerung.md, Abschnitt 3.4), für jede gekrümmte Kugelfläche Radius ±0,1 % relativ, für jede Dicke außer dem Arbeitsabstand ±20 µm und für jedes Glas Brechzahl ±5·10⁻⁴. Das ist die Größenordnung üblicher Fertigungstoleranzen für Präzisionsoptik, ohne Bezug auf einen bestimmten Hersteller; alle Werte sind auf der Kommandozeile einstellbar, 0 schaltet eine Art ab. Verteilung, Abschneiden und Kompensator des Satzes zählen hier nicht, nur die Grenzen δ.

### Bewertung und Störungen

Bewertet wird wie in der Tolerierung die mittlere quadratische Wellenfront nach Abzug von Kolben und Kippung ([numerik.md, Abschnitt 16](numerik.md#16-tolerierung-rms-und-strehl-aus-der-quadratur-m8-t4)), **ohne Kompensator**:

$$Q = \sum_{f,\lambda} \hat w_f\,\hat w_\lambda \Big(\frac{\mathrm{RMS}_{f\lambda}}{\lambda_{\mathrm{ref}}}\Big)^2 = \|\tilde e\|^2 \quad [\text{Wellen}^2],\qquad \tilde e_k = \sqrt{\frac{\hat w_f\,\hat w_\lambda\,W_k}{W_{\mathrm{ok}}}}\;\frac{e_k}{\lambda_{\mathrm{ref}}}$$

mit dem Rest e_k nach Abzug von Kolben und Kippung pro Stützstelle (`tuple_quality` mit der Ausgabe `rest`; versagte Strahlen 0). Q ist das Quadrat von „RMS gesamt“ der Tolerierung. Jede Toleranz hat eine oder zwei Komponenten (Dezentrierung x, y; Verkippung α, β), jede Komponente eine Zufallsgröße ξ mit dem Wert δ ξ. Ein Batch mit 1 + 4 D Varianten (D Komponenten, volle Pupille): nominal, dann pro Komponente ξ = +1, −1, +h, −h, alle übrigen 0 (`ToleranceModel::apply` ab der nominellen Variante, dieselben Formeln wie in Abschnitt 11). Ein Trace, Bewertung auf dem Host, Summen in fester Reihenfolge: bitgleich für jede Threadzahl und, auf demselben Gerät, auf der GPU.

Pro Komponente:

| Größe | Formel | Bedeutung |
|---|---|---|
| Q₊, Q₋ | Q bei ξ = ±1 | direkte Rechnung bei ±δ (Einzelrechnung im Batch) |
| a, b | a = (Q₊ − Q₋)/2, b = (Q₊ + Q₋ − 2 Q₀)/2 | Q(ξ) ≈ Q₀ + a ξ + b ξ² durch die drei Punkte |
| j | (ẽ(+h) − ẽ(−h)) / (2h) | Jacobi-Spalte der Wellenfrontreste nach ξ, also ∂ẽ/∂p mal δ („Spalte mal Toleranz“) |
| g, ‖j‖² | g = 2 ẽ₀ᵀ j, ‖j‖² | Ableitung dQ/dξ bei 0 und Gauß-Newton-Term: Q(ξ) ≈ Q₀ + g ξ + ‖j‖² ξ² |

Standard h = 0,01 (`--step`). Pro Toleranz: ΔQ_max = größtes Q± − Q₀ über Komponenten und Vorzeichen (Rangfolge der Ausgabe), RMS bei δ = √(Q₀ + ΔQ_max) und die Mittel von b und ‖j‖² über die Komponenten. Das Mittel ist die erwartete Zunahme von Q bei einer Störung der Größe δ in zufälliger Richtung (x/y bzw. α/β; die Mischterme mitteln sich weg) bzw. mit zufälligem Vorzeichen.

**Jacobi-Matrix und Einzelrechnung.** Die Spalten j bilden das lineare Modell ẽ(ξ) ≈ ẽ₀ + Σ_k j_k ξ_k für beliebige Kombinationen von Toleranzen; g ist die Ableitung von Q. Der Gauß-Newton-Term ‖j‖² lässt den Beitrag ẽ₀ᵀ ẽ″ der Krümmung der Reste weg. Bei Toleranzen, die Q kaum ändern, ist dieser vergleichbar groß (am Testsystem bis zum Faktor 2 zwischen ‖j‖² und b bei Wirkungen um 10⁻⁷ Wellen², bei den wirksamen Toleranzen < 1 %); einzelne Toleranzen werden deshalb nach den direkten Werten Q± bewertet und sortiert, Kombinationen und Ableitungen über die Jacobi-Spalten. Anders als in entwurfssteuerung.md, Abschnitt 3.4, ursprünglich gedacht, stammt die Jacobi-Matrix nicht aus dem LM am Optimum: Dessen Residuen enthalten die Kippung (Merit, Abschnitt 2), seine Variablen umfassen weder Brechzahlen ohne Buchdahl-Umrechnung noch Dezentrierungen, und nicht jeder tolerierte Parameter ist eine Variable. Die zentralen Differenzen nach den Toleranzparametern kosten einen Trace von 1 + 4 D Varianten (Testsystem: 105 Varianten, 0,2 s auf der CPU).

### Herstellbarkeit

`check_manufacturability` prüft nur die Geometrie der Datei (kein Trace; Geisterplatten zählen nicht), mit dem Halbdurchmesser h der Fläche bzw. h = max(h_s, h_{s+1}) für das Stück zwischen zwei Flächen. Pfeilhöhe und Randdicke wie bei den mechanischen Nebenbedingungen (Abschnitt 2.5). Befunde sind Warnungen (Grenzwert überschritten) oder Fehler (geometrisch unmöglich). Sie sind **nur Hinweise** für Mensch und Entscheider und gehen nie als harte Nebenbedingung in die Optimierung ein (Entscheidung Maintainer 2026-11-19); wer eine Grenze erzwingen will, setzt die mechanischen Nebenbedingungen aus Abschnitt 2.5. Grenzwerte ohne Quelle sind als **Mikroptim-Annahme** gekennzeichnet:

| Prüfung | Größe | Warnung (Standard) | Fehler | Herkunft der Grenze |
|---|---|---|---|---|
| `hemisphere` (Kugelfläche) | h \|c\| | > 0,95 | ≥ 1 (Halbdurchmesser ≥ Radius) | Zhang 2019, §3.2 (Hyperhemisphäre nicht fassbar), wie entwurfssteuerung.md, Abschnitt 3.4 |
| `center_thickness` (Glas) | t | < 0,5 mm | – | Mikroptim-Annahme |
| `edge_thickness` (Glas) | t − sag(c_s, h) + sag(c_{s+1}, h) | < max(0,5 mm; 0,05 · D), D = 2h Durchmesser des Elements | < 0 oder eine der Flächen mit h \|c\| ≥ 1 | Mikroptim-Annahme (durchmesserabhängig, Entscheidung Maintainer 2026-11-19) |
| `air_edge` (Luft zwischen zwei Flächen, nicht der Arbeitsabstand) | wie Randdicke | < 0 mm (Standard: keine Warnung) | < 0 (Berührung) oder h \|c\| ≥ 1 | – |
| `aspect_ratio` (Glas) | 2h / t | > 10 | – | Mikroptim-Annahme (Faustregel: dünne Linsen verformen sich beim Polieren und Fassen) |
| `centering` (Glas, Meniskusform) | Z = (h/2) \|c_s − c_{s+1}\| | < 0,1: „Glockenzentrierung ungeeignet (Z < 0,1), andere Zentrierverfahren nötig“ | – | Mikroptim-Annahme (Richtwert der Glockenzentrierung: flache Menisken zentrieren sich nicht selbst; die Linse bleibt herstellbar) |

Alle Grenzen sind einstellbar (`ManufacturabilityLimits`, CLI `--max-hemisphere`, `--min-center`, `--min-edge`, `--min-edge-relative`, `--min-air-edge`, `--max-aspect`, `--min-centering`); eine Grenze 0 schaltet die Warnung ab (Randdicke: beide Anteile 0), Fehler bleiben. `edge_thickness_limit` liefert die Grenze der Randdicke, `finding_message` den Hinweistext eines Befunds (JSON `message`, Textausgabe, Zustand von `mikroptim.design`). Gekittete Glieder werden pro Glas geprüft (jede Linse wird vor dem Kitten einzeln gefertigt und zentriert). Noch nicht umgesetzt aus entwurfssteuerung.md, Abschnitt 3.4: größter Einfallswinkel (braucht den Trace; als Nebenbedingung `max_incidence_angle` vorhanden), Glas in der Autofluoreszenz-Sperrzone (seit M10-T5 als Glasregel `af_forbidden` der Glasabfrage, noch nicht in dieser Prüfung) und Elementzahl gegen die Abgleichlänge (braucht die Spezifikation des Entwurfs).

### Abnahme (M10-T3)

Tests `tests/test_sensitivity.cpp`, `tests/test_manufacturability.cpp`, `tests/test_cli_sensitivity.cpp`, GPU `CudaSensitivity.*` in `tests/test_cuda_tolerance.cpp`. System: `testsystem_toleranz.json` mit einem Satz aus allen Typen außer Abbe (11 Toleranzen, 15 Komponenten; die Brechzahl am Buchdahl-Glas, damit die Datei sie als n₀ + Δn ausdrücken kann) und fester Sollbrennweite 20 mm (Bildpunkte unabhängig von der Störung, wie im Batch).

| Prüfung | Orakel | Ergebnis (Test) |
|---|---|---|
| Q± jeder Komponente (Radius relativ und in Passen, Dicke, Brechzahl, Flächen- und Elementdezentrierung, Flächen- und Elementverkippung um die Mitte) | Einzelrechnung: gestörte Systemdatei nach ADR-0011 (`decenter`, `tilt_deg` an den Flächen; Elementverkippung mit eigener Formel Scheitel P + R(V − P), Axialanteil in den Dicken), eigener Batch und eigene Bewertung | bitgleich (größte relative Abweichung 0; Toleranz `kSensitivitySingleSystem` = 10⁻¹⁰) (`Sensitivity.PerturbationsMatchSingleSystemFiles`) |
| Jacobi-Empfindlichkeit gegen finite Differenzen | (a) h = 0,01 gegen h = 0,005; (b) Sekante (ẽ(+δ) − ẽ(−δ))/2 aus den beiden Einzelrechnungen: ‖j‖² und 2 ẽ₀ᵀ j; (c) g gegen (Q₊ − Q₋)/2 | relativ zur größten Änderung von Q: (a) 7,1·10⁻⁶, (b) 4,8·10⁻⁶, (c) 2,5·10⁻³ (kubischer Term) (`Sensitivity.JacobianAgreesWithFiniteDifferences`) |
| Invariante Bewegungen | konzentrische Schale um ihren gemeinsamen Krümmungsmittelpunkt gekippt, planparallele Platte quer verschoben: dieselben Flächen | \|ΔQ\|/Q₀ ≤ 6,4·10⁻¹³, ‖j‖²/Q₀ ≤ 1,9·10⁻²⁰; Kontrolle (Linse um ihre Mitte gekippt) wirkt (`Sensitivity.InvariantMotionsHaveNoEffect`) |
| Symmetrie | nur das Feld auf der Achse: Q gerade in der Dezentrierung, x gleich y | ≤ 10⁻¹⁰ Q₀ (`Sensitivity.AxialFieldIsSymmetric`) |
| Determinismus, GPU | bitgleich für 1 und 7 Threads; GPU zweimal bitgleich; GPU gegen CPU (Toleranzsatz der Testdatei, 26 Komponenten) | GPU gegen CPU (RTX 3090): Q relativ 3,4·10⁻¹¹, Jacobi-Terme 6,4·10⁻⁷ relativ zur größten Änderung von Q (`Sensitivity.DeterministicForAnyThreadCount`, `CudaSensitivity.GpuMatchesCpuAndIsReproducible`) |
| Herstellbarkeit | frei erfundenes System: Randdicke, Luftrand, Halbkugelgrad und Z gegen die geschlossene Pfeilhöhe R − √(R² − h²) | ≤ 10⁻¹² mm; 4 Fehler, 6 Warnungen wie konstruiert (`Manufacturability.*`) |
| Randdicke durchmesserabhängig, Hinweistexte (Entscheidung Maintainer 2026-11-19) | Bikonvexlinse mit Rand 1 mm bei D = 30 mm (Grenze 1,5 mm: Warnung) und D = 8 mm (Grenze 0,5 mm: keine); Texte aller Befunde ohne „nicht herstellbar“ | wie konstruiert (`Manufacturability.EdgeLimitScalesWithDiameter`, `Manufacturability.FindingMessages`, `CliSensitivity.DefaultSetOptionsAndManufacturabilityOnly`) |

## 13. Operatoren der Entwurfssteuerung (M10-T4)

Strukturelle Aktionen für Mensch und Entscheider ([entwurfssteuerung.md, Abschnitt 4](entwurfssteuerung.md#4-aktionen-mikroptimdesignaction1)). Der Kern liefert die Geometrie, deterministisch und ohne Entscheider nutzbar (ADR-0012): `include/mikroptim/optim/design_operators.hpp` (Bibliothek `Mikroptim::optim`), Kommandozeile `mikroptim insert` ([cli.md](cli.md#mikroptim-insert)). Die Python-Schicht `mikroptim.design` (`operators.py`) prüft die Vorbedingungen, wählt Vorlagen und fährt Homotopien in Stufen mit Nachoptimierung über `mikroptim optimize`.

```cpp
SystemDescription remap_surfaces(const SystemDescription&, const SurfaceRemap&, std::vector<Surface>);
void freeze_variables(SystemDescription&, std::span<const int> curvatures,
                      std::span<const int> thickness_gaps, std::span<const int> glasses);
int glass_block_end(const SystemDescription&, int first);
bool is_null_element(const SystemDescription&, int first);
RemovalResult remove_element(const SystemDescription&, int first);
SystemDescription collapse_element(const SystemDescription&, int first, double fraction);
void replace_glass(SystemDescription&, int surface, const MaterialSpec&);
GhostDictionaryResult place_ghost_dictionary(const SystemDescription&, const GhostDictionaryPlacement&);
AplanaticShellResult insert_aplanatic_shell(const SystemDescription&, const GlassCatalog&,
                                            const AplanaticShell&);
InsertionResult insert_null_doublet(const SystemDescription&, const NullDoublet&);
```

**Umnummerierung mit entfernten Flächen** (`remap_surfaces`). Wie `renumber_surfaces` (Abschnitt 6), aber mit zwei Abbildungen: Flächen (Krümmungs- und Glasvariablen, Bezugsfläche der Abgleichlänge, mechanische Bedingungen pro Fläche) und Spalte (Dickenvariablen, mechanische Dicken-Bedingungen, Geisterplatten); −1 heißt entfernt, mehrere alte Spalte dürfen auf einen neuen fallen. Doppelte Bezüge bleiben nur beim ersten Vorkommen, auch über mehrere Einträge eines Typs; Einträge ohne Flächen entfallen; die letzte Fläche muss nicht die letzte bleiben (Schale vor dem Objekt). Eine entfernte Bezugsfläche der Abgleichlänge ist ein Fehler.

**Glasblock.** Flächen first … last: Luft vor first, Glas nach first … last − 1, Luft nach last (Einzellinse: last = first + 1; Kittglied mit k Gläsern: first + k). Ein **Null-Element** hat alle Glasdicken 0 und alle Krümmungen gleich; dann heben sich alle Brechungen auf (Abschnitt 6).

**Rückbau** (`remove_element`, Umkehrung von Operator A). Der Spalt nach first − 1 wird t_{first−1} + Σ t_first … t_last (fortlaufend summiert), Geisterplatten nach last liegen danach nach first − 1 mit distance + t_{first−1} + … + t_{last−1}; Dickenbezüge des Spalts nach last gehen auf den vereinigten Spalt über (liegt seine Dicke über der oberen Schranke der Dickenvariable bzw. des Arbeitsabstands, wird die Schranke auf die Dicke angehoben; ebenso in der Rückbau-Homotopie). Ist der Block die Frontgruppe, wird der vereinigte Spalt der Arbeitsabstand (keine Dickenbezüge an der letzten Fläche). Für ein Null-Element bleibt die Merit bis auf Rundung gleich, und nach Operator A bzw. `insert_null_doublet` ergibt sich das Ausgangssystem (die Spaltdicke bis auf die Rundung von p + fl(T − p)). Sonst geht dem Entfernen die **Rückbau-Homotopie** voraus (`collapse_element`): pro Stufe mit Anteil f des verbleibenden Wegs t ← (1 − f) t für die Glasdicken und c ← (1 − f) c + f c_first für die Krümmungen der Flächen first + 1 … last; die abgezogene Dicke kommt zum Spalt nach last (die Lage der folgenden Scheitel bleibt), Krümmungen, Glasdicken und Gläser des Blocks werden festgesetzt (`freeze_variables`). Mit f = 1/K, 1/(K − 1), …, 1 ist nach K Stufen das Null-Element genau erreicht (Dicken genau 0, Krümmungen genau c_first). `mikroptim.design` optimiert nach jeder Stufe die übrigen Variablen (`stages`, Standard 4; `cycles`, Standard 50).

**Glastausch** (`replace_glass`): Das Medium nach der Fläche wird das neue Glas, eine Glasvariable dieses Mediums wird festgesetzt; ob das Glas in den Katalogen steht, prüft das Laden. In `mikroptim.design` mit einem Namen ein Kandidat (Nachoptimierung nur mit `cycles` > 0), mit dem Filter `partner_of_element` die besten Partner aus der Glasvorauswahl (M11-T5) als getrennte Kandidaten, jeder nachoptimiert (Standard 50 Zyklen) und nach Merit geordnet; seit M10-T5 ebenso mit jedem anderen Filter der Glasabfrage `glass_query` (Glasregeln, Bereiche, Transmission; [entwurfssteuerung.md, Abschnitt 5](entwurfssteuerung.md#5-wissensbasis)), ohne das bisherige Glas.

**Geisterplatten-Wörterbuch** (`place_ghost_dictionary`). Eine Platte pro Glas an (after_surface, distance), Ordnung K, Koeffizienten 0, Normierungsradius Vorgabe oder der größere Halbdurchmesser der Spaltflächen. Die neuen Platten stehen hinter allen Platten mit (after_surface, distance) ≤ der Lage (die Sortierung der Datei bleibt); alle Plattenbezüge (Variablen `ghost_coefficient`, Strafe `ghost_penalty`) werden umnummeriert. Variablen: Deckt kein Eintrag mit `"all"` die neuen Platten ab, kommt ein Eintrag mit allen Ordnungen dazu. Strafe: mit eigenem Gewicht μ ein neuer Eintrag (w₀, κ, ε wie der erste vorhandene; Einträge mit `"all"` werden zur Liste der alten Platten, damit sie die neuen nicht doppelt abdecken), sonst decken vorhandene `"all"`-Einträge die Platten ab bzw. sie kommen in den ersten Eintrag mit Liste; ohne jede Strafe ist μ Pflicht. Platten mit a = 0 haben keine Dicke und keine Strafe: Die Merit bleibt gleich.

**Aplanatische Schalenlinse** (`insert_aplanatic_shell`, Vorlage `aplanatic_shell`; Welford, *Aberrations of Optical Systems*, aplanatische Flächen; Zhang 2019, §3.2). Der reale meridionale Randstrahl (ρ = 1, Achse, λ_ref, ohne Geisterplatten) schneidet im Spalt die Achse im Abstand Q von der ersten Schalenfläche (Lage `position`). In Tracerichtung ist die erste Fläche aplanatisch für das Medium n vor ihr und das Glas n′: Mit L = R (n + n′)/n und L′ = R (n + n′)/n′ (Young–Weierstraß-Punkte, für jeden Strahl durch den Punkt exakt) ist

$$R_1 = Q\,\frac{n}{n + n'},\qquad Q' = Q\,\frac{n}{n'},\qquad R_2 = Q' - t,$$

die zweite Fläche (Abstand t = `thickness`) also konzentrisch zum neuen Schnittpunkt Q′. Beide Flächen tragen zur sphärischen Aberration aller Ordnungen (numerik.md, Abschnitt 14, Invariante Q = n u sin U δL) **nichts** bei: An der konzentrischen Fläche ist U′ = U und L′ = L = 1/c, und mit n′u′ = n u − y c (n′ − n) und u l = −y wird

$$n'u'(L - l') - n u (L - l) = L\,(n'u' - n u) + y\,(n' - n) = y\,(n' - n)(1 - cL) = 0;$$

an der aplanatischen Fläche ist I′ = U und I = U′, also sin U′ = (n′/n) sin U, und mit L, L′ von oben heben sich in sin U′ (n′u′ L′ + n′y) − sin U (n u L + n y) alle Terme auf. Die Flächen vor der Schale sehen dieselben Strahlen (Beiträge bitgleich), die Flächen danach ein um n′/n steileres Bündel. Liegt die Schale im Arbeitsabstand (neue Frontlinse vor dem Objekt), bleibt deshalb die Gruppensumme der Frontgruppe erhalten, während die Apertur um n′ wächst – das Prinzip der aplanatischen Frontlinsen. Danach wird der Arbeitsabstand so gesetzt, dass der Abstand der Objektebene zum paraxialen Fokus bleibt (`refocus`, Standard an). Dickenvariablen und mechanische Dicken-Bedingungen des geteilten Spalts gelten für beide Teilstücke, Geisterplatten wie bei Operator A.

**Null-Kittglied** (`insert_null_doublet`, Vorlage `null_doublet`): Operator A mit Dicke 0, dann Operator B mit einer Kittfläche derselben Krümmung c₀ (Anteil 0,5, beide Teile Dicke 0) und dem zweiten Glas dahinter: drei Brechungen an derselben Stelle mit derselben Normale, Merit unverändert. **Gauss-Paar** (Vorlage `gauss_pair`, nur Python): zwei Null-Menisken im Abstand 0,8 min(p, T − p) um die Lage p, Krümmungen ∓c konzentrisch zur Mitte zwischen ihnen (|c| höchstens 0,9/h), Hohlseiten einander zugewandt. Vorlagen mit Dicke 0 wachsen in `insert_module` mit `thickness` über die Dicken-Homotopie (Stufen t_k = t k/K, der Luftspalt dahinter wird entsprechend kürzer, nach jeder Stufe Nachoptimierung).

**Gruppe ersetzen** (`replace_group`, nur Python): alle Glasblöcke der Gruppe (Zuordnung wie im Zustand über die Randstrahlhöhen; ein Kittglied mit einem Element in der Gruppe wird ganz entfernt) über den Rückbau entfernen, dann die Module gleichmäßig verteilt in den vereinigten Spalt einfügen. **Gerüst** (`start_from_skeleton`, nur Python): neuer Wurzelkandidat; pro Modul eine Linse (`gauss_pair`: zwei) bzw. ein Kittglied (`null_doublet`), gleichseitig gebogen mit der dünnen Brechkraft s·w (Zeichen P: w = 1, N: −0,5, bei `PPN_strong_retrofocus` −1, 0: Platte), Kittglied mit ebener Kittfläche und je halber Brechkraft pro Glas; s per Sekante so, dass die paraxiale Brennweite gleich der des Kandidaten ist (`target.focal_length`, sonst paraxial; relativ 10⁻⁹), Blende f · `start_aperture`, Mittendicke 0,3 f, Luftspalte 0,15 f, Arbeitsabstand = hintere Schnittweite. Diese Startwerte (Gewichte w der Zeichen, Mittendicke, Luftspalte, Schranken der Luftspalte) sind **Mikroptim-Annahmen** ohne Quelle, nur als Ausgangspunkt der Nachoptimierung gedacht; die Brechkraftverteilungen selbst stammen aus Zhang Tab. 6.2. Variablen: alle Krümmungen, Luftspalte (0,1 … max(20, 5f) mm) und Arbeitsabstand; Nebenbedingungen ohne Flächen- oder Plattenbezug übernimmt das Gerüst. Die Zeichen gelten pro Gruppe für alle Elemente oder eines pro Element (Gruppen und Listen von der Objektseite aus). Seit M10-T5 stehen Verteilungen und Gewichte in der Wissensbasis (`data/design/knowledge/power_distributions.json`, auch `double_gauss`: PN/P/NP, Mikroptim-Annahme), Module mit den Vorlagen `strong_doublet` (Kittglied) bzw. `petzval_group` (zwei Linsen) sind erlaubt, und ohne Lösung mit positivem Faktor s werden Dicken und Abstände halbiert (höchstens zweimal, `length_scale`); vorher schlug die Sekante bei langen Gerüsten fehl oder hätte mit s < 0 alle Zeichen umgedreht.

**Starkes Kittglied** (Vorlage `strong_doublet`, M10-T5, nur Python): Null-Kittglied wie oben, dessen gemeinsame Krümmung c so gewählt ist, dass der paraxiale Randstrahl an der Lage p (Höhe y = y_s + p u, Steigung u hinter Fläche s) die Kugel unter dem Winkel `incidence_deg` trifft (Schnitt Strahl–Kugel exakt, Bisektion auf jedem Vorzeichenast über |c| ≤ 0,9/h, die kleinere Lösung); Merit unverändert. **Petzval-Gruppe** (Vorlage `petzval_group`, M10-T5, nur Python): zwei Null-Menisken im Abstand d wie beim Gauss-Paar, dann Mittendicke t und Rückflächen aus der dicken Linse φ = (n − 1)(c₁ − c₂) + t (n − 1)² c₁c₂/n mit φ₁ = `power`/f und φ₂ = −φ₁/(1 − d φ₁) (afokal in Dünnlinsennäherung); die Luftspalte dahinter werden um t kürzer, Dickenvariablen dort bekommen höchstens den neuen Wert als untere Schranke. Ändert die Merit.

### Abnahme (M10-T4)

Tests `tests/test_design_operators.cpp`, `tests/test_cli_insert.cpp` (`CliInsert.DesignOperatorsMatchLibrary`: Ergebnisdatei gleich der Bibliothek, JSON-Bericht), Python `python/tests/test_design_operators.py`. Systeme frei erfunden (`testsystem_mehrlinser.json`, `einzellinse_geisterplatten_exakt.json`).

| Prüfung | Orakel | Ergebnis (Test) |
|---|---|---|
| Merit-Invarianz `place_ghost_dictionary` (Plan: wie Abschnitt 6) | F vor und nach dem Einfügen; 7 Fälle: Lage zwischen, an und vor vorhandenen Platten, hinter der Linse, Systeme ohne Platten, Plattenbezüge als `"all"` und als Listen, mit und ohne eigenes μ; alte Platten mit Koeffizienten ≠ 0, damit falsch umnummerierte Bezüge auffielen | 5 von 7 bitgleich, größte relative Änderung 1,1·10⁻¹⁵ (`kInsertionMeritRelative` = 10⁻¹¹) (`DesignOperators.GhostDictionaryLeavesMeritUnchanged`); Python (`test_place_ghost_dictionary_merit_unchanged`) |
| aplanatische Vorlage: Gruppensumme der sphärischen Aberration beim Einfügen (Plan: < 10⁻³) | Σ S_FSPH der Frontgruppe (Element 6–7 des Mehrlinsers) vor und nach dem Einfügen der Schale vor dem Objekt; N-BK7 und CaF₂, zwei Lagen | größte relative Änderung 1,5·10⁻¹³ (`kAplanaticGroupSpherical` = 10⁻³); Gegenprobe mit Krümmungen der Schale ×1,1: ≥ 0,40 (`DesignOperators.AplanaticShellKeepsFrontGroupSphericalSum`); im Inneren (Spalt 3 und 5) Beiträge der Schalenflächen ≤ 5,9·10⁻¹⁴ des größten Flächenbeitrags, Flächen davor bitgleich (`…InsideSystemAddsNoSphericalAberration`); Python über `insert_module` mit `front.shell_ca` (`test_aplanatic_module_keeps_group_sum`) |
| `remove_element` ∘ `insert_null_meniscus` = Ausgangssystem | Flächen, Geisterplatten, aufgelöste Variablen, Flächen der mechanischen Bedingungen, Merit; 8 Fälle (Spalte, Lagen, c₀, Gläser; Platten vor, an und hinter der Lage; Krümmungs- und Dickenvariablen mit `"all"` und Listen, Mindestdicke) | alles gleich außer der Spaltdicke p + fl(T − p) (≤ 1 ulp, `kRemovalRoundtripRelative` = 4,5·10⁻¹⁶), Merit ≤ 10⁻¹¹ (`DesignOperators.RemoveAfterNullMeniscusRestoresSystem`); ebenso nach `insert_null_doublet` (`…NullDoubletRestoresSystem`) und über die Aktionen (`test_remove_after_null_meniscus_restores_system`) |
| Rückbau-Homotopie | 4 Stufen am Kittglied: Null-Element genau erst nach der letzten Stufe, Scheitel dahinter bleiben, Variablen des Blocks fest, Entfernen ändert F nicht | (`DesignOperators.CollapseReachesNullElementAndRemoves`, Python `test_remove_element_homotopy`) |

## 14. Toleranzbewusste Optimierung (M11-T6)

Ziel: von nominal gleichwertigen Lösungen die unempfindlichere wählen, indem die Optimierung die erwartete Merit über feste Störungen eines Toleranzsatzes minimiert ([optimierung-erweiterungen.md, Abschnitt 6](optimierung-erweiterungen.md#6-toleranzbewusste-optimierung-m11-t6)). Bibliothek `Mikroptim::tolerance` (`include/mikroptim/tolerance/tolerance_aware.hpp`, [architektur.md](architektur.md#monte-carlo-tolerierung-m8-t4)), Kommandozeile `mikroptim optimize --tolerance-aware α` ([cli.md](cli.md#toleranzbewusste-optimierung-ab-m11-t6)); Grundsatzentscheidung [ADR-0018](adr/0018-optionale-merit-terme.md). Standard aus; ohne die Option laufen alle Pfade unverändert (bitgleich).

### Merit

$$F_{\text{tol}}(x) = (1 - \alpha)\,F(x) + \frac{\alpha}{S}\sum_{s=1}^{S} F(x \oplus \delta_s),\qquad 0 < \alpha \le 1,$$

mit F = ‖r‖² wie in Abschnitt 2 (Strahl-, Straf-, Zusatz- und trace-abhängige Residuen in Wellen², Bildpunkte fest). Gleichwertig ist der Residuenvektor [√(1 − α)·r(x); √(α/S)·r(x ⊕ δ₁); …; √(α/S)·r(x ⊕ δ_S)]. x ⊕ δ_s ist das System mit der Störung s nach `ToleranceModel::apply` (Abschnitt 11: Radius, Dicke, Brechzahl, Abbe-Zahl, Dezentrierung und Verkippung von Flächen und Elementen nach ADR-0011), ohne Kompensator.

**Störungen** (deterministisch, `ToleranceAwareScheme`):

| Schema | Störungen | Bedeutung |
|---|---|---|
| `plus_minus` (Standard) | S = 2D: für jede der D Zufallsgrößen ξ_d = +1 und ξ_d = −1, alle übrigen 0 | mit Q(ξ) ≈ Q₀ + a ξ + b ξ² (Abschnitt 12) ist der gestörte Anteil (1/S) Σ F = F₀ + (1/D) Σ_d b_d + O(ξ⁴): das Mittel der quadratischen Terme, also die mittlere Krümmung von F in den Toleranzrichtungen |
| `samples` | S Stichproben ξ_s = `deviates(seed, s − 1)` mit Verteilung und Abschneiden des Satzes (Abschnitt 11) | Stichprobenmittel der erwarteten Merit unter der Toleranzverteilung (gemeinsame Zufallszahlen: dieselben ξ_s in jedem Zyklus) |

Toleranzsatz: der Abschnitt `tolerances` der Datei, sonst (oder mit `--tolerance-set default`) der Mikroptim-Standardsatz der Empfindlichkeiten (Abschnitt 12).

**Störungen von Variablen.** Ist der gestörte Parameter eine Variable (relativer Radius bzw. Passe an einer variablen Krümmung, Dicke an einem variablen Spalt, axialer Anteil δz einer Elementbewegung an einer variablen Dicke oder am Arbeitsabstand), würde die Variable die Störung beim Schreiben von x überschreiben. Die Störung wirkt deshalb über den Variablenvektor: Das gestörte System an x ist apply_s(x) (nominelles System mit x, darauf `ToleranceModel::apply`), sein Variablenvektor x′_s = extract(apply_s(x)); die Vorlage der Störung s (einmal pro LM-Aufruf gebaut) trägt die übrigen, festen Störungen (Transformationen, Brechzahlen, nicht variable Radien und Dicken). Die Abbildung ist pro Variable affin, x′_k = x_k/(1 + ε) für relative Radien, sonst x_k + konstant, also

$$J_s(x) = J(x'_s)\,D_s,\qquad D_s = \operatorname{diag}\!\Big(\frac{\partial x'_{s,k}}{\partial x_k}\Big),\qquad \frac{\partial x'_{s,k}}{\partial x_k} = \prod \frac{1}{1 + \varepsilon}$$

(Produkt über die relativen Radius-Toleranzen an der Krümmung k, sonst 1). Die inneren Läufe nutzen die Variablen ohne Schranken (x′_s darf außerhalb liegen; zentrale Differenzen). Brechzahl- und Abbe-Toleranzen an einem Glas mit Glasvariablen sind so nicht darstellbar (die Glasvariable schreibt die Brechzahlen aus den Buchdahl-Koordinaten neu) und werden abgelehnt. Elementbewegungen rechnen wie in Abschnitt 11 mit den nominellen Scheitellagen der Datei, nicht mit den aktuellen Dicken; der Fehler ist von der Ordnung Kippwinkel × Dickenänderung.

### Zusammensetzung im gebündelten LM

`ToleranceAwareBackend` ist ein `LmBackend` (Abschnitt 4) über einem inneren Backend (CPU `CpuLmBackend`, GPU `cuda::GpuLmBackend`). Zu jedem Lauf r mit Vorlage b gehören 1 + S innere Läufe mit den Vorlagen b(1 + S) + s (s = 0 nominal), alle Läufe eines Aufrufs in **einem** Aufruf des inneren Backends: ein FD- bzw. adjungierter Batch und ein Merit-Batch, ein Block pro (Variante, Feld, Wellenlänge) (ADR-0004). Die gestörten Systeme sind also gewöhnliche Varianten im Batch, die Jacobi-Matrix entsteht ohne Sonderfall. Pro Lauf in fester Reihenfolge s = 0 … S mit w₀ = 1 − α, w_s = α/S:

$$A = \sum_s w_s\,D_s A_s D_s,\qquad g = \sum_s w_s\,D_s g_s,\qquad F = \sum_s w_s F_s,$$

A_s = J_sᵀJ_s und g_s = J_sᵀr_s aus dem inneren Backend am Punkt x′_s; versagte und hart versagte Strahlen werden über alle Teilsysteme summiert (ein Schritt, der in einem gestörten System einen Strahl hart versagen lässt, wird abgelehnt). Das ist die Gauß-Newton-Matrix des Residuenvektors oben bis auf die Summationsreihenfolge (Teilsumme pro Teilsystem, dann gewichtet). Das äußere Strafverfahren (`optimize_lm_rounds`, M4-T2) und der Augmented Lagrangian (Abschnitt 2.6) laufen unverändert darüber.

**Batch.** Die inneren Batches sind nicht koaxial mit voller Pupille (`make_non_coaxial`), auch für das nominelle Teilsystem: F₀ entspricht bis auf die Rundung der koaxialen Merit (dieselbe Wellenfront über die volle Pupille), ist aber nicht bitgleich dazu. Kosten pro Zyklus etwa (1 + S)-fach in Varianten, dazu der Faktor der vollen Pupille (Achsfelder: M Speichen statt einer, übrige Felder 2) und die Host-Arbeit für x′_s (vernachlässigbar); Messung unten.

**Determinismus.** Störungen hängen nur vom Toleranzsatz, dem Schema und (bei `samples`) dem Startwert ab; die inneren Läufe sind wie im LM unabhängig voneinander (Abschnitt 4), die Zusammensetzung pro Lauf hat eine feste Reihenfolge. Läufe gemeinsam und einzeln sind bitgleich, auf derselben GPU auch dort; keine atomaren Gleitkommaoperationen.

**Grenzen der ersten Version.** Die gestörten Systeme werden mit der vollen Merit bewertet, also einschließlich der Kippung (Bildlage fest, Abschnitt 2). Eine reine Bildverschiebung, wie sie die Dezentrierung einer Linse mit Brechkraft vor allem erzeugt, zählt damit als Verschlechterung, während Tolerierung und Empfindlichkeiten (Abschnitte 11, 12) nach Abzug der Kippung bewerten. Das behebt der Kippungsabzug (M11-T6b, unten, Teiloption `remove`); Standard bleibt `keep` (bitgleich zu M11-T6). Kompensatoren pro Störung (Arbeitsabstand) wären eigene Variablen und fehlen weiter.

### Abnahme (M11-T6)

Orakel: `data/systems/kittblock_toleranzbewusst.json` (frei erfunden): Glasblock aus zwei Teilen desselben Glases mit gekrümmter Kittfläche c im kollimierten Strahl vor einer Sammellinse (Achse, 546 nm), Variablen c und Arbeitsabstand, Toleranzsatz Brechzahl des hinteren Teils ±0,002. Nominal ist die Kittfläche unsichtbar (gleiche Brechzahl beidseits: keine Brechung, gleiche optische Weglänge), F(c) = F(0) für jedes c. Gestört hat sie die Brechkraft Δn·c; bei c = 0 bleibt nur die ebene Rückfläche im kollimierten Strahl, deren Wirkung ein Kolben ist, also F(0, ±Δn) = F(0). Lösung A (c = 0,05/mm) und Lösung B (c = 0) sind nominal gleichwertig, B ist unempfindlich. Tests `tests/test_tolerance_aware.cpp`, GPU `CudaToleranceAware.*` in `tests/test_cuda_tolerance.cpp`, CLI `CliOptimize.ToleranceAware` und `cli_optimize_tolerance_aware`.

| Prüfung | Orakel | Ergebnis (Test) |
|---|---|---|
| nominal gleichwertig, verschieden empfindlich | F(A) = F(B); F(B ± Δn) = F(B); F(A ± Δn) > 2 F(A) | F(A) = F(B) = 5,294667337·10⁻³ Wellen² bitgleich; B gestört bitgleich wie nominal; A gestört 1,643·10⁻² bzw. 1,648·10⁻² (`ToleranceAware.OracleNominallyEqualDifferentSensitivity`) |
| Einzelrechnung | gestörte Systemdatei mit n₀ ± Δn, eigener Batch | relativ ≤ 10⁻¹⁰ (`kSensitivitySingleSystem`, ebenda) |
| Auswahl | F_tol(A) > F_tol(B) bei α = 0,5 | 1,0875·10⁻² gegen 5,2947·10⁻³ (`ToleranceAware.PicksInsensitiveSolution`) |
| Optimierung ab A | gewöhnlich bleibt c = c_A; toleranzbewusst c → 0 | gewöhnlich c = 0,05 (0 Zyklen, `converged_merit`); toleranzbewusst c = −5,9·10⁻⁸/mm nach 2 Zyklen (`converged_merit`), nominale Merit unverändert, gestörte gleich der nominalen (ebenda) |
| Monte-Carlo-Tolerierung (M8-T4, 200 Stichproben, gleichverteilt, ohne Kompensator) bestätigt | – | RMS Mittel A 0,0958, Ergebnis 0,0728 Wellen (jede Stichprobe = nominal); 95 % A 0,124, Ergebnis 0,0728 (`ToleranceAware.MonteCarloConfirmsBetterDistribution`) |
| Störungen von Variablen | Teilsysteme gegen den direkt gestörten Batch; 2Jᵀr gegen die zentrale Differenz von F_tol (relative Radien 5 % an zwei variablen Krümmungen, variabler Luftspalt, Elementverkippung 20′ mit δz in Spalt und Arbeitsabstand; beide Schemata) | bitgleich; Gradient 2,4·10⁻⁷ relativ zur größten skalierten Komponente (ohne D_s: 0,94) (`ToleranceAware.ChainRuleForPerturbedVariables`) |
| Determinismus | drei Läufe gemeinsam gegen einzeln, beide Schemata | bitgleich (x, Verlauf, Zyklen, Status) (`ToleranceAware.JointRunsBitwiseEqualToSingleRuns`) |
| GPU (RTX 3090) | Teilmerits und Ergebnis gegen die CPU, zwei Läufe | Teilmerits 1,5·10⁻¹¹ relativ; c = 1,9·10⁻⁷/mm (CPU −5,9·10⁻⁸), F_tol relativ 4·10⁻¹²; zwei GPU-Läufe bitgleich (`CudaToleranceAware.PicksInsensitiveSolutionLikeCpu`) |

### Kosten (M11-T6)

Gemessen mit `mikroptim optimize data/systems/zweilinser_petzval.json --hard-failure --tolerance-aware 0.5 --tolerance-set default` (Standardsatz: 14 Toleranzen, 18 Zufallsgrößen, S = 36; Felder 0° und 2°, 10 × 8 Quadratur), Zeit pro LM-Zyklus nach Abzug des Laufs mit `--max-iterations 0`, Release-Build, bestes von drei:

| Backend | gewöhnlich | toleranzbewusst | Faktor |
|---|---|---|---|
| CPU (ein Prozess, Host-Threads nach der Arbeit) | 0,24 ms | 16,7 ms | ≈ 71 |
| GPU (RTX 3090) | 0,50 ms | 1,3 ms | ≈ 2,7 |

Erwartet ist auf der CPU etwa (1 + S) × Faktor der vollen Pupille = 37 × 3,2 (Achsfeld 10 → 80 Strahlen, Feld 2° 40 → 80) ≈ 118; dass es weniger ist, liegt an der festen Host-Arbeit pro Zyklus, die sich auf mehr Varianten verteilt. Auf der GPU füllen die zusätzlichen Varianten vor allem ein sonst kaum ausgelastetes Gerät. Mit S Stichproben (`--tolerance-samples S`) ist der Faktor entsprechend (1 + S)/(1 + 2D) davon.

### Kippungsabzug (M11-T6b)

Mit `ToleranceAwareSettings::tilt = Remove` (`optimize`/`synth --tolerance-tilt remove`) werden die gestörten Teilsysteme s ≥ 1 wie in der Tolerierung und den Empfindlichkeiten bewertet: Strahlresiduen nach Abzug von Kolben und Kippung pro Tupel. Das nominelle Teilsystem s = 0 behält die volle Merit (Bildlage und Verzeichnung sind dort Entwurfsziel, Abschnitt 2). **Standard der CLI ist `remove`** (Entscheidung Maintainer 2026-11-19, [ADR-0018](adr/0018-optionale-merit-terme.md), Nachtrag): `optimize --tolerance-aware` und `synth --tolerance-aware` ohne `--tolerance-tilt` bewerten die gestörten Systeme ohne Kippung, weil sonst eine reine Bildverschiebung (Dezentrierung einer Linse mit Brechkraft) als Verschlechterung zählt, die Tolerierung und Empfindlichkeiten nicht so bewerten. `keep` (volle Merit, bitgleich zu M11-T6 und zum früheren Standard) bleibt als `--tolerance-tilt keep`. In der Bibliothek bleibt `ToleranceAwareSettings::tilt` mit Standard `Keep` (Aufrufer setzen die Art ausdrücklich).

**Definition.** Eine Anpassung für alle drei Verwendungen (`PistonTiltFit`, `include/mikroptim/tolerance/piston_tilt.hpp`; `tuple_quality` nutzt sie seit M11-T6b, bitgleich wie vorher): gewichtete kleinste Quadrate v_k ≈ c₀ + t_x x_k + t_y y_k über die Stützstellen K des Tupels mit x = ρ sin φ, y = ρ cos φ und den Quadraturgewichten W_k, 3 × 3-Normalgleichungen mit Summen in aufsteigender Reihenfolge, Cholesky, bei einem Pivot ≤ 10⁻¹²·Σ_K W nur der Kolben ([numerik.md, Abschnitt 16](numerik.md#16-tolerierung-rms-und-strehl-aus-der-quadratur-m8-t4)). In der Merit sind die Strahlresiduen r_k = s_{f,l}√W_k OPD_k mit der OPD nach Kolbenabzug über die Strahlen in_piston; der Kippungsabzug (`TiltProjection`) ist mit K = {k : in_piston} und b = Σ_K √W_k (1, x_k, y_k) r_k

$$(c_0, t_x, t_y) = \text{solve}(b),\qquad r_k \leftarrow r_k - \sqrt{W_k}\,(c_0 + t_x x_k + t_y y_k)\quad (k \in K),$$

die orthogonale Projektion P_s auf das Komplement von span{√W(1, x, y)} über K. Es gilt P r = s √W e mit dem Rest e der Anpassung der OPD, also ‖P_s r_s‖² über die Strahlzeilen = Σ ŵ_f ŵ_λ (RMS_fλ/λ_ref)² = Q der Tolerierung (ohne Versagen; mit hart versagten Strahlen ist der Beitrag eines Tupels W_ok·RMS² wie in der nominellen Merit, Q dagegen renormiert). Strafresiduen des weichen Versagens und Zusatzresiduen bleiben unverändert. Im Achsfall (eine Speiche) und bei weniger als drei gültigen Strahlen ändert sich nichts (die Residuen sind schon kolbenfrei).

**Jacobi-Matrix.** P_s ist linear und hängt nur von der Maske K am Punkt x′_s ab, also J(P_s r_s) = P_s J_s bei fester Maske (FD-Spalten aus Varianten mit gleicher Maske, adjungierte Zeilen ebenso). Das Backend holt dafür J_s, r_s und den Status vom inneren Backend auf den Host (`LmBackend::host_jacobians`; CPU: dieselbe Rechnung wie `normal_equations`, Status der Variante x₀ des FD-Batches bzw. der adjungierten Rechnung; GPU: Download von `DeviceFdJacobian` bzw. `DeviceAdjointJacobian`, Status bei FD aus einem Lauf RESIDUALS über die Punkte), projiziert r_s und jede Spalte von J_s und bildet

$$A_s = (P_s J_s)^\top (P_s J_s),\qquad g_s = (P_s J_s)^\top (P_s r_s),\qquad F_s = \lVert P_s r_s \rVert^2$$

mit `normal_equations_host` (Summen aufsteigend); s = 0 auf demselben Weg ohne Projektion. Die Merit der Versuchspunkte kommt aus `LmBackend::host_residuals` (CPU `residuals_cpu`, GPU Modus RESIDUALS; Zusatzresiduen wie `fd_extra_residuals`), F_s = Σ (P_s r_s)² aufsteigend, versagte Strahlen aus dem Status. Die Zusammensetzung A = Σ w_s D_s A_s D_s usw. ist unverändert. Keine Reduktion auf dem Gerät mit Atomics, Läufe gemeinsam und einzeln bitgleich. Kosten: J der 1 + S inneren Läufe wird heruntergeladen (P·R Zahlen pro Lauf) und die Gram-Matrix auf dem Host gebildet; am Kittblock (P = 2) nicht messbar, bei großen Systemen bestimmt die Bandbreite den Mehraufwand.

**Warum Kippungsabzug statt Bildlage-Kompensator.** Die Abnahme verlangt Übereinstimmung mit der Bewertung der Monte-Carlo-Tolerierung, und die zieht die Kippung ab (Abschnitt 11). Ein Bildlage-Kompensator (zwei Variablen pro Störung für den Bildpunkt) ist in linearer Näherung dieselbe Projektion (Variablenprojektion: Minimum über die Bildlage ⇒ P J), braucht aber eigene Variablen und innere Optimierungen pro Störung und weicht für endliche Verschiebungen von der Definition der Tolerierung ab. Der Arbeitsabstand-Kompensator der Tolerierung (Fokus) ist nicht enthalten: F_tol entspricht der Tolerierung **ohne** Kompensator (`compensator: none`).

### Synthese und Strahlsuche (M11-T6b)

`synth --tolerance-aware α [--tolerance-samples S] [--tolerance-seed n] [--tolerance-tilt keep|remove]` (Standard aus) setzt `OpticalHomotopyOptions::system_optimizer`: Jeder Schritt (auch innere Homotopien der Operatoren und Kandidaten der Strahlsuche) minimiert F_tol mit `optimize_lm_tolerance_aware_standard`. Der Toleranzsatz ist an jedem Punkt der **Mikroptim-Standardsatz der realisierten Beschreibung** (`tolerance_aware_standard_set`: `default_sensitivity_tolerances`, Abschnitt 12, ohne Brechzahltoleranzen an Gläsern mit Glasvariablen, die nicht darstellbar sind): Die Synthese fügt Elemente ein und nummeriert die Flächen um, ein Toleranzsatz der Datei würde dabei nicht mitgeführt. `HomotopySolve::merit` ist dann F_tol; damit ordnet die Strahlsuche ihre Kandidaten nach F_tol (`beam_rank_before`), und die Wahl zwischen den Starts am Sattelpunkt folgt ebenfalls F_tol. Die Bewertung F_B (Annahme eines Schritts, Schwelle F_max), das Beugungsziel am Ende und der FP32-Vorfilter bleiben nominal. Inneres Backend CPU bzw. mit `--gpu` `cuda::gpu_tolerance_inner`. `--global` (M11-T7) mit `--tolerance-aware` bleibt gesperrt.

### Abnahme (M11-T6b)

Tests `tests/test_tolerance_aware.cpp`, `tests/test_piston_tilt.cpp`, GPU `CudaToleranceAware.TiltRemovalMatchesCpu`, CLI `CliOptimize.ToleranceAwareTilt`, `CliSynth.ToleranceAware`. System: Kittblock (oben) mit Toleranzen der Sammellinse.

| Prüfung | Orakel | Ergebnis (Test) |
|---|---|---|
| Reine Dezentrierung (10 µm) der Sammellinse (f ≈ 19 mm, NA ≈ 0,1) verschiebt das Bild und verschlechtert F_tol nicht | keep: Bildverschiebung zählt; remove: F(±δ) = F₀ bis auf die echte Änderung der Bildgüte, und diese gleich Q± der Empfindlichkeiten | keep: F(±δ) bis 168·F₀, F_tol = 0,447 gegen F₀ = 5,29·10⁻³; remove: \|F(±δ) − F₀\|/F₀ ≤ 6,2·10⁻⁴ (`kToleranceAwareShiftOnly` = 10⁻³), F_tol = 5,296·10⁻³; gegen Q± 4,9·10⁻¹⁶ relativ (`ToleranceAware.TiltRemovalDecenterOnlyShiftsImage`) |
| F_tol stimmt mit der Bewertung der Tolerierung überein | Teilsystem s gegen RMS_gesamt² der Monte-Carlo-Stichprobe s − 1 (20 Stichproben, gleicher Startwert, ohne Kompensator; Dezentrierung, Verkippung, Radien, Brechzahl) und gegen Q± der Empfindlichkeiten (7 Komponenten) | 9,4·10⁻¹⁶ bzw. 1,5·10⁻¹⁵ relativ (`kToleranceAwareVsTolerancing` = 10⁻¹²); Kontrolle keep: Mittel der gestörten F 0,490 gegen 0,0126 (`ToleranceAware.TiltRemovalMatchesTolerancing`) |
| Projektion = Rest der Tolerierung | `tuple_quality` am dezentrierten Mehrlinser (3 Felder, 4 Wellenlängen), hart und weich; idempotent, orthogonal zu √W(1, x, y); Achsfeld unverändert | 9,9·10⁻¹⁶, Q 4,8·10⁻¹⁶, idempotent 5,3·10⁻¹⁵, orthogonal 2,0·10⁻¹⁵ (`TiltProjection.*`, `PistonTiltFit.*`) |
| Jacobi-Matrix | 2g gegen die zentrale Differenz von F_tol (Störungen von Variablen, Dezentrierung 50 µm, Verkippung 20′), beide Schemata, FD und adjungiert | 2,8·10⁻⁷ relativ zur größten skalierten Komponente (`ToleranceAware.TiltRemovalGradientMatchesFiniteDifferences`) |
| Zeilen des CPU-Backends | A, g aus `host_jacobians` gegen `normal_equations`; ‖r‖² aus `host_residuals` gegen die Merit | bitgleich; ≤ 10⁻¹² (`ToleranceAware.CpuBackendHostRows`) |
| Determinismus | Läufe gemeinsam gegen einzeln (remove) | bitgleich (`ToleranceAware.TiltRemovalJointRunsBitwiseEqualToSingleRuns`) |
| Ohne Option bitgleich | Ergebnisse von M11-T6 unverändert (keep ist Standard, eigener Codepfad nur mit remove) | alle Werte von M11-T6 oben unverändert, z. B. F(A) = 5,294667337·10⁻³, c = −5,878·10⁻⁸, GPU c = 1,936·10⁻⁷ (`ToleranceAware.*`, `CudaToleranceAware.PicksInsensitiveSolutionLikeCpu`) |
| GPU (RTX 3090) | Teilmerits und Gradient (FD und adjungiert) gegen die CPU; zwei GPU-Optimierungen | 2,3·10⁻¹¹ bzw. 2,3·10⁻⁷ relativ; bitgleich (`CudaToleranceAware.TiltRemovalMatchesCpu`) |
| Synthese | Merit eines Schritts mit `system_optimizer` gleich F_tol eines eigenen Backends am realisierten Punkt (Standardsatz, S = 20); F_B nominal kleiner | bitgleich; F_tol 519,7 gegen F_B 491,0 (`ToleranceAware.SynthStepUsesExpectedMerit`); Kommandozeile mit und ohne Strahlsuche (`CliSynth.ToleranceAware`) |

## 15. Memetische globale Suche (M11-T7)

Vorschlag und Einordnung: [optimierung-erweiterungen.md, Abschnitt 7.1](optimierung-erweiterungen.md#71-memetische-suche-cma-es-mit-lm). Bibliothek `Mikroptim::optim`: `include/mikroptim/optim/cma_es.hpp` (CMA-ES, eigene Umsetzung ohne externe Abhängigkeit), `include/mikroptim/optim/memetic.hpp` (`memetic_search`); Kommandozeile `mikroptim optimize --global` ([cli.md](cli.md#globale-suche---global-ab-m11-t7)). Die Topologie bleibt fest; gesucht wird über die Variablen der Datei.

### Ablauf

Pro Generation g (höchstens `generations`):

1. **Kandidaten ziehen.** CMA-ES: λ Proben (mit Vorfilter K·λ) aus der Suchverteilung in normierten Koordinaten, zurück in x und auf die Schranken projiziert. Vergleich (`starts`): gleichverteilt gestörte Starts um x₀ wie `--runs` (`perturbed_vector` mit `relative = σ₀` für alle Typen, Startwert `seed`, Variante j = g·K·λ + i), ohne Lernen. Mit `include_start` (Standard) ist der erste Kandidat der Generation 0 x₀ selbst; mit `elitist` (Standard, nur CMA-ES) ist der erste Kandidat jeder späteren Generation der verfeinerte Punkt des bisher besten Individuums, der so weitere L_g Zyklen erhält (Elitismus). Die übrigen Proben behalten ihren Index i.
2. **Vorfilter** (optional, `prefilter`): Die K·λ Kandidaten gehen als ein Batch durch den FP32-Vorfilter (M7-T2, Abschnitt 8; CPU `prefilter_cpu`, GPU `TraceEngine::prefilter_variants`), mit den Zusatzresiduen ohne die trace-abhängigen (die untere Schranke L_v bleibt gültig). Kriterien wie in der Strahlsuche: mehr als die Hälfte der Strahlen versagt sicher oder L_v über dem Zehnfachen der besten oberen Schranke. Optimiert werden die ersten λ behaltenen in Kandidatenreihenfolge; fehlen welche, die aussortierten mit der kleinsten unteren Schranke (Gleichstand: Index). Die FP32-Werte dienen nur dieser Auswahl (AGENTS.md §6.1), nie als Merit oder im Ranking.
3. **Lokale Verfeinerung.** Alle λ Starts in **einem** Aufruf des gebündelten LM (`optimize_lm_cpu` bzw. `cuda::optimize_lm_gpu`, auf der GPU ein Block pro Variante, ADR-0004) mit `max_iterations = L_g` Zyklen pro Individuum; alle übrigen Einstellungen (Quadratur, Strahlversagen und Strafverfahren, Zusatzresiduen, Abbruchkriterien) aus den `LmOptions` des Aufrufers.
4. **Ranking** der Individuen nach dem verfeinerten Ergebnis mit derselben Regel wie `optimize` (`memetic_better`): endliche Merit, dann weniger versagte Strahlen, dann kleinere Merit; Gleichstand nach Index (stabile Sortierung). CMA-ES braucht nur die Rangfolge, die Merits mit verschiedenen w_fail müssen also nicht vergleichbar sein.
5. **Fortschreiben** der Verteilung (nur CMA-ES): Lamarck (Standard) mit den verfeinerten Punkten x*_i, Baldwin (`lamarckian = false`) mit den Startpunkten; beide in der Rangfolge der verfeinerten Merit. Mit `step_size_from_starts` (nur Lamarck, Standard aus) kommt der Pfad p_σ der Schrittweite aus den Startpunkten, Mittelwert und Kovarianz weiter aus den verfeinerten Punkten.

Ergebnis: alle Individuen (Start, `LmRunResult`), Kennzahlen pro Generation und das beste Individuum nach `memetic_better`.

### Normierung

u_k = x_k / d_k mit d_k = max(|x₀_k|, x_typ,k), fest für den ganzen Lauf. Der Vorschlag nennt x/x_typ; mit dem Maximum wie bei der Schrittweite der finiten Differenzen (Abschnitt 4) ist σ₀ eine relative Störung wie `PerturbationScale::relative` und damit mit `--perturbation` vergleichbar, auch wenn x_typ (Standard 0,01/mm für Krümmungen) klein gegen |x₀| ist. Variablen mit x₀ = 0 (Geisterkoeffizienten) haben die Skala x_typ. Rückabbildung x_k = u_k · d_k, danach `clamp_to_bounds`.

### CMA-ES

Nach Hansen (2016, „The CMA Evolution Strategy: A Tutorial“), nur positive Gewichte, n = P:

| Größe | Formel |
|---|---|
| Population | λ = 4 + ⌊3 ln n⌋ (Standard, `population = 0`), μ = ⌊λ/2⌋ |
| Gewichte | w_i = (ln((λ+1)/2) − ln i) / Σ_j (ln((λ+1)/2) − ln j), i = 1 … μ; μ_eff = 1/Σ w_i² |
| Schrittweite | c_σ = (μ_eff + 2)/(n + μ_eff + 5), d_σ = 1 + 2 max(0, √((μ_eff − 1)/(n + 1)) − 1) + c_σ |
| Kovarianz | c_c = (4 + μ_eff/n)/(n + 4 + 2μ_eff/n), c_1 = 2/((n + 1,3)² + μ_eff), c_μ = min(1 − c_1, 2(μ_eff − 2 + 1/μ_eff)/((n + 2)² + μ_eff)) |
| E‖N(0, I)‖ | χ_n = √n (1 − 1/(4n) + 1/(21n²)) |

**Probe** i der Generation g: x_i = m + σ B (D ∘ z), z_k = Σ_{j=1}^{12} u_j − 6 (Irwin-Hall wie im Variantengenerator, Abschnitt 4) aus einem eigenen `std::mt19937_64` mit Startwert `variant_seed(variant_seed(seed, g), i)`, Ziehungen k = 0 … n − 1. Die Probe hängt nur von Startwert, Generation, Index und Zustand ab.

**Fortschreiben** mit λ Punkten x_i (verfeinert oder Starts) und ihrer Rangfolge:

- Schritte y_i = (x_{i:λ} − m)/σ der μ besten. Weil die verfeinerten Punkte nicht aus der Verteilung stammen, wird jeder auf die Mahalanobis-Länge ‖C^{−1/2} y_i‖ ≤ c_y = √n + 2n/(n + 2) gekürzt (Hansen 2011, „Injecting external solutions into CMA-ES“).
- Rekombination y_w = Σ_{i=1}^{μ} w_i y_i, m ← m + σ y_w (c_m = 1).
- p_σ ← (1 − c_σ) p_σ + √(c_σ(2 − c_σ) μ_eff) C^{−1/2} y_w mit C^{−1/2} = B D⁻¹ Bᵀ.
- h_σ = 1, wenn ‖p_σ‖ / √(1 − (1 − c_σ)^{2(g+1)}) < (1,4 + 2/(n + 1)) χ_n, sonst 0; die Potenz als laufendes Produkt.
- p_c ← (1 − c_c) p_c + h_σ √(c_c(2 − c_c) μ_eff) y_w.
- C ← (1 + c_1 δ(h_σ) − c_1 − c_μ) C + c_1 p_c p_cᵀ + c_μ Σ_i w_i y_i y_iᵀ, δ(h_σ) = (1 − h_σ) c_c (2 − c_c).
- σ ← σ · exp(min(1, (c_σ/d_σ)(‖p_σ‖/χ_n − 1))); die Obergrenze 1 (höchstens Faktor e pro Generation) empfiehlt Hansen für eingespeiste Punkte.
- **Schrittweite bei Lamarck.** Die verfeinerten Punkte liegen meist weit außerhalb von σ (der LM läuft ins nächste Becken). p_σ wird dann lang, und σ wächst fast jede Generation (Mehrlinser mit σ₀ = 0,05: 1,4 nach 8 Generationen); die Proben erkunden zunehmend den ganzen Schrankenbereich, der Mittelwert folgt den besten Becken. Am Mehrlinser ist das die wirksame Variante (beste Merit 1,4·10⁻⁴ gegen 1,8·10⁻² mit p_σ aus den Starts, bei der σ etwa konstant bleibt und die Suche im Becken des Elite-Individuums endet). `step_size_from_starts` bleibt als Option für Systeme, bei denen große Proben vor allem Strahlversagen erzeugen.
- C symmetrisieren, Eigenzerlegung C = B diag(λ_k) Bᵀ (zyklisches Jacobi-Verfahren, Rotationen in fester Reihenfolge p < q, Rutishauser-Abbruch, höchstens 64 Durchgänge), D_k = √max(λ_k, 10⁻¹⁴ max λ). Entartet C (nicht endlich oder max λ ≤ 0), beginnt sie wieder bei I.

### Budget

Gezählt werden **LM-Zyklen** (Versuchsschritte einschließlich verworfener) über alle Individuen: Σ `LmRunResult::iterations`, einschließlich der Runden des Strafverfahrens beim weichen Versagen. Budget B = `cycle_budget`, Standard λ · `generations` · `lm_iterations`. Eine Generation erhält L_g = min(L, ⌊(B − verbraucht)/λ⌋) Zyklen pro Individuum und startet nur bei L_g ≥ 1; der Verbrauch bleibt so höchstens B (außer durch zusätzliche Runden des Strafverfahrens, die in beiden Verfahren gleich gezählt werden). Der Vorfilter kostet keine LM-Zyklen (ein FP32-Trace der Kandidaten).

### Determinismus

Kein Zufall außer den Generatoren mit festem Startwert; CMA-ES verwendet nur korrekt gerundete Operationen (+, −, ·, /, √) und exakte Hilfsfunktionen (floor, frexp, ldexp). Die Logarithmen der Gewichte und die Exponentialfunktion der Schrittweite sind eigene Reihen (`portable_log`: x = m·2^e, m ∈ [√½, √2), ln m = 2 artanh((m − 1)/(m + 1)) bis s³¹; `portable_exp`: Cody-Waite-Reduktion x = k ln 2 + r, Taylor-Reihe bis r²⁰), weil `std::exp` und `std::log` zwischen Bibliotheken um ein ulp abweichen dürfen; Abweichung von der Standardbibliothek ≤ 10⁻¹⁵ relativ. Alle Summen laufen in fester Reihenfolge; der gebündelte LM ist bitgleich für jede Threadzahl und Gruppierung (Abschnitt 4). Zwei Läufe mit gleichen Eingaben sind damit bitgleich, auf der GPU auf derselben Karte. MSVC und GCC (Windows) lieferten für `optimize --global cma,generations=4,iterations=10,seed=3` an `einzellinse_bk7_nebenbedingungen.json` dieselbe JSON-Ausgabe; der CMA-ES-Teil ist plattformunabhängig, der LM so weit wie seine Mathematikbibliothek.

### Abnahme (M11-T7)

Tests `tests/test_memetic.cpp` (CPU), `tests/test_cuda_memetic.cpp` (GPU), `CliOptimize.GlobalSearchMatchesLibrary`; Testfall `tests/memetic_test_cases.hpp`. Vergleich mit `perturbed_starts` (`tests/lm_test_systems.hpp`, derselbe `perturbed_vector` wie `optimize --runs`, gleichverteilt mit relativer Störung σ₀, gleicher Startwert): ⌈Z/L⌉ Starts mit je L Zyklen gemeinsam optimiert, Z = verbrauchte Zyklen der memetischen Suche, also **mindestens gleich viele LM-Zyklen** für den Vergleich.

| Prüfung | Einstellung | Ergebnis (Test) |
|---|---|---|
| Mehrlinser (`mehrlinser_lm_problem`, P = 13, Brennweitenbedingung, weiches Versagen), CPU | CMA-ES λ = 11, 8 Generationen, L = 20, σ₀ = 0,1, Startwert 1 (Budget 1.760) | F₀ = 0,731; CMA **1,41·10⁻⁴** (1.760 Zyklen) gegen `perturbed_starts` L = 20: 4,50·10⁻² (88 Läufe, 1.760 Zyklen) und L = 100: 1,86·10⁻² (18 Läufe, 1.800 Zyklen) (`Memetic.CmaBeatsPerturbedStartsOnMehrlinser`, 2,5 s Release) |
| dasselbe, GPU (RTX 3090) | wie oben, `optimize_lm_gpu` | CMA **7,67·10⁻⁴** gegen 4,50·10⁻² (je 1.760 Zyklen), 0,32 s; zwei Läufe bitgleich (`CudaMemetic.CmaBeatsPerturbedStartsOnMehrlinser`). Der Pfad weicht nach einigen Generationen von dem der CPU ab (Rundung der Strahlresiduen ändert die Rangfolge), das Ergebnis liegt im selben Bereich |
| Weitere Startwerte (Mehrlinser, CPU) | Startwerte 2, 3 | CMA 1,56·10⁻⁴ und 1,33·10⁻⁴ gegen `perturbed_starts` L = 20: 4,3·10⁻² und 4,2·10⁻² |
| Determinismus | zwei Läufe, 1 und 3 Host-Threads; Vorfilter auf CPU und GPU | bitgleich (`Memetic.TwoRunsAreBitIdentical`, `Memetic.PrefilterSelectsKeptCandidates`, `CudaMemetic.PrefilterIsDeterministic`) |
| Vergleichsverfahren | `--global starts` | Starts bitgleich zu `perturbed_starts` (`Memetic.PerturbedStartsMethodUsesExistingGenerator`) |
| CMA-ES-Kern | gedrehtes Ellipsoid (Kondition 10⁶, n = 6), Parameter nach Hansen, Eigenzerlegung, exp/log | f < 10⁻¹⁰ in 400 Generationen; Eigenzerlegung ≤ 10⁻¹³; exp/log ≤ 2,2·10⁻¹⁶ relativ zur Standardbibliothek (`CmaEs.*`, `PortableMath.*`) |

**M6-T6-Zwischenstand** (`docs/ergebnisse/m6-t6/a2.json`: Vergleichszweig a2, 4 Elemente, NA 0,94, 17 Variablen, 3 paraxiale und 24 mechanische Bedingungen, F₀ = 561 am Start, der LM steht dort mit `converged_step`). GPU, 16 × 8 Strahlen, σ₀ = 0,02, Vergleich mit gleichem Budget (`--global starts,budget=Z,generations=100` mit Z = Verbrauch der CMA-Suche und denselben übrigen Einträgen); Befehl `mikroptim optimize docs/ergebnisse/m6-t6/a2.json --gpu --rings 16 --quiet --json --global cma,generations=8,iterations=50,sigma=0.02,seed=S`:

| Einstellung | Startwert 1 | 2 | 3 | 4 | Mittel |
|---|---|---|---|---|---|
| CMA, 8 × 50 Zyklen | 3,25·10⁻³ | **2,01·10⁻³** | 3,90·10⁻³ | **2,07·10⁻³** | 2,81·10⁻³ |
| `starts`, gleiches Budget | **3,01·10⁻³** | 4,87·10⁻³ | **2,83·10⁻³** | 3,21·10⁻³ | 3,48·10⁻³ |
| CMA, 20 × 20 Zyklen | **1,95·10⁻³** | 3,39·10⁻³ | 4,01·10⁻³ | **2,29·10⁻³** | 2,91·10⁻³ |
| `starts`, gleiches Budget | 3,03·10⁻³ | **2,74·10⁻³** | **3,14·10⁻³** | 3,17·10⁻³ | 3,02·10⁻³ |

Verbrauch 4.900–5.800 Zyklen bei einem Budget von 4.800 (Strafrunden des weichen Versagens). Hier ist die memetische Suche im Mittel und im besten von vier Läufen besser, aber nicht bei jedem Startwert: Fast alle Läufe enden mit `max_iterations`, weil der LM an den Knicken der aktiven mechanischen Bedingungen kriecht (F10, optimierung-erweiterungen.md, Abschnitt 1); das Ergebnis hängt dann mehr von den verfügbaren Zyklen als vom Start ab. Am glatten Mehrlinser ist der Vorteil dagegen zwei Größenordnungen.
