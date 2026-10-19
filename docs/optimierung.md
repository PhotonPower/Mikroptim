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

Die Variablen stehen im Abschnitt `variables` der Systemdatei ([dateiformat.md](dateiformat.md#variablen-ab-m3-t1)). Implementiert sind Krümmung, Dicke, Arbeitsabstand und (ab M4-T6) Geisterkoeffizienten; Glas (M6-T1) kommt als weiterer `VariableType` hinzu.

`mko::VariableSet` (`include/mikroptim/optim/variable_set.hpp`, Bibliothek `Mikroptim::optim`) wird aus Systembeschreibung und Batch-Topologie gebaut und legt die Komponenten des Variablenvektors x ∈ ℝ^P fest (Reihenfolge von `resolve_variables`):

| Typ | Komponente x_k | Element im Batch |
|---|---|---|
| Krümmung c der Dateifläche i | `surfaces[i].curvature` | `curvature[v][m(i)]` |
| Dicke t der Dateifläche i (i < S_Datei − 1) | `surfaces[i].thickness` | `thickness[v][m(i+1) − 1]`, bei Geisterplatten im Spalt x − d_last (siehe unten) |
| Arbeitsabstand w | `surfaces[S_Datei − 1].thickness` | `thickness[v][S − 1]` (einzige Quelle, `image_distance`) |
| Geisterkoeffizient a_k der Platte g (ab M4-T6) | `ghost_plates[g].coefficients[k − 1]` | `ghost_coeff[v][g][k − 1]` |

Geisterkoeffizienten heißen in Berichten `a[g,k]`; `Variable::ghost_plate`/`ghost_order` nennen Platte und Ordnung, `batch_surface` ist die Tabellenfläche der Platte. Standard x_typ = 10⁻⁴ mm (`kDefaultTypicalGhostCoefficient`): Die FD-Schrittweite bei a = 0 ist dann h = 6·10⁻¹⁰ mm, klein gegen die Glättung ε der L1-Strafe (Abschnitt 3), sodass die zentrale Differenz den glatten Bereich der Strafe auflöst ((h/ε)² ≈ 4·10⁻⁷ bei ε = 10⁻⁶ mm); die Wellenfrontspalten bleiben dabei auf etwa 10⁻⁴ relativ genau (OPD-Änderung (n_G − 1)·h gegen die Rundung ε_m·L der optischen Länge). Schranken sind optional (frei). Störung im Variantengenerator relativ wie bei den übrigen Typen (`PerturbationOptions::ghost_coefficient`, Standard 1 %); a = 0 bleibt ungestört.

m = `table_of_file` der `SurfaceMap` (`make_surface_map`, geprüft gegen die Topologie mit `checked_surface_map`: Flächenzahl, Zahl der Geisterplatten, Typ jeder Tabellenfläche); ohne Geisterplatten die Identität. Das ist die einzige Stelle, an der `VariableSet` den Indexversatz durch Geisterplatten berücksichtigt.

**Spalt mit Geisterplatten.** `make_surface_table` teilt den Luftspalt T nach Dateifläche i an den Plattenabständen d_1 ≤ … ≤ d_m in die Dicken d_1, d_2 − d_1, …, T − d_m. Die Variable t_i behält die Bedeutung der Dateidicke T (Schranken, Startwert und `apply_to_description` wie ohne Platten). Festlegung:

- `apply`: nur das letzte Teilstück (Tabellenfläche m(i+1) − 1, die letzte Platte) wird fl(x − d_m), mit genau dem Ausdruck von `make_surface_table`; die Plattenabstände d_g sind feste Daten der Datei (Lage relativ zum Scheitel der vorderen Fläche) und bleiben. Damit ist ein Batch nach `apply` bitgleich zu dem, der aus der mit `apply_to_description` geänderten Datei neu gebaut wird (reproduzierbares Schreiben und Wiederladen optimierter Systeme).
- `extract`: x = fl(d_m + letztes Teilstück).
- Roundtrip Vektor → Batch → Vektor: Für x ≥ d_m ist fl(d_m + fl(x − d_m)) = x, außer wenn x − d_m genau in der Mitte zweier Gleitkommazahlen derselben Binade wie x liegt (dann ±1 ulp; z. B. d = 0,11 mit x in [0,235; 0,25)). Für Plattenabstände mit wenigen Mantissenbits (0,5, 1,0, 3,5 …) tritt der Fall nicht auf (Tests `VariableSet.GhostGapRoundtrip`, `VariableSet.GhostGapRoundtripTieCase`).
- Die untere Schranke wird auf max(`min`, d_m) angehoben, damit die letzte Platte im Spalt bleibt (validate_system verlangt d ≤ T).
- Der Arbeitsabstand ist nie geteilt (nach der letzten Fläche liegen keine Geisterplatten).

- `extract(batch, v)` → x und `apply(batch, v, x)` lesen bzw. schreiben genau diese Elemente; alles andere (Brechzahlen, Feldwinkel, Feldziele, Geisterkoeffizienten) bleibt unverändert. Der Roundtrip Vektor → Batch → Vektor ist bitgleich (Test `VariableSet.RoundtripVectorBatchVector`; einzige Ausnahme: Rundungsfall „genau halb“ bei einem Spalt mit Geisterplatten, siehe unten). `apply` prüft Länge, Endlichkeit und Topologie, bevor es schreibt.
- `extract(system)` und `apply_to_description(system, x)` tun dasselbe auf den Dateiflächen (Schreiben optimierter Systeme, M3-T5).
- Schranken als Vektoren `lower()`, `upper()` (±∞ für offene Schranken), `typical()` für x_typ; `feasible(x)` und die Projektion `clamp_to_bounds(x)`: x_k ← min(max(x_k, lower_k), upper_k).
- Skalierte Koordinaten gibt es bewusst nicht: Die Marquardt-Skalierung mit diag(JᵀJ) (Abschnitt 4) macht LM invariant gegen die Skalierung der Variablen; x_typ dient nur der Schrittweite.

## 2. Residuenvektor

Die Merit ist F = ‖r‖². Der Residuenvektor einer Variante besteht aus:

1. **Wellenfront:** √(ŵ_f · ŵ_λ · W_i · V_j) · OPD_ij / λ_ref für alle Felder f, Wellenlängen λ, Pupillenstützstellen (i, j), mit den auf Summe 1 normierten Gewichten ŵ_f = w_f/Σw_f, ŵ_λ = w_λ/Σw_λ (ab M4-T1, Definition und Begründung: [numerik.md, „Residuen pro Strahl“](numerik.md#residuen-pro-strahl-m2-gewichte-m4-t1)). Die Residuen sind dimensionslos (Wellen bei λ_ref); ohne Versagen ist ihr Anteil an F der gewichtete mittlere RMS² Σ ŵ_f ŵ_λ (RMS_fλ/λ_ref)² in Wellen².
2. **Paraxiale Nebenbedingungen:** √w · (Ist − Soll) für Brennweite (3,29 mm, über die Brechkraft formuliert), Abgleichlänge (45 mm), ggf. Arbeitsabstand (Gleichung oder Untergrenze); Definitionen in Abschnitt 2.2.
3. **Petzval:** √w · max(0, |κ_P| − κ_max), inklusive Beiträgen der Geisterplatten; Definition und Gewichtswahl in Abschnitt 2.3.
4. **Mechanik (einseitig):** √w · max(0, g) für Durchmesser, Mitten- und Randdicken, maximalen Einfallswinkel (Vorgabe ca. 55–60°).
5. **Weiches Versagen** (ab M4-T2): ein Strafresiduum pro Strahl, p_k = √(w_fail · ŵ_f · ŵ_λ · W_i · V_j) · π_k, mit der Summe π_k der dimensionslosen Strafgrößen seiner Ereignisse (−D, −k, −Radikand, −disc/R_ref²; hart versagt zusätzlich 1). Die Strafresiduen stehen als eigener Block gleicher Länge direkt hinter den Wellenfrontresiduen (vor den Zusatzresiduen 2–7) und entstehen im Kernel; weich versagte Strahlen bleiben mit der OPL ihrer Fortsetzung in Kolben und Wellenfrontresiduen. Gewicht w_fail in Wellen² (`ResidualSettings::fail_weight`, Standard 10⁶), im LM mit äußerem Strafverfahren (Abschnitt 4). Regeln, Grenzfälle und Messungen: [numerik.md, Abschnitt 11](numerik.md#11-weiches-versagen).
6. **Geisterplatten (L1, geglättet):** im Plan √(μ·w_k) · (a_{g,k}² + ε²)^{1/4}, sodass das Quadrat μ·w_k·√(a² + ε²) ≈ μ·w_k·|a| ist. Implementiert (ab M4-T6) ist die gleichwertige Form √(μ·w_k) · a/√(ε + √(a² + ε²)) mit dem Quadrat μ·w_k·(√(a² + ε²) − ε): gleicher Gradient, also gleiches Minimum, aber r(0) = 0 und eine Gauß-Newton-Krümmung, die bei a = 0 exakt stimmt (Begründung in Abschnitt 3).
7. **Glas-Hülle:** √w · Abstand außerhalb der Hülle.

Gewichte w_f, w_λ stammen aus der Systemdatei (`source.field_weights`, `spectrum.weights`); Gewicht 0 schaltet ein Feld bzw. eine Wellenlänge in der Merit ab (Residuen exakt 0, verfolgt wird es weiter). **Einheit der Merit:** Wellen² bei λ_ref. Alle Zusatzresiduen (Punkte 2–7) sind √w · (Größe in ihrer eigenen Einheit) und gehen mit ihrem Quadrat direkt in dieselbe Summe ein; ihre Gewichte haben daher die Einheit 1/(Einheit der Größe)² und sind relativ zu „ein Residuum von 1 entspricht einer Welle RMS“ zu wählen (Faustregel in Abschnitt 2.2). Geisterplatten-Gewichte steigen mit der Ordnung: w_k = w₀ · κ^k (κ ≈ 2).

### 2.1 Zusatzresiduen (Schnittstelle, ab M3-T3)

Residuen, die nicht pro Strahl im Kernel entstehen (Punkte 2–4, 6, 7), liefert der Host pro Variante über `mko::ExtraResiduals` (`include/mikroptim/optim/extra_residuals.hpp`): `count` Werte pro Variante, `evaluate(batch, v, out)` schreibt sie nach `out` (Länge `count`). Im Residuenvektor einer Variante stehen sie hinter den Strahlresiduen (Platz `OutputRequest::extra_residuals`; die Ablage im Jacobi-Aufbau legen M3-T2/M3-T4 fest). `evaluate_extra_residuals(extra, batch, out, first, count, threads)` (`include/mikroptim/optim/paraxial_residuals.hpp`) wertet die Varianten [first, first + count) auf Host-Threads aus, `out[(v − first)·extra.count + i]`; `evaluate` muss dafür threadsicher sein. Mehrere Quellen verkettet `concat_extra_residuals({…})` (ab M4-T6, `extra_residuals.hpp`): count = Σ count_i, die Teile in der angegebenen Reihenfolge, Teile mit count = 0 entfallen. `mikroptim optimize` verwendet paraxiale Nebenbedingungen, dahinter die L1-Strafe der Geisterkoeffizienten. Strafblöcke im Kernel (weiches Versagen, M4-T2) gehören zu den Strahlresiduen und sind davon unabhängig.

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

Direkt mit großem Gewicht zu beginnen, ist ungünstig: Liegt die Überschreitung unter der Änderung von |κ_P| über die FD-Schrittweite (h = ε^{1/3} · max(|x|, x_typ), im Abnahmesystem rund 10⁻⁷/mm), überspannt das zentrale FD-Muster den Knick von max(0, ·). Die Jacobi-Zeile ist dann nur etwa halb so groß wie die einseitige Ableitung, das Gauß-Newton-Modell passt nicht, und LM kommt nur in kleinen Schritten voran. Beobachtet im Abnahmesystem (während der Entwicklung mit Residuen in mm; w = 1 mm² entspricht heute 1,7·10⁶ mm²): Start mit diesem Gewicht direkt vom Startsystem endet nach 500 Zyklen mit `max_iterations` 20 % über dem beschränkten Optimum, dieselbe Bedingung als Gleichung konvergiert in 147 Zyklen. Mit der Fortsetzung liegt der Punkt bei kleinem w im glatten aktiven Bereich (Überschreitung ≫ FD-Änderung), läuft dort zum beschränkten Optimum, und die späteren Stufen mit kleiner Überschreitung brauchen nur noch 8–18 Zyklen bis zur Konvergenz. Die Gewichtsstufen der Systemsynthese übernimmt später der Homotopie-Controller (Abschnitt 7).

**Abnahme** (`tests/test_petzval_optimization.cpp`, GPU `tests/test_cuda_petzval_optimization.cpp`): frei erfundener Zweilinser `data/systems/zweilinser_petzval.json` (Sammellinse n = 1,8, Luftspalt, Zerstreuungslinse CaF₂, f = 20 mm, NA ≈ 0,1, zwei Felder, eine Wellenlänge), Variablen: vier Krümmungen, Luftspalt, Arbeitsabstand (mit Schranken); Merit = Wellenfront (Wellen²) + Brennweite (Gewicht 16.767,7/mm²) + Petzval mit κ_max = 0,01/mm (Petzvalradius ≥ 100 mm, Gewicht 1,677·10⁹ mm²) bei Start-κ_P = −0,0268/mm. Die Gewichte sind die zuerst für Residuen in mm gewählten 0,01/mm² und 1.000 mm², exakt umgerechnet mit 1/(F·L·λ_ref²) = 1/(2·(546,07 nm)²); das Problem ist damit bis auf einen konstanten Faktor dasselbe, und die Zahlen unten stimmen bis auf Rundung mit dem Lauf vor M4-T1 überein. Fortsetzung über sieben Stufen w = 1,677·10³ … 1,677·10⁹ mm² (Faktor 10 pro Stufe, das letzte ist das Gewicht der Datei), LM auf der CPU:

| Kriterium | Ergebnis |
|---|---|
| alle Stufen konvergiert | ja (462, 18, 15, 15, 8, 18, 10 Zyklen; GPU (RTX 3090): 467, 18, 15, 17, 8, 14, 10 mit (\|κ_P\| − κ_max)/κ_max = 1,4·10⁻⁸ und F relativ 1,4·10⁻¹⁰ zur CPU) |
| \|κ_P\| ≤ κ_max nach der Fortsetzung | (\|κ_P\| − κ_max)/κ_max = 1,4·10⁻⁸ (Toleranz `kPetzvalPenaltyRelative` = 10⁻⁶) |
| Bedingung aktiv | Überschreitung > 0 in jeder Stufe, ∝ 1/w; ohne die Bedingung läuft LM vom Ergebnis weg zu κ_P = −0,01067/mm und kleinerem Merit |
| Brennweite eingehalten | f − f_s = 4,8·10⁻⁷ mm (Toleranz `kPetzvalFocalLength` = 10⁻⁵ mm) |

**API** (`include/mikroptim/model/constraints.hpp`, `include/mikroptim/optim/paraxial_residuals.hpp`): `ConstraintType::Petzval`, `ConstraintSpec::max_curvature`, `ConstraintSpec::mode_or_default()` liefert bei Petzval `Max`, `constraint_bound(system, spec)` (Grenze bzw. Sollwert eines Eintrags; `constraint_target` bleibt für Petzval leer). In `ParaxialConstraint` steht κ_max in `target`, der Modus ist `Max`; `check_paraxial_constraints` lehnt andere Modi und κ_max < 0 ab. Die Bedingung braucht den Paraxialtrace (`needs_trace()`) und läuft wie die übrigen über `make_paraxial_residuals` und `evaluate_extra_residuals`.

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

- **Schwelle:** |s_k| < 1, also μ w_k > |g_k|, heißt „μ oberhalb der Schwelle“; dann bleibt |a_k| im Bereich ε |s_k|, sonst wird der Koeffizient aktiv (|a| ≫ ε, dort gilt die Stationarität mit dem L1-Subgradienten). Exakt 0 erreicht die geglättete Strafe nie, aber |a_k| ≈ ε|g_k|/(μ w_k) ist beliebig klein: Für |a| < 10⁻⁸ mm genügt ε = 10⁻⁷ mm bei μ wenig über der Schwelle oder ε = 10⁻⁶ mm bei μ ≥ 100-fach über der Schwelle. Eine Fortsetzung in ε (ε am Ende verkleinern) leistet dasselbe; sie gehört zum Homotopie-Controller (M6-T3). Kleinere ε als etwa 10 h (h = FD-Schrittweite bei a = 0, Abschnitt 1) sollte man nicht wählen, sonst überspannt die zentrale Differenz den glatten Bereich.
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
- Version 2 (M5): Jacobi-Zeilen aus dem adjungierten Kernel; die finite-Differenzen-Variante bleibt als Test-Orakel erhalten.

### Verfahren (ab M3-T4)

Host-API `include/mikroptim/optim/levenberg_marquardt.hpp` (`Mikroptim::optim`), GPU-Backend `include/mikroptim/cuda/levenberg_marquardt.hpp` und Normalgleichungen `include/mikroptim/cuda/normal_equations.hpp` (`Mikroptim::cuda`).

**Zustand pro Lauf:** x, F = ‖r(x)‖² (Strahl-, Straf- und Zusatzresiduen, Abschnitt 2), Zahl versagter und hart versagter Strahlen, Dämpfung μ (Start `initial_damping` = 10⁻³) und ν (Start 2), Status, Zähler. Die Steuerlogik `levenberg_marquardt(backend, P, bounds, x0, options)` ist einmal implementiert; ein **Backend** (`LmBackend`) liefert für eine Liste von Läufen A = JᵀJ und g = Jᵀr an ihren Punkten (`normal_equations`) sowie F und versagte Strahlen (`merit`). Backends: `CpuLmBackend` (CPU, CI), `cuda::GpuLmBackend` (GPU) und in den Tests analytische Residuenfunktionen.

**Zyklus** (alle aktiven Läufe gemeinsam):

1. Für Läufe nach einem angenommenen Schritt (und am Anfang): J und r an x per finiten Differenzen (alle diese Läufe in einem FD-Batch), A = JᵀJ, g = Jᵀr. Danach Skalierung, Gradientenmaß und Modell-Kriterien (unten).
2. Für jeden aktiven Lauf der Schritt mit aktiver Menge (unten), unabhängig pro Lauf auf Host-Threads.
3. Alle Versuchspunkte gemeinsam bewerten: Modus MERIT der Engine bzw. `merit_cpu`, Zusatzresiduen als `extra_merit` = Σ e² (aufsteigend), sodass F genau die Quadratsumme des Residuenvektors ist (bis auf die Summationsreihenfolge).
4. Gewinnverhältnis ρ = (F − F_neu)/(L(0) − L(Δx)) mit L(0) − L(Δx) = −(2Δxᵀg + ΔxᵀAΔx) für den tatsächlichen (projizierten) Schritt Δx. Angenommen, wenn F_neu endlich ist, F_neu < F, die Vorhersage positiv ist und (Standard `reject_new_failures`) am Versuchspunkt nicht mehr Strahlen **hart** versagen als bei x – hart versagte Strahlen sind maskiert (r = 0) und könnten F sonst senken. Weich versagte Strahlen (M4-T2) zählen nicht: Ihre Wellenfront- und Strafresiduen sind stetig und stehen in F; eine Ablehnung jedes Schritts, der einen Strahl knapp über eine Grenze schiebt, würde den LM an Versagensgrenzen festhalten. Im harten Modus sind beide Zahlen gleich (Verhalten wie M3-T4).
5. Nielsen: angenommen μ ← μ·max(1/3, 1 − (2ρ − 1)³), ν ← 2; verworfen (auch bei gescheiterter Cholesky-Zerlegung) μ ← μ·ν, ν ← 2ν.

Fertige Läufe werden eingefroren und gehen in keinen weiteren Batch ein.

**Äußeres Strafverfahren (ab M4-T2).** Die optischen Treiber `optimize_lm_cpu` und `cuda::optimize_lm_gpu` rufen die Steuerlogik über `optimize_lm_rounds` auf. Endet ein Lauf im weichen Modus mit versagten Strahlen (Status außer `NumericalError`), startet er an seinem Endpunkt neu, mit w_fail · `fail_weight_factor` (Standard 10) und frischem Zustand (μ₀, ν = 2, neue Jacobi-Matrix); alle betroffenen Läufe gemeinsam in einem Aufruf, mit ihrer Vorlage (`lm_template_batch`). Das wiederholt sich, bis kein Strahl mehr versagt oder w_fail `max_fail_weight` (10¹⁴) übersteigen würde. Begründung und Messungen: [numerik.md, Abschnitt 11](numerik.md#festlegungen-der-implementierung-m4-t2) (mit festem w_fail endet der LM je nach System in Minima mit versagten Strahlen oder hängt an der Kante zum harten Versagen; mit dem Verfahren enden 108 von 108 bzw. 36 von 36 Starts der Testfamilien versagensfrei). Ergebnis eines Laufs: x, F, Zahlen versagter Strahlen, Status, μ, Gradientenmaß und Modellabnahme der letzten Runde; Zyklen, angenommene Schritte und Jacobi-Auswertungen summiert; `history` aneinandergehängt (jede Runde beginnt mit F am Endpunkt unter dem neuen Gewicht, deshalb springt F dort nach oben, solange Strahlen versagen); `fail_weight` und `penalty_rounds`. `on_iteration` erhält den ursprünglichen Laufindex und fortlaufende Zyklen. Die Entscheidung über einen Neustart hängt nur vom Lauf selbst ab; Gruppe und Einzellauf bleiben bitgleich. `fail_weight_factor` ≤ 1 oder hartes Versagen: ein Durchgang wie in M3-T4.

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
                                    void download(std::vector<double>& a, std::vector<double>& g) const; };
```

Rechentyp: FP64 (die Abnahme). df64 für Jacobi und Merit ist nicht angebunden (`FdJacobianOptions` hat keinen Rechentyp).

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
- Brechzahlen bleiben unverändert (Glas als Variable erst M6).

### Finite-Differenzen-Jacobi (ab M3-T2)

Host-API `include/mikroptim/optim/fd_jacobian.hpp` (`Mikroptim::optim`), Differenzenformeln header-only in `include/mikroptim/optim/fd_formula.hpp` (`fd_derivative`, `MKO_HD`, dieselbe Funktion auf CPU und GPU), GPU-Pfad `include/mikroptim/cuda/fd_jacobian.hpp` (`Mikroptim::cuda`).

**Residuenvektor einer Variante.** R = R_v + E Zeilen: die R_v = `ResidualLayout::trace_rows` Zeilen des Traces (Strahlresiduen, im weichen Modus seit M4-T2 dahinter ebenso viele Strafresiduen), danach die E Zusatzresiduen aus `ExtraResiduals` (Abschnitt [2.1](#21-zusatzresiduen-schnittstelle-ab-m3-t3), z. B. die paraxialen Nebenbedingungen `make_paraxial_residuals`, später Strafterme). Das ist dieselbe Ablage wie im Ausgabepuffer der TraceEngine mit `TraceSettings::extra_residuals = E`. Die Zusatzresiduen werden auf dem Host für jede Variante des FD-Batches ausgewertet (`evaluate_extra_residuals` aus M3-T3 auf `FdJacobianOptions::extra_threads` Host-Threads, bitgleich für jede Threadzahl, Ablage [V·E]).

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

## 5. Geisterplatten

### Platzierung

Mehrere Kandidatenpositionen in Luftspalten: mindestens eine nahe der Blende (feldunabhängige Fehler: sphärische Aberration, Farblängsfehler), eine in der Mitte und eine im Frontbereich (großes Verhältnis Hauptstrahl- zu Randstrahlhöhe: Koma, Astigmatismus, Petzval, Farbquerfehler).

### Glas-Wörterbuch

An jeder Position liegen mehrere Geisterplatten mit festen, repräsentativen Gläsern (z. B. Fluorit-artig, hochbrechendes Kronglas, Flintglas, KZFS-artig). Die L1-Strafe wählt aus, welche gebraucht werden. Freie Glasparameter in Geisterplatten wären nutzlos, weil ihr Gradient bei a = 0 verschwindet.

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

Alle Schwellen stehen in `GhostDiagnosisOptions`. **Empfehlung** des Berichts: die Lage mit dem größten Merit-Gewinn ihrer aktiven Platten (ohne Merit: größte Σ|a|), mit Operator, Glas der aktivsten Platte und dominanter Ordnung; Initialisierung des Operators daraus ist M6-T2.

Tests: `tests/test_ghost_diagnosis.cpp` (Felder gegen unabhängige Rechnungen: t₂ und φ_G, Rand- und Hauptstrahlhöhe im kollimierten Strahl, Merit mit Orakel 1 + Σ a², Abbezahlen und ΔP_g,F von CaF₂, N-BK7 und dem Flint; Klassifikation A/B/B anomal an vorgegebenen Koeffizienten; Text und JSON), Bedarf-Abnahme in Abschnitt 3.

## 6. Einfüge-Operatoren

Beide Operatoren ändern die Merit **exakt nicht** (Test: |ΔF| ≤ 10⁻¹² relativ).

**A – Meniskus ohne Brechkraft (nach Bociort):** In einem Luftspalt werden zwei Flächen gleicher Krümmung c₀ im Abstand 0 mit einem Glas aus dem Wörterbuch eingefügt. Die Dicke wird anschließend über eine Homotopie auf die Mindestdicke gebracht. c₀ und Glas werden aus der aktivsten Geisterplatte an dieser Position initialisiert (Vorzeichen der Biegung, Glastyp). Danach wird das Strafgewicht dieser Geisterplatte erhöht, sodass die Korrektur in das reale Element wandert. Dass eingefügte Elemente Sattelpunkte erzeugen können, wird genutzt: Beide Abstiegsrichtungen werden als getrennte Läufe weiterverfolgt.

**B – Aufspalten in ein Kittglied:** Eine Glaslinse wird an einer inneren Fläche (Krümmung c_s, z. B. aus der Geisterdiagnose oder als Mittel der Außenflächen) in zwei Teile aus **demselben** Glas geteilt. Danach werden die Glasparameter eines Teils freigegeben und wandern stetig, z. B. von einem Kronglas Richtung Fluorit.

## 7. Homotopie-Controller

Homotopie-Parameter: NA, Feld, Spektrum, Strafgewicht μ der Geisterplatten. Standardablauf:

1. monochromatisch (546 nm), auf der Achse, eine Linse, NA 0,3
2. NA in Schritten erhöhen bis 0,95; bei Geisterbedarf über Schwelle Operator A
3. Feld bis 100 % zuschalten, Petzval-Nebenbedingung scharf schalten
4. zwei Wellenlängen (480 und 644 nm): Achromasie, Operator B
5. 436–656 nm: Apochromasie, Operator B mit anomaler Teildispersion
6. 405 und 700 nm mit kleinem Gewicht dazu, Feinoptimierung

Schrittweitensteuerung: Konvergiert LM nach einem Schritt Δ schnell und bleibt die Merit unter der Schwelle, wird Δ vergrößert, sonst halbiert. Fällt Δ unter Δ_min und ist Geisterbedarf vorhanden, wird ein Einfüge-Operator angewandt.

## 8. Kandidatenverwaltung

Die Synthese ist eine Strahlsuche (beam search) über Zweige: Jede Einfügung und jeder Sattelpunkt-Abstieg erzeugt neue Kandidaten. Behalten werden die besten B Kandidaten nach Merit, Elementzahl und (ab M8) Empfindlichkeit. Alle Kandidaten einer Topologie laufen gemeinsam in einem Batch.

### FP32-Vorfilterung im Multistart (ab M7-T2)

Vor der Bewertung eines großen Multistart-Batches in FP64 (oder df64) sortiert `TraceEngine::prefilter_variants` hoffnungslose Varianten aus: auf der RTX 3090 im Kernel rund 19-mal so schnell wie MERIT in FP64 und 4,5-mal so schnell wie df64 ([benchmarks.md](benchmarks.md#fp32-vorfilterung-m7-t2)). Die Entscheidung ist konservativ: Eine Variante wird nur aussortiert, wenn sie das Kriterium auch in FP64 sicher erfüllt ([numerik.md, „FP32-Vorfilterung“](numerik.md#fp32-vorfilterung-m7-t2)); im Zweifel bleibt sie im Batch. Ablauf:

1. Startvarianten erzeugen (`make_perturbed_batch` o. ä.), weitere Residuen E_v (paraxiale Nebenbedingungen, `evaluate_extra_residuals`) in FP64 auf dem Host, Σ e² als `PrefilterSettings::extra_merit`.
2. `prefilter_variants` mit den Kriterien (`PrefilterCriteria`): (a) Anteil versagter Strahlen > `max_failed_fraction` (Standard 0,5), (b) Merit > `max_merit`, (c) Merit > `max_merit_factor` × beste Merit des Batches (b und c Standard aus). Die Margen (`PrefilterMargins`) bleiben auf den Standardwerten.
3. Nur die Varianten mit `keep(v)` in einen neuen Batch kopieren und in FP64/df64 bewerten bzw. optimieren. `merit_lower`/`merit_upper` dürfen für eine Rangfolge der Vorauswahl benutzt werden, **nie** als Merit, Startwert der Dämpfung, Abbruchkriterium oder in der Jacobi-Matrix.

Die Kandidatenverwaltung selbst (Auswahl der besten B, Zweige) ist M6-T4. Ein rangbasiertes Kriterium („nur die schlechtesten p % aussortieren“) lässt sich aus `merit_lower` und `merit_upper` bauen (aussortieren, wenn L_v über dem p-Quantil der U_u liegt), ist aber nicht umgesetzt.

## 9. Glas-Einrasten

Am Ende: für jedes Glaselement die k nächsten Katalog-Gläser im normierten (n₀, ν₁, ν₂)-Raum; Kombinationen werden als Varianten im Batch reoptimiert, das beste Ergebnis gewinnt. Weiche Nebenbedingung auf Transmission bei 420–450 nm (Farbstich), Kennzeichnung spannungsarmer Gläser für DIC.

## 10. Literatur

- F. Bociort, M. van Turnhout et al.: Arbeiten zur Saddle-Point Construction in optischen Systemen (TU Delft, ab ca. 2007).
- M. Isshiki et al.: Global optimization with escape function (Lens design).
- H. B. Nielsen: Damping parameter in Marquardt's method (1999).
- J. Nocedal, S. J. Wright: Numerical Optimization (exakte Penalty-Funktionen).
