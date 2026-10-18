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

Die Variablen stehen im Abschnitt `variables` der Systemdatei ([dateiformat.md](dateiformat.md#variablen-ab-m3-t1)). Implementiert sind Krümmung, Dicke und Arbeitsabstand; Glas (M6-T1) und Geisterkoeffizienten (M4-T6) kommen als weitere `VariableType` hinzu.

`mko::VariableSet` (`include/mikroptim/optim/variable_set.hpp`, Bibliothek `Mikroptim::optim`) wird aus Systembeschreibung und Batch-Topologie gebaut und legt die Komponenten des Variablenvektors x ∈ ℝ^P fest (Reihenfolge von `resolve_variables`):

| Typ | Komponente x_k | Element im Batch |
|---|---|---|
| Krümmung c der Dateifläche i | `surfaces[i].curvature` | `curvature[v][m(i)]` |
| Dicke t der Dateifläche i (i < S_Datei − 1) | `surfaces[i].thickness` | `thickness[v][m(i+1) − 1]`, bei Geisterplatten im Spalt x − d_last (siehe unten) |
| Arbeitsabstand w | `surfaces[S_Datei − 1].thickness` | `thickness[v][S − 1]` (einzige Quelle, `image_distance`) |

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

1. **Wellenfront:** √(w_f · w_λ · W_i · V_j) · OPD_ij / λ_ref für alle Felder f, Wellenlängen λ, Pupillenstützstellen (i, j).
2. **Paraxiale Nebenbedingungen:** √w · (Ist − Soll) für Brennweite (3,29 mm, über die Brechkraft formuliert), Abgleichlänge (45 mm), ggf. Arbeitsabstand (Gleichung oder Untergrenze); Definitionen in Abschnitt 2.2.
3. **Petzval:** √w · max(0, |κ_P| − κ_max), inklusive Beiträgen der Geisterplatten.
4. **Mechanik (einseitig):** √w · max(0, g) für Durchmesser, Mitten- und Randdicken, maximalen Einfallswinkel (Vorgabe ca. 55–60°).
5. **Weiches Versagen:** siehe numerik.md, Abschnitt 11.
6. **Geisterplatten (L1, geglättet):** √(μ·w_k) · (a_{g,k}² + ε²)^{1/4}, sodass das Quadrat μ·w_k·√(a² + ε²) ≈ μ·w_k·|a| ist.
7. **Glas-Hülle:** √w · Abstand außerhalb der Hülle.

Gewichte w_f, w_λ stammen aus der Systemdatei. Geisterplatten-Gewichte steigen mit der Ordnung: w_k = w₀ · κ^k (κ ≈ 2).

### 2.1 Zusatzresiduen (Schnittstelle, ab M3-T3)

Residuen, die nicht pro Strahl im Kernel entstehen (Punkte 2–4, 6, 7), liefert der Host pro Variante über `mko::ExtraResiduals` (`include/mikroptim/optim/extra_residuals.hpp`): `count` Werte pro Variante, `evaluate(batch, v, out)` schreibt sie nach `out` (Länge `count`). Im Residuenvektor einer Variante stehen sie hinter den Strahlresiduen (Platz `OutputRequest::extra_residuals`; die Ablage im Jacobi-Aufbau legen M3-T2/M3-T4 fest). `evaluate_extra_residuals(extra, batch, out, first, count, threads)` (`include/mikroptim/optim/paraxial_residuals.hpp`) wertet die Varianten [first, first + count) auf Host-Threads aus, `out[(v − first)·extra.count + i]`; `evaluate` muss dafür threadsicher sein.

**Parallelisierung auf dem Host.** Feste, zusammenhängende Aufteilung der Varianten auf T Threads (`std::thread`; `std::execution::par` bräuchte unter GCC TBB, also eine neue Abhängigkeit): Thread c bearbeitet [first + ⌊count·c/T⌋, first + ⌊count·(c+1)/T⌋). Jede Variante wird unabhängig und von genau einem Thread gerechnet, die Ergebnisse sind daher für jede Threadzahl bitgleich (keine Reduktion über Varianten). T ≤ 0 heißt `hardware_concurrency`. Wirft eine Variante, wird nach dem Join die Ausnahme der kleinsten werfenden Variante weitergegeben (unabhängig von T).

### 2.2 Paraxiale Nebenbedingungen (ab M3-T3)

Schema in der Systemdatei: [dateiformat.md](dateiformat.md#nebenbedingungen-ab-m3-t3); Sollwerte stehen in `target`. Jede Bedingung liefert ein Residuum

$$r = \sqrt{w}\;\tilde q,\qquad \tilde q = \begin{cases} q & \text{Gleichung (equal)}\\ \min(q, 0) & \text{Untergrenze (min): aktiv nur für Ist < Soll}\\ \max(q, 0) & \text{Obergrenze (max)} \end{cases}$$

Einseitige Bedingungen sind stetig an der Grenze (r = 0 dort), inaktiv exakt 0; ihre Ableitung springt an der Grenze (wie bei allen Straftermen der Form max(0, g)). Alle q sind Längen in mm, das Gewicht w hat die Einheit 1/mm².

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

## 3. Warum L1 für Geisterplatten

Mit einer quadratischen Strafe wäre der Grenznutzen bei a = 0 null; der Optimierer würde immer ein wenig Geisterkorrektur verwenden. Mit L1 gilt die Theorie der exakten Penalty: Ist μ größer als der zugehörige Lagrange-Multiplikator, bleiben die Koeffizienten exakt null, solange das reale System das Ziel alleine erreicht. Zusätzlich entsteht Sparsity: wenige, gezielte Koeffizienten statt vieler kleiner. Die Glättung mit ε (Startwert 10⁻⁶ in Einheiten des Dickenprofils) hält LM differenzierbar.

## 4. Gebündelter Levenberg-Marquardt

- Viele unabhängige Läufe (Multistart, Homotopie-Zweige, beide Abstiegsrichtungen an Sattelpunkten) werden **gleichzeitig** iteriert; jeder Lauf hat eigenen Zustand (x, Dämpfung, Status).
- Jacobi-Matrix in Version 1 per **zentralen finiten Differenzen**: Für P Variablen werden 2P (oder P vorwärts) gestörte Varianten und x₀ selbst in denselben Batch gepackt. Schrittweite h_k = ε_m^{1/3} · max(|x_k|, x_typ,k), ε_m^{1/3} ≈ 6·10⁻⁶. Schablonen an Schranken, Ablage und Genauigkeit: [Finite-Differenzen-Jacobi](#finite-differenzen-jacobi-ab-m3-t2).
- JᵀJ und Jᵀr per `cublasDgemmStridedBatched`, Lösung per `cusolverDnDpotrfBatched` / `potrsBatched`.
- Marquardt-Skalierung mit diag(JᵀJ), Dämpfungssteuerung nach Nielsen (Gewinnverhältnis).
- Schranken per Projektion auf den zulässigen Bereich mit aktiver Menge.
- Abbruch: relative Änderung von F < 10⁻¹⁰ oder Schrittnorm < Toleranz oder Iterationslimit.
- Version 2 (M5): Jacobi-Zeilen aus dem adjungierten Kernel; die finite-Differenzen-Variante bleibt als Test-Orakel erhalten.

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

**Residuenvektor einer Variante.** R = R_v + E Zeilen: die R_v Strahlresiduen in der Ablage von `ResidualLayout`, danach die E Zusatzresiduen aus `ExtraResiduals` (Abschnitt [2.1](#21-zusatzresiduen-schnittstelle-ab-m3-t3), z. B. die paraxialen Nebenbedingungen `make_paraxial_residuals`, später Strafterme). Das ist dieselbe Ablage wie im Ausgabepuffer der TraceEngine mit `TraceSettings::extra_residuals = E`. Die Zusatzresiduen werden auf dem Host für jede Variante des FD-Batches ausgewertet (`evaluate_extra_residuals` aus M3-T3 auf `FdJacobianOptions::extra_threads` Host-Threads, bitgleich für jede Threadzahl, Ablage [V·E]).

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

**Versagte Strahlen.** Hart versagte Strahlen haben r = 0 (M2); an einer Versagensgrenze wird die Jacobi dadurch unstetig. `status_changes` zählt pro Lauf und Spalte die Strahlen, deren Gültigkeit (Status `Ok` oder nicht) sich zwischen x₀ und einem der beiden gestörten Punkte ändert. Nur Diagnose (für M4-T2), keine Behandlung.

**Pfade.**

- CPU (`fd_jacobian_cpu`, CI ohne GPU und Referenz): `make_fd_jacobian_batch` → `residuals_cpu` für alle Varianten → `evaluate_extra_residuals` → `fd_jacobian_from_residuals` (elementweise `fd_derivative`).
- GPU (`mko::cuda::DeviceFdJacobian::compute`, Bequemlichkeit `fd_jacobian_gpu`): Batch, Zusatzresiduen und Schritte auf dem Host; ein `cudaMalloc` für J, r₀, Status-Wechsel, Zusatzresiduen [V·E], Schablonen und Schritte, angelegt vor dem Lauf der Engine (die ihn so als belegt sieht); dann `TraceEngine::run_on_device` mit Tranchen aus ganzen Läufen (Granularität n) und pro Tranche der Kernel `fd_difference_kernel` direkt auf dem Ausgabepuffer der Engine. J und r₀ bleiben auf dem Gerät (`device_jacobian`, `device_residuals`) und werden nur auf Anforderung heruntergeladen. Die Zusatzresiduen gehen nicht durch den Ausgabepuffer der Engine (`extra_residuals = 0` dort), sondern liegen in einem eigenen Gerätepuffer; die Zeilenablage von J und r₀ ist trotzdem die oben beschriebene.
- Differenzen auf dem Host aus einem Download der Engine (`engine.run` mit `extra_residuals = E`, dann `fd_jacobian_from_residuals`): bitgleich zum Gerätepfad für zentrale und gekürzte Spalten ((r_a − r_b)/(d_a − d_b) lässt keine FMA-Kontraktion zu), einseitige innerhalb der Rundung der Formel (Test `CudaFdJacobian.DevicePathMatchesHostDifferences`).

GPU gegen CPU (Test `CudaFdJacobian.MatchesCpu`): Die Formel ist linear in den Residuen, J = Σ_x c_x·r_x, also |J_GPU − J_CPU| ≤ Σ_x |c_x|·δ(r_x) mit der Residuen-Toleranz δ(r) = max(`kCpuGpuFp64`, `kCpuGpuFp64Relative`·|r|) aus M2-T2 (0 für Zusatzresiduen) plus 8ε·Σ|c_x·r_x| für die Rundung der Formel. Gemessen höchstens 2,9·10⁻³ dieser Schranke, relativ pro Spalte 5,9·10⁻⁸. Mehrere Läufe in einem Batch, Tranchen und wiederholte Rechnungen sind bitgleich (ADR-0007; keine Gleitkomma-Reduktion im Differenzkernel, die Status-Wechsel sind eine ganzzahlige Baumreduktion). Laufzeit (RTX 3090, Test `CudaFdJacobian.LargeBatch`, kein Benchmark): 1.000 Läufe des Mehrlinsers (27.000 Varianten, 680 Strahlen, J = 70,7 MB) in 85–125 ms für `compute` einschließlich Batchaufbau auf dem Host.

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

## 9. Glas-Einrasten

Am Ende: für jedes Glaselement die k nächsten Katalog-Gläser im normierten (n₀, ν₁, ν₂)-Raum; Kombinationen werden als Varianten im Batch reoptimiert, das beste Ergebnis gewinnt. Weiche Nebenbedingung auf Transmission bei 420–450 nm (Farbstich), Kennzeichnung spannungsarmer Gläser für DIC.

## 10. Literatur

- F. Bociort, M. van Turnhout et al.: Arbeiten zur Saddle-Point Construction in optischen Systemen (TU Delft, ab ca. 2007).
- M. Isshiki et al.: Global optimization with escape function (Lens design).
- H. B. Nielsen: Damping parameter in Marquardt's method (1999).
- J. Nocedal, S. J. Wright: Numerical Optimization (exakte Penalty-Funktionen).
