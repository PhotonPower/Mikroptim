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
2. **Paraxiale Gleichungsbedingungen:** √w · (Ist − Soll) für Brennweite (3,29 mm), Abgleichlänge (45 mm), ggf. Arbeitsabstand.
3. **Petzval:** √w · max(0, |κ_P| − κ_max), inklusive Beiträgen der Geisterplatten.
4. **Mechanik (einseitig):** √w · max(0, g) für Durchmesser, Mitten- und Randdicken, maximalen Einfallswinkel (Vorgabe ca. 55–60°).
5. **Weiches Versagen:** siehe numerik.md, Abschnitt 11.
6. **Geisterplatten (L1, geglättet):** √(μ·w_k) · (a_{g,k}² + ε²)^{1/4}, sodass das Quadrat μ·w_k·√(a² + ε²) ≈ μ·w_k·|a| ist.
7. **Glas-Hülle:** √w · Abstand außerhalb der Hülle.

Gewichte w_f, w_λ stammen aus der Systemdatei. Geisterplatten-Gewichte steigen mit der Ordnung: w_k = w₀ · κ^k (κ ≈ 2).

## 3. Warum L1 für Geisterplatten

Mit einer quadratischen Strafe wäre der Grenznutzen bei a = 0 null; der Optimierer würde immer ein wenig Geisterkorrektur verwenden. Mit L1 gilt die Theorie der exakten Penalty: Ist μ größer als der zugehörige Lagrange-Multiplikator, bleiben die Koeffizienten exakt null, solange das reale System das Ziel alleine erreicht. Zusätzlich entsteht Sparsity: wenige, gezielte Koeffizienten statt vieler kleiner. Die Glättung mit ε (Startwert 10⁻⁶ in Einheiten des Dickenprofils) hält LM differenzierbar.

## 4. Gebündelter Levenberg-Marquardt

- Viele unabhängige Läufe (Multistart, Homotopie-Zweige, beide Abstiegsrichtungen an Sattelpunkten) werden **gleichzeitig** iteriert; jeder Lauf hat eigenen Zustand (x, Dämpfung, Status).
- Jacobi-Matrix in Version 1 per **zentralen finiten Differenzen**: Für P Variablen werden 2P (oder P+1 vorwärts) gestörte Varianten in denselben Batch gepackt. Schrittweite h_k = ε_m^{1/3} · max(|x_k|, x_typ,k), ε_m^{1/3} ≈ 6·10⁻⁶.
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
- Schranken: zentral x⁺ = x₀ + h, falls ≤ upper, sonst upper; x⁻ = x₀ − h, falls ≥ lower, sonst lower. Liegt x₀ auf einer Schranke, ist der Schritt dorthin 0 und es bleibt eine einseitige Differenz mit dem anderen Punkt; weil lower < upper, ist d⁺ − d⁻ > 0. Vorwärts: x₀ + h, sonst der Rückwärtsschritt x₀ − h, sonst die Schranke mit dem größeren Abstand.
- Rückgabe `FdPattern`: nominelles h_k, die geschriebenen Punkte x⁺_k, x⁻_k und die tatsächlichen Schritte d = fl(x − x₀) (exakt für |x₀| ≥ x_typ und x₀ = 0, Sterbenz-Lemma; sonst auf ½ ulp). Die Ableitung schätzt M3-T2 z. B. als (r⁺ − r⁻)/(d⁺ − d⁻); die endgültige Behandlung an Schranken legt M3-T2 fest.

**Zufällig gestörte Varianten** (`perturbed_vector`, `write_perturbed_variants`, `make_perturbed_batch`), deterministisch und plattformunabhängig:

- Variante j (Zählung im Aufruf ab 0) hat einen eigenen `std::mt19937_64` (im Standard bitgenau festgelegt) mit Startwert splitmix64(seed + j·0x9E3779B97F4A7C15 mod 2⁶⁴) (SplitMix64-Mischfunktion). Variante j hängt also nicht von der Zahl der übrigen Varianten ab.
- Die Verteilungen der Standardbibliothek sind nicht plattformgleich festgelegt; Box-Muller bräuchte log, cos und sin, deren Rundung zwischen Bibliotheken um ein ulp abweichen darf. Daher eigene Umwandlungen nur mit exakten bzw. korrekt gerundeten Operationen: u = (bits ≫ 11)·2⁻⁵³ ∈ [0, 1); gleichverteilt ξ = 2u − 1 ∈ [−1, 1) (`Distribution::Uniform`, eine Ziehung pro Variable) oder Irwin-Hall ξ = Σ_{i=1}^{12} u_i − 6 (`Distribution::IrwinHall`: Mittel 0, Varianz 1, begrenzt auf (−6, 6), nahe der Normalverteilung; 12 Ziehungen pro Variable). Ziehungen in der Reihenfolge k = 0 … P − 1.
- Störung pro Variablentyp: s_k = `absolute` + `relative`·|x₀_k| (`PerturbationScale`, Standard 1 % relativ), y_k = x₀_k + s_k·ξ, beide Ausdrücke mit `std::fma` (unabhängig von der Kontraktion des Compilers), danach Projektion auf die Schranken. `first_is_base`: Variante 0 = x₀.
- Erwartungswerte für Startwert 42 stehen im Test `VariantGenerator.PerturbedValuesArePlatformIndependent` (berechnet mit einer unabhängigen Python-Implementierung); sie gelten unverändert unter MSVC und GCC.
- Brechzahlen bleiben unverändert (Glas als Variable erst M6).

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
