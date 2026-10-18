# Numerik: Konventionen, Formeln, Genauigkeit

Dieses Dokument ist verbindlich für alle Implementierungen im Mathe-Kern, im CPU-Tracer und im CUDA-Backend. Änderungen an Konventionen nur über eine ADR.

## 1. Einheiten

- Längen in **mm**, Krümmungen in **1/mm**.
- Wellenlängen in Dateien und APIs in **nm**; in Dispersionsformeln (Sellmeier, Buchdahl) in **µm**.
- Winkel in Dateien in **Grad**, intern im Bogenmaß.
- OPD intern in mm; Ausgabe zusätzlich in Wellen bei der Referenzwellenlänge.
- Strahlresiduen dimensionslos in Wellen bei der Referenzwellenlänge, die Merit in Wellen² (ab M4-T1, Abschnitt 7).

## 2. Koordinaten und Vorzeichen

- Rechtshändiges System, z ist die optische Achse **in Tracerichtung**.
- Jede Fläche hat lokale Koordinaten mit Scheitel im Ursprung. Das System ist koaxial (keine Dezentrierung bis M8).
- Krümmung c = 1/R ist **positiv**, wenn der Krümmungsmittelpunkt auf der +z-Seite des Scheitels liegt. Planfläche: c = 0.
- Fläche i hat die Dicke t_i = Abstand ihres Scheitels zum Scheitel von Fläche i+1 (entlang +z) und das Medium m_i **nach** der Fläche. Das Medium vor Fläche 0 ist Luft.
- Brechzahlen aus Katalogen sind relativ zu Luft; Luft hat im Programm n = 1.

## 3. Systemaufbau und Tracerichtung

- Getract wird vom Unendlichraum (Seite der Tubuslinse) zur Objektebene.
- Fläche 0 ist die **Aperturblende**, eine Ebene in Luft (ADR-0006). Ihr Halbdurchmesser ist der Radius der Pupille a (Ziel: a = 3,125 mm).
- Die letzte Fläche ist die Frontfläche des Objektivs; ihre Dicke ist der Arbeitsabstand bis zur Objektebene.

## 4. Startstrahlen

### Kollimierte Quelle (Standard)

Feldwinkel θ in der y-z-Ebene, Pupillenkoordinaten (ρ, φ) mit φ gemessen von +y:

$$\mathbf{p}_0 = (a\,\rho \sin\varphi,\; a\,\rho\cos\varphi,\; 0), \qquad \mathbf{d}_0 = (0,\ \sin\theta,\ \cos\theta)$$

Die optische Weglänge startet auf der ebenen Wellenfront senkrecht zu **d₀** durch den Ursprung der Blendenebene:

$$\mathrm{OPL}_0 = \mathbf{p}_0 \cdot \mathbf{d}_0$$

Ohne diesen Term wäre die OPD für θ ≠ 0 um eine Kippung verfälscht.

### Punktquelle (Tests und Gebrauchsrichtung)

Punkt **q** im Medium vor einer Fläche, Richtungen über die Quadratur im Richtungsraum; OPL₀ = 0 am Quellpunkt. Wird für analytische Tests (aplanatische Punkte) benötigt.

Festlegung (M1-T5, `point_source_ray` in `include/mikroptim/trace/trace.hpp`): **q** in lokalen Koordinaten von Fläche 0, Kegelachse +z, Richtungen gleichmäßig im Raum der Richtungskosinus mit denselben Pupillenkoordinaten (ρ, φ) wie bei der kollimierten Quelle:

$$\mathbf{d}_0 = \left(s\rho\sin\varphi,\; s\rho\cos\varphi,\; \sqrt{1 - s^2\rho^2}\right),\qquad s = \sin\vartheta_{\max} = \mathrm{NA}/n_0$$

n₀ ist die Brechzahl des Startmediums, also des Mediums nach Fläche 0 (siehe Abschnitt 5, „Sequenzieller Trace“). Bei Systemen aus Dateien ist das Luft. Für eine Quelle im Glas (aplanatische Punkte) wird eine Flächentabelle benutzt, deren Fläche 0 eine Ebene mit Glas als Medium ist; das ist nur über die Tabellen-API möglich, weil die Systemdatei die Blende in Luft verlangt (ADR-0006).

## 5. Flächenoperationen

Implementiert in `include/mikroptim/trace/surface_ops.hpp`.

**Transfer** in die Tangentialebene der nächsten Fläche (Abstand t):

$$z \leftarrow z - t,\qquad s_0 = -z/d_z,\qquad \mathbf{p} \leftarrow \mathbf{p} + s_0\,\mathbf{d}$$

**Schnitt mit der Sphäre** (Spencer und Murty, Welford), Strahl in der Ebene z = 0:

$$H = c\,(x^2+y^2),\qquad B = d_z - c\,(x\,d_x + y\,d_y),\qquad s = \frac{H}{B + \sqrt{B^2 - cH}}$$

Diese Form ist für c = 0 gültig und vermeidet Auslöschung. B² − cH < 0 oder B + √… ≤ 0 bedeutet „Fläche verfehlt“.

**Brechung** mit μ = n/n′ und Flächennormale **N** = (−c x, −c y, 1 − c z) (Einheitsvektor auf der Sphäre, zeigt in +z-Richtung):

$$\cos I = \mathbf{d}\cdot\mathbf{N},\qquad k = 1 - \mu^2(1-\cos^2 I),\qquad \mathbf{d}' = \mu\,\mathbf{d} + \left(\sqrt{k} - \mu\cos I\right)\mathbf{N}$$

k < 0 bedeutet Totalreflexion.

**Optische Weglänge:** Transfer und Schnitt laufen im Medium vor der Fläche: OPL += n_vor · (s₀ + s).

### Sequenzieller Trace (M1-T5)

Implementiert in `include/mikroptim/trace/trace.hpp` (`trace_ray`, `propagate_surface`, `transfer_to_image_plane`).

- **Start:** Der Strahl steht in lokalen Koordinaten von Fläche 0, im Medium **nach** Fläche 0 (Brechzahl index[0][l]). An Fläche 0 wird nicht gebrochen; bei Systemen aus Dateien ist sie die Blende in Luft, und die kollimierte Quelle startet in ihrer Ebene.
- **Schritt** für s = 1 … S−1: Prüfung d_z > 0, Transfer um thickness[s−1] in die Tangentialebene von Fläche s, Schnitt (Stop und Plane mit c = 0, Sphere mit c = curvature[s]), OPL += n_vor · (s₀ + s), Brechung mit μ = n_vor / n_nach, n_vor = index[s−1][l], n_nach = index[s][l]. Bei einer Geisterplatte (Typ GhostPlate, g-te ihrer Art in Tracerichtung) statt Schnitt und Brechung der Schritt aus Abschnitt 6 mit den Daten der Geisterplatte g.
- **Ende:** im Schnittpunkt mit der letzten Fläche S−1, in deren lokalen Koordinaten, Richtung nach der Brechung, OPL bis zu diesem Punkt. Hier setzt die Referenzsphäre an (Abschnitt 7, Mittelpunkt **M**_f in denselben Koordinaten).
- **Bildebene:** `transfer_to_image_plane` transportiert um thickness[S−1] im Medium index[S−1][l] in die Ebene z = 0 der Bildebene (OPL += n_B · s₀).
- **Versagen:** Der Trace kehrt bei der ersten versagenden Fläche mit deren Status und Nummer zurück (hartes Versagen, kein Werfen). Der Strahlzustand ist dann teilweise fortgeschrieben und wird nicht weiter verwendet. Weiches Versagen (Abschnitt 11) folgt in M4-T2.

### Statusmaske

`TraceStatus` (Werte stabil, als `unsigned` zwischen Host und Device ausgetauscht):

| Wert | Name | Bedeutung |
|---|---|---|
| 0 | `Ok` | Strahl hat alle Flächen passiert |
| 1 | `Missed` | Fläche verfehlt: B² − cH < 0 oder B + √(B² − cH) ≤ 0 |
| 2 | `TIR` | Totalreflexion, k < 0; an einer Geisterplatte Radikand 1 − \|d′_⊥\|² < 0 (Abschnitt 6) |
| 3 | `Backward` | d_z ≤ 0 (oder NaN) vor einem Transfer oder nach der Brechung an der letzten Fläche (vor dem Schritt auf die Referenzsphäre, dann versagende Fläche S); Abschnitt 11 |
| 4 | `Unsupported` | Flächentyp vom Tracer nicht unterstützt (z. B. Fläche vom Typ GhostPlate ohne Geisterdaten in der Tabelle) |

Zu jedem Strahl wird die Nummer der versagenden Fläche gemeldet (−1 ohne Versagen).

## 6. Geisterplatte

Virtuelle dünne Glasplatte in einem **Luftspalt**, modelliert als Ebene mit Phasenfunktion. Normierungsradius r_g, u = (r/r_g)², radiales Dickenprofil als Summe rotationssymmetrischer Zernike-Polynome (äquivalent zu Legendre-Polynomen in u):

$$t(r) = \sum_{k=1}^{K} a_k\,P_k(2u-1)$$

Phase und Ablenkung (Medium n_m = 1, virtuelles Glas n_G(λ)):

$$\Phi = (n_G(\lambda) - n_m)\,t(r), \qquad n_m\,\mathbf{d}'_\perp = n_m\,\mathbf{d}_\perp + \nabla\Phi, \qquad d'_z = \sqrt{1 - |\mathbf{d}'_\perp|^2}$$

$$\nabla t = \frac{4}{r_g^2}\,\sum_k a_k\,P_k'(2u-1)\,(x,\ y)$$

Rekursionen: (k+1)P_{k+1} = (2k+1)ξP_k − kP_{k−1} und P′_{k+1} = P′_{k−1} + (2k+1)P_k. OPL += Φ. Radikand < 0 bedeutet Versagen.

Paraxiale Brechkraft der Geisterplatte (für die Petzval-Nebenbedingung, Beitrag φ_G/n_G):

$$\varphi_G = -2\,(n_G - 1)\,t_2,\qquad t_2 = \frac{1}{r_g^2}\sum_k a_k\,(-1)^{k+1}\,k(k+1)$$

Begründung für das virtuelle Glas statt einer abstrakten Phase: Eine wellenlängenunabhängige Phase wäre achromatisch, und eine ideale Phasenfläche hätte keinen Petzvalbeitrag. Beides würde dem Optimierer genau die schwersten Probleme (Farbe, Bildfeldebnung) „gratis“ lösen.

### Festlegungen der Implementierung (M4-T5)

Implementiert in `include/mikroptim/trace/ghost_plate.hpp` (header-only, `MKO_HD`, templatisiert über R), Zweig `SurfaceKind::GhostPlate` in `propagate_surface` (`include/mikroptim/trace/trace.hpp`). Daten in der Flächentabelle, Batch und Kernel: [architektur.md](architektur.md#geisterplatten-m4-t5), Dateiformat: [dateiformat.md](dateiformat.md#geisterplatten).

**Prüfung der Formeln** (keine Abweichung von den Formeln oben gefunden):

- Gradient: Mit u = (x² + y²)/r_g² ist ∂u/∂x = 2x/r_g² und dξ/du = 2 (ξ = 2u − 1), also dt/du = 2 Σ a_k P′_k(ξ) und ∇t = (dt/du)·(2/r_g²)·(x, y) = (4/r_g²) Σ a_k P′_k(ξ)·(x, y).
- Vorzeichen der Ablenkung: Die Platte verzögert die Welle um Φ; hinter ihr ist das Eikonal S′ = S + Φ und n_m **d** = ∇S, also n_m **d**′_⊥ = n_m **d**_⊥ + ∇Φ. Eine in der Mitte dickere Platte (t₂ < 0) lenkt zur Achse hin ab (sammelnd, φ_G > 0), wie eine Sammellinse aus demselben Glas.
- t₂: Aus P′_k(−1) = (−1)^{k+1} k(k+1)/2 folgt t₂ = dt/d(r²) bei r = 0 = (1/r_g²)·dt/du(0) = (1/r_g²) Σ a_k (−1)^{k+1} k(k+1).
- φ_G: Paraxial ist ∇Φ = 2(n_G − n_m) t₂ (x, y), also n_m u′ = n_m u − y φ_G mit φ_G = −2(n_G − n_m) t₂; für n_m = 1 die Formel oben.
- Petzvalbeitrag: Eine dünne Linse aus dem Glas n_G im Medium n_m hat Σ c (n′ − n)/(n n′) = φ/(n_m n_G); für Luft φ_G/n_G wie oben.

**Schritt an der Geisterplatte.** Der Strahl wird wie bei einer Ebene in die Ebene z = 0 der Geisterplatte transportiert (OPL += n_m·s₀, Schnitt s = 0), dann:

$$u = \frac{x^2 + y^2}{r_g^2},\quad \Phi = (n_G - n_m)\,t(u),\quad \nabla\Phi = (n_G - n_m)\,\frac{dt}{du}\,\frac{2}{r_g^2}\,(x,\ y),\quad \mathrm{OPL} \mathrel{+}= \Phi,\quad n'\,\mathbf{d}'_\perp = n\,\mathbf{d}_\perp + \nabla\Phi,\quad d'_z = \sqrt{1 - |\mathbf{d}'_\perp|^2}$$

mit n_m = n = n_vor und n′ = n_nach. Tabellen aus Systemdateien haben n = n′ = 1 (Luftspalt, Validierung im Dateiformat); n ≠ n′ ist als brechende Ebene mit Phase definiert (Tabellen-API). Der allgemeine Schritt mit gegebenem Φ und ∇Φ ist `phase_surface_step` (z. B. lineare Phase Φ = a·y), das Profil `ghost_profile` (t und dt/du in einem Durchlauf der Rekursionen, Summation in aufsteigender Ordnung k), der Schritt an der Geisterplatte `ghost_plate_step`.

- **Status:** Radikand 1 − |**d**′_⊥|² < 0 meldet `TIR` (kein neuer Statuswert): Es ist dasselbe Ereignis wie die Totalreflexion (die abgelenkte Welle ist evaneszent), und das weiche Versagen (Abschnitt 11) behandelt beide gleich (Strafresiduum −k bzw. −Radikand). Ob eine Geisterplatte versagt hat, zeigt die versagende Fläche (Typ GhostPlate). Eine Fläche vom Typ GhostPlate ohne Daten in der Tabelle ergibt `Unsupported`.
- **Außerhalb von r_g** (u > 1, ξ > 1) wird das Profil ohne Sonderbehandlung ausgewertet; die Polynome wachsen, bleiben aber endlich. Der Normierungsradius sollte daher den genutzten Strahlbereich abdecken.
- **Ordnungen:** K ≤ `kMaxGhostTerms` = 32. Haben die Geisterplatten eines Systems verschiedene Ordnungen, ist K der Tabelle die größte; kürzere Profile werden mit 0 ergänzt, was das Ergebnis bitgenau nicht ändert.
- **Paraxial:** `ghost_profile_t2`, `ghost_plate_power` (φ_G) und `ghost_plate_petzval` (φ_G/(n_m n_G)); Einbindung in den y-nu-Trace in Abschnitt 9.

**Ergebnisse der Abnahme** (Tests in `tests/test_ghost_plate.cpp`, GPU in `tests/test_cuda_residuals.cpp`):

| Kriterium | Ergebnis |
|---|---|
| Lineare Phase Φ = a·y (n = 1; 1,33; 1,7, a von 10⁻⁶ bis 0,4, verschiedene Richtungen): Δ(n d_y) = a | max \|Δ(n d_y) − a\| = 1,1·10⁻¹⁶; OPL wächst exakt um Φ |
| Gradient des Dickenprofils gegen zentrale finite Differenzen mit Richardson-Extrapolation (5 Stufen), K = 1 … 16, r_g = 0,8 … 12 mm, 840 Punkte bis r = 1,1·√2·r_g | max. relativ 1,6·10⁻¹¹ (< 1·10⁻¹⁰) |
| Ideale Fokussierphase Φ(r) = −(√(r² + F²) − F), F = 10 mm, NA 0,3, Geisterplatte direkt hinter der Blende, Legendre-Fit (L²-Projektion mit 16 Gauß-Knoten), Referenzsphäre im Brennpunkt | RMS-OPD 2,5·10⁻⁹ (K = 4), 8,2·10⁻¹³ (K = 6), 3,8·10⁻¹⁶ mm (K = 12, < 1·10⁻⁶) |
| φ_G nach Formel gegen den Achsschnittpunkt realer Strahlen (dünnes Element in Luft, vier Profile bis K = 8) | Längsaberration ∝ h²; bei h = 10⁻⁷ r_g Abweichung ≤ 1,3·10⁻¹⁰ mm bei \|1/φ_G\| bis 4.558 mm |
| Gesamtsystem `einzellinse_geisterplatte.json`: paraxiale Schnittweite gegen realen Strahl (h = 10⁻⁴ mm), Petzvalbeitrag Σ φ_G/n_G | 3,4·10⁻¹³ mm; Petzval innerhalb `kParaxialRelative` |

Die gefittete Phase ist nur auf der Achse ideal; bei 0,5° Feldwinkel beträgt die RMS-OPD 4,5·10⁻⁴ mm (Koma der Phasenplatte, nur dokumentiert).

## 7. Referenzsphäre und OPD

Nach der letzten Fläche (lokale Koordinaten, Medium n_B, Luft): idealer Bildpunkt **M**_f = (0, h_f, w) mit Arbeitsabstand w und h_f = f_soll · tan θ_f. Referenzsphäre um **M**_f mit Radius R_ref = |**M**_f| (durch den Scheitel der letzten Fläche). Mit **q** = **p** − **M**_f, b = −**q**·**d**, g = |**q**|² − R_ref²:

$$s_{\mathrm{ref}} = \frac{g}{\,b + \sqrt{b^2 - g}\,},\qquad \mathrm{OPL} \mathrel{+}= n_B\, s_{\mathrm{ref}}$$

s_ref darf negativ sein. Residuum pro Strahl ist die OPD nach Kolbenabzug:

$$\mathrm{OPD}_i = \mathrm{OPL}_i - \sum_j W_j\,\mathrm{OPL}_j$$

Kippung wird **nicht** entfernt: Sie entspricht Verzeichnung bzw. Brennweitenfehler und soll bestraft werden. Für die Endbewertung (M8) wird die Referenzsphäre durch die paraxiale Austrittspupille gelegt; der Unterschied ist von zweiter Ordnung in der Queraberration.

### Festlegungen der Implementierung (M1-T7)

Implementiert in `include/mikroptim/trace/reference_sphere.hpp` (`reference_sphere_step`, `trace_to_reference_sphere`; header-only, `MKO_HD`), `include/mikroptim/merit/opd.hpp` (`opd_statistics`; header-only, `MKO_HD`) und `include/mikroptim/merit/wavefront.hpp` (Host-API, Bibliothek `Mikroptim::merit`).

**Wahl des Schnittpunkts.** Der Strahl **p** + s **d** (|**d**| = 1) schneidet die Sphäre |**x** − **M**_f| = R_ref, wo s² − 2bs + g = 0, also s_± = b ± √(b² − g) mit s₊ s₋ = g. Gesucht ist der Schnitt mit der Kappe, die dem Scheitel der letzten Fläche zugewandt ist (die Wellenfront durch den Scheitel), d. h. die betragskleinere Nullstelle für Strahlen in der Nähe der Fläche:

- **Reeller Bildpunkt** (M_z ≥ 0, Strahlen laufen auf **M**_f zu): b ≈ |**q**| > 0, gesucht ist s₋ = b − √(b² − g) = g / (b + √(b² − g)). Das ist die Formel oben, in der auslöschungsfreien Form. s₊ läge jenseits von **M**_f auf der Gegenseite der Sphäre.
- **Virtueller Bildpunkt** (M_z < 0, Strahlen laufen von **M**_f weg; z. B. aplanatische Punkte, Abnahme M1-T7): b ≈ −|**q**| < 0, gesucht ist s₊ = g / (b − √(b² − g)). Die Formel oben lieferte hier die ferne Nullstelle und für den Achsstrahl im Scheitel 0/0.

Zusammengefasst, für M_z ≥ 0 identisch mit der Formel oben:

$$s_{\mathrm{ref}} = \frac{g}{\,b + \sigma\sqrt{b^2 - g}\,},\qquad \sigma = \begin{cases} +1 & M_z \ge 0\\ -1 & M_z < 0\end{cases}$$

σ hängt nur vom Bildpunkt ab, der Zweig ist also für alle Strahlen eines Feldes gleich. g wird als **p**·(**p** − 2**M**_f) gerechnet; das ist wegen R_ref = |**M**_f| algebraisch gleich |**q**|² − R_ref², vermeidet aber die Differenz zweier Quadrate der Größe R_ref² (≈ 10⁴ mm²). Danach steht der Strahl auf der Referenzsphäre (**p** += s_ref **d**), OPL += n_B s_ref mit n_B = index[S−1][l] (Medium nach der letzten Fläche).

**Versagen.** Läuft der Strahl nach der Brechung an der letzten Fläche rückwärts (d_z ≤ 0 oder NaN), meldet `trace_to_reference_sphere` `Backward` mit versagender Fläche S, bevor die Sphäre geschnitten wird (`trace_ray` selbst prüft d_z nur vor einem Transfer). b² − g < 0 (Gerade verfehlt die Sphäre) oder σ (b + σ√(b² − g)) ≤ 0 bzw. NaN (die zugewandte Kappe wird nicht getroffen) ergibt den Status `Missed`; als versagende Fläche wird S gemeldet (die Referenzsphäre zählt als Fläche nach der letzten). Der Strahl bleibt dann unverändert.

**Idealer Bildpunkt und f_soll.** w = thickness[S−1] (Arbeitsabstand; negativ bei einem virtuellen Bildpunkt), h_f = f_soll · tan θ_f mit dem Feldwinkel θ_f der kollimierten Quelle. f_soll ist `target.focal_length`, falls gesetzt, sonst die paraxiale Brennweite 1/φ (Abschnitt 9) bei `spectrum.reference_nm`. Ist sie nicht endlich (afokales System), meldet der Host einen Fehler. Bei der Punktquelle (Tests) ist **M** = (0, 0, w); für Quellen außerhalb der Achse gibt der Aufrufer die Bildpunkte explizit vor.

**Quadratur pro Feld.** Standard N = 10, M = 8 (Abschnitt 8), konfigurierbar. Kollimiert: θ_f = 0 → `PupilMode::Axis` (N Stützstellen), sonst `Half` (N·M). Punktquelle: auf der Achse `Axis`, mit x = 0 `Half`, sonst `Full`. Ein vorgegebener Modus gilt für alle Felder.

**Kolbenabzug und RMS** pro Tupel (Feld f, Wellenlänge l) über die Stützstellen k mit den Gewichten W_k = W_i V_j der Quadratur (Summe 1). Versagte Strahlen sind in M1 **hart** ausgeschlossen, die Gewichte der übrigen werden renormiert:

$$W_{\mathrm{ok}} = \sum_{k\,\mathrm{ok}} W_k,\qquad \overline{\mathrm{OPL}} = \frac{1}{W_{\mathrm{ok}}}\sum_{k\,\mathrm{ok}} W_k\,\mathrm{OPL}_k,\qquad \mathrm{OPD}_k = \mathrm{OPL}_k - \overline{\mathrm{OPL}},\qquad \mathrm{RMS} = \sqrt{\frac{1}{W_{\mathrm{ok}}}\sum_{k\,\mathrm{ok}} W_k\,\mathrm{OPD}_k^2}$$

Ohne Versagen ist W_ok = 1 (bis auf Rundung), und das ist genau der Kolbenabzug oben. Gemeldet werden zusätzlich die Zahl der versagten Strahlen und W_ok. Ohne gültigen Strahl ist der RMS ungültig (Host: NaN). In `opd_statistics` ist die OPD versagter Strahlen 0 (maskiert), in der Host-API NaN. Weiches Versagen (Abschnitt 11) folgt in M4-T2.

**Summationsreihenfolge und Determinismus.** Zwei Durchläufe: (1) W_ok und Σ W_k OPL_k, daraus der Kolben; (2) Σ W_k OPD_k². Die Formel √(⟨OPL²⟩ − ⟨OPL⟩²) wäre bei OPL ≈ 100 mm und OPD ≈ 10⁻⁶ mm vollständig ausgelöscht. Die CPU summiert in aufsteigender Reihenfolge des Quadraturindex k (ringweise, innen nach außen). Der Kernel (M2-T2) bildet dieselben Summen als Baumreduktion im Block in fester Reihenfolge (ADR-0007): erst die Paare (W_k, W_k OPL_k), dann den Kolben per Shared Memory an alle Threads, dann W_k OPD_k². Er ist damit bitgleich reproduzierbar, weicht aber von der CPU um Rundungsfehler ab (`kCpuGpuFp64`).

**Einheiten.** OPL, OPD und RMS in mm. In Wellen immer bei der Referenzwellenlänge (Abschnitt 1), auch für die anderen Wellenlängen des Spektrums: OPD [λ] = OPD [mm] / (λ_ref [nm] · 10⁻⁶).

**Vorzeichen.** OPD = OPL − Kolben. Läuft die Wellenfront am Pupillenrand der Referenzsphäre voraus (Unterkorrektur, Randstrahlen schneiden die Achse vor dem paraxialen Fokus), ist die OPL dort kürzer und die OPD negativ. Gegenüber der Wellenaberration W nach Welford (W₀₄₀ = S_I/8 > 0 bei Unterkorrektur) gilt also OPD = −W + const. Der Test `Wavefront.SeidelSphericalAberrationOfSingleLens` prüft das über den Achsschnittpunkt des reellen Randstrahls.

**Ergebnisse der Abnahme** (Tests in `tests/test_wavefront.cpp`):

| Kriterium | Ergebnis |
|---|---|
| Aplanatische Punkte (n = 1,5 und 1,8, sin θ ≤ 0,95, Quelle auf und neben der Achse), Referenz im exakten virtuellen Bildpunkt | RMS-OPD ≤ 1,7·10⁻¹⁴ mm, max \|OPD\| ≤ 2,0·10⁻¹⁴ mm |
| Einzellinse `einzellinse_bk7.json`, Feld 0, Bildebene im paraxialen Fokus, NA = 0,049 (a = 5 mm), drei Wellenlängen | RMS gegen \|W₀₄₀\|·√(4/45): ≤ 1,24·10⁻³ relativ; ρ⁴-Koeffizient des Fits gegen −W₀₄₀: ≤ 7,6·10⁻⁶ relativ |
| dieselbe Linse, NA = 0,025 (a = 2,5 mm) | ≤ 3,1·10⁻⁴ bzw. ≤ 4,7·10⁻⁷ relativ |
| Kugelwelle, Referenzzentrum axial um δz verschoben (konvergent, direkt konstruiert; divergent, über `evaluate_wavefront`) | max \|OPL − exakt\| = 1,4·10⁻¹⁴ mm, max \|OPD − exakt\| = 3,6·10⁻¹⁵ mm |

Die Abweichung des RMS von der Seidel-Näherung ist der Anteil höherer Ordnung (W₀₆₀ ρ⁶ usw., relativ ∝ NA²: Faktor 4 zwischen NA 0,049 und 0,025); der Fit A ρ⁴ + B ρ⁶ + K trennt ihn ab. Die Lage der Referenzsphäre durch den Scheitel statt durch die Austrittspupille ändert auf der Achse nur Terme zweiter Ordnung in der Queraberration und ist darin enthalten.

### Residuen pro Strahl (M2, Gewichte M4-T1)

Festlegung für den Kernel `RESIDUALS` und seine CPU-Referenz (M2-T2, `include/mikroptim/merit/ray_residual.hpp`, Ablage: [architektur.md, „Kernel RESIDUALS“](architektur.md#kernel-residuals-m2-t2)). Pro Variante, Feld f, Wellenlänge l und Stützstelle k:

$$r_k = s_{f,l}\,\sqrt{W_i V_j}\;\mathrm{OPD}_k \quad [\text{Wellen bei } \lambda_{\mathrm{ref}}],\qquad s_{f,l} = \frac{\sqrt{\hat w_f\,\hat w_\lambda}}{\lambda_{\mathrm{ref}}},\qquad \hat w_f = \frac{w_f}{\sum_{f'} w_{f'}},\quad \hat w_\lambda = \frac{w_\lambda}{\sum_{l'} w_{\lambda'}}$$

mit OPD_k nach Kolbenabzug wie oben (hartes Versagen: versagte Strahlen aus dem Kolben ausgeschlossen, Gewichte renormiert, OPD_k = r_k = 0, Status gesetzt; ohne gültigen Strahl im Tupel alle OPD und Residuen 0) und √(W_i V_j) = `residual_weight(sample)`. w_f = `field_weight[f]` (`source.field_weights`, Punktquelle: ein Feld mit Gewicht 1), w_λ = `wavelength_weight[l]` (`spectrum.weights`), λ_ref = `reference_wavelength_nm`·10⁻⁶ mm (`spectrum.reference_nm`, wie die Ausgabe in Wellen in M1-T7). Bildpunkt **M**_f = (0, `field_target[v][f]`, `thickness[v][S−1]`), Pupillenart pro Feld aus der Topologie des Batches. Die OPD in den Ausgaben (`ResidualResult::opd`, `tuple_wavefronts`, `mikroptim trace --gpu`) bleibt in mm und ungewichtet.

**Merit:** F_v = Σ_k r_k² über alle Strahlresiduen der Variante (plus einem vom Aufrufer gelieferten Anteil weiterer Residuen); ohne gültigen Strahl ist F_v = 0 und nur an der Zahl versagter Strahlen erkennbar. Ohne Versagen ist Σ_k W_k OPD_k² = RMS² des Tupels (Quadraturgewichte mit Summe 1), also

$$F_v = \sum_f \sum_\lambda \hat w_f\,\hat w_\lambda \Big(\frac{\mathrm{RMS}_{f\lambda}}{\lambda_{\mathrm{ref}}}\Big)^2 \quad [\text{Wellen}^2],$$

der gewichtete mittlere quadratische RMS-Wellenfrontfehler in Wellen bei λ_ref (Maréchal-Kriterium RMS ≤ λ/14 ⇔ F ≤ 5,1·10⁻³ bei gleichem RMS in allen Tupeln). Mit Versagen ist der Beitrag eines Tupels W_ok·RMS² (die Gewichte der Residuen werden nicht renormiert). Summationsreihenfolge, Max-Reduktionen (größte Strahlhöhe, kleinster cos I pro Fläche) und Toleranzen: [architektur.md, „Kernel MERIT“](architektur.md#kernel-merit-m2-t3).

**Normierung der Gewichte (M4-T1).** w_f und w_λ werden jeweils auf Summe 1 normiert, nicht in der Datei, sondern in `residual_scale`. Gründe: (1) Die Wellenfrontmerit hat damit eine feste Bedeutung (mittlerer RMS² in Wellen²), unabhängig von der Zahl der Felder und Wellenlängen und vom Maßstab der Gewichte in der Datei (Gewichte 1, 1, 1 und 2, 2, 2 sind gleich). (2) Die Homotopie über Feld und Spektrum (optimierung.md, Abschnitt 7) fügt Felder und Wellenlängen hinzu; ohne Normierung würde dabei der Wellenfrontanteil gegenüber den Nebenbedingungen wachsen (bei F·L Tupeln um den Faktor F·L) und deren Gewichte müssten mitgeführt werden. (3) Die Gewichte der Nebenbedingungen lassen sich einmal gegen einen RMS in Wellen wählen (Faustregel in optimierung.md, Abschnitt 2.2). λ_ref statt der Wellenlänge des Tupels, damit alle Tupel dieselbe Längeneinheit haben (wie die Ausgabe in Wellen, Abschnitt 1).

**Gewicht 0.** Ist w_f = 0 oder w_λ = 0, ist s_{f,l} = 0 ohne Division (die Summen dürfen dann 0 sein; in der Datei verlangt die Validierung Summe > 0, im Batch nicht): Das Tupel wird weiter verfolgt (OPD, Status, versagte Strahlen und Max-Reduktionen bleiben, auch für die Vorfilterung), seine Residuen sind exakt 0. Negative Gewichte lehnen Datei- und Batch-Validierung ab.

**Rechnung.** `residual_scale(view, f, l)` (header-only, `MKO_HD`) summiert die Gewichte in aufsteigender Reihenfolge und rechnet √((w_f/Σw_f)·(w_λ/Σw_λ)) / (λ_ref,nm · 10⁻⁶): kein Ausdruck a·b + c (keine FMA-Kontraktion), Division und Wurzel in FP64 auf dem Gerät korrekt gerundet. CPU-Referenz, Kernel `RESIDUALS`, `MERIT` (Rechentyp FP64 und df64; s wird einmal pro Block in den Rechentyp gewandelt) und die FP32-Vorfilterung (s in float) erhalten so denselben Faktor; für FP64 ist er auf CPU und GPU bitgleich. Ein Tupel allein ergibt exakt 1/λ_ref.

**Abnahme M4-T1** (Tests `MeritCpu.EqualsWeightedSumOfWavefrontRms`, `CudaMerit.EqualsWeightedSumOfWavefrontRms`; Mehrlinser mit gleichen, ungleichen und ausgeschalteten Gewichten, Einzellinse mit ungleichen Gewichten, Einzellinse mit Geisterplatten, Punktquelle neben der Achse): F gegen Σ ŵ_f ŵ_λ (RMS_fλ/λ_ref)² mit dem RMS aus `evaluate_wavefront` (M1-T7), Toleranz `kWeightedMerit` = 10⁻¹² relativ. Ergebnisse (RTX 3090):

| Vergleich | größte relative Abweichung von F |
|---|---|
| CPU (`merit_cpu`) gegen Σ ŵ ŵ (RMS/λ_ref)² aus `evaluate_wavefront` (M1-T7) | 4,0·10⁻¹⁶ (OPD bitgleich, nur Rundung von s, r und Summen) |
| GPU FP64 gegen Σ ŵ ŵ (RMS/λ_ref)² aus der OPD desselben Laufs | 2,5·10⁻¹⁶ |
| GPU df64 gegen dasselbe aus der df64-OPD | 5,0·10⁻¹⁵ |
| GPU FP64 gegen M1-T7 (CPU) | 1,3·10⁻¹² (Mehrlinser mit ausgeschalteten Tupeln; übrige ≤ 3,7·10⁻¹³) |
| GPU df64 gegen M1-T7 (CPU) | 4,6·10⁻¹¹ |

Die Gewichtung selbst ist damit auf CPU und GPU auf Rundungsgenauigkeit (< 10⁻¹⁴) belegt. Der Vergleich GPU gegen CPU enthält zusätzlich die Abweichung der OPD (FMA-Kontraktion, `kCpuGpuFp64` = 10⁻¹² mm pro Strahl bzw. `kDf64VsFp64` = 10⁻⁹ mm): Relativ ist δF/F ≈ 2 δOPD/RMS, bei RMS um 10⁻⁴ … 10⁻³ mm also bis ≈ 10⁻¹¹ in FP64. Geprüft wird er deshalb in der Wurzel mit der daraus abgeleiteten Schranke |√F_GPU − √F_M1| ≤ δ·√(Σ_(f,l) s²) + 10⁻¹²·√F (Test `CudaMerit.EqualsWeightedSumOfWavefrontRms`), nicht mit 10⁻¹² relativ.

Der Kernel summiert den Kolben als Baumreduktion (siehe „Summationsreihenfolge und Determinismus“), die CPU-Referenz sequenziell mit `opd_statistics`. Beide summieren die rohen OPL; eine Zentrierung um einen Referenzwert (z. B. die OPL des Hauptstrahls) würde die Auslöschung bei OPL ≈ 100 mm verringern, müsste dann aber in CPU und Kernel gleich definiert werden. Für df64 ist sie nicht nötig: Der Rundungsfehler der OPL in df64 liegt bei einigen u²·OPL ≈ 10⁻¹² mm, die gemessene Abweichung der OPD von FP64 bei einigen 10⁻¹¹ mm (Abschnitt 12, „Zahlentyp df64“).

## 8. Pupillenquadratur

Mittelwert über die Kreispupille als Gauß-Quadratur (Forbes):

$$\langle f\rangle = \frac{1}{\pi}\int_0^{2\pi}\!\!\int_0^1 f\,\rho\,d\rho\,d\varphi \approx \sum_{i=1}^{N}\sum_{j=1}^{M} W_i\,V_j\,f(\rho_i,\varphi_j)$$

- Radial: Gauß-Legendre auf u = ρ² ∈ [0, 1], Knoten u_i, ρ_i = √u_i, Gewichte W_i (Summe 1). Exakt für Polynome in u bis Grad 2N − 1.
- Azimutal, halbe Pupille (Symmetrie x → −x bei Feld entlang y): φ_j = (j − ½)·π/M, V_j = 1/M. Exakt für cos(mφ) mit m < 2M.
- Auf der Achse: M = 1 (nur ein radialer Fächer).
- Standard: N = 10, M = 8. Die Gewichte gehen als √(W_i V_j) in die Residuen ein.

### Festlegungen der Implementierung (M1-T4)

Implementiert in `include/mikroptim/pupil/quadrature.hpp` (header-only, `MKO_HD`).

**Modi.** In 0-basierter Zählung (j = 0 … M−1) und mit V_j = 1/M in allen Modi:

| Modus | Azimut φ_j | exakt für |
|---|---|---|
| `PupilMode::Half` | (j + ½)·π/M, also φ ∈ (0, π), Hälfte x > 0 | cos(mφ), 0 ≤ m ≤ 2M − 1 |
| `PupilMode::Full` | (j + ½)·2π/M, also φ ∈ (0, 2π) | cos(mφ), sin(mφ), 0 ≤ m ≤ M − 1 |
| `PupilMode::Axis` | φ = 0 (M = 1, Fächer entlang +y) | nur rotationssymmetrische Funktionen (m = 0) |

- Volle Pupille: dieselbe Mittelpunktsregel wie die halbe. Die ersten M/2 Speichen einer vollen Regel mit gerader Speichenzahl M sind bitgleich die Speichen der halben Regel mit M/2 Speichen, die übrigen deren Spiegelbilder x → −x (φ → 2π − φ). Gebraucht wird die volle Pupille, sobald die Symmetrie x → −x fehlt (M8-T3).
- Achsfall: Auf der Achse ist die Wellenfront rotationssymmetrisch, ein radialer Fächer genügt. φ = 0 statt des Werts π/2, den die halbe Regel für M = 1 liefert, weil dann sin φ = 0 und cos φ = 1 exakt sind: die Startpunkte p₀ = (0, aρ, 0) liegen exakt im Meridionalschnitt (x = 0), unabhängig von der Rundung von π. `make_pupil_quadrature` ersetzt im Achsfall die Speichenzahl durch 1.

**Index-Abbildung.** Ringweise (ring-major): `index = i·M + j` mit Ring i = 0 … N−1 (u aufsteigend, innen nach außen) und Speiche j = 0 … M−1. Gewicht der Stützstelle W_i·V_j = W_i / M (eine Division), Summe über alle Stützstellen 1. Ein Kernel-Thread berechnet seine Stützstelle mit `sample(q, index)` aus dem Index.

**Knoten und Gewichte.** Tabelle `include/mikroptim/pupil/gauss_legendre_table.hpp` für N = 1 … 16, erzeugt von `tools/gen_gauss_legendre_table.py` (Python-Standardbibliothek `decimal`, 60 Stellen, Newton-Iteration). Die FP64-Werte sind korrekt gerundet; ein Test vergleicht sie bitgenau mit einer unabhängigen Rechnung in doppelt-double-Arithmetik.

**Exaktheitsbereich für Zernike-Polynome** Z = R_n^m cos(mφ) (Tensorprodukt der beiden Regeln):

- Mittelwert ⟨R_n^0⟩ = δ_n0 exakt für gerade n ≤ 4N − 2 (R_n^0 hat Grad n/2 in u).
- Orthogonalität ⟨Z_a Z_b⟩ = δ_ab (1 + δ_m0) / (2(n + 1)) auf der halben Pupille exakt, wenn m + m′ ≤ 2M − 1 und zusätzlich n + n′ ≤ 4N − 2 oder m + m′ ungerade. Für m + m′ gerade ist R_n^m R_n′^m′ ein Polynom vom Grad (n + n′)/2 in u; für m + m′ ungerade verschwindet das Ergebnis schon azimutal exakt. Insbesondere sind alle Z mit n ≤ 2N − 1 paarweise exakt orthogonal (Abnahmekriterium „bis n = 2N − 1“). Auf der vollen Pupille gilt dasselbe mit m + m′ ≤ M − 1, zusätzlich für die Sinusterme.

**API** (Namensraum `mko`):

```cpp
enum class PupilMode : int { Half = 0, Full = 1, Axis = 2 };
inline constexpr int kMaxPupilRings = 16;

template <class R> struct PupilQuadrature {   // trivial kopierbar, per Wert an den Kernel
    int rings, spokes; PupilMode mode;
    R u[kMaxPupilRings], w[kMaxPupilRings];   // Knoten u_i = ρ_i² aufsteigend, Gewichte W_i
};
template <class R> struct PupilSample { R rho, phi, weight; };

// Nur Host; wirft std::invalid_argument bei rings ∉ [1, 16] oder spokes < 1.
template <class R = double>
PupilQuadrature<R> make_pupil_quadrature(int rings, int spokes, PupilMode mode);

template <class R> MKO_HD int sample_count(const PupilQuadrature<R>& q);          // N·M
template <class R> MKO_HD PupilSample<R> sample(const PupilQuadrature<R>& q, int index);
template <class R> MKO_HD R residual_weight(const PupilSample<R>& s);             // √(W_i V_j)
```

## 9. Paraxialoptik

y-nu-Trace mit Brechkraft φ_i = c_i (n′_i − n_i):

$$n'_i u'_i = n_i u_i - y_i\,\varphi_i,\qquad y_{i+1} = y_i + t_i\,u'_i$$

- Brennweite bei kollimierter Eingabe (u₀ = 0): f = −y₀ / u′_letzte.
- Petzvalsumme und Krümmung der Petzvalschale:

$$P = \sum_i c_i\,\frac{n'_i - n_i}{n_i\,n'_i},\qquad \kappa_P = -\,n'_{\mathrm{letzte}}\,P$$

Geisterplatten tragen φ_G/n_G zu P bei (Abschnitt 6).

### Festlegungen der Implementierung (M1-T6)

Implementiert in `include/mikroptim/paraxial/ynu_trace.hpp` (Kern, header-only, `MKO_HD`) und `include/mikroptim/paraxial/paraxial.hpp` (Host-API, Bibliothek `Mikroptim::paraxial`).

**Trace.** u ist der paraxiale Winkel (Steigung dy/dz) im jeweiligen Medium, das Medium vor Fläche 0 ist Luft (n₀ = 1). Pro Fläche s: erst Übergang y += t_{s−1} u (ab s = 1), dann Brechung n′u′ = nu − y c (n′ − n). Die Dicke der letzten Fläche (Abstand zur Bildebene) geht nicht in den Trace ein.

**Systemmatrix.** Zwei Strahlen ab Fläche 0 liefern die Matrix in reduzierten Winkeln von Fläche 0 (vor der Brechung) bis zur letzten Fläche L (nach der Brechung):

$$\begin{pmatrix} y_L \\ n'_L u'_L \end{pmatrix} = \begin{pmatrix} A & B \\ C & D \end{pmatrix} \begin{pmatrix} y_0 \\ u_0 \end{pmatrix},\qquad AD - BC = 1$$

Randstrahl (y₀ = 1, u₀ = 0) → A, C; Hauptstrahl durch die Blendenmitte (y₀ = 0, u₀ = 1) → B, D.

**Kenngrößen.** Alle Lagen sind z-Koordinaten entlang +z (Tracerichtung), gemessen vom Scheitel der genannten Fläche; negativ heißt „vor der Fläche“.

| Größe | Formel | Bezug |
|---|---|---|
| Brechkraft φ | −C | |
| effektive Brennweite f | 1/φ; bei Bildraum Luft gleich −y₀/u′_L (Formel oben) | |
| hintere Schnittweite s′_F | −y_L/u′_L des Randstrahls = n′_L A/φ | letzte Fläche → F′ |
| vordere Schnittweite s_F | −D/φ | Fläche 0 → F |
| vordere Hauptebene z_H | (1 − D)/φ (F liegt um f vor H) | Fläche 0 |
| hintere Hauptebene z_H′ | n′_L (A − 1)/φ (F′ liegt um n′_L f hinter H′) | letzte Fläche |
| Defokus der Bildebene | s′_F − t_L (> 0: Brennpunkt hinter der Bildebene) | |
| Petzvalsumme, -krümmung | P und κ_P wie oben; Petzvalradius 1/κ_P | |
| Austrittspupille z_AP | −y_L/u′_L des Hauptstrahls = −n′_L B/D | letzte Fläche |
| Pupillenvergrößerung m_P | 1/D (Lagrange: n₀u₀/(n′_L u′_L)) | |
| Radius der Austrittspupille | a · \|m_P\| (a = Halbdurchmesser der Blende) | |

Die effektive Brennweite ist 1/φ und nicht −y₀/u′_L, damit sie auch bei einem Bildraum mit n′ ≠ 1 (Immersion) die für die Vergrößerung maßgebliche Brennweite auf der Luftseite (Tubuslinse) ist; bei Bildraum Luft sind beide gleich. Für eine dicke Linse in Luft ergeben sich die Linsenmacherformel 1/f = (n − 1)[c₁ − c₂ + (n − 1) t c₁c₂/n], s′_F = f (1 − (n − 1) t c₁/n) und die Hauptebenen h₁ = −f (n − 1) t c₂/n (vom ersten Scheitel), h₂ = −f (n − 1) t c₁/n (vom zweiten Scheitel).

**Pupillen.** Die Eintrittspupille ist die Blende selbst (Fläche 0, z = 0, Radius a), weil davor der kollimierte Raum liegt. Die Austrittspupille ist das paraxiale Bild der Blende durch alle Flächen.

**Sonderfälle.** Beide werden nur bei exakt verschwindendem Matrixelement erkannt, damit das Ergebnis deterministisch ist; fast afokale Systeme liefern große endliche Werte.

- Afokal (C = 0, z. B. Planplatte): `afocal = true`, φ = +0, f = +∞; Schnittweiten, Hauptebenen und Defokus sind NaN. Petzval und Austrittspupille bleiben definiert.
- Austrittspupille im Unendlichen (D = 0, Blende im vorderen Brennpunkt, bildseitig telezentrisch): `exit_pupil_at_infinity = true`, Lage und Vergrößerung NaN, Radius +∞.
- κ_P = 0: Petzvalradius +∞.

**Wellenlänge.** Die Host-API rechnet entweder mit einer Spalte der Brechzahltabelle (Index in `spectrum.wavelengths_nm`) oder bei einer beliebigen Wellenlänge über `material_index`; Standard ist `spectrum.reference_nm`, die nicht unter den Spektralwellenlängen vorkommen muss. Für Varianten eines SystemBatch (M3-T3) rechnet `batch_paraxial` dieselben Größen mit den Brechzahlen bei λ_ref aus `paraxial_index[v][s]` und `paraxial_ghost_index[v][g]` ([architektur.md](architektur.md#topologie-und-felder-m2-t1)); die paraxialen Nebenbedingungen stehen in [optimierung.md, Abschnitt 2.2](optimierung.md#22-paraxiale-nebenbedingungen-ab-m3-t3).

**Geisterplatten** (M4-T5) gehen als dünne Elemente der Brechkraft φ_G (Abschnitt 6) an ihrer Lage im Luftspalt ein: Der Trace läuft über die Tabellenflächen (Dateiflächen mit eingefügten Geisterplatten, Krümmung 0, Medium des Spalts); an einer Geisterplatte gilt n′u′ = nu − y (c (n′ − n) + φ_G) mit c = 0, in der Petzvalsumme kommt φ_G/(n_m n_G) = φ_G/n_G hinzu. Kern: `paraxial_trace(c, t, n, count, ray, at_surface, power)` und `petzval_sum(c, n, count, petzval)` mit optionalen Feldern der Zusatzbrechkraft bzw. des Petzvalbeitrags pro Fläche (ohne sie bitgleich zu vorher); Host: `compute_paraxial` mit den zusätzlichen Spannen `power`, `petzval` und die Überladungen für Systembeschreibungen, die Geisterplatten automatisch berücksichtigen (n_G aus `IndexTable::ghost_n` bzw. `material_index`). Die Petzval-Nebenbedingung (M4-T4) nutzt diese Summe.

## 10. Glasmodelle

**Sellmeier** (λ in µm): n² = 1 + Σ_i B_i λ² / (λ² − C_i). Basis für Katalogdaten.

**Buchdahl** (für kontinuierliche Glasvariablen): λ₀ = 0,546074 µm, δλ = λ − λ₀,

$$\omega = \frac{\delta\lambda}{1 + 2{,}5\,\delta\lambda},\qquad n(\omega) = n_0 + \nu_1\,\omega + \nu_2\,\omega^2\;(+\,\nu_3\,\omega^3)$$

Die Koordinate ist nur für 1 + 2,5 δλ > 0 definiert, also λ > 0,146074 µm; außerhalb meldet der Host einen Fehler. Implementierung beider Formeln: `include/mikroptim/glass/dispersion.hpp`.

Koeffizienten werden per kleinsten Quadraten an Sellmeier über 400–700 nm angepasst. Optimiert werden (n₀, ν₁, ν₂) innerhalb der konvexen Hülle der Katalog-Gläser (normierte Koordinaten); ν₃ dient der Bewertung.

**Buchdahl-Fit** (`include/mikroptim/glass/buchdahl_fit.hpp`, `mko::fit_buchdahl`):

- *Termzahl:* gezählt werden die Dispersionskoeffizienten ohne n₀. „3 Terme“: (n₀, ν₁, ν₂, ν₃); „2 Terme“: (n₀, ν₁, ν₂) mit ν₃ = 0. Die 2-Term-Koeffizienten sind die Glasvariablen der Optimierung, der 3-Term-Fit dient der Bewertung.
- *Stützstellen:* standardmäßig N = 301 Tschebyschow-Knoten erster Art in λ, λ_k = λ_m − h cos(π (k + ½)/N) mit Mitte λ_m und halber Breite h des Bereichs. Ungewichtete kleinste Quadrate auf diesen Knoten kommen dem gleichmäßig besten (Minimax-)Fit nahe; das passt zum Fehlermaß der Abnahme (Maximum). Alternativ äquidistant in λ mit Randpunkten. Bereich, N und Knotenart sind Parameter (`BuchdahlFitOptions`).
- *Lösung:* Das Modell ist linear in den Koeffizienten. Entwurfsmatrix A_ij = ω_i^j (Vandermonde in ω, j = 0 … Termzahl), rechte Seite n(λ_i); min ‖A x − b‖₂ per Householder-QR in FP64, keine Normalgleichungen (die würden die Konditionszahl quadrieren; cond(A) ≈ 1,2·10² für 2 und 1,4·10³ für 3 Terme über 400–700 nm).
- *Fehlermaß:* max |n_Buchdahl − n_Sellmeier| und RMS auf einem eigenen äquidistanten Prüfgitter mit Randpunkten (Standard 3001 Punkte = 0,1 nm), ausgewertet mit derselben Formel wie die Brechzahltabelle.
- *Fehler:* Fitbereich außerhalb von `range_nm` des Glases, Termzahl ≠ 2, 3, zu wenige Stützstellen, Bereich unterhalb von 146,074 nm → `GlassError`.

Ergebnisse für `data/glass/basis.json` (Standardeinstellungen; „äquidistant“ = 301 äquidistante Knoten zum Vergleich):

| Glas | n₀ | ν₁ | ν₂ | ν₃ | max. Fehler 3 Terme | max. Fehler 2 Terme | äquidistant 3 / 2 Terme |
|---|---|---|---|---|---|---|---|
| N-BK7 | 1,5187250 | −5,1460·10⁻² | 5,2807·10⁻³ | −5,7541·10⁻⁴ | 5,5·10⁻⁶ | 6,1·10⁻⁶ | 9,0·10⁻⁶ / 9,9·10⁻⁶ |
| CAF2 | 1,4349406 | −2,9243·10⁻² | 5,1323·10⁻³ | 7,1493·10⁻³ | 1,1·10⁻⁶ | 1,1·10⁻⁵ | 1,9·10⁻⁶ / 2,0·10⁻⁵ |
| F_SILICA | 1,4600812 | −4,3132·10⁻² | 1,3161·10⁻³ | −3,6838·10⁻³ | 5,7·10⁻⁶ | 9,4·10⁻⁶ | 9,3·10⁻⁶ / 1,5·10⁻⁵ |

(Koeffizienten des 3-Term-Fits. Der Test `BuchdahlFit.AcceptanceBasisCatalog` gibt diese Tabelle aus.)

## 11. Weiches Versagen

Nur in den Optimierungsmodi (`MERIT`, `RESIDUALS`, `ADJOINT`):

| Ereignis | Strafresiduum | Fortsetzung |
|---|---|---|
| Fläche verfehlt (D = B² − cH < 0) | √w_fail · (−D) | mit D := 0 |
| Totalreflexion (k < 0) | √w_fail · (−k) | mit k := 0 |
| Geisterplatte: Radikand < 0 | √w_fail · (−Radikand) | mit Radikand := 0 |
| d_z ≤ 0 (Strahl läuft rückwärts) | hartes Versagen, festes Strafresiduum | Strahl maskiert |

Im Modus `EVAL` ist jedes Versagen hart. Die Statusmaske pro Strahl wird immer mitgeführt.

## 12. Genauigkeit

- Ziel: absoluter Fehler der OPD ≤ 1·10⁻⁶ mm (1 nm) nach Akkumulation über alle Flächen. FP32 (ε ≈ 6·10⁻⁸ bei OPL ≈ 100 mm) erfüllt das nicht; float ist deshalb nur in der FP32-Vorfilterung erlaubt (unten, M7-T2), deren Werte nie in Merit, Jacobi-Matrix oder Optimierung einfließen.
- FP64: typisch 10⁻¹², df64: typisch 10⁻⁹ (Ziel); gemessen für df64 einige 10⁻¹¹ mm, höchstens 9·10⁻¹⁰ mm gegen FP64 (M7-T1, unten).
- FMA-Kontraktion ist erlaubt; CPU/GPU-Unterschiede dadurch werden durch `kCpuGpuFp64` abgedeckt.
- Toleranzen der Tests: `tests/tolerances.hpp`.

### Zahlentyp df64 (M7-T1)

Implementiert in `include/mikroptim/math/df64.hpp` (header-only, `MKO_HD`, Typ `mko::df64`), Einsatz in den Kerneln über `TraceSettings::precision = Precision::Df64` ([architektur.md, „Zahlentypen der Kernel“](architektur.md#zahlentypen-der-kernel-m7-t1)). FP32 bleibt außerhalb von df64 im OPD-Pfad verboten (AGENTS.md §6.1); df64 ist der in ADR-0003 vorgesehene zweite Zahlentyp.

**Darstellung.** x = hi + lo aus zwei float mit |lo| ≤ ulp(hi)/2 (normiert), 8 Byte wie double, trivial kopierbar und trivial default-konstruierbar (Shared Memory). Mit u = 2⁻²⁴ (Einheitsrundung von float) ist die relative Genauigkeit etwa u² = 2⁻⁴⁸ ≈ 3,6·10⁻¹⁵ (FP64: 2⁻⁵³ ≈ 1,1·10⁻¹⁶), also rund 48 statt 53 Bit; benachbarte double sind in df64 im Allgemeinen nicht unterscheidbar.

**Grundoperationen** (Joldes, Muller, Popescu 2017; Wurzel nach Lefèvre, Louvet, Muller, Picot, Rideau 2023; alle Ergebnisse normiert):

| Operation | Verfahren | Schranke (relativ) | gemessen (2·10⁵ Zufallspaare, 16 Dekaden) |
|---|---|---|---|
| a + b, a − b | AccurateDWPlusDW (Alg. 6), auch bei Auslöschung | 3u² + 13u³ | 1,97 u² (Auslöschung 0,48 u²) |
| a · b | DWTimesDW3 (Alg. 12, mit FMA) | 4u² | 3,10 u² |
| a · f, f float | DWTimesFP3 (Alg. 9, mit FMA) | 2u² | 1,50 u² |
| a / b | DWDivDW2 (Alg. 17) | 15u² + 56u³ | 6,28 u² |
| √a | SQRTDWtoDW (ρ₁ = a_hi − s_h² exakt per FMA) | 25/8 u² | 2,68 u² |

Referenz ist FP64 aus den exakt nach double gewandelten Operanden (Test `Df64.ArithmeticAccuracy`, `Df64.SqrtAccuracyAndEdgeCases`; Toleranzen `kDf64AddRelative` … `kDf64SqrtRelative` = Schranke + 2·2⁻⁵³ für die Rundungen der Referenz und der Rückwandlung).

**Fehlerfreie Transformationen und FMA-Kontraktion.** two_sum (6 Operationen), fast_two_sum (3, Vorbedingung |a| ≥ |b|) und two_prod (p = a·b, e = fma(a, b, −p)) sind nur exakt, wenn jede Operation einzeln gerundet wird. Gefährlich ist nicht nur ein Produkt innerhalb der Transformation, sondern auch ein Operand, der selbst ein Produkt ist: nvcc zieht `fmul` und `fadd` auch dann zu einem FMA zusammen, wenn das Produkt noch anderweitig gebraucht wird, und die Summe passte dann nicht mehr zum gerundeten Operanden. Deshalb rechnet df64 im Device-Code ausschließlich über die Intrinsics `__fadd_rn`, `__fsub_rn`, `__fmul_rn`, `__fmaf_rn`, `__fdiv_rn`, `__fsqrt_rn`, die laut CUDA-Dokumentation nie zusammengezogen werden; die Kompilieroptionen bleiben unverändert (keine fast-math-Optionen, `--fmad` wie bisher). Auf dem Host steht jede Operation in einer eigenen Anweisung; GCC zieht im ISO-Modus (`-std=c++20`, `CMAKE_CXX_EXTENSIONS OFF`) nicht zusammen (`-ffp-contract=off`), MSVC mit `/fp:precise` erzeugt ohne `/arch:AVX2` keine FMA-Befehle, `fmaf` ist korrekt gerundet. Geprüft wird das doppelt: `Df64.ErrorFreeTransformations` (2·10⁵ Paare, Summe und Produkt exakt) und `CudaDf64.DeviceMatchesHostBitwise` (10⁵ Eingaben, 12 Operationen einschließlich des zusammengesetzten Ausdrucks a·b + a und der Umwandlungen: Gerät und Host **bitgleich**).

**Umwandlungen.** Aus double: hi = RN(x), lo = RN(x − hi) (x − hi ist in double exakt), relativer Fehler ≤ 2⁻⁴⁹ (gemessen 2⁻⁴⁹); aus Ganzzahlen exakt für |i| < 2⁴⁸; nach double: hi + lo (eine Rundung, exakt für jedes df64 aus einem double). Im Device-Code kostet die Umwandlung aus double eine FP64-Subtraktion: Die Kernel wandeln die Flächentabelle einmal pro Block, pro Strahl nur Pupillenradius und Feldwinkel (Punktquelle: Quellpunkt, NA, Brechzahl) und am Ende OPD und Residuum nach double.

**sin und cos.** `msincos`: Reduktion r = x − q·π/2, q = round(x·2/π), mit π/2 als Summe dreier float (Fehler < 2⁻⁷⁴) und exakten Produkten q·P₀, q·P₁ (two_prod), danach Taylor-Polynome in r² für |r| ≤ π/4 (sin bis r¹⁵, cos bis r¹⁶, Abbruchfehler < 7·10⁻¹⁷) mit Koeffizienten 1/n! als df64-Paaren aus exakten Brüchen. Gemessen ≤ 7,3·10⁻¹⁵ absolut für |x| ≤ 100 (Test `Df64.SinCosAccuracy`, `kDf64Trig` = 10⁻¹⁴); sin 0 = 0 und cos 0 = 1 exakt (der Achsfächer bleibt in der Meridionalebene); |x| ≥ 2²⁰, ±∞ und NaN ergeben NaN. Gebraucht werden sin/cos nur für Startstrahlen (Azimut φ der Stützstelle, Feldwinkel θ). Alternativen: sin/cos in FP64 auf dem Gerät kosten auf der RTX 3090 pro Aufruf etwa 30–40 FP64-Befehle, also bei 1/64 der FP32-Rate so viel wie mehrere tausend FP32-Befehle – mehr als der ganze Trace in df64; eine Tabelle von sin φ_j, cos φ_j im Shared Memory würde die gemeinsame Funktion `collimated_ray` ändern. Die df64-Reihe kostet etwa 20 df64-Multiplikationen und -Additionen pro Aufruf, so viel wie ungefähr eine Fläche.

**Vergleiche** lexikographisch (hi, dann lo), für normierte Werte exakt. NaN verhält sich wie bei double (jeder Vergleich außer ≠ falsch), damit die Prüfungen des Tracers (`!(denom > 0)`, `!(d_z > 0)`, `D < 0`, `k < 0`) nicht endliche Werte wie in FP64 als Versagen melden.

**Wertebereich und Grenzen.**

- Exponentenbereich von float: Ergebnisse über FLT_MAX ≈ 3,4·10³⁸ werden **NaN** (nicht ±∞, weil die fehlerfreien Transformationen dann ∞ − ∞ bilden); die Umwandlung aus double ergibt dort ±∞. Volle relative Genauigkeit für |x| ≥ 2⁻¹⁰⁰ ≈ 7,9·10⁻³¹, darunter wird lo subnormal (absoluter Fehler dann ≈ 10⁻⁴⁵).
- Division durch 0 ergibt NaN (double: ±∞); der Tracer dividiert nur durch geprüfte Größen (d_z > 0, Nenner > 0, Brechzahlen, r_g² > 0, W_ok > 0).
- Zwischenwerte im Tracer (Test `Df64.TracerIntermediateValuesStayInFloatRange`: alle Strahlen der CPU-Testszenarien – gestörte Varianten von Einzellinse, Mehrlinser, Planplatte, Geisterplatte, Punktquelle, 64 Flächen – mit einem Zahlentyp, der jedes Zwischenergebnis festhält): gültige Strahlen |x| ∈ [4,3·10⁻¹⁹, 1,4·10⁷], also auch Produkte und Quadrate (z. B. c·(x² + y²), B² − cH, b² − g) weit innerhalb von float; das Maximum stammt aus dem Legendre-Profil einer Geisterplatte weit außerhalb des Normierungsradius. Kolbenabzug und Residuen reichen bis 1·10⁻³¹ (Quadrate von OPD im Bereich des Rundungsrauschens, ohne Bedeutung). Versagte Strahlen erreichen 6·10¹⁷.
- Geisterplatten: P_K(ξ) wächst wie (2ξ)^K; für K = 32 läuft das Profil bei |ξ| ≳ 8 (r ≳ 2 r_g) über, df64 meldet dann Versagen (NaN), wo FP64 noch rechnet. Der Normierungsradius soll den Strahlbereich ohnehin abdecken (Abschnitt 6).
- **Versagensgrenze:** Der Status entsteht aus Vergleichen von Größen mit relativem Fehler ≈ u² statt ≈ 2⁻⁵³. Strahlen sehr nahe an einer Grenze (Totalreflexion, Fläche verfehlt, d_z = 0) können in df64 anders entschieden werden als in FP64. Test `CudaDf64.StatusAtFailureBoundary`: äußerster Strahl des Achsfeldes von `einzellinse_bk7.json`, Grenzradius a* = 49,5597 mm (Totalreflexion an der zweiten Fläche), Pupillenradius a*·(1 + δ): gleicher Status für |δ| ≥ 10⁻¹⁴, abweichend für δ = 0 und δ = +10⁻¹⁵ (df64 noch Ok, FP64 TIR). In den gestörten Varianten der Testsuite (ca. 4·10⁷ Strahlen einschließlich des abgeschalteten Stresstests) kippt kein Status. Ein gekippter Status ist bei hartem Versagen ein Sprung in Kolben und Merit; das weiche Versagen (M4-T2, Abschnitt 11) macht ihn stetig.
- Determinismus (ADR-0007): Die df64-Operationen sind deterministisch und die Reduktionen nutzen dieselbe Baumstruktur (Shuffle über zwei float-Shuffles); zehn Läufe und Läufe in mehreren Tranchen sind bitgleich (`CudaDf64.RepeatedRunsAreBitIdentical`).

**Abnahme df64 gegen FP64** (Test `CudaDf64.MatchesFp64ForPerturbedVariants` über die TraceEngine, RTX 3090; RESIDUALS mit OPD und Max-Reduktionen, MERIT mit Max-Reduktionen; Vergleich mit FP64 auf der GPU und mit der CPU-Referenz `residuals_cpu`; Status in allen Fällen gleich):

| Szenario | Strahlen (versagt) | max \|Δr\| [mm] | max \|ΔOPD\| [mm] | max \|Δh\| [mm] | max \|Δ cos I\| | max \|Δ√F\| [mm] |
|---|---|---|---|---|---|---|
| `einzellinse_bk7.json`, 1.000 Varianten | 510.000 (35.877) | 1,4·10⁻¹² | 9,9·10⁻¹² | 3,0·10⁻¹³ | 1,2·10⁻¹⁴ | 1,1·10⁻¹² |
| `testsystem_mehrlinser.json`, 1.000 | 680.000 (57.371) | 2,1·10⁻¹² | 1,6·10⁻¹¹ | 2,2·10⁻¹¹ | 2,7·10⁻¹² | 1,9·10⁻¹² |
| dasselbe, Quadratur 6 × 5 | 264.000 (21.203) | 2,2·10⁻¹² | 9,9·10⁻¹² | 1,6·10⁻¹¹ | 1,6·10⁻¹² | 9,7·10⁻¹³ |
| `planplatte_bk7.json`, 500 | 45.000 (1.583) | 3,4·10⁻¹³ | 2,5·10⁻¹² | 1,9·10⁻¹³ | 4,1·10⁻¹⁵ | 2,6·10⁻¹³ |
| `einzellinse_geisterplatte.json`, 500, Koeffizienten und Gläser gestört | 255.000 (36.231) | 8,7·10⁻¹¹ | 9,0·10⁻¹⁰ (ein Strahl, \|OPD\| = 21 mm) | 1,4·10⁻¹⁰ | 1,5·10⁻¹¹ | 7,3·10⁻¹¹ |
| Punktquelle neben der Achse (volle Pupille), 200 | 48.000 (242) | 6,6·10⁻¹³ | 5,9·10⁻¹² | 2,5·10⁻¹³ | 8,8·10⁻¹⁵ | 6,4·10⁻¹³ |
| Stapel mit 64 Flächen, 200 | 36.000 (0) | 8,7·10⁻¹³ | 2,5·10⁻¹² | 1,2·10⁻¹³ | 1,3·10⁻¹³ | 3,4·10⁻¹³ |

Toleranz pro Strahl max(`kDf64VsFp64`, `kDf64VsFp64Relative`·|x|) = max(10⁻⁹ mm, 10⁻¹⁰·|x|), wie bei `kCpuGpuFp64Relative` erst ab |x| > 10 mm relativ; alle Strahlen liegen auch unter 10⁻⁹ mm absolut. Merit über die 2-Norm: |√F_df64 − √F_FP64| ≤ √(Σ_k Schranke_k²); relativ |ΔF|/F ≤ 1,0·10⁻¹⁰. Der abgeschaltete Stresstest `CudaDf64.DISABLED_StressManySeeds` (je zweimal 10.000 Varianten von vier Systemen, 3,9·10⁷ Strahlen) liefert höchstens 3,6·10⁻¹⁰ mm (Geisterplatte, |OPD| = 22 mm), für Systeme ohne Geisterplatte ≤ 1,9·10⁻¹⁰ mm bei |OPD| = 6 mm und sonst ≤ 7·10⁻¹¹ mm, ohne Statuswechsel. Die größten Abweichungen gehören zu schlecht konditionierten Strahlen extremer Varianten (Pupillenradius ×3 … ×5, |OPD| von mm bis einige 10 mm, Geisterplatten weit außerhalb von r_g): Dort verstärken sich die Rundungsfehler relativ um ca. 10⁴ wie in FP64 (vgl. `kCpuGpuFp64Relative`). Auf der CPU liefert derselbe Pfad (`batch_ray_opl` über eine df64-Flächentabelle) ≤ 1,7·10⁻¹¹ mm für OPL und OPD (Test `Df64.CpuTraceMatchesFp64`, läuft auch in der CI); MERIT gegen Σ RESIDUALS² in df64 ≤ 2,3·10⁻¹⁵ relativ (`kMeritVsResidualSumDf64`).

### FP32-Vorfilterung (M7-T2)

**Zweck.** Im Multistart sind viele Varianten hoffnungslos: massives Strahlversagen oder eine Merit, die um Größenordnungen über der besten liegt. Die Vorfilterung bewertet alle Varianten grob in float (auf der RTX 3090 rechnet FP32 64-mal so schnell wie FP64) und sortiert nur die aus, die die Kriterien **auch in FP64 sicher** erfüllen; alle übrigen werden danach in FP64 oder df64 bewertet. Das ist der einzige Pfad mit R = float (AGENTS.md §6.1, ADR-0003): API `mko::cuda::TraceEngine::prefilter_variants` (GPU) bzw. `mko::prefilter_cpu` (CPU-Referenz), Kernel `prefilter_kernel` ([architektur.md, „FP32-Vorfilterung“](architektur.md#fp32-vorfilterung-m7-t2)). Die Werte (`PrefilterResult`) sind Schranken für die Auswahl und dürfen nie in Merit, Residuen, Jacobi-Matrix oder Optimierung einfließen; so steht es in jedem beteiligten Header und Quelltext.

**Pro Strahl** (`prefilter_ray` in `include/mikroptim/merit/prefilter_ray.hpp`, header-only, gemeinsam für CPU und Kernel): dieselben Funktionen wie in FP64 (`batch_source_ray`, `trace_ray`, Prüfung d_z > 0 nach der letzten Brechung, `reference_sphere_step`) in float, Flächentabelle und Quadratur beim Laden nach float gewandelt. Zusätzlich liefert jeder Strahl seinen **Grenzabstand**: den Abstand seiner Prüfgrößen zur Versagensgrenze, einheitlich als Kosinus.

| Prüfung des Tracers | Prüfgröße | Grenzabstand (gültig) | Betrag des Versagens |
|---|---|---|---|
| Richtung vor jedem Transfer und nach der letzten Brechung | d_z > 0 | d_z | −d_z |
| Sphäre getroffen | D = B² − cH ≥ 0 und B + √D > 0 | cos I = **d**·**N** (= √D) | max(√(−D), −B) |
| Brechung | k = 1 − μ²(1 − cos² I) ≥ 0 | cos I′ = **d**′·**N** (= √k) | √(−k) |
| Geisterplatte | 1 − \|**d**′_⊥\|² ≥ 0 | d′_z | √(\|**d**′_⊥\|² − 1) |
| Referenzsphäre | b² − g ≥ 0 und σ(b + σ√(b² − g)) > 0 | √(b² − g)/R_ref und σ(b + σ√(b² − g))/R_ref | √(g − b²)/R_ref bzw. −σ(…)/R_ref |

Für gültige Strahlen wertet ein Beobachter von `trace_ray` die Größen im Schnittpunkt aus (`PrefilterMarginObserver`): Mit **p** = **p**_t + s **d** (Tangentialebene, |**d**| = 1) ist **d**·**N** = d_z − c(**p**·**d**) = B − cs und s = H/(B + √D), also **d**·**N** = (D + B√D)/(B + √D) = √D; aus **d**′ = μ**d** + (√k − μ cos I)**N** folgt **d**′·**N** = √k. Für versagte Strahlen rechnet `prefilter_surface_failure` die verletzte Prüfgröße aus dem Zustand nach, in dem `trace_ray` zurückkehrt (dieselben Formeln wie `surface_ops.hpp` und `ghost_plate.hpp`). Beim Verfehlen genügt das Vorzeichen von B: Wegen cH = c²(x² + y²) ≥ 0 ist √D ≤ |B|, also trifft auch FP64 nur bei B > 0. NaN ergibt den Grenzabstand 0.

**Klassen.** Gültig oder versagt mit Grenzabstand > η heißt *sicher*, sonst *unsicher*. Annahme (A): Sicher entschiedene Strahlen werden in FP64 gleich entschieden. Die Rundungsfehler der Prüfgrößen selbst sind klein (FP32 gegen FP64 über dieselbe Funktion, Grenzabstände bis 1: ≤ 2·10⁻⁵, Test `Prefilter.RayClassesAndErrorModel`), aber streifende Strahlen verstärken die Fehler aller folgenden Flächen: ein Richtungsfehler δd_z erzeugt über die Dicke t einen Querfehler ≈ t·δd_z/d_z², und eine Geisterplatte weit außerhalb ihres Normierungsradius macht daraus einen großen Richtungsfehler. Deshalb ist η groß gewählt (`PrefilterMargins::boundary` = 0,15, Winkel über 81,4°): Im Stresstest (`CudaPrefilter.DISABLED_StressManySeeds`, 3·10⁵ Multistart-Varianten, 1,7·10⁸ Strahlen) war bei η = 0,05 genau ein Strahl falsch sicher (Geisterplatte lenkt auf d_z = 0,05 ab, 2,5 mm Luft versetzen ihn um 50 mm, die zweite Geisterplatte bei r = 2,6 r_g lenkt in FP32 und FP64 verschieden ab; FP64 totalreflektiert an der Linse, FP32 nicht, cos I′ = 0,05), bei η = 0,1, 0,15 und 0,2 keiner. Physikalisch brauchbare Strahlen liegen weit über der Schwelle (Bildraum bei NA 0,95: d_z = 0,31). Grenzabstand an der Totalreflexion als Orakel (`Prefilter.MarginAtTirBoundary`, äußerster Strahl der Einzellinse um den Grenzradius a* = 49,5597 mm): fällt wie √|δ| (1,5·10⁻³ bei |δ| = 10⁻⁶, 0,15 bei 10⁻²), FP32 und FP64 gleich bis 5·10⁻⁵.

**OPL relativ.** Die Start-OPL wird um `prefilter_opl_offset` vermindert, die axiale optische Länge Σ_{s ≤ S−2} n_s t_s des Tupels (jede Konstante lässt die OPD unverändert; diese liegt nahe an allen OPL, weil die Referenzsphäre durch den Scheitel der letzten Fläche geht). Kolben und Quadratsumme werden damit aus Zahlen von einigen mm statt ~100 mm gebildet; der Kolben wird nur über die sicher gültigen Strahlen genommen.

**Fehlermodell der OPD** (sicher gültige Strahlen in Tupeln ohne unsichere Strahlen):

$$|\mathrm{OPD}_{32} - \mathrm{OPD}_{64}| \le \kappa\,u\,\Lambda_v + m\,|\mathrm{OPD}|,\qquad u = 2^{-24},\quad \Lambda_v = \max_k\big(|\mathrm{opl\_offset}| + |\mathrm{OPL}_k - \mathrm{opl\_offset}|\big)$$

Der erste Term ist die Rundung der akkumulierten OPL (einige Rundungen der Größe u·Λ pro Fläche), der zweite die relative Rundung großer OPD extremer Varianten (mm bis einige 10 mm, dort schlecht konditionierte Strahlen). Gemessen (je 600 Varianten, Test `Prefilter.RayClassesAndErrorModel`): nötiges κ bei m = 0 20 / 19 / 162 (Einzellinse / Mehrlinser / Geisterplatte), bei m = 0,01 4,7 / 3,9 / 2,8. Standard κ = 32, m = 0,01 (`PrefilterMargins`); bei Λ = 25 mm ist κuΛ = 4,8·10⁻⁵ mm ≈ 0,09 λ.

**Schranken der Merit.** F32_v = Σ r_k² über die Menge C der sicher gültigen Strahlen. (1) Weglassen von Strahlen verkleinert F: Für A ⊆ B ist F_A = min_x Σ_A W_k (OPL_k − x)² ≤ Σ_A W_k (OPL_k − x_B)² ≤ F_B (Gewichte nicht renormiert, wie in r_k). Nach (A) ist C in der Menge der FP64-gültigen Strahlen enthalten, also F_FP64 ≥ F_FP64(C), mit Gleichheit ohne unsichere Strahlen. (2) Dreiecksungleichung in der gewichteten 2-Norm: |√F32(C) − √F64(C)| ≤ ‖s√W·δOPD‖₂ ≤ Δ_v + m√F64(C) mit Δ_v = κuΛ_v√(Σ_(f,l) s_(f,l)²) (s = `residual_scale` = √(ŵ_f ŵ_λ)/λ_ref seit M4-T1; Σ_k W_k ≤ 1 pro Tupel). Daraus

$$L_v = \Big(\frac{\max(0,\ \sqrt{F_{32}} - \Delta_v)}{1 + m}\Big)^2 + E_v \;\le\; F_{\mathrm{FP64}} \;\le\; U_v = \Big(\frac{\sqrt{F_{32}} + \Delta_v}{1 - m}\Big)^2 + E_v,$$

U_v nur ohne unsichere Strahlen (sonst +∞); E_v ist der optionale Anteil weiterer Residuen (FP64 vom Aufrufer, exakt addiert). NaN in den Rohdaten (Überlauf) ergibt L_v = E_v, U_v = +∞.

**Entscheidungsregel** (`prefilter_decide`; „Wahrheit“ `prefilter_decide_exact` = dieselbe Regel auf FP64-Werten ohne Margen):

| Kriterium | FP32 sortiert aus, wenn | FP64-Wahrheit | warum konservativ |
|---|---|---|---|
| (a) Strahlversagen | sicher versagt > f_max·R_v | versagt > f_max·R_v | sicher versagt ≤ versagt_FP64 ≤ sicher versagt + unsicher (A) |
| (b) Merit absolut | L_v > F_max | F_v > F_max | L_v ≤ F_v |
| (c) Merit relativ | L_v > q·min_u U_u | F_v > q·F_best | Referenzmenge: Varianten ohne unsichere Strahlen, ohne (a), mit mindestens einem gültigen Strahl; dort entscheidet FP64 (a) gleich und F_u ≤ U_u, also F_best ≤ min_u U_u |

F_best der Wahrheit ist die kleinste Merit der Varianten ohne (a) mit mindestens einem gültigen Strahl. Ergebnis: Bitmaske der Gründe pro Variante (`reject`, 0 = behalten), L_v, U_v, Rohdaten (F32, Λ, sicher versagte, unsichere Strahlen), Referenzvariante. Ein Grund, den FP64 nicht hat, kommt nicht vor (getestet).

**Abnahme** (keine falsch aussortierten Varianten; RTX 3090, Test `CudaPrefilter.AcceptanceNoFalseRejections`): je 10.000 Multistart-Varianten von `einzellinse_bk7.json`, `testsystem_mehrlinser.json` und `einzellinse_geisterplatte.json` (Krümmungen ±50 %, Dicken ±30 %, Pupille ×0,5 … 2, jede siebte Variante extrem; 7,1 / 8,5 / 12,5 % versagte Strahlen, 793 / 985 / 1.083 Varianten mit über 50 % Versagen, Merit über 6–8 Größenordnungen), 13 Kriterien pro System (Versagen > 1, 10, 50 % und am 0,9-Quantil der FP64-Werte, Merit über den 0,25/0,5/0,9-Quantilen, über 2-, 10-, 100-, 10⁴-mal die beste, kombiniert, kombiniert mit weiteren Residuen):

| System | FP64-Ablehnungen (Summe über die Kriterien) | falsch aussortiert | zusätzlich behalten | unsichere Strahlen |
|---|---:|---:|---:|---:|
| Einzellinse | 65.266 | 0 | 480 | 12.660 von 5,1·10⁶ |
| Mehrlinser | 64.944 | 0 | 727 | 16.995 von 6,8·10⁶ |
| Geisterplatte | 67.781 | 0 | 478 | 7.401 von 5,1·10⁶ |

Zusätzlich behalten (erlaubt) wird vor allem nahe den Merit-Schwellen (Faktor (1 ± m)² und Δ_v) und beim Kriterium 10⁴-mal die beste (U der besten Variante enthält Δ_v ≈ 10⁻⁴ mm gegen √F_best ≈ 5·10⁻⁴ mm, gemessen mit den Residuen in mm vor M4-T1; beide skalieren mit s). Stresstest mit 10 weiteren Seeds (3·10⁵ Varianten, 390 Entscheidungsläufe): 0 falsch aussortiert, 16.834 von 1.978.009 FP64-Ablehnungen zusätzlich behalten (0,85 %), 0,22 % unsichere Strahlen; L_v ≤ F_FP64 ≤ U_v und die Klammer der versagten Strahlen für alle Varianten. CPU in der CI (`Prefilter.CpuAcceptanceNoFalseRejections`, je 1.000 Varianten gegen `merit_cpu`): 0 falsch aussortiert, 137 von 18.139 zusätzlich behalten. GPU gegen CPU-Referenz (`CudaPrefilter.MatchesCpuReference`): Klassen in allen 900 Varianten gleich, |F_GPU/F_CPU − 1| ≤ 9·10⁻⁵ (FMA-Kontraktion und Summationsreihenfolge). Determinismus: zehn Läufe und vier Tranchen bitgleich (`CudaPrefilter.RepeatedRunsAndTranchesAreBitIdentical`, Baumreduktionen wie MERIT, ADR-0007).

**Grenzen.** Annahme (A) und die Konstanten κ, m sind empirisch belegt, nicht bewiesen; wer η, κ oder m verkleinert, verlässt den getesteten Bereich. float läuft bei |x| > 3,4·10³⁸ über (Geisterplatten weit außerhalb von r_g, wie df64); das ergibt NaN, also unsicher. Hartes Versagen wie in M2: Mit dem weichen Versagen (M4-T2, Strafresiduen) muss die Vorfilterung die Strafen in L_v aufnehmen. Kriterium (c) wird nur angewandt, wenn mindestens eine Variante ohne unsichere Strahlen in die Referenzmenge fällt.

## 13. Literatur

- G. H. Spencer, M. V. R. K. Murty: General ray-tracing procedure, JOSA 52 (1962).
- W. T. Welford: Aberrations of Optical Systems.
- R. Kingslake, R. B. Johnson: Lens Design Fundamentals.
- G. W. Forbes: Optical system assessment for design: numerical ray tracing in the Gaussian pupil, JOSA A 5 (1988).
- P. N. Robb, R. I. Mercado: Calculation of refractive indices using Buchdahl's chromatic coordinate, Applied Optics 22 (1983).
- T. J. Dekker: A floating-point technique for extending the available precision, Numerische Mathematik 18 (1971).
- M. Joldes, J.-M. Muller, V. Popescu: Tight and rigorous error bounds for basic building blocks of double-word arithmetic, ACM Transactions on Mathematical Software 44 (2017).
- V. Lefèvre, N. Louvet, J.-M. Muller, J. Picot, L. Rideau: Accurate calculation of Euclidean norms using double-word arithmetic, ACM Transactions on Mathematical Software 49 (2023).
