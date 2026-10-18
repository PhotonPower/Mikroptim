# Numerik: Konventionen, Formeln, Genauigkeit

Dieses Dokument ist verbindlich für alle Implementierungen im Mathe-Kern, im CPU-Tracer und im CUDA-Backend. Änderungen an Konventionen nur über eine ADR.

## 1. Einheiten

- Längen in **mm**, Krümmungen in **1/mm**.
- Wellenlängen in Dateien und APIs in **nm**; in Dispersionsformeln (Sellmeier, Buchdahl) in **µm**.
- Winkel in Dateien in **Grad**, intern im Bogenmaß.
- OPD intern in mm; Ausgabe zusätzlich in Wellen bei der Referenzwellenlänge.

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

## 7. Referenzsphäre und OPD

Nach der letzten Fläche (lokale Koordinaten, Medium n_B, Luft): idealer Bildpunkt **M**_f = (0, h_f, w) mit Arbeitsabstand w und h_f = f_soll · tan θ_f. Referenzsphäre um **M**_f mit Radius R_ref = |**M**_f| (durch den Scheitel der letzten Fläche). Mit **q** = **p** − **M**_f, b = −**q**·**d**, g = |**q**|² − R_ref²:

$$s_{\mathrm{ref}} = \frac{g}{\,b + \sqrt{b^2 - g}\,},\qquad \mathrm{OPL} \mathrel{+}= n_B\, s_{\mathrm{ref}}$$

s_ref darf negativ sein. Residuum pro Strahl ist die OPD nach Kolbenabzug:

$$\mathrm{OPD}_i = \mathrm{OPL}_i - \sum_j W_j\,\mathrm{OPL}_j$$

Kippung wird **nicht** entfernt: Sie entspricht Verzeichnung bzw. Brennweitenfehler und soll bestraft werden. Für die Endbewertung (M8) wird die Referenzsphäre durch die paraxiale Austrittspupille gelegt; der Unterschied ist von zweiter Ordnung in der Queraberration.

## 8. Pupillenquadratur

Mittelwert über die Kreispupille als Gauß-Quadratur (Forbes):

$$\langle f\rangle = \frac{1}{\pi}\int_0^{2\pi}\!\!\int_0^1 f\,\rho\,d\rho\,d\varphi \approx \sum_{i=1}^{N}\sum_{j=1}^{M} W_i\,V_j\,f(\rho_i,\varphi_j)$$

- Radial: Gauß-Legendre auf u = ρ² ∈ [0, 1], Knoten u_i, ρ_i = √u_i, Gewichte W_i (Summe 1). Exakt für Polynome in u bis Grad 2N − 1.
- Azimutal, halbe Pupille (Symmetrie x → −x bei Feld entlang y): φ_j = (j − ½)·π/M, V_j = 1/M. Exakt für cos(mφ) mit m < 2M.
- Auf der Achse: M = 1 (nur ein radialer Fächer).
- Standard: N = 10, M = 8. Die Gewichte gehen als √(W_i V_j) in die Residuen ein.

## 9. Paraxialoptik

y-nu-Trace mit Brechkraft φ_i = c_i (n′_i − n_i):

$$n'_i u'_i = n_i u_i - y_i\,\varphi_i,\qquad y_{i+1} = y_i + t_i\,u'_i$$

- Brennweite bei kollimierter Eingabe (u₀ = 0): f = −y₀ / u′_letzte.
- Petzvalsumme und Krümmung der Petzvalschale:

$$P = \sum_i c_i\,\frac{n'_i - n_i}{n_i\,n'_i},\qquad \kappa_P = -\,n'_{\mathrm{letzte}}\,P$$

Geisterplatten tragen φ_G/n_G zu P bei (Abschnitt 6).

## 10. Glasmodelle

**Sellmeier** (λ in µm): n² = 1 + Σ_i B_i λ² / (λ² − C_i). Basis für Katalogdaten.

**Buchdahl** (für kontinuierliche Glasvariablen): λ₀ = 0,546074 µm, δλ = λ − λ₀,

$$\omega = \frac{\delta\lambda}{1 + 2{,}5\,\delta\lambda},\qquad n(\omega) = n_0 + \nu_1\,\omega + \nu_2\,\omega^2\;(+\,\nu_3\,\omega^3)$$

Die Koordinate ist nur für 1 + 2,5 δλ > 0 definiert, also λ > 0,146074 µm; außerhalb meldet der Host einen Fehler. Implementierung beider Formeln: `include/mikroptim/glass/dispersion.hpp`.

Koeffizienten werden per kleinsten Quadraten an Sellmeier über 400–700 nm angepasst. Optimiert werden (n₀, ν₁, ν₂) innerhalb der konvexen Hülle der Katalog-Gläser (normierte Koordinaten); ν₃ dient der Bewertung.

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

- Ziel: absoluter Fehler der OPD ≤ 1·10⁻⁶ mm (1 nm) nach Akkumulation über alle Flächen. FP32 (ε ≈ 6·10⁻⁸ bei OPL ≈ 100 mm) erfüllt das nicht.
- FP64: typisch 10⁻¹², df64: typisch 10⁻⁹ (ab M7).
- FMA-Kontraktion ist erlaubt; CPU/GPU-Unterschiede dadurch werden durch `kCpuGpuFp64` abgedeckt.
- Toleranzen der Tests: `tests/tolerances.hpp`.

## 13. Literatur

- G. H. Spencer, M. V. R. K. Murty: General ray-tracing procedure, JOSA 52 (1962).
- W. T. Welford: Aberrations of Optical Systems.
- R. Kingslake, R. B. Johnson: Lens Design Fundamentals.
- G. W. Forbes: Optical system assessment for design: numerical ray tracing in the Gaussian pupil, JOSA A 5 (1988).
- P. N. Robb, R. I. Mercado: Calculation of refractive indices using Buchdahl's chromatic coordinate, Applied Optics 22 (1983).
