# Glossar

**Abgleichlänge (parfocal length):** Abstand von der Anlagefläche des Objektivs bis zur Objektebene, hier 45 mm.

**Buchdahl-Koordinate ω:** chromatische Koordinate ω = δλ/(1 + 2,5·δλ) mit δλ = λ − λ₀ in µm; Brechzahlen werden als Polynom in ω beschrieben.

**df64:** Zahlentyp aus zwei FP32-Werten mit ca. 48 Bit Mantisse; auf GeForce-Karten schneller als natives FP64.

**Einfüge-Operator:** Änderung der Systemtopologie, die die Merit exakt unverändert lässt (Meniskus ohne Brechkraft, Aufspalten einer Linse in ein Kittglied aus gleichem Glas).

**Geisterplatte (ghost plate):** virtuelle dünne Glasplatte mit radialem Dickenprofil an einer Kandidatenposition. Ihre Koeffizienten werden mit einer L1-Strafe belegt; sie werden nur aktiv, wenn das reale System die Aufgabe nicht schafft, und zeigen an, wo und welche Korrektur fehlt.

**Homotopie:** schrittweises Hochfahren eines Parameters (NA, Feld, Spektralbereich, Strafgewicht), sodass die Optimierung einem stetigen Lösungspfad folgt.

**Kernel-Modus:** Betriebsart des Trace-Kernels: `MERIT`, `RESIDUALS`, `ADJOINT`, `EVAL`.

**LM:** Levenberg-Marquardt-Verfahren (gedämpfte kleinste Quadrate).

**OPD / OPL:** optische Wegdifferenz / optische Weglänge.

**Variante:** eine vollständige Systembeschreibung innerhalb eines Batches (z. B. ein gestörtes System für eine Jacobi-Spalte, ein Multistart-Kandidat, eine Toleranzstichprobe).
