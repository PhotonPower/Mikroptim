# Spezifikation des ersten Zielsystems

## Anforderungen

| Größe | Wert |
|---|---|
| Typ | Plan-Apochromat für Auflicht (Epi), trocken, ohne Deckglas |
| Vergrößerung | 50x |
| Numerische Apertur | 0,95 (Luft) |
| Tubussystem | Zeiss ICS, Tubuslinse f_T = 164,5 mm, Tubusfaktor 1x, Eintrittspupille der Tubuslinse 31,6 mm |
| Feldzahl | 25 |
| Abgleichlänge | 45 mm (Anlagefläche bis Objektebene) |
| Spektralbereich | sichtbar; volle Korrektur 436–656 nm, 405 nm und 700 nm mit kleinem Gewicht |
| Flächen | nur sphärisch und plan |
| Korrekturziel | beugungsbegrenzt über das ganze Feld: RMS-Wellenfrontfehler ≤ λ/14 (Strehl ≥ 0,8 nach Maréchal) |

Der Spektralbereich der Tubuslinse (380–700 nm) ist **keine** Korrekturanforderung an das Objektiv.

## Abgeleitete Kenngrößen

| Größe | Formel | Wert |
|---|---|---|
| Brennweite Objektiv | f = f_T / V | 3,29 mm |
| Durchmesser Austrittspupille (Sinusbedingung) | 2·f·NA | 6,25 mm |
| Objektfeld | FZ / V | Ø 0,5 mm (Halbfeld y = 0,25 mm) |
| Feldwinkel im Unendlichraum | atan(FZ/2 / f_T) | ±4,35° |
| Randstrahlwinkel am Objekt | asin(NA) | 71,8° |
| Airy-Radius | 0,61·λ/NA | 0,28 µm (436) / 0,35 µm (546) / 0,42 µm (656) |
| Schärfentiefe | ±λ/(2·NA²) | ±0,24 µm (436) / ±0,30 µm (546) / ±0,36 µm (656) |
| Petzvalradius (ohne Astigmatismus-Ausgleich) | ≥ y²/(2·δ_436) | ≥ ca. 130 mm |
| Vignettierungsfreier Abstand Austrittspupille → Tubuslinse | (31,6/2 − 6,25/2) / tan 4,35° | ≤ ca. 167 mm |

Zum Vergleich: Eine Einzellinse mit f = 3,29 mm hat einen Petzvalradius von etwa n·f ≈ 5 mm. Die Petzvalsumme muss also um den Faktor 25 bis 30 reduziert werden.

## Tracerichtung

Getract wird vom Unendlichraum (Seite der Tubuslinse) zum Objekt: parallele Bündel mit Feldwinkel θ, Durchmesser 6,25 mm, werden in Punkte der Objektebene fokussiert. Der ideale Bildpunkt hat die Höhe h = f·tan θ (verzeichnungsfrei für ein unendlich korrigiertes System). Wegen der Umkehrbarkeit der Lichtwege ist das gleichwertig zur Abbildung in Gebrauchsrichtung.

## Startsysteme und Referenzen

- Referenzobjektiv: Zeiss EC Epiplan-Apochromat 50x/0,95 BD DIC (Modell beim Maintainer)
- Patente als Quellen für Front- und Hintergruppen: DE 103 16 415 (100x/0,95 Luft), US 6,504,653 (100x/1,45 Öl)

Die Linsendaten werden vom Maintainer als JSON in `data/systems/` eingetragen.

## Offene Punkte (vom Maintainer zu entscheiden)

1. **Arbeitsabstand:** Zielwert bzw. Untergrenze.
2. **Gewinde und maximaler Linsendurchmesser:** Bei einer BD-Variante (Hell- und Dunkelfeld) begrenzt der Ringkanal für die Dunkelfeldbeleuchtung den nutzbaren Linsendurchmesser deutlich.
3. **DIC-Tauglichkeit:** spannungsarme Gläser und Fassung; beeinflusst Glasauswahl und Tolerierung.
4. **Lage der Aperturblende:** In Version 1 ist die Blende die erste Fläche in Tracerichtung (ADR-0006).
5. **Spektrale Gewichte:** Kompromiss aus Hellempfindlichkeit V(λ) und flacher Gewichtung, oder Kameraempfindlichkeit × Lichtquelle.
6. **Verzeichnung:** zulässiger Restwert.
