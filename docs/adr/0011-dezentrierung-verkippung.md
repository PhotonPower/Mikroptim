# ADR-0011: Dezentrierung und Verkippung einzelner Flächen

- **Status:** vorgeschlagen (Entscheidung des Maintainers, AGENTS.md §9: Erweiterung der Koordinatenkonventionen)
- **Datum:** 2026-10-20

## Kontext

M8-T3 verlangt allgemeine Flächentransformationen (Dezentrierung, Verkippung) als Grundlage der Monte-Carlo-Tolerierung (M8-T4). Bisher ist das System koaxial (numerik.md §2): Jede Fläche hat lokale Koordinaten mit Scheitel im Ursprung, die Achsen aller Flächen fallen zusammen, und die Pupillenquadratur nutzt die Spiegelsymmetrie x → −x (halbe Pupille). Für Toleranzanalysen genügen kleine Starrkörperbewegungen einzelner Flächen bzw. Elemente; eine allgemeine nicht-sequenzielle Geometrie ist nicht nötig.

## Entscheidung (Vorschlag)

1. **Transformation pro Fläche** relativ zur nominellen (koaxialen) Lage ihres Scheitels: Dezentrierung (Δx, Δy) in mm und Verkippung um die lokale x- und y-Achse (α, β) in Grad (Datei) bzw. Bogenmaß (intern). Reihenfolge beim Eintritt: erst verschieben, dann drehen (R = R_y(β)·R_x(α), aktiv, rechtshändig, positiver Winkel gegen den Uhrzeigersinn mit Blick entgegen der Drehachse). Eine Drehung um z ist bei sphärischen Flächen wirkungslos und entfällt.
2. **Kein Koordinatenbruch für Folgeflächen:** Die Transformation betrifft nur die Fläche selbst („Dezentrieren und zurück“). Der Strahl wird vor dem Schnitt in das gekippte/verschobene lokale System transformiert und nach der Brechung zurück in das nominelle System der Fläche; Dicken bleiben Abstände der nominellen Scheitel entlang +z. Elementweise Bewegungen (eine Linse als Ganzes) entstehen, indem beide Flächen dieselbe Transformation um einen gemeinsamen Drehpunkt erhalten (Drehpunkt-Angabe pro Element, Standard: Scheitel der ersten Fläche).
3. **Pupille:** Sobald eine Transformation ≠ 0 vorliegt, entfällt die Symmetrie; die Quadratur verwendet `PupilMode::Full` (M1-T4) für alle Felder, auch auf der Achse.
4. **Dateiformat:** optionale Felder `decenter: [Δx, Δy]` und `tilt_deg: [α, β]` an Flächen (Standard 0); kompatible Erweiterung von `mikroptim.system/1` (keine neue Hauptversion). Für Toleranzsätze (M8-T4) ein eigener Abschnitt; das Schema legt M8-T4 fest.
5. **Referenzsphäre und Bildpunkt** bleiben im nominellen System der letzten Fläche (numerik.md §7); Verzeichnung und Kippung durch Toleranzen werden damit sichtbar und nicht wegdefiniert.
6. **Numerik:** Die Transformation ist Teil des gemeinsamen Mathe-Kerns (ADR-0005), FP64/df64, ohne Einfluss auf koaxiale Systeme (Transformation 0 ⇒ bitgleiche Ergebnisse, Test-Pflicht).

## Konsequenzen

- numerik.md §2 bekommt einen Unterabschnitt „Dezentrierung und Verkippung“; die Aussage „koaxial bis M8“ entfällt.
- Batch-Layout: pro Variante und Fläche 4 zusätzliche Werte (Δx, Δy, α, β); Kernel lesen sie nur, wenn die Topologie ein Flag „nicht koaxial“ trägt, damit koaxiale Läufe keine Kosten haben.
- Paraxialoptik bleibt koaxial (Toleranzen werden real getract, nicht paraxial bewertet).
- Adjungierte (M5) brauchen Ableitungen nach den Transformationsparametern erst, wenn Dezentrierungen optimiert werden sollen (nicht Teil von M8).

## Verworfene Alternativen

- **Koordinatenbrüche wie in kommerziellen Programmen** (globale Transformation, die für alle Folgeflächen gilt): flexibler, aber fehleranfällig in der Handhabung von Toleranzen (Rückkehr-Brüche nötig) und unnötig für den Zweck.
- **Globale Matrizen pro Fläche:** allgemeiner, aber mehr Parameter, kein direkter Bezug zu Toleranzangaben (Δ, Winkel).
