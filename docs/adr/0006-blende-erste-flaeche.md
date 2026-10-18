# ADR-0006: Blende als erste Fläche, kollimierte Quelle

- **Status:** angenommen
- **Datum:** 2026-10-18

## Kontext
Getract wird vom Unendlichraum zum Objekt. Liegt die Blende im Inneren, braucht jeder Strahl Ray-Aiming (iterativ), was teuer und fehleranfällig ist.

## Entscheidung
In Version 1 ist Fläche 0 die Aperturblende (Ebene in Luft). Pupillenkoordinaten sind direkt die Startkoordinaten des kollimierten Bündels; OPL₀ = **p₀**·**d₀**. Zusätzlich gibt es eine Punktquelle für Tests.

## Konsequenzen
Die Blendenlage ist auf die Rückseite des Objektivs festgelegt (realistisch für Unendlich-Objektive). Eine innenliegende Blende mit Newton-Ray-Aiming kann später als eigene Aufgabe folgen (neue ADR).
