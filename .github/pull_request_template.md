## Aufgabe

Meilenstein und Aufgaben-ID (z. B. `M1-T3`), Link zum Abschnitt in `docs/plan/meilensteine.md`:

## Was wurde gemacht

## Abnahmekriterien

- [ ] Alle Abnahmekriterien der Aufgabe sind erfüllt (bitte einzeln mit Ergebnis auflisten)

## Checkliste

- [ ] Tests ergänzt; `ctest` grün (CPU)
- [ ] Bei CUDA-Änderungen: lokal auf RTX 3090 **und** Tesla P100 getestet (Ausgabe von `mikroptim --devices` beilegen)
- [ ] Keine FP32-Arithmetik im OPD-Pfad, keine fast-math-Optionen
- [ ] Doku aktualisiert (`docs/…`), `CHANGELOG.md` ergänzt
- [ ] Abweichung von Konventionen oder Plan? Dann ADR in `docs/adr/` angelegt
- [ ] Keine Geheimnisse, Tokens oder lizenzpflichtigen Glaskataloge im Commit
