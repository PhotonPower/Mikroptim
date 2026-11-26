# M6-T14b: Stand bei Unterbrechung (2026-11-26, ≈ 01:45)

Auf Anweisung des Koordinators gesichert. PR #121 ist offen, **nicht gemergt**. Code, Tests, Doku und Bericht ([m6-t14b.md](m6-t14b.md)) sind fertig; lokal grün (ctest release-cuda 1156/1156, GCC 962/962, Python 348/31 übersprungen), CI des Code-Stands grün (Lauf der letzten Commits mit Bericht beim Sichern noch nicht abgewartet).

## Läufe (keiner läuft mehr)

| Lauf | Stand | Letzter bewerteter Punkt |
|---|---|---|
| F3ZS (Fortsetzung ab M6-T14 c2_feld, `--zyklus apertur,feld`) | **regulär beendet** („fertig“) nach c3_feld | c2_apertur: NA 0,6825, Feld 52,1 %, 20 El., Strehl ≥ 0,62; c3_feld: NA 0,6825, 52,1 %, 21 El., Strehl 0,64 |
| Z1S (Fortsetzung ab M6-T14 c0_feld mit F25-Fix) | **abgebrochen** am GPU-Budget in c2_apertur, Runde r2 (r1 erreichte NA 0,6425 im Lauf, 18 El.; nicht bewertet) | c1_feld: NA 0,620, Feld 56,8 %, 18 El., Strehl 0,56 (c0_apertur: NA 0,620, Strehl 0,02) |

Ergebnisse unter `docs/ergebnisse/m6-t14b/` (mit `--archiv` verkleinert, Programmausgabe `lauf.log`), Tabelle `m6t14b_tabelle.json`.

## Fortsetzen

1. Build (`tools/bauen.ps1 -Preset release-cuda`), Programm nach `C:\mko\t14b\bin\` kopieren.
2. Z1S ab c1_feld fortsetzen (das archivierte Verzeichnis enthält keine synth-Ergebnisse mehr, daher neuer Zyklus ab dem letzten bewerteten System):

```
OPT="--farb-stetig --farb-ziel voll --farb-modul doublet --farb-chromatisch 0.5 --feldregel --einfuegen-verbessern 4 --rueckbau-punkt 0.85"
python tools/m6t6_zielspezifikation.py --gpu --exe C:/mko/t14b/bin/mikroptim.exe --zyklus apertur,feld --zyklus-runden 4 --start docs/ergebnisse/m6-t14b/Z1S/c1_feld_z2r_s2.json --elementbudget 18 --farb-budget 6 $OPT --out docs/ergebnisse/m6-t14b/Z1S2
python tools/m6t14_tabelle.py Z1S2_c0a=docs/ergebnisse/m6-t14b/Z1S2/c0_apertur_z2r_s2.json --out C:/mko/t14b/tab --gpu --exe C:/mko/t14b/bin/mikroptim.exe
```

   (Die M6-T14-Schalter sind jetzt Standard; je Aperturrunde ≈ 1 h GPU.)
3. Danach Bericht Abschnitt 4 und PR-Tabelle ergänzen, CI von PR #121 abwarten (`gh pr checks 121`), Fragen an den Maintainer (Bericht Abschnitt 7) offen.
