# M6-T14: Stand bei Unterbrechung (2026-11-25, 01:05)

Unterbrochen auf Anweisung des Koordinators (Rechner geht in den Energiesparmodus). PR #120 ist offen, **nicht gemergt**.

## Fertig

- Code, Tests, ADR-0019, Doku, CHANGELOG, Plan, Bericht `docs/ergebnisse/m6-t14-verzeichnung-querfarbe.md` (Stand mit F3ZS bis c2).
- Lokal grün nach Rebase auf main 7e7a91e (M6-T13b): `ctest --preset release-cuda` 1154/1154 (RTX 3090), GCC-Prüfbau 959/959 (vor dem Rebase), Python 317 bestanden / 55 übersprungen.
- CI zu PR #120 lief beim Abbruch noch (nicht abgewartet).

## Läufe (Ergebnisse unter `docs/ergebnisse/m6-t14/`, mit `--archiv` verkleinert, Logs in `logs/`)

| Lauf | Stand | Letzter bewerteter Punkt |
|---|---|---|
| F3ZS (ab F3Z c1, `--bildpunkt schwerpunkt --verzeichnung 0.01 --farb-querfarbe-strahlen`) | **abgebrochen** in c3_apertur nach Runde r1 (NA 0,660, 20 El., `max_elements`), c3 nicht bewertet | c2_feld: NA 0,650, Feld 52,1 %, 22 El., Strehl ≥ 0,71 |
| Z1S (ab Z1f) | beendet mit Fehler zu Beginn von c1_apertur: `mikroptim insert: Fehler: return_to_ghost_plates: keine Lage passt` (`Z1S/c1_apertur_start_bericht.txt`) | c0_feld: NA 0,50, Feld 56,8 %, 18 El., Strehl ≥ 0,81 |
| Z1I (Kontrolle, idealer Bildpunkt, nur Feldstufen) | regulär „fertig“ | c1_feld: NA 0,50, Feld 55,2 %, 18 El., Strehl ≥ 0,60 |
| Z1SF (wie Z1S + `--woerterbuch-glas SYN-28`) | beendet mit demselben Fehler zu Beginn von c1_feld | c0_feld: NA 0,50, Feld 66,2 %, 18 El., Strehl ≥ 0,63 |

## Fortsetzen

1. Build prüfen (`tools/bauen.ps1 -Preset release-cuda`), Programm nach `C:\mko\t14\bin2\` kopieren.
2. F3ZS fortsetzen (das Skript überspringt fertige Stufen): Befehl aus Bericht Abschnitt 9 mit `--exe C:/mko/t14/bin2/mikroptim.exe`; c3_apertur beginnt neu (Fortsetzungsdatei `c3_apertur_r*_restart.json`, falls vorhanden).
3. Danach c3 mit `python tools/m6t14_tabelle.py … F3ZS_c3=docs/ergebnisse/m6-t14/F3ZS/c3_apertur_z2r_s2.json --out C:/mko/t14/tab --gpu` bewerten und in Bericht/Tabelle nachtragen.
4. Z1S/Z1SF: Fortsetzung braucht eine Lösung für F25 (`return_to_ghost_plates` bei 18 El.) – außerhalb dieser Aufgabe.
5. CI von PR #120 abwarten (`gh pr checks 120 --watch`), Fragen an den Maintainer (PR, Bericht Abschnitt 8) offen; erst danach mergen.
