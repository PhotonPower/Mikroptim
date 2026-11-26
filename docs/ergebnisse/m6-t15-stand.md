# M6-T15: Stand bei Unterbrechung (2026-11-26, ≈ 13:20)

Auf Anweisung des Koordinators gesichert (Maintainer braucht die GPU). PR #122 ist offen, **nicht gemergt**. Code, Tests und Doku sind fertig und lokal grün; der Bericht `docs/ergebnisse/m6-t15.md` fehlt noch (Messung unvollständig, Bewertung mit `tools/m6t14_tabelle.py` nicht gelaufen).

## Prüfungen (Code-Stand 1. Commit, unverändert)

- `ctest --preset release-cuda` (RTX 3090): **1161/1161** (vorher 1156; neu `HomotopyControl.PruneRelativeToStateBeforeRemoval`, `…PruneFinishLockKeepsRemovedElements`, `…ApertureRmsLimit`, `OpticalHomotopy.RmsMainOnlyFullyWeightedTuples`, `CliSynth.M6T15PruneRelativeAndApertureLimit`; `BeamSearch.InvalidOptionsThrow` erweitert).
- GCC-Prüfbau (MSYS2, Warnungen als Fehler): **967/967**.
- Python (eigenes CUDA-Wheel, venv `C:\mko\t15\venv`): **347 bestanden, 34 übersprungen**; neu `test_m6t15_standards_and_switches`, `test_main_path_m6t15_standard_and_switch`.
- Bitgleichheit ohne Optionen: `synth` mit der Binärdatei vor M6-T15 (Stand M6-T14b) und der neuen, ohne bzw. mit `--prune --element-budget 2` – Protokoll (ohne Laufzeiten) und Ergebnisdatei gleich (`C:\mko\t15\vergleich.py`).
- CI des ersten Commits: alle Jobs grün.

## Läufe (beide gestoppt, keiner läuft mehr)

Befehle: `C:\mko\t15\run.sh` (wie M6-T14b, `OPT` aus m6-t14b-stand.md, Standards von M6-T15 ohne weitere Schalter, Programm `C:\mko\t15\bin\mikroptim.exe`). Ergebnisse unvollständig unter `docs/ergebnisse/m6-t15/{Z1S,F3ZS}` (nicht archiviert).

| Lauf | Start | Stand beim Stopp |
|---|---|---|
| Z1S | M6-T14b `Z1S/c1_feld_z2r_s2.json` (NA 0,62, 57 % Feld, 18 El.), Budget 18 | c0_apertur: **keine Öffnung** (Aperturgrenze: größter Tupel-RMS am Start 0,135 > 0,1); Rückbau **18 → 14 El.** (2 Kittglieder, Sperre griff: kein Wiedereinfügen), zurück auf NA 0,62 erreicht (14 El.). c1_feld: Feld 0,569 → **0,647** (17 El.), Rückbau entfernt nichts (0/14). c2_apertur Runde r0 lief (abgebrochen). |
| F3ZS | M6-T14 `F3ZS/c2_feld_z2r_s2.json` (NA 0,65, 52 % Feld, 22 El.; derselbe Start wie M6-T14b), Budget 20 | c0_apertur: **keine Öffnung** (größter Tupel-RMS 436–656 nm 0,108–0,120 > 0,1); Rückbau entfernt nichts (F_B,vor = 0,375 F_max, beste Entfernung 1,46 × F_B,vor). c1_feld: Feld 0,522 → **0,569**; Rückbau **21 → 18 El.**, beide Entfernungen nur durch die relative Annahme (F_B 0,00330/0,00318 > 0,5 F_max = 0,00255, ≤ 1,1 × 0,00305), Sperre griff, zurück auf Feld 0,569 erreicht (18 El.). c2_apertur Runde r1 fertig (keine Öffnung), dann gestoppt. |

Zahl der angenommenen Rückbauten in den Rückbau-Läufen des Zyklus: Z1S 2 (c0), F3ZS 2 (c1, beide nur mit relativer Annahme); dazu Rückbauten am Elementbudget in den Stufenläufen (Annahme dort unverändert ρ = 1).

Vorläufige Befunde:

- **Aperturgrenze 0,1 Wellen** sperrt die Öffnung an beiden Startsystemen sofort (Startsysteme liegen mit Geisterplatten schon bei 0,11–0,14 Wellen im schlechtesten voll gewichteten Tupel). Die Regel wirkt wie gewollt (keine Öffnung mit Strehl 0,02 wie M6-T14b Z1S), erreicht aber keine höhere NA. Frage an den Maintainer: Grenze lockern (z. B. 0,12–0,14) oder relativ zum Startwert der Stufe?
- **Rückbau relativ + Sperre**: das Elementbudget wird wieder frei (Z1S 18 → 14 → 17, F3ZS 21 → 18), die Feldstufen öffnen danach (Z1S 57 → 65 %, F3ZS 52 → 57 % – beide über M6-T14b).

## Fortsetzen

1. Bewertung (GPU, wenige Minuten): `python tools/m6t14_tabelle.py Z1S_c0a=docs/ergebnisse/m6-t15/Z1S/c0_apertur_z2r_s2.json Z1S_c1f=docs/ergebnisse/m6-t15/Z1S/c1_feld_z2r_s2.json F3ZS_c0a=docs/ergebnisse/m6-t15/F3ZS/c0_apertur_z2r_s2.json F3ZS_c1f=docs/ergebnisse/m6-t15/F3ZS/c1_feld_z2r_s2.json --out C:/mko/t15/tab --gpu --exe C:/mko/t15/bin/mikroptim.exe --json docs/ergebnisse/m6-t15/m6t15_tabelle.json`
2. Optional Fortsetzung der Zyklen ab `c1_feld_z2r_s2.json` mit `C:\mko\t15\run.sh` (neuer `--out`), ggf. Vergleichslauf mit `--apertur-rms-grenze aus` (nur Rückbau-Regeln).
3. Bericht `docs/ergebnisse/m6-t15.md`, Verzeichnisse mit `--archiv` verkleinern, PR-Beschreibung mit Abnahmekriterien ergänzen.
