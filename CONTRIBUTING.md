# Mitarbeit

Diese Regeln gelten für Menschen und Programmieragenten. Agenten lesen zusätzlich [AGENTS.md](AGENTS.md).

## Ablauf

1. Aufgabe aus [docs/plan/meilensteine.md](docs/plan/meilensteine.md) wählen, deren Abhängigkeiten erledigt sind.
2. Branch anlegen: `m<Meilenstein>/t<Nr>-<kurzname>`, z. B. `m1/t3-gauss-quadratur`.
3. Implementieren, Tests schreiben, Doku und `CHANGELOG.md` aktualisieren.
4. Pull Request mit der Vorlage öffnen. Ein PR pro Aufgabe.
5. Merge erst nach grüner CI und Review durch den Maintainer.

## Commits

Kurze Betreffzeile im Imperativ mit Aufgaben-ID, z. B. `M1-T3: Gauß-Quadratur für die Pupille`.

## Sprache

- Dokumentation, Commit-Nachrichten, PR-Texte: Deutsch.
- Bezeichner im Code: Englisch. Kommentare: Deutsch oder Englisch, innerhalb einer Datei einheitlich.

## Entscheidungen

Grundsätzliche Entscheidungen (Toolchain, Konventionen, Dateiformat, Numerik) werden als ADR in [docs/adr/](docs/adr/) festgehalten. Eine bestehende ADR wird nicht still umgangen, sondern durch eine neue ersetzt.
