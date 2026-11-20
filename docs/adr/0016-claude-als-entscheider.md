# ADR-0016: Claude-API als zusätzlicher Entscheider (optionales Extra)

- **Status:** angenommen (Entscheidung des Maintainers)
- **Datum:** 2026-11-19
- **Ändert:** ADR-0012, Punkt 4 („Kein Cloud-Dienst“) und verworfene Alternative „Cloud-Modelle“

## Kontext

ADR-0012 sieht als Sprachmodell nur ein lokales Modell über Ollama vor. Für die Bewertung (M10-T7) soll zusätzlich ein starkes Modell über die Claude-API verglichen werden können: Zeigt schon ein großes Modell keinen Vorteil gegenüber der Regel (V3), lohnt der Ausbau der lokalen Anbindung nicht; zeigt es einen, ist das die obere Messlatte für lokale Modelle.

Gegen Cloud-Modelle sprach in ADR-0012 der Schutz von Patentdaten und lizenzierten Glaskatalogen (AGENTS.md §8).

## Entscheidung

1. **`ClaudeDecider`** in `mikroptim/design/claude.py` als weitere Umsetzung der Schnittstelle `Decider`, wählbar mit `python -m mikroptim.design run --decider claude [--model …]`. Standardmodell `claude-opus-5-5`; ebenso `claude-sonnet-5-5` und `claude-haiku-5-5` (Modell-IDs ohne Datumsuffix).
2. **Abhängigkeit:** das offizielle Python-SDK `anthropic` nur als optionales Extra `pip install mikroptim[claude]`, Import erst im Entscheider. Kern, Ollama-Pfad und Tests brauchen es nicht; die Tests nutzen eine Attrappe des Clients, ohne Netz und ohne Schlüssel. AGENTS.md §9 nennt das Extra in der Liste der erlaubten Abhängigkeiten.
3. **Zugangsdaten** ausschließlich aus der Umgebung (`ANTHROPIC_API_KEY` oder ein Profil von `ant auth login`, das das SDK selbst findet). Nie in Code, Konfiguration, Protokoll, Kostendatei, Commits oder PR-Texten.
4. **Datenschutz:** Der Zustand und die Ergebnisse der Lese-Werkzeuge verlassen den Rechner. Erlaubt ist das nur für Systeme und Kataloge ohne Schutzbedarf (frei erfundene Testsysteme, Testkataloge, eigene Entwürfe). Der Entscheider verweigert Kandidaten, deren Glaskataloge oder Startsystem unter einem Verzeichnis `private` liegen (`PrivacyError`); Benchmark B4 (Patentsysteme) läuft nur lokal mit Ollama.
5. **Kosten:** Jede Anfrage wird aus `response.usage` (Input, Output, Cache-Lesen, Cache-Schreiben) bewertet, im Protokoll pro Entscheidung festgehalten und in einer Kostendatei (`mikroptim.design.cost/1`) laufend summiert. Vor jeder Anfrage wird geprüft, dass eine Obergrenze (Standard 20 US-$) auch im ungünstigsten Fall nicht überschritten wird; sonst bricht der Lauf ab (`BudgetExceeded`, kein stiller Rückfall auf die Regel).
6. **Reproduzierbarkeit** wie bei Ollama über das Protokoll: Modell, Aufwand, Prüfsummen von Prompt und Werkzeugliste, Request-IDs, Usage, Kosten, Denktext (Zusammenfassung) und Rohantworten. `replay` braucht keine API.

## Konsequenzen

- Zwei Anbindungen von Sprachmodellen mit unterschiedlicher API (Ollama: schema-beschränkte Antwort; Claude: Antwort als Werkzeug `submit_decision`, weil erzwungenes `tool_choice` bei Opus 5.5 und Sonnet 5.5 abgelehnt wird). Beide prüfen die Antwort gegen dasselbe Aktionsschema.
- **Nachtrag M10-T7c (2026-11-20):** Die ursprünglich vorgesehenen strict-Werkzeuge lehnt die API ab (400: höchstens 24 optionale Parameter je Anfrage, die Werkzeuge haben 185: `submit_decision` 133, die Lese-Werkzeuge zusammen 52). Standard ist deshalb `strict` aus; die Prüfung der Antwort gegen das Aktionsschema auf Seite von Mikroptim (Parameter, Aufzählungen, danach die Prüfungen der Ausführung) bleibt und ist damit die einzige Schranke. `strict` bleibt als Option. Die Entscheidung selbst ändert sich nicht.
- Läufe kosten Geld; Obergrenze und Kostendatei sind Pflicht.
- Die CI braucht keinen Schlüssel; echte Läufe nur lokal.

## Verworfene Alternativen

- **Rohes HTTP statt SDK:** keine neue Abhängigkeit, aber Fehlerklassen, Wiederholungen und Zugangsdaten-Profile müssten nachgebaut werden.
- **Claude auch für private Daten:** widerspricht AGENTS.md §8.
