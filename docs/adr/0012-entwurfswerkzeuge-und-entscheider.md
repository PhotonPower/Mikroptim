# ADR-0012: Entwurfswerkzeuge mit Mensch oder Sprachmodell als Entscheider

- **Status:** angenommen
- **Datum:** 2026-10-28

## Kontext

Die automatische Synthese (M6) trifft ihre strukturellen Entscheidungen über feste Heuristiken: Diagnose der Geisterplatten, Operator A oder B, Stufenfolge der Homotopie. Der Lauf auf der Zielspezifikation (M6-T6) ist genau an diesen Stellen gescheitert (Befunde F2–F5, F9, F13): Der Controller fügt nur Menisken ein, findet keine Farbkorrektur und wählt die Stufenfolge nicht passend. Ein Optikdesigner entscheidet dort anhand von Aberrationsbilanzen der Gruppen, Linsenmodulen und Glasregeln (Y. Zhang, *Systematic Design of Microscope Objectives*, 2019).

Ziel ist, dass ein lokales Sprachmodell diese Entscheidungen treffen und den Designer ersetzen *kann*. Der Designer soll aber dieselben Werkzeuge behalten und jederzeit selbst entscheiden können.

Bisher sind alle Entscheidungen der Synthese reine Funktionen der Rechenergebnisse und damit reproduzierbar. Ein Sprachmodell und ein Mensch sind es nicht.

## Entscheidung

1. **Entwurfswerkzeuge als öffentliche API.** Mikroptim bekommt das Python-Paket `mikroptim.design` über den Bindings (M9-T1): Diagnosen (Lese-Werkzeuge), Aktionen mit geprüften Vorbedingungen, Kandidatenbaum und Protokoll. Dieselben Werkzeuge gibt es als Unterbefehle `mikroptim design …`. Der C++/CUDA-Kern bekommt nur neue Diagnosen und Operatoren, die auch ohne Entscheider nutzbar sind.
2. **Austauschbarer Entscheider.** Eine Schnittstelle `Decider` mit den Umsetzungen Mensch (`HumanDecider`), Sprachmodell (`OllamaDecider`), feste Regel (`RuleDecider`), Wiedereinspieler (`ReplayDecider`) und Kombination Vorschlag/Bestätigung (`ConfirmDecider`).
3. **Gleichberechtigung.** Jede Diagnose und jede Aktion steht Mensch und Sprachmodell gleichermaßen zur Verfügung. Es gibt keine Werkzeuge nur für einen von beiden. Die GUI (GUI-Mikroptim) nutzt dafür ausschließlich `mikroptim.design` (dort ADR-0002).
4. **Sprachmodell über Ollama, lokal.** Das Modell ist pro Lauf wählbar, als Instruktions- oder Thinking-Modell. Lese-Werkzeuge als Tool-Calls, die Entscheidung als strukturierte Ausgabe nach `data/design/schemas/design_action.schema.json`. Kein Cloud-Dienst: Patentdaten und lizenzierte Glaskataloge verlassen den Rechner nicht (AGENTS.md §8).
5. **Entscheider schlagen vor, Mikroptim führt aus.** Kein Entscheider schreibt Systemdateien an den Werkzeugen vorbei; jede Aktion wird vor der Ausführung geprüft. Die Rangfolge der Kandidaten bleibt die deterministische Rangfolge der Strahlsuche, Maßstab bleiben Merit und Zielprüfung.
6. **Reproduzierbarkeit über das Protokoll.** Jede Entscheidung wird mit Zustand, Entscheider (bei Ollama Modellname, Digest, Optionen), Rohantwort, Denktext und Ergebnis protokolliert (`mikroptim.design.log/1`). Der `ReplayDecider` wiederholt einen Lauf ohne Modell und ohne Menschen. Die Numerik bleibt nach ADR-0007 reproduzierbar.
7. **Formate.** `mikroptim.design.state/1`, `mikroptim.design.action/1` und `mikroptim.design.log/1` sind versionierte Formate im Sinn von ADR-0008.
8. **GPU-Rollen.** Ollama läuft auf der RTX 3090, der FP64-Trace auf der Tesla P100. Muss der Trace auch auf die 3090, wird sein Speicher über `EngineOptions::memory_limit_bytes` begrenzt.
9. **Wissensbasis in eigenen Worten.** Linsenmodule, Brechkraftverteilungen und Glasregeln liegen als JSON unter `data/design/knowledge/`, formuliert in eigenen Worten mit Quellenangabe. Keine Textauszüge, keine Linsendaten realer Objektive.
10. **Erfolgskriterium.** Das Sprachmodell wird gegen eine regelbasierte Strategie mit denselben Werkzeugen (`RuleDecider`) und, soweit möglich, gegen einen Menschen mit denselben Werkzeugen bewertet. Zeigt es keinen Vorteil gegenüber der Regel, wird die Modellanbindung nicht weiter ausgebaut; die Werkzeuge bleiben.

## Konsequenzen

- Die Entwurfswerkzeuge sind auch ohne Sprachmodell ein Gewinn: Der Mensch bekommt Aberrationsbilanzen, Klassen, Empfindlichkeiten und strukturelle Operatoren mit Protokoll.
- Läufe mit Modell oder Mensch sind nicht durch erneutes Ausführen reproduzierbar, sondern über das Protokoll. Temperatur 0 und fester Seed machen Modellantworten meist, aber nicht garantiert gleich.
- Neue Fehlerarten (ungültige oder unsinnige Vorschläge) werden durch die Prüfung der Aktionen und einen Rückfall-Entscheider abgefangen.
- Abhängigkeit von Ollama und von bestimmten Modellversionen. Beides wird im Protokoll festgehalten und ist austauschbar, weil nur die Schnittstelle `Decider` davon abhängt.
- System-Prompt, Wissensbasis und Schemas brauchen Pflege; Änderungen an den Schemas erhöhen die Formatversion.
- Die GUI braucht eine neue Seite für den Entwurf (Folgeaufgabe in GUI-Mikroptim).

## Verworfene Alternativen

- **Modellanbindung im C++-Kern** (Rückruf im Homotopie-Controller): bindet HTTP und Prompt-Logik an den Kern und macht die Werkzeuge für Menschen schwer zugänglich.
- **Werkzeuge nur für das Modell** (Mensch nur als Beobachter): widerspricht dem Ziel, dass der Designer jederzeit übernehmen kann, und macht den Vergleich Mensch–Modell unmöglich.
- **Cloud-Modelle:** Datenschutz und Lizenzen der Glaskataloge und Patentdaten.
- **Modell bewertet und ordnet Kandidaten selbst:** Ergebnis wäre nicht mehr an Merit und Zielprüfung überprüfbar.
- **llama.cpp-Server oder vLLM direkt:** möglich, aber Ollama vereinfacht Modellwechsel und Verwaltung; ein anderer Server ließe sich später als weiterer `Decider` anbinden.
