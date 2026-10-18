# Glasdaten

- `basis.json`: wenige Gläser mit öffentlich verfügbaren Sellmeier-Koeffizienten für Tests (N-BK7, CaF₂, Quarzglas).
- Vollständige Herstellerkataloge (Schott, Ohara, CDGM, Hikari …) werden **nur** committet, wenn ihre Lizenz das erlaubt. Sonst gehören sie nach `data/glass/private/` (in `.gitignore`) und werden zur Laufzeit geladen.
- Format: siehe [docs/dateiformat.md](../../docs/dateiformat.md#glasdaten).
