# ADR-0008: Eigenes JSON-Systemformat, Raytatouille per Adapter

- **Status:** angenommen
- **Datum:** 2026-10-18

## Kontext
Der Maintainer entwickelt mit Raytatouille ein allgemeines Optikdesign-Programm (Format `*.rtt.json`). Mikroptim ist auf sequenzielle, koaxiale, sphärische Systeme und Synthese spezialisiert.

## Entscheidung
Eigenes, schlankes Format `mikroptim.system/1` (docs/dateiformat.md). Ein Adapter zwischen `*.rtt.json` und `mikroptim.system/1` ist für M9 geplant.

## Konsequenzen
Mikroptim bleibt unabhängig baubar. Feldnamen sollen sich, wo sinnvoll, an Raytatouille anlehnen, um den Adapter einfach zu halten.
