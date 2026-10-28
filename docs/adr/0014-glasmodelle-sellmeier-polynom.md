# ADR-0014: Glasmodelle Sellmeier mit mehr Termen und Polynom

- **Status:** angenommen
- **Datum:** 2026-10-28

## Kontext

`mikroptim.glass/1` kannte nur das Modell `sellmeier` mit genau drei Termen. Viele Hersteller geben ihre Gläser in anderen Formeln an: Schott-Formel und erweiterte Polynomformeln (Hoya, Sumita, Hikari, große Teile von Ohara und CDGM), Sellmeier mit vier oder fünf Termen oder mit Konstante. Der Glasimport (M9-T3) musste diese Gläser mit drei Sellmeier-Termen annähern. Im Plausibilitätslauf über die Herstellergläser der Datenbank refractiveindex.info (1.636 Gläser) wurden dabei rund 100 verworfen und über 290 nur mit eingeschränktem Bereich übernommen. Betroffen waren vor allem hochbrechende Gläser und dichte Flinte, die für Frontlinsen und den Farbquerfehler der Hintergruppe gebraucht werden, sowie die UV- und IR-Ränder (i- bis t-Linie, Multiphoton).

Die Brechzahlen der Katalog-Gläser rechnet Mikroptim ausschließlich auf dem Host (`Glass::index`: Brechzahltabelle, Buchdahl-Fit, Glas-Hülle, Einrasten). Auf dem Gerät gibt es nur die Prüfung des gemeinsamen Kerns (`dispersion_eval`).

## Entscheidung

1. Modell `sellmeier` erweitert: n² = 1 + A + Σ_{i=1}^{k} B_i λ²/(λ² − C_i), 1 ≤ k ≤ 8 (`kMaxSellmeierTerms`), Konstante `A` optional (Standard 0). Dateien mit drei Termen ohne `A` bleiben gültig und rechnen bitgleich wie bisher.
2. Neues Modell `polynomial`: n² = Σ_{j=1}^{m} a_j λ^{p_j}, 1 ≤ m ≤ 12 (`kMaxPolynomialTerms`), Feld `terms` mit Paaren [a_j, p_j]. Ganzzahlige Exponenten werden durch wiederholte Multiplikation gebildet, andere mit `pow`.
3. Der Leser prüft wie bisher Pole im Bereich (Sellmeier) und zusätzlich eine gültige Brechzahl (endlich, n² > 0) auf 201 Punkten des Bereichs, für beide Modelle.
4. Weitere Formeln (Herzberger, Conrady, Handbook of Optics) werden nicht als Modell aufgenommen; der Glasimport fittet sie wie bisher mit drei Sellmeier-Termen. Ein neues Modell braucht eine neue ADR.
5. Die Formeln stehen header-only und `MKO_HD` in `include/mikroptim/glass/dispersion.hpp` (`sellmeier_index_n`, `polynomial_index`, `int_power`), damit eine spätere Auswertung auf dem Gerät denselben Code nutzt (ADR-0005).

## Konsequenzen

- Der Glasimport übernimmt alle Sellmeier- und Polynomformeln der Kataloge exakt: im Plausibilitätslauf 1.633 von 1.636 Gläsern ohne Fit und ohne Einschränkung des Bereichs (verworfen nur zwei tabellierte technische Gläser und eines mit inkonsistenter Quelle).
- `Glass::B` und `Glass::C` sind `std::vector<double>` statt `std::array<double, 3>`. Code, der genau drei Terme erwartet (nur der Test des CUDA-Dispersionskerns), prüft die Termzahl.
- Ältere Versionen von Mikroptim lesen Kataloge mit den neuen Modellen nicht. Der Glasimport kann mit `--legacy` weiterhin nur 3-Term-Sellmeier ausgeben.
- Buchdahl-Fit, Glas-Hülle und Einrasten bleiben unverändert, sie rufen nur `Glass::index` auf.

## Verworfene Alternativen

- **Nur Fit im Konverter:** verliert genau die Gläser, auf die es in Mikroskopobjektiven ankommt, oder schränkt ihren Bereich ein.
- **Je ein Modell pro AGF-Formel:** viele Modelle mit fast gleicher Struktur; die AGF-Formeln 1, 10, 12, 13 und die Polynomformel von refractiveindex.info sind Spezialfälle von `polynomial`, die AGF-Formeln 2, 4, 6, 9, 11 Spezialfälle des erweiterten `sellmeier`.
- **Tabellen mit Interpolation als Modell:** keine glatte Dispersion, schlecht für Ableitungen und den Buchdahl-Fit.
