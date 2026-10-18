#!/usr/bin/env python3
"""Erzeugt include/mikroptim/pupil/gauss_legendre_table.hpp.

Gauss-Legendre-Knoten und -Gewichte fuer N = 1 ... 16 auf dem Intervall u in [0, 1]
(Gewichte mit Summe 1), siehe docs/numerik.md, Abschnitt 8.

Nur Python-Standardbibliothek (decimal). Rechnung mit 60 Dezimalstellen:
Newton-Iteration fuer die Nullstellen x von P_N auf [-1, 1] (Startwert nach Tricomi),
Gewichte w = 2 / ((1 - x^2) P_N'(x)^2), danach u = (1 + x) / 2 und W = w / 2.
Ausgegeben werden 21 signifikante Stellen; der Compiler rundet sie auf das naechste
double, d. h. die Tabelle ist (bis auf extrem unwahrscheinliche Doppelrundung)
korrekt gerundet.

Aufruf (aus der Wurzel des Repositorys):
    python tools/gen_gauss_legendre_table.py > include/mikroptim/pupil/gauss_legendre_table.hpp
"""

import math
import sys
from decimal import Decimal, getcontext

MAX_ORDER = 16
getcontext().prec = 60


def legendre(n, x):
    """P_n(x) und P_n'(x) ueber die Dreitermrekursion."""
    p0, p1 = Decimal(1), x
    if n == 0:
        return p0, Decimal(0)
    for k in range(1, n):
        p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
    dp = n * (x * p1 - p0) / (x * x - 1)
    return p1, dp


def gauss_legendre(n):
    """Knoten u (aufsteigend) und Gewichte W auf [0, 1]."""
    nodes = []
    for i in range(1, n + 1):
        # Startwert (Tricomi), absteigend in x; Index wird unten umgedreht.
        x = Decimal(math.cos(math.pi * (i - 0.25) / (n + 0.5)))
        for _ in range(100):
            p, dp = legendre(n, x)
            dx = p / dp
            x -= dx
            if abs(dx) < Decimal(10) ** -58:
                break
        else:
            raise RuntimeError(f"Newton konvergiert nicht (N={n}, i={i})")
        _, dp = legendre(n, x)
        w = 2 / ((1 - x * x) * dp * dp)
        nodes.append(((1 + x) / 2, w / 2))
    nodes.sort()
    # Kontrolle: Summe der Gewichte und exakte Momente int_0^1 u^k du = 1/(k+1).
    for k in range(2 * n):
        q = sum(W * u**k for u, W in nodes)
        if abs(q - Decimal(1) / (k + 1)) > Decimal(10) ** -50:
            raise RuntimeError(f"Moment {k} falsch (N={n})")
    return nodes


def fmt(d):
    return f"{d:.20e}"


def main():
    tables = [gauss_legendre(n) for n in range(1, MAX_ORDER + 1)]
    total = MAX_ORDER * (MAX_ORDER + 1) // 2
    out = []
    out.append("#pragma once")
    out.append("")
    out.append("// ERZEUGT von tools/gen_gauss_legendre_table.py - nicht von Hand bearbeiten.")
    out.append("//")
    out.append("// Gauss-Legendre-Knoten u_i (aufsteigend) und Gewichte W_i auf u in [0, 1]")
    out.append("// mit Summe der Gewichte 1, fuer N = 1 ... 16 (docs/numerik.md, Abschnitt 8).")
    out.append("// Berechnet mit Python decimal (60 Stellen), ausgegeben mit 21 Stellen.")
    out.append("// Die Werte fuer Ordnung N beginnen bei Index gauss_legendre_offset(N).")
    out.append("// Nur fuer Host-Code; der Kernel erhaelt die Werte ueber PupilQuadrature.")
    out.append("")
    out.append("namespace mko::detail {")
    out.append("")
    out.append(f"inline constexpr int kGaussLegendreMaxOrder = {MAX_ORDER};")
    out.append("")
    out.append("// Erster Index der Ordnung n (1 <= n <= kGaussLegendreMaxOrder) in den Tabellen.")
    out.append("constexpr int gauss_legendre_offset(int n) {")
    out.append("    return n * (n - 1) / 2;")
    out.append("}")
    for name, idx, doc in (("kGaussLegendreU", 0, "Knoten u_i"),
                           ("kGaussLegendreW", 1, "Gewichte W_i")):
        out.append("")
        out.append(f"// {doc}")
        out.append(f"inline constexpr double {name}[{total}] = {{")
        for n, nodes in enumerate(tables, start=1):
            out.append(f"    // N = {n}")
            for node in nodes:
                out.append(f"    {fmt(node[idx])},")
        out.append("};")
    out.append("")
    out.append("} // namespace mko::detail")
    sys.stdout.reconfigure(newline="\n")  # LF auch unter Windows
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
