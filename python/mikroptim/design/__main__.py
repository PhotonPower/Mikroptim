"""``python -m mikroptim.design``: Entwurfssitzung auf der Kommandozeile (Mensch als Entscheider).

Befehle::

    new <system.json> <dir> [--spec spec.json]   Sitzung anlegen (Kandidat 0)
    state <dir> [-c N] [--json]                  Zustand eines Kandidaten
    act <dir> <antwort.json|-> [-c N]            Antwort (mikroptim.design.action/1) ausführen
    tree <dir>                                   Kandidatenbaum
    replay <dir> <neues_dir>                     Protokoll in einer neuen Sitzung abspielen
    run <dir> --decider rule|ollama|human [--confirm] [--model M] [--host URL] [--think|--no-think]
        [--steps N] [-c N] [--fallback rule|human|none]   Folge von Entscheidungen
"""

from __future__ import annotations

import argparse
import json
import sys

from .deciders import ConfirmDecider, HumanDecider, RuleDecider, replay
from .ollama import OllamaDecider
from .session import Session
from .state import format_state


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m mikroptim.design")
    sub = p.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("system")
    n.add_argument("dir")
    n.add_argument("--spec")
    st = sub.add_parser("state")
    st.add_argument("dir")
    st.add_argument("-c", "--candidate", type=int, default=None)
    st.add_argument("--json", action="store_true")
    ac = sub.add_parser("act")
    ac.add_argument("dir")
    ac.add_argument("response")
    ac.add_argument("-c", "--candidate", type=int, default=None)
    tr = sub.add_parser("tree")
    tr.add_argument("dir")
    rp = sub.add_parser("replay")
    rp.add_argument("dir")
    rp.add_argument("new_dir")
    ru = sub.add_parser("run")
    ru.add_argument("dir")
    ru.add_argument("--decider", choices=["rule", "ollama", "human"], required=True)
    ru.add_argument("--confirm", action="store_true", help="Mensch bestätigt jeden Vorschlag")
    ru.add_argument("--model", help="Ollama-Modell, z. B. qwen3:30b")
    ru.add_argument("--host", default="http://localhost:11434")
    th = ru.add_mutually_exclusive_group()
    th.add_argument("--think", dest="think", action="store_true", default=None)
    th.add_argument("--no-think", dest="think", action="store_false")
    ru.add_argument("--steps", type=int, default=10)
    ru.add_argument("-c", "--candidate", type=int, default=None)
    ru.add_argument("--fallback", choices=["rule", "human", "none"], default="none")
    a = p.parse_args(argv)
    try:
        if a.cmd == "new":
            spec = json.load(open(a.spec, encoding="utf-8")) if a.spec else None
            s = Session.create(a.system, a.dir, spec=spec)
            print(format_state(s.state(0)))
            return 0
        s = Session(a.dir)
        if a.cmd == "state":
            c = a.candidate if a.candidate is not None else max(s.open_candidates() or [0])
            state = s.state(c)
            print(json.dumps(state, ensure_ascii=False, indent=2) if a.json else format_state(state))
        elif a.cmd == "act":
            c = a.candidate if a.candidate is not None else max(s.open_candidates() or [0])
            text = sys.stdin.read() if a.response == "-" else open(a.response, encoding="utf-8").read()
            entry = s.apply(c, json.loads(text), decider={"kind": "mensch"}, state=s.state(c))
            print(json.dumps(entry["results"], ensure_ascii=False, indent=2))
            return 0 if all(r["status"] == "ok" for r in entry["results"]) else 1
        elif a.cmd == "tree":
            for cid, c in sorted(s.candidates().items()):
                print(f"{cid:>4}  Eltern {c['parent']!s:>4}  {c['status']:<10} {c['action'] or ''}")
        elif a.cmd == "run":
            if a.decider == "ollama":
                if not a.model:
                    print("Fehler: --model fehlt", file=sys.stderr)
                    return 2
                decider = OllamaDecider(a.model, host=a.host, think=a.think)
            elif a.decider == "rule":
                decider = RuleDecider()
            else:
                decider = HumanDecider()
            if a.confirm and a.decider != "human":
                decider = ConfirmDecider(decider)
            fallback = {"rule": RuleDecider(), "human": HumanDecider(), "none": None}[a.fallback]
            c = a.candidate if a.candidate is not None else max(s.open_candidates() or [0])
            last = s.run(decider, c, max_steps=a.steps, fallback=fallback)
            print(f"letzter Kandidat {last}; {len(s.candidates())} Kandidaten")
        elif a.cmd == "replay":
            r = replay(s.log(), s.meta["start"], a.new_dir, spec=s.meta["spec"],
                       budget=s.meta["budget"])
            print(f"{len(r.log())} Entscheidungen abgespielt, {len(r.candidates())} Kandidaten")
        return 0
    except (OSError, ValueError, KeyError) as e:
        print(f"Fehler: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
