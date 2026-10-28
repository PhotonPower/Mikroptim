"""Entscheider über ein lokales Sprachmodell mit Ollama (M10-T6b, ADR-0012).

Ablauf pro Entscheidung:

1. Chat mit System-Prompt und Zustand (JSON); das Modell darf Lese-Werkzeuge aufrufen
   (Tool-Calls), deren Ergebnisse als Nachrichten zurückgehen (höchstens ``max_tool_calls``).
2. Abschließende Anfrage mit ``format`` = Antwortschema (schema-beschränkte Ausgabe).
3. Prüfung der Antwort; bei Fehlern Rückmeldung an das Modell, höchstens ``max_retries``
   Nachbesserungen, danach ``DecisionError`` (der Aufrufer wählt den Rückfall).

Protokolliert werden Modellname, Digest (``/api/tags``), Optionen, Prüfsumme des Prompts,
Denktext (Thinking-Modelle), Tool-Calls und Rohantworten (``last_trace``). Die Parameternamen
folgen der Chat-API von Ollama (``/api/chat``: ``tools``, ``format``, ``think``, ``options``,
``keep_alive``); bei neuen Versionen gegen die Ollama-Dokumentation prüfen.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request

from .actions import ActionError, read_action, validate_response
from .deciders import Decider, DecisionError
from .prompt import SYSTEM_PROMPT
from .schema import TOOL_ACTIONS, response_schema, tool_specs

DEFAULT_OPTIONS = {"temperature": 0.0, "seed": 0, "num_ctx": 32768}


class OllamaDecider(Decider):
    kind = "ollama"

    def __init__(self, model: str, *, host: str = "http://localhost:11434", think=None,
                 options: dict | None = None, keep_alive: str | None = "30m",
                 max_tool_calls: int = 6, max_retries: int = 2, timeout: float = 600.0,
                 system_prompt: str = SYSTEM_PROMPT):
        self.model = model
        self.host = host.rstrip("/")
        self.think = think
        self.options = {**DEFAULT_OPTIONS, **(options or {})}
        self.keep_alive = keep_alive
        self.max_tool_calls = max_tool_calls
        self.max_retries = max_retries
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.last_trace: dict = {}
        self._digest = None

    # --- HTTP -------------------------------------------------------------------------------

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.host + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def digest(self) -> str | None:
        if self._digest is None:
            try:
                with urllib.request.urlopen(self.host + "/api/tags", timeout=30) as r:
                    tags = json.loads(r.read().decode("utf-8"))
                for m in tags.get("models", []):
                    if m.get("name") == self.model or m.get("model") == self.model:
                        self._digest = m.get("digest")
            except OSError:
                self._digest = None
        return self._digest

    def _chat(self, messages, *, tools=None, fmt=None) -> dict:
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": self.options}
        if self.keep_alive is not None:
            body["keep_alive"] = self.keep_alive
        if self.think is not None:
            body["think"] = self.think
        if tools:
            body["tools"] = tools
        if fmt is not None:
            body["format"] = fmt
        return self._post("/api/chat", body)

    # --- Entscheidung -----------------------------------------------------------------------

    def describe(self) -> dict:
        return {"kind": self.kind, "model": self.model, "digest": self.digest(),
                "think": self.think, "options": self.options,
                "prompt_sha256": hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest(),
                "trace": self.last_trace}

    def decide(self, state: dict, session) -> dict:
        trace = {"tool_calls": [], "thinking": [], "raw": [], "rejected": []}
        self.last_trace = trace
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "Zustand:\n" + json.dumps(state, ensure_ascii=False)}]
        system_path = session.system_path(state["candidate"])

        # 1. Lese-Werkzeuge
        for _ in range(self.max_tool_calls):
            reply = self._chat(messages, tools=tool_specs())
            msg = reply.get("message", {})
            if msg.get("thinking"):
                trace["thinking"].append(msg["thinking"])
            calls = msg.get("tool_calls") or []
            if not calls:
                break
            messages.append({"role": "assistant", "content": msg.get("content", ""),
                             "tool_calls": calls})
            for call in calls:
                fn = call.get("function", {})
                name, args = fn.get("name"), fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if name not in TOOL_ACTIONS:
                    result = {"error": f"unbekanntes Werkzeug {name}"}
                else:
                    try:
                        result = read_action({"action": name, **args}, system_path)
                    except (ActionError, RuntimeError) as e:
                        result = {"error": str(e)}
                trace["tool_calls"].append({"name": name, "arguments": args})
                messages.append({"role": "tool", "tool_name": name,
                                 "content": json.dumps(result, ensure_ascii=False)[:60000]})

        # 2./3. Antwort mit Schema, Prüfung, Nachbesserung
        messages.append({"role": "user", "content":
                         "Antworte jetzt nur mit JSON im Format mikroptim.design.action/1 für "
                         f"Kandidat {state['candidate']}."})
        for attempt in range(self.max_retries + 1):
            reply = self._chat(messages, fmt=response_schema())
            msg = reply.get("message", {})
            if msg.get("thinking"):
                trace["thinking"].append(msg["thinking"])
            content = msg.get("content", "")
            trace["raw"].append(content)
            try:
                response = json.loads(content)
                validate_response(response)
                response["candidate"] = state["candidate"]
                trace["attempts"] = attempt + 1
                return response
            except (json.JSONDecodeError, ActionError) as e:
                trace["rejected"].append(str(e))
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content":
                                 f"Antwort ungültig: {e}. Bitte korrigiert erneut, nur JSON."})
        raise DecisionError(f"Ollama ({self.model}): keine gültige Antwort nach "
                            f"{self.max_retries + 1} Versuchen")
