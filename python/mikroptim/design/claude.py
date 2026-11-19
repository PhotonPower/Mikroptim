"""Entscheider über die Claude-API (M10-T6c, ADR-0016).

Gleiche Rolle wie ``OllamaDecider``: Das Modell sieht den Zustand ``mikroptim.design.state/1``,
darf Lese-Werkzeuge aufrufen und antwortet mit Aktionen ``mikroptim.design.action/1``.
Unterschiede zur Ollama-Anbindung, bedingt durch die API der aktuellen Claude-Modelle:

- Die Antwort kommt als Aufruf des Werkzeugs ``submit_decision`` (``strict: true``) statt als
  schema-beschränkter Text. Erzwungenes ``tool_choice`` liefert bei Claude Opus 5.5 und
  Sonnet 5.5 einen Fehler 400; deshalb ``tool_choice`` ``auto`` und die Anweisung im Prompt,
  genau einmal ``submit_decision`` aufzurufen. Die Eingabe wird trotzdem gegen das
  Aktionsschema geprüft (``validate_response`` und die Parameter aus ``schema.ACTION_DEFS``).
- Thinking adaptiv (bei Opus 5.5 nicht abschaltbar), Aufwand über ``output_config.effort``.
- Server-seitige Rückfälle bei Ablehnung (``fallbacks="default"``) außer bei Haiku.
- Prompt-Caching: fester System-Prompt und sortierte Werkzeugliste vorn.
- Kosten aus ``response.usage`` pro Anfrage, laufend summiert in einem ``CostLedger`` mit
  Obergrenze; vor jeder Anfrage wird geprüft, dass die Obergrenze auch im ungünstigsten Fall
  (volle ``max_tokens``) nicht überschritten wird, sonst ``BudgetExceeded``.

Das Python-SDK ``anthropic`` ist ein optionales Extra (``pip install mikroptim[claude]``) und
wird erst beim ersten Aufruf geladen. Der API-Schlüssel kommt ausschließlich aus der Umgebung
(``ANTHROPIC_API_KEY`` oder ein Profil von ``ant auth login``, das das SDK selbst findet) und
erscheint nie im Protokoll.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib

from . import knowledge
from .actions import ActionError, read_action, validate_response
from .deciders import Decider, DecisionError
from .files import read_system
from .prompt import SYSTEM_PROMPT
from .schema import (ACTION_DEFS, GLASS_QUERY_DESCRIPTION, MODULE_LOOKUP_DESCRIPTION,
                     TOOL_ACTIONS)

DEFAULT_MODEL = "claude-opus-5-5"
SUBMIT_TOOL = "submit_decision"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

#: Preise in US-$ pro Million Tokens (Stand 2026-11; Cache-Schreiben = 1,25 × Input)
PRICES_USD_PER_MTOK = {
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write": 5.0},
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_read": 0.20, "cache_write": 2.5},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01,
                         "cache_write": 0.125},
}
#: Für unbekannte Modelle (z. B. ein server-seitiges Rückfallmodell): bewusst hoch angesetzt
CONSERVATIVE_PRICE = {"input": 5.0, "output": 25.0, "cache_read": 0.50, "cache_write": 6.25}
#: Haiku 5.5: Prompts über 100.000 Tokens kosten das Fünffache
HAIKU_LONG_PROMPT = 100_000

CLAUDE_INSTRUCTIONS = (
    "Arbeitsweise in dieser Sitzung: Rufe bei Bedarf Lese-Werkzeuge auf. Deine Entscheidung "
    "gibst du ausschließlich über das Werkzeug submit_decision ab, und zwar genau einmal pro "
    "Entscheidung, mit 1 bis 4 Aktionen und einer kurzen Einschätzung (assessment). "
    "Schreibe die Entscheidung nicht als Text.")


class BudgetExceeded(RuntimeError):
    """Die Kostenobergrenze würde mit der nächsten Anfrage überschritten; der Lauf bricht ab
    (kein Rückfall-Entscheider)."""


class ClaudeUnavailableError(RuntimeError):
    """SDK fehlt, Zugangsdaten fehlen oder sind ungültig, Modell unbekannt: Konfigurationsfehler,
    der Lauf bricht ab (kein Rückfall-Entscheider)."""


class PrivacyError(RuntimeError):
    """Der Kandidat verwendet Daten aus einem privaten Verzeichnis (lizenzierte Glaskataloge,
    Patentsysteme); diese verlassen den Rechner nicht (AGENTS.md §8, ADR-0016)."""


# --- Kosten -------------------------------------------------------------------------------------


def usage_dict(usage) -> dict:
    """Die vier abgerechneten Token-Arten aus ``response.usage`` (fehlende Felder = 0)."""
    def get(name):
        v = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        return int(v or 0)

    return {"input_tokens": get("input_tokens"), "output_tokens": get("output_tokens"),
            "cache_read_input_tokens": get("cache_read_input_tokens"),
            "cache_creation_input_tokens": get("cache_creation_input_tokens")}


def price_for(model: str) -> dict:
    return PRICES_USD_PER_MTOK.get(model, CONSERVATIVE_PRICE)


def cost_usd(model: str, usage: dict) -> float:
    """Kosten einer Anfrage in US-$ nach ``PRICES_USD_PER_MTOK``."""
    p = dict(price_for(model))
    prompt = (usage["input_tokens"] + usage["cache_read_input_tokens"]
              + usage["cache_creation_input_tokens"])
    if model == "claude-haiku-5-5" and prompt > HAIKU_LONG_PROMPT:
        p = {k: 5.0 * v for k, v in p.items()}
    return (usage["input_tokens"] * p["input"] + usage["output_tokens"] * p["output"]
            + usage["cache_read_input_tokens"] * p["cache_read"]
            + usage["cache_creation_input_tokens"] * p["cache_write"]) / 1e6


class CostLedger:
    """Laufende Summe der Kosten mit Obergrenze, optional in einer JSON-Datei über mehrere
    Läufe und Prozesse hinweg (``mikroptim.design.cost/1``; enthält keine Zugangsdaten)."""

    FORMAT = "mikroptim.design.cost/1"

    def __init__(self, limit_usd: float = 20.0, path=None):
        self.limit_usd = float(limit_usd)
        self.path = pathlib.Path(path) if path else None
        self.spent_usd = 0.0
        self.requests = 0
        self.by_model: dict = {}
        if self.path and self.path.exists():
            d = json.loads(self.path.read_text(encoding="utf-8"))
            self.spent_usd = float(d.get("spent_usd", 0.0))
            self.requests = int(d.get("requests", 0))
            self.by_model = d.get("by_model", {})

    @property
    def remaining_usd(self) -> float:
        return self.limit_usd - self.spent_usd

    def check(self, reserve_usd: float) -> None:
        if self.spent_usd + reserve_usd > self.limit_usd:
            raise BudgetExceeded(
                f"Kostenobergrenze {self.limit_usd:.2f} $: bisher {self.spent_usd:.4f} $, die "
                f"nächste Anfrage könnte bis zu {reserve_usd:.4f} $ kosten – Abbruch")

    def add(self, model: str, usage: dict, cost: float) -> None:
        self.spent_usd += cost
        self.requests += 1
        m = self.by_model.setdefault(model, {"requests": 0, "cost_usd": 0.0,
                                             **{k: 0 for k in usage}})
        m["requests"] += 1
        m["cost_usd"] += cost
        for k, v in usage.items():
            m[k] = m.get(k, 0) + v
        self.save()

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"format": self.FORMAT, "limit_usd": self.limit_usd, "spent_usd": self.spent_usd,
             "requests": self.requests, "by_model": self.by_model},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# --- Schemas für strict-Werkzeuge ---------------------------------------------------------------

#: Von strict-Werkzeugen nicht unterstützte Schlüsselwörter; sie werden entfernt und clientseitig
#: über die Prüfung der Antwort (bzw. die Prüfungen der Aktionen) abgedeckt.
_UNSUPPORTED = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
                "minLength", "maxLength", "minItems", "maxItems", "uniqueItems", "pattern"}


def strict_schema(schema):
    """JSON-Schema für ``strict: true``: ``oneOf`` → ``anyOf``, ``prefixItems`` → ``items``,
    ohne Zahl- und Längenschranken, jedes Objekt mit ``additionalProperties: false``."""
    if isinstance(schema, list):
        return [strict_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k in _UNSUPPORTED:
            continue
        if k == "oneOf":
            out["anyOf"] = strict_schema(v)
        elif k == "prefixItems":
            types = {json.dumps(strict_schema(i), sort_keys=True) for i in v}
            out["items"] = (json.loads(types.pop()) if len(types) == 1
                            else {"anyOf": [json.loads(t) for t in sorted(types)]})
        elif k == "properties":
            out[k] = {name: strict_schema(p) for name, p in v.items()}
        else:
            out[k] = strict_schema(v)
    if out.get("type") == "object" or "properties" in out:
        out.setdefault("type", "object")
        out["additionalProperties"] = False
        out.setdefault("properties", {})
        out.setdefault("required", [])
    return out


def decision_actions() -> list[str]:
    return [k for k in ACTION_DEFS if k not in TOOL_ACTIONS]


def tool_definitions(strict: bool = True) -> list[dict]:
    """Werkzeuge der Claude-API, nach Namen sortiert (stabiles Präfix für das Caching)."""
    desc = {
        "get_surface_table": "Aberrationsbericht pro Fläche (paraxiale Strahlen, Seidel-Beiträge, "
                             "Farbfehler, Sphärochromasie; Wellen nach Welford). Aufrufen, wenn die "
                             "Gruppenbilanzen im Zustand nicht reichen, um die Stelle einer "
                             "strukturellen Änderung zu wählen.",
        "get_ghost_diagnosis": "Diagnose der Geisterplatten (aktive Platten, Ordnungen, Gewinn). "
                               "Aufrufen, wenn das System Geisterplatten hat.",
        "evaluate": "Auswertung auf dem dichten Pupillenraster (RMS, Strehl, optional PSF). "
                    "Aufrufen, wenn die Merit allein für eine Abschlussentscheidung nicht reicht.",
        "glass_preselect": "Glaspaare (order 1, Achromat) bzw. -tripel (order 2, Apochromat) aus "
                           "den Katalogen mit Brechkraftanteilen; element hält dessen Glas fest. "
                           "Aufrufen vor einer Glaswahl für split_lens oder insert_null_meniscus.",
        "sensitivity": "Empfindlichkeit für Dezentrierung, Verkippung, Radius, Dicke und "
                       "Brechzahl (RMS-Zunahme bei ±δ, sortiert) und Befunde der Herstellbarkeit "
                       "(Randdicken, Halbkugelgrad, Zentrierbarkeit). Aufrufen, wenn die "
                       "Kurzfassung im Zustand für eine Abwägung zwischen Korrektur und "
                       "Empfindlichkeit nicht reicht.",
        "glass_query": GLASS_QUERY_DESCRIPTION + ". Aufrufen vor einer Glaswahl nach einer "
                       "Regel der Wissensbasis (z. B. Merte-Partner, Autofluoreszenz).",
        "module_lookup": MODULE_LOOKUP_DESCRIPTION + ". Aufrufen, bevor insert_module, "
                         "replace_group oder start_from_skeleton ein Modul wählt.",
        SUBMIT_TOOL: "Gibt die Entscheidung für den aktuellen Kandidaten ab (Format "
                     "mikroptim.design.action/1 ohne format und candidate): 1 bis 4 Aktionen, "
                     "die nacheinander ausgeführt werden, und eine kurze Einschätzung. Genau "
                     "einmal pro Entscheidung aufrufen, nachdem die nötigen Lese-Werkzeuge "
                     "gelaufen sind.",
    }
    tools = []
    for name in TOOL_ACTIONS:
        params = {k: v for k, v in ACTION_DEFS[name]["properties"].items() if k != "action"}
        tools.append({"name": name, "description": desc[name],
                      "input_schema": {"type": "object", "properties": params}})
    submit = {"type": "object", "required": ["assessment", "actions"], "properties": {
        "assessment": {"type": "string", "maxLength": 600},
        "actions": {"type": "array", "minItems": 1, "maxItems": 4,
                    "items": {"oneOf": [ACTION_DEFS[k] for k in decision_actions()]}}}}
    tools.append({"name": SUBMIT_TOOL, "description": desc[SUBMIT_TOOL], "input_schema": submit})
    out = []
    for t in sorted(tools, key=lambda t: t["name"]):
        t = {**t, "input_schema": strict_schema(copy.deepcopy(t["input_schema"]))}
        if strict:
            t["strict"] = True
        out.append(t)
    return out


def check_actions(actions: list[dict]) -> None:
    """Parameter jeder Aktion gegen ``schema.ACTION_DEFS``: Pflichtparameter vorhanden, keine
    unbekannten Parameter, keine Lese-Werkzeuge in der Entscheidung."""
    allowed = set(decision_actions())
    for a in actions:
        name = a.get("action")
        if name not in allowed:
            raise ActionError("unknown_action",
                              f"{name!r} ist in einer Entscheidung nicht erlaubt (erlaubt: "
                              f"{', '.join(sorted(allowed))}; Lese-Werkzeuge als Werkzeugaufruf)")
        d = ACTION_DEFS[name]
        missing = [k for k in d["required"] if k not in a]
        if missing:
            raise ActionError("missing_parameter", f"{name}: Parameter {', '.join(missing)} fehlt")
        unknown = [k for k in a if k not in d["properties"]]
        if unknown:
            raise ActionError("invalid_parameter",
                              f"{name}: unbekannte Parameter {', '.join(unknown)}")


# --- SDK ------------------------------------------------------------------------------------------


def _import_sdk(required: bool):
    try:
        import anthropic  # noqa: PLC0415 - optionales Extra, bewusst spät geladen
    except ImportError:
        if required:
            raise ClaudeUnavailableError(
                "Das Python-SDK 'anthropic' fehlt: pip install mikroptim[claude]") from None
        return None
    return anthropic


def _block_dict(block) -> dict:
    """Inhaltsblock als dict fürs Protokoll (ohne die Signatur der Thinking-Blöcke)."""
    if isinstance(block, dict):
        d = dict(block)
    elif hasattr(block, "to_dict"):
        d = block.to_dict()
    elif hasattr(block, "model_dump"):
        d = block.model_dump()
    else:
        d = dict(vars(block))
    d.pop("signature", None)
    return d


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


class ClaudeDecider(Decider):
    """Entscheider über die Claude-API (Modell per ``model``, Standard ``claude-opus-5-5``).

    ``client`` erlaubt eine Attrappe in Tests (Methode ``beta.messages.create``); ohne Angabe
    wird ``anthropic.Anthropic()`` erzeugt, das die Zugangsdaten selbst aus der Umgebung holt.
    """

    kind = "claude"

    def __init__(self, model: str = DEFAULT_MODEL, *, effort: str = "medium",
                 max_tokens: int = 16000, max_tool_calls: int = 6, max_retries: int = 2,
                 fallbacks: bool | None = None, strict: bool = True,
                 ledger: CostLedger | None = None, client=None, timeout: float = 600.0,
                 sdk_max_retries: int = 4, allow_private: bool = False,
                 system_prompt: str = SYSTEM_PROMPT):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_tool_calls = max_tool_calls
        self.max_retries = max_retries
        # Server-seitige Rückfälle: für Haiku nicht verfügbar
        self.fallbacks = (not model.startswith("claude-haiku")) if fallbacks is None else fallbacks
        self.strict = strict
        self.ledger = ledger if ledger is not None else CostLedger()
        self.timeout = timeout
        self.sdk_max_retries = sdk_max_retries
        self.allow_private = allow_private
        self.system_prompt = system_prompt + "\n" + CLAUDE_INSTRUCTIONS + "\n"
        self.tools = tool_definitions(strict)
        self._client = client
        self.last_trace: dict = {}

    # --- Beschreibung fürs Protokoll ----------------------------------------------------------

    def describe(self) -> dict:
        sdk = _import_sdk(required=False)
        t = self.last_trace
        return {"kind": self.kind, "model": self.model, "effort": self.effort,
                "thinking": "adaptive", "fallbacks": self.fallbacks, "strict": self.strict,
                "max_tokens": self.max_tokens,
                "sdk_version": getattr(sdk, "__version__", None) if sdk else None,
                "prompt_sha256": hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest(),
                "knowledge_sha256": knowledge.digest(),
                "tools_sha256": hashlib.sha256(json.dumps(self.tools, sort_keys=True)
                                               .encode("utf-8")).hexdigest(),
                "usage": t.get("usage"), "cost_usd": t.get("cost_usd"),
                "ledger_spent_usd": self.ledger.spent_usd, "trace": t}

    # --- API ----------------------------------------------------------------------------------

    def client(self):
        if self._client is None:
            sdk = _import_sdk(required=True)
            try:
                self._client = sdk.Anthropic(timeout=self.timeout, max_retries=self.sdk_max_retries)
            except sdk.AnthropicError as e:  # z. B. keine Zugangsdaten
                raise ClaudeUnavailableError(f"Claude-Client: {e}") from None
        return self._client

    def _reserve(self, messages) -> float:
        """Obere Schranke der Kosten der nächsten Anfrage (Zeichen/2 als Tokens, volle
        max_tokens) zum Modellpreis bzw. zum konservativen Preis, falls höher."""
        chars = len(self.system_prompt) + len(json.dumps(self.tools)) + len(
            json.dumps(messages, default=lambda o: _block_dict(o), ensure_ascii=False))
        p = price_for(self.model)
        if self.fallbacks:  # ein Rückfallmodell kann teurer sein
            p = {k: max(v, CONSERVATIVE_PRICE[k]) for k, v in p.items()}
        return (chars / 2.0 * p["cache_write"] + self.max_tokens * p["output"]) / 1e6

    def _request(self, messages, trace):
        self.ledger.check(self._reserve(messages))
        kw = {"model": self.model, "max_tokens": self.max_tokens,
              "system": [{"type": "text", "text": self.system_prompt,
                          "cache_control": {"type": "ephemeral"}}],
              "tools": self.tools, "tool_choice": {"type": "auto"}, "messages": messages,
              "thinking": {"type": "adaptive", "display": "summarized"},
              "output_config": {"effort": self.effort},
              "cache_control": {"type": "ephemeral"}}
        if self.fallbacks:
            kw["betas"] = [FALLBACK_BETA]
            kw["fallbacks"] = "default"
        sdk = _import_sdk(required=False)
        client = self.client()
        if sdk is None:  # nur mit Attrappe ohne installiertes SDK
            resp = client.beta.messages.create(**kw)
        else:
            try:
                resp = client.beta.messages.create(**kw)
            except (sdk.AuthenticationError, sdk.PermissionDeniedError, sdk.NotFoundError) as e:
                raise ClaudeUnavailableError(f"Claude-API: {type(e).__name__}: {e.message}") \
                    from None
            except sdk.BadRequestError as e:
                raise ClaudeUnavailableError(f"Claude-API: Anfrage abgelehnt (400): {e.message}") \
                    from None
            except sdk.RateLimitError as e:  # nach den Wiederholungen des SDK
                raise DecisionError(f"Claude-API: Ratenlimit: {e.message}") from None
            except sdk.APIStatusError as e:
                raise DecisionError(f"Claude-API: Status {e.status_code}: {e.message}") from None
            except sdk.APIConnectionError as e:  # inkl. Zeitüberschreitung
                raise DecisionError(f"Claude-API: Verbindung: {e}") from None
            except (sdk.AnthropicError, TypeError) as e:
                # z. B. keine Zugangsdaten (das SDK meldet das erst beim Aufruf, als TypeError)
                raise ClaudeUnavailableError(f"Claude-API: {e}") from None
        served = _get(resp, "model") or self.model
        usage = usage_dict(_get(resp, "usage") or {})
        cost = cost_usd(served, usage)
        self.ledger.add(served, usage, cost)
        stop_details = _get(resp, "stop_details")
        trace["requests"].append({
            "request_id": _get(resp, "_request_id") or _get(resp, "id"), "model": served,
            "stop_reason": _get(resp, "stop_reason"),
            "stop_details": _block_dict(stop_details) if stop_details else None,
            "usage": usage, "cost_usd": cost})
        for k, v in usage.items():
            trace["usage"][k] += v
        trace["cost_usd"] += cost
        return resp

    # --- Entscheidung -------------------------------------------------------------------------

    def _check_privacy(self, system_path, session) -> None:
        if self.allow_private:
            return
        data = read_system(system_path)
        paths = [str(p) for p in data.get("glass_catalogs", [])]
        paths.append(str(session.meta.get("start", "")))
        for p in paths:
            parts = {s.lower() for s in pathlib.PurePath(p).parts}
            if "private" in parts:
                raise PrivacyError(f"{p}: private Daten gehen nicht an die Claude-API "
                                   "(lokales Modell über Ollama verwenden)")

    def _run_tool(self, name, args, system_path) -> dict:
        try:
            return read_action({"action": name, **args}, system_path)
        except (ActionError, RuntimeError, ValueError, TypeError, LookupError) as e:
            return {"error": str(e)}

    def decide(self, state: dict, session) -> dict:
        system_path = session.system_path(state["candidate"])
        self._check_privacy(system_path, session)
        trace = {"requests": [], "tool_calls": [], "thinking": [], "raw": [], "rejected": [],
                 "usage": {k: 0 for k in usage_dict({})}, "cost_usd": 0.0}
        self.last_trace = trace
        messages = [{"role": "user", "content":
                     f"Zustand von Kandidat {state['candidate']}:\n"
                     + json.dumps(state, ensure_ascii=False)}]
        tool_calls = 0
        failures = 0
        for _ in range(self.max_tool_calls + self.max_retries + 2):
            resp = self._request(messages, trace)
            stop = _get(resp, "stop_reason")
            content = list(_get(resp, "content") or [])
            trace["raw"].append([_block_dict(b) for b in content])
            for b in content:
                if _get(b, "type") == "thinking" and _get(b, "thinking"):
                    trace["thinking"].append(_get(b, "thinking"))
            if stop == "refusal":
                details = _get(resp, "stop_details")
                raise DecisionError(f"Claude ({self.model}): Anfrage abgelehnt"
                                    + (f" ({_get(details, 'category')})" if details else ""))
            # Antwort unverändert anhängen (inkl. Thinking-Blöcke), Verlauf nur anhängen
            messages.append({"role": "assistant", "content": content})
            uses = [b for b in content if _get(b, "type") == "tool_use"]
            if stop == "max_tokens":
                failures += 1
                trace["rejected"].append("max_tokens: Antwort abgeschnitten")
                note = "Antwort abgeschnitten (max_tokens). "
                results = [{"type": "tool_result", "tool_use_id": _get(b, "id"),
                            "is_error": True, "content": "abgeschnitten, nicht ausgeführt"}
                           for b in uses]
                if failures > self.max_retries:
                    break
                messages.append({"role": "user", "content": results + [{
                    "type": "text", "text": note + "Rufe jetzt submit_decision knapp auf."}]})
                continue
            results = []
            decision = None
            invalid = False
            for b in uses:
                name, args, uid = _get(b, "name"), dict(_get(b, "input") or {}), _get(b, "id")
                if name == SUBMIT_TOOL:
                    response = {"format": "mikroptim.design.action/1",
                                "candidate": state["candidate"],
                                "assessment": args.get("assessment", ""),
                                "actions": args.get("actions")}
                    try:
                        check_actions(validate_response(response))
                    except ActionError as e:
                        invalid = True
                        trace["rejected"].append(str(e))
                        results.append({"type": "tool_result", "tool_use_id": uid,
                                        "is_error": True,
                                        "content": f"Antwort ungültig ({e.code}): {e}"})
                        continue
                    if decision is None:
                        decision = response
                elif name in TOOL_ACTIONS and tool_calls < self.max_tool_calls:
                    tool_calls += 1
                    result = self._run_tool(name, args, system_path)
                    trace["tool_calls"].append({"name": name, "arguments": args})
                    results.append({"type": "tool_result", "tool_use_id": uid,
                                    "content": json.dumps(result, ensure_ascii=False)[:60000]})
                else:
                    reason = ("Werkzeuglimit erreicht" if name in TOOL_ACTIONS
                              else f"unbekanntes Werkzeug {name}")
                    results.append({"type": "tool_result", "tool_use_id": uid, "is_error": True,
                                    "content": f"{reason}; rufe jetzt submit_decision auf"})
            if decision is not None:
                trace["attempts"] = failures + 1
                return decision
            if not uses:
                failures += 1
                trace["rejected"].append("keine Entscheidung über submit_decision")
                if failures > self.max_retries:
                    break
                messages.append({"role": "user", "content":
                                 "Bitte gib deine Entscheidung jetzt über das Werkzeug "
                                 "submit_decision ab (genau ein Aufruf)."})
                continue
            if invalid:
                failures += 1
                if failures > self.max_retries:
                    break
            messages.append({"role": "user", "content": results})
        raise DecisionError(f"Claude ({self.model}): keine gültige Entscheidung nach "
                            f"{failures} Fehlversuchen")
