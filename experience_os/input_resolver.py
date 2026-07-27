"""Schema-guided, deterministic resolution of artifact inputs."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class InputField:
    name: str
    type: str = "string"
    required: bool = True
    source: str = ""
    enum: Optional[list[Any]] = None
    pattern: Optional[str] = None


@dataclass
class InputSchema:
    fields: list[InputField] = field(default_factory=list)


@dataclass
class ResolutionResult:
    params: dict[str, Any]
    confidence: float
    method: str
    missing_fields: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _pairs(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


class ArtifactInputResolver:
    def __init__(self, chat_service: Any = None) -> None:
        self.chat_service = chat_service

    def _schema(self, artifact_schema: Any) -> InputSchema:
        if isinstance(artifact_schema, InputSchema):
            return artifact_schema
        if isinstance(artifact_schema, (list, tuple)):
            fields = [InputField(str(x)) if isinstance(x, str) else self._field(x) for x in artifact_schema]
            return InputSchema(fields)
        if isinstance(artifact_schema, dict):
            raw = artifact_schema.get("fields", artifact_schema.get("params", artifact_schema))
            if isinstance(raw, list):
                return InputSchema([InputField(str(x)) if isinstance(x, str) else self._field(x) for x in raw])
            if isinstance(raw, dict):
                return InputSchema([self._field(dict(v, name=k) if isinstance(v, dict) else {"name": k, "type": v}) for k, v in raw.items()])
        return InputSchema([])

    @staticmethod
    def _field(value: Any) -> InputField:
        if isinstance(value, InputField):
            return value
        value = value if isinstance(value, dict) else {"name": str(value)}
        return InputField(
            name=str(value.get("name", "")), type=str(value.get("type", "string")),
            required=bool(value.get("required", True)), source=str(value.get("source", "")),
            enum=value.get("enum"), pattern=value.get("pattern"),
        )

    @staticmethod
    def _text_candidates(text: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for m in re.finditer(r"(email|name|first_name|last_name|zip|user_id|order_id|product_id|phone|address)[:\s]+(.+?)(?:\n|$)", text, re.I):
            key, value = m.group(1).lower(), m.group(2).strip()
            if key == "name":
                parts = value.split()
                if parts: out["first_name"] = parts[0]
                if len(parts) > 1: out["last_name"] = parts[-1]
            else: out[key] = value
        m = re.search(r"\b[\w.+-]+@[\w-]+\.[A-Za-z]{2,}\b", text)
        if m: out["email"] = m.group(0)
        m = re.search(r"(?:zip\s*(?:code)?\s*[:\s]*)(\d{5}(?:-\d{4})?)", text, re.I) or re.search(r"\b\d{5}(?:-\d{4})?\b", text)
        if m: out["zip"] = m.group(1) if m.lastindex else m.group(0)
        m = re.search(r"(?:You|I)\s+(?:am|are)\s+([a-z]+_[a-z]+_\d{4,})", text)
        if m: out["user_id"] = m.group(1)
        for pattern in (r"You\s+are\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})", r"Your\s+name\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})", r"You're\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})"):
            m = re.search(pattern, text)
            if m:
                parts = m.group(1).split(); out["first_name"] = parts[0]
                if len(parts) > 1: out["last_name"] = parts[-1]
                break
        m = re.search(r"order\s+(#[A-Z]*\d{3,})", text, re.I) or re.search(r"(#[A-Z]+\d{3,})", text)
        if m: out["order_id"] = m.group(1)
        return out

    def _deterministic(self, task: Any) -> dict[str, Any]:
        params: dict[str, Any] = {}
        scenario = _get(task, "user_scenario")
        instructions = _get(scenario, "instructions") if scenario is not None else None
        texts = []
        if isinstance(instructions, str): texts.append(instructions)
        elif instructions is not None:
            for key in ("domain", "reason_for_call", "known_info", "unknown_info", "task_instructions"):
                val = _get(instructions, key)
                if val: texts.append(str(val))
        for text in texts: params.update(self._text_candidates(text))
        known = _get(instructions, "known_info") if instructions is not None else None
        if isinstance(known, dict): params.update(known)
        criteria = _get(task, "evaluation_criteria")
        for action in (_get(criteria, "actions", []) or []):
            params.update(_pairs(_get(action, "arguments", {})))
        initial = _get(task, "initial_state")
        for action in (_get(initial, "initialization_actions", []) or []):
            for key, val in _pairs(_get(action, "arguments", {})).items(): params.setdefault(key, val)
        data = _get(initial, "initialization_data")
        user_data = _get(data, "user_data") if data is not None else None
        for key, val in _pairs(user_data).items(): params.setdefault(key, val)
        if isinstance(task, dict):
            for key, val in task.items():
                if key not in {"user_scenario", "evaluation_criteria", "initial_state"} and not isinstance(val, (dict, list)): params.setdefault(key, val)
        return params

    @staticmethod
    def _validate(params: dict[str, Any], schema: InputSchema) -> tuple[list[str], list[str]]:
        missing, errors = [], []
        types = {"string": str, "str": str, "integer": int, "int": int, "number": (int, float), "float": float, "boolean": bool, "bool": bool, "object": dict, "array": list, "list": list}
        for f in schema.fields:
            if f.required and (f.name not in params or params[f.name] in (None, "")): missing.append(f.name); continue
            if f.name not in params: continue
            value = params[f.name]
            expected = types.get(f.type.lower())
            if expected and (not isinstance(value, expected) or (f.type.lower() not in ("number", "float") and isinstance(value, bool))): errors.append(f"{f.name}: expected {f.type}")
            if f.enum is not None and value not in f.enum: errors.append(f"{f.name}: not in enum")
            if f.pattern and isinstance(value, str) and not re.fullmatch(f.pattern, value): errors.append(f"{f.name}: pattern mismatch")
        return missing, errors

    def resolve(self, task: Any, artifact_schema: Any, task_description: str = "", environment: Any = None, evidence: Any = None, allow_llm: bool = False) -> ResolutionResult:
        schema = self._schema(artifact_schema)
        params = self._deterministic(task)
        missing, errors = self._validate(params, schema)
        method = "deterministic"
        llm_error = None
        if allow_llm and self.chat_service and (missing or errors):
            messages = [{"role": "system", "content": "Return JSON only. Fill only missing or uncertain artifact input fields; do not invent values."}, {"role": "user", "content": f"Task: {task_description}\nSchema: {[f.__dict__ for f in schema.fields]}\nKnown: {params}\nMissing: {missing}"}]
            try:
                fn = getattr(self.chat_service, "complete_json", None) or getattr(self.chat_service, "chat_json")
                completion = fn(messages)
                if not isinstance(completion, dict): raise ValueError("LLM result is not an object")
                for key in missing + [f.name for f in schema.fields if f.name in errors]:
                    if key in completion: params[key] = completion[key]
                method = "deterministic+llm"
            except Exception as exc:
                llm_error = f"llm completion failed: {exc}"
        missing, errors = self._validate(params, schema)
        if llm_error: errors.append(llm_error)
        return ResolutionResult(params, 1.0 if method == "deterministic" and not missing and not errors else (0.8 if method == "deterministic+llm" and not missing and not errors else 0.4), method, missing, errors)
