from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from ..agent.schemas import Phase


class ToolError(RuntimeError):
    pass


@dataclass
class ToolResult:
    call_id: str
    tool_name: str
    ok: bool
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class ToolDefinition:
    function: Callable[..., Any]
    allowed_phases: set[Phase]


class ToolRegistry:
    def __init__(self, audit_path: Path):
        self.audit_path = audit_path
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, name: str, function: Callable[..., Any], allowed_phases: set[Phase]) -> None:
        if name in self._tools:
            raise ToolError(f"Tool already registered: {name}")
        self._tools[name] = ToolDefinition(function, allowed_phases)

    def call(self, name: str, phase: Phase, **kwargs: Any) -> ToolResult:
        if name not in self._tools:
            raise ToolError(f"Unknown tool: {name}")
        definition = self._tools[name]
        if phase not in definition.allowed_phases:
            raise ToolError(f"Tool {name} is not allowed in phase {phase.value}")
        started = time.perf_counter()
        call_id = uuid.uuid4().hex[:12]
        try:
            output = definition.function(**kwargs)
            result = ToolResult(call_id, name, True, output=output)
        except Exception as exc:  # Tool failures are evidence, not search iterations.
            result = ToolResult(call_id, name, False, error=f"{type(exc).__name__}: {exc}")
        result.duration_ms = (time.perf_counter() - started) * 1000.0
        self._audit(phase, kwargs, result)
        return result

    def _audit(self, phase: Phase, arguments: dict[str, Any], result: ToolResult) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"phase": phase.value, "arguments": _safe(arguments), **asdict(result)}
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def _safe(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
