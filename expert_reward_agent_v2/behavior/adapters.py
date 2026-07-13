from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Protocol

from ..agent.schemas import EvidenceKind


class BehaviorAdapter(Protocol):
    name: str
    priority: int

    def extract_metrics(self, trajectories: list[dict[str, Any]]) -> dict[str, Any]: ...

    def classify_outcomes(self, trajectories: list[dict[str, Any]]) -> dict[str, int]: ...

    def detect_exploits(self, trajectories: list[dict[str, Any]]) -> list[str]: ...


class GenericBehaviorAdapter:
    name = "generic"
    priority = 0

    def extract_metrics(self, trajectories: list[dict[str, Any]]) -> dict[str, Any]:
        lengths = [len(item.get("rewards", [])) for item in trajectories]
        returns = [sum(item.get("rewards", [])) for item in trajectories]
        action_magnitudes = [
            abs(float(action))
            for item in trajectories
            for action in item.get("actions", [])
            if isinstance(action, (int, float))
        ]
        return {
            "return_mean": fmean(returns) if returns else 0.0,
            "episode_length_mean": fmean(lengths) if lengths else 0.0,
            "action_abs_mean": fmean(action_magnitudes) if action_magnitudes else None,
        }

    def classify_outcomes(self, trajectories: list[dict[str, Any]]) -> dict[str, int]:
        counts = {"terminated": 0, "truncated": 0, "unknown": 0}
        for item in trajectories:
            if item.get("terminated"):
                counts["terminated"] += 1
            elif item.get("truncated"):
                counts["truncated"] += 1
            else:
                counts["unknown"] += 1
        return counts

    def detect_exploits(self, trajectories: list[dict[str, Any]]) -> list[str]:
        flags: list[str] = []
        for item in trajectories:
            rewards = item.get("rewards", [])
            actions = item.get("actions", [])
            if rewards and sum(rewards) > 0 and actions and all(action == actions[0] for action in actions):
                flags.append("positive_return_with_constant_action")
                break
        return flags


@dataclass
class GeneratedMetric:
    name: str
    operation: str
    field: str
    index: int | None
    evidence_kind: EvidenceKind
    confidence: float


class GeneratedBehaviorAdapter(GenericBehaviorAdapter):
    """Evaluates a restricted metric DSL; it never executes generated code."""

    name = "generated_dsl"
    priority = 10
    OPERATIONS = {"final", "mean", "abs_mean", "delta", "l2_final"}

    def __init__(self, spec: dict[str, Any]):
        self.metrics = [self._parse_metric(item) for item in spec.get("metrics", [])]

    def _parse_metric(self, item: dict[str, Any]) -> GeneratedMetric:
        operation = item["operation"]
        if operation not in self.OPERATIONS:
            raise ValueError(f"Unsupported behavior operation: {operation}")
        confidence = float(item.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        return GeneratedMetric(
            name=item["name"],
            operation=operation,
            field=item.get("field", "observations"),
            index=item.get("index"),
            evidence_kind=EvidenceKind(item.get("evidence_kind", "unknown")),
            confidence=confidence,
        )

    def extract_metrics(self, trajectories: list[dict[str, Any]]) -> dict[str, Any]:
        output = super().extract_metrics(trajectories)
        for metric in self.metrics:
            values = [self._evaluate(metric, trajectory) for trajectory in trajectories]
            numeric = [value for value in values if value is not None and math.isfinite(value)]
            output[metric.name] = fmean(numeric) if numeric else None
            output[f"{metric.name}__evidence"] = metric.evidence_kind.value
            output[f"{metric.name}__confidence"] = metric.confidence
        return output

    @staticmethod
    def _evaluate(metric: GeneratedMetric, trajectory: dict[str, Any]) -> float | None:
        sequence = trajectory.get(metric.field, [])
        if not sequence:
            return None
        values = sequence
        if metric.index is not None:
            values = [row[metric.index] for row in sequence if len(row) > metric.index]
        if not values:
            return None
        if metric.operation == "final":
            return float(values[-1])
        if metric.operation == "mean":
            return fmean(float(value) for value in values)
        if metric.operation == "abs_mean":
            return fmean(abs(float(value)) for value in values)
        if metric.operation == "delta":
            return float(values[-1]) - float(values[0])
        if metric.operation == "l2_final":
            return math.sqrt(sum(float(value) ** 2 for value in values[-1]))
        return None


class AdapterRegistry:
    def __init__(self):
        self._adapters: list[BehaviorAdapter] = []

    def register(self, adapter: BehaviorAdapter) -> None:
        self._adapters.append(adapter)

    def select(self) -> BehaviorAdapter:
        if not self._adapters:
            return GenericBehaviorAdapter()
        return max(self._adapters, key=lambda item: item.priority)
