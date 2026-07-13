from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


class WorkingMemory:
    def __init__(self, path: Path):
        self.path = path

    def save(self, payload: dict[str, Any]) -> None:
        _atomic_json(self.path, payload)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))


class JsonlMemory:
    def __init__(self, path: Path):
        self.path = path

    def append(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


class EpisodicMemory(JsonlMemory):
    """Append-only within-run intervention and outcome history."""


class CaseMemory(JsonlMemory):
    """Validated cross-run cases; never populated from unconfirmed attempts."""

    def query(self, terms: list[str], limit: int = 3) -> list[dict[str, Any]]:
        return _rank_records(self.all(), terms, limit)


class SemanticMemory:
    def __init__(self, cards_dir: Path):
        self.cards_dir = cards_dir

    def load(self) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        if not self.cards_dir.exists():
            return cards
        for path in sorted(self.cards_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            cards.extend(payload if isinstance(payload, list) else [payload])
        return cards

    def retrieve(self, terms: list[str], limit: int = 6) -> list[dict[str, Any]]:
        return _rank_records(self.load(), terms, limit)


def _rank_records(records: list[dict[str, Any]], terms: list[str], limit: int) -> list[dict[str, Any]]:
    normalized = {term.lower() for term in terms if term}
    scored: list[tuple[int, dict[str, Any]]] = []
    for record in records:
        text = json.dumps(record, ensure_ascii=False).lower()
        score = sum(1 for term in normalized if term in text)
        if score:
            scored.append((score, record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored[:limit]]
