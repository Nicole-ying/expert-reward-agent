from __future__ import annotations

from .schemas import Phase


class InvalidTransition(RuntimeError):
    pass


ALLOWED_TRANSITIONS: dict[Phase, set[Phase]] = {
    Phase.INIT: {Phase.PERCEIVE, Phase.FAILED},
    Phase.PERCEIVE: {Phase.CONTRACT, Phase.FAILED},
    Phase.CONTRACT: {Phase.PLAN, Phase.FAILED},
    Phase.PLAN: {Phase.GENERATE, Phase.FAILED},
    Phase.GENERATE: {Phase.VALIDATE, Phase.FAILED},
    Phase.VALIDATE: {Phase.GENERATE, Phase.TRAIN, Phase.FAILED},
    Phase.TRAIN: {Phase.TRAIN, Phase.EVALUATE, Phase.FAILED},
    Phase.EVALUATE: {Phase.DIAGNOSE, Phase.CONFIRM, Phase.STOPPED, Phase.FAILED},
    Phase.DIAGNOSE: {Phase.INTERVENE, Phase.STOPPED, Phase.FAILED},
    Phase.INTERVENE: {Phase.GENERATE, Phase.STOPPED, Phase.FAILED},
    Phase.CONFIRM: {Phase.STOPPED, Phase.DIAGNOSE, Phase.FAILED},
    Phase.STOPPED: set(),
    Phase.FAILED: set(),
}


def transition(current: Phase, target: Phase) -> Phase:
    if target not in ALLOWED_TRANSITIONS[current]:
        raise InvalidTransition(f"Invalid agent transition: {current.value} -> {target.value}")
    return target
