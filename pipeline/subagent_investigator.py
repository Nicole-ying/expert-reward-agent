"""Subagent Investigator — read-only training dynamics evidence scout.

Single-call design: all training data is pre-loaded into one prompt. The subagent
returns a structured JSON signal directly — no multi-turn tool calling, no function
calling protocol fragility. Stability over cleverness.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict


# ── Data readers ────────────────────────────────────────────────────────

def _read_training_summary(train_dir: str) -> str:
    p = Path(train_dir) / "training_summary.json"
    if not p.exists():
        return "(training_summary.json not found)"
    data = json.loads(p.read_text(encoding="utf-8"))
    external = data.get("external_eval", {})
    comp_summary = data.get("component_summary", {})
    comp_stats = comp_summary.get("component_stats", {})
    lines = [
        f"mean_eval_reward={external.get('mean_eval_reward','?')}",
        f"mean_episode_length={external.get('mean_episode_length','?')}",
        f"terminated={sum(1 for t in external.get('episode_terminated',[]) if t)}/{len(external.get('episode_terminated',[]) or [])}",
        "Components:",
    ]
    for name, s in sorted(comp_stats.items()):
        short = name.replace("component.", "")
        lines.append(
            f"  {short}: mean={s.get('mean',0):.4f} nonzero={s.get('nonzero_rate',0):.1%} "
            f"abs_share={s.get('abs_frac_of_total',0):.1%}"
        )
    return "\n".join(lines)


def _read_component_dynamics(train_dir: str) -> str:
    p = Path(train_dir) / "training_summary.json"
    if not p.exists():
        return "(no data)"
    data = json.loads(p.read_text(encoding="utf-8"))
    snapshots = data.get("monitor_snapshots", [])
    if not snapshots:
        return "(no monitor snapshots)"
    all_comps: Dict[str, list] = {}
    for snap in snapshots:
        for comp in snap.get("top_components", []):
            name = comp.get("name", "?")
            all_comps.setdefault(name, []).append({
                "steps": snap.get("steps", 0),
                "sum": comp.get("episode_sum_mean"),
                "active": comp.get("active_rate"),
                "share": comp.get("magnitude_share"),
            })
    lines = []
    for name, history in sorted(all_comps.items()):
        if len(history) < 2:
            continue
        first, last = history[0], history[-1]
        f_sum, l_sum = first.get("sum") or 0, last.get("sum") or 0
        f_act, l_act = first.get("active") or 0, last.get("active") or 0
        trend = "/\\" if l_sum > f_sum else "\\/" if l_sum < f_sum else "--"
        lines.append(
            f"  {name}: {f_sum:.3f}->{l_sum:.3f} {trend} "
            f"active {f_act:.0%}->{l_act:.0%} ({len(history)} checkpoints)"
        )
    return "\n".join(lines) if lines else "(no temporal data)"


def _read_feedback_md(train_dir: str) -> str:
    p = Path(train_dir) / "training_feedback.md"
    if not p.exists():
        return "(not found)"
    text = p.read_text(encoding="utf-8")
    return text[:5000] if len(text) > 5000 else text


def _read_previous_reward(reward_path: str) -> str:
    if not reward_path or not Path(reward_path).exists():
        return "(not available)"
    text = Path(reward_path).read_text(encoding="utf-8")
    return text[:5000] if len(text) > 5000 else text


# ── Main entry point ────────────────────────────────────────────────────

def run_investigator(
    *,
    train_dir: str,
    previous_reward_path: str = "",
    memory_path: str = "",
    client: Any = None,
    model: str = "deepseek-chat",
    max_tokens: int = 800,
) -> Dict[str, Any]:
    """Single-call evidence scout. Pre-loads all data, sends one prompt,
    returns a structured JSON signal. No function calling, no multi-turn."""

    # Pre-load all data
    summary = _read_training_summary(train_dir)
    dynamics = _read_component_dynamics(train_dir)
    feedback = _read_feedback_md(train_dir)
    prev_code = _read_previous_reward(previous_reward_path)

    system_prompt = (
        "You are an EVIDENCE SCOUT. Read the training data below and produce "
        "a compact JSON signal (~400-600 chars total across all fields).\n\n"
        "YOU DO NOT make decisions or suggest reward edits. You report what "
        "you OBSERVED in the data. The reward designer (a separate LLM) owns "
        "all decisions.\n\n"
        "Output valid JSON with these fields:\n"
        "- key_findings: 1-2 most salient facts (score, termination rate, ep len, reward scale)\n"
        "- component_anomalies: which components are dead, dominating (>70% share), "
        "  or self-cancelling (high magnitude but near-zero signed share)\n"
        "- training_dynamics: temporal trends across checkpoints (growth/decay of components, "
        "  scaffold→final drift, plateau detection)\n"
        "- signal_quality: dead gates, thresholds never crossed, coupling between signals, "
        "  missing attractor for desired behavior\n"
        "- confidence: low/medium/high (completeness of evidence, not certainty of diagnosis)\n\n"
        "RULES:\n"
        "- Every claim must reference a metric you see in the data.\n"
        "- Do NOT propose coefficients, operators, or edit strategies.\n"
        "- If data is sparse, say so and set confidence=low.\n"
        "- Be terse. Total JSON should be ~400-600 chars."
    )

    user_prompt = f"""Training data for the most recently completed reward iteration:

=== Training Summary ===
{summary}

=== Component Dynamics (temporal) ===
{dynamics}

=== Training Feedback (final policy) ===
{feedback[:3000]}

=== Previous Reward Code ===
{prev_code[:3000]}

Based on the evidence above, output your JSON signal. Remember: facts only, no recommendations."""

    signal = None
    tool_trace = []

    try:
        resp = client.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        content = (resp.choices[0].message.content or "").strip()

        # Parse JSON from response
        # Try direct parse first
        try:
            signal = json.loads(content)
        except json.JSONDecodeError:
            # Try extracting from code fences or braces
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                try:
                    signal = json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        # Validate required fields
        if signal and isinstance(signal, dict):
            required = ["key_findings", "component_anomalies", "training_dynamics", "signal_quality"]
            missing = [f for f in required if not signal.get(f)]
            if missing:
                tool_trace.append({"warning": f"missing fields: {missing}"})
                for f in missing:
                    signal[f] = "Not reported."
            if signal.get("confidence") not in ("low", "medium", "high"):
                signal["confidence"] = "low"
        else:
            signal = None
            tool_trace.append({"error": "could not parse JSON", "content_preview": content[:300]})

    except Exception as exc:
        tool_trace.append({"error": f"{type(exc).__name__}: {exc}"})

    # Build signal text
    signal_text = ""
    if signal:
        parts = [
            f"**Key Findings**: {signal.get('key_findings', '')}",
            f"**Component Anomalies**: {signal.get('component_anomalies', '')}",
            f"**Training Dynamics**: {signal.get('training_dynamics', '')}",
            f"**Signal Quality**: {signal.get('signal_quality', '')}",
            f"**Evidence Confidence**: `{signal.get('confidence', 'low')}`",
        ]
        signal_text = "\n\n".join(parts)

    return {
        "research_signal": signal,
        "research_signal_text": signal_text,
        "turns_used": 1 if signal else 0,
        "tool_trace": tool_trace,
    }
