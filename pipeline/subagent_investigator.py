"""Subagent Investigator — read-only training dynamics analyst.

This is the "agentic" bridge: a short-lived subagent that reads training outputs
and produces a compact research signal (~400-800 chars) to enrich the reflection
prompt. The subagent has tool access but cannot edit rewards or train.

Pattern borrowed from Agentic_CREATE's EvidenceAnalysisService, but radically
simplified for minimal integration into the expert framework.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from llm_clients.deepseek_client import DeepSeekClient


# ── Tools the subagent can call ──────────────────────────────────────────

def _read_training_summary(train_dir: str) -> str:
    """Read key metrics from training_summary.json."""
    p = Path(train_dir) / "training_summary.json"
    if not p.exists():
        return "(training_summary.json not found)"

    data = json.loads(p.read_text(encoding="utf-8"))
    external = data.get("external_eval", {})
    comp_summary = data.get("component_summary", {})
    comp_stats = comp_summary.get("component_stats", {})

    lines = [
        "## Training Summary",
        f"- mean_eval_reward: {external.get('mean_eval_reward', '?')}",
        f"- mean_episode_length: {external.get('mean_episode_length', '?')}",
        f"- std_eval_reward: {external.get('std_eval_reward', '?')}",
        f"- terminated_count: {sum(1 for t in external.get('episode_terminated', []) if t)}/{len(external.get('episode_terminated', []) or [])}",
        "",
        "### Component Breakdown (per-episode mean, activation rate, share)",
    ]
    for name, stats in sorted(comp_stats.items()):
        short = name.replace("component.", "")
        lines.append(
            f"- {short}: mean={stats.get('mean', 0):.4f}, "
            f"nonzero_rate={stats.get('nonzero_rate', 0):.1%}, "
            f"abs_share={stats.get('abs_frac_of_total', 0):.1%}"
        )
    return "\n".join(lines)


def _read_feedback_md(train_dir: str) -> str:
    """Read the training feedback markdown (full text, bounded)."""
    p = Path(train_dir) / "training_feedback.md"
    if not p.exists():
        return "(training_feedback.md not found)"
    text = p.read_text(encoding="utf-8")
    if len(text) > 6000:
        text = text[:6000] + "\n\n...(truncated for subagent)"
    return text


def _read_component_dynamics(train_dir: str) -> str:
    """Extract per-component time-evolution from training_summary.json.

    Reads monitor snapshots to understand if components are growing, dying, or
    oscillating across training.
    """
    p = Path(train_dir) / "training_summary.json"
    if not p.exists():
        return "(no training_summary.json)"

    data = json.loads(p.read_text(encoding="utf-8"))
    snapshots = data.get("monitor_snapshots", [])
    if not snapshots:
        return "(no monitor snapshots — training may not have completed)"

    # Extract all component names across snapshots
    all_components: Dict[str, List[Dict[str, Any]]] = {}
    for snap in snapshots:
        for comp in snap.get("top_components", []):
            name = comp.get("name", "?")
            all_components.setdefault(name, []).append({
                "steps": snap.get("steps", 0),
                "episode_sum_mean": comp.get("episode_sum_mean"),
                "active_rate": comp.get("active_rate"),
                "magnitude_share": comp.get("magnitude_share"),
            })

    lines = ["## Component Dynamics (across training snapshots)"]
    for name, history in sorted(all_components.items()):
        if len(history) < 2:
            continue
        first = history[0]
        last = history[-1]
        first_sum = first.get("episode_sum_mean") or 0
        last_sum = last.get("episode_sum_mean") or 0
        first_active = first.get("active_rate") or 0
        last_active = last.get("active_rate") or 0
        trend = "↗" if last_sum > first_sum else "↘" if last_sum < first_sum else "→"
        lines.append(
            f"- {name}: {first_sum:.3f}→{last_sum:.3f} {trend} "
            f"active {first_active:.0%}→{last_active:.0%} "
            f"({len(history)} checkpoints)"
        )

    # Add reward error stats if present
    errors = data.get("reward_errors", {})
    if errors:
        lines.append(f"\n### Reward Errors\nerror_count={errors.get('error_count', 0)}, "
                     f"last_error={errors.get('last_error', 'none')}")

    return "\n".join(lines)


def _read_previous_reward(reward_path: str) -> str:
    """Read the previous reward code, bounded."""
    if not reward_path or not Path(reward_path).exists():
        return "(no previous reward code)"
    text = Path(reward_path).read_text(encoding="utf-8")
    if len(text) > 4000:
        text = text[:4000] + "\n# ...(truncated)"
    return text


def _read_memory_table(memory_path: str) -> str:
    """Read the reward memory table, bounded."""
    if not memory_path or not Path(memory_path).exists():
        return "(no reward memory)"
    text = Path(memory_path).read_text(encoding="utf-8")
    if len(text) > 3000:
        # Keep header + last 15 lines
        lines = text.splitlines()
        text = "\n".join(lines[:3] + ["..."] + lines[-15:])
    return text


# ── Tool definitions (OpenAI-compatible function calling format) ──────────

INVESTIGATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_training_summary",
            "description": (
                "Read training_summary.json for this iteration: external evaluation "
                "score, per-component means/activation/share, and episode outcomes. "
                "This is the primary quantitative evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_component_dynamics",
            "description": (
                "Read time-series component dynamics across training checkpoints. "
                "Shows which components grew, died, or oscillated. Use to diagnose "
                "learning scaffolds, vanishing signals, and late-training drift."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_training_feedback",
            "description": (
                "Read the human-readable training_feedback.md with final policy "
                "outcome, component composition table, and episode distribution."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_previous_reward",
            "description": (
                "Read the previous iteration's reward function source code. "
                "The subagent must inspect this to understand what components "
                "exist and how they connect to the observed training dynamics."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_research_signal",
            "description": (
                "Submit the final research signal. This is the ONLY valid final "
                "action. The signal must be compact (~400-800 chars), evidence-cited, "
                "and structured: (1) key training dynamics findings, (2) component "
                "anomalies, (3) one strongest mechanism hypothesis, (4) bounded "
                "decision implication (keep/edit/rebuild which component)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key_findings": {
                        "type": "string",
                        "maxLength": 300,
                        "description": "1-2 most salient training dynamics facts with metrics.",
                    },
                    "component_anomalies": {
                        "type": "string",
                        "maxLength": 250,
                        "description": "Components that are vanishing, dominating, or misbehaving.",
                    },
                    "mechanism_hypothesis": {
                        "type": "string",
                        "maxLength": 250,
                        "description": "One falsifiable causal claim about what's limiting learning.",
                    },
                    "decision_implication": {
                        "type": "string",
                        "maxLength": 250,
                        "description": (
                            "Bounded recommendation: keep/patch/redesign which "
                            "component, with one-sentence rationale."
                        ),
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "How well the evidence supports the hypothesis.",
                    },
                },
                "required": [
                    "key_findings",
                    "component_anomalies",
                    "mechanism_hypothesis",
                    "decision_implication",
                    "confidence",
                ],
            },
        },
    },
]


def _execute_tool(tool_name: str, tool_args: Dict[str, Any], ctx: Dict[str, Any]) -> str:
    """Execute a read-only tool and return the result string."""
    train_dir = ctx.get("train_dir", "")
    if tool_name == "inspect_training_summary":
        return _read_training_summary(train_dir)
    elif tool_name == "inspect_component_dynamics":
        return _read_component_dynamics(train_dir)
    elif tool_name == "inspect_training_feedback":
        return _read_feedback_md(train_dir)
    elif tool_name == "inspect_previous_reward":
        return _read_previous_reward(ctx.get("previous_reward_path", ""))
    elif tool_name == "submit_research_signal":
        # This is handled by the main loop, not executed as a tool
        return json.dumps(tool_args, ensure_ascii=False)
    else:
        return f"Unknown tool: {tool_name}"


def _validate_signal(signal: Dict[str, Any]) -> List[str]:
    """Validate the research signal has all required fields with content."""
    errors = []
    for field in ("key_findings", "component_anomalies", "mechanism_hypothesis", "decision_implication"):
        if not signal.get(field):
            errors.append(f"missing required field: {field}")
    if signal.get("confidence") not in ("low", "medium", "high"):
        errors.append("confidence must be low/medium/high")
    # Total signal should be compact
    total = sum(len(str(signal.get(f, ""))) for f in signal)
    if total > 1500:
        errors.append(f"signal too long ({total} chars); target ~800 max")
    return errors


def run_investigator(
    *,
    train_dir: str,
    previous_reward_path: str = "",
    memory_path: str = "",
    client: Any = None,
    model: str = "deepseek-chat",
    max_turns: int = 3,
    max_tokens: int = 1200,
) -> Dict[str, Any]:
    """Run the read-only subagent investigator.

    Returns a dict with:
      - research_signal: the structured signal dict (or None if failed)
      - research_signal_text: compact markdown rendering
      - turns_used: how many LLM turns were consumed
      - tool_trace: list of tool calls made
    """
    ctx = {
        "train_dir": train_dir,
        "previous_reward_path": previous_reward_path,
        "memory_path": memory_path,
    }

    system_prompt = (
        "You are a read-only training dynamics investigator. Your job is to read "
        "training outputs and produce a compact evidence diagnosis. You cannot edit "
        "rewards or train policies.\n\n"
        "WORKFLOW:\n"
        "1. Call inspect_training_summary to get the quantitative baseline.\n"
        "2. Call inspect_component_dynamics if you need time-series evidence.\n"
        "3. Call inspect_previous_reward if you need to connect dynamics to code.\n"
        "4. Call submit_research_signal ONCE with your final diagnosis.\n\n"
        "RULES:\n"
        "- Make at most 3 function calls total (including submit).\n"
        "- Every factual claim must cite a metric you observed from the tools.\n"
        "- Do NOT propose reward formulas or specific coefficients.\n"
        "- The signal must be compact: Chinese or English, ~400-800 chars total.\n"
        "- If evidence is insufficient, say so and set confidence=low."
    )

    user_context = (
        f"The training run at `{train_dir}` has completed. Inspect its outputs "
        f"and produce a research signal that will help the reward designer make "
        f"one evidence-based edit decision.\n\n"
        f"Previous reward: {previous_reward_path or '(initial generation, no previous)'}\n"
        f"Memory: {memory_path or '(no memory yet)'}"
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_context},
    ]

    tool_trace: List[Dict[str, Any]] = []
    signal: Dict[str, Any] | None = None
    submit_received = False
    queries_seen: set[str] = set()
    repair_attempts = 0

    for turn in range(1, max_turns + 3):  # +extra for repair turns
        final_only = turn > max_turns
        tools = INVESTIGATOR_TOOLS
        if final_only:
            # On repair turns, only allow submit
            tools = [
                t for t in INVESTIGATOR_TOOLS
                if t["function"]["name"] == "submit_research_signal"
            ]

        try:
            resp = client.completion(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            tool_trace.append({"turn": turn, "error": f"{type(exc).__name__}: {exc}"})
            break

        choice = resp.choices[0]
        message = choice.message
        content = (message.content or "").strip()
        tool_calls = list(message.tool_calls or [])

        # Handle empty response
        if not content and not tool_calls:
            messages.append({"role": "assistant", "content": content or "(empty)"})
            messages.append({
                "role": "user",
                "content": "Call inspect_* tools to read training data, then submit_research_signal.",
            })
            continue

        # Try to parse signal from text content (DeepSeek sometimes embeds function
        # calls in text rather than using the tool_calls protocol)
        if not tool_calls and "submit_research_signal" in content.lower():
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                try:
                    candidate = json.loads(match.group(0))
                    if isinstance(candidate, dict) and "key_findings" in candidate:
                        signal = candidate
                        submit_received = True
                        tool_trace.append({
                            "turn": turn,
                            "parsed_from_text": True,
                            "signal": signal,
                        })
                        break
                except json.JSONDecodeError:
                    pass

        if not tool_calls:
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    "You must use function calls. Call inspect_training_summary, "
                    "inspect_component_dynamics, inspect_training_feedback, or "
                    "inspect_previous_reward to read data. Then call "
                    "submit_research_signal with your diagnosis."
                ),
            })
            continue

        # Process tool calls.  DeepSeek requires: assistant (with tool_calls)
        # BEFORE tool messages. Collect results first, then emit in order.
        pending_results: list[tuple[Any, str, str]] = []  # (tc, tool_name, result_text)

        for tc in tool_calls[:2]:  # max 2 calls per turn
            name = tc.function.name
            raw_args = tc.function.arguments or "{}"
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}
            else:
                args = raw_args or {}

            tool_trace.append({
                "turn": turn,
                "tool": name,
                "args": args,
            })

            if name == "submit_research_signal":
                signal = args
                submit_received = True
                errors = _validate_signal(signal)
                if errors:
                    tool_trace[-1]["validation_errors"] = errors
                    repair_attempts += 1
                    if repair_attempts > 2:
                        signal["confidence"] = "low"
                        signal.setdefault("key_findings", "Validation failed after repairs")
                        signal.setdefault("component_anomalies", "Unable to determine")
                        signal.setdefault("mechanism_hypothesis", "Insufficient evidence")
                        signal.setdefault("decision_implication", "Review training feedback manually")
                        tool_trace[-1]["accepted_with_errors"] = True
                        break
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Signal validation failed: {'; '.join(errors)}. "
                            "Fix the issues and re-submit. Make signal compact "
                            "~400-800 chars total."
                        ),
                    })
                    signal = None
                    submit_received = False
                    continue
                tool_trace[-1]["signal_valid"] = True
                break

            # Execute read-only tool
            sig = f"{name}:{json.dumps(args, sort_keys=True)}"
            if sig in queries_seen:
                result = "(duplicate query — same arguments as earlier)"
                tool_trace[-1]["cached"] = True
            else:
                queries_seen.add(sig)
                result = _execute_tool(name, args, ctx)
                tool_trace[-1]["result_len"] = len(result)

            pending_results.append((tc, name, result))

        if submit_received and signal:
            break

        # Emit assistant + tool messages in required order
        if pending_results:
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": content or None,
                "tool_calls": [],
            }
            for idx, (tc, name, result) in enumerate(pending_results):
                assistant_msg["tool_calls"].append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": tc.function.arguments,
                    },
                })
            messages.append(assistant_msg)
            for idx, (tc, name, result) in enumerate(pending_results):
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

    # Build compact signal text
    signal_text = ""
    if signal:
        parts = [
            f"**Key Findings**: {signal.get('key_findings', '')}",
            f"**Component Anomalies**: {signal.get('component_anomalies', '')}",
            f"**Mechanism Hypothesis**: {signal.get('mechanism_hypothesis', '')}",
            f"**Decision Implication**: {signal.get('decision_implication', '')}",
            f"**Confidence**: `{signal.get('confidence', 'low')}`",
        ]
        signal_text = "\n\n".join(parts)

    return {
        "research_signal": signal,
        "research_signal_text": signal_text,
        "turns_used": len([t for t in tool_trace if "tool" in t]),
        "tool_trace": tool_trace,
    }
