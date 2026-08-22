"""Retrieval tools for the reflection agent — simple keyword-based, no ChromaDB needed."""

import json
import re
from pathlib import Path

_techniques_text = None
_skeletons_data = None
_transformations_data = None


def _load_techniques():
    global _techniques_text
    if _techniques_text is None:
        p = Path("knowledge_base/iteration/reward_design_techniques.md")
        _techniques_text = p.read_text(encoding="utf-8") if p.exists() else ""
    return _techniques_text


def _load_skeletons():
    global _skeletons_data
    if _skeletons_data is None:
        p = Path("knowledge_base/iteration/skeleton_details.json")
        _skeletons_data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return _skeletons_data


def _load_transformations():
    global _transformations_data
    if _transformations_data is None:
        path = Path("knowledge_base/iteration/reward_transformations.json")
        _transformations_data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _transformations_data


def _split_sections(md_text):
    """Split markdown into per-##-heading sections."""
    sections = []
    current = []
    for line in md_text.splitlines():
        if line.startswith("## ") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return sections


def search_reward_design_knowledge(query: str) -> str:
    """Search the technique library for matching entries. Returns top 2 matches."""
    text = _load_techniques()
    sections = _split_sections(text)
    if not sections:
        return "(技法库为空)"
    query_lower = query.lower()
    scored = []
    for sec in sections:
        sec_lower = sec.lower()
        # Score: keyword match count in section
        keywords = re.findall(r"[a-z_]{3,}", query_lower)
        score = sum(1 for kw in keywords if kw in sec_lower)
        # Bonus for heading match
        heading = sec.split("\n")[0].lower() if sec else ""
        score += sum(2 for kw in keywords if kw in heading)
        if score > 0:
            scored.append((score, sec))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored or scored[0][0] < 2:
        return "No sufficiently relevant knowledge card was found. Do not treat retrieval as support for the current hypothesis."
    # Return top 2, keep it compact
    results = []
    for _, sec in scored[:2]:
        # Extract just the heading + 症状 + 修复 lines
        lines = sec.splitlines()
        compact = []
        for line in lines:
            s = line.strip()
            if s.startswith("## ") or s.startswith("- 症状") or s.startswith("- 修复") or s.startswith("- 原理"):
                compact.append(s)
        results.append("\n".join(compact) if compact else lines[0])
    warning = (
        "NOTE: These are legacy symptom heuristics. Any numeric ratio, coefficient, or "
        "threshold is a candidate starting point, not a universal decision rule. "
        "Current final-policy evidence and transformation reasoning take priority.\n\n"
    )
    return warning + "\n---\n".join(results)


def get_reward_transformation(query: str) -> str:
    """Retrieve general reward-structure transformations by diagnosis or operator name."""
    transformations = _load_transformations()
    if not transformations:
        return "(reward transformation library is empty)"
    keywords = set(re.findall(r"[a-z_]{3,}", query.lower()))
    scored = []
    for name, card in transformations.items():
        searchable = " ".join([name, *card.values()]).lower()
        score = sum(2 if keyword in name else 1 for keyword in keywords if keyword in searchable)
        if name.lower() in query.lower() or query.lower() in name.lower():
            score += 5
        scored.append((score, name, card))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [item for item in scored if item[0] > 0][:3] or scored[:3]
    results = []
    for _, name, card in selected:
        results.append(
            f"## {name}\n"
            f"- diagnosis: {card['diagnosis']}\n"
            f"- transform: {card['transform']}\n"
            f"- why: {card['why']}\n"
            f"- risks: {card['risks']}\n"
            f"- verify_next_round: {card['verify']}"
        )
    return "\n---\n".join(results)


def get_skeleton_detail(skeleton_name: str) -> str:
    """Get the math form, rationale, pitfalls, and usage of a skeleton."""
    skeletons = _load_skeletons()
    sk = skeletons.get(skeleton_name)
    if not sk:
        # Try fuzzy match
        for key in skeletons:
            if skeleton_name.lower() in key.lower():
                sk = skeletons[key]
                skeleton_name = key
                break
    if not sk:
        return f"骨架 '{skeleton_name}' 未在 skeleton_details.json 中找到。"
    return (
        "NOTE: Skeleton coefficients and ratio ranges are legacy starting points, not universal constraints.\n"
        f"## {skeleton_name}\n"
        f"- 数学形态: {sk['math_form']}\n"
        f"- 设计原理: {sk['design_rationale']}\n"
        f"- 常见陷阱: {sk['pitfalls']}\n"
        f"- 推荐配合: {sk['recommended_with']}"
    )


# ── Context-aware tools: let the reflection LLM read its own experiment history ──

_context = {}  # {run_root, prefix, seed, current_iter}


def set_reflection_context(run_root, prefix, seed, current_iter):
    """Set the experiment context for history-reading tools."""
    _context["run_root"] = run_root
    _context["prefix"] = prefix
    _context["seed"] = seed
    _context["current_iter"] = current_iter


def read_memory() -> str:
    """Read the full reward memory table — all past iterations at a glance.

    Shows iter, skeleton, score, best, delta, len, key_signal, action for every round.
    Use this FIRST to get the big picture before diving into specifics.
    """
    if not _context:
        return "(context not set)"
    base = Path(_context["run_root"]) / _context["prefix"] / f"seed_{_context['seed']}"
    mem_path = base / "memory" / "reward_memory.md"
    if not mem_path.exists():
        return "(no memory file yet)"
    return mem_path.read_text(encoding="utf-8")


def read_training_feedback(iteration: int) -> str:
    """Read the training feedback (eval_result.json) for a specific past iteration.

    Returns score, len, termination breakdown, and per-component ep_sum/active_rate/shares.
    Use this to get exact numbers when the memory table summary isn't enough.
    """
    if not _context:
        return "(context not set)"
    import json
    base = Path(_context["run_root"]) / _context["prefix"] / f"seed_{_context['seed']}"
    ef = base / f"iter_{iteration:02d}" / "training" / "eval_result.json"
    if not ef.exists():
        return f"(no eval_result for iteration {iteration})"
    ev = json.loads(ef.read_text(encoding="utf-8"))
    comps = ev.get("final_policy_component_evaluation", {})
    lines = [
        f"=== Iteration {iteration} training feedback ===",
        f"score: {ev['mean_eval_reward']:.2f}",
        f"len: {ev['mean_episode_length']:.1f}",
        f"terminated: {ev['termination_breakdown'].get('terminated', '?')}/{ev.get('eval_episodes', '?')}",
        f"",
        f"Components:",
    ]
    for name, c in comps.items():
        lines.append(
            f"  {name}: ep_sum={c.get('episode_sum_mean', 0):.3f}, "
            f"active={c.get('active_rate', 0)*100:.1f}%, "
            f"magnitude_share={c.get('magnitude_share', 0)*100:.1f}%, "
            f"signed_share={c.get('signed_contribution_share', 0)*100:.1f}%"
        )
    return "\n".join(lines)


def read_reward_code(iteration: int) -> str:
    """Read the reward function code from a specific past iteration.

    Use this to compare what changed between two iterations at the code level.
    """
    if not _context:
        return "(context not set)"
    base = Path(_context["run_root"]) / _context["prefix"] / f"seed_{_context['seed']}"
    rp = base / f"iter_{iteration:02d}" / "generation" / f"reward_v{iteration}.py"
    if not rp.exists():
        return f"(no reward code for iteration {iteration})"
    return rp.read_text(encoding="utf-8")


def read_past_reflection(iteration: int) -> str:
    """Read the reflection agent's own response from a past iteration.

    Returns the evidence, behavior_diagnosis, selected_level, and selected_intervention
    fields — exactly what you decided and why at that round.
    """
    if not _context:
        return "(context not set)"
    base = Path(_context["run_root"]) / _context["prefix"] / f"seed_{_context['seed']}"
    resp_path = base / f"iter_{iteration:02d}" / "generation" / "response_records" / "agent_reflection.md"
    if not resp_path.exists():
        return f"(no reflection record for iteration {iteration})"
    text = resp_path.read_text(encoding="utf-8")
    # Extract the key diagnostic fields
    fields = {}
    for key in ["evidence", "behavior_diagnosis", "selected_level", "selected_intervention"]:
        m = re.search(rf'(?:\d+\.\s*)?`?{key}`?\s*[：:]\s*(.+?)(?=\n\d+\.\s*`|\n`|\n```|\n\Z)', text, re.DOTALL)
        if m:
            fields[key] = m.group(1).strip()[:300]
    if not fields:
        return f"(iteration {iteration} reflection: could not parse fields)\nFirst 400 chars:\n{text[:400]}"
    lines = [f"=== Your reflection at iteration {iteration} ==="]
    for key in ["evidence", "behavior_diagnosis", "selected_level", "selected_intervention"]:
        if key in fields:
            lines.append(f"\n{key}: {fields[key]}")
    return "\n".join(lines)


def get_component_history(component_name: str) -> str:
    """Get per-iteration values of a specific component across all past iterations.

    Shows ep_sum_mean and active_rate for each past iteration, helping you spot:
    - Dead→alive transitions (active_rate jumping from <2% to >10%)
    - Magnitude jumps (ep_sum_mean suddenly 10x larger)
    - Components that just started working and should be PROTECTED not replaced.
    """
    if not _context:
        return "(context not set)"
    import json
    base = Path(_context["run_root"]) / _context["prefix"] / f"seed_{_context['seed']}"
    lines = [f"=== History of component '{component_name}' ===", ""]
    found = False
    for it in range(1, _context["current_iter"]):
        ef = base / f"iter_{it:02d}" / "training" / "eval_result.json"
        if not ef.exists():
            continue
        ev = json.loads(ef.read_text(encoding="utf-8"))
        comps = ev.get("final_policy_component_evaluation", {})
        if component_name in comps:
            c = comps[component_name]
            found = True
            lines.append(
                f"  iter_{it:02d}: ep_sum_mean={c.get('episode_sum_mean', 0):.3f}, "
                f"active_rate={c.get('active_rate', 0)*100:.1f}%, "
                f"signed_share={c.get('signed_contribution_share', 0)*100:.1f}%, "
                f"magnitude_share={c.get('magnitude_share', 0)*100:.1f}%"
            )
    if not found:
        # Try fuzzy search — component names change across iterations
        alt_names = set()
        for it in range(1, _context["current_iter"]):
            ef = base / f"iter_{it:02d}" / "training" / "eval_result.json"
            if ef.exists():
                ev = json.loads(ef.read_text(encoding="utf-8"))
                comps = ev.get("final_policy_component_evaluation", {})
                alt_names.update(comps.keys())
        if alt_names:
            lines.append(f"(component '{component_name}' not found in any past iteration)")
            lines.append(f"Available component names in history: {', '.join(sorted(alt_names))}")
        else:
            lines.append(f"(component '{component_name}' not found, no history available)")
        return "\n".join(lines)
    lines.append("")
    # Detect patterns
    vals = []
    for it in range(1, _context["current_iter"]):
        ef = base / f"iter_{it:02d}" / "training" / "eval_result.json"
        if ef.exists():
            ev = json.loads(ef.read_text(encoding="utf-8"))
            c = ev.get("final_policy_component_evaluation", {}).get(component_name, {})
            vals.append(c.get("episode_sum_mean", 0) if c else 0)
    if len(vals) >= 2 and vals[-1] > 0:
        prev_max = max(vals[:-1]) if vals[:-1] else 0
        if prev_max > 0 and vals[-1] > prev_max * 5:
            lines.append(
                f"⚠️ WARNING: {component_name} just jumped from ~{prev_max:.1f} to {vals[-1]:.1f} "
                f"(>{int(vals[-1]/max(prev_max,0.001))}x increase). "
                f"If magnitude_share > 90%, this component may be exploited — fix IT, don't protect it."
            )
    return "\n".join(lines)


def read_checkpoint_trend(iteration: int = None) -> str:
    """Read checkpoint evaluation data showing per-component trends during training.

    If iteration is None, reads the most recent (previous) iteration's checkpoint data.
    Each checkpoint captures component ep_sum_mean and active_rate at that training stage.

    Use this to cross-validate your diagnosis:
    - If contact's active_rate surges while velocity_damping's share drops, exploit is from
      speed threshold being too loose, not distance threshold.
    - If a component is 0 at all checkpoints, it's dead (no gradient).
    - If score peaks mid-training then drops, the reward is being exploited.
    """
    if not _context:
        return "(context not set)"
    import json
    base = Path(_context["run_root"]) / _context["prefix"] / f"seed_{_context['seed']}"
    if iteration is None:
        iteration = _context["current_iter"] - 1
    ckpt_path = base / f"iter_{iteration:02d}" / "training" / "checkpoint_evals.json"
    if not ckpt_path.exists():
        return f"(no checkpoint data for iteration {iteration})"
    ckpt_raw = json.loads(ckpt_path.read_text(encoding="utf-8"))
    if not ckpt_raw:
        return f"(empty checkpoint data for iteration {iteration})"
    # Collect component names
    all_comp_names = set()
    for r in ckpt_raw:
        comps = r.get("components", {})
        all_comp_names.update(comps.keys())
    lines = [f"=== Checkpoint trend for iteration {iteration} ===", ""]
    # Overall score row
    pcts = [str(r["pct"]) + "%" for r in ckpt_raw]
    scores = [f"{r.get('score_mean', 0):.1f}" for r in ckpt_raw]
    lines.append("score:  " + "  ".join(f"cp{p}={s}" for p, s in zip(pcts, scores)))
    lines.append("")
    # Per-component ep_sum trend
    for name in sorted(all_comp_names):
        vals = []
        for r in ckpt_raw:
            c = r.get("components", {}).get(name, {})
            v = c.get("episode_sum_mean", 0) if c else 0
            vals.append(f"{v:.3f}")
        line = f"{name} (ep_sum):  " + "  ".join(f"cp{p}={v}" for p, v in zip(pcts, vals))
        lines.append(line)
    lines.append("")
    # Per-component active_rate trend
    for name in sorted(all_comp_names):
        vals = []
        for r in ckpt_raw:
            c = r.get("components", {}).get(name, {})
            v = (c.get("active_rate", 0) or 0) * 100
            vals.append(f"{v:.1f}%")
        line = f"{name} (active%):  " + "  ".join(f"cp{p}={v}" for p, v in zip(pcts, vals))
        lines.append(line)
    return "\n".join(lines)


def read_environment_card() -> str:
    """Read the environment card — task goal, observation space, action space, termination conditions.

    Use this when you need to understand what signals are available for reward design,
    or when you need to verify which observation fields you can use.
    """
    if not _context:
        return "(context not set)"
    base = Path(_context["run_root"]) / _context["prefix"] / f"seed_{_context['seed']}"
    # Environment card is typically in iter_01/generation/
    card_path = base / "iter_01" / "generation" / "environment_card.md"
    if not card_path.exists():
        return "(no environment card found)"
    return card_path.read_text(encoding="utf-8")
