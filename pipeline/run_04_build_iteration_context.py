"""Build iteration context for reward revision.

New flow (v2):
1. Read training_feedback + agent_memory + previous_code
2. Call analysis LLM → diagnostic JSON (failure_modes, hacking_risks, etc.)
3. Retrieve matched expert cards from reward_misalignment_cards.md
4. Look up skeleton suggestions from route_catalog for this task type
5. Assemble complete iteration_context.md
"""

import argparse
import json
from pathlib import Path

from .common import load_config, read_text, write_text, record_prompt, record_response
from llm_clients.deepseek_client import DeepSeekClient


FAILURE_MODE_NAMES = [
    "speed_penalty_dominance",
    "stability_penalty_dominance",
    "sparse_completion_proxy",
    "early_failure_or_crash",
    "high_reward_without_success",
    "goal_near_oscillation",
    "contact_reward_hacking",
    "generated_reward_negative_average",
]

HACKING_RISK_NAMES = [
    "speed_penalty_dominance",
    "stability_penalty_dominance",
    "sparse_completion_proxy",
    "early_failure_or_crash",
    "high_reward_without_success",
    "goal_near_oscillation",
    "contact_reward_hacking",
    "generated_reward_negative_average",
]


def extract_card(cards_md, card_id):
    marker = f"## {card_id}"
    start = cards_md.find(marker)
    if start < 0:
        return ""
    next_start = cards_md.find("\n## ", start + len(marker))
    if next_start < 0:
        return cards_md[start:].strip()
    return cards_md[start:next_start].strip()


def compress_card(block, max_chars=200):
    """Keep only signal+fix lines, discard full description."""
    if len(block) <= max_chars:
        return block
    lines = block.splitlines()
    kept = []
    for line in lines:
        s = line.strip()
        if s.startswith("- signal:") or s.startswith("- risk:") or s.startswith("- fix:"):
            kept.append(line)
    if kept:
        header = lines[0] if lines else ""
        return header + "\n" + "\n".join(kept)
    return block[:max_chars] + "..."


def retrieve_cards(cards_md, mode_names, top_k=4, max_chars_per_card=200):
    blocks = []
    seen = set()
    for name in mode_names:
        if name in seen:
            continue
        seen.add(name)
        block = extract_card(cards_md, name)
        if block:
            blocks.append(compress_card(block, max_chars=max_chars_per_card))
    return blocks[:max(1, int(top_k))]


def skeleton_signature(code_text):
    """Extract a short skeleton signature from reward code."""
    comps = []
    for line in code_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and any(
            kw in stripped.lower()
            for kw in ["component", "progress", "stability", "landing", "penalty",
                       "bonus", "shaping", "anchor", "distance", "speed", "energy",
                       "action", "terminal", "gated", "potential", "curriculum"]
        ):
            comps.append(stripped.lstrip("# ").strip())
    return " + ".join(comps[:8]) if comps else "unknown_skeleton"


def render_memory_table(memory_md):
    """Render a compact history table from reward_memory.md."""
    lines = []
    lines.append("| iter | score | best | skeleton_summary | trend |")
    lines.append("|------|-------|------|------------------|-------|")
    in_latest = False
    for line in memory_md.splitlines():
        s = line.strip()
        if s.startswith("### iter_"):
            in_latest = True
            iter_id = s.replace("###", "").strip()
            lines.append(f"| {iter_id} |")
        elif in_latest and s.startswith("- external_score:"):
            lines[-1] += f" {s.split(':')[1].strip()} |"
        elif in_latest and s.startswith("- best_score_so_far:"):
            lines[-1] += f" {s.split(':')[1].strip()} |"
        elif in_latest and s.startswith("- reward_structure:"):
            lines[-1] += f" {s.split(':', 1)[1].strip()} |"
        elif in_latest and s.startswith("- decision:"):
            lines[-1] += f" {s.split(':')[1].strip()} |"
        elif in_latest and s.startswith("---"):
            in_latest = False
    return "\n".join(lines)


def get_skeleton_suggestions(route_catalog_path, task_route_id):
    """Read route_catalog and return recommended skeletons for the task type."""
    try:
        data = json.loads(Path(route_catalog_path).read_text(encoding="utf-8"))
    except Exception:
        return []
    routes = data.get("routes", [])
    for route in routes:
        if route.get("route_id") == task_route_id:
            return route.get("recommended_skeletons", [])
    return []


def filter_skeleton_suggestions(skeletons, forbidden=None):
    """Remove skeletons blocked by environment constraints."""
    if forbidden is None:
        forbidden = {"terminal_success_reward", "terminal_failure_penalty"}
    return [s for s in skeletons if s not in forbidden]


def run_analysis_llm(feedback_md, memory_md, previous_code, system_prompt, config_path, expert_context_md="", best_code="", mock=False, analysis_dir=None):
    """Call analysis LLM and return diagnostic JSON dict."""
    cfg = load_config(config_path)
    llm_cfg = cfg["llm"]

    best_block = ""
    if best_code:
        best_block = f"""
# best_reward.py (historical highest score)
```python
{best_code}
```
"""

    user_prompt = f"""# training_feedback
{feedback_md}

# agent_memory
{memory_md}

# previous_reward.py (current, being analyzed)
```python
{previous_code}
```
{best_block}
# expert_knowledge_context
{expert_context_md}

# known_failure_modes
{', '.join(FAILURE_MODE_NAMES)}

# known_hacking_risks
{', '.join(HACKING_RISK_NAMES)}

Based on the evidence above, output a diagnostic JSON.
If best_reward.py is provided and scored higher, compare its coefficients to previous_reward.py
and identify exactly which changes caused the regression. Recommend whether to revert.
"""

    if mock:
        return {
            "failure_modes": ["early_failure_or_crash"],
            "hacking_risks": [],
            "component_analysis": {},
            "skeleton_assessment": {
                "current_skeleton": [],
                "iterations_on_this_skeleton": 1,
                "best_score_this_skeleton": 0,
                "stagnant": False,
                "skeleton_family": "unknown",
            },
            "recommended_action": "mix",
            "reasoning": "mock diagnosis",
        }

    client = DeepSeekClient(api_key_env=llm_cfg["api_key_env"], base_url=llm_cfg["base_url"])
    response = client.chat(
        model=llm_cfg["model_env"],
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=2048,
        json_mode=True,
    )
    if analysis_dir:
        record_prompt(analysis_dir, "04_analysis", system_prompt, user_prompt)
        record_response(analysis_dir, "04_analysis", response)

    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            return json.loads(match.group(0))
        return {}


def find_expert_context(seed_root):
    """Find the most recent expert_reward_context.md."""
    for d in sorted(Path(seed_root).glob("iter_*/generation/expert_reward_context.md"), reverse=True):
        return read_text(str(d))
    return ""


def build_context(
    train_run_dir,
    memory_path,
    cards_path,
    top_k=4,
    config_path="configs/env001_deepseek_rag.yaml",
    task_route_id="navigation_goal_reaching",
    route_catalog_path="knowledge_base/stage_generation/route_catalog_03.json",
    mock=False,
):
    train_dir = Path(train_run_dir)
    iter_dir = train_dir.parent  # e.g. iter_03/
    seed_root = iter_dir.parent  # e.g. seed_0/
    gen_dir = iter_dir / "generation"
    feedback_md = read_text(train_dir / "training_feedback.md")
    memory_md = read_text(memory_path) if Path(memory_path).exists() else ""
    cards_md = read_text(cards_path)

    # Previous reward code
    reward_files = sorted(gen_dir.glob("reward_v*.py"))
    previous_code = read_text(str(reward_files[-1])) if reward_files else ""

    # Expert knowledge context (from iter_01 or last fresh restart)
    expert_context_md = find_expert_context(str(seed_root))

    # Find best reward code for comparison
    best_code = ""
    best_dir = seed_root / "best"
    if (best_dir / "best_reward.py").exists():
        best_code = read_text(str(best_dir / "best_reward.py"))

    # Step 1: Analysis LLM
    analysis_prompt = read_text("prompts/04_analysis_prompt.md")
    diagnosis = run_analysis_llm(
        feedback_md, memory_md, previous_code, analysis_prompt, config_path,
        expert_context_md=expert_context_md, best_code=best_code, mock=mock, analysis_dir=str(gen_dir),
    )

    # Step 2: RAG cards
    matched_names = diagnosis.get("failure_modes", []) + diagnosis.get("hacking_risks", [])
    card_blocks = retrieve_cards(cards_md, matched_names, top_k=top_k)

    # Step 3: Write analysis_report.md (text) + iteration_cards.md
    rec_action = diagnosis.get("recommended_action", "mix")
    reasoning = diagnosis.get("reasoning", "")
    skel = diagnosis.get("skeleton_assessment", {})
    comps = diagnosis.get("component_analysis", {})

    report = []
    report.append(f"# Analysis Report\n")
    report.append(f"## Recommended Action: {rec_action}")
    report.append(f"{reasoning}\n")
    report.append(f"## Skeleton Status")
    report.append(f"- family: {skel.get('skeleton_family', '?')}")
    report.append(f"- stagnant: {skel.get('stagnant', False)}")
    report.append(f"- iterations_on_skeleton: {skel.get('iterations_on_this_skeleton', 1)}\n")
    report.append(f"## Component Analysis")
    for name, info in comps.items():
        report.append(f"- {name}: role={info.get('role','?')} dir={info.get('direction','?')} issue={info.get('issue','none')}")
    fm = diagnosis.get("failure_modes", [])
    hr = diagnosis.get("hacking_risks", [])
    report.append(f"\n## Detected Issues")
    if fm: report.append(f"- failure_modes: {', '.join(fm)}")
    if hr: report.append(f"- hacking_risks: {', '.join(hr)}")
    report.append("")
    analysis_text = "\n".join(report)
    write_text(str(gen_dir / "analysis_report.md"), analysis_text)
    write_text(str(gen_dir / "iteration_cards.md"), ("\n\n".join(card_blocks) if card_blocks else "(none)") + "\n")

    # Step 4: Memory table
    memory_table = render_memory_table(memory_md) if memory_md else "(no history)"

    # Step 5: Assemble iteration_context.md
    lines = []
    lines.append("# Iteration Context")
    lines.append("")
    lines.append("## Recommended Action")
    lines.append(f"**{rec_action}** — {reasoning}")
    lines.append("")
    lines.append("## Agent Memory")
    lines.append(memory_table)
    lines.append("")
    if card_blocks:
        lines.append("## Expert Cards")
        lines.append("\n\n".join(card_blocks))
        lines.append("")
    lines.append("## Training Evidence")
    lines.append(feedback_md)
    return "\n".join(lines).strip() + "\n", diagnosis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-run-dir", required=True)
    ap.add_argument("--memory", default="runs/env_001/memory/reward_memory.md")
    ap.add_argument("--cards", default="knowledge_base/iteration/reward_misalignment_cards.md")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/env001_deepseek_rag.yaml")
    ap.add_argument("--task-route-id", default="navigation_goal_reaching")
    ap.add_argument("--route-catalog", default="knowledge_base/stage_generation/route_catalog_03.json")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    context, diagnosis = build_context(
        train_run_dir=args.train_run_dir,
        memory_path=args.memory,
        cards_path=args.cards,
        top_k=args.top_k,
        config_path=args.config,
        task_route_id=args.task_route_id,
        route_catalog_path=args.route_catalog,
        mock=args.mock,
    )
    write_text(args.out, context)
    # Write raw diagnosis JSON so the orchestrator can extract new_lessons
    diag_path = str(Path(args.out).parent / "diagnosis.json")
    import json as _json
    write_text(diag_path, _json.dumps(diagnosis, ensure_ascii=False, indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
