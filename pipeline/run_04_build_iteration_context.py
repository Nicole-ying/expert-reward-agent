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

from .common import load_config, read_text, write_text
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


def retrieve_cards(cards_md, mode_names, top_k=4):
    blocks = []
    seen = set()
    for name in mode_names:
        if name in seen:
            continue
        seen.add(name)
        block = extract_card(cards_md, name)
        if block:
            blocks.append(block)
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


def run_analysis_llm(feedback_md, memory_md, previous_code, system_prompt, config_path, mock=False):
    """Call analysis LLM and return diagnostic JSON dict."""
    cfg = load_config(config_path)
    llm_cfg = cfg["llm"]

    user_prompt = f"""# training_feedback
{feedback_md}

# agent_memory
{memory_md}

# previous_reward.py
```python
{previous_code}
```

# failure_mode_names
{json.dumps(FAILURE_MODE_NAMES, ensure_ascii=False)}

# hacking_risk_names
{json.dumps(HACKING_RISK_NAMES, ensure_ascii=False)}

Output ONLY the JSON. No markdown, no code block, no extra text.
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
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            return json.loads(match.group(0))
        return {}


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
    feedback_md = read_text(train_dir / "training_feedback.md")
    memory_md = read_text(memory_path) if Path(memory_path).exists() else ""
    cards_md = read_text(cards_path)

    # Read previous reward code for analysis
    reward_path = train_dir.parent.parent / "generation"
    reward_files = sorted(reward_path.glob("reward_v*.py"))
    previous_code = ""
    if reward_files:
        previous_code = read_text(reward_files[-1])

    # Step 1: Call analysis LLM
    analysis_prompt = read_text("prompts/04_analysis_prompt.md")
    diagnosis = run_analysis_llm(
        feedback_md, memory_md, previous_code, analysis_prompt, config_path, mock=mock
    )

    # Step 2: Retrieve expert cards based on diagnosis
    matched_names = diagnosis.get("failure_modes", []) + diagnosis.get("hacking_risks", [])
    card_blocks = retrieve_cards(cards_md, matched_names, top_k=top_k)

    # Step 3: Get skeleton suggestions from route_catalog
    all_suggestions = get_skeleton_suggestions(route_catalog_path, task_route_id)
    suggestions = filter_skeleton_suggestions(all_suggestions)
    skeleton_family = diagnosis.get("skeleton_assessment", {}).get("skeleton_family", "")

    # Step 4: Render memory table
    memory_table = render_memory_table(memory_md) if memory_md else "(no history)"

    # Step 5: Assemble context
    lines = []
    lines.append("# Iteration Context for Reward Revision")
    lines.append("")
    lines.append("## Agent Memory (history)")
    lines.append("")
    lines.append(memory_table)
    lines.append("")
    lines.append("## Analysis Report")
    lines.append("")
    lines.append(f"```json\n{json.dumps(diagnosis, ensure_ascii=False, indent=2)}\n```")
    lines.append("")
    lines.append("## Training Feedback")
    lines.append("")
    lines.append(feedback_md)
    lines.append("")
    lines.append("## Matched Expert Cards")
    lines.append("")
    if card_blocks:
        lines.append("\n\n".join(card_blocks))
    else:
        lines.append("- No specific card matched. Consider broader exploration.")
    lines.append("")
    lines.append("## Skeleton Suggestions (from expert knowledge base)")
    lines.append("")
    if suggestions:
        lines.append(f"- Current skeleton family: {skeleton_family or 'unknown'}")
        lines.append(f"- Task type: {task_route_id}")
        lines.append(f"- Expert-recommended skeletons not yet tried: {', '.join(suggestions)}")
    else:
        lines.append("- No specific skeleton suggestions available.")
    lines.append("")
    lines.append("## Revision Boundary")
    lines.append("")
    lines.append("- Revise the previous reward instead of generating from scratch.")
    lines.append("- Keep the function signature unchanged.")
    lines.append("- Do not use original_reward or unavailable info fields.")
    lines.append("- Do not add terminal success/failure rewards without explicit signals.")
    return "\n".join(lines).strip() + "\n"


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

    context = build_context(
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
    print(args.out)


if __name__ == "__main__":
    main()
