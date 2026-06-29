import argparse
from pathlib import Path


CARD_IDS = [
    "speed_penalty_dominance",
    "stability_penalty_dominance",
    "sparse_completion_proxy",
    "early_failure_or_crash",
    "high_reward_without_success",
    "goal_near_oscillation",
    "contact_reward_hacking",
    "generated_reward_negative_average",
]


def read_text(path):
    return Path(path).read_text(encoding="utf-8")


def write_text(path, text):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def extract_card(cards_md, card_id):
    marker = f"## {card_id}"
    start = cards_md.find(marker)
    if start < 0:
        return ""
    next_start = cards_md.find("\n## ", start + len(marker))
    if next_start < 0:
        return cards_md[start:].strip()
    return cards_md[start:next_start].strip()


def matched_card_ids(feedback_md, top_k=4):
    text = feedback_md.lower()
    out = []
    rules = [
        ("speed_penalty_dominance", ["speed_penalty dominates", "speed_penalty_dominance"]),
        ("stability_penalty_dominance", ["stability_penalty dominates", "stability_penalty_dominance"]),
        ("sparse_completion_proxy", ["soft_landing_bonus is too sparse", "sparse_completion_proxy", "rarely reached"]),
        ("early_failure_or_crash", ["early_failure_or_crash", "short episode length"]),
        ("high_reward_without_success", ["high_reward_without_success"]),
        ("goal_near_oscillation", ["goal_near_oscillation"]),
        ("contact_reward_hacking", ["contact_reward_hacking"]),
        ("generated_reward_negative_average", ["generated reward is negative", "total_reward mean: -"]),
    ]
    for card_id, keys in rules:
        if any(k in text for k in keys):
            out.append(card_id)
    return out[:max(1, int(top_k))]


def extract_section(md, heading):
    marker = f"## {heading}"
    start = md.find(marker)
    if start < 0:
        return ""
    next_start = md.find("\n## ", start + len(marker))
    if next_start < 0:
        return md[start:].strip()
    return md[start:next_start].strip()


def compact_feedback(feedback_md):
    parts = []
    for heading in ["2. External evaluation", "3. Reward execution health", "4. Key component evidence", "5. Preliminary failure hints"]:
        sec = extract_section(feedback_md, heading)
        if sec:
            parts.append(sec)
    return "\n\n".join(parts).strip()


def compact_memory(memory_md):
    latest = extract_section(memory_md, "Latest Iter Detail")
    lessons = extract_section(memory_md, "Stable Lessons")
    blocks = []
    if lessons:
        blocks.append(lessons)
    if latest:
        blocks.append(latest)
    return "\n\n".join(blocks).strip()


def build_skeleton_plan(feedback_md):
    text = feedback_md.lower()
    lines = []
    lines.append("## Skeleton Revision Plan")
    lines.append("")
    lines.append("### keep")
    lines.append("- progress_delta_reward")
    lines.append("")
    lines.append("### weaken")
    if "speed_penalty" in text:
        lines.append("- speed_penalty")
    if "stability_penalty dominates" in text:
        lines.append("- stability_penalty")
    if lines[-1] == "### weaken":
        lines.append("- none")
    lines.append("")
    lines.append("### revise")
    if "soft_landing" in text:
        lines.append("- soft_landing_proxy -> smoother landing-quality shaping")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("### consider_add")
    lines.append("- distance_reward as a small anchor if progress-only guidance remains weak")
    lines.append("")
    lines.append("### still_defer")
    lines.append("- terminal_success_reward")
    lines.append("- terminal_failure_penalty")
    lines.append("- energy_penalty")
    lines.append("- time_penalty")
    lines.append("- gated_reward")
    return "\n".join(lines)


def build_context(train_run_dir, memory_path, cards_path, top_k=4):
    train_dir = Path(train_run_dir)
    feedback_md = read_text(train_dir / "training_feedback.md")
    memory_md = read_text(memory_path)
    cards_md = read_text(cards_path)

    card_blocks = []
    ids = matched_card_ids(feedback_md, top_k=top_k)
    for card_id in ids:
        block = extract_card(cards_md, card_id)
        if block:
            card_blocks.append(block)

    lines = []
    lines.append("# Iteration Context for Reward Revision")
    lines.append("")
    lines.append("This is the single compact context file for the next reward revision LLM.")
    lines.append("Do not treat expert cards as templates; use them as diagnostic guidance.")
    lines.append("")
    lines.append("## Previous Training Feedback")
    lines.append("")
    lines.append(compact_feedback(feedback_md))
    lines.append("")
    lines.append("## Short Memory")
    lines.append("")
    lines.append(compact_memory(memory_md))
    lines.append("")
    lines.append("## Matched Expert Cards")
    lines.append("")
    if card_blocks:
        lines.append("\n\n".join(card_blocks))
    else:
        lines.append("- none")
    lines.append("")
    lines.append(build_skeleton_plan(feedback_md))
    lines.append("")
    lines.append("## Reward Revision Boundary")
    lines.append("")
    lines.append("- Revise the previous reward instead of generating from scratch.")
    lines.append("- Keep the function signature unchanged.")
    lines.append("- Do not use original_reward or unavailable info fields.")
    lines.append("- Do not add terminal success/failure rewards without explicit signals.")
    lines.append("- Prefer fewer components with clear roles over adding many skeletons.")
    return "\n".join(lines).strip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-run-dir", required=True)
    ap.add_argument("--memory", default="runs/env_001/memory/reward_memory.md")
    ap.add_argument("--cards", default="knowledge_base/iteration/reward_misalignment_cards.md")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    context = build_context(args.train_run_dir, args.memory, args.cards, top_k=args.top_k)
    write_text(args.out, context)
    print(args.out)


if __name__ == "__main__":
    main()
