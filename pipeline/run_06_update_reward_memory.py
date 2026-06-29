import argparse
import json
import re
from pathlib import Path


HEADER = """# Reward Memory for Env_001

This file stores compact cross-iteration lessons. It is not a file index.
Do not copy full reward code, full logs, or full training summaries here.
"""

STABLE_LESSONS = """## Stable Lessons

- Use external evaluation reward as the fitness signal; generated reward alone is not enough.
- Keep terminal_success_reward blocked until an explicit success signal is available.
- Keep terminal_failure_penalty blocked until failure reason is available.
- Contact flags are only usable inside a guarded landing proxy: near target + low speed + stable angle + contact.
- Avoid speed or stability penalties dominating the main progress signal.
- Avoid a hard sparse completion bonus as the only landing guidance.
- Keep memory short: record component structure, key evidence, diagnosis, and next action only.
"""

TABLE_HEADER = """## Evolution Summary

| iter | reward_structure | external_score | len | gen_reward | key_component_signal | verdict | diagnosis | next_action |
|---:|---|---:|---:|---:|---|---|---|---|
"""


def read_text(path, default=""):
    p = Path(path)
    if not p.exists():
        return default
    return p.read_text(encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def f2(value):
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "N/A"


def f3(value):
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "N/A"


def component_name(name):
    if name.startswith("component."):
        return name.split(".", 1)[1]
    return name


def reward_structure(component_stats):
    names = []
    for key in component_stats:
        if not key.startswith("component."):
            continue
        name = component_name(key)
        if name == "total_reward":
            continue
        names.append(name)
    if not names:
        return "unknown"
    return " + ".join(names[:6])


def get_stat(component_stats, name):
    return component_stats.get(name, {})


def key_signal(component_stats):
    parts = []
    progress = get_stat(component_stats, "component.progress_reward")
    speed = get_stat(component_stats, "component.speed_penalty")
    stability = get_stat(component_stats, "component.stability_penalty")
    soft = get_stat(component_stats, "component.soft_landing_bonus")
    landing = get_stat(component_stats, "component.landing_quality")

    if progress:
        parts.append(f"progress {f3(progress.get('mean'))}")
    if speed:
        parts.append(f"speed {f3(speed.get('mean'))}")
    if stability:
        parts.append(f"stability {f3(stability.get('mean'))}")
    if soft:
        parts.append(f"soft {100 * float(soft.get('nonzero_rate', 0.0)):.2f}%")
    if landing:
        parts.append(f"landing {f3(landing.get('mean'))}")
    if not parts:
        return "component stats unavailable"
    return "; ".join(parts[:4])


def diagnosis_and_action(feedback_md, component_stats, mean_score):
    text = feedback_md.lower()
    diagnosis = []
    actions = []

    if "early_failure_or_crash" in text or mean_score < 0:
        diagnosis.append("early_failure_or_crash")
        actions.append("add smoother approach/landing guidance")
    if "speed_penalty dominates" in text:
        diagnosis.append("speed_penalty_dominance")
        actions.append("weaken speed penalty")
    if "stability_penalty dominates" in text:
        diagnosis.append("stability_penalty_dominance")
        actions.append("weaken stability penalty")
    if "soft_landing_bonus is too sparse" in text or "rarely reached" in text:
        diagnosis.append("sparse_completion_proxy")
        actions.append("smooth landing proxy")
    if "generated reward is negative" in text:
        diagnosis.append("generated_reward_negative_average")
        actions.append("rebalance progress and penalties")

    if not diagnosis:
        diagnosis.append("needs_review")
    if not actions:
        actions.append("inspect component balance")

    return "; ".join(dict.fromkeys(diagnosis)), "; ".join(dict.fromkeys(actions))


def existing_rows(memory_md):
    rows = []
    for line in memory_md.splitlines():
        if re.match(r"^\|\s*\d+\s*\|", line):
            rows.append(line)
    return rows


def latest_detail(iter_id, structure, mean_score, mean_len, reward_error_count, key_component, diagnosis, action):
    return f"""## Latest Iter Detail

### iter_{iter_id}

- reward_structure: {structure}
- external_score: {f2(mean_score)}
- mean_episode_length: {f2(mean_len)}
- reward_error_count: {reward_error_count}

#### component_evidence

- {key_component}

#### diagnosis

- {diagnosis}

#### next_action

- {action}
"""


def build_memory(memory_path, iter_id, training_run_dir):
    run_dir = Path(training_run_dir)
    summary = read_json(run_dir / "training_summary.json")
    feedback_md = read_text(run_dir / "training_feedback.md")
    component_summary = summary.get("component_summary", {})
    component_stats = component_summary.get("component_stats", {})
    external_eval = summary.get("external_eval", {})

    mean_score = float(external_eval.get("mean_eval_reward", 0.0))
    mean_len = float(external_eval.get("mean_episode_length", 0.0))
    gen_reward = get_stat(component_stats, "generated_reward").get("mean", "N/A")
    reward_error_count = component_summary.get("reward_error_count_max", 0)

    structure = reward_structure(component_stats)
    signal = key_signal(component_stats)
    diagnosis, action = diagnosis_and_action(feedback_md, component_stats, mean_score)
    verdict = "success" if mean_score >= 200 else "failure"

    row = (
        f"| {iter_id} | {structure} | {f2(mean_score)} | {f2(mean_len)} | "
        f"{f3(gen_reward)} | {signal} | {verdict} | {diagnosis} | {action} |"
    )

    memory_md = read_text(memory_path, "")
    rows = [r for r in existing_rows(memory_md) if not re.match(rf"^\|\s*{iter_id}\s*\|", r)]
    rows.append(row)
    rows = sorted(rows, key=lambda r: int(r.split("|")[1].strip()))

    text = HEADER.strip() + "\n\n" + TABLE_HEADER + "\n".join(rows) + "\n\n" + STABLE_LESSONS.strip() + "\n\n"
    text += latest_detail(iter_id, structure, mean_score, mean_len, reward_error_count, signal, diagnosis, action)

    p = Path(memory_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--train-run-dir", required=True)
    ap.add_argument("--memory", default="runs/env_001/memory/reward_memory.md")
    args = ap.parse_args()

    out = build_memory(args.memory, args.iter, args.train_run_dir)
    print(out)


if __name__ == "__main__":
    main()
