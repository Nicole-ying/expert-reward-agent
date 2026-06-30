import argparse
import json
import re
from pathlib import Path


HEADER = """# Reward Memory for Env_001

This file stores compact cross-iteration history. It records:
- Full reward code path per iteration (the code lives in the generation directory)
- Per-iteration component structure and evidence
- Score trend and diagnosis

The history table is used by the analysis LLM to detect stagnation and repeated skeletons.
"""

STABLE_LESSONS = """## Stable Lessons

- Use external evaluation reward as the fitness signal; generated reward alone is not enough.
- Target solved threshold for Env_001: mean external evaluation score >= 200.
- Preserve best-so-far reward; final reward should be the best reward, not necessarily the last reward.
- If the task has been solved and a later revision drops below target, stop and keep the best reward.
- Keep terminal_success_reward blocked until an explicit success signal is available.
- Keep terminal_failure_penalty blocked until failure reason is available.
- Contact flags are only usable inside a guarded landing proxy: near target + low speed + stable angle + contact.
- Avoid speed or stability penalties dominating the main progress signal.
- Avoid a hard sparse completion bonus as the only landing guidance.
"""

TABLE_HEADER = """## Evolution Summary

| iter | reward_structure | external_score | best_so_far | delta_from_best | len | gen_reward | key_component_signal | verdict | decision | diagnosis | next_action |
|---:|---|---:|---:|---:|---:|---:|---|---|---|---|---|
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
    """Report ALL components, not just hardcoded names."""
    parts = []
    skip = {"component.total_reward", "component.generated_reward"}
    for name in sorted(component_stats.keys()):
        if not name.startswith("component.") or name in skip:
            continue
        item = component_stats[name]
        short = component_name(name)
        mean_val = float(item.get("mean", 0))
        trigger = float(item.get("nonzero_rate", 1.0))
        if trigger < 0.99:
            parts.append(f"{short} {f3(mean_val)} ({trigger:.2%})")
        else:
            parts.append(f"{short} {f3(mean_val)}")
    if not parts:
        return "component stats unavailable"
    return "; ".join(parts[:8])


def diagnosis_and_action(feedback_md, component_stats, mean_score, target_score):
    text = feedback_md.lower()
    diagnosis = []
    actions = []

    if mean_score >= target_score:
        diagnosis.append("solved")
        actions.append("protect best reward; only continue with small evidence-driven changes")
    if "early_failure_or_crash" in text or "persistent_negative_score" in text:
        diagnosis.append("early_failure_or_crash")
        if "persistent_negative_score" in text:
            actions.append("agent survives but never achieves positive return: may need stronger progress shaping or velocity damping")
        else:
            actions.append("agent crashes early: need stability guidance before termination")
    if "penalty_dominance" in text:
        diagnosis.append("penalty_dominance_detected")
        actions.append("a penalty term may be overpowering the progress signal: weaken or conditionalize it")
    if "sparse_proxy" in text:
        diagnosis.append("sparse_completion_proxy")
        actions.append("a bonus-like component has very low trigger rate: replace with continuous shaping")
    if "generated_reward_negative_average" in text or "generated reward is negative" in text:
        diagnosis.append("generated_reward_negative_average")
        actions.append("rebalance progress and penalties so mean reward is not punitive")
    if "partial_progress" in text:
        diagnosis.append("partial_progress")
        actions.append("positive but below target: inspect component balance, consider slightly increasing main signal weight")

    if not diagnosis:
        diagnosis.append("needs_review")
        actions.append("review all component signals and score trend; consider trying a different skeleton family")

    return "; ".join(dict.fromkeys(diagnosis)), "; ".join(dict.fromkeys(actions))


def existing_rows(memory_md):
    rows = []
    for line in memory_md.splitlines():
        if re.match(r"^\|\s*\d+\s*\|", line):
            rows.append(line)
    return rows


def latest_detail(iter_id, structure, mean_score, best_score, best_iter, mean_len, reward_error_count, key_component, verdict, decision, diagnosis, action, code_path=""):
    block = f"""### iter_{iter_id}

- reward_structure: {structure}
- external_score: {f2(mean_score)}
- best_score_so_far: {f2(best_score)}
- best_iter: {best_iter}
- mean_episode_length: {f2(mean_len)}
- reward_error_count: {reward_error_count}
- verdict: {verdict}
- decision: {decision}

#### component_evidence

- {key_component}

#### diagnosis

- {diagnosis}

#### next_action

- {action}
"""
    if code_path:
        block += f"\n#### reward_code_path\n- {code_path}\n"
    return block


def build_memory(memory_path, iter_id, training_run_dir, target_score, best_score=None, best_iter=None, decision="continue"):
    run_dir = Path(training_run_dir)
    summary = read_json(run_dir / "training_summary.json")
    feedback_md = read_text(run_dir / "training_feedback.md")
    component_summary = summary.get("component_summary", {})
    component_stats = component_summary.get("component_stats", {})
    external_eval = summary.get("external_eval", {})

    mean_score = float(external_eval.get("mean_eval_reward", 0.0))
    mean_len = float(external_eval.get("mean_episode_length", 0.0))
    gen_reward = get_stat(component_stats, "component.generated_reward").get("mean", "N/A")
    reward_error_count = component_summary.get("reward_error_count_max", 0)

    if best_score is None:
        best_score = mean_score
    if best_iter is None:
        best_iter = iter_id
    delta_from_best = mean_score - float(best_score)

    structure = reward_structure(component_stats)
    signal = key_signal(component_stats)
    diagnosis, action = diagnosis_and_action(feedback_md, component_stats, mean_score, target_score)
    verdict = "success" if mean_score >= target_score else "failure"

    # Store reward code path for reference
    code_path = str(run_dir.parent.parent / "generation" / f"reward_v{iter_id}.py")

    row = (
        f"| {iter_id} | {structure} | {f2(mean_score)} | {f2(best_score)} | {f2(delta_from_best)} | "
        f"{f2(mean_len)} | {f3(gen_reward)} | {signal} | {verdict} | {decision} | {diagnosis} | {action} |"
    )

    memory_md = read_text(memory_path, "")
    rows = [r for r in existing_rows(memory_md) if not re.match(rf"^\|\s*{iter_id}\s*\|", r)]
    rows.append(row)
    rows = sorted(rows, key=lambda r: int(r.split("|")[1].strip()))

    text = HEADER.strip() + "\n\n" + TABLE_HEADER + "\n".join(rows) + "\n\n" + STABLE_LESSONS.strip() + "\n\n"
    text += latest_detail(iter_id, structure, mean_score, best_score, best_iter, mean_len, reward_error_count, signal, verdict, decision, diagnosis, action, code_path=code_path)

    p = Path(memory_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--train-run-dir", required=True)
    ap.add_argument("--memory", default="runs/env_001/memory/reward_memory.md")
    ap.add_argument("--target-score", type=float, default=200.0)
    ap.add_argument("--best-score", type=float, default=None)
    ap.add_argument("--best-iter", type=int, default=None)
    ap.add_argument("--decision", default="continue")
    args = ap.parse_args()

    out = build_memory(
        args.memory,
        args.iter,
        args.train_run_dir,
        target_score=args.target_score,
        best_score=args.best_score,
        best_iter=args.best_iter,
        decision=args.decision,
    )
    print(out)


if __name__ == "__main__":
    main()
