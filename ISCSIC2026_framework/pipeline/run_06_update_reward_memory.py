import argparse
import json
import re
from pathlib import Path


HEADER = "# Reward Memory"
TABLE_HEADER = "| iter | skeleton | score | best | delta | len | key_signal | action |\n|---:|---|---:|---:|---:|---:|---|---|\n"


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
    """Compact component summary for table cell."""
    parts = []
    skip = {"component.total_reward", "component.generated_reward"}
    for name in sorted(component_stats.keys()):
        if not name.startswith("component.") or name in skip:
            continue
        item = component_stats[name]
        short = component_name(name)
        m = float(item.get("mean", 0))
        parts.append(f"{short}={f3(m)}")
    return " ".join(parts[:5]) if parts else "—"


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


def build_memory(memory_path, iter_id, training_run_dir, target_score, best_score=None, best_iter=None, decision="continue", new_lessons=None):
    run_dir = Path(training_run_dir)
    summary = read_json(run_dir / "training_summary.json")
    feedback_md = read_text(run_dir / "training_feedback.md")
    component_summary = summary.get("component_summary", {})
    component_stats = component_summary.get("component_stats", {})
    external_eval = summary.get("external_eval", {})

    mean_score = float(external_eval.get("mean_eval_reward", 0.0))
    mean_len = float(external_eval.get("mean_episode_length", 0.0))

    if best_score is None:
        best_score = mean_score
    if best_iter is None:
        best_iter = iter_id
    delta = mean_score - float(best_score)

    structure = reward_structure(component_stats)
    signal = key_signal(component_stats)
    diagnosis, action = diagnosis_and_action(feedback_md, component_stats, mean_score, target_score)

    row = (
        f"| {iter_id} | {structure} | {f2(mean_score)} | {f2(best_score)} | "
        f"{f2(delta)} | {f2(mean_len)} | {signal} | {decision} |"
    )

    memory_md = read_text(memory_path, "")

    rows = [r for r in existing_rows(memory_md) if not re.match(rf"^\|\s*{iter_id}\s*\|", r)]
    rows.append(row)
    rows = sorted(rows, key=lambda r: int(r.split("|")[1].strip()))

    text = HEADER.strip() + "\n\n" + TABLE_HEADER + "\n".join(rows) + "\n"

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
    ap.add_argument("--new-lessons", default=None)
    args = ap.parse_args()

    out = build_memory(
        args.memory,
        args.iter,
        args.train_run_dir,
        target_score=args.target_score,
        best_score=args.best_score,
        best_iter=args.best_iter,
        decision=args.decision,
        new_lessons=args.new_lessons,
    )
    print(out)


if __name__ == "__main__":
    main()
