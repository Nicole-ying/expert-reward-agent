import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .common import load_config


def run_cmd(cmd):
    cmd = [sys.executable if arg == "python" else arg for arg in cmd]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def pad_iter(i):
    return f"{i:02d}"


def validate_rounds(rounds):
    if rounds < 1:
        raise ValueError("iteration.total_rounds must be >= 1")


def validate_output_path_length(cfg, prefix, seed, rounds):
    """Fail before API calls when a Windows output path is too long."""
    if os.name != "nt":
        return
    probe = (
        Path.cwd()
        / cfg["experiment"]["run_root"]
        / prefix
        / f"seed_{seed}"
        / f"iter_{rounds:02d}"
        / "generation"
        / "prompt_records"
        / "02_reward_generator.prompt_stats.md"
    )
    if len(str(probe)) >= 248:
        raise ValueError(
            f"Experiment prefix is too long for this Windows workspace "
            f"({len(str(probe))} chars): {prefix!r}. Use a shorter prefix."
        )


def maybe_reset_memory(memory_path, reset):
    p = Path(memory_path)
    if reset and p.exists():
        p.unlink()
    p.parent.mkdir(parents=True, exist_ok=True)


def build_restart_context(memory_path, target_score):
    """Summarize tried component structures without treating every attempt as a failure."""
    path = Path(memory_path)
    if not path.exists():
        return ""

    structures = {}
    attempts = []
    overall_best = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---") or "| iter |" in line:
            continue
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        if len(columns) < 4:
            continue
        try:
            iteration = int(columns[0])
            structure = columns[1]
            score = float(columns[2])
        except (TypeError, ValueError):
            continue
        if not structure:
            continue
        record = structures.setdefault(structure, {"attempts": 0, "best": score, "latest": score, "latest_iter": iteration})
        record["attempts"] += 1
        record["best"] = max(record["best"], score)
        if iteration >= record["latest_iter"]:
            record["latest"] = score
            record["latest_iter"] = iteration
        overall_best = score if overall_best is None else max(overall_best, score)
        attempts.append((iteration, structure, score))

    if not structures:
        return ""

    lines = [
        "# Fresh Restart Evidence",
        "",
        f"- target_score: {target_score:.3f}",
        f"- best_score_so_far: {overall_best:.3f}",
        "",
        "## Tried component structures",
        "",
        "| structure | attempts | best_score | latest_score | status |",
        "|---|---:|---:|---:|---|",
    ]
    for structure, record in sorted(structures.items(), key=lambda item: item[1]["best"], reverse=True):
        status = "solved" if record["best"] >= target_score else "unsolved"
        lines.append(
            f"| {structure} | {record['attempts']} | {record['best']:.3f} | "
            f"{record['latest']:.3f} | {status} |"
        )

    seed_root = path.parent.parent
    lines.extend(["", "## Previous interventions", ""])
    intervention_found = False
    labels = (
        "selected_level", "selected_intervention", "selected_transformation",
        "selected level", "selected intervention", "selected transformation",
        "diagnosis_dimension", "diagnosis dimension", "本轮行动", "本轮修改",
        "修改方案", "修改策略", "核心变换",
    )
    for iteration, structure, score in sorted(attempts):
        generation_dir = seed_root / f"iter_{iteration:02d}" / "generation"
        reward_files = sorted(generation_dir.glob("reward_*.md")) if generation_dir.exists() else []
        if not reward_files:
            continue
        text = reward_files[-1].read_text(encoding="utf-8")
        summary = []
        text_lines = text.splitlines()
        for index, raw_line in enumerate(text_lines):
            line = raw_line.strip().strip("#*- ")
            if not any(label.lower() in line.lower() for label in labels):
                continue
            if len(line) < 24:
                for following in text_lines[index + 1:index + 4]:
                    following = following.strip().strip("#*- ")
                    if following:
                        line = f"{line}: {following}"
                        break
            line = " ".join(line.split())[:220]
            if line and line not in summary:
                summary.append(line)
            if len(summary) >= 3:
                break
        if summary:
            intervention_found = True
            lines.append(f"- iter {iteration} (score={score:.3f}, structure={structure}): " + " | ".join(summary))
    if not intervention_found:
        lines.append("- No structured intervention fields were available in the historical responses.")

    lines.extend([
        "",
        "## Restart instruction",
        "",
        "The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.",
        "Compare the tried structures and their scores before choosing the next direction.",
        "If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.",
        "Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.",
    ])
    return "\n".join(lines) + "\n"


def experiment_root_for(cfg, prefix, seed):
    """Top-level directory for one experiment prefix, nesting everything inside."""
    run_root = Path(cfg["experiment"]["run_root"])
    return run_root / prefix / f"seed_{seed}"


def build_paths(cfg, prefix, iteration_index, seed):
    """All paths live under experiment_root_for/iter_XX/."""
    iter_pad = pad_iter(iteration_index)
    seed_root = experiment_root_for(cfg, prefix, seed)
    iter_root = seed_root / f"iter_{iter_pad}"
    gen_dir = iter_root / "generation"
    train_dir = iter_root / "training"
    gen_run_name = f"{prefix}/seed_{seed}/iter_{iter_pad}/generation"
    context_path = iter_root / "iteration_context.md"
    return {
        "iter_pad": iter_pad,
        "seed_root": seed_root,
        "iter_root": iter_root,
        "gen_dir": gen_dir,
        "gen_run_name": gen_run_name,
        "train_dir": train_dir,
        "context_path": context_path,
    }


def reward_path_for(cfg, gen_run_name, version):
    return Path(cfg["experiment"]["run_root"]) / gen_run_name / f"reward_v{version}.py"


def reward_md_path_for(cfg, gen_run_name, version):
    return Path(cfg["experiment"]["run_root"]) / gen_run_name / f"reward_v{version}.md"


def validation_path_for(cfg, gen_run_name, version):
    return Path(cfg["experiment"]["run_root"]) / gen_run_name / "validations" / f"reward_v{version}.validation.json"


def check_reward_valid(cfg, gen_run_name, version, stop_on_invalid):
    path = validation_path_for(cfg, gen_run_name, version)
    if not path.exists():
        raise FileNotFoundError(f"Missing validation file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if stop_on_invalid and not data.get("valid", False):
        errors = data.get("errors", [])
        warnings = data.get("warnings", [])
        details = "; ".join(str(item) for item in errors) or "unknown validation error"
        if any("import" in str(item).lower() for item in errors):
            details += (
                "; 修复要求：删除所有import和numpy调用；平方根使用**0.5，"
                "有界函数使用max/min或不需要库的代数表达式"
            )
        if warnings:
            details += "; warnings: " + "; ".join(str(item) for item in warnings)
        raise RuntimeError(f"Reward v{version} failed validation: {details} (record: {path})")


def read_training_score(train_dir):
    summary_path = Path(train_dir) / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing training_summary.json: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    external = summary.get("external_eval", {})
    return float(external.get("mean_eval_reward", 0.0)), summary


def code_signature(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if (
                isinstance(body, list)
                and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                del body[0]
        body = ast.dump(tree, include_attributes=False)
    except Exception:
        lines = []
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
        body = "\n".join(lines)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def skeleton_fingerprint(reward_path):
    """Extract the component-key set from a reward.py as a skeleton fingerprint.

    Uses the sorted tuple of component dictionary keys so that different
    component sets are treated as different skeletons, even when they share
    generic words like "velocity" or "forward".
    """
    try:
        tree = ast.parse(Path(reward_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key_node in node.keys:
                if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                    keys.add(key_node.value)
    if not keys:
        return None
    return tuple(sorted(keys))


def _first_iter_score(memory_path):
    """Read the first iteration's score from the reward memory file."""
    import re
    mem = Path(memory_path)
    if not mem.exists():
        return None
    for line in mem.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and re.match(r"\|\s*1\s*\|", line):
            cols = [c.strip() for c in line.split("|")]
            try:
                return float(cols[3])
            except (ValueError, IndexError):
                return None
    return None


def is_identical_reward(path_a, path_b):
    if not path_a or not path_b:
        return False
    if not Path(path_a).exists() or not Path(path_b).exists():
        return False
    return code_signature(path_a) == code_signature(path_b)


def find_identical_historical_reward(cfg, prefix, seed, current_iteration, current_reward):
    """Return the earlier iteration and path matching the current executable code."""
    if not current_reward or not Path(current_reward).exists():
        return None
    current_signature = code_signature(current_reward)
    seed_root = experiment_root_for(cfg, prefix, seed)
    for iteration_index in range(1, current_iteration):
        generation_dir = seed_root / f"iter_{iteration_index:02d}" / "generation"
        for historical_path in sorted(generation_dir.glob("reward_*.py")):
            if code_signature(historical_path) == current_signature:
                return iteration_index, historical_path
    return None


def append_noop_retry_instruction(context_path, attempt):
    p = Path(context_path)
    text = p.read_text(encoding="utf-8")
    text += f"""

## No-op Retry Instruction

The previous revision attempt #{attempt} produced a reward function that is semantically identical to previous_reward.py.
This is not an acceptable iteration while the task is still unsolved.
You must perform a concrete tune/delete/add/mix action justified by the training evidence.
Do not return identical reward code again.
"""
    p.write_text(text, encoding="utf-8")


def copy_best_artifacts(cfg, prefix, seed, iteration_index, reward_path, reward_md_path, train_dir, best_score, target_score):
    best_dir = experiment_root_for(cfg, prefix, seed) / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(reward_path, best_dir / "best_reward.py")
    if Path(reward_md_path).exists():
        shutil.copy2(reward_md_path, best_dir / "best_reward.md")
    feedback = Path(train_dir) / "training_feedback.md"
    summary = Path(train_dir) / "training_summary.json"
    if feedback.exists():
        shutil.copy2(feedback, best_dir / "best_training_feedback.md")
    if summary.exists():
        summary_data = json.loads(summary.read_text(encoding="utf-8"))
        normalization = summary_data.get("normalization", {})
        vecnormalize_path = normalization.get("vecnormalize_path")
        if vecnormalize_path and Path(vecnormalize_path).exists():
            best_vecnormalize = best_dir / "best_vecnormalize.pkl"
            shutil.copy2(vecnormalize_path, best_vecnormalize)
            normalization["vecnormalize_path"] = str(best_vecnormalize)
        (best_dir / "best_training_summary.json").write_text(
            json.dumps(summary_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    summary_text = f"""# Best Reward Summary

- best_iter: {iteration_index}
- best_score: {best_score:.3f}
- target_score: {target_score:.3f}
- solved: {best_score >= target_score}
- reward_path: {reward_path}
- training_dir: {train_dir}

Final reward should use `best_reward.py`, not necessarily the last generated reward.
"""
    (best_dir / "best_summary.md").write_text(summary_text, encoding="utf-8")
    return best_dir


def write_experiment_summary(cfg, prefix, seed, stopped_reason, best_iter, best_score, target_score, rounds_completed):
    exp_root = experiment_root_for(cfg, prefix, seed)
    exp_root.mkdir(parents=True, exist_ok=True)
    disk_best = None
    for summary_path in sorted(exp_root.glob("iter_*/training/training_summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            score = float(summary.get("external_eval", {}).get("mean_eval_reward"))
            iteration_index = int(summary_path.parents[1].name.split("_")[-1])
            reward_path = Path(summary.get("reward_path", ""))
            if not reward_path.is_absolute():
                reward_path = Path.cwd() / reward_path
            if reward_path.exists() and (disk_best is None or score > disk_best[0]):
                disk_best = (score, iteration_index, reward_path, summary_path.parent)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    if disk_best is not None:
        best_score, best_iter, reward_path, train_dir = disk_best
        copy_best_artifacts(
            cfg=cfg,
            prefix=prefix,
            seed=seed,
            iteration_index=best_iter,
            reward_path=reward_path,
            reward_md_path=reward_path.with_suffix(".md"),
            train_dir=train_dir,
            best_score=best_score,
            target_score=target_score,
        )
    best_dir = exp_root / "best"
    best_score_text = "N/A" if best_score is None else f"{best_score:.3f}"
    text = f"""# Iterative Experiment Summary

- prefix: {prefix}
- seed: {seed}
- rounds_completed: {rounds_completed}
- target_score: {target_score:.3f}
- best_iter: {best_iter}
- best_score: {best_score_text}
- solved: {best_score is not None and best_score >= target_score}
- stopped_reason: {stopped_reason}

## Best Artifact

- best_dir: {best_dir}
- best_reward: {best_dir / 'best_reward.py'}
- best_summary: {best_dir / 'best_summary.md'}
"""
    (exp_root / "experiment_summary.md").write_text(text, encoding="utf-8")


def run_iterative_experiment(config_path, prefix=None, rounds=None, total_timesteps=None, eval_episodes=None, mock=None, seed=0, resume_from=None):
    cfg = load_config(config_path)
    iter_cfg = cfg.get("iteration", {})
    train_cfg = cfg.get("training", {})
    rag_cfg = cfg.get("rag", {})
    ablation_cfg = cfg.get("ablation", {})

    prefix = prefix or iter_cfg.get("experiment_prefix", "exp_iter")
    rounds = int(rounds if rounds is not None else iter_cfg.get("total_rounds", 10))
    validate_rounds(rounds)
    validate_output_path_length(cfg, prefix, seed, rounds)

    total_timesteps = int(total_timesteps if total_timesteps is not None else train_cfg.get("total_timesteps", 1000000))
    eval_episodes = int(eval_episodes if eval_episodes is not None else train_cfg.get("eval_episodes", 10))
    use_mock = bool(iter_cfg.get("use_mock_llm", False) if mock is None else mock)
    stop_on_invalid = bool(iter_cfg.get("stop_on_invalid_reward", True))
    cards_top_k = int(iter_cfg.get("cards_top_k", 4))

    target_score = float(iter_cfg.get("target_score", 200.0))
    min_improvement = float(iter_cfg.get("min_meaningful_improvement", 5.0))
    stop_after_solved_drop = bool(iter_cfg.get("stop_after_solved_drop", True))
    stop_when_solved_and_identical = bool(iter_cfg.get("stop_when_solved_and_identical", True))
    patience_after_solved = int(iter_cfg.get("no_improvement_patience_after_solved", 2))
    patience_unsolved = int(iter_cfg.get("no_improvement_patience_unsolved", 3))
    retry_identical_unsolved = bool(iter_cfg.get("retry_identical_when_unsolved", True))
    max_identical_retries = int(iter_cfg.get("max_identical_revision_retries", 2))
    use_reflection_agent = bool(iter_cfg.get("use_reflection_agent", False))

    memory_path = str(experiment_root_for(cfg, prefix, seed) / "memory" / "reward_memory.md")
    # When memory is disabled, pass a non-existent path so agents see empty history
    agent_memory_path = memory_path if not ablation_cfg.get("disable_memory", False) else "__disabled_memory__"
    restart_memory_path = agent_memory_path
    cards_path = rag_cfg.get("reward_misalignment_cards_path", "knowledge_base/iteration/reward_misalignment_cards.md")

    # Resume support
    start_iter = 1
    previous_reward = None
    best_score = None
    best_iter = None
    best_reward = None
    solved_seen = False
    no_improve_count = 0
    stopped_reason = "completed_all_rounds"
    rounds_completed = 0
    last_score = None
    force_fresh_restart = False
    restart_count = 0
    same_skeleton_count = 0
    last_skeleton_fingerprint = None

    if resume_from and resume_from > 1:
        start_iter = resume_from
        print(f"Resuming from iteration {resume_from}")
        # Don't reset memory when resuming
        maybe_reset_memory(memory_path, False)
        # Read state from previous iteration
        prev_iter = resume_from - 1
        prev_paths = build_paths(cfg, prefix, prev_iter, seed)
        reward_path = reward_path_for(cfg, prev_paths["gen_run_name"], prev_iter)
        if Path(reward_path).exists():
            previous_reward = reward_path
        # Scan historical reward files to reconstruct skeleton fingerprint counter
        for i in range(1, start_iter):
            ipaths = build_paths(cfg, prefix, i, seed)
            rpath = reward_path_for(cfg, ipaths["gen_run_name"], i)
            fp = skeleton_fingerprint(rpath) if Path(rpath).exists() else None
            if fp and fp == last_skeleton_fingerprint:
                same_skeleton_count += 1
            else:
                same_skeleton_count = 1
            last_skeleton_fingerprint = fp
        print(f"Resume: reconstructed same_skeleton_count={same_skeleton_count} from {start_iter-1} historical iterations")
        # Reconstruct no_improve_count from memory (count consecutive no_improvement from the end)
        mem = Path(memory_path)
        if mem.exists():
            lines = mem.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines):
                if "no_meaningful_improvement" in line:
                    no_improve_count += 1
                elif "new_best" in line or "target_solved" in line:
                    break
        print(f"Resume: reconstructed no_improve_count={no_improve_count}")
        # Read best score from memory
        import re
        mem = Path(memory_path)
        if mem.exists():
            for line in mem.read_text(encoding="utf-8").splitlines():
                if line.startswith("|") and re.match(r"\|\s*\d+\s*\|", line):
                    cols = [c.strip() for c in line.split("|")]
                    try:
                        s = float(cols[3])
                        if best_score is None or s > best_score:
                            best_score = s
                            best_iter = int(cols[1])
                    except: pass
            if best_score is not None:
                solved_seen = best_score >= target_score
                # Find best reward path
                bp = experiment_root_for(cfg, prefix, seed) / "best" / "best_reward.py"
                if bp.exists():
                    best_reward = str(bp)
        # Check if reconstructed state already meets restart conditions
        stuck_rounds = same_skeleton_count
        if (not solved_seen) and same_skeleton_count >= 4 and (best_score is not None and best_score < 0):
            force_fresh_restart = True
            restart_count += 1
            same_skeleton_count = 0
            no_improve_count = 0
            print(f">>> Resume: same skeleton for {stuck_rounds} rounds, best={best_score:.1f} still negative. Triggering fresh restart immediately.")
        elif (not solved_seen) and same_skeleton_count >= 4 and (best_score is not None and best_score > 0) and no_improve_count >= 1:
            force_fresh_restart = True
            restart_count += 1
            same_skeleton_count = 0
            no_improve_count = 0
            print(f">>> Resume: same skeleton for {stuck_rounds} rounds, best={best_score:.1f} positive but oscillating. Triggering fresh restart immediately.")
    else:
        maybe_reset_memory(memory_path, bool(iter_cfg.get("reset_memory_at_start", True)))

    print("=" * 60)
    print("Config-driven iterative reward experiment")
    print("=" * 60)
    print(f"config          : {config_path}")
    print(f"prefix          : {prefix}")
    print(f"seed            : {seed}")
    print(f"rounds          : {rounds}")
    print(f"start_iter      : {start_iter}")
    print(f"target_score    : {target_score}")
    print(f"total_timesteps : {total_timesteps}")
    print(f"eval_episodes   : {eval_episodes}")
    print(f"memory_path     : {memory_path}")
    print(f"cards_path      : {cards_path}")
    print(f"cards_top_k     : {cards_top_k}")
    print(f"mock_llm        : {use_mock}")

    for iteration_index in range(start_iter, rounds + 1):
        version = iteration_index
        paths = build_paths(cfg, prefix, iteration_index, seed)
        mock_args = ["--mock"] if use_mock else []

        print("\n" + "-" * 60)
        print(f"Iteration {iteration_index}/{rounds}")
        print("-" * 60)

        if iteration_index == 1 or force_fresh_restart:
            if force_fresh_restart:
                print(">>> Fresh restart: injecting failed-skeleton context into generation")
                version = 1
                failed_info = build_restart_context(restart_memory_path, target_score)
                # Write restart context where run_03 can read it
                gen_dir = Path(cfg["experiment"]["run_root"]) / paths["gen_run_name"]
                gen_dir.mkdir(parents=True, exist_ok=True)
                (gen_dir / "restart_context.md").write_text(failed_info, encoding="utf-8")
            else:
                version = iteration_index
            force_fresh_restart = False
            run_cmd([
                "python", "-m", "pipeline.run_direct_generation_pipeline",
                "--config", config_path,
                "--run-name", paths["gen_run_name"],
                "--seed", str(seed + restart_count * 100),
                *mock_args,
            ])
            current_reward = reward_path_for(cfg, paths["gen_run_name"], 1)
        else:
            prev_paths = build_paths(cfg, prefix, iteration_index - 1, seed)

            if use_reflection_agent:
                # Single-agent reflection mode
                best_arg = []
                if best_reward and str(best_reward) != str(previous_reward):
                    best_arg = ["--best-reward", str(best_reward)]
                run_cmd([
                    "python", "-m", "pipeline.run_reflection_agent",
                    "--config", config_path,
                    "--previous-reward", str(previous_reward),
                    "--environment-card", str(build_paths(cfg, prefix, 1, seed)["gen_dir"] / "environment_card.md"),
                    "--train-run-dir", str(prev_paths["train_dir"]),
                    "--memory", agent_memory_path,
                    "--out-run-name", paths["gen_run_name"],
                    "--reward-version", f"v{version}",
                    *best_arg,
                    *mock_args,
                ])
                current_reward = reward_path_for(cfg, paths["gen_run_name"], version)
            else:
                # Legacy: run_04 analysis + run_05 revision
                run_cmd([
                    "python", "-m", "pipeline.run_04_build_iteration_context",
                    "--train-run-dir", str(prev_paths["train_dir"]),
                    "--memory", agent_memory_path,
                    "--cards", cards_path,
                    "--top-k", str(cards_top_k),
                    "--out", str(paths["context_path"]),
                    "--config", config_path,
                    *mock_args,
                ])

                identical_after_retries = False
                for attempt in range(max_identical_retries + 1):
                    if attempt > 0:
                        append_noop_retry_instruction(paths["context_path"], attempt)
                    best_arg = []
                    if best_reward and str(best_reward) != str(previous_reward):
                        best_arg = ["--best-reward", str(best_reward)]
                    run_cmd([
                        "python", "-m", "pipeline.run_05_reward_revision",
                        "--config", config_path,
                        "--previous-reward", str(previous_reward),
                        "--iteration-context", str(paths["context_path"]),
                        "--out-run-name", paths["gen_run_name"],
                        "--reward-version", f"v{version}",
                        *best_arg,
                        *mock_args,
                    ])
                    current_reward = reward_path_for(cfg, paths["gen_run_name"], version)
                    if not is_identical_reward(previous_reward, current_reward):
                        break
                    identical_after_retries = True
                    if solved_seen and stop_when_solved_and_identical:
                        stopped_reason = "stop_solved_identical_reward_keep_best"
                        print("Identical reward after solved. Stop and keep best reward.")
                        write_experiment_summary(cfg, prefix, seed, stopped_reason, best_iter, best_score, target_score, rounds_completed)
                        return
                    if not retry_identical_unsolved:
                        break

                if identical_after_retries and is_identical_reward(previous_reward, current_reward):
                    stopped_reason = "stop_identical_reward_after_retries_keep_best"
                    print("Revision remained identical after retries. Stop to avoid white-run training.")
                    write_experiment_summary(cfg, prefix, seed, stopped_reason, best_iter, best_score, target_score, rounds_completed)
                    return

        # Validate, with retries if invalid
        valid = False
        for retry in range(3):
            try:
                check_reward_valid(cfg, paths["gen_run_name"], version, True)
                valid = True
                break
            except (RuntimeError, FileNotFoundError) as e:
                print(f"Reward validation failed (retry {retry+1}/3): {e}")
                if retry == 2:
                    break
                # Retry: re-run the same LLM call with validation errors
                if iteration_index == 1 or force_fresh_restart:
                    cmd = [
                        "python", "-m", "pipeline.run_direct_generation_pipeline",
                        "--config", config_path,
                        "--run-name", paths["gen_run_name"],
                        "--seed", str(seed + restart_count * 100 + retry),
                        "--validation-retry", str(e),
                        *mock_args,
                    ]
                elif use_reflection_agent:
                    cmd = [
                        "python", "-m", "pipeline.run_reflection_agent",
                        "--config", config_path,
                        "--previous-reward", str(previous_reward),
                        "--environment-card", str(build_paths(cfg, prefix, 1, seed)["gen_dir"] / "environment_card.md"),
                        "--train-run-dir", str(prev_paths["train_dir"]),
                        "--memory", agent_memory_path,
                        "--out-run-name", paths["gen_run_name"],
                        "--reward-version", f"v{version}",
                        "--validation-retry", str(e),
                    ]
                    if best_reward and str(best_reward) != str(previous_reward):
                        cmd += ["--best-reward", str(best_reward)]
                    cmd += mock_args
                else:
                    cmd = [
                        "python", "-m", "pipeline.run_05_reward_revision",
                        "--config", config_path,
                        "--previous-reward", str(previous_reward),
                        "--iteration-context", str(paths["context_path"]),
                        "--out-run-name", paths["gen_run_name"],
                        "--reward-version", f"v{version}",
                    ]
                    if best_reward and str(best_reward) != str(previous_reward):
                        cmd += ["--best-reward", str(best_reward)]
                    cmd += mock_args
                run_cmd(cmd)
        if not valid:
            print("Invalid code after 3 retries. Skipping iteration, forcing fresh restart next.")
            force_fresh_restart = True
            restart_count += 1
            no_improve_count += 1
            continue

        # Check if current reward is identical to best (not just previous) — skip redundant training
        duplicate_match = find_identical_historical_reward(
            cfg, prefix, seed, iteration_index, current_reward
        )
        if duplicate_match:
            duplicate_resolved = False
            if use_reflection_agent and iteration_index > 1:
                for retry in range(max_identical_retries):
                    duplicate_iter, duplicate_path = duplicate_match
                    print(
                        f"Revision is identical to historical iter {duplicate_iter}; requesting a new reward "
                        f"({retry + 1}/{max_identical_retries})."
                    )
                    duplicate_cmd = [
                        "python", "-m", "pipeline.run_reflection_agent",
                        "--config", config_path,
                        "--previous-reward", str(previous_reward),
                        "--environment-card", str(build_paths(cfg, prefix, 1, seed)["gen_dir"] / "environment_card.md"),
                        "--train-run-dir", str(prev_paths["train_dir"]),
                        "--memory", agent_memory_path,
                        "--out-run-name", paths["gen_run_name"],
                        "--reward-version", f"v{version}",
                        "--duplicate-retry",
                        f"The previous generation duplicated iter {duplicate_iter} ({duplicate_path}). "
                        f"Retry {retry + 1}: generate a materially different reward function.",
                    ]
                    if best_reward and str(best_reward) != str(previous_reward):
                        duplicate_cmd += ["--best-reward", str(best_reward)]
                    duplicate_cmd += mock_args
                    run_cmd(duplicate_cmd)
                    current_reward = reward_path_for(cfg, paths["gen_run_name"], version)
                    try:
                        check_reward_valid(cfg, paths["gen_run_name"], version, True)
                    except (RuntimeError, FileNotFoundError) as exc:
                        print(f"Duplicate retry produced invalid code: {exc}")
                        continue
                    duplicate_match = find_identical_historical_reward(
                        cfg, prefix, seed, iteration_index, current_reward
                    )
                    if not duplicate_match:
                        duplicate_resolved = True
                        break

            if not duplicate_resolved and duplicate_match:
                print("Duplicate persisted after retries. Fresh-regenerating within the same iteration.")
                restart_count += 1
                gen_dir = Path(cfg["experiment"]["run_root"]) / paths["gen_run_name"]
                gen_dir.mkdir(parents=True, exist_ok=True)
                (gen_dir / "restart_context.md").write_text(
                    build_restart_context(restart_memory_path, target_score), encoding="utf-8"
                )
                run_cmd([
                    "python", "-m", "pipeline.run_direct_generation_pipeline",
                    "--config", config_path,
                    "--run-name", paths["gen_run_name"],
                    "--seed", str(seed + restart_count * 100),
                    *mock_args,
                ])
                version = 1
                current_reward = reward_path_for(cfg, paths["gen_run_name"], version)
                try:
                    check_reward_valid(cfg, paths["gen_run_name"], version, True)
                except (RuntimeError, FileNotFoundError) as exc:
                    print(f"Fresh duplicate recovery produced invalid code: {exc}")
                    stopped_reason = "stop_duplicate_recovery_invalid_keep_best"
                    write_experiment_summary(
                        cfg, prefix, seed, stopped_reason, best_iter, best_score,
                        target_score, rounds_completed,
                    )
                    return
                duplicate_match = find_identical_historical_reward(
                    cfg, prefix, seed, iteration_index, current_reward
                )
                if duplicate_match:
                    stopped_reason = "stop_duplicate_after_fresh_restart_keep_best"
                    print("Fresh regeneration is still identical. Stop without redundant training.")
                    write_experiment_summary(
                        cfg, prefix, seed, stopped_reason, best_iter, best_score,
                        target_score, rounds_completed,
                    )
                    return

        if best_reward and is_identical_reward(best_reward, current_reward):
            for retry in range(max_identical_retries):
                if iteration_index == 1 or force_fresh_restart or not use_reflection_agent:
                    break
                print(
                    f"Revision is identical to best (iter {best_iter}); "
                    f"requesting a local modification ({retry + 1}/{max_identical_retries})."
                )
                run_cmd([
                    "python", "-m", "pipeline.run_reflection_agent",
                    "--config", config_path,
                    "--previous-reward", str(previous_reward),
                    "--environment-card", str(build_paths(cfg, prefix, 1, seed)["gen_dir"] / "environment_card.md"),
                    "--train-run-dir", str(prev_paths["train_dir"]),
                    "--memory", agent_memory_path,
                    "--out-run-name", paths["gen_run_name"],
                    "--reward-version", f"v{version}",
                    "--best-reward", str(best_reward),
                    "--duplicate-retry",
                    "生成代码与历史 best 完全相同。请以 best 为基线完成诊断，并只修改一个目标组件；不得原样返回 best。",
                    *mock_args,
                ])
                current_reward = reward_path_for(cfg, paths["gen_run_name"], version)
                try:
                    check_reward_valid(cfg, paths["gen_run_name"], version, True)
                except (RuntimeError, FileNotFoundError) as e:
                    print(f"Best-revision retry produced invalid code: {e}")
                    continue
                if not is_identical_reward(best_reward, current_reward):
                    break

            if is_identical_reward(best_reward, current_reward):
                stopped_reason = "stop_identical_to_best_after_retries_keep_best"
                print("Revision remained identical to best. Stop to avoid a redundant training round.")
                write_experiment_summary(
                    cfg, prefix, seed, stopped_reason, best_iter, best_score,
                    target_score, rounds_completed,
                )
                return

        run_cmd([
            "python", "-m", "training.train_sb3_wrapper",
            "--config", config_path,
            "--reward", str(current_reward),
            "--run-name", paths["gen_run_name"].replace("/generation", "/training"),
            "--save-dir", str(paths["train_dir"]),
            "--total-timesteps", str(total_timesteps),
            "--eval-episodes", str(eval_episodes),
            "--seed", str(seed),
        ])
        rounds_completed = iteration_index

        current_score, _summary = read_training_score(paths["train_dir"])
        last_score = current_score

        previous_best_score = best_score
        is_new_best = previous_best_score is None or current_score > previous_best_score
        meaningful_improvement = (
            previous_best_score is None
            or current_score > previous_best_score + min_improvement
        )
        if is_new_best:
            best_score = current_score
            best_iter = iteration_index
            best_reward = current_reward
            copy_best_artifacts(
                cfg=cfg,
                prefix=prefix,
                seed=seed,
                iteration_index=iteration_index,
                reward_path=current_reward,
                reward_md_path=reward_md_path_for(cfg, paths["gen_run_name"], version),
                train_dir=paths["train_dir"],
                best_score=best_score,
                target_score=target_score,
            )
            decision = "new_best"
        else:
            decision = "no_meaningful_improvement"

        if meaningful_improvement:
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Track skeleton persistence: same component set = same skeleton family
        current_fingerprint = skeleton_fingerprint(current_reward)
        if current_fingerprint and current_fingerprint == last_skeleton_fingerprint:
            same_skeleton_count += 1
        else:
            same_skeleton_count = 1
        last_skeleton_fingerprint = current_fingerprint

        if current_score >= target_score:
            solved_seen = True
            if decision == "new_best":
                decision = "target_solved_new_best"
            else:
                decision = "target_solved_no_improvement"

        stop_now = False
        if solved_seen and stop_after_solved_drop and current_score < target_score:
            decision = "stop_after_solved_drop_keep_best"
            stopped_reason = decision
            stop_now = True
        elif solved_seen and no_improve_count >= patience_after_solved:
            decision = "stop_solved_no_improvement_keep_best"
            stopped_reason = decision
            stop_now = True
        elif (not solved_seen) and same_skeleton_count >= 4 and best_score < 0:
            decision = "same_skeleton_persistent_negative_fresh_restart"
            force_fresh_restart = True
            restart_count += 1
            stuck_rounds = same_skeleton_count
            same_skeleton_count = 0
            no_improve_count = 0
            print(f">>> Same skeleton for {stuck_rounds} rounds, best={best_score:.1f} still negative. Fresh restart #{restart_count}. Seed offset +{restart_count * 100}.")
        elif (
            (not solved_seen)
            and same_skeleton_count >= 4
            and 0 < best_score < 0.05 * target_score
            and no_improve_count >= 2
        ):
            decision = "same_skeleton_oscillation_fresh_restart"
            force_fresh_restart = True
            restart_count += 1
            stuck_rounds = same_skeleton_count
            same_skeleton_count = 0
            no_improve_count = 0
            print(f">>> Same skeleton for {stuck_rounds} rounds, best={best_score:.1f} positive but oscillating (<5% target). Fresh restart #{restart_count}. Seed offset +{restart_count * 100}.")
        elif (
            (not solved_seen)
            and no_improve_count >= patience_unsolved
            and best_score >= 0.05 * target_score
        ):
            decision = "unsolved_high_achievement_continue_from_best"
            no_improve_count = 0
            print(
                f">>> Stagnation detected, but best={best_score:.1f} already reaches "
                f"{best_score / target_score:.1%} of target. Keep best-centered local search; no fresh restart."
            )
        elif (not solved_seen) and no_improve_count >= patience_unsolved:
            # Check improvement ratio before triggering full restart.
            # If best_score improved significantly over the very first attempt,
            # the current direction has merit — keep searching locally.
            first_score = _first_iter_score(memory_path)
            if first_score is not None and first_score < 0 and best_score > 0:
                improvement_ratio = (best_score - first_score) / max(0.01, abs(first_score))
                if improvement_ratio > 2.0:
                    decision = "unsolved_improving_continue_from_best"
                    no_improve_count = 0
                    print(
                        f">>> Stagnation but best={best_score:.1f} is a {improvement_ratio:.1f}x "
                        f"improvement over first={first_score:.1f}. Continue local search; no fresh restart."
                    )
                else:
                    decision = "unsolved_stagnation_fresh_restart"
                    force_fresh_restart = True
                    restart_count += 1
                    same_skeleton_count = 0
                    no_improve_count = 0
                    print(f">>> Unsolved stagnation after {patience_unsolved} iters. Fresh restart #{restart_count} (full regeneration, seed offset +{restart_count * 100}).")
            else:
                decision = "unsolved_stagnation_fresh_restart"
                force_fresh_restart = True
                restart_count += 1
                same_skeleton_count = 0
                no_improve_count = 0
                print(f">>> Unsolved stagnation after {patience_unsolved} iters. Fresh restart #{restart_count} (full regeneration, seed offset +{restart_count * 100}).")

        new_lessons_arg = []
        if iteration_index > 1:
            diag_file = paths["iter_root"] / "diagnosis.json"
            if diag_file.exists():
                try:
                    diag = json.loads(diag_file.read_text(encoding="utf-8"))
                    nl = diag.get("new_lessons", [])
                    if nl:
                        new_lessons_arg = ["--new-lessons", json.dumps(nl, ensure_ascii=False)]
                except Exception:
                    pass
        run_cmd([
            "python", "-m", "pipeline.run_06_update_reward_memory",
            "--iter", str(iteration_index),
            "--train-run-dir", str(paths["train_dir"]),
            "--memory", memory_path,
            "--target-score", str(target_score),
            "--best-score", str(best_score),
            "--best-iter", str(best_iter),
            "--decision", decision,
            *new_lessons_arg,
        ])

        previous_reward = current_reward

        if stop_now:
            print(f"Early stop: {stopped_reason}")
            break

    write_experiment_summary(cfg, prefix, seed, stopped_reason, best_iter, best_score, target_score, rounds_completed)

    print("\n" + "=" * 60)
    print("Done. Key files")
    print("=" * 60)
    for iteration_index in range(1, rounds_completed + 1):
        paths = build_paths(cfg, prefix, iteration_index, seed)
        version = iteration_index
        print(f"iter_{paths['iter_pad']}:")
        print(f"  reward   : {reward_path_for(cfg, paths['gen_run_name'], version)}")
        print(f"  feedback : {paths['train_dir'] / 'training_feedback.md'}")
        if iteration_index > 1:
            print(f"  context  : {paths['context_path']}")
    print(f"memory: {memory_path}")
    print(f"best: {experiment_root_for(cfg, prefix, seed) / 'best' / 'best_reward.py'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/env001_deepseek_rag.yaml")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--total-timesteps", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume-from", type=int, default=None)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    mock = True if args.mock else None
    run_iterative_experiment(
        config_path=args.config,
        prefix=args.prefix,
        rounds=args.rounds,
        total_timesteps=args.total_timesteps,
        eval_episodes=args.eval_episodes,
        mock=mock,
        seed=args.seed,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
