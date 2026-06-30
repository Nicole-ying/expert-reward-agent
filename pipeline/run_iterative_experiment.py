import argparse
import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from .common import load_config


def run_cmd(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def pad_iter(i):
    return f"{i:02d}"


def validate_rounds(rounds):
    if rounds < 1:
        raise ValueError("iteration.total_rounds must be >= 1")


def maybe_reset_memory(memory_path, reset):
    p = Path(memory_path)
    if reset and p.exists():
        p.unlink()
    p.parent.mkdir(parents=True, exist_ok=True)


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
        raise RuntimeError(f"Reward v{version} failed validation: {path}")


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


def is_identical_reward(path_a, path_b):
    if not path_a or not path_b:
        return False
    if not Path(path_a).exists() or not Path(path_b).exists():
        return False
    return code_signature(path_a) == code_signature(path_b)


def build_agent_context_header(iteration_index, target_score, best_score, best_iter, last_score, no_improve_count, solved_seen):
    best_text = "N/A" if best_score is None else f"{best_score:.3f}"
    best_iter_text = "N/A" if best_iter is None else str(best_iter)
    last_text = "N/A" if last_score is None else f"{last_score:.3f}"

    if solved_seen:
        trend = "solved"
        guidance = "Prefer small tunes over large changes. Preserve what works."
        suggest = "tune"
    elif last_score is not None and best_score is not None and last_score < best_score - 20:
        trend = "declining_from_best"
        guidance = "Score dropped significantly from best. Investigate what changed and revert harmful modifications."
        suggest = "tune (revert harmful changes)"
    elif no_improve_count >= 3:
        trend = "stagnant"
        guidance = "Current skeleton family has been tried 3+ rounds with no improvement. The KB skeleton suggestions below are DIFFERENT architectures — you MUST try one. Do NOT stay on the same skeleton with coefficient tuning."
        suggest = "rebuild"
    elif no_improve_count >= 1:
        trend = "stalling"
        guidance = "Current skeleton is not improving. Consider mix (add/delete components) or rebuild (switch skeleton family from KB suggestions)."
        suggest = "mix or rebuild"
    elif last_score is not None and best_score is not None and last_score < best_score:
        trend = "below_best"
        guidance = "Investigate why score dropped from best. Consider reverting harmful changes."
        suggest = "tune or mix"
    else:
        trend = "searching"
        guidance = "Continue refining based on evidence."
        suggest = "tune or mix"

    return f"""# Agent Context

- iteration: {iteration_index}
- target_score: {target_score:.3f}
- best_score: {best_text} (iter {best_iter_text})
- current_score: {last_text}
- stagnation_rounds: {no_improve_count}
- trend: {trend}
- guidance: {guidance}
- suggested_action: {suggest}

When suggested_action is rebuild, the current skeleton family has FAILED.
You MUST pick a different architecture from the KB Skeleton Suggestions below.
Do NOT return another coefficient-tuned variant of the same skeleton.
"""


def prepend_agent_context(context_path, header_text):
    p = Path(context_path)
    old = p.read_text(encoding="utf-8") if p.exists() else ""
    p.write_text(header_text.strip() + "\n\n" + old, encoding="utf-8")


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
        shutil.copy2(summary, best_dir / "best_training_summary.json")

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


def run_iterative_experiment(config_path, prefix=None, rounds=None, total_timesteps=None, eval_episodes=None, mock=None, seed=0):
    cfg = load_config(config_path)
    iter_cfg = cfg.get("iteration", {})
    train_cfg = cfg.get("training", {})
    rag_cfg = cfg.get("rag", {})

    prefix = prefix or iter_cfg.get("experiment_prefix", "exp_iter")
    rounds = int(rounds if rounds is not None else iter_cfg.get("total_rounds", 10))
    validate_rounds(rounds)

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

    memory_path = str(experiment_root_for(cfg, prefix, seed) / "memory" / "reward_memory.md")
    cards_path = rag_cfg.get("reward_misalignment_cards_path", "knowledge_base/iteration/reward_misalignment_cards.md")
    maybe_reset_memory(memory_path, bool(iter_cfg.get("reset_memory_at_start", True)))

    print("=" * 60)
    print("Config-driven iterative reward experiment")
    print("=" * 60)
    print(f"config          : {config_path}")
    print(f"prefix          : {prefix}")
    print(f"seed            : {seed}")
    print(f"rounds          : {rounds}")
    print(f"target_score    : {target_score}")
    print(f"total_timesteps : {total_timesteps}")
    print(f"eval_episodes   : {eval_episodes}")
    print(f"memory_path     : {memory_path}")
    print(f"cards_path      : {cards_path}")
    print(f"cards_top_k     : {cards_top_k}")
    print(f"mock_llm        : {use_mock}")

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

    for iteration_index in range(1, rounds + 1):
        version = iteration_index
        paths = build_paths(cfg, prefix, iteration_index, seed)
        mock_args = ["--mock"] if use_mock else []

        print("\n" + "-" * 60)
        print(f"Iteration {iteration_index}/{rounds}")
        print("-" * 60)

        if iteration_index == 1 or force_fresh_restart:
            print(">>> Fresh restart: re-running full generation pipeline")
            if force_fresh_restart:
                version = 1  # generation pipeline always outputs reward_v1
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
            run_cmd([
                "python", "-m", "pipeline.run_04_build_iteration_context",
                "--train-run-dir", str(prev_paths["train_dir"]),
                "--memory", memory_path,
                "--cards", cards_path,
                "--top-k", str(cards_top_k),
                "--out", str(paths["context_path"]),
                "--config", config_path,
                *mock_args,
            ])
            header_text = build_agent_context_header(
                iteration_index=iteration_index,
                target_score=target_score,
                best_score=best_score,
                best_iter=best_iter,
                last_score=last_score,
                no_improve_count=no_improve_count,
                solved_seen=solved_seen,
            )
            prepend_agent_context(paths["context_path"], header_text)

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

        check_reward_valid(cfg, paths["gen_run_name"], version, stop_on_invalid)

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

        improved = best_score is None or current_score > best_score + min_improvement
        if improved:
            best_score = current_score
            best_iter = iteration_index
            best_reward = current_reward
            no_improve_count = 0
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
            no_improve_count += 1
            decision = "no_meaningful_improvement"

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
        elif (not solved_seen) and no_improve_count >= patience_unsolved:
            decision = "unsolved_stagnation_fresh_restart"
            force_fresh_restart = True
            restart_count += 1
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
    )


if __name__ == "__main__":
    main()
