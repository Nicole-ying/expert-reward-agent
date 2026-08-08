"""Run the vNext environment-card -> initial-reward chain.

The script generates one shared environment card and then compares two reward
generation conditions by default:

1. card_only: the concise environment card only;
2. historical_expert: the same card plus the historical fixed expert schema.

Every prompt, raw response, extracted reward, and validation report is saved so
the effect of expert context can be audited instead of assumed.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
FRAMEWORK_ROOT = SCRIPT_DIR.parent
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))

from llm_clients.deepseek_client import DeepSeekClient  # noqa: E402
from pipeline.common import load_config  # noqa: E402
from pipeline.run_03_direct_reward_generator import (  # noqa: E402
    estimate_tokens,
    extract_code,
    validate_code,
)
from rag.direct_context_builder import EXPERT_SCHEMA_CONTEXT  # noqa: E402


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def resolve_from_root(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else FRAMEWORK_ROOT / path


def resolve_prompt_file(prompt_dir: Path, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        path = prompt_dir / name
        if path.is_file():
            return path
    expected = ", ".join(candidates)
    raise FileNotFoundError(f"Prompt directory {prompt_dir} contains none of: {expected}")


def strip_environment_heading(response: str) -> str:
    """Avoid two top-level card headings after controller composition."""
    text = response.strip()
    text = re.sub(
        r"\A#\s+(?:Environment Semantics Card|Environment Analysis)\s*\n+",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    return text.strip()


def extract_task_description(task_spec: str) -> str:
    parsed = yaml.safe_load(task_spec) or {}
    description = parsed.get("task_description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("task_spec must contain a non-empty task_description string")
    return description.strip()


def compose_environment_card(task_description: str, analysis: str) -> str:
    return (
        "# Environment Semantics Card\n\n"
        "## 0. Original anonymized task description\n\n"
        f"{task_description.strip()}\n\n"
        f"{strip_environment_heading(analysis)}\n"
    )


def prompt_stats(system_prompt: str, user_prompt: str) -> dict[str, int]:
    return {
        "system_chars": len(system_prompt),
        "user_chars": len(user_prompt),
        "total_chars": len(system_prompt) + len(user_prompt),
        "estimated_tokens": estimate_tokens(system_prompt + "\n" + user_prompt),
    }


def component_keys(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if not any(name in {"components", "reward_components", "reward_terms"} for name in names):
            continue
        keys = []
        for key in node.value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.append(key.value)
            else:
                keys.append("<dynamic-key>")
        return keys
    return []


def save_prompt_record(path: Path, system_prompt: str, user_prompt: str) -> None:
    write_text(
        path,
        "# Prompt Record\n\n"
        "## System Prompt\n\n"
        f"````text\n{system_prompt.strip()}\n````\n\n"
        "## User Prompt\n\n"
        f"````markdown\n{user_prompt.strip()}\n````\n",
    )


def build_environment_user_prompt(
    task_spec: str,
    masked_step: str,
    reward_clip: float | None,
    episode_step_limit: int | None,
) -> str:
    clip_line = (
        "Runtime total-reward clipping: disabled."
        if reward_clip is None
        else f"Runtime total-reward clipping: [-{float(reward_clip)}, +{float(reward_clip)}] after compute_reward returns."
    )
    step_limit_line = (
        "Configured maximum episode steps: unknown."
        if episode_step_limit is None
        else f"Configured maximum episode steps: {int(episode_step_limit)}."
    )
    return (
        "# ANONYMIZED_TASK_SPEC\n\n"
        f"{task_spec.strip()}\n\n"
        "# MASKED_STEP_SOURCE\n\n"
        f"```python\n{masked_step.strip()}\n```\n\n"
        "# REWARD_INTERFACE_CONTRACT\n\n"
        "```text\n"
        "def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):\n\n"
        "Runtime-accessible inputs:\n"
        "- obs, action, next_obs, training_progress\n"
        "- info[\"terminated\"]: bool\n"
        "- info[\"truncated\"]: bool\n"
        "- info[\"done\"]: bool(terminated or truncated)\n\n"
        "Runtime episode-limit semantics:\n"
        "- the masked raw-step may return truncated=False, but gym.make can add an outer TimeLimit wrapper\n"
        "- therefore info[\"truncated\"] may become True at the configured episode-step limit\n"
        "- treat truncation as budget exhaustion, not automatically as success or failure\n\n"
        f"{clip_line}\n\n"
        f"{step_limit_line}\n\n"
        "Forbidden:\n"
        "- original_reward and the official environment reward\n"
        "- bare variables terminated, truncated, or done\n"
        "- undeclared info fields\n\n"
        "The termination boolean does not expose its cause. Distinguish success/failure only by combining "
        "info[\"terminated\"] with legal state evidence.\n"
        "```\n"
    )


def missing_environment_sections(text: str) -> list[int]:
    return [index for index in range(1, 9) if not re.search(rf"(?m)^## {index}\.\s", text)]


def undefined_names(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not functions:
        return []
    function = functions[0]
    defined = {arg.arg for candidate in functions for arg in candidate.args.args}
    defined.update(
        node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )
    defined.update(node.name for node in ast.walk(function) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    safe_builtins = {
        "abs", "all", "any", "bool", "dict", "float", "int", "isinstance", "len",
        "list", "max", "min", "range", "round", "sum", "tuple",
    }
    loaded = {
        node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    return sorted(loaded - defined - safe_builtins)


def extra_function_names(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name != "compute_reward"
    ]


def audit_reward_code(code: str) -> tuple[dict[str, object], list[str]]:
    validation = validate_code(code)
    keys = component_keys(code)
    missing_names = undefined_names(code)
    helpers = extra_function_names(code)
    if missing_names:
        validation["errors"].append(f"Undefined names: {', '.join(missing_names)}")
    if helpers:
        validation["errors"].append(f"Extra or nested functions are forbidden: {', '.join(helpers)}")
    validation["valid"] = not validation["errors"]
    validation["component_keys"] = keys
    validation["component_count"] = len(keys)
    validation["component_budget_2_to_4"] = 2 <= len(keys) <= 4
    if keys and not validation["component_budget_2_to_4"]:
        validation["warnings"].append(
            "Initial reward falls outside the recommended 2–4 component budget; inspect the design justification."
        )
    return validation, keys


def build_reward_user_prompt(
    environment_card: str,
    expert_context: str | None,
    reward_clip: float | None,
    episode_step_limit: int | None,
) -> str:
    clip_line = (
        "- total-reward clipping: disabled"
        if reward_clip is None
        else f"- total-reward clipping after compute_reward returns: [-{float(reward_clip)}, +{float(reward_clip)}]"
    )
    step_limit_line = (
        "- maximum episode steps: unknown; use a conservative accumulation bound"
        if episode_step_limit is None
        else f"- maximum episode steps: {int(episode_step_limit)}"
    )
    parts = [
        environment_card.strip(),
        "# Authoritative Reward Runtime Contract\n"
        f"{clip_line}\n"
        f"{step_limit_line}\n"
        "- this contract overrides any missing or conflicting statement in the Environment Card\n"
        "- design event magnitudes and the scale audit using the effective post-clip reward seen by PPO",
    ]
    if expert_context:
        parts.extend(
            [
                "# Optional Expert Context (advisory only)",
                "Use this only after the environment card determines the task semantics and legal signals. "
                "Do not add a component merely because this context mentions it.",
                expert_context.strip(),
            ]
        )
    return "\n\n".join(parts) + "\n"


def reward_variants(mode: str, custom_context: Path | None) -> dict[str, str | None]:
    variants: dict[str, str | None] = {}
    if mode in {"card_only", "both"}:
        variants["card_only"] = None
    if mode in {"historical_expert", "both"}:
        variants["historical_expert"] = EXPERT_SCHEMA_CONTEXT
    if custom_context:
        variants["custom_expert"] = read_text(custom_context)
    return variants


def run(args: argparse.Namespace) -> Path:
    config_path = resolve_from_root(args.config)
    cfg = load_config(config_path)
    task_spec_path = resolve_from_root(args.task_spec or cfg["inputs"]["task_spec_path"])
    masked_step_path = resolve_from_root(args.masked_step or cfg["inputs"]["masked_step_path"])
    prompt_dir = resolve_from_root(args.prompt_dir)
    environment_prompt_path = resolve_prompt_file(
        prompt_dir,
        ("01_environment_semantics_prompt.md", "01_environment_analyzer_prompt.md"),
    )
    reward_prompt_path = resolve_prompt_file(
        prompt_dir,
        ("02_initial_reward_generator_prompt.md", "02_reward_generator_prompt.md"),
    )

    task_spec = read_text(task_spec_path)
    task_description = extract_task_description(task_spec)
    masked_step = read_text(masked_step_path)
    environment_system = read_text(environment_prompt_path)
    reward_system = read_text(reward_prompt_path)
    train_cfg = cfg.get("training", {})
    reward_clip = train_cfg.get("reward_clip", 20.0)
    episode_step_limit = train_cfg.get("episode_step_limit")
    if episode_step_limit is None:
        try:
            import gymnasium as gym

            env_spec = gym.spec(train_cfg["runner_env_id"])
            episode_step_limit = env_spec.max_episode_steps
        except Exception:
            episode_step_limit = None
    environment_user = build_environment_user_prompt(
        task_spec,
        masked_step,
        reward_clip,
        episode_step_limit,
    )

    llm_cfg = cfg["llm"]
    env_model = args.env_model or args.model or llm_cfg["model_env"]
    reward_model = args.reward_model or args.model or llm_cfg["model_reward"]
    if args.resume_run:
        output_root = resolve_from_root(args.resume_run)
        if not output_root.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {output_root}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = resolve_from_root(args.output_root) / (args.run_name or timestamp)
        output_root.mkdir(parents=True, exist_ok=False)

        write_text(output_root / "inputs/task_spec_anonymized.yaml", task_spec)
        write_text(output_root / "inputs/masked_step_source.py", masked_step)
        save_prompt_record(
            output_root / "environment/prompt_record.md",
            environment_system,
            environment_user,
        )
        write_json(
            output_root / "environment/prompt_stats.json",
            prompt_stats(environment_system, environment_user),
        )

    manifest = {
        "config": str(config_path),
        "task_spec": str(task_spec_path),
        "masked_step": str(masked_step_path),
        "prompt_dir": str(prompt_dir),
        "environment_prompt": str(environment_prompt_path),
        "reward_prompt": str(reward_prompt_path),
        "env_model": env_model,
        "reward_model": reward_model,
        "expert_mode": args.expert_mode,
        "dry_run": args.dry_run,
        "resumed": bool(args.resume_run),
    }
    write_json(output_root / "manifest.json", manifest)

    if args.dry_run:
        write_text(
            output_root / "DRY_RUN.md",
            "Prompts were assembled but no LLM request was made. Remove --dry-run to execute.\n",
        )
        return output_root

    client = DeepSeekClient(api_key_env=args.api_key_env, base_url=args.base_url or llm_cfg["base_url"])
    environment_card_path = output_root / "environment/environment_card.md"
    if args.resume_run:
        if not environment_card_path.exists():
            raise FileNotFoundError(f"Resume run has no environment card: {environment_card_path}")
        environment_card = read_text(environment_card_path)
    else:
        environment_response = client.chat(
            model=env_model,
            system_prompt=environment_system,
            user_prompt=environment_user,
            temperature=args.env_temperature,
            max_tokens=args.env_max_tokens,
        )
        write_text(output_root / "environment/raw_response.md", environment_response)
        missing = missing_environment_sections(environment_response)
        write_json(
            output_root / "environment/validation.json",
            {"complete": not missing, "missing_sections": missing},
        )
        if missing:
            raise RuntimeError(
                f"Environment card is incomplete; missing sections {missing}. "
                "No reward-generation calls were made. Inspect environment/raw_response.md."
            )
        environment_card = compose_environment_card(task_description, environment_response)
        write_text(environment_card_path, environment_card)

    custom_context = resolve_from_root(args.expert_context_file) if args.expert_context_file else None
    variants = reward_variants(args.expert_mode, custom_context)
    comparison = []
    for name, expert_context in variants.items():
        variant_dir = output_root / "rewards" / name
        reward_user = build_reward_user_prompt(
            environment_card,
            expert_context,
            reward_clip,
            episode_step_limit,
        )
        save_prompt_record(variant_dir / "prompt_record.md", reward_system, reward_user)
        stats = prompt_stats(reward_system, reward_user)
        write_json(variant_dir / "prompt_stats.json", stats)
        if expert_context:
            write_text(variant_dir / "expert_context.md", expert_context)

        validation_path = variant_dir / "validation.json"
        reward_path = variant_dir / "reward_v1.py"
        response_path = variant_dir / "raw_response.md"
        reuse_completed = bool(
            args.resume_run
            and validation_path.exists()
            and reward_path.exists()
            and response_path.exists()
        )
        if reuse_completed:
            validation, keys = audit_reward_code(read_text(reward_path))
            write_json(validation_path, validation)
            if args.retry_invalid and not validation["valid"]:
                reuse_completed = False
        if not reuse_completed:
            response = client.chat(
                model=reward_model,
                system_prompt=reward_system,
                user_prompt=reward_user,
                temperature=args.reward_temperature,
                max_tokens=args.reward_max_tokens,
            )
            write_text(response_path, response)
            code = extract_code(response)
            write_text(reward_path, code + ("\n" if code else ""))
            validation, keys = audit_reward_code(code)
            write_json(validation_path, validation)
        comparison.append(
            {
                "variant": name,
                "valid": validation["valid"],
                "component_count": len(keys),
                "component_keys": keys,
                "prompt_estimated_tokens": stats["estimated_tokens"],
                "errors": validation["errors"],
                "warnings": validation["warnings"],
            }
        )

    write_json(output_root / "comparison.json", comparison)
    table = [
        "# Initial Reward A/B Comparison",
        "",
        "| variant | valid | components | estimated prompt tokens |",
        "|---|---|---|---:|",
    ]
    for row in comparison:
        table.append(
            f"| {row['variant']} | {row['valid']} | "
            f"{', '.join(row['component_keys']) or 'none'} ({row['component_count']}) | "
            f"{row['prompt_estimated_tokens']} |"
        )
    table.extend(
        [
            "",
            "Static validity and component count do not establish reward quality. "
            "Inspect both designs first; policy-training/native-evaluation evidence is required for a performance conclusion.",
            "",
        ]
    )
    write_text(output_root / "comparison.md", "\n".join(table))
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a vNext environment card and A/B initial rewards with or without expert context."
    )
    parser.add_argument("--config", default="configs/env001_paper_v4.yaml")
    parser.add_argument("--task-spec")
    parser.add_argument("--masked-step")
    parser.add_argument("--prompt-dir", default="prompt_candidates_vnext")
    parser.add_argument("--output-root", default="runs/vnext_initial_ab")
    parser.add_argument("--run-name")
    parser.add_argument("--resume-run", help="Resume an existing output directory and skip completed stages.")
    parser.add_argument(
        "--retry-invalid",
        action="store_true",
        help="When resuming, regenerate only reward variants whose saved code fails validation.",
    )
    parser.add_argument("--expert-mode", choices=["card_only", "historical_expert", "both"], default="both")
    parser.add_argument("--expert-context-file")
    parser.add_argument("--model")
    parser.add_argument("--env-model")
    parser.add_argument("--reward-model")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url")
    parser.add_argument("--env-temperature", type=float, default=0.0)
    parser.add_argument("--reward-temperature", type=float, default=0.15)
    parser.add_argument("--env-max-tokens", type=int, default=5000)
    parser.add_argument("--reward-max-tokens", type=int, default=6000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(result)
