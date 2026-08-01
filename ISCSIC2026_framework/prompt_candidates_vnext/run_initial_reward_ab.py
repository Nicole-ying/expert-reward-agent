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


def compose_environment_card(task_spec: str, analysis: str) -> str:
    return (
        "# Environment Semantics Card\n\n"
        "## 0. Original anonymized task specification\n\n"
        "```yaml\n"
        f"{task_spec.strip()}\n"
        "```\n\n"
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


def build_environment_user_prompt(task_spec: str, masked_step: str) -> str:
    return (
        "# ANONYMIZED_TASK_SPEC\n\n"
        f"{task_spec.strip()}\n\n"
        "# MASKED_STEP_SOURCE\n\n"
        f"```python\n{masked_step.strip()}\n```\n"
    )


def build_reward_user_prompt(environment_card: str, expert_context: str | None) -> str:
    parts = [environment_card.strip()]
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

    task_spec = read_text(task_spec_path)
    masked_step = read_text(masked_step_path)
    environment_system = read_text(SCRIPT_DIR / "01_environment_semantics_prompt.md")
    reward_system = read_text(SCRIPT_DIR / "02_initial_reward_generator_prompt.md")
    environment_user = build_environment_user_prompt(task_spec, masked_step)

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

    llm_cfg = cfg["llm"]
    env_model = args.env_model or args.model or llm_cfg["model_env"]
    reward_model = args.reward_model or args.model or llm_cfg["model_reward"]
    manifest = {
        "config": str(config_path),
        "task_spec": str(task_spec_path),
        "masked_step": str(masked_step_path),
        "env_model": env_model,
        "reward_model": reward_model,
        "expert_mode": args.expert_mode,
        "dry_run": args.dry_run,
    }
    write_json(output_root / "manifest.json", manifest)

    if args.dry_run:
        write_text(
            output_root / "DRY_RUN.md",
            "Prompts were assembled but no LLM request was made. Remove --dry-run to execute.\n",
        )
        return output_root

    client = DeepSeekClient(api_key_env=args.api_key_env, base_url=args.base_url or llm_cfg["base_url"])
    environment_response = client.chat(
        model=env_model,
        system_prompt=environment_system,
        user_prompt=environment_user,
        temperature=args.env_temperature,
        max_tokens=args.env_max_tokens,
    )
    write_text(output_root / "environment/raw_response.md", environment_response)
    environment_card = compose_environment_card(task_spec, environment_response)
    write_text(output_root / "environment/environment_card.md", environment_card)

    custom_context = resolve_from_root(args.expert_context_file) if args.expert_context_file else None
    variants = reward_variants(args.expert_mode, custom_context)
    comparison = []
    for name, expert_context in variants.items():
        variant_dir = output_root / "rewards" / name
        reward_user = build_reward_user_prompt(environment_card, expert_context)
        save_prompt_record(variant_dir / "prompt_record.md", reward_system, reward_user)
        stats = prompt_stats(reward_system, reward_user)
        write_json(variant_dir / "prompt_stats.json", stats)
        if expert_context:
            write_text(variant_dir / "expert_context.md", expert_context)

        response = client.chat(
            model=reward_model,
            system_prompt=reward_system,
            user_prompt=reward_user,
            temperature=args.reward_temperature,
            max_tokens=args.reward_max_tokens,
        )
        write_text(variant_dir / "raw_response.md", response)
        code = extract_code(response)
        write_text(variant_dir / "reward_v1.py", code + ("\n" if code else ""))
        validation = validate_code(code)
        keys = component_keys(code)
        validation["component_keys"] = keys
        validation["component_count"] = len(keys)
        validation["component_budget_2_to_4"] = 2 <= len(keys) <= 4
        if keys and not validation["component_budget_2_to_4"]:
            validation["warnings"].append(
                "Initial reward falls outside the recommended 2–4 component budget; inspect the design justification."
            )
        write_json(variant_dir / "validation.json", validation)
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
    parser.add_argument("--output-root", default="runs/vnext_initial_ab")
    parser.add_argument("--run-name")
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
