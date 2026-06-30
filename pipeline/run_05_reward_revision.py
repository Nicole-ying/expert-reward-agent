import argparse
from pathlib import Path

from .common import load_config, read_text, write_text, write_json, record_prompt, record_response
from .run_03_direct_reward_generator import extract_code, validate_code, estimate_tokens
from llm_clients.deepseek_client import DeepSeekClient


ENV_CONTRACT = """# environment_contract

- function_signature: def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
- allowed: obs[0..7], next_obs[0..7], action (0=noop,1=left,2=main,3=right), info (no reliable fields)
- forbidden: original_reward, official_reward, fitness_score, individual_reward, info['success'], info['failure'], info['termination_reason']
- terminal_success_reward and terminal_failure_penalty remain blocked (no explicit success/failure flag available).
"""


MOCK_REVISION_MD = """# reward_revision.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = 6.0 * (current_dist - next_dist)
    distance_anchor = -0.05 * next_dist

    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    speed_penalty = -0.08 * speed
    stability_penalty = -0.08 * abs(next_obs[4]) - 0.04 * abs(next_obs[5])

    near_target_quality = 1.0 / (1.0 + 4.0 * next_dist)
    low_speed_quality = 1.0 / (1.0 + 2.0 * speed)
    angle_quality = 1.0 / (1.0 + 5.0 * abs(next_obs[4]))
    landing_quality = 0.4 * near_target_quality * low_speed_quality * angle_quality

    total_reward = progress_reward + distance_anchor + speed_penalty + stability_penalty + landing_quality
    components = {
        "progress_reward": progress_reward,
        "distance_anchor": distance_anchor,
        "speed_penalty": speed_penalty,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality,
        "total_reward": total_reward,
    }
    return float(total_reward), components
```

# revision notes

- kept progress_delta_reward;
- weakened speed_penalty;
- revised sparse soft_landing_bonus into smooth landing_quality;
- added a small distance_anchor;
- terminal success/failure rewards remain blocked.
"""


def write_revision_prompt_stats(run_dir, system_prompt, user_prompt):
    total_text = system_prompt + "\n" + user_prompt
    lines = []
    lines.append("# Reward Revision Prompt Stats")
    lines.append("")
    lines.append("Token count is an estimate because the exact tokenizer depends on the LLM provider.")
    lines.append("")
    lines.append("| part | chars | lines | estimated_tokens |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| system_prompt | {len(system_prompt)} | {system_prompt.count(chr(10)) + 1} | {estimate_tokens(system_prompt)} |")
    lines.append(f"| user_prompt | {len(user_prompt)} | {user_prompt.count(chr(10)) + 1} | {estimate_tokens(user_prompt)} |")
    lines.append(f"| total | {len(total_text)} | {total_text.count(chr(10)) + 1} | {estimate_tokens(total_text)} |")
    write_text(Path(run_dir) / "prompt_records/03_reward_revision.prompt_stats.md", "\n".join(lines) + "\n")


def run(config_path, previous_reward_path, iteration_context_path, out_run_name, reward_version, mock=False, best_reward_path=None):
    cfg = load_config(config_path)
    run_dir = Path(cfg["experiment"]["run_root"]) / out_run_name
    for sub in ["llm_inputs", "prompt_records", "response_records", "validations"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    prompt_path = cfg.get("prompts", {}).get("reward_revision", "prompts/03_reward_revision_prompt.md")
    system_prompt = read_text(prompt_path)
    previous_reward = read_text(previous_reward_path)
    iteration_context = read_text(iteration_context_path)

    # Find expert_reward_context, strip v1-generation-only sections
    expert_ctx = ""
    gen_dir = Path(previous_reward_path).parent
    ctx_path = gen_dir / "expert_reward_context.md"
    if not ctx_path.exists():
        seed_dir = gen_dir.parent.parent
        for d in sorted(Path(seed_dir).glob("iter_*/generation/expert_reward_context.md"), reverse=True):
            ctx_path = d
            break
    if ctx_path.exists():
        expert_ctx = read_text(str(ctx_path))
        # Strip v1 generation instructions (noise for revision)
        import re as _re
        expert_ctx = _re.split(r"\n## \d+\. reward_v1 生成要求", expert_ctx)[0]

    # Find environment_card for task context
    env_card = ""
    for d in sorted(Path(seed_dir).glob("iter_*/generation/environment_card.md"), reverse=True):
        # Extract sections 1-5 (task goal, type, obs, action, termination)
        raw = read_text(str(d))
        parts = raw.split("## 6. reward")[0] if "## 6. reward" in raw else raw
        env_card = parts.strip()
        break

    # Extract score info from memory for previous/best annotations
    import json as _json
    prev_score = "?"
    best_score = "?"
    memory_path = str(Path(iteration_context_path).parent.parent / "memory" / "reward_memory.md")
    try:
        for line in Path(memory_path).read_text(encoding="utf-8").splitlines():
            if line.startswith("|"):
                cols = [c.strip() for c in line.split("|")]
                if len(cols) >= 4:
                    try:
                        s = float(cols[3])
                        if s > float(best_score.replace("?", "-999")):
                            best_score = cols[3]
                    except: pass
    except: pass
    # Get prev score from training_feedback
    prev_train = gen_dir.parent / "training" / "training_feedback.md"
    if prev_train.exists():
        fb = read_text(str(prev_train))
        m = _re.search(r"score=([-\d.]+)", fb)
        if m: prev_score = m.group(1)

    user_parts = [
        "# Environment",
        env_card if env_card else ENV_CONTRACT,
        ENV_CONTRACT,
    ]
    if expert_ctx:
        user_parts += [
            "# Expert Knowledge",
            expert_ctx,
        ]
    user_parts += [
        f"# previous_reward.py (score: {prev_score})",
        "```python\n" + previous_reward.strip() + "\n```",
    ]
    if best_reward_path and Path(best_reward_path).exists():
        best_reward = read_text(best_reward_path)
        user_parts += [
            f"# best_reward.py (score: {best_score})",
            "```python\n" + best_reward.strip() + "\n```",
        ]
    user_parts += [
        "# iteration_context.md",
        iteration_context,
    ]
    user_prompt = "\n\n".join(user_parts)

    write_text(run_dir / f"llm_inputs/reward_{reward_version}_revision.input.md", user_prompt)
    record_prompt(run_dir, "03_reward_revision", system_prompt, user_prompt)
    write_revision_prompt_stats(run_dir, system_prompt, user_prompt)

    if mock:
        out_md = MOCK_REVISION_MD
    else:
        llm_cfg = cfg["llm"]
        client = DeepSeekClient(api_key_env=llm_cfg["api_key_env"], base_url=llm_cfg["base_url"])
        out_md = client.chat(
            model=llm_cfg["model_reward"],
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=llm_cfg["temperature_reward_generator"],
            max_tokens=llm_cfg["max_tokens_reward"],
            json_mode=False,
        )

    code = extract_code(out_md)
    validation = validate_code(code)

    write_text(run_dir / f"reward_{reward_version}.md", out_md)
    write_text(run_dir / f"reward_{reward_version}.py", code)
    write_json(run_dir / f"validations/reward_{reward_version}.validation.json", validation)
    record_response(run_dir, "03_reward_revision", out_md)

    if not validation["valid"]:
        print("WARNING: reward revision validation failed")
        for e in validation["errors"]:
            print(" -", e)
    if validation.get("warnings"):
        print("reward revision validation warnings")
        for w in validation["warnings"]:
            print(" -", w)

    print(run_dir / f"reward_{reward_version}.py")
    print(run_dir / f"reward_{reward_version}.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/env001_deepseek_rag.yaml")
    ap.add_argument("--previous-reward", required=True)
    ap.add_argument("--best-reward", default=None)
    ap.add_argument("--iteration-context", required=True)
    ap.add_argument("--out-run-name", required=True)
    ap.add_argument("--reward-version", default="v2")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    run(args.config, args.previous_reward, args.iteration_context, args.out_run_name, args.reward_version, args.mock, args.best_reward)
