import argparse
import ast
import json
import re
from pathlib import Path
from .common import load_config, read_text, write_text, write_json, record_prompt, record_response
from llm_clients.deepseek_client import DeepSeekClient

MOCK_REWARD_MD = """# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    old_x = obs[0]
    old_y = obs[1]
    new_x = next_obs[0]
    new_y = next_obs[1]
    new_vx = next_obs[2]
    new_vy = next_obs[3]
    new_angle = next_obs[4]
    new_angular_velocity = next_obs[5]

    old_distance = (old_x * old_x + old_y * old_y) ** 0.5
    new_distance = (new_x * new_x + new_y * new_y) ** 0.5
    next_speed = (new_vx * new_vx + new_vy * new_vy) ** 0.5

    progress_reward = old_distance - new_distance
    stability_penalty = -0.03 * next_speed - 0.03 * abs(new_angle) - 0.01 * abs(new_angular_velocity)

    total_reward = progress_reward + stability_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "total_reward": total_reward
    }
    if isinstance(info, dict):
        info["reward_terms"] = components

    return float(total_reward), components
```

# reward_v1 设计说明

- 使用组件：progress_delta_reward + light stability_penalty。
- progress_delta_reward 负责主学习信号：奖励智能体每一步更接近目标区域。
- stability_penalty 负责轻量稳定约束：防止高速、姿态过大或角速度过大。
- 没有使用 terminal_success_reward，因为环境卡片说明 explicit_success_flag_available=false。
- 没有使用 terminal_failure_penalty，因为没有 explicit_failure_flag 或 termination_reason。
- 后续迭代可以根据训练现象加入 derived soft landing proxy、energy_penalty、time_penalty 或 gated_reward。
- 训练后重点观察：goal_near_oscillation、fast_crash_near_goal、high_reward_without_success、agent_afraid_to_move。
"""


def extract_code(md):
    m = re.search(r"```python\s*(.*?)```", md, flags=re.S)
    if m:
        return m.group(1).strip()
    if "def compute_reward" in md:
        return md.strip()
    return ""


def estimate_tokens(text):
    """Rough tokenizer-independent estimate for mixed Chinese/English prompts."""
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    ascii_chars = len(text) - non_ascii
    return int(non_ascii * 1.1 + ascii_chars / 4.0)


def write_prompt_stats(run_dir, system_prompt, user_prompt):
    total_text = system_prompt + "\n" + user_prompt
    lines = []
    lines.append("# Reward Generator Prompt Stats")
    lines.append("")
    lines.append("Token count is an estimate because the exact tokenizer depends on the LLM provider.")
    lines.append("")
    lines.append("| part | chars | lines | estimated_tokens |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| system_prompt | {len(system_prompt)} | {system_prompt.count(chr(10)) + 1} | {estimate_tokens(system_prompt)} |")
    lines.append(f"| user_prompt | {len(user_prompt)} | {user_prompt.count(chr(10)) + 1} | {estimate_tokens(user_prompt)} |")
    lines.append(f"| total | {len(total_text)} | {total_text.count(chr(10)) + 1} | {estimate_tokens(total_text)} |")
    write_text(Path(run_dir) / "prompt_records/02_reward_generator.prompt_stats.md", "\n".join(lines) + "\n")


def _return_has_component_tuple(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            value = node.value
            if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == 2:
                return True
    return False


def _has_components_dict_assignment(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"components", "reward_components", "reward_terms"}:
                    if isinstance(node.value, ast.Dict):
                        return True
    return False


def validate_code(code):
    errors = []
    warnings = []
    exact_sig = "def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):"
    if exact_sig not in code:
        errors.append("缺少准确函数签名")

    body = code.replace(exact_sig, "")
    if "original_reward" in body:
        errors.append("函数体中使用了 original_reward")

    forbidden = [
        "info.get(\"success\"", "info.get('success'", "info[\"success\"]", "info['success']",
        "info.get(\"failure\"", "info.get('failure'", "info[\"failure\"]", "info['failure']",
        "info.get(\"termination_reason\"", "info.get('termination_reason'",
        "obs[0:3]", "next_obs[0:3]",
        "import ", "class ", "try:", "except ", "eval(", "exec(", "open("
    ]
    for token in forbidden:
        if token in code:
            errors.append(f"出现禁止模式: {token}")

    if re.search(r"\bobs\s*\[\s*\d+\s*:\s*\d*", code) or re.search(r"\bnext_obs\s*\[\s*\d+\s*:\s*\d*", code):
        errors.append("出现未允许的 obs/next_obs 切片")

    try:
        tree = ast.parse(code)
    except Exception as e:
        errors.append(f"代码无法解析 AST: {e}")
        return {"valid": False, "errors": errors, "warnings": warnings}

    try:
        compile(code, "<reward_v1.py>", "exec")
    except Exception as e:
        errors.append(f"代码无法编译: {e}")

    if not _has_components_dict_assignment(tree):
        errors.append("没有发现 components/reward_components/reward_terms 字典赋值")

    if not _return_has_component_tuple(tree):
        warnings.append("建议返回 (float(total_reward), components)。当前 wrapper 兼容 float，但 tuple 返回更利于诊断。")

    if "total_reward" not in code and "reward" not in code:
        warnings.append("未发现明显的 total_reward/reward 变量名")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def run(config_path, run_name, mock=False):
    cfg = load_config(config_path)
    run_dir = Path(cfg["experiment"]["run_root"]) / run_name
    system_prompt = read_text(cfg["prompts"]["reward_generator"])
    env_md = read_text(run_dir / "environment_card.md")
    expert_md = read_text(run_dir / "expert_reward_context.md")

    user_parts = [
        "# environment_card.md",
        env_md,
        "",
        "# expert_reward_context.md",
        expert_md,
    ]
    if cfg.get("context", {}).get("include_masked_step_in_reward_generator", False):
        user_parts += ["", "# masked_step_source.py", read_text(cfg["inputs"]["masked_step_path"])]
    user_prompt = "\n\n".join(user_parts)

    write_text(run_dir / "llm_inputs/02_reward_generator.input.md", user_prompt)
    record_prompt(run_dir, "02_reward_generator", system_prompt, user_prompt)
    write_prompt_stats(run_dir, system_prompt, user_prompt)

    if mock:
        out_md = MOCK_REWARD_MD
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

    write_text(run_dir / "reward_v1.md", out_md)
    write_text(run_dir / "reward_v1.py", code)
    write_json(run_dir / "validations/reward_v1.validation.json", validation)
    record_response(run_dir, "02_reward_generator", out_md)

    if not validation["valid"]:
        print("WARNING: reward_v1 validation failed")
        for e in validation["errors"]:
            print(" -", e)
    if validation.get("warnings"):
        print("reward_v1 validation warnings")
        for w in validation["warnings"]:
            print(" -", w)

    print(run_dir / "reward_v1.py")
    print(run_dir / "reward_v1.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/env001_deepseek_rag.yaml")
    ap.add_argument("--run-name", default="mock_run")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    run(args.config, args.run_name, args.mock)
