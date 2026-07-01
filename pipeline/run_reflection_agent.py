"""Single-agent reflection: replaces run_04 (analysis) + run_05 (revision).

The agent reads feedback, memory, expert context, and uses tools to
retrieve design techniques and skeleton details. It produces revised code.
"""

import argparse
import json
import re
from pathlib import Path

from .common import load_config, read_text, write_text, write_json, record_prompt, record_response
from .reflection_tools import search_reward_design_knowledge, get_skeleton_detail
from .run_03_direct_reward_generator import extract_code, validate_code
from llm_clients.deepseek_client import DeepSeekClient


def build_user_prompt(feedback_md, memory_md, previous_code, best_code, env_contract, expert_context, env_card):
    """Assemble the reflection agent's user prompt."""
    parts = []

    if env_card:
        # Trim env_card to key sections (1-5)
        env_trimmed = env_card.split("## 6.")[0] if "## 6." in env_card else env_card
        parts.append(f"# 环境信息\n{env_trimmed}")

    parts.append(f"# 环境契约\n{env_contract}")

    if expert_context:
        # Strip v1 generation instructions
        ctx = re.split(r"\n## \d+\. reward_v1 生成要求", expert_context)[0]
        parts.append(f"# 专家知识骨架\n{ctx}")

    parts.append(f"# 当前奖励函数代码\n```python\n{previous_code.strip()}\n```")

    if best_code and best_code != previous_code:
        parts.append(f"# 历史最佳奖励函数代码\n```python\n{best_code.strip()}\n```")

    parts.append(f"# 训练反馈\n{feedback_md}")

    if memory_md:
        parts.append(f"# 历史记忆\n{memory_md}")

    return "\n\n".join(parts)


def run_reflection_agent(
    config_path,
    previous_reward_path,
    best_reward_path,
    train_run_dir,
    memory_path,
    out_run_name,
    reward_version,
    mock=False,
):
    cfg = load_config(config_path)
    run_dir = Path(cfg["experiment"]["run_root"]) / out_run_name
    for sub in ["llm_inputs", "prompt_records", "response_records", "validations"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    system_prompt = read_text("prompts/reflection_agent_prompt.md")
    previous_code = read_text(previous_reward_path)
    feedback_md = read_text(str(Path(train_run_dir) / "training_feedback.md"))
    memory_md = ""
    if Path(memory_path).exists():
        memory_md = read_text(memory_path)

    # Best code
    best_code = ""
    if best_reward_path and Path(best_reward_path).exists():
        best_code = read_text(best_reward_path)

    # Expert context
    seed_dir = Path(previous_reward_path).parent.parent
    expert_context = ""
    for d in sorted(Path(seed_dir).glob("iter_*/generation/expert_reward_context.md"), reverse=True):
        expert_context = read_text(str(d))
        break

    # Environment card
    env_card = ""
    for d in sorted(Path(seed_dir).glob("iter_*/generation/environment_card.md"), reverse=True):
        env_card = read_text(str(d))
        break

    env_contract = (
        "- function_signature: def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):\n"
        "- allowed: obs[0..7], next_obs[0..7], action (0=noop,1=left,2=main,3=right), info (no reliable fields)\n"
        "- forbidden: original_reward, official_reward, terminal_success_reward, terminal_failure_penalty\n"
    )

    user_prompt = build_user_prompt(
        feedback_md, memory_md, previous_code, best_code, env_contract, expert_context, env_card
    )

    write_text(run_dir / f"llm_inputs/reward_{reward_version}_reflection_agent.input.md", user_prompt)
    record_prompt(run_dir, "agent_reflection", system_prompt, user_prompt)

    if mock:
        from .run_05_reward_revision import MOCK_REVISION_MD
        out_md = MOCK_REVISION_MD
    else:
        llm_cfg = cfg["llm"]
        client = DeepSeekClient(api_key_env=llm_cfg["api_key_env"], base_url=llm_cfg["base_url"])

        # Tool definitions (OpenAI-compatible format)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_reward_design_knowledge",
                    "description": "搜索奖励设计技法库。输入症状描述（自然语言），返回匹配的技法和修复方案。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "症状的自然语言描述，如 'penalty dominating progress signal' 或 'landing bonus never triggers'"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_skeleton_detail",
                    "description": "获取某个骨架的数学形态、设计原理、常见陷阱和推荐配合。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skeleton_name": {"type": "string", "description": "骨架名称，如 'progress_delta_reward', 'potential_based_shaping', 'bounded_proximity_reward'"}
                        },
                        "required": ["skeleton_name"]
                    }
                }
            }
        ]

        # Agent loop
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        max_tool_rounds = 3
        for _ in range(max_tool_rounds):
            resp = client.client.chat.completions.create(
                model=llm_cfg["model_reward"],
                messages=messages,
                temperature=llm_cfg["temperature_reward_generator"],
                max_tokens=llm_cfg["max_tokens_reward"],
                tools=tools,
                tool_choice="auto",
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                messages.append({"role": "assistant", "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]})
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    if tc.function.name == "search_reward_design_knowledge":
                        result = search_reward_design_knowledge(args.get("query", ""))
                    elif tc.function.name == "get_skeleton_detail":
                        result = get_skeleton_detail(args.get("skeleton_name", ""))
                    else:
                        result = f"Unknown tool: {tc.function.name}"
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            else:
                out_md = msg.content or ""
                break
        else:
            # Max rounds reached, take last response
            resp = client.client.chat.completions.create(
                model=llm_cfg["model_reward"],
                messages=messages,
                temperature=llm_cfg["temperature_reward_generator"],
                max_tokens=llm_cfg["max_tokens_reward"],
            )
            out_md = resp.choices[0].message.content or ""

    record_response(run_dir, "agent_reflection", out_md)

    code = extract_code(out_md)
    validation = validate_code(code)

    write_text(run_dir / f"reward_{reward_version}.md", out_md)
    write_text(run_dir / f"reward_{reward_version}.py", code)
    write_json(run_dir / f"validations/reward_{reward_version}.validation.json", validation)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/env001_deepseek_rag.yaml")
    ap.add_argument("--previous-reward", required=True)
    ap.add_argument("--best-reward", default=None)
    ap.add_argument("--train-run-dir", required=True)
    ap.add_argument("--memory", default="runs/env_001/memory/reward_memory.md")
    ap.add_argument("--out-run-name", required=True)
    ap.add_argument("--reward-version", default="v2")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    run_reflection_agent(
        config_path=args.config,
        previous_reward_path=args.previous_reward,
        best_reward_path=args.best_reward,
        train_run_dir=args.train_run_dir,
        memory_path=args.memory,
        out_run_name=args.out_run_name,
        reward_version=args.reward_version,
        mock=args.mock,
    )


if __name__ == "__main__":
    main()
