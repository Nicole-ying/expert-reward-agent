"""Single-agent reflection: replaces run_04 (analysis) + run_05 (revision).

The agent reads feedback, memory, environment-specific task/profile context,
and optional tool results. It produces revised reward code.
"""

import argparse
import json
import re
from pathlib import Path

from .common import load_config, read_text, write_text, write_json, record_prompt, record_response
from .reflection_tools import (
    get_reward_transformation,
    get_skeleton_detail,
    search_reward_design_knowledge,
)
from .run_03_direct_reward_generator import extract_code, validate_code
from llm_clients.deepseek_client import DeepSeekClient


def _environment_summary(environment_card_md):
    """Keep task and interface facts needed to interpret reward code.

    Sections 9-12 (expert task profile, reward roles, signal mapping, failure modes)
    are intentionally excluded from reflection. They are designed for the initial
    reward generator, not for iterative diagnosis. The reflection agent needs raw
    environment facts (1-7) and training evidence, not pre-digested design advice.
    """
    if not environment_card_md:
        return ""
    wanted = {1, 3, 4, 5, 7}
    sections = re.split(r"(?=^## \d+\.)", environment_card_md, flags=re.MULTILINE)
    selected = []
    for section in sections:
        match = re.match(r"^## (\d+)\.", section)
        if match and int(match.group(1)) in wanted:
            selected.append(section.strip())
    return "\n\n".join(selected)


def _compact_route_context(cfg, environment_card_md):
    """Legacy hook kept for backward compatibility.

    Route-to-skeleton recommendations are intentionally no longer inserted into
    reflection prompts. The environment card's reward-role decomposition is the
    source of truth for task-specific expert guidance.
    """
    return ""


def build_user_prompt(feedback_md, memory_md, previous_code, best_code, environment_card_md="", cfg=None):
    """Assemble the reflection agent's user prompt — focused, no generic templates."""
    parts = []

    prev_score = "?"
    m = re.search(r"score=([-\d.]+)", feedback_md)
    if m:
        prev_score = m.group(1)

    target_score = float((cfg or {}).get("iteration", {}).get("target_score", 0.0))
    current_score = float(prev_score) if prev_score != "?" else None
    if target_score > 0 and current_score is not None:
        gap = target_score - current_score
        ratio = current_score / target_score
        parts.append(
            "# Search objective\n"
            f"- target_score: {target_score:.6f}\n"
            f"- current_score: {current_score:.6f}\n"
            f"- gap_to_target: {gap:.6f}\n"
            f"- target_achievement_ratio: {ratio:.3%}\n"
            "Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula."
        )

    parts.append(f"# 上一轮奖励函数代码（该轮得分: {prev_score}）\n```python\n{previous_code.strip()}\n```")

    if False:  # 历史最佳代码已移除，用历史记忆表格替代
        pass

    parts.append(f"# 训练反馈（上一轮代码的训练结果）\n{feedback_md}")

    environment_summary = _environment_summary(environment_card_md)
    if environment_summary:
        parts.append(
            "# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）\n"
            f"{environment_summary}"
        )

    route_context = _compact_route_context(cfg or {}, environment_card_md)
    if route_context:
        parts.append(f"# Compact expert route context\n{route_context}")

    if memory_md:
        parts.append(f"# 历史记忆\n{memory_md}")

    return "\n\n".join(parts)


def _score_only_feedback(feedback_md):
    outcome = re.search(
        r"## Final-policy outcome\s*(.*?)(?=\n## |\Z)",
        feedback_md,
        flags=re.DOTALL,
    )
    distribution = re.search(
        r"## Evaluation distribution\s*(.*?)(?=\n## |\Z)",
        feedback_md,
        flags=re.DOTALL,
    )
    blocks = ["# Score-Only Feedback Ablation"]
    if outcome:
        blocks.extend(["## Final-policy outcome", outcome.group(1).strip()])
    if distribution:
        blocks.extend(["## Evaluation distribution", distribution.group(1).strip()])
    return "\n\n".join(blocks)


def _score_only_memory(memory_md):
    rows = []
    for line in memory_md.splitlines():
        if not line.startswith("|") or line.startswith("|---") or "| iter |" in line:
            continue
        columns = [item.strip() for item in line.strip().strip("|").split("|")]
        if len(columns) >= 6:
            rows.append(columns[:6])
    if not rows:
        return ""
    lines = [
        "# Score-Only Reward Memory",
        "",
        "| iter | skeleton | score | best | delta | len |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _tool_definitions():
    return [
        {
            "type": "function",
            "function": {
                "name": "search_reward_design_knowledge",
                "description": "搜索奖励设计技法库。输入症状描述（自然语言），返回匹配的技法和修复方案。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "症状的自然语言描述，如 'penalty dominating progress signal' 或 'landing bonus never triggers'",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_skeleton_detail",
                "description": "获取某个骨架的数学形态、设计原理、常见陷阱和推荐配合。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skeleton_name": {
                            "type": "string",
                            "description": "骨架名称，如 'progress_delta_reward', 'potential_based_shaping', 'bounded_proximity_reward'",
                        }
                    },
                    "required": ["skeleton_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_reward_transformation",
                "description": "Retrieve general reward-structure transformations from diagnosis evidence, including rationale, risks, and next-round verification targets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The diagnosed problem dimension or desired transformation, such as persistent proxy farming, sparse credit, product collapse, or global constraint interference.",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
    ]


def _run_tool_call(tool_call):
    args = json.loads(tool_call.function.arguments)
    if tool_call.function.name == "search_reward_design_knowledge":
        result = search_reward_design_knowledge(args.get("query", ""))
    elif tool_call.function.name == "get_skeleton_detail":
        result = get_skeleton_detail(args.get("skeleton_name", ""))
    elif tool_call.function.name == "get_reward_transformation":
        result = get_reward_transformation(args.get("query", ""))
    else:
        result = f"Unknown tool: {tool_call.function.name}"
    return args, result


def run_reflection_agent(
    config_path,
    previous_reward_path,
    best_reward_path,
    train_run_dir,
    memory_path,
    out_run_name,
    reward_version,
    environment_card_path=None,
    mock=False,
    validation_retry=None,
    duplicate_retry=None,
):
    cfg = load_config(config_path)
    ablation_cfg = cfg.get("ablation", {})
    run_dir = Path(cfg["experiment"]["run_root"]) / out_run_name
    for sub in ["llm_inputs", "prompt_records", "response_records", "validations"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    reflection_mode = ablation_cfg.get("reflection_mode", "structured")
    prompt_path = (
        "prompts/reflection_agent_unconstrained_prompt.md"
        if reflection_mode == "unconstrained"
        else "prompts/reflection_agent_prompt.md"
    )
    system_prompt = read_text(prompt_path)
    previous_code = read_text(previous_reward_path)
    feedback_md = read_text(str(Path(train_run_dir) / "training_feedback.md"))
    if ablation_cfg.get("feedback_mode") == "score_only":
        feedback_md = _score_only_feedback(feedback_md)

    environment_card_md = ""
    if environment_card_path and Path(environment_card_path).exists():
        environment_card_md = read_text(environment_card_path)

    memory_md = ""
    if not ablation_cfg.get("disable_memory", False) and Path(memory_path).exists():
        memory_md = read_text(memory_path)
        if ablation_cfg.get("feedback_mode") == "score_only":
            memory_md = _score_only_memory(memory_md)

    best_code = ""
    if best_reward_path and Path(best_reward_path).exists():
        best_code = read_text(best_reward_path)

    if duplicate_retry:
        duplicate_draft_path = run_dir / f"reward_{reward_version}.py"
        duplicate_draft = read_text(duplicate_draft_path) if duplicate_draft_path.exists() else ""
        user_prompt = (
            "# Duplicate reward retry\n"
            f"{duplicate_retry}\n"
            "The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. "
            "Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. "
            "Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. "
            "Return a complete reward function whose executable code is materially different from every historical reward. "
            "Do not merely rename variables or comments.\n\n"
            f"# Rejected duplicate draft\n```python\n{duplicate_draft}\n```\n\n"
        ) + build_user_prompt(feedback_md, memory_md, previous_code, best_code, environment_card_md, cfg)
    elif validation_retry:
        failed_draft_path = run_dir / f"reward_{reward_version}.md"
        failed_draft = read_text(failed_draft_path) if failed_draft_path.exists() else ""
        user_prompt = (
            f"# ⚠️ 上一版代码验证失败\n"
            f"错误信息：{validation_retry}\n"
            "这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。"
            "直接输出修复后的完整 Python 代码。\n\n"
            f"# 被截断或无效的上一版草稿\n{failed_draft}\n\n"
        ) + build_user_prompt(feedback_md, "", previous_code, best_code, environment_card_md, cfg)
    else:
        user_prompt = build_user_prompt(feedback_md, memory_md, previous_code, best_code, environment_card_md, cfg)

    write_text(run_dir / f"llm_inputs/reward_{reward_version}_reflection_agent.input.md", user_prompt)
    record_prompt(run_dir, "agent_reflection", system_prompt, user_prompt)

    tool_trace = []
    trace_path = run_dir / f"response_records/reward_{reward_version}_tool_trace.json"
    previous_invocations = []
    if trace_path.exists():
        previous_trace = json.loads(read_text(trace_path))
        previous_invocations = previous_trace.get("invocations", [])
        if not previous_invocations and previous_trace.get("calls") is not None:
            previous_invocations = [{
                "validation_retry": None,
                "status": "legacy_completed",
                "calls": previous_trace.get("calls", []),
            }]

    def save_tool_trace(status):
        invocations = previous_invocations + [{
            "validation_retry": bool(validation_retry),
            "duplicate_retry": bool(duplicate_retry),
            "status": status,
            "calls": tool_trace,
        }]
        write_json(trace_path, {
            "call_count": sum(len(item.get("calls", [])) for item in invocations),
            "invocation_count": len(invocations),
            "invocations": invocations,
            "calls": [call for item in invocations for call in item.get("calls", [])],
        })

    if mock:
        from .run_05_reward_revision import MOCK_REVISION_MD
        out_md = MOCK_REVISION_MD
    else:
        llm_cfg = cfg["llm"]
        client = DeepSeekClient(api_key_env=llm_cfg["api_key_env"], base_url=llm_cfg["base_url"])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        max_tool_rounds = 1
        tools = _tool_definitions()
        for tool_round in range(max_tool_rounds):
            request = dict(
                model=llm_cfg["model_reward"],
                messages=messages,
                temperature=llm_cfg["temperature_reward_generator"],
                max_tokens=llm_cfg["max_tokens_reward"],
            )
            if not validation_retry and not ablation_cfg.get("disable_expert_rag", False):
                request.update(tools=tools, tool_choice="auto")
            try:
                resp = client.completion(**request)
            except Exception:
                save_tool_trace("llm_error")
                raise
            msg = resp.choices[0].message

            if msg.tool_calls:
                messages.append({"role": "assistant", "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]})
                for tc in msg.tool_calls:
                    args, result = _run_tool_call(tc)
                    tool_trace.append({
                        "round": tool_round + 1,
                        "name": tc.function.name,
                        "arguments": args,
                        "result": result,
                    })
                    save_tool_trace("tools_returned")
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            else:
                out_md = msg.content or ""
                break
        else:
            try:
                resp = client.completion(
                    model=llm_cfg["model_reward"],
                    messages=messages,
                    temperature=llm_cfg["temperature_reward_generator"],
                    max_tokens=llm_cfg["max_tokens_reward"],
                )
            except Exception:
                save_tool_trace("llm_error_after_tools")
                raise
            out_md = resp.choices[0].message.content or ""

    record_response(run_dir, "agent_reflection", out_md)
    save_tool_trace("completed")

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
    ap.add_argument("--environment-card", default=None)
    ap.add_argument("--best-reward", default=None)
    ap.add_argument("--train-run-dir", required=True)
    ap.add_argument("--memory", default="runs/env_001/memory/reward_memory.md")
    ap.add_argument("--out-run-name", required=True)
    ap.add_argument("--reward-version", default="v2")
    ap.add_argument("--validation-retry", default=None)
    ap.add_argument("--duplicate-retry", default=None)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    run_reflection_agent(
        config_path=args.config,
        previous_reward_path=args.previous_reward,
        environment_card_path=args.environment_card,
        best_reward_path=args.best_reward,
        train_run_dir=args.train_run_dir,
        memory_path=args.memory,
        out_run_name=args.out_run_name,
        reward_version=args.reward_version,
        mock=args.mock,
        validation_retry=args.validation_retry,
        duplicate_retry=args.duplicate_retry,
    )


if __name__ == "__main__":
    main()
