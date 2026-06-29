import json
import re
from pathlib import Path


def load_jsonl(path):
    items = []
    p = Path(path)
    if not p.exists():
        return items
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def extract_selected_route_id(environment_card_md):
    m = re.search(r"selected_route_id\s*:\s*([a-zA-Z0-9_]+)", environment_card_md)
    if m:
        return m.group(1).strip()
    # fallback keyword
    if "导航" in environment_card_md or "到达目标" in environment_card_md or "target" in environment_card_md.lower():
        return "navigation_goal_reaching"
    return "navigation_goal_reaching"


def build_expert_reward_context(environment_card_md, chunks_path, max_chars=6500):
    route_id = extract_selected_route_id(environment_card_md)
    chunks = load_jsonl(chunks_path)

    route_chunks = [c for c in chunks if c.get("knowledge_type") == "task_route" and c.get("route_id") == route_id]
    route_recommended = []
    if route_chunks:
        route_recommended = route_chunks[0].get("recommended_skeletons", [])

    # Always include route recommended + likely interface-relevant stability/action terms, but compress.
    desired = set(route_recommended) | {"stability_penalty", "energy_penalty", "progress_delta_reward", "distance_reward"}
    skeleton_chunks = [c for c in chunks if c.get("knowledge_type") == "reward_skeleton" and c.get("skeleton_id") in desired]

    lines = []
    lines.append("# 专家奖励知识上下文（RAG 压缩版）")
    lines.append("")
    lines.append("## 1. 检索策略")
    lines.append(f"- selected_route_id: {route_id}")
    lines.append("- 生成阶段只使用 02 奖励骨架库 与 03 任务类型路由知识。")
    lines.append("- 这里是给 Reward Generator 直接阅读的压缩专家上下文，不是完整知识库。")
    lines.append("")

    lines.append("## 2. 当前任务类型专家知识")
    for c in route_chunks[:1]:
        lines.append(f"### {c.get('heading','selected route')}")
        txt = c.get("raw_text","").strip()
        lines.append(txt[:1800])
        lines.append("")
    if route_recommended:
        lines.append("推荐骨架：")
        for sid in route_recommended:
            lines.append(f"- {sid}")
    lines.append("")

    lines.append("## 3. 相关奖励骨架知识")
    for c in skeleton_chunks:
        sid = c.get("skeleton_id")
        txt = c.get("raw_text","").strip()
        lines.append(f"### {sid}")
        lines.append(txt[:900])
        lines.append("")

    lines.append("## 4. 本阶段生成要求")
    lines.append("- 直接生成 reward_v1.py。")
    lines.append("- 从简单到复杂，但不要把 minimal-first 理解成只能用一个组件。")
    lines.append("- 如果 success/failure 显式信号不存在，不要用 terminal_success_reward / terminal_failure_penalty 作为 v1 核心项。")
    lines.append("- 优先使用明确可由 obs/next_obs/action 得到的信号。")
    lines.append("- 对目标到达且存在速度/姿态风险的任务，建议考虑：主学习信号 + 轻量稳定约束。")
    lines.append("- 后续迭代再考虑 terminal、failure、energy、time、gated 等项。")

    md = "\n".join(lines)
    if len(md) > max_chars:
        md = md[:max_chars] + "\n\n<!-- truncated to max_expert_context_chars -->\n"
    return route_id, md
