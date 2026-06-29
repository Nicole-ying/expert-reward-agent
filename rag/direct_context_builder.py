import re


def extract_selected_route_id(environment_card_md):
    m = re.search(r"selected_route_id\s*:\s*([a-zA-Z0-9_]+)", environment_card_md)
    if m:
        return m.group(1).strip()
    return "navigation_goal_reaching"


SKELETON_SUMMARY = {
    "progress_delta_reward": {
        "role": "主学习引导",
        "math": "d(obs, goal) - d(next_obs, goal)",
        "need": "obs[0], obs[1], next_obs[0], next_obs[1]",
        "use": "推荐作为 v1 主信号：奖励每一步更接近目标。",
        "risk": "目标附近震荡。",
        "later": "可 clip；后续配合成功、时间或稳定信号。",
    },
    "distance_reward": {
        "role": "密集过程引导",
        "math": "-d(obs, goal)",
        "need": "obs[0], obs[1]",
        "use": "可用，但单独使用可能只鼓励靠近，不鼓励稳定完成。",
        "risk": "接近目标但不完成；不关心速度和姿态。",
        "later": "训练后检查 high_reward_without_success。",
    },
    "stability_penalty": {
        "role": "轻量稳定约束",
        "math": "-lambda_v*|velocity| - lambda_a*|angle|",
        "need": "next_obs[2], next_obs[3], next_obs[4], 可选 next_obs[5]",
        "use": "如果任务要求稳定接近/着陆，v1 可以小权重加入。",
        "risk": "过强会保守或不敢动。",
        "later": "若高速撞击或姿态失稳，增大权重。",
    },
    "terminal_success_reward": {
        "role": "任务目标奖励",
        "math": "R_success * I[success]",
        "need": "显式 success flag",
        "use": "若 explicit_success_flag_available=false，不作为 v1 核心项。",
        "risk": "会诱导 LLM 发明 info['success']。",
        "later": "当 wrapper 明确暴露 success 后再加。",
    },
    "terminal_failure_penalty": {
        "role": "失败惩罚",
        "math": "-R_failure * I[failure]",
        "need": "显式 failure flag 或 termination_reason",
        "use": "若 explicit_failure_flag_available=false，不作为 v1 核心项。",
        "risk": "误判终止原因。",
        "later": "当能区分失败终止后再加。",
    },
    "time_penalty": {
        "role": "效率约束",
        "math": "-lambda_time",
        "need": "每步调用",
        "use": "通常后续加入，不建议 v1 太早加入。",
        "risk": "可能导致冒险或快速失败。",
        "later": "若能接近但拖太久，再小权重加入。",
    },
    "energy_penalty": {
        "role": "动作/能耗约束",
        "math": "-lambda_action * engine_use(action)",
        "need": "action",
        "use": "通常后续加入，v1 太早加入可能不敢动。",
        "risk": "agent_afraid_to_move。",
        "later": "能接近并稳定后再优化燃料。",
    },
    "gated_reward": {
        "role": "安全门控",
        "math": "if unsafe then penalty else task_reward",
        "need": "明确 unsafe 条件",
        "use": "v1 不建议使用复杂门控。",
        "risk": "门控过严导致学不到。",
        "later": "若安全被进度奖励抵消，再加入。",
    },
    "potential_based_shaping": {
        "role": "势能塑形",
        "math": "gamma*Phi(next_obs)-Phi(obs)",
        "need": "可定义 Phi",
        "use": "不作为 v1 首选；比 progress_delta 更抽象。",
        "risk": "Phi 错误会误导学习。",
        "later": "如果需要更标准的 shaping，再替换或补充。",
    },
}


def build_expert_reward_context(environment_card_md, chunks_path=None, max_chars=6500):
    route_id = extract_selected_route_id(environment_card_md)
    text = environment_card_md.lower()
    related = ["progress_delta_reward", "distance_reward"]
    if any(k in text for k in ["速度", "姿态", "稳定", "velocity", "angle", "settle", "landing", "着陆"]):
        related.append("stability_penalty")
    related += [
        "terminal_success_reward",
        "terminal_failure_penalty",
        "time_penalty",
        "energy_penalty",
        "gated_reward",
        "potential_based_shaping",
    ]

    lines = []
    lines.append("# 专家奖励知识上下文（RAG 压缩版）")
    lines.append("")
    lines.append("这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。")
    lines.append("")
    lines.append("## 1. 当前任务类型")
    lines.append(f"- selected_route_id: {route_id}")
    if route_id == "navigation_goal_reaching":
        lines.append("- 核心规则: 目标到达任务需要密集过程引导；如果 success flag 不可用，不要依赖终点成功奖励。")
        lines.append("- 优先检查失败模式: goal_near_oscillation, high_reward_without_success, fast_crash_near_goal")
    lines.append("")
    lines.append("## 2. 相关奖励骨架摘要")
    for sid in related:
        d = SKELETON_SUMMARY[sid]
        lines.append(f"### {sid}")
        lines.append(f"- 角色: {d['role']}")
        lines.append(f"- 数学形态: {d['math']}")
        lines.append(f"- 需要信号: {d['need']}")
        lines.append(f"- 本轮建议: {d['use']}")
        lines.append(f"- 风险: {d['risk']}")
        lines.append(f"- 后续迭代: {d['later']}")
        lines.append("")
    lines.append("## 3. reward_v1 生成要求")
    lines.append("- 直接生成 reward_v1.py，不再生成 reward_design_plan.json。")
    lines.append("- 推荐结构: 主学习信号 + 0~1 个轻量约束项。")
    lines.append("- 如果 success/failure 显式信号不存在，不要使用 terminal_success_reward / terminal_failure_penalty 作为 v1 核心项。")
    lines.append("- 如果速度/姿态信号明确可用，且任务需要稳定接近或着陆，可以加入轻量 stability_penalty。")
    lines.append("- energy_penalty、time_penalty、gated_reward 默认后续迭代再加入。")
    lines.append("- 每个 reward term 必须写入 info['reward_terms']，便于训练后诊断。")
    md = "\n".join(lines)
    if len(md) > max_chars:
        md = md[:max_chars] + "\n\n<!-- truncated to max_expert_context_chars -->\n"
    return route_id, md
