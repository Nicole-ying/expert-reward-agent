import re


def extract_selected_route_id(environment_card_md):
    m = re.search(r"selected_route_id\s*:\s*([a-zA-Z0-9_]+)", environment_card_md)
    if m:
        return m.group(1).strip()
    return "navigation_goal_reaching"


SKELETON_SUMMARY = {
    "progress_delta_reward": {
        "role": "密集学习引导",
        "math": "d(obs, goal) - d(next_obs, goal)",
        "need": "obs[0], obs[1], next_obs[0], next_obs[1]",
        "use": "奖励每一步更接近目标。适合目标位置已知的导航/到达任务。",
        "risk": "目标附近震荡。",
        "later": "可 clip；后续配合成功、时间或稳定信号。",
    },
    "distance_reward": {
        "role": "密集过程引导",
        "math": "-d(obs, goal)",
        "need": "obs[0], obs[1]",
        "use": "连续负距离信号，引导 agent 靠近目标。与 progress_delta_reward 同时大权重使用会产生重复信号。",
        "risk": "接近目标但不完成；不关心速度和姿态。",
        "later": "训练后检查 high_reward_without_success。",
    },
    "stability_penalty": {
        "role": "轻量稳定约束",
        "math": "-lambda_v*|velocity| - lambda_a*|angle| - lambda_w*|angular_velocity|",
        "need": "next_obs[2], next_obs[3], next_obs[4], next_obs[5]",
        "use": "抑制高速、大角度或高角速度。适合需要稳定运动或姿态控制的任务。",
        "risk": "过强会保守或不敢动。",
        "later": "若高速失稳，增大权重。",
    },
    "soft_landing_proxy": {
        "role": "任务完成近似信号",
        "math": "small_bonus if near_target and low_speed and stable_angle and both_contact else 0",
        "need": "position, velocity, angle, contact flags",
        "use": "多条件组合的完成近似。不能直接把 contact 当 success。",
        "risk": "如果条件太宽，会变成 contact reward hacking。",
        "later": "如果 high_reward_without_success，收紧条件或移除。",
    },
    "terminal_success_reward": {
        "role": "任务目标奖励",
        "math": "R_success * I[success]",
        "need": "显式 success flag",
        "use": "当环境提供显式 success flag 时可用。若 explicit_success_flag_available=false，不可使用。",
        "risk": "会诱导 LLM 发明 info['success']。",
        "later": "当 wrapper 明确暴露 success 后再加。",
    },
    "terminal_failure_penalty": {
        "role": "失败惩罚",
        "math": "-R_failure * I[failure]",
        "need": "显式 failure flag 或 termination_reason",
        "use": "当环境提供显式 failure flag 时可用。若 explicit_failure_flag_available=false，不可使用。",
        "risk": "误判终止原因。",
        "later": "当能区分失败终止后再加。",
    },
    "time_penalty": {
        "role": "效率约束",
        "math": "-lambda_time",
        "need": "每步调用",
        "use": "惩罚每步耗时。先完成任务再加入，不建议 v1 使用。",
        "risk": "可能导致冒险或快速失败。",
        "later": "若能接近但拖太久，再小权重加入。",
    },
    "energy_penalty": {
        "role": "动作/能耗约束",
        "math": "-lambda_action * engine_use(action)",
        "need": "action",
        "use": "惩罚动作幅度/能耗。先完成任务再加入，v1 太早加入可能不敢动。",
        "risk": "agent_afraid_to_move。",
        "later": "能完成任务并稳定后再优化能耗。",
    },
    "gated_reward": {
        "role": "安全门控",
        "math": "if unsafe then penalty else task_reward",
        "need": "明确 unsafe 条件",
        "use": "按条件切换奖励模式。v1 不建议使用复杂门控。",
        "risk": "门控过严导致学不到。",
        "later": "若安全被进度奖励抵消，再加入。",
    },
    "potential_based_shaping": {
        "role": "势能塑形",
        "math": "gamma*Phi(next_obs)-Phi(obs)",
        "need": "可定义 Phi",
        "use": "基于势能差分的塑形信号。比 progress_delta 更抽象，当任务有明确的势能定义时可使用。",
        "risk": "Phi 错误会误导学习。",
        "later": "如果需要更标准的 shaping，再替换或补充。",
    },
    "forward_progress_reward": {
        "role": "前进方向引导",
        "math": "lambda_fwd * forward_velocity",
        "need": "forward velocity component",
        "use": "奖励沿前进方向的速度。适合 locomotion 类任务。",
        "risk": "快速前进但容易摔倒。",
        "later": "若 fast_then_fail，配合稳定性约束。",
    },
    "alive_bonus": {
        "role": "存活激励",
        "math": "lambda_alive * I[not_done]",
        "need": "done flag",
        "use": "每步给予小额存活奖励，鼓励 agent 不提前终止。适合 locomotion/balance 类任务。",
        "risk": "hover_or_stand_still（原地不动来获取存活奖励）。",
        "later": "若 agent 不动，减小权重或配合前向奖励。",
    },
    "action_smoothness_penalty": {
        "role": "动作平滑约束",
        "math": "-lambda_smooth * |action - previous_action|",
        "need": "previous action or action history",
        "use": "惩罚动作的剧烈变化。适合连续控制任务。",
        "risk": "离散动作空间不可用（无历史信息）。",
        "later": "若动作抖动，增大权重。",
    },
    "event_reward": {
        "role": "事件目标奖励",
        "math": "R_event * I[event]",
        "need": "event flag",
        "use": "对特定事件给予奖励。适合 resource_gathering 等离散目标任务。",
        "risk": "event_reward_farming（反复触发事件）。",
        "later": "若事件被 exploit，加 cooldown 或递减。",
    },
}


# Route-based skeleton groups: each route maps to the skeletons most relevant to it.
# These are derived from route_catalog_03.json; the full catalog is the authoritative source.
# This mapping is used as a fallback when the catalog cannot be loaded at context-build time.
_ROUTE_SKELETON_MAP = {
    "navigation_goal_reaching": [
        "progress_delta_reward", "distance_reward", "potential_based_shaping",
        "stability_penalty", "soft_landing_proxy",
        "terminal_success_reward", "terminal_failure_penalty",
        "time_penalty", "energy_penalty", "gated_reward",
    ],
    "locomotion_continuous_control": [
        "forward_progress_reward", "alive_bonus", "stability_penalty",
        "energy_penalty", "action_smoothness_penalty",
        "terminal_failure_penalty", "time_penalty",
        "gated_reward", "potential_based_shaping",
    ],
    "manipulation_precision": [
        "distance_reward", "progress_delta_reward", "potential_based_shaping",
        "stability_penalty", "event_reward",
        "terminal_success_reward", "terminal_failure_penalty",
        "time_penalty", "energy_penalty", "gated_reward",
    ],
    "balance_stabilization": [
        "stability_penalty", "alive_bonus", "distance_reward",
        "potential_based_shaping", "action_smoothness_penalty",
        "terminal_failure_penalty", "time_penalty",
        "gated_reward", "energy_penalty",
    ],
    "racing_time_trial": [
        "forward_progress_reward", "time_penalty", "distance_reward",
        "stability_penalty", "energy_penalty",
        "terminal_success_reward", "terminal_failure_penalty",
        "action_smoothness_penalty", "gated_reward",
    ],
    "resource_gathering": [
        "event_reward", "distance_reward", "time_penalty",
        "energy_penalty", "alive_bonus",
        "terminal_success_reward", "terminal_failure_penalty",
        "gated_reward", "potential_based_shaping",
    ],
    "evasion_avoidance": [
        "distance_reward", "progress_delta_reward", "stability_penalty",
        "alive_bonus", "time_penalty",
        "terminal_success_reward", "terminal_failure_penalty",
        "energy_penalty", "gated_reward", "potential_based_shaping",
    ],
}

# Skeletons that are supplementary across all routes (not primary candidates)
_SUPPLEMENTARY_SKELETONS = [
    "intrinsic_exploration_reward",
    "dynamic_curriculum_reward",
    "learned_preference_reward",
    "weighted_sum_reward",
]

# Task-agnostic route summary lines — one generic line + route-specific observation hints.
_ROUTE_SUMMARIES = {
    "navigation_goal_reaching": (
        "navigation_goal_reaching：任务目标是接近/到达指定位置。"
        "重点观察 goal_near_oscillation / high_reward_without_success / fast_crash_near_goal。"
    ),
    "locomotion_continuous_control": (
        "locomotion_continuous_control：任务目标是稳定前进通过地形。"
        "重点观察 fast_then_fail / hover_or_stand_still / over_conservative_policy。"
    ),
    "balance_stabilization": (
        "balance_stabilization：任务目标是维持平衡/姿态。"
        "重点观察 over_conservative_policy / agent_afraid_to_move / exploration_without_completion。"
    ),
    "racing_time_trial": (
        "racing_time_trial：任务目标是尽快到达终点。"
        "重点观察 reckless_behavior / fast_then_fail / weight_domination。"
    ),
    "resource_gathering": (
        "resource_gathering：任务目标是收集分散资源。"
        "重点观察 event_reward_farming / goal_near_oscillation / exploration_without_completion。"
    ),
    "evasion_avoidance": (
        "evasion_avoidance：任务目标是避开障碍/威胁到达安全区域。"
        "重点观察 over_conservative_policy / fast_crash_near_goal / reckless_behavior。"
    ),
    "manipulation_precision": (
        "manipulation_precision：任务目标是精确操控物体到目标位姿。"
        "重点观察 goal_near_oscillation / weight_domination / sparse_reward_no_learning。"
    ),
}


def build_expert_reward_context(environment_card_md, chunks_path=None, max_chars=6500):
    route_id = extract_selected_route_id(environment_card_md)

    # Select skeletons based on route_id (task type), not hardcoded per environment.
    related = list(_ROUTE_SKELETON_MAP.get(route_id, _ROUTE_SKELETON_MAP["navigation_goal_reaching"]))
    # Append supplementary skeletons for all routes
    for sk in _SUPPLEMENTARY_SKELETONS:
        if sk not in related:
            related.append(sk)

    # Also check environment card for additional contextual keywords to include
    # skeletons that might otherwise be missed by the route map alone.
    text = environment_card_md.lower()
    if "soft_landing_proxy" not in related and any(k in text for k in ["contact", "leg", "foot", "touch"]):
        related.append("soft_landing_proxy")
    if "stability_penalty" not in related and any(k in text for k in ["angle", "velocity", "姿态", "稳定"]):
        related.append("stability_penalty")
    if "event_reward" not in related and any(k in text for k in ["event", "事件", "trigger", "collect", "收集"]):
        related.append("event_reward")

    lines = []
    lines.append("# 专家奖励知识上下文（RAG 压缩版）")
    lines.append("")
    lines.append("这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。")
    lines.append("以下骨架由任务路由检索生成，不预设特定组合。具体选择由环境接口中可用信号决定。")
    lines.append("")
    lines.append("## 1. 任务路由摘要")
    lines.append(f"- {_ROUTE_SUMMARIES.get(route_id, f'{route_id}：按该任务类型选择信号，并先检查接口可用性。')}")
    lines.append("")
    lines.append("## 2. 相关奖励骨架摘要（按任务路由检索）")
    lines.append("")
    lines.append("以下骨架由任务路由检索推荐。是否使用某个骨架取决于：")
    lines.append("1. 该骨架所需信号是否在环境接口中实际可用；")
    lines.append("2. 是否与该任务阶段匹配（v1 优先设计核心学习信号，效率/安全类后续迭代加入）。")
    lines.append("")
    for sid in related:
        if sid not in SKELETON_SUMMARY:
            continue
        d = SKELETON_SUMMARY[sid]
        lines.append(f"### {sid}")
        lines.append(f"- 角色: {d['role']}")
        lines.append(f"- 数学形态: {d['math']}")
        lines.append(f"- 需要信号: {d['need']}")
        lines.append(f"- 使用说明: {d['use']}")
        lines.append(f"- 风险: {d['risk']}")
        lines.append(f"- 后续迭代: {d['later']}")
        lines.append("")
    lines.append("## 3. reward_v1 生成要求")
    lines.append("- 直接生成 reward_v1.py，不再生成 reward_design_plan.json。")
    lines.append("- 使用 role-based component budget：每个组件必须有明确角色，不能为了显得完整而堆叠。")
    lines.append("- 从上述任务路由推荐的骨架中选择，优先选择所需信号在环境接口中可用的骨架。")
    lines.append("- 如果 success/failure 显式信号不存在，不要使用 terminal_success_reward / terminal_failure_penalty。")
    lines.append("- 效率类骨架（energy_penalty、time_penalty）和复杂门控（gated_reward）默认后续迭代再加入。")
    lines.append("- 每个组件的设计要考虑可利用风险：agent 可能找到哪些捷径？条件信号是否容易被 exploit？")
    lines.append("- 返回格式建议为 return float(total_reward), components；components 必须是 dict。")
    md = "\n".join(lines)
    if len(md) > max_chars:
        md = md[:max_chars] + "\n\n<!-- truncated to max_expert_context_chars -->\n"
    return route_id, md
