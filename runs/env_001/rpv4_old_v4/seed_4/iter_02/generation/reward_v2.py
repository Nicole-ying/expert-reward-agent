def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v2。
    
    框架变更（L3）：从 improvement_delta（势能差）切换为密集状态型吸引奖励。
    核心思路：每步独立给予距离相关奖励，提供清晰连续梯度，突破势能差框架的总量上限。
    
    Args:
        obs: 当前观测 (8,) [x_pos, y_pos, x_vel, y_vel, body_angle, ang_vel, left_contact, right_contact]
        action: 当前动作 (0-3: no_engine, left_orient, main_engine, right_orient)
        next_obs: 下一观测 (8,)
        original_reward: 原始奖励（不使用）
        info: 信息字典（为空，不使用）
        training_progress: 训练进度（v2不使用）
    
    Returns:
        float: 总奖励
        dict: 各奖励组件
    """

    # ========== 信号提取 ==========
    # 当前状态
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact, right_contact = obs[6], obs[7]

    # 下一状态
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # ========== 辅助量 ==========
    # 到目标垫板的欧氏距离
    current_distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_distance = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5

    # 下一状态的合速度（标量）
    next_speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5

    # 接近度门控：距离越近值越高，用于激活近目标约束
    # exp(-2*d): d=0→1, d=0.5→0.37, d=1→0.14, d=2→0.018, d=3→0.002
    proximity_gate = 2.718281828 ** (-2.0 * next_distance)

    # ========== Component A: 目标接近（L3：密集状态型吸引奖励） ==========
    # role: goal_proximity
    # 核心改动：不再使用 improvement_delta，改用状态型公式。
    # 每步独立计算：距离越近奖励越高，提供连续梯度。
    # scale/(offset+distance): d=0→6.67, d=0.5→2.5, d=1→1.54, d=2→0.87, d=3→0.61
    state_based_reward = 2.0 / (0.3 + next_distance)

    # 小幅 improvement bonus：直接奖励「距离减小」的步，强化即时因果
    distance_improvement = current_distance - next_distance
    improvement_bonus = 1.5 * max(0.0, distance_improvement)

    goal_proximity = state_based_reward + improvement_bonus

    # ========== Component B: 稳定停靠（L2：门控速度惩罚） ==========
    # role: settling_stability
    # 改动：从 improvement-based 切换为门控绝对速度惩罚。
    # 仅在接近目标时激活，鼓励减速停靠。
    # 远距离时 proximity_gate≈0，完全不干预自由飞行。
    settling_stability = -proximity_gate * next_speed * 0.8

    # ========== Component C: 姿态约束（L2：门控绝对惩罚） ==========
    # role: attitude_constraint
    # 改动：从 improvement-based 切换为门控绝对惩罚。
    # 消除 v1 中全程持续的负 drain。
    # 仅在接近目标时要求姿态平稳（角度小 + 角速度低）。
    attitude_constraint = -proximity_gate * (
        abs(next_body_angle) * 0.5 + abs(next_ang_vel) * 0.3
    )

    # ========== Component D: 安全着陆接触（L2：放宽条件 + 增强奖励） ==========
    # role: landing_contact
    # 改动：near_target 阈值从 0.2 放宽到 0.3，增加首次双接触 bonus。
    near_target = next_distance < 0.3
    both_contacts = (next_left_contact > 0.5 and next_right_contact > 0.5)
    prev_both_contacts = (left_contact > 0.5 and right_contact > 0.5)

    landing_contact = 0.0
    if near_target and both_contacts:
        # 维持双接触的持续奖励
        landing_contact = 3.0
        # 首次获得双接触的额外奖励（状态转移事件）
        if not prev_both_contacts:
            landing_contact += 5.0

    # ========== Component E: 燃料效率（保持轻量） ==========
    # role: fuel_efficiency
    # 对任何引擎使用施加轻微惩罚
    fuel_efficiency = -0.03 * float(action != 0)

    # ========== 汇总奖励 ==========
    total_reward = (
        goal_proximity
        + settling_stability
        + attitude_constraint
        + landing_contact
        + fuel_efficiency
    )

    components = {
        'goal_proximity': goal_proximity,
        'settling_stability': settling_stability,
        'attitude_constraint': attitude_constraint,
        'landing_contact': landing_contact,
        'fuel_efficiency': fuel_efficiency
    }

    return float(total_reward), components