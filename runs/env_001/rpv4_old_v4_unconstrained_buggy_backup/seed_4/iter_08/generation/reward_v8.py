def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v8。

    核心改动（L1）：landing_contact 系数 1.2 → 0.9，进一步降低接触奖励的
    magnitude share（v7=70%），使 penalty 组件获得更合理的相对权重，
    减少训练中期的策略振荡，提升最终策略的一致性。

    设计原则（继承 v4/v7）：
    - goal_proximity: 二次距离代价 → 引导接近（远处强、近处弱）
    - settling_stability: 二次速度代价 → 引导减速（高速强、低速弱）
    - landing_contact: 连续接触奖励 → 奖励"接近+慢速+接触"的着陆过程
    - attitude_constraint: 姿态代价 → 保持平稳
    - fuel_efficiency: 燃料代价 → 节省推力

    Args:
        obs: 当前观测 (8,)
        action: 当前动作 (0-3)
        next_obs: 下一观测 (8,)
        original_reward: 未使用
        info: 未使用
        training_progress: 未使用
    """

    # ========== 信号提取（使用 next_obs 评估转移后状态） ==========
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ========== 辅助量 ==========
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5

    # proximity_gate: 1.0 在目标处，~0.1 在 distance≈3
    proximity_gate = 1.0 / (1.0 + 3.0 * distance)

    # ========== Component A: goal_proximity — 二次距离代价 ==========
    # 梯度 = -0.2 * distance，远处梯度强，近处梯度弱（天然 curriculum）
    goal_proximity = -0.1 * distance ** 2

    # ========== Component B: settling_stability — 二次速度代价 ==========
    # 梯度随 speed 和 proximity 增强，接近目标时减速约束更强
    settling_stability = -0.3 * speed ** 2 * (0.2 + 0.8 * proximity_gate)
    settling_stability += -0.08 * ang_vel ** 2 * (0.2 + 0.8 * proximity_gate)

    # ========== Component C: attitude_constraint — 姿态代价 ==========
    angle_error = abs(body_angle)
    ang_vel_mag = abs(ang_vel)
    # gate 范围：远距离 0.3，近距离 1.0
    attitude_gate = 0.3 + 0.7 * proximity_gate
    attitude_constraint = -0.4 * angle_error * attitude_gate
    attitude_constraint += -0.15 * ang_vel_mag * attitude_gate

    # ========== Component D: landing_contact — 连续接触奖励 ==========
    # v8：系数 1.2 → 0.9（L1 微调），降低 magnitude share 以减少训练振荡。
    # 仅在实际有支撑腿接触、靠近垫板、且速度足够低时发放奖励。
    # 三个条件共同防止 agent 在无关阶段获得误奖励：
    #   - near_pad: 距离 < 0.5，确保只在目标附近激活
    #   - slow_enough: 速度 < 1.0，防止高速撞击时领奖
    #   - contact_frac > 0: 至少一条腿接触，推动完成最后一步
    near_pad = distance < 0.5
    slow_enough = speed < 1.0
    contact_frac = (left_contact + right_contact) / 2.0  # 0.0 ~ 1.0

    if near_pad and slow_enough and contact_frac > 0.0:
        # 连续缩放因子：越近越好、越慢越好
        prox_factor = 1.0 - distance / 0.5      # 1.0@d=0, 0.0@d=0.5
        speed_factor = 1.0 - speed / 1.0        # 1.0@s=0, 0.0@s=1.0
        # v8: 系数 0.9（v7=1.2, v4=2.0），平衡 reward composition
        landing_contact = 0.9 * prox_factor * speed_factor * contact_frac
    else:
        landing_contact = 0.0

    # ========== Component E: fuel_efficiency — 燃料代价 ==========
    fuel_efficiency = -0.03 * float(action != 0)

    # ========== 汇总 ==========
    total_reward = (
        goal_proximity +
        settling_stability +
        attitude_constraint +
        landing_contact +
        fuel_efficiency
    )

    components = {
        'goal_proximity': goal_proximity,
        'settling_stability': settling_stability,
        'attitude_constraint': attitude_constraint,
        'landing_contact': landing_contact,
        'fuel_efficiency': fuel_efficiency
    }

    return float(total_reward), components