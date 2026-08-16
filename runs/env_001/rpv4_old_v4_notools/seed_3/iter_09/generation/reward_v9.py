def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    新主信号：基于距离差值的进展奖励 + 速度/姿态约束 + 阶段门控的着陆大奖
    目标：克服自杀式坠毁，引导 agent 学习接近、减速、稳定着陆
    """
    # ========== 权重 ==========
    w_progress    = 1.0    # 进展信号权重（每步靠近的奖励）
    w_speed       = 0.3    # 速度惩罚系数
    w_attitude    = 0.1    # 姿态/角速度惩罚系数
    w_contact     = 0.2    # 双腿接触即时奖励
    w_landing     = 100.0  # 成功着陆大奖

    # ========== 状态提取 ==========
    # 当前步绝对位置
    x, y = obs[0], obs[1]
    # 下一步
    nx, ny   = next_obs[0], next_obs[1]
    vx, vy   = next_obs[2], next_obs[3]
    angle    = next_obs[4]
    ang_vel  = next_obs[5]
    left_con = next_obs[6]
    right_con = next_obs[7]

    # 当前与下一步距离
    current_dist = (x**2 + y**2) ** 0.5
    next_dist    = (nx**2 + ny**2) ** 0.5

    # ========== 组件计算 ==========

    # 1. 稠密进展：每步靠近目标的正奖励（无门控）
    delta_dist = current_dist - next_dist
    progress_reward = w_progress * delta_dist

    # 2. 速度惩罚：抑制高速移动
    abs_speed = abs(vx) + abs(vy)
    speed_penalty = -w_speed * abs_speed

    # 3. 姿态/角速度惩罚（全局但权重适度）
    attitude_penalty = -w_attitude * (abs(angle) + abs(ang_vel))

    # 4. 双腿接触即时反馈
    both_legs = left_con * right_con  # 0 或 1
    contact_reward = w_contact * both_legs

    # 5. 成功着陆大奖（阶段门控：接触、低速、近心）
    landing_condition = (both_legs > 0.5) and (abs_speed < 0.5) and (next_dist < 1.0)
    landing_bonus = w_landing if landing_condition else 0.0

    # 6. 主引擎燃料惩罚（保留但默认权重为 0，待后续观察再启用）
    main_engine_used = 1.0 if action == 2 else 0.0
    energy_penalty = 0.0 * main_engine_used   # 当前关闭

    # ========== 总奖励 ==========
    total_reward = (
        progress_reward
        + speed_penalty
        + attitude_penalty
        + contact_reward
        + landing_bonus
        + energy_penalty
    )

    components = {
        "progress_reward":   progress_reward,
        "speed_penalty":     speed_penalty,
        "attitude_penalty":  attitude_penalty,
        "contact_reward":    contact_reward,
        "landing_bonus":     landing_bonus,
        "energy_penalty":    energy_penalty,
    }

    return float(total_reward), components