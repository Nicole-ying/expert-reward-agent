def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D着陆器奖励函数 v1
    使用 observation [8] 和 action (0-3) 计算奖励
    """
    # ---------- 从观测中提取变量 ----------
    x_obs, y_obs = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ---------- 核心进展：到目标(0,0)的距离变化 ----------
    dist_obs = (x_obs**2 + y_obs**2)**0.5
    dist_next = (x_next**2 + y_next**2)**0.5
    progress_delta = dist_obs - dist_next   # 正 = 接近目标

    # ---------- soft_health_gate：根据下一步速度与姿态衰减进展奖励 ----------
    speed_sq = vx_next**2 + vy_next**2      # 线速度平方
    speed = speed_sq**0.5
    # 门控系数：速度越高、倾角越大，gate 越小
    gate_speed = 1.0 / (1.0 + 0.5 * speed_sq)
    gate_angle = 1.0 / (1.0 + 2.0 * abs(angle_next))
    gate_health = gate_speed * gate_angle

    # progress_reward 已经带健康门控
    progress_reward = progress_delta * gate_health

    # ---------- soft_landing_contact：双腿同时接触并满足稳定条件 ----------
    contact_bonus = 0.0
    if left_contact_next > 0.5 and right_contact_next > 0.5:
        # 速度与姿态的指数衰减因子
        speed_temp = 0.2      # 速度尺度
        angle_temp = 0.1      # 角度尺度
        stable_factor = (2.718281828 ** (-speed_sq / speed_temp)) * \
                        (2.718281828 ** (-(angle_next**2) / angle_temp))
        contact_bonus = 10.0 * stable_factor   # 基础分10，由稳定性缩放

    # ---------- fuel_penalty：惩罚引擎使用 ----------
    fuel_cost = 0.05
    fuel_penalty = -fuel_cost if action != 0 else 0.0

    # ---------- orientation_stability：接近地面时额外惩罚姿态偏差 ----------
    orientation_penalty = 0.0
    if y_next < 0.5:   # 高度阈值，接近着陆垫
        orientation_penalty = -0.5 * (angle_next**2) - 0.2 * (abs(angvel_next)**2)

    # ---------- 组合总奖励 ----------
    total_reward = progress_reward + contact_bonus + fuel_penalty + orientation_penalty

    # 记录各组件，便于调试
    components = {
        "progress": progress_reward,
        "contact_bonus": contact_bonus,
        "fuel_penalty": fuel_penalty,
        "orientation_penalty": orientation_penalty
    }
    return float(total_reward), components