def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_obs, y_obs = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ---------- progress reward ----------
    dist_obs = (x_obs**2 + y_obs**2)**0.5
    dist_next = (x_next**2 + y_next**2)**0.5
    progress_delta = dist_obs - dist_next

    speed_sq = vx_next**2 + vy_next**2
    gate_speed = 1.0 / (1.0 + 0.1 * speed_sq)
    gate_angle = 1.0 / (1.0 + 0.5 * abs(angle_next))
    gate_health = gate_speed * gate_angle
    progress_reward = 5.0 * progress_delta * gate_health

    # ---------- contact_bonus with curriculum ----------
    descent_target_vy = -0.5
    vy_error = vy_next - descent_target_vy

    # 调度因子：初期放大方差（容忍度更高），后期收缩至原值
    scale = 1.0 + 4.0 * (1.0 - training_progress)   # training_progress 0→1 ⇒ scale 5→1

    # 近地引导：使用宽松因子
    near_soft_factor = (2.718281828 ** (-(vy_error**2) / (0.1 * scale))) * \
                       (2.718281828 ** (-(vx_next**2) / (0.1 * scale))) * \
                       (2.718281828 ** (-(angle_next**2) / (0.05 * scale))) * \
                       (2.718281828 ** (-(angvel_next**2) / (0.05 * scale)))

    # 接触奖励：始终使用严格因子（原参数）
    contact_soft_factor = (2.718281828 ** (-(vy_error**2) / 0.1)) * \
                          (2.718281828 ** (-(vx_next**2) / 0.1)) * \
                          (2.718281828 ** (-(angle_next**2) / 0.05)) * \
                          (2.718281828 ** (-(angvel_next**2) / 0.05))

    contact_bonus = 0.0
    if y_next < 1.0:
        contact_bonus = 5.0 * near_soft_factor   # 近地鼓励良好的下降与姿态（宽容）

    if left_contact_next > 0.5 and right_contact_next > 0.5:
        contact_bonus += 30.0 * contact_soft_factor # 接触时大额稳定性奖励（严格）

    # ---------- fuel penalty ----------
    fuel_cost = 0.05
    fuel_penalty = -fuel_cost if action != 0 else 0.0

    # ---------- near-ground orientation penalty ----------
    orientation_penalty = 0.0
    if y_next < 0.5:
        orientation_penalty = -0.5 * (angle_next**2) - 0.2 * (abs(angvel_next)**2)

    total_reward = progress_reward + contact_bonus + fuel_penalty + orientation_penalty

    components = {
        "progress": progress_reward,
        "contact_bonus": contact_bonus,
        "fuel_penalty": fuel_penalty,
        "orientation_penalty": orientation_penalty
    }
    return float(total_reward), components