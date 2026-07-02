def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── 提取当前状态 ──
    x, y = obs[0], obs[1]
    curr_vx, curr_vy = obs[2], obs[3]
    curr_angle = obs[4]

    # ── 提取下一时刻状态（反映动作后果）──
    next_x, next_y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 距离 ──
    dist = (x**2 + y**2) ** 0.5
    next_dist = (next_x**2 + next_y**2) ** 0.5

    # ── 速率标量 ──
    speed = (vx**2 + vy**2) ** 0.5
    curr_speed = (curr_vx**2 + curr_vy**2) ** 0.5

    # ── 主学习信号：potential-based shaping ──
    # Φ = -(distance + α*speed + β*|angle|)
    # F = γ * Φ(next) - Φ(curr)，天然引导接近+减速+姿态稳定
    alpha = 0.1
    beta = 0.2
    gamma = 0.99

    phi_curr = -(dist + alpha * curr_speed + beta * abs(curr_angle))
    phi_next = -(next_dist + alpha * speed + beta * abs(angle))
    potential_shaping = gamma * phi_next - phi_curr

    # ── 软着陆 proxy（连续乘积，阈值放宽）──
    # 用 bounded max(0, 1-x/threshold) 提供连续梯度
    prox_factor = max(0.0, 1.0 - next_dist / 1.0)               # 距离 < 1.0
    vel_factor = max(0.0, 1.0 - speed / 0.5)                    # 速率 < 0.5
    angle_factor = max(0.0, 1.0 - abs(angle) / 0.3)             # 倾角 < 0.3
    ang_vel_factor = max(0.0, 1.0 - abs(ang_vel) / 0.3)         # 角速度 < 0.3
    contact_factor = min(left_contact, right_contact)            # 双脚触地

    soft_landing_proxy = (
        prox_factor * vel_factor * angle_factor * ang_vel_factor * contact_factor
    )

    # ── 轻量辅助惩罚 ──
    # 角速度惩罚：抑制剧烈旋转
    angular_vel_penalty = -0.01 * abs(ang_vel)
    # 能量惩罚：鼓励高效移动
    energy_penalty = -0.01 * (abs(vx) + abs(vy))

    # ── 总奖励 ──
    total_reward = (
        1.0 * potential_shaping +
        2.0 * soft_landing_proxy +
        1.0 * angular_vel_penalty +
        1.0 * energy_penalty
    )

    components = {
        "potential_shaping": potential_shaping,
        "soft_landing_proxy": soft_landing_proxy,
        "angular_vel_penalty": angular_vel_penalty,
        "energy_penalty": energy_penalty,
    }

    return float(total_reward), components