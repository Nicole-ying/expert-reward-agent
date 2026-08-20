def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === 势能塑形 (Potential-Based Shaping) ===
    # 原理：F = γ * Φ(s') - Φ(s)，是唯一保证最优策略不变的塑形方式。
    # Φ(s) = -(dist + α*speed + β*|angle|)，同时引导靠近、减速、姿态稳定。
    # 相比 distance_reward = -dist'（只关心绝对位置），势能塑形奖励"改善量"，
    # 提供更稠密的梯度：靠近有奖、减速有奖、摆正有奖。

    # 当前状态
    dist = ((obs[0] - 0.0) ** 2 + (obs[1] - 0.0) ** 2) ** 0.5
    speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    angle = abs(obs[4])

    # 下一状态
    dist_next = ((next_obs[0] - 0.0) ** 2 + (next_obs[1] - 0.0) ** 2) ** 0.5
    speed_next = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_next = abs(next_obs[4])

    # 势能函数：负的加权状态代价
    alpha_speed = 0.5   # 速度权重，使 speed 分量与 dist 分量量级可比
    beta_angle = 0.5    # 角度权重，使 angle 分量与 dist 分量量级可比
    phi_now = -(dist + alpha_speed * speed + beta_angle * angle)
    phi_next = -(dist_next + alpha_speed * speed_next + beta_angle * angle_next)

    gamma = 0.99  # 接近 1，轻微折扣提供时间偏好
    potential_shaping = gamma * phi_next - phi_now

    # === 独立接触奖励（加法，不被乘积归零） ===
    # 左右腿接触均为 [0,1] 连续值
    contact_bonus = (next_obs[6] + next_obs[7]) * 0.1

    # === 总奖励 ===
    total_reward = potential_shaping + contact_bonus

    components = {
        'potential_shaping': potential_shaping,
        'contact_bonus': contact_bonus,
        'total_reward': total_reward
    }

    return float(total_reward), components