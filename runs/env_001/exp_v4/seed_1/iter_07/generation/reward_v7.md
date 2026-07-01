```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward（大幅提高权重） ==========
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_delta_reward = 20.0 * progress_delta  # 从10.0提高到20.0，增强驱动

    # ========== 稳定约束：stability_penalty（大幅降低权重，优先惩罚速度） ==========
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = abs(next_obs[4])
    angular_vel_penalty = abs(next_obs[5])
    # 降低所有系数，speed保持相对权重，angle和angular_vel大幅降低
    stability_penalty = -0.2 * speed - 0.1 * angle_penalty - 0.05 * angular_vel_penalty

    # ========== 着陆接近 shaping：替换稀疏的 soft_landing_proxy ==========
    # 当接近目标时，根据速度和姿态给予连续 shaping 奖励，鼓励减速和稳定
    near_target = next_dist < 0.5
    if near_target:
        # 速度越低、角度越小，奖励越大（连续 shaping）
        speed_quality = max(0.0, 1.0 - speed / 0.5)  # speed 0 -> 1.0, speed 0.5 -> 0.0
        angle_quality = max(0.0, 1.0 - abs(next_obs[4]) / 0.5)  # angle 0 -> 1.0, angle 0.5 -> 0.0
        landing_shaping = 1.0 * speed_quality + 0.5 * angle_quality
    else:
        landing_shaping = 0.0

    # ========== 动作代价：energy_penalty（保持不变） ==========
    energy_penalty = -0.1 if action != 0 else 0.0

    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + landing_shaping + energy_penalty

    # ========== 组件字典 ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping": landing_shaping,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components
```