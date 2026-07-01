# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward（增强驱动力） ==========
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    # 系数从10.0提升至20.0，增强接近目标的驱动力
    progress_delta_reward = 20.0 * progress_delta

    # ========== 稳定约束：stability_penalty（大幅降低权重，避免过度保守） ==========
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = abs(next_obs[4])
    angular_vel_penalty = abs(next_obs[5])
    # speed系数从0.5降至0.2，angle系数从0.3降至0.1，angular_vel系数从0.1降至0.05
    stability_penalty = -0.2 * speed - 0.1 * angle_penalty - 0.05 * angular_vel_penalty

    # ========== 任务完成 proxy：soft_landing_proxy（放宽条件，提高触发率） ==========
    # near_target阈值从0.3放宽至0.5，low_speed阈值从0.2放宽至0.3，stable_angle阈值从0.2放宽至0.3
    near_target = next_dist < 0.5
    low_speed = speed < 0.3
    stable_angle = abs(next_obs[4]) < 0.3
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    # 奖励从2.0提升至3.0，增强着陆激励
    soft_landing_proxy = 3.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # ========== 动作代价：energy_penalty（保持小权重） ==========
    energy_penalty = -0.1 if action != 0 else 0.0

    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy + energy_penalty

    # ========== 组件字典 ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components
```
