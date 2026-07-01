```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward（恢复 best 系数 10.0） ==========
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_delta_reward = 10.0 * progress_delta

    # ========== 稳定约束：stability_penalty（恢复 best 系数，但将 angle 权重从 0.3 微调至 0.25） ==========
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = abs(next_obs[4])
    angular_vel_penalty = abs(next_obs[5])
    # 小幅降低 angle 权重（0.3→0.25），缓解 stability_penalty_dominance 风险
    stability_penalty = -0.5 * speed - 0.25 * angle_penalty - 0.1 * angular_vel_penalty

    # ========== 任务完成 proxy：soft_landing_proxy（恢复 best 严格条件，但将奖励从 2.0 微调至 2.5） ==========
    near_target = next_dist < 0.3
    low_speed = speed < 0.2
    stable_angle = abs(next_obs[4]) < 0.2
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    # 小幅提升奖励（2.0→2.5），在保持稀疏性的同时增强着陆激励
    soft_landing_proxy = 2.5 if (near_target and low_speed and stable_angle and both_contact) else 0.0

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