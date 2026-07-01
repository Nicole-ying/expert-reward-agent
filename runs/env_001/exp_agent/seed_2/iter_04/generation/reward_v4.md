## 诊断分析

**核心问题：agent 在 72 步内全部提前终止（100%），score 极负（-111），说明 agent 在到达目标前就坠毁了。**

从 ratio_to_progress 看：
1. **progress_delta_reward** (ratio=1.0, mean=0.16) — 正常，agent 确实在向目标移动
2. **stability_penalty** (ratio=-0.24, mean=-0.039) — 绝对值 < 0.5，不算主导，但当前系数可能仍不够
3. **soft_landing_bonus** (nonzero_rate=1.8%) — 几乎从不触发，无法提供着陆引导
4. **energy_penalty** (ratio=-0.055) — 很小，不是问题

**根因分析**：agent 在接近目标的过程中姿态失稳坠毁。stability_penalty 的 ratio 只有 -0.24，说明惩罚不够强到让 agent 学会保持稳定。同时 soft_landing_bonus 的 nonzero_rate 仅 1.8%，说明 agent 几乎从未成功着陆——它在到达着陆条件前就 crash 了。

**修改策略（层次 1+2 混合）**：
1. **增强 stability_penalty**：当前系数太弱（angle=-0.1, angular_vel=-0.05, speed=-0.03），agent 不在乎姿态。增大系数，让 agent 学会在移动时保持直立。
2. **改进 soft_landing_bonus**：当前已经是连续乘积形式，但 nonzero_rate 太低。增大权重并放宽条件，让它在更早的阶段就能提供引导信号。
3. **保持 progress_delta_reward** 不变 — 它工作正常。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. Main learning signal: progress_delta_reward (unchanged)
    # Agent is making progress toward target — keep this working signal
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_delta_reward = 10.0 * progress_delta
    
    # 2. Stability penalty — INCREASED significantly
    # Agent crashes at ~72 steps because it falls over while moving toward target.
    # Current penalties (angle=-0.1, angular_vel=-0.05, speed=-0.03) are too weak.
    # Need stronger signal to teach upright posture during approach.
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -1.0 * abs(next_body_angle)       # was -0.1, 10x increase
    angular_vel_penalty = -0.5 * abs(next_angular_vel) # was -0.05, 10x increase
    speed_penalty = -0.2 * speed                        # was -0.03, ~7x increase
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. Soft landing proxy — INCREASED weight and relaxed conditions
    # nonzero_rate=1.8% is too low — agent almost never reaches landing state.
    # Increase bonus magnitude and make the factors less strict so agent gets
    # gradient signal earlier (before perfect landing conditions are met).
    proximity_factor = 1.0 / (1.0 + 3.0 * next_dist)   # was 5.0, now broader
    speed_factor = 1.0 / (1.0 + 3.0 * speed)            # was 5.0, now broader
    angle_factor = 1.0 / (1.0 + 5.0 * abs(next_body_angle))  # was 10.0, now broader
    angular_vel_factor = 1.0 / (1.0 + 3.0 * abs(next_angular_vel))  # was 5.0, now broader
    contact_factor = next_left_contact * next_right_contact  # continuous in [0,1]
    
    soft_landing_bonus = 20.0 * proximity_factor * speed_factor * angle_factor * angular_vel_factor * contact_factor  # was 10.0, now 20.0
    
    # 4. Small energy penalty for using engines (unchanged)
    energy_penalty = 0.0
    if action != 0:
        energy_penalty = -0.05
    
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + energy_penalty
    
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```