根据分析，当前奖励函数导致 agent 获得负外部得分并频繁早期失败，说明核心信号有误。我们需要完全重写，采用新的主信号框架：每步给予生存惩罚以鼓励快速完成任务，同时给予距离减少奖励、速度和角度惩罚（接近地面时加重），以及稳定着陆时的奖励，以及燃油惩罚。这样能使 agent 学习快速、软着陆并停靠在目标垫上。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, omega, lc, rc = obs
    nx, ny, nvx, nvy, nangle, nomega, nlc, nrc = next_obs

    # --- 生存惩罚，推动尽快完成任务（避免长时间徘徊） ---
    step_penalty = -0.1

    # --- 距离减少奖励（势场塑形） ---
    old_dist_sq = x * x + y * y
    new_dist_sq = nx * nx + ny * ny
    dist_reward = 1.0 * (old_dist_sq - new_dist_sq)

    # --- 速度惩罚，接近地面时显著加强（强迫软着陆） ---
    vel_norm = abs(nvx) + abs(nvy)
    ground_factor = 3.0 if ny < 0.5 else 1.0
    vel_penalty = -0.05 * vel_norm * ground_factor

    # --- 角度惩罚（尽量保持垂直） ---
    angle_penalty = -0.2 * abs(nangle)

    # --- 稳定着陆奖励 ---
    # 当双腿接触、高度很低、角度较小且速度很低时给予额外奖励
    if nlc > 0 and nrc > 0 and ny < 0.2 and abs(nangle) < 0.2 and vel_norm < 0.5:
        contact_bonus = 2.0
    else:
        contact_bonus = 0.0

    # --- 燃油惩罚 ---
    fuel_penalty = -0.02 if action != 0 else 0.0

    total = (
        dist_reward
        + vel_penalty
        + angle_penalty
        + contact_bonus
        + step_penalty
        + fuel_penalty
    )

    components = {
        "dist_reward": dist_reward,
        "vel_penalty": vel_penalty,
        "angle_penalty": angle_penalty,
        "contact_bonus": contact_bonus,
        "step_penalty": step_penalty,
        "fuel_penalty": fuel_penalty,
    }
    return float(total), components
```