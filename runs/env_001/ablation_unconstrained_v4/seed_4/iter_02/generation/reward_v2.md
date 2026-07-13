analysis：agent 在所有评估episode中提前终止（terminated=20/20），平均长度仅68.7步，score极低（-101.89）。奖励构成中，landing_vel_penalty 的负向贡献占据主导（magnitude_share 56.7%），而 progress 正向贡献不足（23.9%），stable_bonus 几乎不被触发（active_rate 1.9%）。结合观测，agent 可能因水平位置漂移出界或水平速度过大导致 crash。现有奖励仅通过高度门控合速度惩罚，无法有效约束水平方向和越界趋势，且 progress 权重较低，不足以引导 agent 快速靠近目标。修改方向：增强到达目标的吸引力（提高 progress 权重），并引入明确的水平位置约束和水平速度惩罚，同时移除从未提供有效梯度的 stable_bonus。这些改动与历史奖励函数完全不同（删除了 stable_bonus 且新增两个惩罚组件），构成实质性差异。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- hyperparameters ----------
    w_prox   = 5.0   # distance reduction weight (increased to dominate)
    w_vel    = 0.3   # landing velocity penalty weight
    w_angle  = 0.2   # body angle penalty weight
    w_angvel = 0.1   # angular velocity penalty weight
    w_xcenter = 0.5  # penalty for horizontal offset from center
    w_hspeed  = 0.2  # penalty for horizontal speed

    k_gate_y = 20.0  # steepness of height gate

    # ---------- 1. progress: distance reduction ----------
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress  = prev_dist - next_dist
    progress_reward = w_prox * progress

    # ---------- 2. soft landing velocity penalty ----------
    y_clipped = max(0.0, next_obs[1])   # y positive above pad
    gate_vel  = 1.0 / (1.0 + k_gate_y * y_clipped)
    speed     = (next_obs[2]**2 + next_obs[3]**2) ** 0.5
    landing_vel_penalty = -w_vel * speed * gate_vel

    # ---------- 3. upright stability penalty ----------
    upright_penalty = -w_angle * abs(next_obs[4]) - w_angvel * abs(next_obs[5])

    # ---------- 4. horizontal offset penalty ----------
    x_center_penalty = -w_xcenter * abs(next_obs[0])

    # ---------- 5. horizontal speed penalty ----------
    hspeed_penalty = -w_hspeed * abs(next_obs[2])

    # ---------- total reward ----------
    total_reward = (progress_reward +
                    landing_vel_penalty +
                    upright_penalty +
                    x_center_penalty +
                    hspeed_penalty)

    components = {
        'progress': progress_reward,
        'landing_vel_penalty': landing_vel_penalty,
        'upright_penalty': upright_penalty,
        'x_center_penalty': x_center_penalty,
        'hspeed_penalty': hspeed_penalty
    }
    return float(total_reward), components
```