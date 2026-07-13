1. `evidence`：iter 5 加入 engine_penalty（权重 0.05）后，score 从 253.71 降到 249.34，episode 长度从 362 升到 374.85；engine_penalty 的 episode_sum_mean 为 -17.03，是 progress（+6.91）的 2.46 倍，远超健康比例经验线（~0.5）；active_rate 90.8% 说明几乎每步都在扣分。
2. `behavior_diagnosis`：agent 已能成功着陆（20/20 terminated），但过重的引擎惩罚迫使策略过度节流，导致接近速度变慢、episode 拉长、最终得分反而下降。
3. `signal_completeness`：所有必要职责（引导、近端减速、姿态稳定、着陆质量、引擎效率）均已覆盖且可达；问题不在缺失信号，而在 engine_penalty 的尺度失衡。
4. `selected_level`：Level 1 尺度修复 — engine_penalty 职责和数学形态合理，但其系数 0.05 产生的累计幅度压倒 progress 主信号，符合 penalty_magnitude_control 触发条件。
5. `selected_intervention`：将 engine_weight 从 0.05 降至 0.005（10 倍降低），其他组件完全不动。
6. `falsifiable_hypothesis`：降低引擎惩罚因子后，agent 在导航阶段不再过度抑制引擎使用，能够更快接近目标；进度效率提升应反映为 episode 长度下降、score 回到或超过 253.71。
7. `expected_next_round`：engine_penalty 的 episode_sum_mean 应从 -17 降至约 -1.7；progress 可能小幅上升；episode 长度应降到 ~365 或更低；score 应回升到 253+。
8. `main_risk`：惩罚过轻可能导致 agent 在悬停或微调阶段多用一些引擎，但 progress 主信号和 velocity_penalty 近端减速仍会约束整体效率，不会出现无节制喷射。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current state
    x_pos, y_pos = obs[0], obs[1]
    # Next state
    nx_pos, ny_pos = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Distances to target
    prev_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (nx_pos ** 2 + ny_pos ** 2) ** 0.5

    # 1. Progress reward: positive for moving toward target, negative for moving away
    progress_weight = 5.0
    progress_reward = progress_weight * (prev_dist - next_dist)

    # 2. Velocity penalty - active only when close to target
    proximity_factor = 1.0 / (1.0 + 10.0 * next_dist)
    vel_weight = 1.0
    velocity_penalty = -vel_weight * proximity_factor * (nx_vel ** 2 + ny_vel ** 2)

    # 3. Orientation stability - keep body upright and avoid spinning
    orient_weight = 0.5
    orientation_penalty = -orient_weight * (n_angle ** 2 + 0.2 * n_angvel ** 2)

    # 4. Landing reward - staged: accessible contact gradient + quality bonus
    contact_score = left_contact + right_contact
    contact_weight = 0.3
    contact_reward = contact_weight * proximity_factor * contact_score

    vel_magnitude = (nx_vel ** 2 + ny_vel ** 2) ** 0.5
    settling_factor = 1.0 / (1.0 + 5.0 * vel_magnitude + 2.0 * abs(n_angle) + 1.0 * abs(n_angvel))
    contact_product = left_contact * right_contact
    settling_bonus_weight = 0.5
    settling_bonus = settling_bonus_weight * contact_product * proximity_factor * settling_factor

    landing_reward = contact_reward + settling_bonus

    # 5. Engine efficiency penalty - discourage unnecessary thrust
    #    action=0 (no_engine) is free; any engine use incurs a small constant cost
    engine_weight = 0.005
    engine_penalty = -engine_weight * (1.0 if action != 0 else 0.0)

    total_reward = progress_reward + velocity_penalty + orientation_penalty + landing_reward + engine_penalty

    components = {
        'progress': float(progress_reward),
        'velocity_penalty': float(velocity_penalty),
        'orientation_penalty': float(orientation_penalty),
        'landing_reward': float(landing_reward),
        'engine_penalty': float(engine_penalty)
    }

    return float(total_reward), components
```