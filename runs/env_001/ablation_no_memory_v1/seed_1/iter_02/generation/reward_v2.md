`evidence`: 全部20轮episode在68步内提前终止，得分在[-129, -100]区间，distance_reward 均值-66.5占magnitude 89%且始终为负，soft_landing_reward 均值仅0.29，agent从未到达目标垫就撞毁或飞出边界。
`behavior_diagnosis`: agent快速失败，从未靠近目标；每一步都承受负奖励累积，缺乏任何正向进展信号，信用分配极差，无法区分"接近目标"与"远离目标"的行为。
`signal_completeness`: 缺少正向进展信号——distance_reward始终为负（只是靠近时负得少），agent处于"最小化累积惩罚"而非"最大化进展"的奖励体制，必要进展反馈缺失。
`selected_level`: Level 2 — state_to_improvement。distance_reward是持久状态惩罚（persistent state penalty），满足"占据状态即可获得相对更好奖励但无法引导完成"的诊断条件，应转换为进展增量。
`selected_intervention`: 将 distance_reward 从 `-next_distance` 改为 `3.0 * (prev_distance - next_distance)`，从持久距离惩罚变为距离进展增量奖励。系数3.0用于匹配新值域：若agent以~0.1/步接近目标，则每步获得+0.3正向奖励，可与stability_penalty形成有效平衡。
`falsifiable_hypothesis`: 正向进展信号应使agent学会主动靠近目标而非徘徊；episode长度应增加（能存活更久），得分应显著改善，且early terminal比例应下降。
`expected_next_round`: score应上升（负值减小或转正），episode_length应增加（agent存活更久），distance_reward的episode_sum_mean应变正（若agent学会接近目标），soft_landing_reward magnitude应上升（agent更频繁进入近目标低速状态）。
`main_risk`: 纯进展增量无维持奖励——agent到达目标附近后可能因缺乏维持信号而漂移或振荡；若soft_landing_reward强度不足以稳定最终着陆，可能需要后续强化完成信号。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D vehicle-like trajectory optimization:
    Reach and stabilize on the central target pad with minimal engine use.
    """
    # Previous observation
    px_prev = obs[0]
    py_prev = obs[1]
    prev_distance = (px_prev**2 + py_prev**2)**0.5

    # Next observation
    px = next_obs[0]
    py = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    # Distance to target pad center
    next_distance = (px**2 + py**2)**0.5
    # Speed magnitude
    speed = (vx**2 + vy**2)**0.5

    # 1. Main learning signal: progress delta reward
    #    Positive when approaching target, negative when retreating.
    #    Transforms persistent state penalty into improvement signal.
    progress_delta = 3.0 * (prev_distance - next_distance)

    # 2. Stability constraint: light penalty on high speed, large angle, high angular velocity
    #    Helps agent learn to slow down and keep stable attitude near target
    stability_penalty = -0.1 * speed - 0.05 * abs(angle) - 0.05 * abs(ang_vel)

    # 3. Soft approaching proxy: reward getting close and slow simultaneously
    #    Acts as a smoothed "landing" surrogate without contact signals.
    sigma_dist = 0.2
    sigma_speed = 0.3
    nearness = 2.718281828 ** (-(next_distance**2) / (2.0 * sigma_dist**2))
    slowness = 2.718281828 ** (-(speed**2) / (2.0 * sigma_speed**2))
    soft_landing_reward = 1.0 * nearness * slowness

    # Combine components
    total_reward = progress_delta + stability_penalty + soft_landing_reward

    components = {
        'progress_delta': progress_delta,
        'stability_penalty': stability_penalty,
        'soft_landing_reward': soft_landing_reward
    }

    return float(total_reward), components
```