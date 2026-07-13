`evidence`：iter 4 得分 253.71（已超 target 200），terminated=20/20 全部成功着陆，episode 长度 362。landing_reward 占 signed_share 73.9% 但 active_rate 仅 11.3%（稀疏终端事件），progress 占 14.6% 活跃 95.9%，velocity_penalty 和 orientation_penalty 尺度合理。上一轮提交被判定为 iter 4 的语义重复。当前设计缺少任务明确要求的"以最小的引擎推力"信号。

`behavior_diagnosis`：策略已经学会成功着陆（全部 20 回合 terminated 成功），但可能使用了不必要的引擎推力——任务要求最小引擎使用，而当前奖励对此没有任何约束，agent 可能在飞行过程中过度使用引擎。

`signal_completeness`：核心引导信号（progress、着陆、姿态、速度约束）完备且可达，但缺少引擎效率/燃料节约职责——任务描述明确将"以最小的引擎推力"列为核心目标，当前奖励完全缺失该信号。

`selected_level`：Level 2 — 任务要求的引擎效率信号完全缺失，需新增一个组件而非调整现有系数。

`selected_intervention`：新增 `engine_penalty` 组件，对任何非零引擎动作（action ∈ {1,2,3}）施加小额恒定惩罚，鼓励 agent 在不需要推力时选择滑行（action=0）。保持其他四个组件完全不变。

`falsifiable_hypothesis`：新增引擎惩罚后，agent 应减少不必要的引擎点火，episode 长度可能缩短（更高效轨迹），同时保持成功着陆率（terminated 仍接近 20/20）。若 engine_penalty 的 episode_sum_mean 为负且 agent 仍成功，说明引擎使用确实存在冗余。

`expected_next_round`：engine_penalty 组件应出现非零负值且 active_rate 反映引擎使用频率；score 可能略降但不应跌破 200；terminated 保持 20/20；episode 长度可能缩短。

`main_risk`：若惩罚系数过大，agent 可能为避免引擎使用而不敢修正轨迹，导致着陆失败或偏离目标；选择 0.05 的小权重以降低此风险，但若效果不显可能需要后续调整。

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

    # 2. Velocity penalty – active only when close to target
    proximity_factor = 1.0 / (1.0 + 10.0 * next_dist)
    vel_weight = 1.0
    velocity_penalty = -vel_weight * proximity_factor * (nx_vel ** 2 + ny_vel ** 2)

    # 3. Orientation stability – keep body upright and avoid spinning
    orient_weight = 0.5
    orientation_penalty = -orient_weight * (n_angle ** 2 + 0.2 * n_angvel ** 2)

    # 4. Landing reward – staged: accessible contact gradient + quality bonus
    contact_score = left_contact + right_contact
    contact_weight = 0.3
    contact_reward = contact_weight * proximity_factor * contact_score

    vel_magnitude = (nx_vel ** 2 + ny_vel ** 2) ** 0.5
    settling_factor = 1.0 / (1.0 + 5.0 * vel_magnitude + 2.0 * abs(n_angle) + 1.0 * abs(n_angvel))
    contact_product = left_contact * right_contact
    settling_bonus_weight = 0.5
    settling_bonus = settling_bonus_weight * contact_product * proximity_factor * settling_factor

    landing_reward = contact_reward + settling_bonus

    # 5. Engine efficiency penalty – discourage unnecessary thrust
    #    action=0 (no_engine) is free; any engine use incurs a small constant cost
    engine_weight = 0.05
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