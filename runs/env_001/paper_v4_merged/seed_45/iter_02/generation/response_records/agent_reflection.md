# Response Record

1. evidence：外部得分128.47，episode全超时1000步无terminated，progress_delta步均0.002但总和仅1.40，speed/angle惩罚active_rate<2%近乎僵尸，left/right接触维度完全未用，存在着陆引导信号缺失。
2. behavior_diagnosis：策略在超时徘徊中存活，既不crash也不成功着陆，因缺乏接触与终局激励，主体无法学会精确下降并稳定接触以触发终止。
3. signal_completeness：当前信号缺少着陆成功/接触激励，未使用的left/right接触恰能填补缺口，职责尚不完备。
4. selected_level：Level 2 — 信号缺口触发add新组件（使用已声明但未用的obs维度）。
5. selected_intervention：新增组件`contact_reward`，依据next_obs[6]和[7]给予双脚接触正向激励。
6. falsifiable_hypothesis：添加接触奖励会激励agent学习降低速度并精确着陆以获取持续接触奖励，从而缩短episode并触发终止，提升外部得分。
7. expected_next_round：episode_length应缩短（出现terminated而非全truncated），contact_reward active_rate>0且magnitude_share上升，外部score从128上升。
8. main_risk：接触奖励可能过强导致agent过早粗暴动作，增加crash失败率，若下一轮出现大量早期终止且得分下降则需削弱该系数。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observations
    x = obs[0]
    y = obs[1]
    vx = obs[2]
    vy = obs[3]
    angle = obs[4]
    angvel = obs[5]
    # left_contact, right_contact from obs (not used)
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_vx = next_obs[2]
    next_vy = next_obs[3]
    next_angle = next_obs[4]
    next_angvel = next_obs[5]
    next_left = next_obs[6]
    next_right = next_obs[7]

    # 1. Progress towards center (0,0)
    dist = (x**2 + y**2)**0.5 + 1e-6
    next_dist = (next_x**2 + next_y**2)**0.5 + 1e-6
    progress_delta = dist - next_dist

    # 2. Orientation stability penalty (hinge)
    angle_threshold = 0.3
    angvel_threshold = 0.5
    angle_violation = max(0.0, abs(next_angle) - angle_threshold)
    angvel_violation = max(0.0, abs(next_angvel) - angvel_threshold)
    orientation_penalty = -0.1 * angle_violation - 0.05 * angvel_violation

    # 3. Speed safety penalty (hinge)
    speed_threshold = 0.5
    vx_violation = max(0.0, abs(next_vx) - speed_threshold)
    vy_violation = max(0.0, abs(next_vy) - speed_threshold)
    speed_penalty = -0.05 * (vx_violation + vy_violation)

    # 4. NEW: contact encouragement (dense reward for feet on ground)
    contact_reward = 0.1 * (next_left + next_right)

    total_reward = progress_delta + orientation_penalty + speed_penalty + contact_reward

    components = {
        'progress_delta': progress_delta,
        'orientation_penalty': orientation_penalty,
        'speed_penalty': speed_penalty,
        'contact_reward': contact_reward
    }

    return float(total_reward), components
```
