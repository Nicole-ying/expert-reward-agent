# Response Record

1. `evidence`：v2 的 success_landing_bonus 激活率仅 0.1% 但 score 达 -119.5；v3 新增 crash_penalty 导致 success 死掉、progress 塌陷、angle 和 action 惩罚大涨，score 暴跌至 -203.4，所有终止均为失败。
2. `behavior_diagnosis`：crash_penalty 使 agent 学会避免一切地面接触，长期悬停并累积姿态和动作惩罚，progress 信号也因远离目标而减弱，无法接近软着陆触发区。
3. `signal_completeness`：缺少可触发的成功奖励和有效的近地减速引导；原 success_landing_bonus 阈值过严，agent 几乎不可能在随机探索中获得正向信号。
4. `selected_level`：Level 2，修改 success_landing_bonus 的触发条件，放宽阈值，同时移除 crash_penalty 以恢复 v2 有效结构。
5. `selected_intervention`：删除 crash_penalty，将 success_landing_bonus 的判定距离阈值放宽到 0.2，速度阈值放宽到 0.5，角度阈值放宽到 0.2。
6. `falsifiable_hypothesis`：放宽条件后 success_landing_bonus 激活率将显著上升，为 agent 提供可达的正向终极奖励，驱动减速和软着陆，从而使 score 大幅提升。
7. `expected_next_round`：success_landing_bonus 的 ep_sum_mean 应 > 0，active_rate 应 > 1%，score 应超过 -119.5，len 可能因学习成功着陆而下降或维持中等。
8. `main_risk`：放宽阈值可能让 agent 在未真正稳定着陆时获得 100 奖励，带来奖励冲刷风险，但 progress_shaping 和角度惩罚仍会引导更精确的行为，风险可控。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 0. 解析观测
    x_pos, y_pos = obs[0], obs[1]
    body_angle = obs[4]

    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # 1. 主学习信号: 距离势能塑造 (progress_shaping)
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    potential_curr = -dist
    potential_next = -next_dist
    progress_reward = 5.0 * (potential_next - potential_curr)

    # 2. 成功着陆奖励 (success_landing_bonus) —— 放宽阈值
    contact_any = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    landed = (
        next_dist < 0.2 and
        abs(next_x_vel) < 0.5 and
        abs(next_y_vel) < 0.5 and
        abs(next_body_angle) < 0.2 and
        contact_any
    )
    success_bonus = 100.0 if landed else 0.0

    # 3. 姿态惩罚
    angle_penalty = -0.5 * (body_angle ** 2)

    # 4. 动作代价
    action_cost = -0.02 if action != 0 else 0.0

    # 5. 总奖励
    total_reward = (
        progress_reward +
        success_bonus +
        angle_penalty +
        action_cost
    )

    components = {
        "progress_shaping": progress_reward,
        "success_landing_bonus": success_bonus,
        "angle_penalty": angle_penalty,
        "action_cost": action_cost
    }

    return float(total_reward), components
```
