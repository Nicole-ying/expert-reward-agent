# Response Record

1. `evidence`：v2 的 soft_landing 修改使分数从 -214 升至 -109，length 从 183 压到 68，说明全程减速/稳定惩罚是有效行为驱动源；但由于所有 episode 仍以 failure termination 结束（score<-50 且在 150 步前终止），且 crash_avoidance 完全未激活（active 0%），说明缺少正向着陆完成信号。
2. `behavior_diagnosis`：agent 学会了快速下坠并尝试维持低速和直立，但最终无法完成安全着陆——可能因水平越界或高速触地而失败，且没有任何成功 episode，导致平均分数仍为负值。
3. `signal_completeness`：目标方向引导（goal_approach）和减速/稳定惩罚（soft_landing）已具备，但缺少着陆成功的正向激励来区分安全着陆与其他终止状态；环境已提供接触、速度、角度等信号，可达且合法。
4. `selected_level`：Level 2（修改公式形式）——crash_avoidance 为死组件，替换其函数形式为着陆成功奖励是最小且可验证的干预。
5. `selected_intervention`：唯一目标组件为 `crash_avoidance`，将其替换为 `landing_success`：当双腿接触且满足低速、直立、低角速度条件时，给予一次性正向大奖励。
6. `falsifiable_hypothesis`：增加 100.0 的着陆成功奖励后，agent 将在 soft_landing 的引导下逐步探索到安全着陆状态，一旦成功即获得巨大正奖励，从而使平均分数由负转正；下一轮将观察到 landing_success 的 active_rate > 0 且 episode_sum_mean > 0，score 明显上升。
7. `expected_next_round`：score 应由负转正（至少 -50 以上），部分 episode 成功终止（不再全部 early terminal），landing_success 出现正激活和正 ep_sum，len 可能因成功而略增或持平，soft_landing 负贡献可能因成功前减速而略微减少。
8. `main_risk`：着陆成功条件阈值可能过严导致 agent 始终无法触发奖励，从而无改进；或过松导致非稳态着陆被误奖励，引入欺骗性正分但未真正解决安全着陆。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current observations
    x_pos = obs[0]
    y_pos = obs[1]
    # Unpack next observations
    x_pos_next = next_obs[0]
    y_pos_next = next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    body_angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # 1. goal_approach: potential-based shaping (distance to center)
    dist_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    dist_next = (x_pos_next ** 2 + y_pos_next ** 2) ** 0.5
    goal_approach = dist_current - dist_next   # positive when approaching

    # 2. soft_landing: full-range velocity, angle, angular velocity penalty
    vel_penalty = abs(x_vel_next) + abs(y_vel_next)
    angle_penalty = abs(body_angle_next)       # rad, should be near 0
    ang_vel_penalty = abs(ang_vel_next)
    soft_landing = -0.2 * vel_penalty - 0.2 * angle_penalty - 0.2 * ang_vel_penalty

    # 3. landing_success: sparse positive reward for safe touchdown
    BOTH_LEGS = (left_contact_next > 0.5) and (right_contact_next > 0.5)
    STABLE = (abs(x_vel_next) < 0.5 and abs(y_vel_next) < 0.5 and 
              abs(body_angle_next) < 0.2 and abs(ang_vel_next) < 0.2)
    if BOTH_LEGS and STABLE:
        landing_success = 100.0
    else:
        landing_success = 0.0

    total_reward = goal_approach + soft_landing + landing_success

    components = {
        "goal_approach": goal_approach,
        "soft_landing": soft_landing,
        "landing_success": landing_success,
    }

    return float(total_reward), components
```
