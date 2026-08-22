1. `evidence`：v2 将 soft_landing 改为全高度线性惩罚后 score 从 -214 涨到 -109，但 v3 将 crash_avoidance 替换为稀疏的 landing_success 后 score 回落到 -115，且 landing_success 成为死组件（active_rate 0.4%），soft_landing 仍是全覆盖惩罚。
2. `behavior_diagnosis`：当前策略在各轨迹中均因累积 reward 过早触发 early_terminal，未能完成稳定着陆；agent 快速下降但未学会维持双腿接触并减速至成功终止条件。
3. `signal_completeness`：现有信号中缺少任何可学习的着陆成功引导——landing_success 过于稀疏无法提供梯度，soft_landing 虽然压制高速但无法驱动接触；需要增加一个连续的双腿接触奖励，以形成“接触→减速→稳定”的完整链条。
4. `selected_level`：Level 2，将死组件 landing_success 的稀疏奖励改为连续的双腿接触奖励，使其在接触的每一步都产生正向激励，不再依赖稳定条件的罕见同时满足。
5. `selected_intervention`：唯一目标组件为 landing_success，将其替换为 contact_reward（每步当 left_contact_next>0.5 且 right_contact_next>0.5 时给予 +1.0），并移除 crash_avoidance。
6. `falsifiable_hypothesis`：连续接触奖励会为 agent 提供直接梯度，促使其尽快放下双腿接触垫面，同时 soft_landing 仍会惩罚高速/倾斜，从而引导它学会在接触后继续减速并保持稳定，最终提高成功着陆率并拉升 score。
7. `expected_next_round`：下一轮 soft_landing 的 ep_sum 可能略微减小（因速度降低），contact_reward 会有较高 active_rate 和正的 episode_sum，score 应显著上升（趋于正值），len 可能变化，但 early_terminal 比例应明显下降。
8. `main_risk`：agent 可能学会快速碰触双腿但随后保持高速滚动或轻微弹跳，仅靠接触奖励无法独立保证稳定着陆，若 soft_landing 惩罚不足以迫使完全稳定，可能导致大量 episode 仍无法触发成功终止并持续负分。

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

    # 3. contact_reward: continuous per-step reward for having both legs on pad
    BOTH_LEGS = (left_contact_next > 0.5) and (right_contact_next > 0.5)
    if BOTH_LEGS:
        contact_reward = 1.0
    else:
        contact_reward = 0.0

    total_reward = goal_approach + soft_landing + contact_reward

    components = {
        "goal_approach": goal_approach,
        "soft_landing": soft_landing,
        "contact_reward": contact_reward,
    }

    return float(total_reward), components
```