# Response Record

1. `evidence`：v5得分-119，landing_success ep_sum_mean=25但active_rate仅0.4%，几乎死组件；soft_landing ep_sum_mean=-20.64全覆盖惩罚；历史最佳v2得分-109.58，使用goal_approach+soft_landing+crash_avoidance，crash_avoidance几乎不触发，但整体行为优于当前。
2. `behavior_diagnosis`：策略在约68步内被终止（全部为着陆终止），soft_landing主导迫使快速减速和姿态控制，但因缺乏可触发的正向着陆引导，agent无法学会在接触垫后维持稳定的双腿着垫姿态，导致大量失败使得平均分仍为较大负值。
3. `signal_completeness`：已有接近信号（goal_approach）和全程稳定性惩罚（soft_landing），但缺少梯度连续、风险可控的正向着陆成功指引，当前稀疏bonus几乎不可触发，无法抵消soft_landing的压制。
4. `selected_level`：Level 2，因landing_success活性极低，数学形式无效，需将稀疏bonus改为连续舒适奖励，基于历史最佳v2骨架替换一个未触发组件。
5. `selected_intervention`：基于v2代码，把crash_avoidance（几乎不触发）替换为landing_comfort，该组件在双腿接触时根据速度和姿态误差给出连续奖励，最大每步1.0，从而提供逐步改善的梯度。
6. `falsifiable_hypothesis`：加入连续舒适奖励后，agent将在双腿接触后为获取更高奖励而主动降低速度、减小倾斜和角速度，从而提升成功着陆率，使score上升并突破-109.58。
7. `expected_next_round`：score > -109.58，len可能略增，landing_comfort的ep_sum_mean为正值且active_rate与双腿接触事件率一致。
8. `main_risk`：landing_comfort的引导可能仍不够强，不足以对抗soft_landing的惩罚，导致行为无明显改善；后续可能需调整系数。

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
    angle_penalty = abs(body_angle_next)
    ang_vel_penalty = abs(ang_vel_next)
    soft_landing = -0.2 * vel_penalty - 0.2 * angle_penalty - 0.2 * ang_vel_penalty

    # 3. landing_comfort: continuous reward when both legs touch, scaled by stability
    BOTH_LEGS = (left_contact_next > 0.5) and (right_contact_next > 0.5)
    if BOTH_LEGS:
        # comfort decreases with velocity and orientation errors
        stability_error = vel_penalty + angle_penalty + ang_vel_penalty
        comfort = max(0.0, 1.0 - stability_error)
        landing_comfort = comfort  # max +1.0 per step when perfectly stable
    else:
        landing_comfort = 0.0

    total_reward = goal_approach + soft_landing + landing_comfort

    components = {
        "goal_approach": goal_approach,
        "soft_landing": soft_landing,
        "landing_comfort": landing_comfort,
    }

    return float(total_reward), components
```
