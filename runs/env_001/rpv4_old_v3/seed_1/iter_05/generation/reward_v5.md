1. evidence：当前iter=4得分-111.06，soft_landing(-19.30)主导，contact_reward仅5.3%份额且active_rate=1.7%为死组件；历史best iter=2得分-109.58，使用crash_avoidance但该组件同样为死组件，表明代理从未获得真实成功信号，仅靠goal_approach+soft_landing完成快速下落；v3的landing_success(100)激活率0.004未改善分数。
2. behavior_diagnosis：代理学会了快速飞向中心垫，但未学习减速与稳定姿态，平均68步即因硬着陆（速度/角度过大）触发失败终止，从未完成成功着陆，因此持续受soft_landing全程惩罚，score深负。
3. signal_completeness：当前奖励缺少可触发的成功着陆正向信号；goal_approach提供接近引导，soft_landing压制速度和旋转，但缺失着陆成功时的巨大正反馈，导致代理无法区分成功与失败终点。
4. selected_level：Level 2—contact_reward为死组件（active_rate<2%），其数学形态（每步+1仅双腿接触）无法激励稳定着陆，需将稀疏的接触奖励改为一次性成功条件奖励。
5. selected_intervention：唯一目标组件contact_reward，将其替换为landing_success，当同时满足双腿接触、低速、直立、低角速度条件时给予一次性+100奖励。
6. falsifiable_hypothesis：一旦landing_success可被触发，代理将获得巨大成正奖励，远超soft_landing负惩罚，从而学会在接近垫面时主动减速并维持姿态，进而提高成功着陆率并拉升score。
7. expected_next_round：landing_success的active_rate应>0且episode_sum_mean>0，soft_landing负值可能略减小（因着陆速度降低），score有望显著改善至-50以上甚至接近0；回合长度可能略微增加（因减速过程）。
8. main_risk：若阈值过严，landing_success仍为死组件，则分数无变化；若阈值过松，代理可能满足条件但未真正稳定着陆即获奖励，导致hard-coded成功但无法泛化。下一轮需根据active_rate微调阈值。

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

    # 3. landing_success: one-time bonus for stable two-leg touchdown
    BOTH_LEGS = (left_contact_next > 0.5) and (right_contact_next > 0.5)
    STABLE = (abs(x_vel_next) < 1.0 and abs(y_vel_next) < 1.0 and
              abs(body_angle_next) < 0.5 and abs(ang_vel_next) < 1.0)
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