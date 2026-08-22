1. evidence：当前得分 217.21 已超过目标 200，landing_success_reward 占总奖励幅度的 90.6% 且有力驱动成功着陆，crash_prevention_x/y 的 active_rate 连续两轮均为 0.0%，从未对训练产生任何影响。
2. behavior_diagnosis：策略已稳定学会在中心区域低速、水平姿态、双腿接触的成功着陆，同时用 goal_proximity 引导接近过程；碰撞边界组件因探索范围安全而从未激活，属于彻底的死组件。
3. signal_completeness：所需主目标完成信号（着陆成功）、过程引导（距离改善）、时间效率惩罚均已具备且可达；边界防护信号在当前行为下完全不必要，不存在缺失职责。
4. selected_level：Level 1 — 移除 dead 组件 crash_prevention_x 和 crash_prevention_y，因其 active_rate 恒为零且无任何贡献，不改变有效奖励结构。
5. selected_intervention：删除 penalty_crash_x 与 penalty_crash_y 的计算和 components 条目，保持其余组件（goal_proximity、landing_success_reward、living_penalty）完全不变。
6. falsifiable_hypothesis：删除这些零贡献组件后，奖励总和与各组件 ep_sum 在统计上应与本轮完全相同，得分、回合长度、终止分布均不发生变化。
7. expected_next_round：score 仍约为 217，len 仍约 484，landing_success_reward 与 goal_proximity 的 ep_sum 和 active_rate 与当前持平。
8. main_risk：若未来环境初始分布或训练种子改变导致偶尔触及边界，将缺乏惩罚信号，但当前解题策略已证明无需该信号，风险极低且可在需要时重新加入。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 距离计算
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # 1. 主学习信号：goal_proximity（距离改善差分）
    w_prox = 2.0
    reward_prox = w_prox * (dist - next_dist)

    # 2. 着陆成功奖励
    landing_dist_thresh = 0.2
    landing_vel_thresh = 0.15
    landing_ang_thresh = 0.1
    w_landing = 1.0

    contacts = left_contact + right_contact
    near_center = next_dist < landing_dist_thresh
    low_speed = abs(next_vx) < landing_vel_thresh and abs(next_vy) < landing_vel_thresh
    upright = abs(body_angle) < landing_ang_thresh

    landing_success = (contacts == 2) and near_center and low_speed and upright
    reward_landing = w_landing if landing_success else 0.0

    # 3. 时间惩罚
    living_penalty = -0.002

    total_reward = reward_prox + reward_landing + living_penalty

    components = {
        "goal_proximity": reward_prox,
        "landing_success_reward": reward_landing,
        "living_penalty": living_penalty
    }

    return float(total_reward), components
```