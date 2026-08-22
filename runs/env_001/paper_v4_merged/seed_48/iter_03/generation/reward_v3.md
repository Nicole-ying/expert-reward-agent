1. `evidence`：当前 score=243.73 已超目标，soft_landing_progress 占据 89.8% 的 signed_share，distance_delta 仅 8.0%，engine_penalty 负向控制有效；active_rate 全部正常，无死信号；terminated 率 95% 但未捕获双腿接触关键成功条件，contact 维度 obs[6/7] 未被使用。
2. `behavior_diagnosis`：策略已能快速抵达目标区域并低角慢速着陆，但着陆时双腿接触与否未被显式奖励，可能导致部分着陆姿态不理想（单腿或机腹触地），着陆稳固性仍有优化空间。
3. `signal_completeness`：双腿同时接触是任务定义中的“成功停靠”硬性标志，当前奖励未对该信号进行任何塑造，信号缺失；其余速度、角度、距离均已覆盖。
4. `selected_level`：Level 2 结构变换 — 依据“add 新组件（使用已声明但未用的 obs 维度）”将双腿接触因子注入 soft_landing_progress，作为乘积因子放大奖励，补全缺失的着陆稳固信号。
5. `selected_intervention`：在 soft_landing_progress 计算中引入接触放大因子 `(1.0 + 2.0 * (next_obs[6] * next_obs[7]))`，强化双腿同时接触时的即时奖励，保持非接触阶段原梯度。
6. `falsifiable_hypothesis`：接触因子提升双腿触地阶段的局部奖励，应推动策略更早、更稳定达成双腿着陆，从而使 episode 平均 score 进一步上升且 episode length 可能缩短（更快达到终止条件）。
7. `expected_next_round`：soft_landing_progress 的 episode_sum_mean 小幅增长；score 可能突破 245；episode length 可能轻微下降；contact 因子会使得最后几步的奖励尖峰更明显。
8. `main_risk`：接触放大因子可能使 agent 在接近平台时过度追求双腿接触（如频繁微小姿态调整），导致引擎使用增加或原地徘徊，需关注 engine_penalty 项是否恶化。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # obs / next_obs : [x, y, vx, vy, angle, ang_vel, left_contact, right_contact]
    # goal is at origin (0,0); x,y relative to target pad
    dist_old = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_new = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    delta_distance = dist_old - dist_new  # positive when approaching

    # Soft landing progress: encourage closeness, low speed, low angle,
    # boosted when both legs contact the target platform
    near_goal = 1.0 / (1.0 + 5.0 * dist_new)
    speed_sq = next_obs[2]**2 + next_obs[3]**2
    low_speed = 1.0 / (1.0 + 10.0 * speed_sq)
    abs_angle = abs(next_obs[4])
    low_angle = 1.0 / (1.0 + 20.0 * abs_angle)
    # contact factor: 1.0 when none or one leg, 1.0+2.0=3.0 when both legs contact
    contact_bonus = 1.0 + 2.0 * (next_obs[6] * next_obs[7])
    soft_progress = near_goal * low_speed * low_angle * contact_bonus

    # Engine usage penalty: penalize any thrust action (discrete actions 1,2,3)
    engine_penalty = 1.0 if action != 0 else 0.0

    # Weights
    w_dist = 10.0
    w_soft = 2.0
    w_engine = 0.01

    total = (w_dist * delta_distance +
             w_soft * soft_progress -
             w_engine * engine_penalty)

    components = {
        'distance_delta': w_dist * delta_distance,
        'soft_landing_progress': w_soft * soft_progress,
        'engine_penalty': -w_engine * engine_penalty,
    }
    return float(total), components
```