# Response Record

`evidence`：最终策略score=-70.35，全部20回合truncated在1000步，无提前终止；proximity episode_sum_mean=1448.02占magnitude_share 96.7%且active_rate=100%，landing_reward active_rate=0.0%从未触发，fuel_penalty每步-0.05全覆盖。agent靠持续占据中近位置获取高额状态奖励，不与真实着陆目标对齐。

`behavior_diagnosis`：agent在1000步全程使用发动机、保持y≈0.38以上、从未进入着陆奖励激活区（y<0.2），单纯在中近距离徘徊消耗时间收割proximity状态值，不尝试着陆。

`signal_completeness`：当前缺少“进展”信号——proximity是状态值而非改善量，允许agent停在好状态无限收获；landing_reward在y<0.2设硬门控且active_rate=0%，意味着着陆信号对当前策略不可达；燃料约束存在但量级被proximity淹没。

`selected_level`：Level 2，触发条件为证据模式“占据好状态即可持续获奖”→state_to_improvement变换，且landing_reward active_rate=0%确认不是单纯尺度过强而是结构性问题。

`selected_intervention`：唯一目标组件为proximity，将其从状态值（`1/(1+k|z|)`有界函数）变为基于欧氏距离的改善量（`dist_old - dist_new`），移除可被静态占位收割的性质。

`falsifiable_hypothesis`：将proximity改为距离改善量后，agent无法通过静止在中间位置获得奖励，必须持续减小与目标垫的距离才能获得正反馈，从而被驱动进入y<0.2区域并首次触发landing_reward。

`expected_next_round`：progress的episode_sum_mean应显著低于原proximity的1448（因不再能静止收割）；landing_reward的active_rate应从0%上升（首次出现非零值）；episode_length可能开始出现提前终止（成功或失败着陆而非truncated）；score应有方向性改善。

`main_risk`：progress在agent到达垫子附近后趋近于零，若landing_reward仍因乘积塌缩难以触发，agent可能在垫子附近无梯度地带随机游走，需要下一轮对landing_reward做dense_to_task_event或product_to_noncollapsing_joint变换。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x = obs[0]
    y = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    angle = obs[4]
    left_contact = obs[6]
    right_contact = obs[7]

    next_x = next_obs[0]
    next_y = next_obs[1]

    # 1. Progress: reward for reducing Euclidean distance to target pad center
    #    State-to-improvement transform: agent cannot farm by occupying a good state
    dist_old = (x ** 2 + y ** 2) ** 0.5
    dist_new = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = 5.0 * (dist_old - dist_new)

    # 2. Soft landing reward: unchanged structure, only active near the pad
    threshold_y = 0.2
    landing_weight = 2.0
    if y < threshold_y:
        contact_factor = (left_contact + right_contact) / 2.0
        speed_factor = 1.0 / (1.0 + 5.0 * (abs(x_vel) + abs(y_vel)))
        angle_factor = 1.0 / (1.0 + 10.0 * abs(angle))
        landing_bonus = landing_weight * contact_factor * speed_factor * angle_factor
        landing_reward = landing_bonus
    else:
        landing_reward = 0.0

    # 3. Fuel efficiency penalty: unchanged
    fuel_penalty = -0.05 if action != 0 else 0.0

    total_reward = progress + landing_reward + fuel_penalty

    components = {
        "progress": progress,
        "landing_reward": landing_reward,
        "fuel_penalty": fuel_penalty
    }

    return float(total_reward), components
```
