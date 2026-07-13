`evidence`：landing_bonus 当前 episode_sum_mean=3456、active_rate=7.1%，按转移事件设计应每回合最多触发一次，但实际每回合平均触发 ~42 次（597×7.1%），形成弹跳重复收集漏洞。shaped_progress 均值仅 4.36，占总奖励份额 0.1%，引导信号极弱。12/20 terminated、8/20 truncated，40% 的回合无法在时限内完成着陆。

`behavior_diagnosis`：策略学会在着陆垫上反复弹跳来重复收割转移事件奖励（landing_bonus），而非执行单次稳定着陆。同时进度信号过弱，导致 40% 回合无法有效引导至着陆垫而超时。

`signal_completeness`：进度信号（距离改善）存在但被 landing_bonus 完全淹没；着陆完成信号存在但数学形态（转移事件）允许弹跳重复触发，缺乏"仅首次着陆有效"的机制。必要职责表面完备但结构有漏洞。

`selected_level`：Level 2。着陆奖励的转移事件结构被弹跳行为系统性利用（active_rate 7.1% 而非预期的 <1%），单纯调整系数无法消除重复触发漏洞。触发 `persistent_to_transition_event` 的反向变换：将转移事件改为持续状态奖励，消除弹跳的增量收益。

`selected_intervention`：将 landing_bonus 从转移事件（`100 × landing_transition × gate`）改为持续着陆质量奖励（`10 × is_landed × near_target × low_speed × upright`），每步处于良好着陆状态时持续获得奖励。弹跳（离开再接触）不再产生额外收益，因为总奖励仅取决于累积着地时长而非接触次数。

`falsifiable_hypothesis`：弹跳行为应消失，landing_quality 的 episode_sum_mean 应从当前 ~3456 下降到与实际着地时长成比例的水平；shaped_progress 的份额应相对上升；truncated 比例应下降，整体 score 应提升。

`expected_next_round`：landing_quality 的 active_rate 应反映实际着地时间占比（~5-10%），episode_sum_mean 大幅下降；shaped_progress 的 signed_share 上升；terminated 比例提高；score 向 200 靠近。

`main_risk`：持续状态奖励可能激励智能体在着陆后停留而不触发环境终止（若环境终止条件较慢），导致 episode 延长或 stall。但当前 12/20 已能正常终止，风险可控。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 主学习信号：距离改善 × 姿态门控（保持不变）
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_raw = dist_current - dist_next

    angle_abs = abs(next_obs[4])
    gate      = 1.0 / (1.0 + 5.0 * angle_abs)
    shaped_progress = 5.0 * progress_raw * gate

    # 2. 安全约束：角速度轻量惩罚（保持不变）
    ang_vel_penalty = -0.1 * (next_obs[5] ** 2)

    # 3. 着陆质量奖励：持续状态奖励（替代单次转移事件）
    # 消除弹跳重复触发：双腿着地期间按质量持续给分，离开再接触无额外收益
    is_landed = next_obs[6] * next_obs[7]

    near_target = 1.0 / (1.0 + 5.0 * abs(next_obs[0]))
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    low_speed = 1.0 / (1.0 + 2.0 * speed)
    upright = 1.0 / (1.0 + 5.0 * angle_abs)

    landing_quality = 10.0 * is_landed * near_target * low_speed * upright

    total_reward = shaped_progress + ang_vel_penalty + landing_quality
    components = {
        "shaped_progress": shaped_progress,
        "angular_vel_penalty": ang_vel_penalty,
        "landing_quality": landing_quality
    }
    return (float(total_reward), components)
```