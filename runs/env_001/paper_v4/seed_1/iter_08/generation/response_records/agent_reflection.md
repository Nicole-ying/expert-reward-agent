# Response Record

`evidence`: score=24.3远低于best=120.0, landing_bonus episode_sum_mean=390.3占magnitude_share 98.1%, 但active_rate仅6.3%, shaped_progress均值仅3.49且被淹没, 3/20 episode早期终止且得分<-50. 上一轮(iter 7)与best(iter 5)结构同族但landing_bonus从~3.4膨胀到~390, 乘积式质量因子导致bonus既稀疏又极值支配。

`behavior_diagnosis`: agent在部分episode中成功触发接触过渡并获得密集小增量累积(active_rate 6.3% × ~575步≈36次), 乘积式质量因子让bonus在接触阶段膨胀到390, 完全淹没shaped_progress; 另3/20 episode快速失败得分<-492, 说明agent未学到稳定接近行为, 因为稠密引导信号shaped_progress在奖励中几乎不可见。

`signal_completeness`: 当下shaped_progress提供稠密距离改善引导, angular_vel_penalty提供姿态约束, landing_bonus提供完成信号。但landing_bonus采用`new_contact × near_target × upright × low_speed`四因子乘积, 任一因子略低则整体塌缩至接近零, 导致有效信用仅在一小部分步数中释放, 且释放时量级失控。必要职责存在但landing_bonus的数学形态使关节满足条件过于苛刻且尺度不可控。

`selected_level`: Level 2 — product_to_noncollapsing_joint。landing_bonus的多因子乘积导致塌缩和极值支配, 符合结构变换条件; Level 1尺度假说已隐含在iter 5→7的退化中(同结构, 不同实现, 均值从3.4→390), 仅调系数不足以解决乘积塌缩根因。

`selected_intervention`: 将landing_bonus从乘积聚合`15.0 * A * B * C * D`改为加权加性聚合`A * (w1*B + w2*C + w3*D)`, 保持new_contact转移门控但用加性加权和替代乘积, 同步设置与新值域匹配的系数(2.0, 0.5, 0.5)。不改变shaped_progress和angular_vel_penalty。

`falsifiable_hypothesis`: 加性聚合使任一质量因子的改善都能独立贡献奖励, 消除"三因子须同时完美"的塌缩瓶颈。landing_bonus不再需要单步极端值来补偿稀疏性, episode_sum_mean应从~390显著下降, magnitude_share下降至<50%, 同时active_rate应略升(因为更多中等质量接触步获得非零奖励), 外部score应回升并向best=120靠近。

`expected_next_round`: episode_sum_mean of landing_bonus降至~10-80区间, magnitude_share降至30-60%, shaped_progress的signed_share升至10-30%, 外部score回升至50+, 早期终止(<150步且<-50)减少到≤1/20。

`main_risk`: 加性聚合可能允许agent通过"接近目标但不直立"或"直立但远离目标"的局部最优获取部分bonus, 补偿效应可能导致着陆质量下降。需要观察下一轮upright和near_target的各自贡献, 如出现单因子刷分则下一轮可用软最小或门控修正。

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

    # 3. 着陆奖励：转移事件门控 × 加性质量因子（product→weighted sum，消除塌缩）
    contact_now = next_obs[6] * next_obs[7]
    contact_before = obs[6] * obs[7]
    new_contact = contact_now - contact_before
    if new_contact < 0.0:
        new_contact = 0.0

    near_target = 1.0 / (1.0 + 3.0 * abs(next_obs[0]))
    upright = 1.0 / (1.0 + 3.0 * angle_abs)
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    low_speed = 1.0 / (1.0 + speed)

    # 加性聚合：各质量因子独立贡献，避免乘积塌缩
    landing_bonus = new_contact * (2.0 * near_target + 0.5 * upright + 0.5 * low_speed)

    total_reward = shaped_progress + ang_vel_penalty + landing_bonus
    components = {
        "shaped_progress": shaped_progress,
        "angular_vel_penalty": ang_vel_penalty,
        "landing_bonus": landing_bonus
    }
    return (float(total_reward), components)
```
