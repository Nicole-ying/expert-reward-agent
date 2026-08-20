# Response Record

1. `evidence`: score=-24.46 but some episodes reach +234；terminated=16/20，len=396；potential_diff mean=0.88，angle_penalty mean=-4.16；现有骨架第一次刷新best但远未达target，表明接近目标过程中的姿态惩罚主导负分，且成功着陆的episode未能被奖励明确识别。
2. `behavior_diagnosis`: 策略多数episode因姿态不稳或着陆失败导致负分，少数成功episode得分很高但无完成激励，整体平均负，表明agent缺少对“精确软着陆并接触”这一完成状态的定向信号。
3. `signal_completeness`: 现有组件（势差+姿态罚）提供了过程引导，但**缺失任务完成事件**——双腿接触同时速角小、位置近零的成功着陆没有明确奖励，导致代理目标（减距离、保姿态）与最终任务完成未对齐。
4. `selected_level`: Level 2，触发条件为缺失必要职责（完成信号），且当前过程组件已能引导agent到达目标附近，需要将代理目标与任务完成对齐（proxy_to_completion_alignment）。
5. `selected_intervention`: 新增一个稀疏完成奖励组件`success_bonus` —— 当检测到双脚接触且姿态、位置、速度均满足软着陆条件时给予固定正奖励；其他所有组件保持不变。
6. `falsifiable_hypothesis`: 加入完成奖励后，成功episode会受到额外正反馈，而失败episode保持现状，平均得分应显著提升，且len应保持较短（成功终止更快），同时score_range的上限可能进一步升高而负分episode占比下降。
7. `expected_next_round`: 下一轮score应明显向正移动并超过0，`success_bonus`的active_rate应为低（仅成功时触发），平均success_bonus贡献呈正且显著，angle_penalty的负分影响相对下降，len可能因成功终止而继续缩短。
8. `main_risk`: 成功条件定义过严或过松，若过严则bonus永远为零无法学习，过松则agent可能过早触发假成功而停止优化，需根据下一轮反馈验证阈值合理性。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -- Extract state variables --
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle, angular_v = obs[4], obs[5]

    x_n, y_n = next_obs[0], next_obs[1]
    vx_n, vy_n = next_obs[2], next_obs[3]
    angle_n, angular_v_n = next_obs[4], next_obs[5]
    left_contact_n = next_obs[6]
    right_contact_n = next_obs[7]

    # -- Component 1: Potential-based progress + stability --
    w_dist = 1.0
    w_vel  = 0.3
    dist_current = (x**2 + y**2) ** 0.5
    dist_next    = (x_n**2 + y_n**2) ** 0.5
    vel_current  = abs(vx) + abs(vy)
    vel_next     = abs(vx_n) + abs(vy_n)

    potential_current = w_dist * dist_current + w_vel * vel_current
    potential_next    = w_dist * dist_next    + w_vel * vel_next

    potential_diff = potential_current - potential_next

    # -- Component 2: Attitude penalty --
    k_angle = 0.5
    k_angvel = 0.1
    angle_penalty = -k_angle * (angle_n**2) - k_angvel * (angular_v_n**2)

    # -- Component 3: Sparse completion bonus --
    # Success condition: both support legs in contact, near zero velocity,
    # upright angle, and close to target center.
    success = False
    if left_contact_n == 1.0 and right_contact_n == 1.0:
        if abs(x_n) < 0.2 and abs(y_n) < 0.3:
            if abs(vx_n) < 0.2 and abs(vy_n) < 0.2:
                if abs(angle_n) < 0.2:
                    success = True
    success_bonus = 200.0 if success else 0.0

    # -- Total reward --
    total_reward = potential_diff + angle_penalty + success_bonus

    components = {
        "potential_diff": float(potential_diff),
        "angle_penalty": float(angle_penalty),
        "success_bonus": float(success_bonus)
    }

    return float(total_reward), components
```
