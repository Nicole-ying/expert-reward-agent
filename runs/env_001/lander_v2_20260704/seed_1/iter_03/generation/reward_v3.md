`evidence`：episode长度从68骤增至752.80，approach_landing_reward以93.6%占比和100% active_rate支配总奖励（episode_sum_mean=6315），但外部得分仅从-119.56微升至-98.89，说明agent学会了在着陆垫附近滞留收集稠密代理奖励，而非高效完成着陆。

`behavior_diagnosis`：agent已学会接近着陆垫（距离成本从-1.111改善至-0.521），但continuous状态奖励使其滞留平台——近距、低速、好姿态和持续接触每条产生≈8.4的收益，752步累积数千奖励后缓慢settle。这是典型的dense proxy exploitation，代理目标（收集approach奖励）与外部任务（快速稳定着陆）发生错位。

`signal_completeness`：引导信号（proximity/velocity/angle/contact）齐全且可达，但缺失两项关键职责：①着陆完成信号（接触后无额外激励去快速settle）；②引擎效率约束（未惩罚推力使用）。当前稠密形式使"在目标附近逗留"优于"快速完成着陆"。

`selected_level`：Level 2 — `dense_to_task_event`。稠密代理已教会agent到达着陆垫（ep长度从68→752，不再crash），但现在主导奖励并阻止高效完成。证据模式完全匹配："dense proxy forming medium score plateau"和"proxy提高但外部任务不升"。根据知识库指引：应reduce/difference/localize该dense proxy。

`selected_intervention`：将approach_landing_reward从**持续性状态值**（state-value：`e^(-dist²)`、`e^(-vel²)`、contact连续值）重构为**改善量**（improvement-based：`max(0, prev - curr)`形式），并将velocity improvement以proximity gate约束（仅近端减速获奖励）。四个子项统一改为有界差分，消除"持续占优状态即可无限获利"的漏洞，总奖励不再随episode长度增长。

`falsifiable_hypothesis`：改善量形式使approach_landing_reward成为有界信号（每episode上限≈60），agent无法通过滞留获利，被迫尽早完成着陆以停止距离/稳定性惩罚累积。预期episode_length显著下降（从752→200以内），terminated率保持或改善，外部得分应突破-50。

`expected_next_round`：approach_landing_reward的episode_sum_mean应大幅下降（从6315→约40-80），active_rate保持高位（improvement几乎每步都可能发生），distance_cost的signed_share上升（因为总奖励缩小），episode_length降至300以下，外部score改善。

`main_risk`：改善量信号比稠密状态值更稀疏，可能削弱已学会的reachability，导致agent重新出现早期快速失败（退化为iter 1的~68步crash模式）。proximity_gate对velocity_improvement的约束也可能在边界处产生不连续性。需在下一轮密切监控早期终止数量。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取当前和下一步状态
    px, py = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    omega = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 距离与稳定性成本（保持不变）
    w_dist = 1.0
    w_vel = 0.1
    w_angle = 0.1
    w_angvel = 0.05

    distance_cost = -w_dist * (nx ** 2 + ny ** 2)
    stability_cost = -(
        w_vel * (vx ** 2 + vy ** 2)
        + w_angle * abs(angle)
        + w_angvel * abs(omega)
    )

    # 改善量型 approach_landing_reward：有界差分替代持续状态值
    prev_dist = (px ** 2 + py ** 2) ** 0.5
    curr_dist = (nx ** 2 + ny ** 2) ** 0.5
    prev_vel = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    curr_vel = (vx ** 2 + vy ** 2) ** 0.5
    prev_contact = (obs[6] + obs[7]) / 2.0
    curr_contact = (left_contact + right_contact) / 2.0

    # 接近改善：向目标靠近即获奖
    proximity_improvement = 5.0 * max(0.0, prev_dist - curr_dist)

    # 减速改善：仅在靠近着陆垫时奖励减速（避免远距离时惩罚必要的高速接近）
    proximity_gate = max(0.0, 1.0 - curr_dist / 1.5)
    velocity_improvement = 10.0 * proximity_gate * max(0.0, prev_vel - curr_vel)

    # 姿态改善：减小倾角即获奖
    angle_improvement = 3.0 * max(0.0, abs(obs[4]) - abs(angle))

    # 触地改善：建立支撑腿接触的转移奖励（非持续占有）
    contact_improvement = 30.0 * max(0.0, curr_contact - prev_contact)

    approach_landing_reward = proximity_improvement + velocity_improvement + angle_improvement + contact_improvement

    total_reward = distance_cost + stability_cost + approach_landing_reward

    components = {
        'distance_cost': distance_cost,
        'stability_cost': stability_cost,
        'approach_landing_reward': approach_landing_reward
    }

    return float(total_reward), components
```