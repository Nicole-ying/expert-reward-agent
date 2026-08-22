# Response Record

# 设计理由
审计发现：当前奖励函数完全未使用 LIDAR 测距信号（obs[14:24]），而任务描述明确指出机器人必须利用前方 10 个 LIDAR 预判崎岖地形以动态调整步态。该信号缺口直接导致 agent 在阶梯、树桩、坑洞等突变地形上没有预警信息，最终摔倒终止。现有组件 active_rate 均 >83%，无僵尸组件；连续三轮预判 ❌ 表明仅靠步态激励+姿态门控在此地形上已经走到尽头。因此本轮引入一个地形粗糙度惩罚（基于 LIDAR std），在崎岖地形上按速度比例惩罚，迫使 agent 在危险地形减速调整步态，平滑地形上仍可全速前进。惩罚尺度已按主信号 0.3× 校准（系数 0.02，预计 per-step 约 -0.02，远低于主信号 per-step 0.258）。此修改为核心信号缺口补齐，而非修补现有逻辑。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # =========================================================================
    # 1. 提取观测信号
    # =========================================================================
    hull_angle = obs[0]
    hull_ang_vel = obs[1]
    horizontal_speed = obs[2]
    leg_1_contact = obs[12]
    leg_2_contact = obs[13]
    # LIDAR 地形预览（显式索引，避免切片）
    lidar_0 = obs[14]
    lidar_1 = obs[15]
    lidar_2 = obs[16]
    lidar_3 = obs[17]
    lidar_4 = obs[18]
    lidar_5 = obs[19]
    lidar_6 = obs[20]
    lidar_7 = obs[21]
    lidar_8 = obs[22]
    lidar_9 = obs[23]

    next_hull_angle = next_obs[0]
    next_hull_ang_vel = next_obs[1]
    next_horizontal_speed = next_obs[2]
    next_leg_1_contact = next_obs[12]
    next_leg_2_contact = next_obs[13]

    # =========================================================================
    # 2. 前向速度奖励 (主学习信号)
    # =========================================================================
    forward_speed = max(0.0, horizontal_speed)
    forward_reward = 1.0 * forward_speed

    # =========================================================================
    # 3. 姿态稳定门
    # =========================================================================
    tilt_safe_bound = 0.3
    tilt_danger_bound = 0.7
    tilt_margin = tilt_danger_bound - tilt_safe_bound

    abs_tilt = abs(hull_angle)
    if abs_tilt <= tilt_safe_bound:
        tilt_gate = 1.0
    elif abs_tilt >= tilt_danger_bound:
        tilt_gate = 0.0
    else:
        tilt_gate = 1.0 - (abs_tilt - tilt_safe_bound) / tilt_margin

    ang_vel_thresh = 2.0
    ang_vel_margin = 4.0
    abs_ang_vel = abs(hull_ang_vel)
    if abs_ang_vel <= ang_vel_thresh:
        ang_vel_factor = 1.0
    elif abs_ang_vel >= ang_vel_thresh + ang_vel_margin:
        ang_vel_factor = 0.3
    else:
        ang_vel_factor = 1.0 - 0.7 * (abs_ang_vel - ang_vel_thresh) / ang_vel_margin

    stability_gate = tilt_gate * ang_vel_factor

    # =========================================================================
    # 4. 接触切换奖励
    # =========================================================================
    leg1_change = abs(next_leg_1_contact - leg_1_contact)
    leg2_change = abs(next_leg_2_contact - leg_2_contact)
    contact_transition_reward = 0.05 * (leg1_change + leg2_change)

    # =========================================================================
    # 5. 能量效率惩罚
    # =========================================================================
    action_sq_sum = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = 0.005 * action_sq_sum

    # =========================================================================
    # 6. 地形粗糙度惩罚 (新增 — 利用 LIDAR 预判崎岖，减速避险)
    # =========================================================================
    lidar_mean = (lidar_0 + lidar_1 + lidar_2 + lidar_3 + lidar_4 +
                  lidar_5 + lidar_6 + lidar_7 + lidar_8 + lidar_9) / 10.0
    lidar_var = (
        (lidar_0 - lidar_mean) ** 2 +
        (lidar_1 - lidar_mean) ** 2 +
        (lidar_2 - lidar_mean) ** 2 +
        (lidar_3 - lidar_mean) ** 2 +
        (lidar_4 - lidar_mean) ** 2 +
        (lidar_5 - lidar_mean) ** 2 +
        (lidar_6 - lidar_mean) ** 2 +
        (lidar_7 - lidar_mean) ** 2 +
        (lidar_8 - lidar_mean) ** 2 +
        (lidar_9 - lidar_mean) ** 2
    ) / 10.0
    roughness = lidar_var ** 0.5
    roughness_penalty = 0.02 * roughness * abs(horizontal_speed)

    # =========================================================================
    # 7. 组合并返回
    # =========================================================================
    gated_forward = forward_reward * stability_gate
    total_reward = gated_forward + contact_transition_reward - energy_penalty - roughness_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_gate": stability_gate,
        "gated_forward": gated_forward,
        "contact_transition_reward": contact_transition_reward,
        "energy_penalty": -energy_penalty,
        "roughness_penalty": -roughness_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 观测扫描发现 LIDAR 信号缺失（obs[14:24] 从未使用），而任务明确要求利用 LIDAR 预判崎岖地形；该缺口极可能是 floor_fallen 终止率高的主因。
- **behavior**: agent 在平坦或缓坡上保持姿态稳定、步态交替，但遇到台阶/树桩等突变地形时因无预警而突然摔倒，导致 episode 提前终止。
- **signal**: 缺少地形危险预警信号，导致 agent 在崎岖区域仍维持高速前进，摔倒后无负反馈。
- **level**: Level 2（信号缺口 → 新增组件）
- **hypothesis**: 新增基于 LIDAR std 的地形粗糙度速度乘积惩罚后，agent 将学会在崎岖地形减速以降低惩罚，从而赢得更多调整步态时间，减少无预判摔倒，延长存活距离并提升总 forward reward。
- **risk**: 若 roughness 阈值不匹配实际地形尺度，可能过度惩罚平坦区域或对极端崎岖仍不足以减速；但系数已降至主信号的 0.08x，预期副作用为轻度减速，不会冻结探索。
