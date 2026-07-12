# Response Record

# 设计理由
本次诊断基于 **信号覆盖审计**：代码在安全区 (`hull_angle<0.4`) 内没有任何姿态代价，导致 iter 8 移除“安全区减速税”后，agent 在平地上过于松弛，生存长度从 917 下降到 863，总得分未能突破 300。修复方法是在 `balance_modulation` 中加入一个对 **`hull_angle` 的微弱二次惩罚**。这会在安全区提供连续的直立引导，促使策略保持更稳定的竖直姿态，但又不会像 iter 7 那样过度惩罚导致减速。修改点很小，属于添加一种连续的微小代价（Level 2 结构变换）。系数经过校准，每步惩罚约 0.002 ~ 0.02，远小于主信号 per‑step（~3.5），不会干扰前进激励。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取信号
    horizontal_vel = obs[2]
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    leg1_contact = obs[8]
    leg2_contact = obs[13]

    # --- 1. 主奖励：前进速度，线性+凸化 ---
    forward = max(0.0, horizontal_vel)
    forward_reward = 5.0 * forward + 2.0 * (forward ** 2)

    # --- 2. 步态效率：与前进速度绑定的交替奖励 ---
    gait_quality = 1.0 - abs(leg1_contact - leg2_contact)
    gait_efficiency = 0.5 * gait_quality * forward

    # --- 3. 姿态健康门控（hinge 形式）+ 角速度惩罚 + 微弱姿态代价 ---
    angle_excess = abs(hull_angle)
    if angle_excess < 0.4:
        balance_gate = 1.0
    elif angle_excess > 0.5:
        balance_gate = 0.1
    else:
        balance_gate = 1.0 - 0.9 * (angle_excess - 0.4) / 0.1

    angular_vel_penalty = -0.05 * (hull_angular_vel ** 2)
    posture_penalty = -0.2 * (hull_angle ** 2)          # 新加的微弱直立代价

    balance_modulation = (forward_reward + gait_efficiency) * (balance_gate - 1.0) + angular_vel_penalty + posture_penalty

    # --- 4. 双脚离地惩罚 ---
    contact_sum = leg1_contact + leg2_contact
    if contact_sum < 0.5:
        both_off_penalty = -0.1 * (1.0 - contact_sum)
    else:
        both_off_penalty = 0.0

    # --- 5. 动作能耗惩罚 ---
    action_penalty = -0.01 * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    total_reward = forward_reward + gait_efficiency + balance_modulation + both_off_penalty + action_penalty

    components = {
        "forward_reward": forward_reward,
        "gait_efficiency": gait_efficiency,
        "balance_modulation": balance_modulation,
        "both_off_penalty": both_off_penalty,
        "action_penalty": action_penalty,
        "gait_quality": gait_quality,
        "balance_gate": balance_gate,
        "posture_penalty": posture_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少安全区的直立引导，导致移除减速税后稳定性变差。
- **behavior**: 安全区内姿态过于松弛，偶尔摔倒，生存长度下降。
- **signal**: 需要一种持续但微弱的直立牵引力。
- **level**: Level 2
- **hypothesis**: 添加对 `hull_angle` 的二次微小惩罚能在不压制前进的前提下，略微提升竖直稳定性，从而延长 episode 并突破 300 分。
- **risk**: 若系数偏大可能过度抑制前倾，降低速度。但所选系数 (0.2) 极小，预期风险可控。
