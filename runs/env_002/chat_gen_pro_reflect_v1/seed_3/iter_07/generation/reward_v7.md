# 设计理由

**修改组件：** 姿态健康门控 (`balance_gate` 的计算方式)。  
**问题：** 当前线性衰减门控（在角度 0.3~0.7 rad 之间从 1.0 线性降到 0.3）对即将摔倒的大角度抑制作用不足：即使在 0.7 rad（接近很多双足机器人摔倒阈值），门控仍有 0.3，前进奖励仍然有较大比例保留，导致 agent 缺乏足够强的梯度来紧急恢复平衡。最终所有评估 episode 都以摔倒终止，虽然行走长度较长（~748 步），但若能避免摔倒，将可能走完全程并达到目标分数 300。  
**变换：** 将角度门控改为平方映射，容忍角设为 0.5 rad，超过 0.5 rad 后门控迅速降到 0.1 的下限。这样在角度接近危险区域时给予急剧衰减的奖励，迫使策略更积极地维持竖直姿态。  
**数学形式：**  
- 若 `|hull_angle| ≥ 0.5`，`base_gate = 0.0`；  
- 否则 `base_gate = 1.0 - (|hull_angle| / 0.5)²`。  
- 最终 `balance_gate = 0.1 + 0.9 * base_gate`（下限 0.1 保证 gate 永不塌缩到 0，满足“不理想但安全区域 gate≥0.3”的要求：在 0.3 rad 时 gate≈0.676）。  
**系数校准：** 没有引入新的独立惩罚，仅强化了原有的前进奖励门控。平方映射使 per-step 惩罚上限可控，总惩罚负担仍远低于主奖励的 0.5×，不会导致长度剧烈崩坏。

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

    # --- 3. 姿态健康门控 + 角速度惩罚（修改：更陡峭的角度门控）---
    # 使用平方映射，容忍角 0.5 rad，在超过 0.5 时 gate 降至接近 0.1
    angle_excess = abs(hull_angle)
    if angle_excess >= 0.5:
        base_gate = 0.0
    else:
        base_gate = 1.0 - (angle_excess / 0.5) ** 2
    # 硬下限 0.1 保证梯度不彻底消失
    balance_gate = 0.1 + 0.9 * base_gate
    angular_vel_penalty = -0.05 * (hull_angular_vel ** 2)
    balance_modulation = (forward_reward + gait_efficiency) * (balance_gate - 1.0) + angular_vel_penalty

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
        "balance_gate": balance_gate
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少对即将摔倒的大角度（>0.5 rad）的强抑制信号；当前线性门控过于温和。
- **behavior**: agent 行走较长距离（~748 步）但最终必然摔倒，无法完成地形。
- **signal**: 需要更陡峭的角度-奖励衰减，迫使 agent 在倾斜增大时紧急恢复平衡。
- **level**: Level 2
- **hypothesis**: 平方映射门控会在接近临界角度时急剧压低奖励，给予策略强梯度来避免摔倒，从而延长生存、可能走完全程并突破 300 分。
- **risk**: 若门控过强，可能抑制行走探索，导致步数略微缩短或速度降低，但下限 0.1 可防止奖励完全归零。