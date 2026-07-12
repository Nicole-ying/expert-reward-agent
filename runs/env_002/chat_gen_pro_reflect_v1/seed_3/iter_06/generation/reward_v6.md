# 设计理由
**改了什么组件**：将原有的独立 `balance_penalty`（角度 hinge 二次惩罚 + 角速度二次惩罚）替换为 **姿态健康门控 + 小幅角速度惩罚** 的组件 `balance_modulation`。  
**为什么改**：当前平衡惩罚 per-step 仅约 -0.007，被主前进奖励（~3.0）完全淹没，无法有效防止摔倒。而 `terminated 20/20` 表明每次 eval 都以终止结束（可能是成功到达终点或摔倒），要进一步提高分数需要更强的摔倒前兆信号。  
**数学形式**：
- `forward = max(0, horizontal_vel)`  
- `base_gate = clamp01(1.0 - max(0, abs(hull_angle)-0.3) / 0.4)` （`abs_angle≤0.3`→1.0，`≥0.7`→0.0）
- `balance_gate = 0.3 + 0.7 * base_gate`（最低 0.3，避免梯度塌缩）
- `angular_vel_penalty = -0.05 * hull_angular_vel**2`
- `balance_modulation = (forward_reward + gait_efficiency) * (balance_gate - 1.0) + angular_vel_penalty`
**系数校准**：门限 0.3 rad 对应安全倾斜，0.7 rad 对应可能摔倒的前兆边缘（终止边界 0.8 rad 的 80%）。最低门控 0.3 保证即使严重倾斜仍然保留 30% 的前进梯度，避免冻结探索。角速度惩罚保持不变（原组件中已有）。

**为什么不用更大独立惩罚**：更大的二次惩罚可能过度抑制灵活的前倾探索，且不如门控直接与主目标联动敏感。乘性门控让平衡优先级自然高于盲目奔跑的奖励。

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

    # --- 3. 姿态健康门控 + 角速度惩罚（替代原 balance_penalty）---
    # 基础门控：abs(hull_angle) <= 0.3 为 1.0，>= 0.7 为 0.0
    angle_excess = max(0.0, abs(hull_angle) - 0.3)
    base_gate = max(0.0, min(1.0, 1.0 - angle_excess / 0.4))
    # 加下限 0.3，保证即使严重倾斜仍有微弱正向梯度
    balance_gate = 0.3 + 0.7 * base_gate
    angular_vel_penalty = -0.05 * (hull_angular_vel ** 2)
    # 总效果：当 hull 端正时相当于只受角速度惩罚；倾斜时压低了前进相关奖励
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
- **audit**: 当前 reward 缺少对摔倒的强前兆信号，平衡惩罚太弱被前进奖励淹没。  
- **behavior**: agent 已学会高速前进（score 186），但 eval 20 集全部以终止结束，可能仍有部分因摔倒导致低分，需要更稳定的姿态控制。  
- **signal**: 引入乘性姿态门控，在倾斜增大时压低前进奖励，强于原独立惩罚；同时保留角速度惩罚。  
- **level**: Level 2  
- **hypothesis**: 姿态门控会使 agent 在倾斜时被迫更关注恢复平衡，减少摔倒次数，从而整体得分提升并接近 300 分目标。  
- **risk**: 若门控下限 0.3 仍在大倾斜时给出非零奖励，agent 可能学会“容忍”大角度来换取速度，需观察实际摔倒率变化。