# 设计理由
上一轮的门控 `balance_gate` 使用平方映射，对任何倾斜都产生持续惩罚（因为 `balance_gate` 始终 < 1，从而 `balance_modulation` 为负）。即使倾斜很小（如 0.2 rad），也会令速度奖励打 0.96 折，相当于每步都有微弱的“减速税”。在分数已经达到 299.05、只差 0.95 突破 300 的背景下，这种持续惩罚可能让某些 episode 的前进速度偏慢，导致少数 episode 未能走完全部地形，拉低了平均分。

**修改组件**：`balance_gate`（基于 `hull_angle` 的门控因子）。  
**变换**：由全局二次惩罚改为 **hinge 门控**——角度低于 0.4 rad 时完全不触发惩罚（`gate = 1.0`），角度在 0.4 ~ 0.5 rad 之间线性下降至 0.1，超过 0.5 rad 时仍保持在 0.1（与原来一致）。这样在安全区域内前进速度不再被削弱，仅在接近临界倾斜时才快速压缩奖励，给予强梯度制止摔倒。

**系数校准**：  
- 安全阈 0.4 rad 约为假定摔倒边界（约 1 rad）的 40%，处于终止边界的 60‑80% 建议范围的低端，稍保守但安全。  
- `gate` 在 0.5 rad 处仍为 0.1，确保危险区域惩罚与上一轮一致。  
- 其余系数（`angular_vel_penalty`、`both_off_penalty`、`action_penalty`）保持不变，惩罚总负担远小于主信号。

**预期**：Agent 在大部分时间可以全速前进，不必因微小倾斜而减速，从而提升平均走行距离，使更多 episode 成功触达地形终点，将平均分推过 300。

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

    # --- 3. 姿态健康门控（改为 hinge 形式）+ 角速度惩罚 ---
    angle_excess = abs(hull_angle)
    # 安全区 (angle < 0.4 rad) 不惩罚；危险区 (0.4~0.5) 线性过渡；崩塌区 (>0.5) 保持低位
    if angle_excess < 0.4:
        balance_gate = 1.0
    elif angle_excess > 0.5:
        balance_gate = 0.1
    else:
        balance_gate = 1.0 - 0.9 * (angle_excess - 0.4) / 0.1

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
- **audit**: 尚无到达终点的显式信号，但通过速度激励可弥补；当前奖励职责基本齐全。  
- **behavior**: agent 已能稳定行走，仅少数 episode 可能未走完全程（平均分 299.05，接近满分 300）。  
- **signal**: 姿态门控对安全倾斜仍施加持续惩罚，可能抑制了平均速度，导致个别 episode 未能跑完地形。  
- **level**: Level 2（结构变换：将二次门控改为 hinge 门控）。  
- **hypothesis**: 移除安全区域的减速税后，agent 的平均前进速度将提高，使更多 episode 坚持到地形终点，突破 300 分。  
- **risk**: 门控变宽松可能导致少数 episode 摔倒风险略微上升，但基于角速度惩罚和双脚离地惩罚，预期仍能保持整体稳定。