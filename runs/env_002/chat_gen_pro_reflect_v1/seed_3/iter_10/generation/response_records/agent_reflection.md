# Response Record

# 设计理由

**material different 的选择**：之前 iter 8 / iter 9 的修改都在现有 balance_gate + posture_penalty 的框架内调整系数/阈值或增减小惩罚，但得分连续下滑（293→249），预判连续 ❌。  
本轮进行两个实质性变更：  
1. **balance_gate 形状变换**：从 iter 8 的线性 hinge（0.4–0.5）改回更激进的平方映射，并把干预起点提前到 `|angle| > 0.3`，在接近临界危险区域时提供强梯度，有效避免摔倒。  
2. **新增 LIDAR 安全组件**：利用从未用过的 `obs[14..23]` LIDAR 距离传感器，对前方太近的障碍物施加温和惩罚，引导策略避开易致摔倒的紧贴地形。  
同时移除 iter 9 证明有害的 `posture_penalty`。

## 设计校准
- 主信号 per‑step ≈ 2769/842 ≈ 3.3。  
- 新增 `lidar_penalty` 只在前方任意测距 < 0.3 时触发，步均负担远低于主信号的 0.3 倍。  
- 动作惩罚、双脚离地惩罚等维持历史低负担。  
- `balance_gate` 采用平方映射 `gate = 1.0 - 0.9 * ((angle - 0.3)/(0.5 - 0.3)) ** 2`，在 `|angle| = 0.5` 时 gate = 0.1，提供安全网。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 信号提取
    horizontal_vel = obs[2]
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    leg1_contact = obs[8]
    leg2_contact = obs[13]
    # LIDAR 前方距离 (obs[14..23])
    min_lidar = min(obs[14], obs[15], obs[16], obs[17], obs[18],
                    obs[19], obs[20], obs[21], obs[22], obs[23])

    # --- 1. 主奖励：前进速度，线性+凸化 ---
    forward = max(0.0, horizontal_vel)
    forward_reward = 5.0 * forward + 2.0 * (forward ** 2)

    # --- 2. 步态效率：交替接触绑定前进速度 ---
    gait_quality = 1.0 - abs(leg1_contact - leg2_contact)
    gait_efficiency = 0.5 * gait_quality * forward

    # --- 3. 姿态健康门控（平方映射，阈值提前到0.3）---
    angle_abs = abs(hull_angle)
    if angle_abs < 0.3:
        balance_gate = 1.0
    elif angle_abs > 0.5:
        balance_gate = 0.1
    else:
        # 平方映射，从0.3开始急剧压低
        balance_gate = 1.0 - 0.9 * ((angle_abs - 0.3) / 0.2) ** 2

    # 角速度惩罚保持
    angular_vel_penalty = -0.05 * (hull_angular_vel ** 2)
    # 移除 posture_penalty
    balance_modulation = (forward_reward + gait_efficiency) * (balance_gate - 1.0) + angular_vel_penalty

    # --- 4. 双脚离地惩罚 ---
    contact_sum = leg1_contact + leg2_contact
    if contact_sum < 0.5:
        both_off_penalty = -0.1 * (1.0 - contact_sum)
    else:
        both_off_penalty = 0.0

    # --- 5. 动作能耗惩罚 ---
    action_penalty = -0.01 * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    # --- 6. 新增：LIDAR 近场惩罚（避免冲向障碍物）---
    if min_lidar < 0.3:
        lidar_penalty = -0.5 * (0.3 - min_lidar)
    else:
        lidar_penalty = 0.0

    total_reward = (forward_reward + gait_efficiency + balance_modulation +
                    both_off_penalty + action_penalty + lidar_penalty)

    components = {
        "forward_reward": forward_reward,
        "gait_efficiency": gait_efficiency,
        "balance_modulation": balance_modulation,
        "both_off_penalty": both_off_penalty,
        "action_penalty": action_penalty,
        "lidar_penalty": lidar_penalty,
        "gait_quality": gait_quality,
        "balance_gate": balance_gate,
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少对前方地形距离的感知信号，同时平衡门控在 0.4‑0.5 线性段梯度不足，近期加入的姿态二次惩罚被证实有害。
- **behavior**: agent 仍然因摔倒提前终止，iter 9 的速度‑平衡折衷导致得分降至 249。
- **signal**: 缺少 LIDAR 预警；balance_gate 形状过弱，无法在接近危险倾斜时给出强梯度。
- **level**: Level 2（结构变换：gate 形状升级为平方映射 + 新增 LIDAR 近场惩罚）
- **hypothesis**: 提前激活的平方门控更强制约倾斜，LIDAR 惩罚迫使策略远离会引发摔倒的近障碍物，两者合力延长生存距离，有望突破 300 分。
- **risk**: 平方门控可能压缩安全倾斜内的探索空间；LIDAR 惩罚若阈值偏高可能导致减速过多，但当前系数 0.5 和阈值 0.3 非常保守，副作用可控。
