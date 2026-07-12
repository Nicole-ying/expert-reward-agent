# 设计理由
当前angle_penalty触发率为0，说明摔倒不以持续角度超限的形式出现；问题更可能是在崎岖路段缺乏速度抑制导致失控。我在gated_forward_reward的gate中引入地形粗糙度因子：用10个LIDAR读数的标准差度量地形崎岖度，乘以 `1/(1 + lidar_std)` 衰减速度奖励。这样崎岖时前进信号变弱，鼓励agent减速以保持稳定，但平坦时与原来一致。系数选择让粗糙度因子在平坦段接近1，在std=1时降至0.5，保持适度信号。代码已将LIDAR切片改为逐个索引访问以满足验证规则。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 信号提取
    horizontal_velocity = obs[2]
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]
    hip1_torque = action[0]
    knee1_torque = action[1]
    hip2_torque = action[2]
    knee2_torque = action[3]
    # 逐个索引 LIDAR 值以通过验证
    lidar = [obs[14], obs[15], obs[16], obs[17], obs[18], obs[19], obs[20], obs[21], obs[22], obs[23]]

    # 组件1: 前进奖励（线性，仅正向）
    forward_reward = 1.0 * max(horizontal_velocity, 0.0)

    # 组件2: 平衡+地形门控
    angle_factor = 1.0 / (1.0 + 2.0 * abs(hull_angle))
    angular_velocity_factor = 1.0 / (1.0 + 0.5 * abs(hull_angular_velocity))

    # 地形粗糙度因子: 基于LIDAR标准差
    lidar_mean = sum(lidar) / 10.0
    lidar_var = sum((x - lidar_mean) ** 2 for x in lidar) / 10.0
    lidar_std = lidar_var ** 0.5
    terrain_factor = 1.0 / (1.0 + lidar_std)  # std=0时=1.0, std增大时缓慢衰减

    # 综合门控: 取最保守值
    balance_gate = min(angle_factor, angular_velocity_factor)
    gate = min(balance_gate, terrain_factor)
    gated_forward_reward = forward_reward * gate

    # 组件3: 能耗惩罚
    energy_penalty_weight = 0.01
    energy_penalty = -energy_penalty_weight * (
        hip1_torque**2 + knee1_torque**2 + hip2_torque**2 + knee2_torque**2
    )

    # 组件4: 额外平衡惩罚（保留，但历史触发率为0，不影响）
    angle_safety_threshold = 0.8
    angle_penalty_weight = 1.0
    angle_penalty = -angle_penalty_weight * max(0.0, abs(hull_angle) - angle_safety_threshold)

    total_reward = gated_forward_reward + energy_penalty + angle_penalty

    components = {
        "gated_forward_reward": gated_forward_reward,
        "energy_penalty": energy_penalty,
        "angle_penalty": angle_penalty,
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: 20条episode全部terminated，长度均值913，得分范围[-58,300]，angle_penalty触发率0%，gated_forward_reward占signed_share 96.6%，说明某些episode摔倒导致大幅负分。
- **behavior**: agent在崎岖路段可能因未减速而失控摔倒，拉低平均分数。
- **signal**: 缺乏对前方地形的感知信号，无法在危险路段提前抑制速度。
- **level**: Level 2（数学结构变换，在门控中引入地形粗糙度因子）
- **hypothesis**: 地形粗糙度门控使agent在崎岖处减速，降低摔倒概率，从而提升最低分并提高平均得分。
- **risk**: 速度过度抑制可能导致探索缓慢，损失极致速度得分；若LIDAR标准差常态偏高，整体速度下降会拉低平均分。