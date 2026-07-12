# Response Record

# 设计理由
**移除**僵尸组件 `height_penalty`（height_penalty 触发率 0%，从未提供梯度），**替换为** `descend_penalty`：利用 `body_z_velocity`（观测索引 15）作为向前跌落的前兆信号。  
当身体垂直下降速度超过安全阈值（‑0.5 m/s）时施加线性 hinge 惩罚，鼓励 agent 减缓坠落、延长存活，从而使 `forward_reward` 有足够的展开时间。  
系数校准：主信号 `forward_reward` 每步均约 0.87，新惩罚每步期望 < 0.2 × 0.87 ≈ 0.17；阈值设在快速坠落但未崩溃的幅度，且无惩罚区间保证正常站立/行走不被误伤。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]
    vx = next_obs[13]
    vy = next_obs[14]
    vz = next_obs[15]    # body_z_velocity

    # 姿态健康因子
    pose_error = quat_x**2 + quat_y**2 + quat_z**2
    posture_factor = 2.718281828 ** (-10.0 * pose_error)

    # 高度健康门控：从危险区 (0.25m) 到安全区 (0.55m) 线性提升
    height_gate = 0.0
    if body_z >= 0.55:
        height_gate = 1.0
    elif body_z <= 0.25:
        height_gate = 0.0
    else:
        height_gate = (body_z - 0.25) / 0.3

    # 前进奖励（基础量 × 姿态门控 × 高度门控）
    base_forward = max(0.0, vx)
    forward_reward = 1.0 * base_forward * posture_factor * height_gate

    # 侧向速度惩罚
    lateral_penalty = -0.5 * (vy ** 2)

    # 动作幅度惩罚
    action_penalty = -0.1 * sum(a ** 2 for a in action)

    # 下降速度惩罚：阻止过快坠落（防止瞬间跌倒）
    descend_penalty = -0.2 * max(0.0, -0.5 - vz)

    total_reward = forward_reward + lateral_penalty + action_penalty + descend_penalty

    components = {
        "forward_reward": forward_reward,
        "lateral_penalty": lateral_penalty,
        "action_penalty": action_penalty,
        "descend_penalty": descend_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺失高度下降前兆信号，导致 agent 快速跌倒，现有高度惩罚从未触发。
- **evidence**: height_penalty active_rate 0%；len=13.75，全部 terminated；forward_reward 仅 59.3%。
- **behavior**: agent 在极短时间（~14 步）内摔倒导致终止，无法持续前进。
- **signal**: 需要 body_z_velocity 的向下过快惩罚作为“即将摔倒”的连续警告。
- **level**: Level 2 — 将有源僵尸组件替换为连续前兆信号。
- **hypothesis**: 下降速度惩罚会学习减缓身体坠落，延长 episode，使前进奖励累积，提升得分。
- **risk**: 若阈值 ‑0.5 设置过低会约束正常步态下降（如迈步小幅下沉），但当前系数温和，仅为引导作用；若后续反馈中 descend_penalty 占比过高可下调系数至 0.1。
