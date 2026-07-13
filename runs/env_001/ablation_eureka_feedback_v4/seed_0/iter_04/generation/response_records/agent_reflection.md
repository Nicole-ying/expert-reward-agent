# Response Record

# 设计理由
上一轮将 stability 作为乘积因子乘到负距离惩罚 `proximity` 上，导致不稳定状态（低 stability）减小了负奖励，反而奖励 agent 保持高速和倾斜，最终全部 episode 以失败终止（score -582.65）。本轮只修改 proximity 组件：将其从绝对值负奖励 `-distance` 改为有界正向进度信号 `1/(1+distance)`，保留乘积结构 `stability * progress`。由于 progress 始终为正，低 stability 会减小正向奖励，agent 只有保持稳定并靠近目标才能获得更高回报，从而消除错误激励。系数无须额外校准，progress 最大 1.0，乘积量级与接触奖励匹配。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    distance = (x**2 + y**2) ** 0.5
    progress = 1.0 / (1.0 + distance)  # 正向接近奖励

    speed_sq = vx**2 + vy**2
    stability = 1.0 / (1.0 + 0.1 * speed_sq + 0.04 * (ang_vel**2) + 1.0 * (angle**2))

    contact = 0.5 * (left_contact + right_contact)

    total = stability * progress + contact

    components = {
        "progress": progress,
        "stability": stability,
        "contact": contact
    }
    return float(total), components
```

# 诊断摘要
- **evidence**: score=-582.65, terminated=20/20, proximity 均值 -65.89, stability 均值 28.46（因子累计后偏高），乘积使不稳定状态奖励更大。
- **behavior**: agent 保持高速或倾斜以减少距离惩罚，导致全部 episode 以失败终止。
- **signal**: proximity 为负值且存在稳定性乘法门，造成“不稳定→惩罚减小”的错误梯度方向。
- **level**: Level 2，将 proximity 从负距离值转为有界正向进度信号，保持乘法结构但逆转语义。
- **hypothesis**: progress 变为正后，低 stability 会削弱正向奖励，迫使 agent 同时追求接近目标与姿态稳定，恢复正确梯度。
- **risk**: 进度奖励 bounded 可能使早期信号较弱，但随接近目标逐步增强，配合接触奖励应足以引导学习。
