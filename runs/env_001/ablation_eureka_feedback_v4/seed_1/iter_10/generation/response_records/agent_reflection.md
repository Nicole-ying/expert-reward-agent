# Response Record

# 设计理由
将稀疏的 contact_bonus 从仅双腿同时接触（lc * rc）的强条件改为接触腿数（lc + rc）的连续化方案，系数由 10.0 降至 2.0。这使 agent 在只有一条腿接触时也能获得部分奖励，提供中间梯度，引导从接近平台到稳定双接触的逐步改善；同时降低最大单步奖励，避免成功时过度累积导致刷分，且不影响其他组件。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    av = next_obs[5]
    lc = next_obs[6]
    rc = next_obs[7]

    dist = (x**2 + y**2) ** 0.5
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    speed = (vx**2 + vy**2) ** 0.5

    # 进步奖励：向目标靠近
    progress = prev_dist - dist
    progress_reward = 2.0 * progress

    # 速度惩罚：只惩罚超出安全阈值的速度，避免压制正常机动
    velocity_penalty = -0.1 * max(0.0, speed - 1.0) ** 2

    # 姿态惩罚
    orientation_penalty = -0.1 * (angle**2) - 0.05 * (av**2)

    # 接触奖励：接触腿数 × 低速 × 竖直，提供从单腿接触到双腿的连续梯度
    contact_bonus = 2.0 * (lc + rc) * max(0.0, 1.0 - speed) * max(0.0, 1.0 - abs(angle))

    total = progress_reward + velocity_penalty + orientation_penalty + contact_bonus

    components = {
        'progress_reward': progress_reward,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'contact_bonus': contact_bonus
    }

    return float(total), components
```

# 诊断摘要
- **evidence**: terminated=60%，score 均值 -95，范围 [-204,192]，contact_bonus 在少数成功 episode 极高但多数失败 episode 为 0，两极化严重。
- **behavior**: agent 能在部分 episode 达到着陆奖励高峰，但整体成功率低，多数 episode 远离目标或未能完成稳定着陆，导致严重负分。
- **signal**: 着陆奖励只对双腿同时接触有效，缺少从接近到单腿接触的中间梯度，使学习信号稀疏且不稳定。
- **level**: Level 2（将稀疏二值条件改为连续接触腿数 proxy，数学结构变换，同时降低系数）。
- **hypothesis**: 提供单腿接触的部分奖励能让 agent 在接近平台时更早获得正反馈，从而提升着陆成功率，减少两极分化。
- **risk**: 单腿接触的奖励可能让 agent 满足于仅一条腿接触而不再追求完全着陆，但因双腿接触奖励是单腿的两倍，仍会向双接触优化；若出现平台边缘摩擦保持，可能被困。
