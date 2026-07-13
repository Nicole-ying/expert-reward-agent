# Response Record

# 设计理由
本轮将 `stability * progress` 换为几何平均 `(stability * progress) ** 0.5`。乘积在远端（distance 大、progress 小）时塌缩到很低的数值（例如 stability=0.9, progress=0.1 → 0.09），导致大量时间步的奖励信号过弱，难以驱动 agent 快速接近目标。几何平均在两侧都不为零时提供明显更大的梯度（同例得 0.3），同时保持近端最优值仍为 1，不改变最优行为的方向。contact 部分保持不变，整体系数无需重新校准。

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
    progress = 1.0 / (1.0 + distance)

    speed_sq = vx**2 + vy**2
    stability = 1.0 / (1.0 + 0.1 * speed_sq + 0.04 * (ang_vel**2) + 1.0 * (angle**2))

    contact = 0.5 * (left_contact + right_contact)

    # 几何平均替代乘积，避免中远距离奖励塌缩
    joint = (stability * progress + 1e-12) ** 0.5
    total = joint + contact

    components = {
        "progress": progress,
        "stability": stability,
        "contact": contact,
    }
    return float(total), components
```

# 诊断摘要
- **evidence**: terminated=50%, episode_length=803.9, score_mean=46.22（远低于目标 200）
- **behavior**: agent 能长时间存活（很少快速坠毁），但接近目标的驱动力不足，常徘徊或缓慢移动，最终被截断或少数成功
- **signal**: stability 与 progress 的乘积在距离较远时塌缩至极小值，造成“信号平坦”区域，早期梯度不足以有效引导 agent 向目标加速
- **level**: Level 2（乘积→几何平均）
- **hypothesis**: 用几何平均提升 mid‑field 奖励强度，使 agent 在非完美稳定状态下也能获得有意义的接近信号，从而更快驶向目标垫并最终得分上升
- **risk**: 稳定性要求暂时放宽后，部分 episode 可能因过快接近导致碰撞终止（terminated 率略微增加），但整体得分预期提升，后续可用门控进一步加固着陆阶段
