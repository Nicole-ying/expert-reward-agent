# 设计理由
我将 progress 组件从“距离缩小量”改为“速度朝向目标的对齐量”。当前 agent 已能成功着陆，但 progress 信号极弱，每步平均仅 0.014，几乎被 contact 主导。替换为径向速度对准后，agent 在飞向目标的过程中每一步都可获得连续梯度，尤其在早期阶段便学会对准目标方向，从而加快接近速度、缩短 episode 时间。使用 `alignment = max(0, vx*ux + vy*uy)` 并保留 `joint = (stability * alignment_factor * 0.2 + 1e-12)**0.5`，保持与稳定性耦合的几何平均结构，确保 progress 不会单独膨胀。系数 0.2 经校准可使 joint 的每步贡献与 contact 保持可比范围（约 0.3），不会破坏已收敛的接触行为。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # next state
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # distance to target (landing pad)
    dist = (x**2 + y**2)**0.5 + 1e-6
    # unit vector pointing toward target
    ux = -x / dist
    uy = -y / dist

    # velocity alignment with target direction (positive when moving closer)
    alignment = vx * ux + vy * uy
    progress = 0.2 * max(0.0, alignment)

    # stability: favor low speed, low angle, low angular velocity
    speed_sq = vx**2 + vy**2
    stability = 1.0 / (1.0 + 0.1 * speed_sq + 0.04 * (ang_vel**2) + 1.0 * (angle**2))

    # contact reward: base contact + bonus for both legs contacting while stable
    contact = 0.5 * (left_contact + right_contact) + 0.2 * left_contact * right_contact * stability

    # joint progress/stability signal
    joint = (stability * progress + 1e-12) ** 0.5

    # small engine penalty
    engine_penalty = -0.03 * (1.0 if action != 0 else 0.0)

    total = joint + contact + engine_penalty

    components = {
        "progress": progress,
        "stability": stability,
        "contact": contact,
        "engine_penalty": engine_penalty,
    }
    return float(total), components
```

# 诊断摘要
- **evidence**: terminated=19/20, score=239.5, progress/ep mean=0.014, contact/ep mean=0.637
- **behavior**: agent 稳定着陆，但 progress 信号几乎不起作用，主要靠接触奖励引导
- **signal**: 缺少从早期远距离开始向目标移动的密集梯度
- **level**: Level 2（数学结构变换）——将稀疏的“距离缩小量”替换为连续的“速度对准”信号
- **hypothesis**: 更强的方向性驱动会让 agent 更早对准目标，缩短到达时间，同时保持稳定着陆
- **risk**: 若 agent 过早高速朝向目标，可能导致 overshoot 后反复修正；但 stability 耦合会抑制高速，风险可控