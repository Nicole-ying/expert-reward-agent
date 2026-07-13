# 设计理由
观察到episode全部截断，没有触发成功终止，说明agent虽然学会接近目标并部分接触（contact均值约0.29），但未能稳定地两腿同时触垫并使身体完全静止。当前contact组件仅考虑单腿接触总和，无法区分单腿或双腿接触，对“完全稳定着陆”的引导不足。因此，将contact组件改造为复合信号：保留基础接触奖励（避免勘探倒退），同时加入左腿×右腿×稳定性乘积项，对双腿同时触垫且姿态/速度极低的状态给予额外奖励。系数0.2使bonus的per-step最大值约为0.2，小于历史contact贡献的2倍以避免过快支配总奖励，但仍能形成朝向完全着陆的强梯度。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # current position
    x0 = obs[0]
    y0 = obs[1]
    # next position
    x1 = next_obs[0]
    y1 = next_obs[1]

    dist0 = (x0**2 + y0**2) ** 0.5
    dist1 = (x1**2 + y1**2) ** 0.5
    delta = dist0 - dist1  # >0 when approaching pad
    progress = 5.0 * max(0.0, delta)

    # next state
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    speed_sq = vx**2 + vy**2
    stability = 1.0 / (1.0 + 0.1 * speed_sq + 0.04 * (ang_vel**2) + 1.0 * (angle**2))

    # contact reward: base contact + bonus for both legs contacting while stable
    contact = 0.5 * (left_contact + right_contact) + 0.2 * left_contact * right_contact * stability

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
- **evidence**: terminated=0/20, truncated=20/20, len=1000, contact mean≈0.29，progress mean≈0.0076，stability mean≈0.994.
- **behavior**: 智能体学会靠近目标且姿态稳定，但双腿接触不频繁，无法触发成功终止，全部episode跑满步数截断。
- **signal**: 缺乏对“双腿同时触垫且完全静止”状态的强化，现有contact信号对单腿接触与双腿接触等同，不能驱动agent完成最终收束。
- **level**: Level 2（数学结构变换：在contact组件中引入双腿接触×稳定性的乘积项，形成联合条件奖励）。
- **hypothesis**: 加入双腿×稳定性bonus后，agent为了获得额外奖励会更主动地实现双腿着垫并迅速消除残余运动，从而更早触发物理休眠，提升总得分。
- **risk**: 若双腿稳定着陆难以探索，bonus可能极少触发，但基础接触部分保留，不会破坏已有接近行为；bonus比例较小（最大0.2），不致引发显著的奖励偏移。