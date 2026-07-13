# 设计理由
当前agent虽然能频繁接触目标垫并获得高稳定性，但progress极低（episode sum 7），表明接近速度过慢；同时大量episode因时间截断终止，暗示着陆后settle效率不足。历史修改已调整过progress和stability结构，但未对燃料使用施加任何代价。考虑到环境明确要求“减少引擎使用”，我新增一个小的引擎惩罚（action≠0时-0.03），在不破坏现有接近与接触信号的前提下，推动策略减少不必要的点火、加快稳定并降低能耗。系数根据主信号 joint 每步约0.09、contact约0.49 校准，预期每步惩罚约0.01，远小于主奖励且不会支配学习。

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

    # small engine penalty to discourage unnecessary thrust
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
- **evidence**: terminated 6/20, truncated 14/20, progress ep sum=7.05（极低）, stability ep sum=830.7, contact=408.0, score=145.
- **behavior**: agent慢速接近垫、频繁使用引擎，大部分episode在时间截断时仍处于接触但未settle状态。
- **signal**: 缺少对引擎使用的代价，导致policy可能浪费燃料并延长稳定时间。
- **level**: Level 2（新增引擎惩罚组件），因历史已多次调整现有组件边界，此次引入新信号以对齐外部燃料效率需求。
- **hypothesis**: 轻微惩罚非零动作将减少不必要的点火，迫使策略更快稳定并贴垫，从而提升外部得分。
- **risk**: 惩罚可能过度抑制必要的姿态调整，造成着陆失败率上升；但系数极小（-0.03），预期安全。