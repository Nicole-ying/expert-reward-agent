# 设计理由
我将 `velocity_penalty` 从“全速度二次惩罚 + 距离门控”改为“超速阈值惩罚”（hinge）。反馈显示 `velocity_penalty` 绝对值（-34.07）远超 `progress_reward`（0.61），且 `contact_bonus=0`，说明速度惩罚压制了所有移动动机，导致 agent 不敢使用引擎，从未满足着陆条件。改为只惩罚速度超过 1.0 的部分，系数 0.1，使 per-step 惩罚量与进步奖励（约 0.02~0.04）保持同量级，消除对低速探索的抑制，同时保留对高速碰撞的约束。

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

    # 接触奖励：双接触 + 低速 + 竖直
    contact_bonus = 10.0 * lc * rc * max(0.0, 1.0 - speed) * max(0.0, 1.0 - abs(angle))

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
- **evidence**: `velocity_penalty`=-34.07，`progress_reward`=0.61，`contact_bonus`=0.0，terminated=15/20，ep_len=668
- **behavior**: agent 为避免速度惩罚而极少使用引擎，无法朝目标移动，长期漂移后以失败终止，从无着陆接触
- **signal**: 速度惩罚过强且覆盖所有速度，压倒了唯一的正向进步信号，导致学习坍缩
- **level**: Level 2（全时二次惩罚 → hinge 惩罚），对应证据模式“惩罚 active_rate≈100% 但 terminated 率仍高”
- **hypothesis**: 消除对低速的惩罚后，agent 愿意使用引擎朝目标移动，`progress_reward` 将能有效引导，逐步靠近平台并触发 `contact_bonus`
- **risk**: 接近地面时若速度仍超过 1.0 会受到惩罚，但不会坍塌；hinge 完全放开 0~1 的速度可能导致 agent 维持在 0.8~1.0 巡航而不主动减速到零，后续可能需要针对性强化减速信号