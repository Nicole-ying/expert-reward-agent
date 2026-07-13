# Response Record

# 设计理由
考虑到当前主信号（joint）已在大多数 episode 成功着陆并稳定停靠，但 engine 使用率仍高达约 76%，且历史最佳得分对应更短的 episode 长度，说明效率仍有提升空间。为提高接近速度并间接减少引擎浪费，我将 progress 信号从“速度对齐”（连续状态值）改为“每步距离减少量”（改善量），使 agent 在向目标靠近的每一步都直接获得正向梯度，而不再仅依靠瞬时速度分量。保留系数 0.2 以维持与现有 stability 大致匹配的量级，joint 仍为两者的几何平均，迫使 agent 在快速靠近的同时保持姿态稳定。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # current state
    cx = obs[0]
    cy = obs[1]
    # next state
    nx = next_obs[0]
    ny = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # distance to target from current and next positions
    curr_dist = (cx**2 + cy**2)**0.5 + 1e-12
    next_dist = (nx**2 + ny**2)**0.5 + 1e-12
    # progress: reward for reducing distance to target
    progress = 0.2 * max(0.0, curr_dist - next_dist)

    # stability: favor low speed, low angle, low angular velocity
    speed_sq = vx**2 + vy**2
    stability = 1.0 / (1.0 + 0.1 * speed_sq + 0.04 * (ang_vel**2) + 1.0 * (angle**2))

    # contact reward
    contact = 0.5 * (left_contact + right_contact) + 0.2 * left_contact * right_contact * stability

    # joint progress/stability
    joint = (stability * progress + 1e-12) ** 0.5

    # engine penalty
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
- **evidence**: terminated=15/20, len=513.6, engine active rate 约 76%，progress per-step 仅 0.024
- **behavior**: agent 已能稳定着陆，但 episode 偏长且引擎使用频繁，部分 episode 被截断
- **signal**: progress 依赖瞬时速度对齐，在减速阶段信号微弱，无法有效推动快速接近
- **level**: Level 2 – 数学结构变换（状态值→改善量）
- **hypothesis**: 改为每步距离减少量后，agent 在朝向目标移动的每一步都获得梯度，会更快接近并更早触发 stable settlement，从而缩短 episode、减少引擎罚分累积
- **risk**: 若仅靠距离减少量而缺乏速度约束，可能诱使 agent 高速冲向目标导致 crash；但 joint 仍与 stability 绑定，可抑制过度冒险
