# 设计理由
将 `stability` 从独立的二次惩罚重构为一个取值在 (0,1] 的 gate 因子，乘到 `proximity` 上。原稳定性惩罚对速度/角速度/角度施加固定负分，但没有与距离目标的行为耦合。新形态使 agent 在向目标靠近时，必须同时保持低速和小姿态才能获得完整的接近奖励，从而迫使软着陆。gate 采用 `1/(1+...)` 的软饱和形式，保证在"速度较快但仍可纠正"的区域（v≈2）gate 不低于 0.6，避免学习信号塌缩。

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
    proximity = -1.0 * distance

    # stability 重构为 gate 因子：值域 (0,1]，低速度小角度时接近 1
    speed_sq = vx**2 + vy**2
    stability = 1.0 / (1.0 + 0.1 * speed_sq + 0.04 * (ang_vel**2) + 1.0 * (angle**2))

    contact = 0.5 * (left_contact + right_contact)

    total = stability * proximity + contact

    components = {
        "proximity": proximity,
        "stability": stability,
        "contact": contact
    }
    return float(total), components
```

# 诊断摘要
- **evidence**: terminated 20/20，len 69.75，proximity=-66.87，stability=-9.48，contact=1.85，score=-73.53
- **behavior**: agent 试图靠近中心但未能有效减速，以高速或大角度撞地/飞出，全部 episode 失败
- **signal**: 独立的稳定性惩罚没有与接近行为耦合，接近信号（proximity）未受速度姿态约束
- **level**: Level 2——将独立惩罚转为 gate，实现 soft health gate
- **hypothesis**: gate 迫使 agent 在向目标移动时控制速度和角度以获得完整的 proximity 回报，从而学习软着陆
- **risk**: gate 可能轻微减慢初始靠近进度，但衰减区间保守设计保证了学习信号不会中断