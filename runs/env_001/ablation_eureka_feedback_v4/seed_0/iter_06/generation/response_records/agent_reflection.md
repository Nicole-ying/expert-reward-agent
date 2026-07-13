# Response Record

# 设计理由
当前 agent 存活满 1000 步但从未触地（contact=0），说明它学会了在目标垫附近保持高 progress 和 stability 以获取持续奖励，却不完成着陆。为此将 progress 组件从依赖绝对距离的状态值改为每步距离减少量的改善量：`progress = 5.0 * max(0, dist_current - dist_next)`。这样悬停不再获得奖励，只有向垫移动才能获得正向信号，迫使 agent 持续接近并最终触发接触。系数 5.0 将典型步长改善量（0.02~0.1）映射到 0.1~0.5，与 stability 的数值范围匹配，避免任何单组件过强。

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

    contact = 0.5 * (left_contact + right_contact)

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
- **evidence**: episode_len=1000，truncated=20/20，contact=0，progress/stability 都很高，外部 score=-14
