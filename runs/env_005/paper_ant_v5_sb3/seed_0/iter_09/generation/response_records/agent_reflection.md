# Response Record

# 设计理由
当前 agent 主要因身体倾斜导致高度降低而频繁失败（17/20 episodes terminated），但现有 `height_pen` 只在高度已低于 0.3 时才发出微弱警告，无法从根源上抑制倾倒。我添加一个直立惩罚组件 `upright_pen`，采用 hinge 形态：当 `body_up_z < 0.8` 时开始施加线性惩罚，系数 0.5。这样在身体开始倾斜时即给予梯度信号，促使保持直立，从而防止跌倒。系数校准：前进奖励每步典型值约 1.0，直立惩罚在 `body_up_z=0.5` 时每步约 0.15，控制在 0.1~0.3 范围，不会压制前进信号。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_x_vel = next_obs[13]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    body_z = next_obs[0]

    # 直立因子（body_up_z ∈ [-1, 1]，1 为全直立）
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_factor = max(0.0, body_up_z)

    # 前向奖励（仅当身体朝上时）
    forward = 1.0 * body_x_vel * upright_factor

    # 身体高度铰链惩罚：在高度接近 0.2 前发出警告
    height_threshold = 0.3
    w_height_pen = 1.0
    height_pen = -w_height_pen * max(0.0, height_threshold - body_z)

    # 直立惩罚：当身体不够直立时施加，防止因倾斜导致跌倒
    w_upright_pen = 0.5
    upright_threshold = 0.8
    upright_pen = -w_upright_pen * max(0.0, upright_threshold - body_up_z)

    total_reward = forward + height_pen + upright_pen

    components = {
        "forward": forward,
        "height_pen": height_pen,
        "upright_pen": upright_pen
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: 17/20 episodes terminated，score=-228，forward 每集平均 65.9 但总奖励均值极负，height_pen 触发率 42.7% 但惩罚极小（-3.28）。
- **behavior**: agent 因身体倾斜导致高度快速跌破 0.2 而频繁跌倒，少数存活 episode 能获得正向前进奖励。
- **signal**: 缺少对倾斜的早期惩罚，现有高度惩罚只在即将失败时才微弱预警，无法抑制倾倒。
- **level**: Level 2——添加缺失的直立约束组件（hinge 惩罚）。
- **hypothesis**: 通过在倾斜初期施加适度惩罚，驱动策略保持更好的直立姿态，从而减少高度崩塌，提高存活率和前进效率。
- **risk**: 过度惩罚可能使策略偏向保守姿态、牺牲前进速度；若出现此情况后续可降低惩罚系数或改用 gate 形态。
