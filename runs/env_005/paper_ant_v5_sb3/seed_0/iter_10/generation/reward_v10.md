# 设计理由
当前 agent 能保持直立并以较高速度前进，但所有 episode 最终因身体高度低于 0.2 而失败终止。原 height_pen 阈值设在 0.3，active_rate=0% 表明高度从未低于 0.3，说明跌倒过程在 0.3 以下发生得太快，该惩罚没有提供任何梯度。因此将阈值提高到 0.4（终止边界 0.2 的 2 倍，落在 60‑80% 的危险区间），让 agent 在高度从 0.4 下降时就收到渐进警告信号，从而学会维持健康高度。惩罚系数保持 1.0，每步预期惩罚（在 0.35 左右约 -0.05）控制在前进奖励典型值的 3‑5%，不会过度压制前进行为。

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
    height_threshold = 0.4  # 从 0.3 提高到 0.4，提前预警跌倒
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
- **evidence**: terminated 20/20，episode_mean_length 357.5，height_pen active_rate 0% 但所有 episode 在高度 <0.2 时失败终止。
- **behavior**: agent 学会了快速向前 + 直立，但高度逐渐下降直至跌倒，过程中从未触发高度惩罚。
- **signal**: 高度约束信号完全缺失——阈值 0.3 过低，无法在跌倒过程中提供梯度。
- **level**: Level 1（尺度修复），仅调整 height_pen 阈值。
- **hypothesis**: 将阈值提升到 0.4 后，agent 在高度下降早期就会受到连续惩罚，从而学会维持 0.4 以上的安全高度，减少跌倒，延长存活并提高总分。
- **risk**: 如果惩罚过强，agent 可能过分保守、不敢前进；但系数 1.0、阈值 0.4 产生的单步惩罚很小，最可能的副作用是前进速度略微下降，但存活优势应覆盖。