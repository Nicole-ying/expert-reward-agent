# 设计理由
当前 agent 快速坠毁（episode 平均长度仅 68.4，全部 terminated），说明奖励缺少对垂直速度的约束。现有的 `progress`（向目标移动）虽然为正但无法阻止高速撞击，而 `landing_event` 过于稀疏无法提供中途引导。本轮新增一个 **soft_landing 连续奖励**，在飞行器接近垫面（ny < 0.3）且垂直速度较小时给予正向激励，引导它在最后阶段减速软着陆。选用高度‑速度乘积形式，系数 0.2 控制在每步最大 0.2，约为当前 `progress` 典型步均值的 10 倍但限于局部数步，不会淹没主信号。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前状态与下一状态
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    prev_left, prev_right = obs[6], obs[7]
    next_left, next_right = next_obs[6], next_obs[7]

    # 1. Proximity: 向目标移动的距离减少量
    dist_curr = (x ** 2 + y ** 2) ** 0.5
    dist_next = (nx ** 2 + ny ** 2) ** 0.5
    progress = max(0.0, dist_curr - dist_next)

    # 2. Landing event: 一次性双足着陆事件奖励
    landing_event = 0.0
    prev_both = (prev_left >= 0.5 and prev_right >= 0.5)
    next_both = (next_left >= 0.5 and next_right >= 0.5)
    if not prev_both and next_both:
        horiz_factor = 1.0
        if abs(nx) > 0.2:
            horiz_factor = max(0.0, 1.0 - (abs(nx) - 0.2) / 0.3)

        speed = (nvx ** 2 + nvy ** 2) ** 0.5
        speed_factor = 1.0
        if speed > 0.3:
            speed_factor = max(0.0, 1.0 - (speed - 0.3) / 0.4)

        angle_factor = 1.0
        if abs(nangle) > 0.2:
            angle_factor = max(0.0, 1.0 - (abs(nangle) - 0.2) / 0.3)

        quality = horiz_factor * speed_factor * angle_factor
        landing_event = 10.0 * quality

    # 3. Energy penalty: 惩罚不必要的引擎使用
    energy_penalty = -0.01 if action != 0 else 0.0

    # 4. Soft landing: 接近垫面时鼓励低垂直速度
    soft_landing = 0.0
    landing_altitude = 0.3
    max_landing_speed = 0.5
    if ny < landing_altitude:
        height_factor = max(0.0, 1.0 - ny / landing_altitude)
        speed_factor = max(0.0, 1.0 - abs(nvy) / max_landing_speed)
        soft_landing = 0.2 * height_factor * speed_factor

    total_reward = progress + landing_event + energy_penalty + soft_landing

    components = {
        "proximity": progress,
        "landing_event": landing_event,
        "energy_penalty": energy_penalty,
        "soft_landing": soft_landing
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: terminated=20/20, len=68.4, score=-123.4; proximity 仅 1.13，landing_event 0.62，说明快速坠毁且无减速信号。
- **behavior**: agent 一味向下冲向目标垫，无姿态或速度控制，最终高速撞地或飞出视口。
- **signal**: 缺少终端减速引导，progress 鼓励靠近但不约束下坠速度。
- **level**: Level 2（数学结构变换 —— 新增连续 soft landing 信号作为局部门控）。
- **hypothesis**: 在接近垫面时给予由速度和高度构成的乘积奖励，能形成梯度引导飞行器主动减速，避免硬着陆。
- **risk**: 若阈值设得偏高，agent 可能在较高处就开始盘旋刷分而不完成着陆；当前将高度上限设为 0.3、奖励系数 0.2，应能控制此风险。