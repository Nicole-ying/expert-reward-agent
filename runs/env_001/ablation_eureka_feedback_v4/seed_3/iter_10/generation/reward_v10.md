# 设计理由
修改 `soft_landing` 组件：从乘积形式的正向奖励改为线性差值信号，当垂直速度低于阈值时给正奖励、高于阈值时给负惩罚。原乘积形式对极端低速和超高速的梯度不均，且无法主动抑制危险高速。新形式是连续有符号信号，既鼓励低速又惩罚超速，且仅在高度低于 0.3 时触发，不干扰高空探索。系数 0.2 使最大单步贡献约 0.1，量级适中，不会覆盖全程的 progress 信号，但与着陆阶段的行为直接对齐。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前与下一状态
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

    # 2. Landing event: 一次性双足着陆事件奖励（保持不变）
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

    # 4. Soft landing: 接近垫面时鼓励低垂直速度，惩罚超速
    soft_landing = 0.0
    landing_altitude = 0.3
    max_landing_speed = 0.5
    if ny < landing_altitude:
        height_factor = max(0.0, 1.0 - ny / landing_altitude)
        speed_diff = max_landing_speed - abs(nvy)
        soft_landing = 0.2 * height_factor * speed_diff

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
- **evidence**: terminated 仅 8/20，truncated 12/20（平均长度 787），landing_event 高触发频次（疑似被反复着陆滥用），得分分布在 11.5~251.4 之间。
- **behavior**: agent 多数 episode 徘徊至时间截断而无法稳定终止，部分 episode 成功但整体效率不高，可能存在反复弹跳着陆以刷取事件奖励的行为。
- **signal**: 靠近地面时的速度引导偏弱（纯正向乘积），缺少对危险高速的惩罚，导致着陆阶段不稳定。
- **level**: Level 2（数学结构变换），将 soft_landing 从乘积正向奖励改为带惩罚的线性差值信号。
- **hypothesis**: 基于垂直速度的对称线性信号能更清晰地区分安全低速与危险高速，同时保留接近地面的引导力，从而帮助 agent 在着陆时稳定、减少截断，并抑制反复着陆动机。
- **risk**: 惩罚端可能使部分早期帧产生微小负值，若 agent 在未学会控制速度前接近地面，可能短期抑制下降动机，但高度门控和系数温和可缓解。