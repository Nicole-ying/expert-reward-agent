# 设计理由
上一轮把 `soft_landing` 换成二值接触信号 `landing_success`，导致着陆奖励 active_rate 降为 0，所有 episode 被截断在 1000 步且无终止。因此把着陆组件从“接触二值×安全门”改成连续空间质量函数：在垫面附近（x,y 小）即便还没接触也能获得梯度，同时保留接触信号作为额外加成，从而让 agent 在接近着陆区域时有动力调整姿态和速度。数学形式上，用指数衰减的三维（位置、速度、姿态）相乘加上接触加成，系数沿用原 safety_gate 的速度/姿态衰减强度，并加入位置因子使激励集中在着陆区。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from next observation
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # A. Goal proximity shaping (unchanged)
    dist = (x**2 + y**2) ** 0.5
    proximity = 2.71828 ** (-dist)

    # B. Continuous landing quality (replaces landing_success)
    # Combines spatial proximity, safe posture, and contact bonus
    speed_sq = vx**2 + vy**2
    angle_sq = angle**2
    spatial_factor = 2.71828 ** (-2.0 * x**2) * 2.71828 ** (-0.5 * abs(y))
    safety_factor = 2.71828 ** (-10.0 * angle_sq) * 2.71828 ** (-5.0 * speed_sq)
    contact_bonus = 1.0 + left_contact + right_contact
    landing_quality = spatial_factor * safety_factor * contact_bonus

    # C. Energy efficiency (unchanged)
    energy_penalty = -0.01 if action != 0 else 0.0

    # D. Terminal velocity penalty (unchanged)
    vel_penalty = 0.0
    if y < 0.05 and abs(x) < 0.1:
        if vy < -0.3:
            vel_penalty = -0.5 * max(0.0, -vy - 0.3)

    total_reward = 1.0 * proximity + 1.0 * landing_quality + energy_penalty + vel_penalty

    components = {
        "proximity": proximity,
        "landing_quality": landing_quality,
        "energy_penalty": energy_penalty,
        "terminal_velocity_penalty": vel_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: 上一轮 `landing_success` 均值 0.0，所有 episode 1000 步截断无终止，当前得分 -17.09 远低于历史最佳 115.51。
- **behavior**: agent 长时间徘徊在上空，完全不尝试完成着陆，没有成功 settle 也没有 crash。
- **signal**: 着陆信号完全缺失梯度——只有成功接触时才给出奖励，而 agent 从未获得接触。
- **level**: Level 2（稀疏→连续 proxy），因为 active_rate 接近 0，需要结构变换而非调系数。
- **hypothesis**: 新的 `landing_quality` 在接近着陆区域时即提供连续梯度（位置、速度、姿态），激励 agent 精细调整并最终触发接触，从而恢复着陆能力。
- **risk**: 可能鼓励在垫面附近低速悬停而不急于接触，但接触加成使双足触地的分值明显高于单纯悬停，不至于固化高悬停行为。