# 设计理由
我将 `soft_landing` 从依赖二值 contact 的乘积式改为连续几何平均式着陆进度奖励。原公式在 `contact_factor=0` 时整体为 0，active_rate 持续为 0，切断了着陆引导的梯度。新形式使用高度因子（`exp(-y/0.3)`）、速度因子（`exp(-2*speed_sq)`）和角度因子（`exp(-10*angle²)`）的几何平均，确保即使没有接触也每步有非零信号，且随着接近垫面、减速、摆正而递增。系数设为 1.0，与当前主信号 `proximity` 的 per‑step 典型值（~0.7）可比，且不超过主信号的 2 倍，避免喧宾夺主。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from next observation
    x = next_obs[0]          # x position relative to target pad center
    y = next_obs[1]          # y position relative to target pad height
    vx = next_obs[2]         # horizontal velocity
    vy = next_obs[3]         # vertical velocity (negative = downward)
    angle = next_obs[4]      # body angle
    angvel = next_obs[5]     # angular velocity
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # A. Goal proximity shaping (mandatory)
    dist = (x**2 + y**2) ** 0.5
    proximity = 2.71828 ** (-dist)   # bounded positive: [0,1], peak at (0,0)

    # B. Soft landing progress (continuous proxy, replaces product-based)
    # Height factor: encourages descending toward pad
    h_factor = 2.71828 ** (-y / 0.3)        # ~1 when y≈0, decays as y grows
    # Velocity factor: encourages low speed
    speed_sq = vx**2 + vy**2
    v_factor = 2.71828 ** (-2.0 * speed_sq) # ~1 when speed≈0, drops quickly
    # Angle factor: encourages upright posture
    angle_factor = 2.71828 ** (-10.0 * angle**2)
    # Geometric mean avoids collapse when any single factor is still developing
    soft_landing = (h_factor * v_factor * angle_factor) ** (1.0 / 3.0)

    # C. Energy efficiency (mandatory)
    energy_penalty = -0.01 if action != 0 else 0.0

    # D. Terminal velocity penalty (conditional, only near the pad)
    vel_penalty = 0.0
    if y < 0.05 and abs(x) < 0.1:
        # penalize excessively large downward speed
        if vy < -0.3:
            vel_penalty = -0.5 * max(0.0, -vy - 0.3)

    total_reward = 1.0 * proximity + 1.0 * soft_landing + energy_penalty + vel_penalty

    components = {
        "proximity": proximity,
        "soft_landing": soft_landing,
        "energy_penalty": energy_penalty,
        "terminal_velocity_penalty": vel_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: terminated=0/20, truncated=20/20, soft_landing per‑episode mean=0, active_rate≈0
- **behavior**: agent 在目标垫上方徘徊，不进行着陆，所有回合均被截断
- **signal**: 着陆引导缺失 – 原依赖 contact 的乘积当 contact 为 0 时整体坍塌，无法提供梯度
- **level**: Level 2，乘积塌缩导致的数学形态变换
- **hypothesis**: 用连续几何平均着陆进度代替乘积，使 agent 在靠近垫子、减速、摆正时获得递增奖励，从而学会着陆并最终触发接触
- **risk**: 可能仅奖励减速和平稳悬停而不主动完成最终接触（后续可引入接触专项奖励）