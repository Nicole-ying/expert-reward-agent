# 设计理由
agent 在目标垫附近通过悬停积累了高额 soft_landing 和 proximity 密集奖励，导致许多 episode 被截断而未成功着陆，外部 score 停滞在 115。我将 soft_landing 从全时状态值改为仅当支撑腿接触垫面时才激活的局部强奖励 `landing_success`，并用指数衰减姿态安全 gate 防止高速碰撞奖励。系数选为 1.0，使最大 per‑step 奖励约 2.0，不超过 proximity 的两倍，保持接近引导不变。这应迫使 agent 冒险下降以获取新奖励，提高着陆成功率。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from next observation
    x = next_obs[0]          # x position relative to target pad center
    y = next_obs[1]          # y position relative to target pad height
    vx = next_obs[2]         # horizontal velocity
    vy = next_obs[3]         # vertical velocity (negative = downward)
    angle = next_obs[4]      # body angle
    angvel = next_obs[5]     # angular velocity (unused)
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # A. Goal proximity shaping (unchanged)
    dist = (x**2 + y**2) ** 0.5
    proximity = 2.71828 ** (-dist)   # [0,1], peak at (0,0)

    # B. Landing success reward (replaces soft_landing)
    # Only active when legs are in contact with the pad, gated by safe posture
    speed_sq = vx**2 + vy**2
    angle_sq = angle**2
    safety_gate = 2.71828 ** (-10.0 * angle_sq) * 2.71828 ** (-5.0 * speed_sq)
    landing_success = 1.0 * (left_contact + right_contact) * safety_gate

    # C. Energy efficiency (unchanged)
    energy_penalty = -0.01 if action != 0 else 0.0

    # D. Terminal velocity penalty (unchanged)
    vel_penalty = 0.0
    if y < 0.05 and abs(x) < 0.1:
        if vy < -0.3:
            vel_penalty = -0.5 * max(0.0, -vy - 0.3)

    total_reward = 1.0 * proximity + 1.0 * landing_success + energy_penalty + vel_penalty

    components = {
        "proximity": proximity,
        "landing_success": landing_success,
        "energy_penalty": energy_penalty,
        "terminal_velocity_penalty": vel_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: soft_landing=544.8, proximity=507.2, terminal_velocity_penalty=0.0, terminated=60%, score=115.5 (target 200)
- **behavior**: agent 在垫面上方悬停积累密集奖励，8/20 episode 达到最大步数截断，未成功着陆
- **signal**: 缺乏针对着陆成功的强奖励，持续悬停即可获得高额回报，抑制冒险下降
- **level**: Level 2 – 将 soft_landing（状态值）改为 landing_success（接触时才激活的局部强信号）
- **hypothesis**: 悬停无法获得新奖励，agent 被迫下降以触发接触奖励，从而提高着陆成功率和外部 score
- **risk**: 如果接触探索困难，初期可能丢失密集奖励导致学习变慢，但 proximity 仍提供接近引导