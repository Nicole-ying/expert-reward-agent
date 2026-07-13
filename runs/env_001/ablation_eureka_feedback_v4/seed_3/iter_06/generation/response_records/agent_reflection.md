# Response Record

# 设计理由
我选择将 soft_landing 组件从状态值形式改为改善量形式（landing_improvement），奖励姿态和速度的逐步减小，避免 agent 依靠良好状态持续获得奖励而徘徊。同时保持 proximity 作为主要进展信号不变。Landing_improvement 使用 `max(0, angle² - nangle²) + max(0, speed² - nspeed²)` 的数学形式，系数设为 0.5，使每步的预期奖励低于主进展信号（progress）的 2 倍。Terminal velocity penalty 的阈值放宽至 `ny<0.1` 以更可靠地惩罚硬着陆风险。本次修改为 Level 2 结构变换（状态值→改善量），目的是缩短着陆时间、提升外部任务得分。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前状态
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    left, right = obs[6], obs[7]

    # 下一状态
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    nleft, nright = next_obs[6], next_obs[7]

    # 1. Proximity: 向目标移动的距离减少量
    dist_curr = (x ** 2 + y ** 2) ** 0.5
    dist_next = (nx ** 2 + ny ** 2) ** 0.5
    progress = max(0.0, dist_curr - dist_next)

    # 2. Landing improvement: 奖励姿态和速度的改善（平方减少）
    angle_sq = angle ** 2
    nangle_sq = nangle ** 2
    speed_sq = vx ** 2 + vy ** 2
    nspeed_sq = nvx ** 2 + nvy ** 2

    angle_improvement = max(0.0, angle_sq - nangle_sq)
    speed_improvement = max(0.0, speed_sq - nspeed_sq)
    landing_improvement = angle_improvement + speed_improvement

    # 3. Energy penalty
    energy_penalty = -0.01 if action != 0 else 0.0

    # 4. Terminal velocity penalty: 近地时向下速度过大
    vel_penalty = 0.0
    if ny < 0.1 and nvy < -0.3:
        vel_penalty = -1.0 * abs(nvy + 0.3)

    total_reward = progress + landing_improvement * 0.5 + energy_penalty + vel_penalty

    components = {
        "proximity": progress,
        "landing_improvement": landing_improvement * 0.5,
        "energy_penalty": energy_penalty,
        "terminal_velocity_penalty": vel_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: score=-113.59, episode_len=105.25, terminated=20/20, 上一轮的 progress_gate_reward 均值仅 0.904，信号塌缩导致快速失败。
- **behavior**: agent 在约 105 步就因 crash/越界终止，外部得分极低，内部奖励近乎为零，没有有效学习信号。
- **signal**: 主奖励信号过弱（被 gate 严重衰减），且 soft_landing 的状态值形式在 best 轮中可能导致中分段徘徊（len=725.7），无法进一步逼近满分。
- **level**: Level 2，状态值→改善量变换，解决“占据好状态持续获奖”导致的徘徊问题。
- **hypothesis**: 将 soft_landing 改为改善量可以推动 agent 主动降低姿态和速度，缩短到达稳定停靠的时间，从而提升外部任务得分。
- **risk**: 改善量奖励的尺度可能仍偏小，若 agent 早期无姿态改善则组件贡献不足，后续或需微调系数或开启新的 progress 凸化。
