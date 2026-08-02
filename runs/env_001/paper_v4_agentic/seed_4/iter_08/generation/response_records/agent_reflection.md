# Response Record

# 设计理由
连续 4 轮预判 ❌，且所有 episode 均以 truncation (1000步) 结束，没有一次成功着陆。当前奖励函数中 `progress_gated` 组件是一个基于“朝向目标速度”的绝对值正奖励：只要 agent 保持姿态并朝向目标，即可持续获得奖励。这导致 agent 学会了在目标附近徘徊获取持续收益，而不再追求触发软着陆终止。本次修改将 `progress_gated` 从**绝对值速度奖励**切换为**距离改善量奖励**（Level 2 结构变换），使 agent 只在**真正接近目标**时才获得正奖励，徘徊或远离时奖励为 0 或负。配合原有的 `landing` 组件（含 touchdown 一次性大奖励），有望引导 agent 尽快完成着陆以获取累计收益。

数学形式：`delta_dist = max(0, dist - next_dist) * gate`，gate 保持原有的姿态门控（角度、速度、角速度）。系数沿用 8.0（可接受范围内），保证每步奖励量级与原信号相近但仅在接近时产生。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------- unpack observations -------------------
    x,  y  = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle      = obs[4]
    angvel     = obs[5]
    left_leg   = obs[6]
    right_leg  = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle  = next_obs[4]
    n_angvel = next_obs[5]
    n_left   = next_obs[6]
    n_right  = next_obs[7]

    # ------------------- helper quantities -------------------
    dist      = (x**2  + y**2)  ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    next_vel_abs  = (nvx**2 + nvy**2) ** 0.5

    # ------------------- thresholds & weights -------------------
    w_progress = 8.0
    w_fuel = 0.2

    th_angle  = 0.5
    th_vel    = 1.0
    th_angvel = 2.0

    w_approach   = 1.0
    w_touchdown  = 10.0

    # ------------------- 1. distance-improvement progress signal -------------------
    delta_dist = dist - next_dist   # positive when approaching target

    gate_min = 0.1
    gate_angle  = max(gate_min, 1.0 - abs(n_angle)  / th_angle)
    gate_vel    = max(gate_min, 1.0 - next_vel_abs   / th_vel)
    gate_angvel = max(gate_min, 1.0 - abs(n_angvel)  / th_angvel)
    gate = gate_angle * gate_vel * gate_angvel

    progress_delta = w_progress * max(0.0, delta_dist) * gate

    # ------------------- 2. landing (approach + touchdown) -------------------
    contact_next = (n_left + n_right) / 2.0

    pos_factor    = max(0.0, 1.0 - abs(nx) / 0.5)
    height_factor = max(0.0, 1.0 - abs(ny) / 0.5)
    vel_factor    = max(0.0, 1.0 - next_vel_abs / 0.5)
    angle_factor  = max(0.0, 1.0 - abs(n_angle) / 0.3)
    angvel_factor = max(0.0, 1.0 - abs(n_angvel) / 0.5)

    touchdown_reward = 0.0
    approach_reward  = 0.0

    if contact_next > 0.1:
        quality = pos_factor * height_factor * vel_factor * angle_factor * angvel_factor * contact_next
        touchdown_reward = w_touchdown * quality
    else:
        dist_factor   = max(0.2, 1.0 - next_dist / 1.5)
        h_factor      = max(0.2, 1.0 - abs(ny) / 1.5)
        vel_factor_a  = max(0.2, 1.0 - next_vel_abs / 0.8)
        angle_factor_a = max(0.2, 1.0 - abs(n_angle) / 0.5)
        angvel_factor_a = max(0.2, 1.0 - abs(n_angvel) / 1.0)
        approach_reward = w_approach * dist_factor * h_factor * vel_factor_a * angle_factor_a * angvel_factor_a

    landing = approach_reward + touchdown_reward

    # ------------------- 3. fuel penalty -------------------
    engine_on = 1.0 if action != 0 else 0.0
    fuel_penalty = -w_fuel * engine_on

    # ------------------- total reward -------------------
    total_reward = progress_delta + landing + fuel_penalty

    components = {
        'progress_delta':   progress_delta,
        'landing':          landing,
        'fuel_penalty':     fuel_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测均已使用，terminated=0 且全部 truncated，agent 停在软着陆门槛外持续徘徊，非信号缺失而是 main progress reward 结构诱发了 exploit。
- **behavior**: agent 通过保持朝向目标的速度获取持续 progress 奖励，配合 approach 奖励徘徊在目标上方，从未触发稳定着陆（terminated=0）。
- **signal**: `progress_gated` 的绝对值朝向奖励提供了无限续航的正反馈，淹没了 touchdown 一次性奖励的相对价值。
- **level**: Level 2 — 结构变换（绝对值速度奖励 → 距离改善量奖励）。
- **hypothesis**: 改为仅奖励“实际缩短距离”的 delta 后，徘徊净收益归零，agent 会选择尽快着陆以获取 landing 中的 touchdown 大奖励，从而突破 truncated 循环。
- **risk**: 早期探索阶段 agent 可能无法连续接近目标导致总奖励偏低，训练初期可能更慢，但 `approach` 奖励（非接触时）仍提供引导。后续若 improvement 不足可微调系数。
