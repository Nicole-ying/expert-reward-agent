# Response Record

# 设计理由
**修改组件**：将 `fuel_penalty`（常数引擎惩罚）替换为 `brake_reward`（局部门控的主引擎减速奖励）。  
**数学形式**：`brake_reward = w_brake * max(0, descend_threshold - nvy)`，仅在 `action == 2` 且 `nvy < descend_threshold`（即下降速度超过阈值）时发放，否则为 0。  
**系数校准**：`w_brake = 0.5`，`descend_threshold = -0.3`，使得在典型下坠（`nvy ≈ -1.0`）时每步奖励约 0.35，预期总贡献与 `progress_delta` 量级相当（~4‑6），不会支配总奖励。  
**为什么这样改**：累积记录显示，自从引入 `fuel_penalty`（iter 3）后 episode 长度从 372 暴跌至 68 且始终未恢复，agent 在训练中学会了一个“糟糕的捷径”——永远选择 `action=0` 以避免任何燃料惩罚，即使这必然导致坠毁。常数惩罚没有区分“必要的引擎减速”与“浪费燃料”，在当前奖励信号（`progress_delta` 和 `landing` 均 <0.1/step）不足以抵偿惩罚的情况下，agent 彻底放弃使用引擎。  
**变换层级**：Level 2 结构变换（全局惩罚 → 局部门控奖励）。  
**预期改善**：移除常数值惩罚后，agent 不再有“避免一切引擎”的动机；新奖励仅在垂直下降过快时对主引擎点火给予明确正反馈，引导 agent 在下坠过程中主动减速，从而延长 episode 并有机会改善姿态、接近着陆。

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
    w_approach   = 1.0
    w_touchdown  = 10.0

    th_angle  = 0.5
    th_vel    = 1.0
    th_angvel = 2.0

    # brake reward weights/thresholds
    w_brake = 0.5
    descend_threshold = -0.3

    # ------------------- 1. distance-improvement progress signal -------------------
    delta_dist = dist - next_dist   # positive when approaching target

    gate_min = 0.1
    gate_angle  = max(gate_min, 1.0 - abs(n_angle)  / th_angle)
    gate_vel    = max(gate_min, 1.0 - next_vel_abs   / th_vel)
    gate_angvel = max(gate_min, 1.0 - abs(n_angvel)  / th_angvel)
    gate = (gate_angle * gate_vel * gate_angvel) ** (1.0/3.0)

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

    # ------------------- 3. brake reward (replaces fuel penalty) -------------------
    brake_reward = 0.0
    if action == 2 and nvy < descend_threshold:
        brake_reward = w_brake * (descend_threshold - nvy)

    # ------------------- total reward -------------------
    total_reward = progress_delta + landing + brake_reward

    components = {
        'progress_delta':   progress_delta,
        'landing':          landing,
        'brake_reward':     brake_reward
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测信号已纳入奖励，但常数引擎惩罚在正向奖励微弱时彻底抑制了引擎探索，导致所有 episode 快速坠毁。
- **behavior**: agent 为完全避免 `fuel_penalty`（active_rate 4.5%）而选择不动引擎自由落体，68 步即撞毁，无任何着陆尝试。
- **signal**: 缺少在坠落危急时鼓励使用主引擎减速的正向信号，常数惩罚不分必要性，纯粹起抑制作用。
- **level**: Level 2（结构变换：全局惩罚 → 局部门控奖励）
- **hypothesis**: 移除常数惩罚并仅在下降过快时对主引擎点火给予明确奖励，将引导 agent 在下坠过程中主动减速，从而延长 episode、改善姿态，为后续着陆组件发挥作用创造条件。
- **risk**: 可能鼓励 agent 过度使用主引擎导致长时间悬停或为获取奖励刻意保持下坠，但由于奖励严格依赖于垂直速度超过阈值，悬停时 nvy≈0 不会触发，下坠幅度也会随减速自然减小，滥用风险较低。
