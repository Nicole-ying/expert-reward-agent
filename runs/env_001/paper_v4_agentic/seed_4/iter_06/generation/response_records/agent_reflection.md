# Response Record

# 设计理由
改动了 **landing_progress** 组件。  
- 旧组件在非一次性着陆时使用 `w_delta * distance_reduction * stab`，`stab` 由三个无底门的 `max(0, ...)` 乘积构成，大量步中 `stab=0`，导致实际 per‑step 奖励极低（episode_sum_mean 仅 17.3），无法形成有效的着陆引导。  
- 新组件改为 **密集的着陆准备乘积因子**：把位置、高度、速度、姿态、角速度、接触六个连续因子全部绑定在同一乘积内，每个因子都用 `max(0.2, ...)` 保证不塌缩至零。权重设为 `w_lp=3.0`，在近乎理想状态下每步可提供约 3.0 的正奖励，远大于 `progress_gated` (~0.31) 和 `fuel_penalty` (−0.2) 的步均量级，从而驱动 agent 从徘徊状态进入着陆流程。  
- 保留了原有的一次性着陆条件逻辑（但实现中已移除，因为连续乘积足以引导至成功终止，且避免未触发的稀疏信号浪费参数空间）。  
- 系数校准：各因子阈值基于终止边界的 60–80% 估计（由环境事实中 crash/成功条件推断）。

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

    # ------------------- 1. velocity-toward-target progress signal -------------------
    dir_x = -nx / (next_dist + 1e-6)
    dir_y = -ny / (next_dist + 1e-6)
    toward_speed = nvx * dir_x + nvy * dir_y

    gate_min = 0.1
    gate_angle  = max(gate_min, 1.0 - abs(n_angle)  / th_angle)
    gate_vel    = max(gate_min, 1.0 - next_vel_abs   / th_vel)
    gate_angvel = max(gate_min, 1.0 - abs(n_angvel)  / th_angvel)
    gate = gate_angle * gate_vel * gate_angvel

    progress_gated = w_progress * max(0.0, toward_speed) * gate

    # ------------------- 2. landing preparation (dense continuous proxy) -------------------
    w_lp = 3.0

    dist_factor   = max(0.2, 1.0 - next_dist / 1.5)
    height_factor = max(0.2, 1.0 - abs(ny) / 1.5)
    vel_factor    = max(0.2, 1.0 - next_vel_abs / 0.8)
    angle_factor  = max(0.2, 1.0 - abs(n_angle) / 0.5)
    angvel_factor = max(0.2, 1.0 - abs(n_angvel) / 1.0)
    contact_factor = max(0.3, (n_left + n_right) / 2.0)

    landing_prep = w_lp * dist_factor * height_factor * vel_factor * angle_factor * angvel_factor * contact_factor

    # ------------------- 3. fuel penalty -------------------
    engine_on = 1.0 if action != 0 else 0.0
    fuel_penalty = -w_fuel * engine_on

    # ------------------- total reward -------------------
    total_reward = progress_gated + landing_prep + fuel_penalty

    components = {
        'progress_gated':  progress_gated,
        'landing_prep':    landing_prep,
        'fuel_penalty':    fuel_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测均可使用且未被遗漏，未发现信号缺口；组件活跃率均正常，问题属于信号校准而非缺失。
- **behavior**: agent 学会了保持姿态并在中心附近缓慢移动，以赚取 `progress_gated` 的每步正向奖励，从未触发着陆接触，所有 episode 超时 truncation。
- **signal**: 着陆相关奖励（原 `landing_progress`）密度过低且频繁为零，无法抗衡 `progress_gated` 的“徘徊收益”。
- **level**: Level 2（结构变换：将稀疏/低效的改善项改为密集的连续因子乘积）
- **hypothesis**: 新的密集乘积会为接近软着陆的状态提供每步约 3.0 的强正奖励，远超徘徊收益，驱策 agent 完成下降、减速、摆正姿态并放下支撑腿。
- **risk**: 乘积因子数量多，可能在恶劣状态下参数极小，但所有因子都设了最低值 `0.2/0.3`，不会真正塌缩；最可能副作用是 agent 过早下降导致坠毁，但 `angle_factor` 和 `vel_factor` 的门控会抑制该行为。
