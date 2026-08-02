# 设计理由
本轮修改 **landing_prep** 组件（一个永远激活的密集乘积奖励，占总奖励 57.5%），将其替换为 **接触驱动的分级着陆奖励**。  
现状：agent 学会了在目标附近保持良好姿态和低速，但几乎从不触发着陆终止（15/20 截断），因为现有 `landing_prep` 在无腿接触时仍给出大量奖励（乘积因子最低仍为 0.3），形成了稳定的悬停收益。  
方案：将原组件拆成两部分，仍然保留一个较小的接近引导（`approach`，w=1.0），主要奖励转移到仅在支撑腿接触地面时才触发的 `touchdown` 质量奖励（w=10.0）。`touchdown` 只在接触时发放，且数值与速度、角度、位置等软着陆质量因子相乘，促使 agent 在接触瞬间同时满足低速、竖直、接近中心的条件。这样形成“接近获得保底 → 接触获得高额报酬”的梯度，迫使 agent 从悬停转向实际着陆。  
系数校准：`progress_gated` 每步约 0.47，`approach` 每步平均约 0.3‑0.5，合计正信号约 0.8‑1.0，仍远超燃料惩罚（-0.2），不会导致长度崩溃。`touchdown` 单次最高约 10，相当于约 20 步的净收益，足以驱动行为突变。

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

    # Landing component weights
    w_approach   = 1.0   # 连续接近鼓励（小，防止崩塌）
    w_touchdown  = 10.0  # 接触着陆成功奖励（大）

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

    # ------------------- 2. landing (approach + touchdown) -------------------
    contact_next = (n_left + n_right) / 2.0  # range 0..1

    # quality factors for touchdown (only active when contacting)
    pos_factor    = max(0.0, 1.0 - abs(nx) / 0.5)
    height_factor = max(0.0, 1.0 - abs(ny) / 0.5)
    vel_factor    = max(0.0, 1.0 - next_vel_abs / 0.5)
    angle_factor  = max(0.0, 1.0 - abs(n_angle) / 0.3)
    angvel_factor = max(0.0, 1.0 - abs(n_angvel) / 0.5)

    touchdown_reward = 0.0
    approach_reward  = 0.0

    if contact_next > 0.1:   # at least one leg touching
        quality = pos_factor * height_factor * vel_factor * angle_factor * angvel_factor * contact_next
        touchdown_reward = w_touchdown * quality
        # no approach reward when touching (the touchdown dominates)
    else:
        # original continuous approach factors (without leg contact)
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
    total_reward = progress_gated + landing + fuel_penalty

    components = {
        'progress_gated':  progress_gated,
        'landing':          landing,
        'fuel_penalty':     fuel_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 观测维已全部使用，缺失的是“接触着陆”这一终点事件的强信号，密集奖励 sustain 了悬停而不触发终止。
- **behavior**: agent 在目标附近保持良好姿态和低速，但 85% 的回合因超时截断，未完成着陆。
- **signal**: `landing_prep` 在无腿接触时仍给出大量奖励，缺乏只有接触才能获得的高额报酬来改变平衡。
- **level**: Level 2
- **hypothesis**: 在保持安全接近奖励的前提下，加入仅在支撑腿接触时发放的高权重着陆质量奖励，将驱动 agent 从悬停转向实际触地并停稳。
- **risk**: 若初期随机接触不足，agent 可能仍陷于悬停；或者为了追求接触而过早猛撞，导致 crash 率上升、长度缩短。