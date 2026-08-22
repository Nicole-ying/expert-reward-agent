# Response Record

# 设计理由
本轮属于 **Level 2 结构变换**：增加一个新组件 `fuel_penalty`，以弥补当前奖励函数在燃料消耗信号上的缺失。  
环境事实 §1 明确要求“尽量减少发动机使用量（燃料消耗）”，但现有 reward 完全没有体现这一点。  
虽然 `proximity_stability` 组件已成功引导 agent 学会存活并接近目标（iter 2 将 score 从 -110 提升至 98.8，len 增至 372），但由于缺少燃料消耗的反向信号，agent 可能频繁且不必要地使用引擎，导致环境内置的评估分数（目标 200）仍存在较大差距。  

**数学形式**：  
`fuel_penalty = -w_fuel * float(action != 0)`，其中 `w_fuel = 0.2`。  
校准依据：主信号 `proximity_stability` 的 per‑step 约 5.94，设 `w_fuel=0.2` 使 per‑step 惩罚不超过主信号的 3.4%，远低于 0.3 倍上限。该轻量惩罚足以提供梯度信号，却不会破坏已习得的着陆策略。  
本次不改动 `progress_gated`（虽贡献极低，留待后续评估是否需要移除或重塑），也不调整 `proximity_stability` 权重，以遵循“每轮只改一个组件”原则。

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
    vel_abs       = (vx**2 + vy**2) ** 0.5
    next_vel_abs  = (nvx**2 + nvy**2) ** 0.5

    # ------------------- thresholds & weights -------------------
    w_progress = 1.0
    w_proximity = 10.0
    w_fuel = 0.2   # new: light per-step fuel penalty

    th_angle  = 0.5
    th_vel    = 1.0
    th_angvel = 2.0
    th_dist   = 0.5

    gate_min = 0.1
    gate_min_stab = 0.2

    # ------------------- 1. progress signal (distance delta) -------------------
    delta_dist = max(0.0, dist - next_dist)

    gate_angle  = max(gate_min, 1.0 - abs(angle)  / th_angle)
    gate_vel    = max(gate_min, 1.0 - vel_abs      / th_vel)
    gate_angvel = max(gate_min, 1.0 - abs(angvel)  / th_angvel)
    gate = gate_angle * gate_vel * gate_angvel

    progress_gated = w_progress * delta_dist * gate

    # ------------------- 2. proximity + stability reward -------------------
    prox_factor = max(0.0, 1.0 - next_dist / th_dist)

    a_stab  = max(gate_min_stab, 1.0 - abs(n_angle)  / th_angle)
    v_stab  = max(gate_min_stab, 1.0 - next_vel_abs   / th_vel)
    av_stab = max(gate_min_stab, 1.0 - abs(n_angvel)  / th_angvel)
    stab = a_stab * v_stab * av_stab

    contact_flag = 1.0 if (n_left + n_right) >= 1.0 else 0.0
    contact_mult = 1.0 + 0.5 * contact_flag

    proximity_stability_reward = w_proximity * prox_factor * stab * contact_mult

    # ------------------- 3. fuel penalty (new) -------------------
    engine_on = 1.0 if action != 0 else 0.0   # any engine usage
    fuel_penalty = -w_fuel * engine_on

    # ------------------- total reward -------------------
    total_reward = progress_gated + proximity_stability_reward + fuel_penalty

    components = {
        'progress_gated':   progress_gated,
        'proximity_stability': proximity_stability_reward,
        'fuel_penalty':      fuel_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号缺失燃料消耗惩罚和时间激励，proximity_stability 与任务目标对齐且有效，但比重过高，本轮仅补全缺失的燃料信号。
- **behavior**: agent 已学会存活并完成软着陆（terminated 率高，len 合理），但可能因缺少燃料惩罚而频繁使用引擎，导致环境 score 被压至 98.8（目标 200）。
- **signal**: 缺少燃料消耗抑制信号；progress_gated 贡献极微但非僵尸，暂不动。
- **level**: Level 2
- **hypothesis**: 加入轻量 per-step 燃料消耗惩罚（‑0.2/step）将梯度引导策略减少不必要的引擎点火，从而提升环境内置评估中的燃料效率得分，同时不损害已学得的着陆行为。
- **risk**: 惩罚虽轻，仍可能略微改变动作分布，若 agent 在着陆阶段完全不敢用主引擎可能导致 crash，但 w_fuel 极小（<主信号 4%）且贴近安全着陆所需的短暂引擎使用不会被过度抑制。最坏情况下需要后续调整。
