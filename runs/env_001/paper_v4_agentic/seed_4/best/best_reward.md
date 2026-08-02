# 设计理由
第一轮反思：当前奖励函数存在**量级严重不足**的问题。环境内置的燃料/坠毁惩罚（约 -1.53/step）占绝对主导，生成奖励的 episode_sum_mean 仅 0.6，完全无法抵抗。`soft_landing` 组件的 active_rate 仅 0.7%，是一个僵尸组件——它要求支撑腿接触（二值条件），而 agent 在早期探索中根本达不到着陆条件，该信号在 99.3% 的 step 中无效。`progress_gated` 虽然活跃（92%），但平均每步 0.0021，过于微弱。子代理调研明确指出“需要约 100 倍的幅度提升”才能让奖励信号与 -1.53 的环境惩罚竞争。

本轮对 `soft_landing` 组件执行 **Level 2 结构变换**：从二进制接触奖励转变为**连续的接近度-稳定性密集奖励（proximity_stability）**。移除对接触的硬性依赖，改为基于 `next_dist` 的接近度因子与姿态/速度的稳定性因子的乘积，并保留接触时的乘性 boost（1.5 倍），在无接触时仍能提供强大的定向梯度。同时将权重 `w_proximity` 设为 10.0，并给所有稳定性因子施加 `gate_min=0.2` 防止乘积塌缩。预期 per-step 平均奖励达到约 2.0，能在 episode 长度内累积超过环境总惩罚（~104），使 score 转正，引导 agent 学会减速、靠近目标并最终着陆。

`progress_gated` 本轮未改动，保留其方向引导。

# 代码
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
    w_proximity = 10.0   # 大幅提升 soft_landing 继承者的权重

    th_angle  = 0.5    # radians, about 30 degrees
    th_vel    = 1.0    # speed magnitude
    th_angvel = 2.0    # rad/s
    th_dist   = 0.5    # proximity to target

    gate_min = 0.1     # for progress gate
    gate_min_stab = 0.2  # for stability factors to prevent collapse

    # ------------------- 1. progress signal (distance delta) -------------------
    delta_dist = max(0.0, dist - next_dist)

    gate_angle  = max(gate_min, 1.0 - abs(angle)  / th_angle)
    gate_vel    = max(gate_min, 1.0 - vel_abs      / th_vel)
    gate_angvel = max(gate_min, 1.0 - abs(angvel)  / th_angvel)
    gate = gate_angle * gate_vel * gate_angvel

    progress_gated = w_progress * delta_dist * gate

    # ------------------- 2. proximity + stability reward (replaces soft_landing) -------------------
    # proximity: how close to target (0,0)
    prox_factor = max(0.0, 1.0 - next_dist / th_dist)

    # stability after the step
    a_stab  = max(gate_min_stab, 1.0 - abs(n_angle)  / th_angle)
    v_stab  = max(gate_min_stab, 1.0 - next_vel_abs   / th_vel)
    av_stab = max(gate_min_stab, 1.0 - abs(n_angvel)  / th_angvel)
    stab = a_stab * v_stab * av_stab

    # contact gives a 1.5x multiplier, but is not required
    contact_flag = 1.0 if (n_left + n_right) >= 1.0 else 0.0
    contact_mult = 1.0 + 0.5 * contact_flag

    proximity_stability_reward = w_proximity * prox_factor * stab * contact_mult

    # ------------------- total reward -------------------
    total_reward = progress_gated + proximity_stability_reward

    components = {
        'progress_gated':   progress_gated,
        'proximity_stability': proximity_stability_reward
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号齐全但量级不足，环境惩罚 (-1.53/step) 占绝对主导；soft_landing 僵尸组件（active_rate 0.7%）完全失效。
- **behavior**: agent 快速触发 crash/出界终止（len≈68，全部 terminated），无生存或靠近目标的动力。
- **signal**: 正向奖励总量（~0.6/episode）完全无法对抗环境惩罚（~-104/episode），需要约 100× 的量级提升并需要密集引导信号。
- **level**: Level 2（将 soft_landing 从二值接触奖励变换为密集的接近度-稳定性奖励）
- **hypothesis**: 密集的 proximity_stability 信号（预期 ~2.0/step）将对抗环境惩罚，引导 agent 学习减速、保持姿态并靠近目标，从而突破快速终止的困局。
- **risk**: 若权重过高，可能使 agent 在目标区域上方徘徊而不完成实际着陆；后续可通过逐步收紧稳定性阈值或降低权重来应对。