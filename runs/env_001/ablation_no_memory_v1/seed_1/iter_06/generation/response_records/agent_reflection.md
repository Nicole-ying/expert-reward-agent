# Response Record

`evidence`：当前分数-115.44，全部20条episode在~68步内提前终止（early_terminal），landing_bonus的active_rate仅0.4%（稀疏二值信号几乎不可学习），stability_penalty与progress_delta幅值相近（-1.16 vs +1.12）导致净学习信号接近零，上一轮将best的三处关键设计（3x progress缩放、距离门控稳定性、连续软着陆代理）全部移除，得分崩溃。

`behavior_diagnosis`：agent在所有episode中快速撞击或飞出边界，未能靠近目标垫。progress_delta几乎不积累正值，说明agent没有学到向目标移动；always-on稳定性惩罚在远离目标时惩罚必要的速度和角度变化，与progress信号相互抵消。

`signal_completeness`：当前代码缺少可学习的接近/着陆反馈（landing_bonus稀疏到0.4%），稳定性约束无距离门控导致全局压制探索；best代码已具备这三项职责，但软着陆代理使用乘积形式可能在中等距离/速度下值塌缩到接近零。

`selected_level`：Level 2 — product_to_noncollapsing_joint。当前失败的直接原因是上一轮移除了best的全部关键设计；以best为基础，针对其软着陆奖励的乘积塌缩风险做一次结构变换。

`selected_intervention`：将best中的`soft_landing_reward = 1.0 * nearness * slowness`改为几何平均形式`soft_landing_reward = 1.0 * (nearness * slowness)**0.5`，其他组件完全保持best设计不变。

`falsifiable_hypothesis`：乘积形式在nearness=0.044、slowness=0.249（距离0.5、速度0.5）时塌缩到0.011，而几何平均给出0.105，提供约10倍梯度。这应在agent靠近目标但尚未完全减速的过渡阶段提供更平滑的学习信号，使landing-like行为更早出现。

`expected_next_round`：soft_landing_reward的episode_sum_mean应上升，active_rate保持高位（连续信号），agent应能稳定接近目标垫而非快速撞击，score显著改善，episode_length增加。

`main_risk`：几何平均虽然缓解塌缩但仍保留联合约束，在agent完全远离目标时值仍接近零（nearness≈0），不会干扰progress主导的接近阶段；主要风险是系数1.0在叠加progress_delta(3x)后可能使总奖励尺度变化，但范围可控。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Previous observation
    px_prev = obs[0]
    py_prev = obs[1]
    prev_distance = (px_prev**2 + py_prev**2)**0.5

    # Next observation
    px = next_obs[0]
    py = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    # Distance to target pad center
    next_distance = (px**2 + py**2)**0.5
    # Speed magnitude
    speed = (vx**2 + vy**2)**0.5

    # 1. Main learning signal: progress delta reward
    #    Positive when approaching target, negative when retreating.
    progress_delta = 3.0 * (prev_distance - next_distance)

    # 2. Stability constraint: light penalty on high speed, large angle, high angular velocity
    #    Distance-gated: only active when agent is near the target pad (within ~2 units).
    #    This frees the agent to move decisively when far away.
    raw_stability = -0.1 * speed - 0.05 * abs(angle) - 0.05 * abs(ang_vel)
    gate_radius = 2.0
    distance_gate = max(0.0, 1.0 - next_distance / gate_radius)
    stability_penalty = raw_stability * distance_gate

    # 3. Soft approaching proxy: geometric mean of nearness and slowness
    #    Uses geometric mean instead of product to avoid reward collapse
    #    when one factor is moderate and the other is low.
    #    Still preserves the joint requirement: both factors must be non-zero.
    sigma_dist = 0.2
    sigma_speed = 0.3
    nearness = 2.718281828 ** (-(next_distance**2) / (2.0 * sigma_dist**2))
    slowness = 2.718281828 ** (-(speed**2) / (2.0 * sigma_speed**2))
    soft_landing_reward = 1.0 * (nearness * slowness)**0.5

    # Combine components
    total_reward = progress_delta + stability_penalty + soft_landing_reward

    components = {
        'progress_delta': progress_delta,
        'stability_penalty': stability_penalty,
        'soft_landing_reward': soft_landing_reward
    }

    return float(total_reward), components
```
