`evidence`：当前奖励得分252.56已超target 200，20/20 episode均以terminated结束（无truncation），episode_length=314.7。组件中soft_landing_reward以155.2均值占signed_share 93.7%绝对主导，progress_delta仅3.9（2.4%），stability_penalty为-6.4（3.9%）。上一轮（iter 6副本）被判定为语义重复，当前需做实质性不同的修改。

`behavior_diagnosis`：策略已能稳定到达并settle在目标垫上（20/20成功终止），但奖励完全没有对引擎使用的约束。任务明确要求"尽可能少地使用引擎推力"，当前314步的episode长度暗示agent可能在soft_landing_reward的持续奖励下过度使用引擎维持姿态，而非尽快高效着陆。

`signal_completeness`：必要职责中缺少动作效率/燃料约束。任务要求最小化引擎推力使用，但当前三个组件均不涉及action选择，这是明确缺失的必要约束信号。

`selected_level`：Level 2 — 添加缺失的必要动作约束组件（fuel_penalty），属于信号完备性修复，非单纯尺度调整。

`selected_intervention`：新增fuel_penalty组件，对非零动作（action 1/2/3）施加每步-0.02的轻量惩罚，action 0（no_engine）不受罚。其他三个组件保持与iter 6完全一致。

`falsifiable_hypothesis`：添加燃料惩罚后，agent应学会在不需要推力时选择action 0，从而降低episode内非零动作比例、缩短episode_length，同时保持成功终止率。若fuel_penalty的active_rate与非零动作比例同步下降，且score因效率提升而保持在target附近或以上，则假设成立。

`expected_next_round`：fuel_penalty的episode_sum_mean应为负值（量级约-3到-8），active_rate与非零动作使用率对应；episode_length应下降（减少无谓引擎使用）；terminated保持高位；score可能小幅下降但应仍在target 200以上。

`main_risk`：若-0.02的惩罚过轻，燃料效率改善不可见；若在早期探索阶段惩罚累积过快，可能抑制必要的主引擎使用，导致agent不敢加速接近目标，反而延长episode或降低成功率。系数可能需要后续调整。

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
    raw_stability = -0.1 * speed - 0.05 * abs(angle) - 0.05 * abs(ang_vel)
    gate_radius = 2.0
    distance_gate = max(0.0, 1.0 - next_distance / gate_radius)
    stability_penalty = raw_stability * distance_gate

    # 3. Soft approaching proxy: geometric mean of nearness and slowness
    sigma_dist = 0.2
    sigma_speed = 0.3
    nearness = 2.718281828 ** (-(next_distance**2) / (2.0 * sigma_dist**2))
    slowness = 2.718281828 ** (-(speed**2) / (2.0 * sigma_speed**2))
    soft_landing_reward = 1.0 * (nearness * slowness)**0.5

    # 4. Fuel efficiency penalty: penalize any engine use to encourage minimal thrust
    #    Action 0 (no_engine) is free; actions 1/2/3 incur a small per-step cost.
    fuel_penalty = -0.02 if action != 0 else 0.0

    total_reward = progress_delta + stability_penalty + soft_landing_reward + fuel_penalty

    components = {
        'progress_delta': progress_delta,
        'stability_penalty': stability_penalty,
        'soft_landing_reward': soft_landing_reward,
        'fuel_penalty': fuel_penalty
    }

    return float(total_reward), components
```