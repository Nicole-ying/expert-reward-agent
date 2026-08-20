`evidence`: 所有20个episode均被truncate在1000步，contact_reward的active_rate=0%，approach_and_soft_landing累积episode_sum_mean=1664.98(每步≈1.66)，说明agent在持续收割接近奖励而从未触发着陆；上一轮将approach系数从1.0提到3.0后得分未改善，放大proxy farming问题。

`behavior_diagnosis`: agent学会在目标附近悬停以持续收割稠密接近奖励，完全避免执行着陆动作——contact从未触发，episode全部跑满1000步被截断。

`signal_completeness`: 必要职责基本齐全，但approach组件采用持久状态值（state-based），使agent占据"靠近但不接触"的状态即可无限获利，导致sparse contact信号永远无法进入信用分配。

`selected_level`: Level 2 — 证据符合`state_to_improvement`变换："A policy can remain in a rewarding state, hover, stall, or farm a persistent state bonus without completing the task."

`selected_intervention`: 将`approach_reward`从持久状态值`w*(1/(1+dist_next))`改为势能差`Φ(s')-Φ(s)`，其中`Φ(s)=5.0/(1+dist)`；保持velocity penalty门控结构、stabilization和contact_reward不变。

`falsifiable_hypothesis`: 改为改善量奖励后，悬停不再产生收益（Δ≈0），agent必须持续靠近目标或完成着陆才能获得正奖励，应迫使策略从"悬停收割"转向"下降着陆"。

`expected_next_round`: contact_reward的active_rate应从0%上升；episode中应出现terminated（自然终止）而非全部truncated；approach组件episode_sum_mean将显著下降（因不再累积持久奖励）；外部score应改善。

`main_risk`: 纯势能差奖励在初始探索阶段可能过于稀疏——agent在学会基本靠近之前几乎收不到正反馈，可能导致学习停滞或更差的早期失败率。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Goal: Guide a 2D lander to softly touch down on the target pad with both feet.
    
    Key change: approach reward is now potential-based improvement (delta),
    not persistent state value. Hovering yields zero; only getting closer pays.
    """
    # Unpack current and next positions for potential difference
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Component 1: Approach and soft landing ---
    # Potential-based improvement: reward for getting closer, not for staying close
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    
    potential_curr = 5.0 / (1.0 + dist_curr)
    potential_next = 5.0 / (1.0 + dist_next)
    approach_reward = potential_next - potential_curr

    # Velocity penalty gated by proximity: heavier when close to target
    gate_vel = 1.0 / (1.0 + dist_next)
    vel_penalty = (vx**2 + vy**2) * gate_vel

    w_vel = 0.1
    comp_approach_landing = approach_reward - w_vel * vel_penalty

    # --- Component 2: Upright stabilization ---
    angle_penalty = angle**2
    angvel_penalty = ang_vel**2
    w_angle = 0.1
    w_angvel = 0.1
    comp_stabilization = -w_angle * angle_penalty - w_angvel * angvel_penalty

    # --- Component 3: Successful contact reward ---
    w_contact = 3.0
    comp_contact = w_contact * (left_contact * right_contact)

    total_reward = comp_approach_landing + comp_stabilization + comp_contact
    components = {
        "approach_and_soft_landing": comp_approach_landing,
        "upright_stabilization": comp_stabilization,
        "contact_reward": comp_contact
    }
    return float(total_reward), components
```