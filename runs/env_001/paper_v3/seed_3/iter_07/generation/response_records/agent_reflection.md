# Response Record

`evidence`：当前策略 external score=-31.49，episode平均长度960.65，17/20被truncated（生存满步数）；approach_and_soft_landing组件的episode_sum_mean高达686.97（步均~0.72），但其结构为per-step state value `1/(1+dist)`，agent可靠近目标后持续领取该奖励而不完成着陆；contact_reward active_rate仅0.2%，着陆行为几乎不发生。上一轮在best基础上为stabilization加了距离门控，得分恶化，说明门控不是根因。

`behavior_diagnosis`：agent学会了靠近目标区域并保持存活，但拒绝完成着陆——因为着陆会终止episode、停止收集per-step approach reward，而悬停或徘徊可以持续收割。这是一个典型的"state farming"行为，外部得分为负说明实际任务未完成且可能消耗燃料被惩罚。

`signal_completeness`：当前奖励职责基本完备（approach、soft landing velocity、upright stabilization、contact），但approach组件的数学形态是per-step state value而非improvement，导致"占据好状态即可持续获奖"，缺失的是"进展"语义。contact reward在数值上也被持续的approach reward淹没。

`selected_level`：Level 2 — `state_to_improvement`变换。触发条件是："占据好状态即可持续获奖"证据明确（高episode_sum_mean + 高truncation率 + 低contact active_rate + 负external score），且这是结构性问题而非单纯尺度问题。

`selected_intervention`：将approach组件从per-step state value `1/(1+dist)` 改为distance improvement `max(0, prev_dist - curr_dist)`，乘以系数使单步信号可学习。保留velocity penalty的距离门控、全局upright stabilization和contact reward不变（以best为基底）。

`falsifiable_hypothesis`：improvement-based reward的总奖励上限受初始距离约束（~14），消除了"无限悬停收割"的激励；agent在目标附近静止时approach reward归零，唯一正奖励来自完成着陆的contact reward，因此应驱动agent实际着陆而非徘徊。

`expected_next_round`：episode平均长度应显著下降（不再悬停满960步），terminated比例上升，contact_reward的active_rate上升，external score改善。若external score不升反降可能是系数不当或纯improvement信号过于稀疏。

`main_risk`：纯improvement reward在早期学习阶段信号密度低于state value，可能导致探索困难或学习变慢；若agent初始化策略不做任何有意义的移动，delta_dist恒为0，整个approach组件无反馈。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Previous state (pre-action)
    prev_x = obs[0]
    prev_y = obs[1]
    
    # Next state (post-action)
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Component 1: Approach improvement (state_to_improvement) ---
    # Reward reduction in distance to target, not the state of being close.
    prev_dist = (prev_x**2 + prev_y**2) ** 0.5
    curr_dist = (x**2 + y**2) ** 0.5
    delta_dist = prev_dist - curr_dist  # positive = moved closer
    w_approach = 10.0
    comp_approach = w_approach * max(0.0, delta_dist)

    # --- Component 2: Soft landing velocity penalty (distance-gated) ---
    gate_vel = 1.0 / (1.0 + curr_dist)
    vel_penalty = (vx**2 + vy**2) * gate_vel
    w_vel = 0.1
    comp_soft_landing = -w_vel * vel_penalty

    # --- Component 3: Upright stabilization (global, from best) ---
    angle_penalty = angle**2
    angvel_penalty = ang_vel**2
    w_angle = 0.1
    w_angvel = 0.1
    comp_stabilization = -w_angle * angle_penalty - w_angvel * angvel_penalty

    # --- Component 4: Successful contact reward ---
    w_contact = 3.0
    comp_contact = w_contact * (left_contact * right_contact)

    total_reward = comp_approach + comp_soft_landing + comp_stabilization + comp_contact
    components = {
        "approach_improvement": comp_approach,
        "soft_landing_velocity": comp_soft_landing,
        "upright_stabilization": comp_stabilization,
        "contact_reward": comp_contact
    }
    return float(total_reward), components
```
