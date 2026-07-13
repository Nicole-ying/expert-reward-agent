`evidence`：当前所有episode都被truncate到1000步，approach_proximity每episode累积均值584.9（占reward 99.4%）而contact_reward激活率0%，external score=-56——说明agent悬浮在目标附近持续收割状态型接近奖励，从未真正着陆。

`behavior_diagnosis`：agent学会了在着陆垫上方低高度、直立、低速悬浮满1000步以最大化`1/(1+dist)`的累积收益，不执行最终着陆动作（双腿从未同时接触），外部评分为负反映了燃料/时间浪费。

`signal_completeness`：四个主职责（接近、软着陆减速、姿态稳定、接触奖励）的名义存在性完备，但`approach_target`的数学形态为`state_to_improvement`漏洞——奖励"处于好状态"而非"变好"，导致agent占据好状态即可无限获利。

`selected_level`：Level 2 — `state_to_improvement`。证据直接命中"占据好状态即可持续获奖"模式：contact始终不触发、episode全长截断、状态型接近奖励占总reward的99%+。

`selected_intervention`：将`approach_proximity`从状态值`1/(1+dist)`改为势能差分`1/(1+new_dist) - 1/(1+old_dist)`，同步设置系数w_approach=5.0以适应新值域；其他三个组件不变。

`falsifiable_hypothesis`：势能差分使agent无法通过悬浮在同一位置获利——只有实际缩小距离才有正奖励。每episode总接近奖励被势能差上限约束（≈5.0），使contact_reward（3.0/步）成为完成任务的支配性正信号，agent应被驱动执行着陆动作以获取contact奖励。

`expected_next_round`：episode_length应下降（不再满1000步悬浮），contact_reward的active_rate应从0%上升，approach组件episode_sum_mean从584→个位数，external score应改善。

`main_risk`：势能差分降低每步信号强度约50倍（从~0.58/步降至~0.01/步），早期信用分配可能变难；若approach_improvement信号太弱导致探索困难，下一轮可尝试增大w_approach或补充稀疏到稠密的局部引导。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current state (pre-action)
    old_x = obs[0]
    old_y = obs[1]

    # Next state (post-action)
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Component 1: Improvement-based approach reward ---
    # Potential-based shaping: reward = Φ(next) - Φ(current)
    # Φ = 1/(1+dist), bounded in [0, 1]
    # Agent only earns reward for reducing distance, not for staying near target.
    old_dist = (old_x**2 + old_y**2) ** 0.5
    new_dist = (x**2 + y**2) ** 0.5
    w_approach = 5.0
    comp_approach = w_approach * (1.0 / (1.0 + new_dist) - 1.0 / (1.0 + old_dist))

    # --- Component 2: Soft landing velocity penalty (distance-gated) ---
    gate_vel = 1.0 / (1.0 + new_dist)
    vel_penalty = (vx**2 + vy**2) * gate_vel
    w_vel = 0.2
    comp_soft_landing = -w_vel * vel_penalty

    # --- Component 3: Upright stabilization ---
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