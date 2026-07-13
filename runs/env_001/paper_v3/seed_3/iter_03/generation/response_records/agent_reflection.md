# Response Record

1. `evidence`：最终策略在20个评估episode中全部truncated（1000步截断），terminated=0，contact_reward触发率为0%，episode_sum_mean=0；approach_and_soft_landing组件累计均值340.06，magnitude_share=99.9%，signed_share=99.9%，active_rate=100%，说明agent每步都在收取基于状态的趋近奖励却从未着陆；外部score=-21.82，远低于target=200；上一轮将w_approach从best的1.0降到0.5，但外部行为未改善，agent仍在proxy上稳定刷分。

2. `behavior_diagnosis`：agent学会在目标附近（推断dist≈1.94）悬停1000步持续收取1/(1+dist)≈0.34/步的趋近奖励，累计≈340，同时避免触发任何接触（双腿从未同时着垫），形成典型的proxy farming——占据好状态即可持续获奖而不完成任务。

3. `signal_completeness`：趋近信号存在但形态为state_to_improvement问题（状态值而非改善量）；速度惩罚存在但被gate弱化；接触奖励存在但稀疏且从未触发（3.0一次性bonus远不足以与340的累积proxy竞争）；缺少任何时间代价或悬停惩罚来打破平台期。

4. `selected_level`：Level 2——证据直接否定当前数学形态（state_to_improvement模式），agent在状态值奖励上无限farm而不完成接触任务，需要结构变换而非单纯降系数。

5. `selected_intervention`：将`approach_reward`从状态值`1/(1+dist_next)`变换为势能差`dist_curr - dist_next`（即potential=-dist的改善量），同步设置w_approach=5.0匹配新值域；其他组件（velocity penalty、stabilization、contact）保持不变。

6. `falsifiable_hypothesis`：势能差形式使悬停产生≈0的趋近奖励（dist不变→delta=0），agent无法再通过停留在同一距离持续刷分；唯一获取正向奖励的途径是实际缩小与目标的距离，到达dist≈0后趋近信号饱和，只有触发contact_reward才能获得额外奖励，这应迫使agent探索并完成着陆。

7. `expected_next_round`：approach_and_soft_landing的episode_sum_mean应大幅下降（不再反映持续刷分），contact_reward的active_rate应从0%上升（agent被推向着陆），terminated比例应打破0/20，外部score应改善且episode_length应缩短。

8. `main_risk`：势能差信号在单步内幅度较小（典型步进0.05-0.3），可能导致早期趋近梯度不足，agent徘徊不学；若contact_reward=3.0仍不足以在dist≈0时驱动最终着陆动作，可能需要在下一轮增大接触权重或添加微小时间惩罚。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Goal: Guide a 2D lander to softly touch down on the target pad with both feet,
    staying upright and near zero velocity.
    
    Key change: approach reward is now potential-based improvement (dist_curr - dist_next)
    instead of a persistent state value, preventing proxy farming via hovering.
    """
    # Unpack current and next observations
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Component 1: Approach improvement and soft landing ---
    # Distance-based potential: closer = higher potential
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5

    # Improvement = potential(next) - potential(curr) = dist_curr - dist_next
    # Positive when approaching target, zero when hovering, negative when retreating
    approach_delta = dist_curr - dist_next

    # Velocity penalty gated by proximity: stronger when close to target
    gate_vel = 1.0 / (1.0 + dist_next)
    vel_penalty = (vx**2 + vy**2) * gate_vel

    w_approach = 5.0
    w_vel = 0.1
    comp_approach_landing = w_approach * approach_delta - w_vel * vel_penalty

    # --- Component 2: Upright stabilization ---
    angle_penalty = angle**2
    angvel_penalty = ang_vel**2
    w_angle = 0.1
    w_angvel = 0.1
    comp_stabilization = -w_angle * angle_penalty - w_angvel * angvel_penalty

    # --- Component 3: Successful contact reward ---
    # Both landing legs must touch simultaneously
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
