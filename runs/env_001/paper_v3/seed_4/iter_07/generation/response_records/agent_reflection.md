# Response Record

## 1. evidence
上一轮score=4.29、episode长度~950步、15/20截断、5/20终止，proximity均值694占78.4%且100%激活，descent_quality均值138占15.7%且100%激活，contact_quality激活率仅1.5%，速度/姿态惩罚系数0.01几乎无作用；agent在全长episode中持续收取proximity和descent_quality的稠密奖励却几乎不触地，external score极低但内部奖励总和很高，呈现典型的中等分平台。

## 2. behavior_diagnosis
agent学会了在目标附近徘徊生存，通过保持中等距离和低速姿态持续获取proximity（~0.73/步）和descent_quality（~0.15/步无接触泄漏）的稳定收益，但从未真正着陆——contact_quality仅1.5%激活率，绝大多数episode被时间截断而非成功终止。

## 3. signal_completeness
引导职责（proximity）、着陆质量（descent_quality）、接触奖励（contact_quality）名义上覆盖了接近、软着陆和稳定停靠，但proximity作为绝对状态值允许"占据好状态即持续获奖"，descent_quality的0.05无接触泄漏进一步支撑了不触地的生存策略，缺少对"完成着陆"的强激励——这是典型的`state_to_improvement`证据模式。

## 4. selected_level
Level 2，触发条件为`state_to_improvement`：proximity作为持久状态值被agent在中等距离上无限收割，策略占据proxy最优状态而不完成任务；contact_quality激活率1.5%接近0也满足Level 2入口条件。

## 5. selected_intervention
仅修改proximity组件的数学形态，从绝对状态值`1/(1+dist_next)`改为势能差`1/(1+dist_next) - 1/(1+dist)`，同步将系数从1.0提升至100.0以匹配新值域（单步delta约0.001~0.05，完整成功episode的telescoping总和约1.0）。

## 6. falsifiable_hypothesis
将proximity从"占据状态即获奖"改为"接近才获奖"后，徘徊行为将失去主要收益来源（hovering时delta≈0），agent被迫通过实际着陆获取descent_quality和contact_quality的回报，从而打破中等分平台。

## 7. expected_next_round
proximity的episode_sum_mean应大幅下降（从694降至成功episode约100以内，失败/徘徊episode接近0），episode_length应下降（不再无限徘徊），contact_quality激活率和external score应上升，如果agent成功学会着陆。

## 8. main_risk
纯势能差奖励在远距离时单步信号微弱（0.001~0.01量级），可能导致早期到达性信用分配不足，agent在远距离阶段学习困难或退化为早期失败。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract signals from current and next observations
    x_old = obs[0]
    y_old = obs[1]
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Proximity: state-to-improvement transform ---
    # Replace persistent state value with improvement (potential difference).
    # Only rewards actual approach, not occupying a proxy-optimal distance.
    old_dist = (x_old**2 + y_old**2) ** 0.5
    new_dist = (x**2 + y**2) ** 0.5
    old_prox = 1.0 / (1.0 + old_dist)
    new_prox = 1.0 / (1.0 + new_dist)
    proximity_delta = new_prox - old_prox          # positive when getting closer, range ~[-1,1]
    w_proximity = 100.0                             # scales telescoping sum to meaningful range
    comp_prox = w_proximity * proximity_delta

    # --- Descent quality (unchanged from current) ---
    speed_norm = (vx**2 + vy**2) ** 0.5
    factor_vel = 1.0 / (1.0 + speed_norm)
    factor_angle = 1.0 / (1.0 + abs(angle) + abs(angular_vel))
    height_factor = 1.0 / (1.0 + abs(y))
    contact_sum = left_contact + right_contact
    contact_gate = 0.05 + 0.95 * min(1.0, contact_sum)
    descent_quality = contact_gate * height_factor * factor_vel * factor_angle
    w_descent = 3.0
    comp_descent = w_descent * descent_quality

    # --- Contact quality (unchanged from current) ---
    both_contact = (left_contact + right_contact) >= 1.5
    if both_contact:
        contact_quality = factor_vel * factor_angle
        w_contact = 5.0
        comp_contact = w_contact * contact_quality
    else:
        comp_contact = 0.0

    # --- Penalties (unchanged from current) ---
    w_vel_pen = 0.01
    vel_pen = -w_vel_pen * (vx**2 + vy**2)

    w_att_pen = 0.01
    att_pen = -w_att_pen * (angle**2 + angular_vel**2)

    total_reward = comp_prox + comp_descent + comp_contact + vel_pen + att_pen

    reward_components = {
        'proximity': comp_prox,
        'descent_quality': comp_descent,
        'contact_quality': comp_contact,
        'velocity_penalty': vel_pen,
        'attitude_penalty': att_pen,
    }
    return float(total_reward), reward_components
```
