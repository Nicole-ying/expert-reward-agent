1. `evidence`：当前episode长达550步（best仅255步），landing_quality episode_sum_mean高达392.6（62.2% magnitude），distance_reward仅-208.9（33.1%），agent在目标附近以低距离惩罚持续获取状态型landing_quality奖励而未完成着陆。
2. `behavior_diagnosis`：agent学会了在目标平台附近悬停——保持较近距离、低速、良好姿态和部分接触——利用状态型landing_quality持续累积正奖励，而指数饱和距离惩罚不足以驱动其完成最终着陆。
3. `signal_completeness`：distance_reward提供接近引导但梯度饱和，stability_penalty提供运动约束，landing_quality作为持久状态奖励缺少"改善"语义——奖励"处于好状态"而非"变得更好"，导致悬停利用漏洞。
4. `selected_level`：Level 2——`state_to_improvement`变换。证据直接表明持久状态奖励被利用（长episode、高landing_quality累积、低外部进展），且上轮指数距离修改未消除悬停行为，需要结构变换而非尺度调整。
5. `selected_intervention`：将landing_quality从状态值`0.2 * sum_of_factors(next_obs)`变换为势能差`5.0 * (sum_of_factors(next_obs) - sum_of_factors(obs))`，系数5.0匹配改善形式的值域（单episode最大改善约5.0→总贡献约25，与distance/stability可比较）。
6. `falsifiable_hypothesis`：改善型landing_quality消除悬停收益——停留在同一状态得零分——迫使agent持续改善着陆质量直至完成；episode长度应缩短，landing_quality episode_sum_mean应大幅下降（不再累积），外部score应改善。
7. `expected_next_round`：landing_quality episode_sum_mean降至~5-25范围（不再持续累积），episode_length从550降至接近或低于best的255，terminated比例上升，score应刷新best（>-33.72）。
8. `main_risk`：纯改善奖励可能在终端状态附近产生振荡（来回移动以重复获取改善奖励），且改善奖励总和受限于初始到完美的质量差，可能导致总奖励尺度塌缩使distance_reward过度主导。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current state observables
    x_pos_curr = obs[0]
    y_pos_curr = obs[1]
    x_vel_curr = obs[2]
    y_vel_curr = obs[3]
    body_angle_curr = obs[4]
    left_contact_curr = obs[6]
    right_contact_curr = obs[7]

    # Next state observables
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Distance reward: bounded exponential saturation to prevent rushing
    distance_to_target = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -2.0 * (1.0 - 2.718281828 ** (-distance_to_target / 2.0))

    # 2. Light stability penalty
    stability_penalty = -(
        0.15 * abs(x_vel) +
        0.05 * abs(y_vel) +
        0.2 * abs(body_angle) +
        0.2 * abs(angular_vel)
    )

    # 3. Landing quality: improvement-based (potential difference)
    #    Measures how much the agent improved its landing quality this step.
    #    Persistent hovering yields zero, eliminating the state-reward exploit.
    curr_dist = (x_pos_curr ** 2 + y_pos_curr ** 2) ** 0.5
    curr_prox = max(0.0, 1.0 - curr_dist / 2.0)
    curr_speed_x = max(0.0, 1.0 - abs(x_vel_curr) / 0.8)
    curr_speed_y = max(0.0, 1.0 - abs(y_vel_curr) / 0.8)
    curr_angle = max(0.0, 1.0 - abs(body_angle_curr) / 0.4)
    curr_contact = 0.2 + 0.8 * (left_contact_curr + right_contact_curr) / 2.0
    curr_sum = curr_prox + curr_speed_x + curr_speed_y + curr_angle + curr_contact

    next_prox = max(0.0, 1.0 - distance_to_target / 2.0)
    next_speed_x = max(0.0, 1.0 - abs(x_vel) / 0.8)
    next_speed_y = max(0.0, 1.0 - abs(y_vel) / 0.8)
    next_angle = max(0.0, 1.0 - abs(body_angle) / 0.4)
    next_contact = 0.2 + 0.8 * (left_contact + right_contact) / 2.0
    next_sum = next_prox + next_speed_x + next_speed_y + next_angle + next_contact

    landing_quality = 5.0 * (next_sum - curr_sum)

    total_reward = distance_reward + stability_penalty + landing_quality

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components
```