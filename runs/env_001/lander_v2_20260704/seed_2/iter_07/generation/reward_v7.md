**evidence**：score 从 -108 大幅跃升至 -33.7，episode 长度从 68 增至 254.8，terminated 19/20；组件 landing_quality（169.97）与 distance_reward（-156.10）量级相当、符号相反，两者几乎抵消；stability_penalty 仅贡献 -14.64；active_rate 全 100%。

**behavior_diagnosis**：agent 已学会持续靠近目标并尝试着陆（长 episode、大量 terminate），但行为偏慢（~255 步），长时间在中等距离范围（平均距离 ~0.61）维持良好姿态与低速以掠取 landing_quality，形成“舒适区”均衡，导致净得分仍为负且未体现“尽快”要求。

**signal_completeness**：具备距离、姿态/速度约束与着陆质量信号，但缺少明确的速度激励，landing_quality 权重过高使 agent 可在不迅速接近目标的情况下获得足够补偿，任务完成紧迫性不足。

**selected_level**：Level 1。landing_quality 与 distance_reward 量级抵消，且 landing_quality 过大是形成舒适区的直接原因，应通过系数调整重建相对梯度，不改变数学形态。

**selected_intervention**：仅将 landing_quality 系数从 0.2 降至 0.1，其他组件不变。

**falsifiable_hypothesis**：降低 landing_quality 将使 distance_reward 的相对重要性上升，agent 为减少累计距离惩罚会加速靠近目标，episode 长度缩短、累计负距离奖励减少、最终得分提升。

**expected_next_round**：episode 长度应下降（但不应崩溃至 <150），landing_quality 份额降低，distance_reward 占比相对升高（因步长缩短其绝对值可能减小），score 改善（负值变小或转正）。

**main_risk**：若 landing_quality 过弱，agent 可能丢失姿态与触地引导，退化为快速坠落或碰撞，导致 episode 长度骤降、得分恶化。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observables
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Distance reward: continuous negative Euclidean distance to goal
    distance_to_target = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -1.0 * distance_to_target

    # 2. Light stability penalty (preserved from best)
    stability_penalty = -(
        0.15 * abs(x_vel) +
        0.05 * abs(y_vel) +
        0.2 * abs(body_angle) +
        0.2 * abs(angular_vel)
    )

    # 3. Landing quality: sum-based joint satisfaction, coefficient reduced from 0.2 to 0.1
    prox_factor = max(0.0, 1.0 - distance_to_target / 2.0)
    speed_x_factor = max(0.0, 1.0 - abs(x_vel) / 0.8)
    speed_y_factor = max(0.0, 1.0 - abs(y_vel) / 0.8)
    angle_factor = max(0.0, 1.0 - abs(body_angle) / 0.4)
    contact_factor = 0.2 + 0.8 * (left_contact + right_contact) / 2.0

    sum_of_factors = prox_factor + speed_x_factor + speed_y_factor + angle_factor + contact_factor
    landing_quality = 0.1 * sum_of_factors  # Reduced from 0.2 to strengthen distance gradient

    total_reward = distance_reward + stability_penalty + landing_quality

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components
```