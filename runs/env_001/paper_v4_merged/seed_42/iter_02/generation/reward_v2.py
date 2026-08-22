def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 航向进展：距离目标越近越好（improvement_delta）
    d_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    d_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = d_prev - d_next
    goal_progress = 1.0 * progress

    # 2. 稳定停靠奖励：靠近目标时鼓励低速、竖直、双腿接触
    proximity_thresh = 0.5
    proximity_gate = max(0.0, 1.0 - d_next / proximity_thresh)

    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    vel_thresh = 0.2
    velocity_bonus = 0.5 * max(0.0, 1.0 - speed / vel_thresh)

    angle_thresh = 0.1
    angle_bonus = 0.2 * max(0.0, 1.0 - abs(next_obs[4]) / angle_thresh)

    contact_bonus = 1.0 * next_obs[6] * next_obs[7]

    stable_bonus = proximity_gate * (velocity_bonus + angle_bonus + contact_bonus)

    # 3. 燃料效率惩罚
    fuel_penalty = -0.01 if action != 0 else 0.0

    # 4. 密集距离奖励：越接近目标奖励越大（连续有界）
    approach_reward = 0.1 / (1.0 + d_next)

    total_reward = goal_progress + stable_bonus + fuel_penalty + approach_reward
    components = {
        'goal_progress': float(goal_progress),
        'stable_bonus': float(stable_bonus),
        'fuel_penalty': float(fuel_penalty),
        'approach_reward': float(approach_reward)
    }
    return float(total_reward), components