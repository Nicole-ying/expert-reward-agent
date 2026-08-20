def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Potential-based shaping reward: reward = V(next_obs) - V(obs)
    where V(s) penalizes distance, speed, angle and angular velocity.
    This encourages the agent to move towards smaller distances and lower velocities.
    """
    def state_value(o):
        # Position penalty (L1 distance to target)
        pos_penalty = abs(o[0]) + abs(o[1])
        # Speed penalty (sum of L1 velocities)
        vel_penalty = abs(o[2]) + abs(o[3])
        # Orientation penalty
        ang_penalty = abs(o[4])
        # Angular velocity penalty
        angvel_penalty = abs(o[5])

        # Weighted combination; the scaling factors are chosen so that typical ranges
        # of the observables produce comparable contributions.
        return -pos_penalty - 0.5 * vel_penalty - 0.2 * ang_penalty - 0.1 * angvel_penalty

    v_obs = state_value(obs)
    v_next = state_value(next_obs)
    shaping = v_next - v_obs

    total_reward = shaping
    components = {"shaping_reward": shaping}
    return float(total_reward), components