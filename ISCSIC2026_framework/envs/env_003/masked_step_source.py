def step(self, action):
    # State integration and force dynamics are omitted for compactness.
    # Official reward computation is masked and must not be reconstructed.
    state = [
        base_position,
        base_velocity,
        pole_angle,
        pole_angular_velocity,
    ]
    terminated = abs(base_position) > 2.4 or abs(pole_angle) > 0.20943951
    truncated = elapsed_steps >= 500
    masked_reward = <OFFICIAL_REWARD_MASKED>
    return state, masked_reward, terminated, truncated, {}

