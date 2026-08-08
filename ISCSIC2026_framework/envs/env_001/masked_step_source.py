def step(self, action):
    # Action validation, physics step, engine impulses and wind are omitted for compactness.
    # Official reward computation is masked and must not be reconstructed.
    state = [
        x_position_relative_to_target,
        y_position_relative_to_pad_height,
        x_velocity,
        y_velocity,
        body_angle,
        angular_velocity,
        1.0 if left_support_contact else 0.0,
        1.0 if right_support_contact else 0.0,
    ]
    terminated = crash_or_body_contact or horizontal_position_outside_viewport or body_not_awake_or_settled
    masked_reward = <OFFICIAL_REWARD_MASKED>
    return state, masked_reward, terminated, False, {}
