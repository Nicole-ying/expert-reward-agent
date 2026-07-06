def step(self, action):
    # MuJoCo simulation, articulated-body dynamics, and contact resolution are omitted.
    state = [
        torso_height,
        torso_angle,
        upper_joint_angle,
        lower_joint_angle,
        foot_joint_angle,
        forward_velocity,
        vertical_velocity,
        torso_angular_velocity,
        upper_joint_speed,
        lower_joint_speed,
        foot_joint_speed,
    ]

    terminated = (
        body_height_outside_healthy_range
        or torso_angle_outside_healthy_range
        or state_value_outside_finite_healthy_range
    )
    truncated = time_limit_reached
    masked_reward = <OFFICIAL_REWARD_MASKED>

    # The real environment exposes official reward terms in info; they are masked
    # and forbidden for generated reward functions in this experiment.
    return state, masked_reward, terminated, truncated, {}

