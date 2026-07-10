def step(self, action):
    # MuJoCo simulation of a 3D quadruped with 8 torque-controlled joints.
    # Body dynamics, ground contact, and joint actuation are omitted.
    # 27D observation: 13 position + 14 velocity (no contact forces, x/y excluded).
    state = [
        body_z,
        quat_w, quat_x, quat_y, quat_z,
        joint_1_angle, joint_2_angle, joint_3_angle, joint_4_angle,
        joint_5_angle, joint_6_angle, joint_7_angle, joint_8_angle,
        body_x_velocity, body_y_velocity, body_z_velocity,
        body_roll_velocity, body_pitch_velocity, body_yaw_velocity,
        joint_1_velocity, joint_2_velocity, joint_3_velocity, joint_4_velocity,
        joint_5_velocity, joint_6_velocity, joint_7_velocity, joint_8_velocity,
    ]

    terminated = (
        body_height_outside_healthy_range
        or state_value_outside_finite_range
    )
    truncated = time_limit_reached
    masked_reward = <OFFICIAL_REWARD_MASKED>

    # The real environment exposes official reward terms in info; they are masked
    # and forbidden for generated reward functions in this experiment.
    return state, masked_reward, terminated, truncated, {}
