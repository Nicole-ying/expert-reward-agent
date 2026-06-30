def step(self, action):
    # Physics simulation, joint motor control, terrain contact and friction are omitted for compactness.
    # Official reward computation is masked and must not be reconstructed.
    state = [
        hull_angle,
        hull_angular_velocity,
        horizontal_velocity,
        vertical_velocity,
        hip1_angle,
        hip1_speed,
        knee1_angle,
        knee1_speed,
        hip2_angle,
        hip2_speed,
        knee2_angle,
        knee2_speed,
        1.0 if leg1_ground_contact else 0.0,
        1.0 if leg2_ground_contact else 0.0,
        lidar_0, lidar_1, lidar_2, lidar_3, lidar_4,
        lidar_5, lidar_6, lidar_7, lidar_8, lidar_9,
    ]
    terminated = body_fallen_over or reached_end_of_terrain
    masked_reward = <OFFICIAL_REWARD_MASKED>
    return state, masked_reward, terminated, False, {}
