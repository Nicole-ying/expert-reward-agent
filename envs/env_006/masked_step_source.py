def step(self, action):
    # Joint torques, physics simulation, lidar updates, and terrain collision
    # are omitted for compactness. Official reward computation is masked.
    state = [
        hull_angle,
        hull_angular_velocity,
        horizontal_speed,
        vertical_speed,
        hip_1_angle, hip_1_speed,
        knee_1_angle, knee_1_speed,
        hip_2_angle, hip_2_speed,
        knee_2_angle, knee_2_speed,
        1.0 if leg_1_contact else 0.0,
        1.0 if leg_2_contact else 0.0,
        lidar[0], lidar[1], lidar[2], lidar[3], lidar[4],
        lidar[5], lidar[6], lidar[7], lidar[8], lidar[9],
    ]
    terminated = body_fallen_over or reached_end_of_terrain
    masked_reward = <OFFICIAL_REWARD_MASKED>
    return state, masked_reward, terminated, False, {}
