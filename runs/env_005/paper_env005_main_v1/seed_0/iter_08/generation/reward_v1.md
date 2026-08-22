```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Four-legged locomotion reward v1.
    Core design hypothesis: focus on height management and foot contact rhythm.
    Forward velocity is deliberately absent from v1 to prevent fast-then-fail collapse.
    """

    # ---------- Extract next observation signals ----------
    # Body height
    body_z = next_obs[0]

    # Body orientation (quaternion) — individual indices, no slicing
    quat_w = next_obs[4]  # scalar component
    quat_x = next_obs[1]
    quat_y = next_obs[2]
    quat_z = next_obs[3]

    # Body velocities
    forward_vel = next_obs[13]   # desired forward movement
    lateral_vel = next_obs[14]   # sideways drift
    vertical_vel = next_obs[15]  # bouncing

    # Angular velocities (body rotation)
    roll_rate = next_obs[16]
    pitch_rate = next_obs[17]

    # Foot contact z-force — access with single indices, no slicing allowed
    fl_foot_z = next_obs[47]   # front-left foot fz (contact force z-component)
    fr_foot_z = next_obs[65]   # front-right foot fz
    rl_foot_z = next_obs[83]   # rear-left foot fz
    rr_foot_z = next_obs[101]  # rear-right foot fz

    # ---------- Component 1: Height Management (主学习信号) ----------
    # Role: core driving force — stay standing
    # Mathematical form: symmetric penalty around target height with deadzone
    target_height = 0.6
    height_error = body_z - target_height

    # Symmetric penalty: penalizes both being too low (collapsing) and too high (jumping)
    # Deadzone of +/-0.05 to allow natural body oscillation
    height_deviation = abs(height_error) - 0.05
    if height_deviation < 0.0:
        height_deviation = 0.0

    # Linear penalty beyond deadzone, capped to avoid explosion
    height_penalty = 4.0 * min(height_deviation, 0.4)
    height_management = -height_penalty

    # ---------- Component 2: Foot Contact Regularization (步态引导) ----------
    # Role: gait regularization — encourage rhythmic foot contact patterns
    # Use sigmoid on contact forces to convert to ~[0,1] contact indicator
    # Then reward having at least two feet on ground (diagonal pairs preferred)

    contact_threshold = 50.0  # minimum contact force to consider "grounded"
    sensitivity = 0.05

    # Simple sigmoid activation for each foot
    fl_contact = 1.0 / (1.0 + (2.718281828 ** (-sensitivity * (fl_foot_z - contact_threshold))))
    fr_contact = 1.0 / (1.0 + (2.718281828 ** (-sensitivity * (fr_foot_z - contact_threshold))))
    rl_contact = 1.0 / (1.0 + (2.718281828 ** (-sensitivity * (rl_foot_z - contact_threshold))))
    rr_contact = 1.0 / (1.0 + (2.718281828 ** (-sensitivity * (rr_foot_z - contact_threshold))))

    # Total ground contact (0 = no feet, 4 = all feet on ground)
    total_contact = fl_contact + fr_contact + rl_contact + rr_contact

    # Reward: prefer 2-3 feet on ground (walking), penalize 0-1 (falling) or 4 (standing still)
    # Use Gaussian-like reward centered at 2.5 (dynamic gait)
    contact_target = 2.5
    contact_deviation = abs(total_contact - contact_target)
    contact_reward = 1.5 * (2.718281828 ** (-2.0 * (contact_deviation ** 2)))

    # ---------- Component 3: Body Stability (姿态约束) ----------
    # Role: stability constraint — keep body upright and suppress roll/pitch
    # Compute body "up" vector from quaternion (projection of body z-axis onto world z-axis)
    body_up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)

    # Reward closeness to 1.0 (perfectly upright)
    tilt_penalty = 1.5 * (1.0 - body_up_z)

    # Also penalize angular velocity (roll + pitch rate)
    angular_rate_penalty = 0.3 * (abs(roll_rate) + abs(pitch_rate))

    stability_cost = tilt_penalty + angular_rate_penalty

    # ---------- Component 4: Forward Motivation (弱前进信号) ----------
    # Role: weak forward bias — gentle encouragement to move forward
    forward_encouragement = 0.3 * forward_vel

    # ---------- Component 5: Lateral Drift Penalty (侧向抑制) ----------
    # Role: suppress side-to-side sway
    lateral_penalty = 0.6 * abs(lateral_vel)

    # ---------- Assemble total reward ----------
    total_reward = (
        height_management
        + contact_reward
        - stability_cost
        + forward_encouragement
        - lateral_penalty
    )

    # ---------- Components dictionary ----------
    components = {
        'height_management': height_management,
        'contact_reward': contact_reward,
        'stability_cost': -stability_cost,
        'forward_encouragement': forward_encouragement,
        'lateral_penalty': -lateral_penalty
    }

    return float(total_reward), components
```