def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- distance to target (target at origin (0,0)) ---
    dist_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress = dist_prev - dist_next                # >0 when moving closer to target

    # --- velocity components ---
    vx = next_obs[2]
    vy = next_obs[3]

    # vertical velocity penalty, amplified when close to the ground (y near 0)
    altitude_factor = 1.0 / (1.0 + 10.0 * abs(next_obs[1]))
    vel_vertical_penalty = -0.02 * (vy ** 2) * altitude_factor

    # horizontal velocity penalty (small constant weight)
    vel_horizontal_penalty = -0.01 * (vx ** 2)

    velocity_penalty = vel_vertical_penalty + vel_horizontal_penalty

    # --- orientation penalty: body angle and angular velocity ---
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.02 * (next_obs[5] ** 2)

    # --- contact guidance: state-value with comprehensive settling gates ---
    # Reverting from iter_6's transition-based approach (which caused bouncing
    # exploitation: ep_sum=9810, active=12%) back to state-value (iter_1/5).
    #
    # Key L2 change from iter_1/5: the settling gate now checks total speed
    # (vx AND vy, not just |vy|) plus body angle. This closes two exploitation
    # loopholes:
    #   1. Horizontal sliding: iter_1/5 only gated |vy|, so agent could slide
    #      sideways (vx≠0, vy≈0) and collect full contact reward.
    #   2. Tilted stance: iter_1/5 had no angle gate on contact, so agent could
    #      stay tilted on the platform collecting reward without settling.
    #
    # With all three gates (proximity + speed + angle) simultaneously tight,
    # the agent only earns contact reward when it is truly settled — which
    # should trigger body_not_awake_or_settled termination within a few steps.
    contact_val = float(next_obs[6] + next_obs[7])   # 0.0, 1.0, or 2.0
    proximity_factor = max(0.0, 1.0 - dist_next / 5.0)
    speed = (vx**2 + vy**2) ** 0.5
    velocity_ok = max(0.0, 1.0 - speed / 0.5)       # speed < 0.5 for any reward
    angle_ok = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)  # |angle| < 0.3 for any reward
    contact_guidance = 0.3 * contact_val * proximity_factor * velocity_ok * angle_ok

    # --- combine all components ---
    # Progress weight=0.5 matches iter_1 (score=171, best ever).
    # Weights >0.5 (iter_4/5) caused rushing-and-crashing or lingering.
    total_reward = (0.5 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 0.5 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return total_reward, components