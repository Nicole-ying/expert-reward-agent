def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v10 — Replace contact-gated soft_landing_bonus with height-gated landing_readiness.

    Diagnosis from v9:
    - soft_landing_bonus active_rate=3.4%, ep_sum=2.3 — dead signal across all checkpoints.
      The agent never discovers landing because the contact gate is a terminal event:
      the agent must already have reached the pad (and touched it) to receive any signal.
    - Graduated contact (0.5x single-leg) failed to fix this: agent crashes before reaching
      the pad, so single-leg contact is just as sparse as dual-leg contact in practice.
    - The fundamental gap: there is NO continuous signal bridging proximity_progress
      (all-phase navigation) and soft_landing_bonus (contact-only terminal event).

    Single semantic change (L2):
    - Replace contact gate with continuous height gate: `near_ground = 1/(1+5*height)`.
      Height is always observable and provides gradient at any altitude, unlike contact
      which is only informative at touchdown. This creates a "slow down near the ground"
      pressure that the agent can discover long before it ever touches the pad.
    - Proximity kept at 30.0 (v8-level, proven non-crash-diving) because the new
      height-gated signal now provides additional downward motivation.
    """

    # ---------- proximity_progress: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 30.0 * (old_dist - new_dist)

    # ---------- attitude_penalty: keep tilt and spin under control ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- landing_readiness: height-gated continuous quality signal ----------
    # Key change: binary contact gate → continuous height gate.
    # near_ground provides gradient at any altitude, bridging approach → landing.
    height = next_obs[1]                                 # vertical distance to pad surface
    near_ground = 1.0 / (1.0 + 5.0 * height)             # 1.0 at h=0, 0.5 at h=0.2, 0.17 at h=1

    v_mag = abs(next_obs[2]) + abs(next_obs[3])
    velocity_quality = 1.0 / (1.0 + 5.0 * v_mag)        # → 1.0 at v=0, → 0.17 at v=1
    angle_quality    = 1.0 / (1.0 + 5.0 * abs(angle))   # → 1.0 at θ=0, → 0.17 at |θ|=1

    landing_readiness = 3.0 * near_ground * velocity_quality * angle_quality

    # ---------- total reward ----------
    total_reward = proximity_progress + attitude_penalty + landing_readiness

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "landing_readiness": landing_readiness
    }
    return float(total_reward), components