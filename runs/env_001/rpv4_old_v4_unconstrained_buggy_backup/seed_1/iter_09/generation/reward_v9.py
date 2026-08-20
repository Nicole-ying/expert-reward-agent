def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v9 — Graduated contact gate for soft_landing_bonus.

    Diagnosis from v8:
    - soft_landing_bonus active_rate = 3.4% — far too sparse for reliable learning.
      The binary gate (both legs must contact simultaneously) means the agent almost
      never experiences the landing reward signal.
    - Checkpoint trend shows the agent discovers landing mid-training (cp50-60%)
      but then forgets it (cp70-100%) — classic sparse-reward collapse as exploration
      decays. Agent retreats to safe hovering where proximity_progress gives steady
      ~25/episode without risk.
    - v4 proved the skeleton works (score=135, 19/20 landed) but needed proximity=40
      to force approach. At proximity=30 the approach gradient is too weak.

    Single semantic change (L2):
    - soft_landing_bonus: binary both_contact → graduated contact_level.
      Single-leg contact now earns half credit (0.5×), creating a stepping-stone
      gradient: "touch the pad" → "land with both legs." This directly addresses
      the sparsity problem by increasing the effective active_rate.
    - Proximity nudged 30→35 and bonus coeff 4.0→5.0 to support the same semantic
      direction (making landing more discoverable and rewarding).
    """

    # ---------- proximity_progress: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 35.0 * (old_dist - new_dist)

    # ---------- attitude_penalty: keep tilt and spin under control ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- soft_landing_bonus: graduated contact for progressive learning ----------
    left_contact = 1.0 if next_obs[6] > 0.5 else 0.0
    right_contact = 1.0 if next_obs[7] > 0.5 else 0.0
    contact_level = 0.5 * left_contact + 0.5 * right_contact   # 0.0, 0.5, or 1.0

    v_mag = abs(next_obs[2]) + abs(next_obs[3])
    velocity_quality = 1.0 / (1.0 + 5.0 * v_mag)       # → 1.0 at v=0, → 0.17 at v=1
    angle_quality    = 1.0 / (1.0 + 5.0 * abs(angle))  # → 1.0 at θ=0, → 0.17 at |θ|=1

    soft_landing_bonus = 5.0 * contact_level * velocity_quality * angle_quality

    # ---------- total reward ----------
    total_reward = proximity_progress + attitude_penalty + soft_landing_bonus

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components