def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v9: Continuous proximity replaces discrete landing bonus.

    ROOT CAUSE OF v4-v8 FAILURES:
    Every iteration with a discrete "landing bonus" (per-step or one-time) was exploited:
    - v4: agent sat on pad collecting 50/step → landing_bonus=2627, 97.4% magnitude
    - v8: agent bounced to re-trigger one-time bonus → landing_bonus=922, 94.7% magnitude
    Discrete events are inherently exploitable — the agent always finds a way to re-trigger them.

    v9 APPROACH:
    - NO discrete landing bonus. NO touchdown detection. NO "just_touched_down" triggers.
    - proximity_reward: continuous Gaussian centered on pad (0,0). Always active, smooth
      gradient, max 0.5/step at center. Cannot be exploited: bouncing AWAY reduces reward.
    - On-pad: proximity + contact ≈ 1.0/step. Far away: ≈ 0.05/step. 20:1 natural gradient.
    - All components are continuous → no exploit cliff, no collapse-recovery cycle.
    """
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    px = obs[0]
    py = obs[1]

    # ── Derived quantities ──
    dist = (x ** 2 + y ** 2) ** 0.5
    prev_dist = (px ** 2 + py ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5

    # ── 1. Progress: delta-distance (navigation signal, proven in all iterations) ──
    progress_reward = 2.0 * (prev_dist - dist)

    # ── 2. Proximity: continuous Gaussian pull toward pad center ──
    # Replaces the exploitable discrete landing bonus.
    # max 0.5 at (0,0), decays smoothly: 0.37 at dist=0.5, 0.14 at dist=1.0, ~0 at dist=2.0
    # Always active → agent always feels gradient toward pad. Bouncing reduces reward.
    proximity_reward = 0.5 * (2.718281828 ** (-dist ** 2 / 0.8))

    # ── 3. Orientation penalty: stable, always active, proven scales ──
    orientation_penalty = -0.15 * (angle ** 2) - 0.06 * (angvel ** 2)

    # ── 4. Velocity damping near pad: prevent hard landings ──
    near_pad = 2.718281828 ** (-dist ** 2 / 0.5)
    safe_speed = 0.25
    excess_speed = max(0.0, speed - safe_speed)
    vel_damping = -0.5 * near_pad * excess_speed

    # ── 5. Contact: reward ground interaction for discoverability ──
    contact_reward = 0.3 * (left_contact + right_contact)

    # ── 6. Action efficiency: gentle penalty, glide is free ──
    if action == 0:
        action_reward = 0.0
    elif action == 2:          # main engine — highest cost
        action_reward = -0.04
    else:                      # orientation engines (1, 3) — moderate cost
        action_reward = -0.02

    # ── Aggregate ──
    total_reward = (
        progress_reward
        + proximity_reward
        + orientation_penalty
        + vel_damping
        + contact_reward
        + action_reward
    )

    components = {
        "progress_reward": progress_reward,
        "proximity_reward": proximity_reward,
        "orientation_penalty": orientation_penalty,
        "vel_damping": vel_damping,
        "contact_reward": contact_reward,
        "action_reward": action_reward,
    }
    return float(total_reward), components