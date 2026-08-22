def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 10): soft_landing_envelope (absolute per-step reward,
    exploited via hovering at 98.8% magnitude share) replaced with
    landing_approach — a delta-quality reward that only pays when the
    landing state actually improves.

    landing_approach = 20.0 * max(0, quality(next) - quality(prev))
    where quality = exp(-h/0.3) * exp(-5x^2) * exp(-(vx^2+vy^2)) * exp(-|angle|)

    This is bounded (~20 max per episode), cannot be exploited by hovering
    (steady state gives zero delta), and provides gradient across all four
    landing dimensions (height, centering, speed, uprightness).

    Unlike iter 6's landing_approach (which used only height+centering in a
    linear potential with scale 2.0 and negative deltas), this version:
      - includes velocity and angle in the quality function
      - uses scale 20.0 for stronger per-step signal
      - uses max(0, delta) to avoid penalizing exploration regression
    """
    # --- goal proximity: linear delta-distance for uniform gradients ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 4.0 * (prev_dist - next_dist)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 at ground, decays smoothly with height (k=1.5)
    height = max(next_obs[1], 0.0)
    gate = 2.718281828 ** (-height / 1.5)
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- landing approach: delta-quality reward ---
    # Quality function: multiplicative envelope peaking at 1.0 for the
    # ideal landing state (h=0, x=0, v=0, angle=0). Each factor in (0,1].
    def landing_quality(h, x, vx, vy, angle):
        h_safe = max(h, 0.0)
        h_factor = 2.718281828 ** (-h_safe / 0.3)
        x_factor = 2.718281828 ** (-5.0 * x**2)
        v_factor = 2.718281828 ** (-(vx**2 + vy**2))
        a_factor = 2.718281828 ** (-abs(angle))
        return h_factor * x_factor * v_factor * a_factor

    curr_quality = landing_quality(
        next_obs[1], next_obs[0], next_obs[2], next_obs[3], next_obs[4]
    )
    prev_quality = landing_quality(
        obs[1], obs[0], obs[2], obs[3], obs[4]
    )

    # Reward only improvement (delta > 0). Steady-state hovering gives zero.
    # Scale 20.0: max episode total ~20 (quality range ~0 -> ~1).
    landing_approach = 20.0 * max(0.0, curr_quality - prev_quality)

    total_reward = goal_proximity + vel_penalty + upright_penalty + landing_approach

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "landing_approach": landing_approach
    }
    return float(total_reward), components