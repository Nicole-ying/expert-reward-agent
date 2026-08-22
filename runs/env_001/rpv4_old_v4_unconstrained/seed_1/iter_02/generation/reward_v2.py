def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v2 — restructured to fix hovering and landing-sparsity problems.

    Core changes from v1:
    - Added altitude_reward: continuous incentive to descend (fixes hovering).
    - Added fuel_penalty + step_penalty: make lingering near the pad costly.
    - contact_reward now uses continuous n_contact (0/1/2) not binary product,
      so partial contact provides gradient. Per-step value is modest to prevent
      bounce-exploit (fuel+step costs outweigh repeated tiny contact gains).
    - Reduced proximity_progress weight — was 82.9% magnitude share, too dominant.
    - All components balanced so net reward near pad w/o landing is negative.
    """
    # ── current state ──────────────────────────────────────────────
    x  = next_obs[0]          # horizontal position relative to pad center
    y  = next_obs[1]          # vertical position (height above pad)
    vx = next_obs[2]          # horizontal velocity
    vy = next_obs[3]          # vertical velocity
    angle      = next_obs[4]  # body tilt angle
    angular_vel = next_obs[5] # angular velocity
    left_contact  = next_obs[6]  # 1.0 if left leg on ground
    right_contact = next_obs[7]  # 1.0 if right leg on ground

    ox = obs[0]               # previous x
    oy = obs[1]               # previous y

    # ── 1. proximity_progress ──────────────────────────────────────
    # Reward closing distance to the pad centre. Reduced weight vs v1
    # (was 30.0, now 5.0) so it doesn't drown out landing signals.
    old_dist = (ox ** 2 + oy ** 2) ** 0.5
    new_dist = (x  ** 2 + y  ** 2) ** 0.5
    proximity_progress = 5.0 * (old_dist - new_dist)

    # ── 2. altitude_reward ─────────────────────────────────────────
    # Continuous reward for being at low height.  Creates the *gradient*
    # to descend that v1 was missing.  Max 0.15/step at y=0.
    altitude_reward = 0.15 / (1.0 + 5.0 * (y ** 2) ** 0.5)

    # ── 3. speed_penalty ───────────────────────────────────────────
    # Discourage high speed generally; extra weight when close to ground
    # (ground_proximity factor) so the agent slows down for landing.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    ground_proximity = 1.0 / (1.0 + 5.0 * (y ** 2) ** 0.5)
    speed_penalty = -0.12 * speed * (1.0 + ground_proximity)

    # ── 4. attitude_penalty ────────────────────────────────────────
    # Keep body level and non-rotating.  Same form as v1, slightly
    # reduced angular-vel coefficient.
    attitude_penalty = -0.3 * (angle ** 2) - 0.08 * (angular_vel ** 2)

    # ── 5. contact_reward ──────────────────────────────────────────
    # Reward leg-ground contact (0, 1, or 2 legs), scaled by quality:
    # low speed and upright attitude.  *Not* binary — partial credit
    # for single-leg contact provides a learning gradient.
    n_contact = left_contact + right_contact          # 0.0, 1.0, or 2.0
    contact_quality = 1.0 / ((1.0 + 3.0 * speed) * (1.0 + 2.0 * (angle ** 2) ** 0.5))
    contact_reward = 1.5 * n_contact * contact_quality

    # ── 6. fuel_penalty ────────────────────────────────────────────
    # Action 0 = no engine; actions 1,2,3 fire at least one engine.
    # Discourages unnecessary thrust and makes hovering expensive.
    fuel_penalty = -0.25 if action != 0 else 0.0

    # ── 7. step_penalty ────────────────────────────────────────────
    # Slight time pressure — makes indefinite hovering net-negative.
    step_penalty = -0.02

    # ── total ──────────────────────────────────────────────────────
    total_reward = (
        proximity_progress
        + altitude_reward
        + speed_penalty
        + attitude_penalty
        + contact_reward
        + fuel_penalty
        + step_penalty
    )

    components = {
        "proximity_progress": proximity_progress,
        "altitude_reward":    altitude_reward,
        "speed_penalty":      speed_penalty,
        "attitude_penalty":   attitude_penalty,
        "contact_reward":     contact_reward,
        "fuel_penalty":       fuel_penalty,
        "step_penalty":       step_penalty,
    }

    return float(total_reward), components