def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v10: Stronger progress + descent incentive to fix hovering.

    ROOT CAUSE OF v9 HOVERING:
    progress_reward (0.5% share) was 100x weaker than proximity_reward (56.5% share).
    Agent could collect ~0.3/step proximity just by staying near pad, vs ~0.004/step
    from progress. Rational choice: hover indefinitely, occasionally land.

    v10 CHANGES:
    - progress_reward: 2.0 → 10.0 (5x), makes "get closer" the dominant per-step signal
    - proximity_reward: 0.5 → 0.35 max, reduces hovering incentive while keeping gradient
    - vel_damping: wider sigma (1.0), stronger coefficient (1.5), prevents crash dives
    - NEW descent_bonus: rewards downward velocity when horizontally aligned with pad.
      Directly punishes hovering: 0 when hovering, positive when descending.
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

    # ── 1. Progress: strong navigation drive (5x boost from v9) ──
    # v9: 2.0 gave ~2.5 ep_sum vs proximity's 289.5 → 100:1 ratio
    # v10: 10.0 targets ~12-15 ep_sum vs proximity's ~150 → 10:1 ratio
    progress_reward = 10.0 * (prev_dist - dist)

    # ── 2. Proximity: continuous gradient, moderated to reduce loitering ──
    # Reduced from 0.5 to 0.35 max. At center with contact: 0.35 + 0.6 = 0.95/step.
    # At dist=0.5: 0.35*exp(-0.25/0.8)=0.256. Far away (dist=2): ~0.002.
    proximity_reward = 0.35 * (2.718281828 ** (-dist ** 2 / 0.8))

    # ── 3. Orientation penalty: stable, proven, unchanged ──
    orientation_penalty = -0.15 * (angle ** 2) - 0.06 * (angvel ** 2)

    # ── 4. Velocity damping: wider activation + stronger coefficient ──
    # v9: sigma=0.5, coeff=0.5 → only active very close to pad, easily ignored
    # v10: sigma=1.0, coeff=1.5 → activates at dist=1.0 with 37% strength,
    #   creates natural braking gradient that counters descent_bonus near pad
    near_pad = 2.718281828 ** (-dist ** 2 / 1.0)
    safe_speed = 0.25
    excess_speed = max(0.0, speed - safe_speed)
    vel_damping = -1.5 * near_pad * excess_speed

    # ── 5. Descent incentive: reward downward velocity when above pad ──
    # Key anti-hovering signal. When agent is horizontally aligned and descending,
    # this adds 0.1-0.5/step. When hovering (vy≈0), this is ~0.
    # Above_pad gates on horizontal alignment: at |x|=0.5 → 61%, at |x|=1.0 → 14%.
    # Natural balancing with vel_damping near pad:
    #   dist=0.5, speed=0.8: vel_damping=-0.64 vs descent=+0.40 → net negative → slow down
    #   dist=1.0, speed=1.0: vel_damping=-0.41 vs descent=+0.50 → net positive → descend OK
    above_pad = 2.718281828 ** (-x ** 2 / 0.5)
    descent_bonus = 0.5 * above_pad * max(0.0, -vy)

    # ── 6. Contact: ground interaction discoverability, unchanged ──
    contact_reward = 0.3 * (left_contact + right_contact)

    # ── 7. Action efficiency: gentle penalty, unchanged ──
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
        + descent_bonus
        + contact_reward
        + action_reward
    )

    components = {
        "progress_reward": progress_reward,
        "proximity_reward": proximity_reward,
        "orientation_penalty": orientation_penalty,
        "vel_damping": vel_damping,
        "descent_bonus": descent_bonus,
        "contact_reward": contact_reward,
        "action_reward": action_reward,
    }
    return float(total_reward), components