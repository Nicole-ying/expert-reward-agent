## Component-by-Component Analysis

### (1) `proximity_reward` — **ROOT CAUSE OF EXPLOIT, MUST REMOVE**
- ep_sum_mean=1175, magnitude_share=**89%**, active=100%
- Checkpoint trend: cp80%=687 → cp90%=**1262** (spike!), while contact_bonus and landing_quality at cp80% were 90 and 142 but dropped to **zero** at cp90%.
- **Diagnosis**: The agent discovered at cp80% it could land (score=160), then at cp90% discovered a better strategy: **hover near the pad forever** to collect 1.5 per step × 1000 steps ≈ 1500, instead of landing (which terminates the episode, ending the reward stream). The proximity_reward dwarfs every other signal (89% share), so the agent maximizes it by never landing. This is the classic "hovering exploit" — the exponential proximity gives unbounded cumulative reward if you never terminate.
- **Action**: **Completely remove this component.** Replace with nothing — the approach_reward already drives the agent to the pad.

### (2) `approach_reward` — **Healthy but undersized**
- ep_sum_mean=18.9, magnitude_share=1.8%, active=100%
- Checkpoint trend: rose from -131 to +24 over training — learning is happening
- **Diagnosis**: This is a proper potential-based signal. It's bounded (max = w × initial_distance) and cannot be exploited. At w=20 it should produce ~100 per episode (initial dist ~5), but proximity_reward steals all the credit. It works correctly but is overpowered.
- **Action**: Reduce coefficient from 20 to 10 (since there's no competing proximity signal), keeping it as the primary approach driver.

### (3) `vel_penalty` — **Functional but needs better gating**
- ep_sum_mean=-86.2, magnitude_share=6.5%, active=100%
- Checkpoint trend: steadily increasing in magnitude (-49 to -90), tracking proximity.
- **Diagnosis**: The near/far gating with exponential works but the far penalty (0.02) is too small to matter and the near penalty (1.0 × near_gate) adds unnecessary complexity. The exponential gating also creates weird interactions.
- **Action**: Simplify to `1/(1+5*dist)` gating with a single coefficient, making it strong near pad and negligible far away.

### (4) `orientation_cost` — **Dead signal, nearly invisible**
- ep_sum_mean=-0.94, magnitude_share=0.1%, active=100%
- Checkpoint trend: flat at -0.7 to -1.5 across all checkpoints — zero learning progression.
- **Diagnosis**: The coefficients (0.3, 0.05) are too small relative to the proximity reward. Even at cp80% when the agent was landing, orientation_cost was only -1.22.
- **Action**: Increase to w_angle=0.5, w_angvel=0.1, use absolute values instead of squares for linear penalty.

### (5) `contact_bonus` — **Works when agent lands, but irrelevant otherwise**
- ep_sum_mean=2.0, magnitude_share=0.2%, active=0.2%
- Checkpoint: zero except cp80%=90 and cp100%=7. Proves the agent only touches ground when landing.
- **Diagnosis**: Contact is a valid signal but can't compete with proximity. At w=0.5 per foot, it gives 1.0/step when both feet touch — but the agent prefers 1.5/step from proximity without needing to land.
- **Action**: Increase to w=1.0 per foot, keep as supporting signal.

### (6) `landing_quality` — **Good signal, can't compete with proximity**
- ep_sum_mean=2.5, magnitude_share=0.2%, active=0.2%
- Checkpoint: zero except cp80%=142. Same pattern as contact.
- **Diagnosis**: This is a solid signal — it only fires when both feet are down and rewards stability. But it can't compete with 1500 from hovering.
- **Action**: Increase to w=5.0, keep the formula.

### (7) `fuel_cost` — **Stable, functional**
- ep_sum_mean=-29.7, magnitude_share=2.3%, active=99%
- **Diagnosis**: Works fine, steady across checkpoints. No changes needed.

---

## New Design: Replace proximity exploit with one-shot landing success bonus

The core fix: **remove proximity_reward entirely** and add a **one-shot success bonus** that fires only on the transition to a stable two-foot landing near the pad. This gives the agent a large terminal reward for landing that exceeds anything it could earn by hovering. With no continuous proximity signal, there is zero incentive to hover.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: potential-based approach shaping as primary driver,
    one-shot landing success bonus for terminal incentive, proximity-gated
    velocity damping, and supporting contact/landing-quality signals.
    No continuous proximity reward — prevents hovering exploit.
    """
    x_prev, y_prev = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    prev_left = obs[6]
    prev_right = obs[7]

    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist = (x ** 2 + y ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5

    # 1. Potential-based approach shaping — primary driver toward the pad.
    # Bounded: total per episode = w_approach * initial_distance. Cannot be exploited.
    w_approach = 10.0
    approach_reward = w_approach * (dist_prev - dist)

    # 2. One-shot landing success bonus.
    # Fires only on the transition from not-landed to landed, near the pad,
    # with low speed and upright attitude. Gives a large terminal incentive
    # that exceeds any possible cumulative reward from hovering.
    prev_both = prev_left * prev_right
    now_both = left_contact * right_contact
    just_landed = float(now_both > 0.5 and prev_both < 0.5)
    near_pad = float(dist < 0.5)
    soft_landing = float(speed < 1.5)
    upright = float(abs(angle) < 0.5)
    w_success = 200.0
    success_bonus = w_success * just_landed * near_pad * soft_landing * upright

    # 3. Landing quality — continuous reward while settled on both feet.
    # Rewards stability: low speed + upright + both feet down.
    w_landing = 5.0
    landing_quality = w_landing * (left_contact * right_contact) / (1.0 + abs(angle) + speed)

    # 4. Proximity-gated velocity damping.
    # Strong near the pad (enforces soft landing), negligible far away.
    near_gate = 1.0 / (1.0 + 5.0 * dist)
    w_vel = 0.5
    vel_penalty = -w_vel * near_gate * speed

    # 5. Orientation cost — keep the lander upright.
    w_angle = 0.5
    w_angvel = 0.1
    orientation_cost = -w_angle * abs(angle) - w_angvel * abs(angvel)

    # 6. Per-foot contact bonus — encourage ground contact.
    w_contact = 1.0
    contact_bonus = w_contact * (left_contact + right_contact)

    # 7. Fuel efficiency — light pressure against unnecessary engine use.
    w_fuel = 0.03
    fuel_cost = -w_fuel * float(action != 0)

    total_reward = (
        approach_reward + success_bonus + landing_quality +
        vel_penalty + orientation_cost + contact_bonus + fuel_cost
    )

    components = {
        "approach_reward": approach_reward,
        "success_bonus": success_bonus,
        "landing_quality": landing_quality,
        "vel_penalty": vel_penalty,
        "orientation_cost": orientation_cost,
        "contact_bonus": contact_bonus,
        "fuel_cost": fuel_cost,
    }
    return float(total_reward), components
```