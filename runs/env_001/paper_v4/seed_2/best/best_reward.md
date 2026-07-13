`evidence`: Final-policy score=-17.9, all 20 episodes truncated at 1000 steps with zero early terminations; position_proximity dominates reward (episode_sum_mean=749.3, 99.7% magnitude share) while contact_completion has active_rate=0%, contributing literally zero reward in all evaluation episodes. The agent collects steady state-reward by hovering near the target but never lands.

`behavior_diagnosis`: The agent has settled into a proxy plateau — it hovers at ~0.33 units from the target, collecting ~0.75 position_proximity per step (≈750 total per episode), with no incentive to close the final gap and actually land. The contact_completion component, a product of 6 factors including two binary leg-contact signals, remains exactly zero throughout because the product collapses to zero whenever either leg is off the ground — which is virtually always during hovering.

`signal_completeness`: The reward has the right conceptual ingredients (proximity, velocity, orientation, contact) but the contact_completion is structurally unreachable (product of 6 factors with binary gates yields zero 100% of the time). This is a classic sparse completion proxy failure: the landing bonus provides no gradient anywhere in the state space the agent occupies. The position_proximity state reward (not improvement-based) further enables the hovering exploit since constant-distance loitering pays indefinitely.

`selected_level`: **Level 2** — contact_completion has 0% active_rate, which is a structural reachability failure. No amount of coefficient scaling (Level 1) can fix a signal that never fires. The product form creates a dead zone covering the entire hovering regime. This matches the `product_to_noncollapsing_joint` transformation pattern.

`selected_intervention`: Transform **contact_completion** from a product of 6 factors into a sum of 5 independently contributing bounded factors. Each factor (x_proximity, y_proximity, velocity, angle, leg_contact) now contributes its own [0,1] gradient, so the agent receives partial credit whenever any landing condition improves — even before both legs touch simultaneously. The product collapse is eliminated.

`falsifiable_hypothesis`: Breaking the product into a sum gives the agent nonzero gradient for each landing sub-skill independently (reduce velocity, straighten angle, approach center, make leg contact). The agent should now receive incremental reward as it improves any single condition, creating a bridge from hovering behavior toward full landing. Active rate should jump from 0% to near 100%, and external score should improve as the agent begins landing at least occasionally instead of hovering for all 1000 steps.

`expected_next_round`: contact_completion active_rate should rise from 0% to near 100%; episode_sum_mean for contact_completion should become positive; external score should increase (less negative); some episodes may begin terminating before 1000 steps via the `body_not_awake_or_settled` success condition if the agent starts landing.

`main_risk`: The sum structure could reward partial conditions (e.g., being upright in mid-air or having one leg down) without requiring full landing, potentially creating a new proxy where the agent optimizes a subset of factors while still never completing the task. The combined magnitude (5 factors × max 1.0 = max 5.0 per step) may also be too strong relative to position_proximity (max 1.0), causing excessive focus on landing conditions at the expense of actually reaching the pad.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant states from the next observation (post-transition)
    x_next = next_obs[0]
    y_next = next_obs[1]
    vx_next = next_obs[2]
    vy_next = next_obs[3]
    angle_next = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- Component A: Position Proximity (Main Learning Signal) ----------
    # Dense, bounded reward encouraging the craft to reach (0,0).
    # Uses soft inverse distance so that reward saturates near the target.
    dist = (x_next**2 + y_next**2) ** 0.5
    pos_reward = 1.0 / (1.0 + dist)

    # ---------- Component B: Soft Landing Velocity (Bounded Penalty) ----------
    # Penalise high horizontal/vertical velocity only when close to the ground.
    # Activation gate: high when |y| is small (near platform level), negligible in high sky.
    activation = 1.0 / (1.0 + 10.0 * abs(y_next))
    vel_penalty = -0.1 * activation * (vx_next**2 + vy_next**2)

    # ---------- Component C: Stable Orientation (Quadratic Penalty) ----------
    # Light penalty on body tilt to encourage horizontal attitude.
    angle_penalty = -0.5 * (angle_next ** 2)

    # ---------- Component D: Contact Completion (Non-Collapsing Joint Sum) ----------
    # Each factor is an independent bounded [0,1] measure of a desired landing condition.
    # Using a SUM instead of a product ensures each condition provides its own gradient,
    # so the agent receives partial credit even before all conditions are simultaneously met.
    k_x = 5.0
    factor_x = 1.0 / (1.0 + k_x * x_next**2)

    k_y = 5.0
    factor_y = 1.0 / (1.0 + k_y * y_next**2)

    # Combined velocity factor: rewards low total speed.
    k_v = 2.0
    factor_v = 1.0 / (1.0 + k_v * (vx_next**2 + vy_next**2))

    k_angle = 3.0
    factor_angle = 1.0 / (1.0 + k_angle * angle_next**2)

    # Contact factor: 0.5 per leg, max 1.0 when both legs touch.
    factor_contact = 0.5 * left_contact + 0.5 * right_contact

    # Sum of independent factors: range [0, 5]. Each factor contributes partial credit.
    # Coefficient 0.4 keeps max at 2.0 per step — strong enough to incentivize landing
    # without dwarfing the position_proximity signal.
    contact_reward = 0.4 * (factor_x + factor_y + factor_v + factor_angle + factor_contact)

    # ---------- Total Reward ----------
    total_reward = pos_reward + vel_penalty + angle_penalty + contact_reward

    components = {
        'position_proximity': pos_reward,
        'soft_landing_velocity': vel_penalty,
        'stable_orientation': angle_penalty,
        'contact_completion': contact_reward,
    }

    return float(total_reward), components
```