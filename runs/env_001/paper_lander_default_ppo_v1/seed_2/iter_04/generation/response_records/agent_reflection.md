# Response Record

`evidence`: 20/20 episodes terminate early (~68.75 steps), all crashes; `soft_landing_proxy` active_rate=0.3% (<1% threshold, effectively unreachable); `distance_progress` is positive but small (episode_sum_mean=2.24); `stability_penalty` ratio to progress is ~0.35 (scale fix from last round worked, no longer dominating). Score improved from -109.88 to -98.51 after last scale fix, but gap remains huge.

`behavior_diagnosis`: Agent makes slow progress toward target but crashes in every episode without learning to slow down, orient, or make leg contact. The sole landing signal fires only 0.3% of steps, leaving approach/landing behavior unguided.

`signal_completeness`: Progress and stability signals exist but the landing guidance is effectively absent — a binary sparse proxy at 0.3% active rate cannot provide usable gradient. The agent lacks any intermediate feedback for deceleration, upright orientation, or proximity-to-pad during the final approach.

`selected_level`: Level 2 — `sparse_to_dense` transformation on the landing component, triggered by active_rate=0.3% falling below the ~1% threshold for sparse completion proxies.

`selected_intervention`: Replace binary `soft_landing_proxy` with a dense continuous `landing_approach` reward using geometric mean of three bounded [0,1] factors (proximity, velocity, angle) multiplied by a contact factor (0.3 without legs, 1.0 with both legs), scaled by 0.3.

`falsifiable_hypothesis`: The geometric-mean form provides non-zero gradient for partial approach states (e.g., close but fast, or close/slow but tilted), so the policy can learn deceleration and orientation adjustments before leg contact becomes necessary. Contact_factor=0.3 base ensures signal even before legs touch, while the 3.3× boost for both-leg contact preserves incentive for the final step.

`expected_next_round`: `landing_approach` active_rate should rise substantially above 0.3%; episode length may increase (agent survives longer while learning approach); score should improve if the gradient successfully guides toward landing conditions. Crashes should decrease.

`main_risk`: The dense state-value reward could create a new proxy optimum — hovering near the pad without making leg contact, collecting ~0.07/step indefinitely. Mitigated by distance_progress giving zero during hover, stability_penalty providing mild deterrent, and the 3.3× contact multiplier incentivizing the final contact step. If hovering emerges, next step is `state_to_improvement` or tightening contact_factor base.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract current and next observation variables
    prev_x, prev_y = obs[0], obs[1]
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    omega = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Progress signal: reward getting closer to target, penalize moving away
    prev_dist = (prev_x**2 + prev_y**2) ** 0.5
    curr_dist = (x**2 + y**2) ** 0.5
    reward_dist = 2.0 * (prev_dist - curr_dist)

    # 2. Stability constraint: light penalty on high speeds and large angles
    reward_stability = (
        -0.008 * abs(vx) -
        0.008 * abs(vy) -
        0.008 * abs(angle) -
        0.008 * abs(omega)
    )

    # 3. Dense landing approach guidance: geometric mean of bounded factors
    #    Replaces sparse binary proxy (old active_rate=0.3%, unreachable)
    prox = max(0.0, 1.0 - curr_dist / 2.0)
    vel_mag = abs(vx) + abs(vy)
    vel_factor = max(0.0, 1.0 - vel_mag / 2.0)
    ang_factor = max(0.0, 1.0 - abs(angle) / 0.5)

    # Geometric mean prevents product collapse while preserving joint requirement
    approach_quality = (prox * vel_factor * ang_factor) ** (1.0 / 3.0)

    # Contact factor: partial credit without legs, full credit with both
    both_legs = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_factor = 0.3 + 0.7 * both_legs

    reward_landing = 0.3 * approach_quality * contact_factor

    total_reward = reward_dist + reward_stability + reward_landing

    components = {
        "distance_progress": reward_dist,
        "stability_penalty": reward_stability,
        "landing_approach": reward_landing
    }

    return float(total_reward), components
```
