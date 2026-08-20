# Response Record

`evidence`: landing_quality_reward dominates with episode_sum_mean=403.49 (95.4% magnitude share) at 65.3% active_rate across ~443-step episodes — the continuous state-based proxy is being farmed. progress_reward contributes only 12.38 (3.2% magnitude). terminated=17/20 shows most episodes do eventually land, but the 3 truncated and score range [88.8, 278.97] suggest inconsistent efficiency. The previous round introduced this proxy as a "continuous bounded proxy replacing sparse binary bonus" — it succeeded at providing gradient but introduced farming.

`behavior_diagnosis`: The agent reaches the landing zone (d<0.3) and then lingers for ~289 steps (65.3% of 443), collecting per-step state-based rewards for proximity × quality. Once in a good state, staying there pays indefinitely — the agent has no incentive to complete touchdown quickly. The persistent state-value form directly enables this stationary reward collection.

`signal_completeness`: Progress and stability signals exist with sound mathematical forms but are numerically overwhelmed. The landing quality signal is structurally flawed: rewarding state occupation rather than state improvement makes it a farmable proxy, not a completion guide.

`selected_level`: Level 2 — persistent_to_transition_event / state_to_improvement. The evidence (65.3% active_rate, 95.4% magnitude share on a state-value component) directly matches the "persistent state farming" pattern. Pure coefficient reduction (Level 1) would weaken but not eliminate the structural incentive to linger — any positive per-step reward for occupying a good state remains farmable.

`selected_intervention`: Transform landing_quality_reward from persistent state-based (`2.0 * prox * Σqualities`) to improvement-based (`15.0 * max(0, quality(next_obs) − quality(obs))`). Quality is computed identically (prox × sum of bounded vel/angle/contact factors), but reward fires only on positive delta. Entering the zone from outside counts as full improvement since q_curr=0 when d_curr≥0.3.

`falsifiable_hypothesis`: Removing the per-step state reward eliminates the farming plateau — the agent can no longer collect reward by maintaining a good state. The active_rate should drop sharply (only improvement steps fire), magnitude_share should fall below 50%, and episode_length should decrease as the agent lands more directly. The progress_reward will become the primary continuous guide.

`expected_next_round`: landing_quality_reward episode_sum_mean should collapse from ~403 to ~20–50; active_rate should drop far below 65%; progress_reward magnitude_share should rise above 30%; episode_length should decrease; terminated rate should stay ≥15/20; total score may drop below 200 initially as proxy inflation is removed.

`main_risk`: `max(0, delta)` only rewards improvement, never penalizes degradation — this could incentivize oscillatory "degrade-then-improve" cycles within the landing zone. The environment's body_not_awake_or_settled termination should cap this, but the 3 truncated episodes may increase if the agent oscillates without settling.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Compute Euclidean distance to target (landing pad center)
    d_curr = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    d_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # Main learning signal: progress towards the target
    progress = d_curr - d_next
    progress_clipped = max(-0.5, min(0.5, progress))
    progress_reward = 10.0 * progress_clipped

    # Stability penalty: discourage high speed, large angle and angular velocity
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    stability_penalty = -0.05 * abs(vx) - 0.02 * abs(vy) - 0.05 * abs(angle) - 0.01 * abs(angular_vel)

    # Landing quality improvement reward
    # Replaces persistent state-based proxy with improvement-based reward.
    # Only rewards getting better at landing, not occupying a good state.
    landing_zone = 0.3

    if d_next < landing_zone:
        # Quality of current state (before step) — zero if outside landing zone
        if d_curr < landing_zone:
            prox_c = max(0.0, 1.0 - d_curr / landing_zone)
            speed_c = (obs[2]**2 + obs[3]**2)**0.5
            vel_q_c = max(0.0, 1.0 - speed_c / 0.5)
            angle_q_c = max(0.0, 1.0 - abs(obs[4]) / 0.3)
            contact_q_c = 0.5 * (obs[6] + obs[7])
            q_curr = prox_c * (vel_q_c + angle_q_c + 0.5 * contact_q_c)
        else:
            q_curr = 0.0

        # Quality of next state (after step)
        prox_n = max(0.0, 1.0 - d_next / landing_zone)
        speed_n = (next_obs[2]**2 + next_obs[3]**2)**0.5
        vel_q_n = max(0.0, 1.0 - speed_n / 0.5)
        angle_q_n = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)
        contact_q_n = 0.5 * (next_obs[6] + next_obs[7])
        q_next = prox_n * (vel_q_n + angle_q_n + 0.5 * contact_q_n)

        quality_improvement = max(0.0, q_next - q_curr)
        landing_quality_reward = 15.0 * quality_improvement
    else:
        landing_quality_reward = 0.0

    total_reward = progress_reward + stability_penalty + landing_quality_reward

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'landing_quality_reward': landing_quality_reward
    }
    return float(total_reward), components
```
