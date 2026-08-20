# Response Record

`evidence`: score=-94.32, 19/20 episodes crash early (<150 steps, score<-50), progress_reward mean=1.19 (barely moves toward target before dying), soft_landing_proxy active_rate=0.6% (essentially never triggers), stability_penalty magnitude only 7.2% after v2 scale reduction, indicating the agent falls fast, gets small positive delta_dist from falling, never learns to brake, and crashes.

`behavior_diagnosis`: agent reduces distance to target (positive mean delta_dist) but at uncontrolled speed — likely free-falling toward the ground, collecting small progress rewards, then crashing hard before ever reaching the landing conditions.

`signal_completeness`: progress_reward (delta_dist) incentivizes any distance reduction including dangerous free-fall; soft_landing_proxy is too sparse (0.6% trigger rate) to provide approach-quality gradient; stability_penalty exists but at current scale fails to prevent crash behavior. Missing: a dense signal that rewards controlled, slow, upright approach — so the agent can learn to brake before it's too late.

`selected_level`: Level 2 — `sparse_to_dense` on the landing proxy. The soft_landing_proxy binary condition (active_rate=0.6%) gives almost no learning signal. The agent never discovers what a good landing looks like because it crashes first. A continuous approach-quality signal gives gradient at every step, rewarding proximity × slowness × uprightness.

`selected_intervention`: replace `soft_landing_proxy` (sparse binary 0.5 bonus when near+slow+upright+both_contact) with `approach_quality_reward` — a continuous product of soft bounded factors: `proximity = 1/(1+5*dist)`, `speed_factor = 2/(2+|vx|+|vy|)`, `angle_factor = 1/(1+|angle|)`. Coefficient set to 1.0 so the signal is strong enough to matter without dominating progress. Other components unchanged.

`falsifiable_hypothesis`: a continuous approach-quality gradient at every step should teach the agent that slowing down and staying upright near the target is rewarding, reducing crash rate; the soft factors never collapse to exactly zero, so the agent always has gradient toward better approach behavior even when far away, unlike the old binary proxy that gave zero signal until all conditions were simultaneously met.

`expected_next_round`: active_rate of approach_quality_reward should rise to near 100% (continuous signal fires every step); episode_length should increase as agent survives longer; early_terminal count should drop below 19/20; score should improve above -94. Magnitude_share of approach_quality_reward may become significant but that's acceptable at this stage.

`main_risk`: the continuous approach-quality product may become an exploitable proxy — the agent could learn to hover near the target at moderate speed collecting approach_quality_reward without actually landing, creating a medium-score plateau. If this occurs, next round would need `dense_to_task_event` or adding a contact-gated completion bonus.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3: sparse_to_dense — replace binary soft_landing_proxy (0.6% active)
    with continuous approach_quality_reward that gives gradient everywhere.
    Progress delta and stability penalty unchanged from v2.
    """
    # -- Distance to target
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # -- 1. Progress delta (unchanged)
    delta_dist = dist_current - dist_next
    w_progress = 1.0
    progress_reward = w_progress * delta_dist

    # -- 2. Stability penalty (unchanged from v2)
    w_vel    = 0.001
    w_angle  = 0.001
    w_angvel = 0.0001

    stability_penalty = (
        -w_vel    * (abs(next_obs[2]) + abs(next_obs[3]))
        -w_angle  * abs(next_obs[4])
        -w_angvel * abs(next_obs[5])
    )

    # -- 3. Continuous approach quality (replaces sparse soft_landing_proxy)
    # Soft bounded factors: all in (0, 1], never collapse to exactly zero
    proximity    = 1.0 / (1.0 + 5.0 * dist_next)
    speed        = abs(next_obs[2]) + abs(next_obs[3])
    speed_factor = 2.0 / (2.0 + speed)
    angle_factor = 1.0 / (1.0 + abs(next_obs[4]))

    approach_quality = proximity * speed_factor * angle_factor
    w_approach = 1.0
    approach_quality_reward = w_approach * approach_quality

    # -- Total reward
    total_reward = progress_reward + stability_penalty + approach_quality_reward

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "approach_quality_reward": approach_quality_reward
    }

    return float(total_reward), components
```
