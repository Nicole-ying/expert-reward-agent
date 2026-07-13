# Response Record

`evidence`: landing_quality dominates at 99.4% magnitude share (episode_sum_mean=892.81) while shaped_progress is negligible at 0.5% (4.45); 9/20 episodes truncated at time limit despite high proxy accumulation; score plateaued at 168.34 with 84% target achievement — the agent gets massive continuous reward just for occupying good states near the ground.

`behavior_diagnosis`: the agent exploits the continuous state-value landing_quality by staying near the ground with low speed, upright posture, and centered position, accumulating ~1.36 per step for 657 steps on average, but fails to complete landing efficiently in 45% of episodes — it is farming the proxy rather than finishing the task.

`signal_completeness`: progress signal (shaped_progress) and landing quality signal both exist and are reachable, but landing_quality's structure as a persistent state-value reward enables exploitation through occupancy rather than completion — the signal's mathematical form is the problem, not its absence.

`selected_level`: Level 2 — state_to_improvement transformation. The landing_quality component matches the exact diagnosis pattern: a persistent state value q(s_next) that the policy can farm by remaining in rewarding states without completing the task. Level 1 coefficient reduction would not fix the structural exploitation; the agent could still accumulate by hovering even with a smaller coefficient.

`selected_intervention`: transform landing_quality from a persistent state-value reward `q(next_obs)` into an improvement-based reward `coefficient * (q(next_obs) - q(obs))`, using the full signed difference so that deterioration is penalized and steady-state hovering yields near-zero reward. Only the landing_quality component changes; shaped_progress and angular_vel_penalty are untouched.

`falsifiable_hypothesis`: removing the occupancy-farming incentive should force the agent to make genuine progress toward the landing pad rather than hovering near it. The truncated episode rate should drop as the agent can no longer accumulate reward by staying in place, and the score should move closer to target as landing completions become the primary way to earn quality reward.

`expected_next_round`: landing_quality episode_sum_mean should drop dramatically from 892 to a bounded range (max ~100 if all episodes achieve full improvement); truncated count should decrease below 9/20; external score should increase toward 200; episode_length may decrease as the agent stops dawdling.

`main_risk`: improvement-only reward gives zero signal for maintaining a good terminal state, which could cause oscillation or unstable landings if the agent overshoots and deteriorates; the coefficient 50.0 is a best-guess starting point and may need scaling if the improvement signal is too weak or too noisy relative to shaped_progress.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Helper: landing quality evaluated at a given observation
    def _landing_quality(o):
        height = o[1]
        near_ground = 1.0 / (1.0 + 5.0 * abs(height))
        speed = (o[2] ** 2 + o[3] ** 2) ** 0.5
        low_speed = 1.0 / (1.0 + speed)
        angle_abs = abs(o[4])
        upright = 1.0 / (1.0 + 3.0 * angle_abs)
        x_offset = abs(o[0])
        centered = 1.0 / (1.0 + 3.0 * x_offset)
        return 0.5 * near_ground * (2.0 * low_speed + 1.0 * upright + 1.0 * centered)

    # 1. Main learning signal: distance improvement × attitude gate (unchanged)
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_raw = dist_current - dist_next
    angle_abs_next = abs(next_obs[4])
    gate = 1.0 / (1.0 + 5.0 * angle_abs_next)
    shaped_progress = 5.0 * progress_raw * gate

    # 2. Safety constraint: angular velocity light penalty (unchanged)
    ang_vel_penalty = -0.1 * (next_obs[5] ** 2)

    # 3. Landing quality: state→improvement (state_to_improvement transform)
    #     Reward becoming better, not occupying a good state.
    quality_current = _landing_quality(obs)
    quality_next    = _landing_quality(next_obs)
    landing_quality_improvement = 50.0 * (quality_next - quality_current)

    total_reward = shaped_progress + ang_vel_penalty + landing_quality_improvement
    components = {
        "shaped_progress": shaped_progress,
        "angular_vel_penalty": ang_vel_penalty,
        "landing_quality_improvement": landing_quality_improvement
    }
    return (float(total_reward), components)
```
