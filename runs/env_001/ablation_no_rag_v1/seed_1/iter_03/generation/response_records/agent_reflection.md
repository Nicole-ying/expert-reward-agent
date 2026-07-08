# Response Record

```yaml
evidence: >
  The previous round added a velocity_penalty that dominated reward (84% magnitude share), resulting in a score drop from -100.51 to -110.36. All 20 evaluation episodes were early terminations (crash or out-of-view) with a mean length of 68.45. The velocity_penalty punished |vy| uniformly, including the downward motion necessary to reach the platform, which is structurally misaligned with the task. The best historical code (distance_progress + tilt_penalty) had no velocity term and performed relatively better. Removing the harmful penalty and adding a guiding signal for controlled descent is indicated.

behavior_diagnosis: >
  The agent rapidly crashes or flies out of bounds, failing to learn a safe descent. The velocity penalty suppressed all vertical speed, preventing the agent from exploiting downward motion to reach the target or from discovering the engine actions needed to decelerate.

signal_completeness: >
  A progress signal and a tilt constraint exist, which provide basic goal direction and attitude stability. However, a signal to encourage deceleration against gravity is missing; the agent receives no reward for firing the main engine to soften the landing, leaving early exploration vulnerable to terminal crashes before any landing contact can occur.

selected_level: >
  Level 2 (structural transformation). The velocity_penalty's |vy| formulation is mathematically inappropriate for a task that requires downward motion. A different signal—rewarding upward acceleration to counteract gravity—replaces the penalty while preserving the intent of promoting a controlled descent.

selected_intervention: >
  Replace the velocity_penalty with a vertical_accel_reward: +0.1 * (next_obs[3] - obs[3]) clamped to non-negative values (only reward upward accelerations). This targets the transition from free-fall to powered descent without punishing the necessary downward speed.

falsifiable_hypothesis: >
  If the main engine can produce upward thrust, rewarding positive Δvy will incentivise deceleration, increasing episode length and enabling the agent to approach the platform more slowly. This should raise the final score above the historical best (-100.51).

expected_next_round: >
  Score should improve above -100.51; episode length should increase as early crashes become less frequent. The vertical_accel_reward active_rate will be high, and its magnitude_share will be moderate and positive, while distance_progress remains the main goal signal.

main_risk: >
  The agent might learn to "pulse fire" the engine to generate many small positive Δvy rewards without actually reaching the platform, or excessive upward accelerations could waste fuel and delay descent. Mitigation by the low coefficient is expected, but further tuning or a bounded formulation may be needed.
```

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract positions
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    body_angle = obs[4]

    # Euclidean distances to target platform
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress = dist_curr - dist_next

    # Stability: penalise large body tilt
    tilt_penalty = -0.1 * abs(body_angle)

    # Controlled descent: reward upward acceleration (deceleration against gravity)
    vy_curr = obs[3]
    vy_next = next_obs[3]
    delta_vy = vy_next - vy_curr
    # Only reward deceleration (positive delta_vy), avoid penalising natural downward acceleration
    vertical_accel_reward = 0.1 * max(0.0, delta_vy)

    total_reward = progress + tilt_penalty + vertical_accel_reward

    components = {
        'distance_progress': progress,
        'tilt_penalty': tilt_penalty,
        'vertical_accel_reward': vertical_accel_reward
    }

    return float(total_reward), components
```
