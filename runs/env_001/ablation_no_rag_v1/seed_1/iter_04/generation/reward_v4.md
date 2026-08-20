```text
evidence: score=-107.65, all episodes terminated early with failures, while distance_progress mean=1.12 is positive but insufficient to offset a large implicit step cost (~-1.57/step); previous vertical_accel_reward mean=0.15 did not help.
behavior_diagnosis: agent keeps crashing quickly, never achieving stable landing, and the reward lacks any completion signal to guide safe touchdown.
signal_completeness: necessary dense progress and tilt penalty exist, but a completion/landing signal (leg contacts) is missing, making the task of safe landing unreachable.
selected_level: Level 2 — necessary landing signal is absent, a mathematical structure change (addition of a sparse-to-dense completion component) is required.
selected_intervention: replace the ineffective vertical_accel_reward with a new landing_support_reward that gives +2.0 per step when both support legs contact the target platform.
falsifiable_hypothesis: by providing a positive reward for dual-leg contact, the agent will learn to achieve safe landing, reducing failure terminations and strongly improving total score.
expected_next_round: score should increase substantially (possibly positive), failure terminations should decrease, and the landing_support component will become active in successful episodes.
main_risk: agent might learn to spam hard contacts or partial contacts to trigger the reward without truly stable landing, but environmental termination conditions may still label those as failures.
```

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Positions
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    body_angle = obs[4]

    # Distance progress toward target (0,0)
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress = dist_curr - dist_next

    # Stability: penalise large body tilt
    tilt_penalty = -0.1 * abs(body_angle)

    # Landing reward: positive every step both legs are on the platform
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    landing_support = 2.0 if (left_contact and right_contact) else 0.0

    total_reward = progress + tilt_penalty + landing_support

    components = {
        'distance_progress': progress,
        'tilt_penalty': tilt_penalty,
        'landing_support': landing_support
    }

    return float(total_reward), components
```