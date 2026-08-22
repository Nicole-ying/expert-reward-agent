# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extracting relevant observations
    hull_angle_abs = abs(next_obs[0])
    hull_ang_vel_abs = abs(next_obs[1])
    horizontal_speed = next_obs[2]
    vertical_speed = next_obs[3]

    # Core forward progress: only reward positive horizontal speed
    forward_speed = max(0.0, horizontal_speed)

    # Soft health gate: reduces forward reward when posture deteriorates
    # Coefficients are chosen so that typical walking produces gate in [0.4, 0.8],
    # while large tilt or fast rotation significantly attenuate the reward.
    k_angle = 5.0
    k_ang_vel = 0.5
    gate = 1.0 / (1.0 + k_angle * hull_angle_abs + k_ang_vel * hull_ang_vel_abs)

    # Gated forward progress (main learning signal)
    w_fwd = 1.0
    progress_gated = w_fwd * forward_speed * gate

    # Vertical bounce penalty: only penalize excessive up/down oscillations
    vert_threshold = 0.5
    if abs(vertical_speed) > vert_threshold:
        excess = abs(vertical_speed) - vert_threshold
        vert_penalty = -0.1 * (excess ** 2)
    else:
        vert_penalty = 0.0

    total_reward = progress_gated + vert_penalty
    components = {
        'progress_gated': progress_gated,
        'vertical_penalty': vert_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## Selected task profile
- **task_family**: locomotion_continuous_control
- **dynamics_subtype**: planar_bipedal_gait on irregular terrain
- The agent must make fast, stable forward progress while avoiding falls; energy efficiency is secondary.

## Selected reward roles and signal mapping
| role | signal | formula operator | description |
|---|---|---|---|
| `forward_progress_reward` (mandatory) | `next_obs[2]` (horizontal speed) | `dense_state_signal` (positive linear) | Rewards positive forward speed; uses `max(0, speed)` to avoid rewarding backward motion. |
| `posture_stability_penalty` (mandatory, reformulated as soft gate) | `next_obs[0]` (hull angle), `next_obs[1]` (hull angular velocity) | `soft_health_gate` (reciprocal gate) | Instead of a separate penalty, the forward reward is multiplied by a gate that smoothly decays from 1.0 when the body is tilted or spinning quickly. This avoids an extra negative term and directly ties stability to the main reward. |
| `vertical_bounce_penalty` (conditional) | `next_obs[3]` (vertical speed) | `hinge + quadratic_penalty` | Lightly penalises excessive vertical speed (absolute >0.5 m/s) to discourage wasteful bouncing and to help maintain ground contact. The weight is kept small (`0.1`) so it does not dominate or suppress necessary small hops. |

## Roles excluded from this version
- **`joint_effort_penalty`**: energy/action cost is left for later iterations; v1 focuses on learning stable forward motion.
- **`step_pattern_constraint`**: bilateral contact signals are available but would over‑constrain early exploration on uneven terrain.
- **`termination_based_completion_reward`**: no explicit success/failure flag exists in `info`; termination events cannot be reliably detected online, so terminal‑event signals are omitted.
- **`efficiency/action smoothness`**: not included to avoid suppressing exploration of large control moments needed for obstacle negotiation.

## Design rationale and novelty
Previous attempts stacked multiple independent penalties (posture, angular velocity, air time, action cost) on top of a raw forward speed reward, leading to consistently negative total returns (best score –52.7). The hypothesis is that **too many negative components overwhelm the positive progress signal**, especially when small but safe oscillations are constantly penalised.

This version adopts a **soft health gate** as the main stability mechanism:
- Forward progress is the sole positive signal (`speed × gate`), so the agent can only increase total reward by keeping good posture while moving fast.
- The gate does not produce a penalty by itself; it merely attenuates the progress reward when posture worsens. This avoids the “walking in a straight‑jacket” effect where every small wiggle draws a penalty.
- The reciprocal form `1/(1 + k1·|angle| + k2·|ang_vel|)` gives a smooth gradient, encouraging the agent to improve posture without a hard dead zone.

## Why no terminal success/failure reward
Neither an explicit success flag nor a failure flag is available. While a fall can be inferred from extreme `hull_angle`, a terminal‑event penalty would be sparse and provide no intermediate gradient. The soft gate already addresses the precursor conditions of a fall (large tilt / fast rotation) in a dense manner, so adding a hard post‑factum penalty is unnecessary and could hurt learning stability.

## What is left for future iterations
- Energy/action cost (once a stable gait has been acquired).
- Potential‑based shaping or improvement delta if purely velocity‑based progress is insufficient (e.g., agent gets stuck in pits).
- More sophisticated ground‑contact monitoring to balance leg usage, if a persistent one‑sided gait emerges.

## Failure modes to monitor after initial training
- **Persistent low speed**: if the gate is too sensitive, the agent may learn to stand upright and barely move. Check that the average gate value is not far below 0.5 during normal walking.
- **Frequent falls with high gate values**: if the gate coefficients are too lenient, falls may still occur without sufficient attenuation; then `k_angle` or `k_ang_vel` may need to be increased.
- **Excessive bouncing**: if the vertical speed frequently exceeds 0.5, the `vertical_penalty` may be too weak; its weight can be slightly raised or the threshold lowered.
