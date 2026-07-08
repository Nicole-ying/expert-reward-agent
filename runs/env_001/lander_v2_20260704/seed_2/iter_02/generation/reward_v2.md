## 1. evidence
All 20 evaluation episodes terminate early (len=68.4, terminated=20/20, all score<-50), the `soft_landing_bonus` fires in only 0.5% of steps with episode_sum_mean=0.175, and the unbounded `distance_reward` dominates at -66.5 mean sum.

## 2. behavior_diagnosis
The agent rapidly minimizes distance to the platform (avg per-step distance ≈0.97 near termination) but consistently crashes or fails to complete landing; the binary soft_landing_bonus is effectively invisible as a learning signal, so the agent never receives gradient toward proper landing posture.

## 3. signal_completeness
A position-guiding signal (`distance_reward`) and a stability constraint (`stability_penalty`) exist, but the completion signal (`soft_landing_bonus`) is a hard binary gate that is unreachable during early learning — this is the critical missing gradient for the landing phase.

## 4. selected_level
**Level 2** — the 0.5% active_rate binary bonus matches the `sparse_to_dense` evidence pattern exactly; a coefficient tweak cannot fix structural sparsity.

## 5. selected_intervention
Transform `soft_landing_bonus` from a hard binary gate into a continuous `landing_quality` product of bounded factors (`max(0, 1−x/D)`) with a soft floor on the contact factor, scaled by 0.8 to match the new value range.

## 6. falsifiable_hypothesis
Replacing the all-or-nothing bonus with a product of continuous proximity/speed/angle/contact factors will raise `active_rate` from ~0.5% to a substantially higher value (target 40%+), provide usable gradient for the landing approach, and reduce early crash terminations.

## 7. expected_next_round
`landing_quality` active_rate should jump well above 1%; `episode_sum_mean` of this component should grow noticeably from 0.175; episode length may initially increase (agent explores near-platform behavior) and external score should trend upward if landing posture improves.

## 8. main_risk
Product form with a low contact floor may still create a "proximity plateau" where the agent hovers near the platform collecting partial credit without committing to leg contact — the 0.2 floor prevents full collapse but does not guarantee terminal landing behavior.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observables
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Primary learning signal: continuous negative Euclidean distance to goal
    distance_to_target = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -1.0 * distance_to_target

    # 2. Stability penalty: light constraint on excessive speed, tilt, and spin
    w_vx = 0.15
    w_vy = 0.05
    w_angle = 0.2
    w_angvel = 0.2
    stability_penalty = -(
        w_vx * abs(x_vel) +
        w_vy * abs(y_vel) +
        w_angle * abs(body_angle) +
        w_angvel * abs(angular_vel)
    )

    # 3. Gradual landing quality: continuous product of bounded factors
    #    Replaces binary soft_landing_bonus to provide dense gradient toward
    #    proximity, low speed, upright attitude, and dual-leg contact.
    prox_factor = max(0.0, 1.0 - distance_to_target / 2.0)
    speed_x_factor = max(0.0, 1.0 - abs(x_vel) / 0.8)
    speed_y_factor = max(0.0, 1.0 - abs(y_vel) / 0.8)
    angle_factor = max(0.0, 1.0 - abs(body_angle) / 0.4)
    # Soft floor on contact so the product does not vanish before legs touch
    contact_factor = 0.2 + 0.8 * (left_contact + right_contact) / 2.0

    landing_quality = 0.8 * prox_factor * speed_x_factor * speed_y_factor * angle_factor * contact_factor

    total_reward = distance_reward + stability_penalty + landing_quality

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components
```