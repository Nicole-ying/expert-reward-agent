## 1. evidence
external score 36.5/200 with 15/20 episodes terminating, contact_reward active only 6.8% of steps while goal_proximity mean is -161.17, velocity_penalty contribution is negligible at -2.5% magnitude share despite 99.8% active_rate because proximity gate `1/(1+5*dist²)` collapses to <0.05 beyond dist≈2, giving the agent almost no deceleration feedback until it is already on top of the pad.

## 2. behavior_diagnosis
the agent occasionally reaches the pad (contact_reward accumulates +90 over brief contact windows) but most episodes end in termination—likely crashes or unstable landings—because the velocity penalty only activates meaningfully within ~1–2 units of the target, too late for the agent to decelerate from approach speed.

## 3. signal_completeness
all four mandatory roles are present (goal_proximity, soft_landing_velocity, orientation_stability, contact_reward), but the velocity penalty's activation gate is so narrow that it is effectively unreachable as a control signal during approach; the missing settlement_bonus is secondary to the approach-control failure.

## 4. selected_level
Level 2 (`global_to_local_gate`) — the velocity penalty's proximity gate is too narrow, making the deceleration constraint discontinuous and unreachable until it is too late to avoid crash.

## 5. selected_intervention
split the single `proximity` gate into `proximity_narrow` (alpha=5.0, for contact_reward) and `proximity_wide` (alpha=1.5, for velocity_penalty), so the velocity penalty provides usable deceleration feedback from dist≈3–5 instead of only dist<2; all other components and coefficients unchanged.

## 6. falsifiable_hypothesis
widening the velocity penalty's activation zone will cause the agent to begin decelerating earlier on approach, reducing high-speed collisions and increasing the fraction of episodes that reach stable dual-leg contact on the pad.

## 7. expected_next_round
velocity_penalty magnitude_share should rise from ~2.5% to a larger share, contact_reward active_rate should increase above 6.8% because more episodes reach the pad safely, terminated ratio should drop below 15/20, and external score should improve.

## 8. main_risk
a wider velocity penalty may mildly suppress approach speed in early exploration and slow progress toward the target; if velocity_penalty grows too dominant, the agent could hover indecisively at intermediate distances.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components
    x = obs[0]            # horizontal position relative to target pad
    y = obs[1]            # vertical position relative to target pad
    vx = obs[2]           # horizontal velocity
    vy = obs[3]           # vertical velocity
    angle = obs[4]        # body angle
    ang_vel = obs[5]      # angular velocity
    left_contact = obs[6] # left support leg contact (0 or 1)
    right_contact = obs[7]# right support leg contact (0 or 1)

    # Hyperparameters
    w_goal = 1.0
    w_vel = 0.5
    w_angle = 0.2
    w_angvel = 0.1
    w_contact = 3.0

    # Distance to target center (squared)
    dist_sq = x**2 + y**2

    # Narrow proximity gate for contact reward: only reward leg contact very close to target
    proximity_narrow = 1.0 / (1.0 + 5.0 * dist_sq)

    # Wide proximity gate for velocity penalty: provide deceleration feedback earlier on approach
    proximity_wide = 1.0 / (1.0 + 1.5 * dist_sq)

    # 1. Main progress: drive toward target center (dense quadratic penalty on distance)
    goal_proximity = -w_goal * dist_sq

    # 2. Soft landing velocity penalty: active over a wider approach zone
    velocity_penalty = -w_vel * (vx**2 + vy**2) * proximity_wide

    # 3. Orientation stability penalty: penalize tilt and spin everywhere (light weight)
    orientation_penalty = -w_angle * (angle**2) - w_angvel * (ang_vel**2)

    # 4. Contact reward: reward both legs touching the pad, gated by narrow proximity
    both_legs_contact = left_contact * right_contact  # 1 only if both are 1
    contact_reward = w_contact * both_legs_contact * proximity_narrow

    # Total reward
    total_reward = goal_proximity + velocity_penalty + orientation_penalty + contact_reward

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward
    }

    return float(total_reward), components
```