**1. `evidence`**：10/20 episodes terminate, 10/20 truncate; score range [-42.9, 298.3] shows half landing well, half failing entirely; contact_reward active_rate=3.5% (~24 steps/ep) and magnitude_share=25.6% despite w_contact=2.0, while goal_proximity dominates at 68.9% magnitude share with mean -117.66 — when contact fires it matters, but the per-step gradient from "hovering near target" to "both legs down" is only +2/step, too weak to pull the failing 50% across the finish line.

**2. `behavior_diagnosis`**：The agent reliably reaches the target vicinity (goal_proximity sum only -117.66 over 675 steps → avg dist²≈0.17 → avg dist≈0.41) but fails to consistently execute the final landing maneuver; half the episodes truncate without achieving the settled state, producing the wide score variance and ~90-point gap to target.

**3. `signal_completeness`**：All mandatory roles are present and reachable (goal_proximity, velocity_penalty, orientation_penalty, contact_reward). No missing signal category. The issue is relative scale, not missing structure.

**4. `selected_level`**：Level 1 — contact_reward has correct mathematical form (gated by proximity, both-legs AND), correct symbol (positive), and fires 3.5% of steps (above the ~1% sparsity danger zone). Evidence points to insufficient magnitude relative to the goal_proximity basin, not wrong structure.

**5. `selected_intervention`**：Increase `w_contact` from 2.0 to 8.0. All other coefficients, gates, and component forms unchanged.

**6. `falsifiable_hypothesis`**：A 4× stronger per-step contact reward makes the landed state's value dominant enough to propagate a strong gradient backward through the pre-landing action sequence, pulling the currently-failing 50% of episodes into successful termination and raising mean score toward 200.

**7. `expected_next_round`**：contact_reward magnitude_share should rise significantly; terminated ratio should increase above 10/20; score mean should rise and variance should shrink; goal_proximity should remain stable (approach behavior already works); no regression in velocity or orientation metrics.

**8. `main_risk`**：Stronger contact reward could incentivize crash-landing (high-speed impact that momentarily triggers both-legs contact). The existing velocity_penalty (active_rate=99.4%, proximity-gated) should counteract this, but if next round shows velocity_penalty magnitude spiking alongside higher contact_reward, the velocity gate may need tightening.

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
    alpha_proximity = 5.0          # controls the activation radius for proximity-based terms
    w_vel = 0.5
    w_angle = 0.2
    w_angvel = 0.1
    w_contact = 8.0                # increased from 2.0 to strengthen landing gradient

    # Distance to target center (squared)
    dist_sq = x**2 + y**2

    # Soft proximity weight: ~1 when close to target, ~0 when far
    proximity = 1.0 / (1.0 + alpha_proximity * dist_sq)

    # 1. Main progress: drive toward target center (dense quadratic penalty on distance)
    goal_proximity = -w_goal * dist_sq

    # 2. Soft landing velocity penalty: active only near the target
    velocity_penalty = -w_vel * (vx**2 + vy**2) * proximity

    # 3. Orientation stability penalty: penalize tilt and spin everywhere (light weight)
    orientation_penalty = -w_angle * (angle**2) - w_angvel * (ang_vel**2)

    # 4. Contact reward: encourage both legs touching, gated by proximity
    both_legs_contact = left_contact * right_contact  # 1 only if both are 1
    contact_reward = w_contact * both_legs_contact * proximity

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