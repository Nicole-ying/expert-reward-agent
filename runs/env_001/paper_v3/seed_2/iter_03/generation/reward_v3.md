`evidence`: 20/20 episodes terminate early (len=69.4, all score<-50), progress mean +5.59 dominates (88.4% magnitude), landing_quality active only 3.5% (contact rarely reached), orientation_penalty negligible (-0.009 mean). Previous round increased w_progress 1.0→5.0 without fixing approach-speed control, score worsened from best.

`behavior_diagnosis`: The agent moves toward the target (positive progress) but crashes at high speed before achieving leg contact. The progress signal rewards any distance reduction regardless of velocity, so the agent learns to plummet toward the target rather than decelerate for a soft landing.

`signal_completeness`: The `soft_landing_condition` role is only partially fulfilled — `landing_quality` activates on contact (3.5% of steps), which is too late. There is no approach-phase signal encouraging velocity reduction as the agent nears the target. `crash_and_out_of_bounds_prevention` is also missing, relying solely on a negligible orientation penalty.

`selected_level`: Level 2 — `global_to_local_gate` transformation. The missing approach-phase velocity constraint is a necessary safety signal that should activate only near the target (local), not during early exploration (global). This is not a simple coefficient fix because no such component exists in the current reward.

`selected_intervention`: Add a new component `approach_velocity_penalty` that penalizes total velocity magnitude, gated by proximity to target (`1/(1+next_dist)`). Base on best code (w_progress=1.0) since current (w_progress=5.0) amplified the crash behavior. New coefficient w_approach_vel=0.3.

`falsifiable_hypothesis`: A proximity-gated velocity penalty creates an incentive to slow down when near the target, transforming the policy from "dive toward target" to "approach then decelerate." This should reduce crash rate and increase landing_quality active_rate as the agent survives long enough to reach leg contact.

`expected_next_round`: Episode length should increase beyond 69 steps, `approach_velocity_penalty` should show meaningful non-zero activation near target, `landing_quality` active_rate should rise from 3.5%, and score should improve from -85. Crash rate (early terminal count) should drop below 20/20.

`main_risk`: The velocity penalty might discourage the agent from approaching the target at all if it cannot control velocity precisely enough — it could learn to hover far away where proximity≈0 and the penalty is negligible, causing progress to stall and episode length to increase without landing success.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # weights and sensitivities
    w_progress = 1.0
    w_angle_penalty = 0.2
    w_landing = 1.0
    w_approach_vel = 0.3   # proximity-gated velocity penalty near target
    a_v = 10.0             # sensitivity for vertical speed in landing quality
    b_angle = 10.0         # sensitivity for body angle in landing quality

    # current and next distances to target (0,0)
    dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 1. Main progress signal: improvement towards target
    progress = w_progress * (dist - next_dist)

    # 2. Orientation stability: penalty for body angle to prevent tumbling
    orientation_penalty = -w_angle_penalty * (next_obs[4] ** 2)

    # 3. Landing quality bonus: soft success proxy when leg contacts are active
    contact = max(next_obs[6], next_obs[7])  # any leg contact
    vertical_speed_factor = 1.0 / (1.0 + a_v * (next_obs[3] ** 2))
    angle_factor = 1.0 / (1.0 + b_angle * (next_obs[4] ** 2))
    landing_quality = w_landing * contact * vertical_speed_factor * angle_factor

    # 4. Approach-phase velocity penalty: penalize high speed near target
    #    Proximity gates the penalty: negligible when far, active when close
    proximity = 1.0 / (1.0 + next_dist)
    velocity_sq = next_obs[2] ** 2 + next_obs[3] ** 2
    approach_velocity_penalty = -w_approach_vel * proximity * velocity_sq

    total_reward = progress + orientation_penalty + landing_quality + approach_velocity_penalty

    components = {
        "progress": progress,
        "orientation_penalty": orientation_penalty,
        "landing_quality": landing_quality,
        "approach_velocity_penalty": approach_velocity_penalty
    }

    return float(total_reward), components
```