1. `evidence`：latest score 0.62 is best so far but far below target; all 20 eval episodes terminated early (avg length 117.7 steps); `landing_quality_reward` active_rate only 0.4%, magnitude_share 7.8%, indicating near-sparse feedback; `angle_penalty` (−1.15) almost cancels `potential_diff` (+1.07) leaving tiny net reward; previous iteration replaced `success_bonus` with this sparse landing product and improved from −24.46 to 0.62, but the extremely low trigger rate still starves the agent of useful final-phase guidance.

2. `behavior_diagnosis`：The agent likely flies downward quickly, gets some progress reward, but then crashes or settles prematurely because the angle penalty is large enough to offset progress and there is almost no informative signal near the pad; the sparse landing reward rarely fires, so the agent cannot learn to achieve a soft two-foot landing.

3. `signal_completeness`：Progress guidance exists, attitude constraint exists, but a dense final-approach incentive for positional, velocity, angular and contact proximity is missing; the current landing component is a harsh all-or-nothing product that fails to provide gradual feedback, making it unreachable for learning.

4. `selected_level`：Level 2 — the evidence pattern “task event almost never triggers, local feedback missing” matches `sparse_to_dense`; a structural change from a multiplicative sparse bonus to a continuous, additive proximity-based sum is warranted.

5. `selected_intervention`：Change the `landing_quality_reward` component from a product of hard thresholds to a dense sum of independent proximity measures (position, velocity, angle, contact) using bounded linear decays, with a coefficient recalibrated to give approximately 0.5–1.0 per step when near the pad.

6. `falsifiable_hypothesis`：A dense additive landing guidance will give the agent usable step-by-step rewards as it approaches the origin, small velocities, upright attitude and two-leg contact, thereby enabling learning of precise final-phase control and raising the successful landing rate, which should increase the score and episode length.

7. `expected_next_round`：`landing_quality_reward` active_rate should rise substantially (>>1%), its `episode_sum_mean` should grow, total external score should increase above 0.62, and the number of early-terminal (<150 steps) episodes should decrease; episode length may initially lengthen as the agent learns to stay alive longer to collect the new rewards.

8. `main_risk`：The dense guidance might encourage the agent to hover indefinitely near the target without actually landing, since the reward persists as long as it stays close; this could cause episode length to increase without successful termination, and the overall score may plateau at a moderate level if the signal is not strong enough to push the agent to complete the final contact and velocity thresholds.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -- Extract state variables --
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle, angular_v = obs[4], obs[5]

    x_n, y_n = next_obs[0], next_obs[1]
    vx_n, vy_n = next_obs[2], next_obs[3]
    angle_n, angular_v_n = next_obs[4], next_obs[5]
    left_contact_n = next_obs[6]
    right_contact_n = next_obs[7]

    # -- Component 1: Potential-based progress (unchanged) --
    w_dist = 1.0
    w_vel  = 0.3
    dist_current = (x**2 + y**2) ** 0.5
    dist_next    = (x_n**2 + y_n**2) ** 0.5
    vel_current  = abs(vx) + abs(vy)
    vel_next     = abs(vx_n) + abs(vy_n)

    potential_current = w_dist * dist_current + w_vel * vel_current
    potential_next    = w_dist * dist_next    + w_vel * vel_next
    potential_diff = potential_current - potential_next

    # -- Component 2: Attitude penalty (unchanged) --
    k_angle = 0.5
    k_angvel = 0.1
    angle_penalty = -k_angle * (angle_n**2) - k_angvel * (angular_v_n**2)

    # -- Component 3: Dense landing guidance (replaces sparse product) --
    # Proximity measures that are high when close to the ideal landing state.
    # Each dimension uses max(0, 1 - |error|/threshold) to give a fraction in [0,1].
    x_proximity        = max(0.0, 1.0 - abs(x_n) / 0.4)
    y_proximity        = max(0.0, 1.0 - abs(y_n) / 0.4)
    speed_n            = (vx_n**2 + vy_n**2) ** 0.5
    speed_proximity    = max(0.0, 1.0 - speed_n / 0.3)
    angle_proximity    = max(0.0, 1.0 - abs(angle_n) / 0.2)
    contact_proximity  = (left_contact_n + right_contact_n) / 2.0   # 0, 0.5 or 1

    # Weighted sum gives a dense indicator of landing progress.
    # Weights chosen so that the maximum per-step reward is on the order of 1.0.
    w_x   = 0.15
    w_y   = 0.25
    w_spd = 0.25
    w_ang = 0.25
    w_con = 0.10

    landing_guidance = w_x * x_proximity + w_y * y_proximity + \
                       w_spd * speed_proximity + w_ang * angle_proximity + \
                       w_con * contact_proximity

    k_landing_guidance = 2.0
    landing_reward = k_landing_guidance * landing_guidance

    # -- Total reward --
    total_reward = potential_diff + angle_penalty + landing_reward

    components = {
        "potential_diff": float(potential_diff),
        "angle_penalty": float(angle_penalty),
        "landing_reward": float(landing_reward)
    }

    return float(total_reward), components
```