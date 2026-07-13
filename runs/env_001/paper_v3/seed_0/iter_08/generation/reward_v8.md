`evidence`: Current score 96.5/200, landing_proxy dominates at 65.5% magnitude share (active_rate 46%), distance_penalty at 31.4%, velocity and orientation penalties negligible at ~1.5% each. Terminated 16/20 suggests most episodes reach a settled-like state, but the wide score range [-166, 239] and 48% target achievement indicate landing quality varies drastically between episodes. The sum-of-averages landing_proxy allows proximity or contact to compensate for poor speed or angle.

`behavior_diagnosis`: The agent reliably reaches the target region and achieves some form of contact (16/20 terminated), but the additive landing proxy creates a local optimum where partial satisfaction—e.g., being near target with single contact but high speed or tilted—still yields substantial reward. This prevents the agent from learning the precise, simultaneous, multi-condition satisfaction needed for high-quality landings.

`signal_completeness`: All four mandatory roles exist (position_approaching, velocity_reduction, orientation_stabilization, safe_landing_confirmation). However, the landing_proxy implements safe_landing_confirmation as a compensatory weighted sum, which fails to enforce the joint (AND-like) requirement that ALL conditions must be simultaneously satisfied for a true landing.

`selected_level`: Level 2 — `independent_to_joint` transformation. The arithmetic mean of bounded factors allows one dimension to compensate for another (e.g., good proximity + contact offsets high speed). Evidence from score variance and the gap to target indicates the agent exploits this compensation rather than jointly optimizing all landing dimensions.

`selected_intervention`: Replace the landing_proxy arithmetic mean (`(a+b+c+d)/4`) with a geometric mean (`(a*b*c*d)^0.25`). Remove the redundant contact_gate since contact participates directly in the geometric mean via `contact_factor = (left_contact + right_contact) / 2`. Adjust coefficient from 3.0 to 8.0 to account for the geometric mean being strictly ≤ arithmetic mean for the same inputs.

`falsifiable_hypothesis`: The geometric mean enforces joint satisfaction—if any single factor (proximity, speed, angle, or contact) is poor, the entire landing_proxy is suppressed. This eliminates the compensation loophole and forces the agent to improve all four dimensions simultaneously, raising the quality floor of landings and pushing average score toward target.

`expected_next_round`: landing_proxy active_rate may drop initially as partial-credit landings yield less reward, but the quality per active step should increase. Score should rise as the agent learns to satisfy all conditions jointly. The distance_penalty magnitude share may temporarily increase. Score variance should decrease as mediocre landings are penalized more heavily.

`main_risk`: The geometric mean with `contact_factor = (left+right)/2` still yields 0 when both contacts are absent. During no-contact approach phases, the landing_proxy provides zero gradient for velocity and angle improvement, relying entirely on velocity_penalty and orientation_penalty for those signals. This could slow learning of approach deceleration and leveling.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D vehicle landing task.

    Components:
    - distance_penalty: encourages moving towards the target platform (position=0,0)
    - velocity_penalty: penalizes high speed when near target, gated by distance
    - orientation_penalty: penalizes tilt and angular velocity
    - landing_proxy: geometric mean of bounded factors rewarding simultaneous
      proximity, low speed, upright posture, and ground contact;
      joint satisfaction required (all factors must be good, no compensation)
    """
    # next_obs unpacking
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Euclidean distance to target
    dist = (x**2 + y**2) ** 0.5

    # Component A: distance penalty (core progress signal)
    w_dist = 1.0
    distance_penalty = -w_dist * dist

    # Component B: velocity penalty (damped by distance to target)
    w_vel = 0.2
    gate = 1.0 / (1.0 + dist)
    speed_sq = vx**2 + vy**2
    velocity_penalty = -w_vel * speed_sq * gate

    # Component C: orientation stabilization penalty
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * abs(angle) - w_angvel * abs(ang_vel)

    # Component D: landing quality via geometric mean (joint satisfaction)
    w_landing = 8.0
    # proximity factor: 1 at dist=0, 0 when dist >= 2.5
    prox_factor = max(0.0, 1.0 - dist / 2.5)
    # speed factor: 1 at speed=0, 0 when speed >= 2.0 m/s
    speed = (vx**2 + vy**2) ** 0.5
    vel_factor = max(0.0, 1.0 - speed / 2.0)
    # angle factor: 1 at angle=0, 0 when |angle| >= 0.5 rad
    angle_factor = max(0.0, 1.0 - abs(angle) / 0.5)
    # contact factor: 0.0 (none), 0.5 (single), 1.0 (double contact)
    contact_factor = (left_contact + right_contact) / 2.0

    # Geometric mean enforces joint satisfaction across all four dimensions.
    # If any factor is 0, the entire term is 0 -- no compensation allowed.
    # Using exponent 0.25 (fourth root) prevents the collapse of a raw product
    # while retaining the AND-like pressure on all dimensions.
    landing_quality = (prox_factor * vel_factor * angle_factor * contact_factor) ** 0.25
    landing_proxy = w_landing * landing_quality

    # Total reward
    total_reward = distance_penalty + velocity_penalty + orientation_penalty + landing_proxy

    # Component dictionary
    components = {
        "distance_penalty": distance_penalty,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "landing_proxy": landing_proxy
    }

    return float(total_reward), components
```