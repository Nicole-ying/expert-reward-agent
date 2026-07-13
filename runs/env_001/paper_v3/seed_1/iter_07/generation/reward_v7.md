`evidence`: contact_reward contributes 94% of signed reward (1960.93 episode_mean) with 100% active_rate using the formula `proximity * speed_quality * angle_quality` that contains no contact requirement; 17/20 episodes hit truncation at len≈896; goal_proximity is only -109; the agent farms persistent positive proxy without ever needing to land.

`behavior_diagnosis`: agent hovers or drifts near the target pad (~avg dist 0.35), slow and upright, collecting steady continuous "landing quality" reward indefinitely without actually touching both legs to the pad; episodes survive full length and almost never terminate.

`signal_completeness`: goal_proximity (distance gradient), velocity_penalty (near-target speed damping), and orientation_penalty (angle stability) are all present and reasonably shaped — but the contact_reward is missing its essential gating dimension: `both_legs_contact`. The "landing" reward has no landing requirement.

`selected_level`: Level 2 — `proxy_to_completion_alignment`. Evidence fits the pattern exactly: generated reward is high, external completion is poor (3/20 terminates, likely not all successful), and the proxy is missing the core completion dimension of leg contact.

`selected_intervention`: gate `contact_reward` by `both_legs_contact = left_contact * right_contact`, changing from `proximity * speed_quality * angle_quality` to `both_legs_contact * proximity * speed_quality * angle_quality`. Keep `w_landing=3.0` since the signal will now be much sparser and needs strong per-step incentive when contact is achieved.

`falsifiable_hypothesis`: contact_reward will drop from ~1960 to a much smaller per-episode sum (only firing during actual contact), breaking the proxy plateau; goal_proximity will become the dominant gradient driving the agent to reach and stay on the pad; truncation rate should fall as the agent learns to terminate by settling.

`expected_next_round`: contact_reward episode_sum_mean drops dramatically (e.g. <100); magnitude_share of goal_proximity rises; truncated episodes should decrease from 17/20; terminated episodes should increase; score may initially dip before improving as the agent shifts from proxy-farming to actual landing behavior.

`main_risk`: with contact_reward now a sparse event, the agent may take longer to discover the landing behavior; if goal_proximity alone is insufficient to drive the final descent, the agent could stall near the target (similar to current behavior but without the proxy). If this occurs next round, adding a sparse settlement bonus or slightly increasing w_goal would be the next step.

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
    alpha_proximity = 5.0
    w_vel = 0.5
    w_angle = 0.2
    w_angvel = 0.1
    w_landing = 3.0           # weight for contact-gated landing quality reward
    beta_speed = 10.0          # speed quality decay
    beta_angle = 10.0          # angle quality decay

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

    # 4. Contact-gated landing quality reward: only fires when both legs touch the pad
    both_legs_contact = left_contact * right_contact  # 1 only if both are 1
    speed_sq = vx**2 + vy**2
    speed_quality = 1.0 / (1.0 + beta_speed * speed_sq)
    angle_sq = angle**2
    angle_quality = 1.0 / (1.0 + beta_angle * angle_sq)
    contact_reward = w_landing * both_legs_contact * proximity * speed_quality * angle_quality

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