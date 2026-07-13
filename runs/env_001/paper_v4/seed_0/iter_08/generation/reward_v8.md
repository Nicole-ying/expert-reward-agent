`evidence`: Current score=-388.67, len=109.5, 19/20 early terminal; touchdown_bonus and stable_landed both active_rate=0%, meaning the agent never reaches the landing pad; approach_shaping episode_sum_mean=-0.93 (net negative, agent moves away); descent_safety triggers only 6.2% of time but contributes -37.7% signed share; iter 6 best used state-based safe_proximity and achieved len=911, score=-18.28; iter 7 replaced it with delta-based approach_shaping and collapsed to len=109.

`behavior_diagnosis`: The agent crashes within ~109 steps in 19/20 episodes. The delta-based approach_shaping provides noisy, weak, and often negative guidance (net -0.93), failing to create a consistent gradient toward the pad. Combined with fuel_penalty (-0.05 per engine use) and rare-but-heavy descent_safety spikes, the agent has no reliable positive signal to follow, so it never discovers the pad region.

`signal_completeness`: Missing a consistent, state-based gradient toward the landing zone. The delta euclidean-distance signal couples horizontal and vertical progress into a single noisy difference that is zero-mean when stationary and frequently negative when perturbed. The landing events (touchdown, stable) are unreachable because the agent never reaches the pad. A state-based proximity signal that directly rewards being centered and low is necessary for the agent to discover the pad region.

`selected_level`: Level 2 — `state_to_improvement` evidence pattern: iter 6's state-based proximity reached the pad area (len=911) but led to proxy farming without landing; iter 7's delta-based approach collapsed entirely. The fix is to restore state-based proximity but add an anti-farming height factor, transforming the signal from pure delta euclidean to a product of bounded horizontal proximity and height incentive.

`selected_intervention`: Replace `approach_shaping` (delta euclidean: `2.0*(prev_dist-curr_dist)`) with `landing_proximity` (state-based product: `2.0 * 1/(1+5|x|) * max(0,1-y/2)`). This provides a consistent gradient toward x=0 that strengthens as the agent descends, making hovering at altitude unrewarding.

`falsifiable_hypothesis`: A state-based product signal gives the agent a stationary gradient that is positive whenever it is near the pad center and low — unlike the delta signal which vanishes when stationary and goes negative when perturbed. The height factor prevents the farming seen in iter 6 because being centered at high altitude yields near-zero reward. This should guide the agent to the pad region and encourage descent, making touchdown_bonus and stable_landed reachable for the first time.

`expected_next_round`: landing_proximity episode_sum_mean should become positive; episode length should increase substantially from 109 toward iter 6's ~911; early termination rate should drop; touchdown_bonus and stable_landed active_rate may rise above 0%; score should improve from -388 toward and beyond iter 6's -18.

`main_risk`: The height_factor rewards being at y=0, which could incentivize the agent to dive straight down and crash, bypassing the soft landing entirely. If descent_safety is too weak to counteract this, the agent may achieve high landing_proximity at the cost of crash terminations, yielding superficially better scores but no actual landing.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Previous state (needed for touchdown transition detection)
    prev_lc = obs[6]
    prev_rc = obs[7]

    # Next state
    x = next_obs[0]
    y = next_obs[1]
    xv = next_obs[2]
    yv = next_obs[3]
    ang = next_obs[4]
    lc = next_obs[6]
    rc = next_obs[7]

    # 1. Landing proximity: state-based reward for horizontal centering AND low height
    horizontal_proximity = 1.0 / (1.0 + 5.0 * abs(x))
    height_factor = max(0.0, 1.0 - y / 2.0)
    landing_proximity = 2.0 * horizontal_proximity * height_factor

    # 2. Descent safety: penalize dangerous high-speed descent near ground
    descent_safety = 0.0
    if yv < 0.0 and y > 0.02:
        total_speed = (xv ** 2 + yv ** 2) ** 0.5
        height_urgency = 1.0 / (1.0 + y)
        descent_safety = -2.0 * total_speed * height_urgency

    # 3. Touchdown bonus: first moment both legs achieve ground contact
    prev_contacts = prev_lc + prev_rc
    next_contacts = lc + rc
    touchdown_bonus = 0.0
    if next_contacts >= 2.0 and prev_contacts < 2.0:
        speed_quality = max(0.0, 1.0 - (abs(xv) + abs(yv)))
        angle_quality = max(0.0, 1.0 - abs(ang) / 0.5)
        touchdown_bonus = 3.0 * (1.0 + speed_quality * angle_quality)

    # 4. Stable landed: ongoing reward for maintaining a successful landing state
    stable_landed = 0.0
    if lc > 0.5 and rc > 0.5 and y < 0.3:
        speed_score = max(0.0, 1.0 - (abs(xv) + abs(yv)))
        angle_score = max(0.0, 1.0 - abs(ang))
        stable_landed = speed_score * angle_score

    # 5. Fuel penalty: small cost for any engine use
    fuel_penalty = -0.05 if action != 0 else 0.0

    total_reward = landing_proximity + descent_safety + touchdown_bonus + stable_landed + fuel_penalty

    components = {
        "landing_proximity": landing_proximity,
        "descent_safety": descent_safety,
        "touchdown_bonus": touchdown_bonus,
        "stable_landed": stable_landed,
        "fuel_penalty": fuel_penalty
    }

    return (float(total_reward), components)
```