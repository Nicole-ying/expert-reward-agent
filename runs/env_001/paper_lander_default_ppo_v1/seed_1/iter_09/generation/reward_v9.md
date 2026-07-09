## 1. Evidence
Current score 144.11 with best 162.94, soft_landing_proxy dominates at 643.31 episode sum (87.1% magnitude share) while progress_reward is negligible at 1.20 (0.2%), and episode length 305.30 vs best's 463.70 suggests the agent farms state-value proxy by lingering in good configurations without completing efficiently; last intervention adding fuel_penalty dropped score from best.

## 2. Behavior Diagnosis
The agent reaches the pad area and maintains good landing posture (high proximity, low speed, good angle, partial contact) but stalls or hovers instead of settling to termination, repeatedly collecting the persistent state-value soft_landing_proxy without converting it into actual task completion.

## 3. Signal Completeness
A completion-aligned signal is missing — soft_landing_proxy is a state-value reward that can be farmed by occupying good states, and no component incentivizes the agent to finish rather than linger; the proxy-to-completion alignment is broken.

## 4. Selected Level
Level 2 — state_to_improvement: the persistent state-value reward is being exploited by the policy stalling in proxy-optimal states, matching the `state_to_improvement` diagnosis exactly.

## 5. Selected Intervention
Transform `soft_landing_proxy(next_obs)` into `landing_improvement = 5.0 * (landing_quality(next_obs) - landing_quality(obs))`, converting the dominant proxy from a farmable state value to an improvement delta. No other component is changed. Working from best code (iter 5) without fuel_penalty.

## 6. Falsifiable Hypothesis
Paying for improvement rather than occupancy should eliminate the hovering incentive — the agent cannot collect reward by staying still regardless of configuration quality, and must instead continuously improve toward a settled landing state, resulting in faster termination and higher external task score.

## 7. Expected Next Round
`landing_improvement` magnitude_share will be far smaller than the original `soft_landing_proxy` (no more 643+ accumulation); episode length should decrease notably; external score should increase toward target; `terminated` episodes should still dominate but complete faster.

## 8. Main Risk
The per-step total reward may become weakly negative (progress ~0.004, speed_tracking ~-0.3, improvement ~0.1), potentially slowing credit assignment; improvement delta may be noisy step-to-step, creating gradient variance that delays convergence.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    观测:
    obs[0]: x_position_relative_to_target
    obs[1]: y_position_relative_to_pad_height
    obs[2]: x_velocity
    obs[3]: y_velocity
    obs[4]: body_angle
    obs[5]: angular_velocity
    obs[6]: left_contact_flag (0.0/1.0)
    obs[7]: right_contact_flag (0.0/1.0)
    """

    def distance(obs_arr):
        return (obs_arr[0] ** 2 + obs_arr[1] ** 2 + 1e-8) ** 0.5

    # ---- 1. Progress reward: 距离减少量 ----
    dist_old = distance(obs)
    dist_new = distance(next_obs)
    progress_reward = dist_old - dist_new

    # ---- 2. Speed tracking reward: 期望速度引导 ----
    max_speed = 5.0
    d_ref = 1.0
    desired_speed = max_speed * min(dist_new / d_ref, 1.0)

    cur_speed = (next_obs[2] ** 2 + next_obs[3] ** 2 + 1e-8) ** 0.5
    speed_error = abs(cur_speed - desired_speed)
    lambda_speed = 0.2
    speed_tracking_reward = -lambda_speed * speed_error

    # ---- 3. Landing improvement: 状态改善量 (state_to_improvement) ----
    proximity_threshold = 0.5
    speed_threshold = 0.25
    angle_threshold = 0.2

    def landing_quality(o):
        d = distance(o)
        s = (o[2] ** 2 + o[3] ** 2 + 1e-8) ** 0.5
        proximity_score = max(0.0, 1.0 - d / proximity_threshold)
        speed_low_score = max(0.0, 1.0 - s / speed_threshold)
        angle_score = max(0.0, 1.0 - abs(o[4]) / angle_threshold)
        contact_score = (o[6] + o[7]) * 0.5
        return proximity_score + speed_low_score + angle_score + 2.0 * contact_score

    quality_old = landing_quality(obs)
    quality_new = landing_quality(next_obs)
    landing_improvement = 5.0 * (quality_new - quality_old)

    # ---- 总奖励 ----
    total_reward = progress_reward + speed_tracking_reward + landing_improvement

    reward_components = {
        'progress_reward': progress_reward,
        'speed_tracking_reward': speed_tracking_reward,
        'landing_improvement': landing_improvement
    }
    return float(total_reward), reward_components
```