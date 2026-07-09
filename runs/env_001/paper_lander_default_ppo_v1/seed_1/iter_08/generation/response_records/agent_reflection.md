# Response Record

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

    # ---- 3. Soft landing proxy: 状态值着陆质量 ----
    proximity_threshold = 0.5
    speed_threshold = 0.25
    angle_threshold = 0.2

    proximity_score = max(0.0, 1.0 - dist_new / proximity_threshold)
    speed_low_score = max(0.0, 1.0 - cur_speed / speed_threshold)
    angle_score = max(0.0, 1.0 - abs(next_obs[4]) / angle_threshold)
    contact_score = (next_obs[6] + next_obs[7]) * 0.5

    soft_landing_proxy = (
        proximity_score +
        speed_low_score +
        angle_score +
        2.0 * contact_score
    )

    # ---- 4. Fuel efficiency penalty: 引擎使用惩罚 ----
    fuel_penalty = -0.01 if action in (1, 2, 3) else 0.0

    # ---- 总奖励 ----
    total_reward = progress_reward + speed_tracking_reward + soft_landing_proxy + fuel_penalty

    reward_components = {
        'progress_reward': progress_reward,
        'speed_tracking_reward': speed_tracking_reward,
        'soft_landing_proxy': soft_landing_proxy,
        'fuel_penalty': fuel_penalty
    }
    return float(total_reward), reward_components
```
