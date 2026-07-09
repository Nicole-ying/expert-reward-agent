`evidence`：soft_landing_proxy 以 episode_sum_mean=1186.39 占 signed_share 90.5%，在 463.7 步长回合中按步累积，而 progress_reward 仅 1.24（0.1%）。15/20 terminated 表明多数回合最终着陆，但步长过长说明 agent 在近距离慢速徘徊以持续获取状态奖励。

`behavior_diagnosis`：agent 学会了在着陆垫附近保持近距离、低速、直立姿态来"刷取"soft_landing_proxy 的按步奖励（hover farming），而非高效完成着陆。这是典型的 proxy-optimal 行为：proxy 高分但任务完成效率差。

`signal_completeness`：progress_reward 承担远距离引导职责但被淹没；speed_tracking_reward 提供速度约束但持续为负；着陆奖励以状态值形式存在，缺少"改善量"语义，使 agent 可以在好状态停滞而不推进。

`selected_level`：Level 2 — `state_to_improvement`。证据直接否定当前数学形态：soft_landing_proxy 支付"处于好状态"的每步奖励，导致 episode 极长且 proxy 份额畸形。结构上需要从状态值变为状态改善量。

`selected_intervention`：将 `soft_landing_proxy` 从 `Q(next_obs)` 变换为 `coefficient × max(0, Q(next_obs) − Q(obs))`，即着陆质量改善量。Q 函数保持相同的 proximity/speed/angle/contact 线性组合，系数设为 5.0 使单次完整着陆的改善累积量（约 25）具有足够引导力但远小于原来的 1186。

`falsifiable_hypothesis`：移除"原地保持好状态即可获奖"的激励后，agent 必须通过实际推进着陆质量来获取奖励，因此 episode 长度应显著下降，徘徊行为减少，proxy 份额大幅降低。

`expected_next_round`：episode_length 下降（不再徘徊），landing_improvement 的 episode_sum_mean 远低于当前 1186，progress_reward 相对份额上升，score 可能短期波动但最终应改善。

`main_risk`：改善量奖励在远距离处为 0（proximity 阈值 0.5 以外 Q=0），可能导致终段引导信号出现"悬崖"——agent 在距离 0.5 附近得不到改善奖励时可能犹豫，需下一轮验证是否需要放宽 proximity 阈值或使用无界势能差。

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

    # ---- 2. Speed tracking reward ----
    max_speed = 5.0
    d_ref = 1.0
    desired_speed = max_speed * min(dist_new / d_ref, 1.0)

    cur_speed = (next_obs[2] ** 2 + next_obs[3] ** 2 + 1e-8) ** 0.5
    speed_error = abs(cur_speed - desired_speed)
    lambda_speed = 0.2
    speed_tracking_reward = -lambda_speed * speed_error

    # ---- 3. Landing improvement: 着陆质量改善量 (state_to_improvement) ----
    proximity_threshold = 0.5
    speed_threshold = 0.25
    angle_threshold = 0.2

    def landing_quality(o):
        d = (o[0] ** 2 + o[1] ** 2 + 1e-8) ** 0.5
        s = (o[2] ** 2 + o[3] ** 2 + 1e-8) ** 0.5
        prox = max(0.0, 1.0 - d / proximity_threshold)
        slow = max(0.0, 1.0 - s / speed_threshold)
        ang = max(0.0, 1.0 - abs(o[4]) / angle_threshold)
        contact = (o[6] + o[7]) * 0.5
        return prox + slow + ang + 2.0 * contact

    Q_old = landing_quality(obs)
    Q_new = landing_quality(next_obs)

    landing_improvement_coeff = 5.0
    landing_improvement = landing_improvement_coeff * max(0.0, Q_new - Q_old)

    # ---- 总奖励 ----
    total_reward = progress_reward + speed_tracking_reward + landing_improvement

    components = {
        'progress_reward': progress_reward,
        'speed_tracking_reward': speed_tracking_reward,
        'landing_improvement': landing_improvement
    }
    return float(total_reward), components
```