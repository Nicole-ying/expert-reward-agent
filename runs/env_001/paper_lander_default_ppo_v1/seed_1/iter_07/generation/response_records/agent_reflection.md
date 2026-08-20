# Response Record

`evidence`：iter 6 将 iter 5 的 state-value `soft_landing_proxy` 改为 state-to-improvement `landing_improvement` 后，episode 从 463 骤降至 69、全部 early terminal，score 从 162.94 跌至 -111.07；speed_tracking 负份额达 -65%，但这是崩溃后高速振荡的结果而非根因。best（iter 5）的 soft_landing_proxy 作为 state-value 成功引导了稳定着陆行为，但 episode 偏长（463 步）、progress_reward 近乎为零（0.004），提示 agent 可能在好状态中逗留。

`behavior_diagnosis`：iter 6 的 agent 快速崩溃——state-to-improvement 使靠近着陆垫后不再有维持奖励，导致接近阶段失去引导、姿态失控后坠毁。iter 5 的 agent 能成功着陆但效率不足，可能在着陆条件满足后不急于完成。

`signal_completeness`：现有信号缺少燃料效率约束。任务明确要求"尽可能少地使用引擎推力"，但所有历史奖励均未对此建模。progress + speed_tracking + soft_landing_proxy 覆盖了接近、减速、着陆质量，但缺少节油压力。

`selected_level`：Level 2——添加缺失的任务必要信号（fuel_penalty），同时从 iter 6 的 state-to-improvement 恢复为 iter 5 已验证的 state-value soft_landing_proxy。这是基于 best 骨架的一个新的、有证据的单组件干预。

`selected_intervention`：以 best（iter 5）为基础，新增 `fuel_penalty` 组件：对引擎点火动作（action 1/2/3）施加 -0.01 的固定惩罚，action 0（无推力）为 0。不修改其他三个组件。

`falsifiable_hypothesis`：燃料惩罚提供节油梯度，促使 agent 减少不必要的悬停和过度修正，从而在保持成功着陆的前提下缩短 episode 并提高 net score；同时 state-value soft_landing_proxy 恢复 iter 5 已验证的稳定着陆引导。

`expected_next_round`：score 应回升至正区间（目标 >0 并向 200 靠近），episode_length 相比 iter 5 的 463 应有所缩短，fuel_penalty episode_sum_mean 为负但 magnitude_share 不超过 10-20%，soft_landing_proxy 保持为主要正信号。

`main_risk`：燃料惩罚可能抑制早期探索中必要的引擎使用，导致 agent 不敢点火从而无法学习飞行控制；若 -0.01 过强，agent 可能选择全程无推力（直接坠落）而非学习高效着陆。

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

    # ---- 3. Soft landing proxy: 状态值着陆质量（state-value，best结构） ----
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

    # ---- 4. Fuel efficiency penalty: 引擎使用惩罚（新增） ----
    # action 0 = 无推力, action 1/2/3 = 引擎点火
    fuel_penalty = -0.01 if action in (1, 2, 3) else 0.0

    # ---- 总奖励 ----
    total_reward = progress_reward + speed_tracking_reward + soft_landing_proxy + fuel_penalty

    components = {
        'progress_reward': progress_reward,
        'speed_tracking_reward': speed_tracking_reward,
        'soft_landing_proxy': soft_landing_proxy,
        'fuel_penalty': fuel_penalty
    }
    return float(total_reward), components
```
