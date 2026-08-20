# Response Record

`evidence`：所有20个评估episode全部truncated（1000步），无early termination，策略已学会稳定存活；forward_progress以2170的episode_sum_mean占signed_share的95.8%，是唯一正向驱动信号；stability惩罚合计仅约4.1% magnitude_share，未抑制探索；但得分3168距目标3800差632分（83.4%），说明策略收敛到一个保守步态——能稳定前进但速度不足，缺少推动更高动态跳跃的信号。

`behavior_diagnosis`：策略找到了一个稳定但次优的步态局部最优解——躯干保持直立、持续中低速前进，但无法突破到更高速度，可能在进行低振幅" shuffling"而非高能跳跃，因为当前奖励对高度/弹跳质量完全没有反馈。

`signal_completeness`：缺少跳跃高度相关信号。在单腿跳跃任务中，前进速度与跳跃高度/弹跳动力学高度耦合——仅有前进速度奖励无法区分低振幅shuffling和高能跳跃。`obs[0]`（torso_height）是可用的直接信号，当前未被使用。这属于缺失的必要过程引导职责。

`selected_level`：Level 2。需要添加一个新的信号类型（height-based reward），而非仅调整现有组件的系数。已有证据表明策略陷入保守步态平台期，单一前向速度信号不足以引导更动态的跳跃行为。

`selected_intervention`：新增`height_reward`组件——基于躯干高度与目标值的偏差给予有界正向奖励。使用三角形核函数：`max(0, 1 - |height - target| / tolerance)`，target=1.0，tolerance=0.5，weight=0.5。保持其他三个现有组件不变。

`falsifiable_hypothesis`：添加高度奖励后，策略将被引导到更高的跳跃振幅和更有力的着地推蹬，从而连带提高前向速度均值；预期forward_progress的episode_sum_mean上升，height_reward获得非零active_rate且为正贡献，外部score向3800靠近。

`expected_next_round`：height_reward应活跃（active_rate接近100%），episode_sum_mean为正（预期在0.2-0.5范围，取决于实际高度分布）；forward_progress应上升（目标从2170→2400+）；总score应上升；episode仍应全部truncated（保持稳定）。

`main_risk`：若target_height=1.0偏离环境实际健康高度中心，height_reward可能引导策略偏离真正高效的运动姿态，反而降低前向速度；tolerance=0.5若过窄也可能限制正常跳跃振幅。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    V2 reward: adds height-based shaping to encourage dynamic hopping.
    Drives forward progress while keeping the torso upright and at a healthy hopping height.
    """
    # Weights
    w_fwd = 1.0          # main progress drive
    w_angle = 0.5        # torso tilt penalty
    w_angvel = 0.1       # torso angular velocity penalty
    w_height = 0.5       # height tracking reward

    # Main learning signal: forward velocity (after transition)
    forward_progress = w_fwd * max(0.0, next_obs[5])

    # Stability constraints based on next observation
    torso_angle_penalty = -w_angle * abs(next_obs[1])
    torso_angular_velocity_penalty = -w_angvel * abs(next_obs[7])

    # Height tracking: reward proximity to target hopping height
    # Uses a triangular kernel: full reward at target, linearly decreasing to 0 at target +/- tolerance
    target_height = 1.0
    tolerance = 0.5
    height_deviation = abs(next_obs[0] - target_height)
    height_reward = w_height * max(0.0, 1.0 - height_deviation / tolerance)

    total_reward = forward_progress + torso_angle_penalty + torso_angular_velocity_penalty + height_reward

    components = {
        "forward_progress": forward_progress,
        "torso_angle_penalty": torso_angle_penalty,
        "torso_angular_velocity_penalty": torso_angular_velocity_penalty,
        "height_reward": height_reward,
    }

    return float(total_reward), components
```
