# Response Record

1. evidence：上一轮代码使用旧状态 obs 而非 next_obs 计算速度门控和前进奖励，导致 forward_reward 的 episode_sum_mean 异常高达 61419，而外部 score 为 -48.96，两者严重背离，表明奖励信号与真实任务目标错位；同时所有 episode 均存活至 1000 步（truncated=20/20），说明该结构虽能维持生存但未驱动有效前进。

2. behavior_diagnosis：策略很可能利用基于旧状态的错误信用分配，通过非前进的摆动或振荡获取巨大的虚拟前进速度奖励，实际位移微小，致使外部评分极低。

3. signal_completeness：代码中包含前进速度引导、高度及直立健康门控、侧向漂移惩罚和角速度惩罚，基本职责齐备，但因未基于动作后的 next_obs 计算奖励，导致前进信号无法与真实行为对齐，必要职责虽可达但被错误状态源破坏。

4. selected_level：Level 2（结构变换），将奖励计算从 obs 切换到 next_obs，以纠正信用分配错误；这符合“修正观测源”这类最小但可验证的数学结构修改。

5. selected_intervention：唯一目标组件 forward_reward。修改其计算：用 next_obs 替换 obs 来获取 body_x_vel 以及计算 combined_gate（高度和直立），其他结构、系数和惩罚项保持不变。

6. falsifiable_hypothesis：修复观测源后，forward_reward 将如实反映动作产生的真实前进速度，策略转而追求实际前行，外部 score 应转为正值并显著超过当前 -48.96，同时 episode 长度仍能维持在 1000 左右（持续存活）。

7. expected_next_round：forward_reward 的 episode_sum_mean 将大幅下降至与真实速度累积匹配的水平；score 转正且有望超过之前 best 26.07；lat_penalty 和 ang_penalty 仍保持小幅贡献；episode_length 继续保持或接近 1000。

8. main_risk：门控乘法可能导致早期速度奖励稀疏，但凭借健康区域的速度梯度仍可学习；奖励尺度若仍偏大可能造成数值不稳定，但首轮先观察，下一轮再视情况调整。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 提取 next_obs 的状态 ---
    body_z = next_obs[0]
    quat_w = next_obs[1]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]
    # 速度
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]
    roll_vel  = next_obs[16]
    pitch_vel = next_obs[17]
    yaw_vel   = next_obs[18]

    # --- 健康门控因子（基于 next_obs）---
    # 1. 高度健康：body_z >= 0.5 完全健康，0.5->0.2 线性衰减到 0
    if body_z >= 0.5:
        health_gate = 1.0
    elif body_z <= 0.2:
        health_gate = 0.0
    else:
        health_gate = (body_z - 0.2) / 0.3

    # 2. 直立健康
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    if body_up_z >= 0.5:
        upright_gate = 1.0
    else:
        upright_gate = max(0.0, body_up_z / 0.5)

    combined_gate = health_gate * upright_gate

    # --- 奖励组件 ---
    w_fwd = 1000.0
    forward_reward = w_fwd * body_x_vel * combined_gate

    w_lat = 2.0
    lat_penalty = w_lat * body_y_vel**2

    w_ang = 0.5
    ang_penalty = w_ang * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)

    total_reward = forward_reward - lat_penalty - ang_penalty

    components = {
        'forward_reward': forward_reward,
        'lat_penalty': -lat_penalty,
        'ang_penalty': -ang_penalty
    }

    return float(total_reward), components
```
