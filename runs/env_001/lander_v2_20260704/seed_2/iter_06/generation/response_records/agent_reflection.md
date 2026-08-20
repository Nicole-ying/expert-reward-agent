# Response Record

`evidence`：上一轮得分-109.22未刷新best(-108.37)，early_terminal 20/20全部快速终止(≈68步)；weighted_stability_penalty占magnitude_share 68.9%压制progress_delta(31.1%)，|penalty/progress|≈2.3远超v6经验阈值0.5；iter 3 best代码中stability仅占约13%，且有landing_quality(0.139)提供着陆引导，当前代码完全缺失着陆信号。

`behavior_diagnosis`：agent在所有episode中约68步即终止，处于快速失败模式；稳定性惩罚过强迫使策略不敢执行有效机动，同时缺少着陆阶段引导信号使末端行为盲目。

`signal_completeness`：当前奖励缺少着陆/终端引导职责——progress_delta只能鼓励接近，无法区分"悬停在上方"与"稳定着陆"，且越靠近目标progress增量越小，稳定性惩罚反而因altitude_factor放大，形成末端信号塌缩。

`selected_level`：Level 2——着陆信号缺失属于必要职责不完备，且Level 1单纯降稳定性系数无法弥补缺失的终端引导。

`selected_intervention`：以best代码为基础，将landing_quality从乘积几何平均(product^0.2)改为和基联合满足(sum_of_factors)，保留distance_reward和light stability_penalty不变。

`falsifiable_hypothesis`：和基结构在任一因子为零时仍对其他因子保留梯度，避免乘积塌缩导致的信用分配中断，应使landing_quality更稳定地提供着陆引导，从而改善末端行为。

`expected_next_round`：landing_quality的active_rate应保持高值(>90%)，magnitude_share应显著高于iter 3的~14%，整体score应刷新best且early_terminal比例应下降。

`main_risk`：和基允许因子间补偿(如高速但姿态好仍得分)，可能引导agent以不安全的着陆姿态刷分；若出现此现象需改用soft-min或门控结构。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observables
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Distance reward: continuous negative Euclidean distance to goal
    distance_to_target = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -1.0 * distance_to_target

    # 2. Light stability penalty (preserved from best)
    stability_penalty = -(
        0.15 * abs(x_vel) +
        0.05 * abs(y_vel) +
        0.2 * abs(body_angle) +
        0.2 * abs(angular_vel)
    )

    # 3. Landing quality: sum-based joint satisfaction
    #    Uses additive combination instead of product to prevent gradient collapse
    #    when any single factor is zero during early approach.
    prox_factor = max(0.0, 1.0 - distance_to_target / 2.0)
    speed_x_factor = max(0.0, 1.0 - abs(x_vel) / 0.8)
    speed_y_factor = max(0.0, 1.0 - abs(y_vel) / 0.8)
    angle_factor = max(0.0, 1.0 - abs(body_angle) / 0.4)
    contact_factor = 0.2 + 0.8 * (left_contact + right_contact) / 2.0

    sum_of_factors = prox_factor + speed_x_factor + speed_y_factor + angle_factor + contact_factor
    landing_quality = 0.2 * sum_of_factors  # max=1.0 when all conditions perfect

    total_reward = distance_reward + stability_penalty + landing_quality

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components
```
