# Response Record

```python
import numpy as np


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # 目标位置（平台中心）定义为 (0,0)，因为 obs[0], obs[1] 是相对偏移
    # ------------------------------------------------------------

    # 1. 主学习信号：朝向目标的进度
    dist_curr = np.sqrt(obs[0] ** 2 + obs[1] ** 2)
    dist_next = np.sqrt(next_obs[0] ** 2 + next_obs[1] ** 2)
    progress = 1.0 * (dist_curr - dist_next)  # 正值表示接近目标

    # 2. 稳定性约束：抑制高速、大角度和高角速度，确保着陆平稳
    linear_speed_cost = abs(next_obs[2]) + abs(next_obs[3])
    angle_cost = abs(next_obs[4])
    angular_cost = abs(next_obs[5])

    w_vel = 0.01
    w_angle = 0.01
    w_angvel = 0.005
    stability_penalty = (
        -w_vel * linear_speed_cost - w_angle * angle_cost - w_angvel * angular_cost
    )

    # 3. 着陆质量：连续有界信号，替代稀疏 soft_landing_bonus
    #    在目标附近提供渐进梯度，引导减速、调姿、双足接触
    speed_next = np.sqrt(next_obs[2] ** 2 + next_obs[3] ** 2)

    # 接近度：高斯衰减，dist=0 时值为 1，dist≈1 时降至 ~0.37
    proximity = np.exp(-(dist_next ** 2) / 1.0)

    # 速度质量：低速得高分，speed≈0.7 时降至 ~0.37
    speed_quality = np.exp(-(speed_next ** 2) / 0.5)

    # 角度质量：水平姿态得高分，|angle|≈0.3 时降至 ~0.37
    angle_quality = np.exp(-(next_obs[4] ** 2) / 0.1)

    # 接触质量：乘积形式要求双足同时接触才能显著得分
    contact_quality = next_obs[6] * next_obs[7]

    # 综合着陆质量：proximity 作为全局门控，内部三因子取均值
    landing_quality = (
        2.0 * proximity * (speed_quality + angle_quality + contact_quality) / 3.0
    )

    # 总奖励
    total_reward = progress + stability_penalty + landing_quality

    components = {
        "progress": progress,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality,
    }

    return float(total_reward), components
```
