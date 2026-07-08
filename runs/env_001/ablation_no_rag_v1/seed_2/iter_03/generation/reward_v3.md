```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：向目标中心的 progress（potential-based）
    dist_old = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_new = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = dist_old - dist_new

    # 稳定着陆引导：连续乘积状态奖励，消除 transition exploit
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 接触因子：平均双脚接触（0~1），鼓励双脚均着地
    contact_factor = (left_contact + right_contact) / 2.0

    # curriculum：早期使用宽松的高斯尺度，后期收紧以要求精确
    sigma_p = 2.0 - 1.5 * training_progress  # 位置尺度：2.0 → 0.5
    sigma_v = 2.0 - 1.5 * training_progress  # 速度尺度：2.0 → 0.5
    sigma_a = 0.6 - 0.45 * training_progress  # 角度尺度：0.6 → 0.15

    # 组合指数项，同时要求靠近、低速、竖直
    exponent = -(
        (x ** 2 + y ** 2) / (2.0 * sigma_p ** 2) +
        (vx ** 2 + vy ** 2) / (2.0 * sigma_v ** 2) +
        (angle ** 2) / (2.0 * sigma_a ** 2)
    )

    # 使用 2.718281828**exponent 替代 exp；系数 10.0 提供足够推动力
    stable_landing_reward = 10.0 * contact_factor * (2.718281828 ** exponent)

    total_reward = progress_reward + stable_landing_reward

    components = {
        "progress_reward": progress_reward,
        "stable_landing_reward": stable_landing_reward
    }

    return float(total_reward), components
```