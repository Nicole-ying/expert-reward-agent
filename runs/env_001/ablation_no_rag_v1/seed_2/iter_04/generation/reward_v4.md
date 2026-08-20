```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：每一步向目标中心靠近的 progress
    dist_old = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_new = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = dist_old - dist_new

    # 着陆引导奖励：从乘积塌缩结构改为加性结构，无接触时也能获得部分奖励
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 比之前更宽松的尺度，让奖励在较远位置就开始提供引导
    sigma_p = 1.5
    sigma_v = 1.5
    sigma_a = 0.5

    exponent = -(
        (x ** 2 + y ** 2) / (2.0 * sigma_p ** 2) +
        (vx ** 2 + vy ** 2) / (2.0 * sigma_v ** 2) +
        (angle ** 2) / (2.0 * sigma_a ** 2)
    )
    stability_score = 2.718281828 ** exponent

    # 基础稳定性奖励，不依赖双脚接触
    base_stability = 5.0 * stability_score

    # 接触额外奖励，鼓励最终着地
    contact_factor = (left_contact + right_contact) / 2.0
    contact_bonus = 5.0 * contact_factor * stability_score

    stable_landing_reward = base_stability + contact_bonus

    total_reward = progress_reward + stable_landing_reward

    components = {
        "progress_reward": progress_reward,
        "stable_landing_reward": stable_landing_reward
    }

    return float(total_reward), components
```