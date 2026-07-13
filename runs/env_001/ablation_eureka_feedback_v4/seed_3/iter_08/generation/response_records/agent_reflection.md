# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前状态与下一状态
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    prev_left, prev_right = obs[6], obs[7]
    next_left, next_right = next_obs[6], next_obs[7]

    # 1. Proximity: 向目标移动的距离减少量
    dist_curr = (x ** 2 + y ** 2) ** 0.5
    dist_next = (nx ** 2 + ny ** 2) ** 0.5
    progress = max(0.0, dist_curr - dist_next)

    # 2. Landing event: 一次性双足着陆事件奖励，只在从非双接触到双接触时发放
    landing_event = 0.0
    prev_both = (prev_left >= 0.5 and prev_right >= 0.5)
    next_both = (next_left >= 0.5 and next_right >= 0.5)
    if not prev_both and next_both:
        # 着陆质量因子：水平偏移、速度大小、机体角度
        horiz_factor = 1.0
        if abs(nx) > 0.2:
            horiz_factor = max(0.0, 1.0 - (abs(nx) - 0.2) / 0.3)

        speed = (nvx ** 2 + nvy ** 2) ** 0.5
        speed_factor = 1.0
        if speed > 0.3:
            speed_factor = max(0.0, 1.0 - (speed - 0.3) / 0.4)

        angle_factor = 1.0
        if abs(nangle) > 0.2:
            angle_factor = max(0.0, 1.0 - (abs(nangle) - 0.2) / 0.3)

        quality = horiz_factor * speed_factor * angle_factor
        landing_event = 10.0 * quality

    # 3. Energy penalty: 惩罚不必要的引擎使用
    energy_penalty = -0.01 if action != 0 else 0.0

    total_reward = progress + landing_event + energy_penalty

    components = {
        "proximity": progress,
        "landing_event": landing_event,
        "energy_penalty": energy_penalty
    }
    return float(total_reward), components
```
