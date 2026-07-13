# Response Record

# 设计理由
当前 `landing_quality` 组件从未触发（active_rate=0），因为条件 `both_contact` 要求两条腿同时接触，这使着陆信号完全稀疏，缺乏梯度引导。我将其中的二值因子 `both_contact` 替换为连续因子 `(contactL + contactR) / 2.0`，这样只要任意一条腿触地就能获得部分奖励，同时保留与距离、速度负相关的平滑因子，鼓励在靠近且低速时完成最终接触。其他组件（速度惩罚、角度稳定等）保持不变，只改了这一个组件的数学形态（稀疏二值 → 连续 proxy），符合 Level 2 干预。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current and next positions relative to landing pad center
    dx_curr, dy_curr = obs[0], obs[1]
    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_curr = (dx_curr**2 + dy_curr**2) ** 0.5
    dist_next = (dx_next**2 + dy_next**2) ** 0.5

    # Approach progress: positive when getting closer to the pad
    approach_delta = dist_curr - dist_next

    # Next‑step velocity and safety threshold
    vx_next, vy_next = next_obs[2], next_obs[3]
    speed_next = (vx_next**2 + vy_next**2) ** 0.5
    safe_speed = 0.2

    # Next‑step orientation and angular velocity
    angle_next = next_obs[4]
    angvel_next = next_obs[5]

    # Contact signals after the step
    contactL = next_obs[6]
    contactR = next_obs[7]

    # --- Reward weights ---
    w_approach = 2.0
    w_vel_penalty = 5.0           # per unit speed above safe_speed
    w_angle = 0.5                 # penalty on squared body angle
    w_angvel = 0.1                # penalty on squared angular velocity
    w_landing = 5.0               # bonus for landing (now continuous factor)

    # Component 1: dense progress towards the pad
    approach_reward = w_approach * approach_delta

    # Component 2: speed constraint (hinge – only penalise when exceeding safe threshold)
    vel_penalty = -w_vel_penalty * max(0.0, speed_next - safe_speed)

    # Component 3: angular stability (quadratic penalties)
    angle_stability = -w_angle * (angle_next**2) - w_angvel * (angvel_next**2)

    # Component 4: soft landing‑quality proxy (continuous contact factor)
    dist_factor = 1.0 / (1.0 + 1.0 * dist_next)
    speed_factor = 1.0 / (1.0 + 1.0 * speed_next)
    # Replace binary both_contact with continuous contact factor
    contact_factor = (contactL + contactR) / 2.0
    landing_reward = w_landing * dist_factor * speed_factor * contact_factor

    total_reward = approach_reward + vel_penalty + angle_stability + landing_reward

    components = {
        "approach_progress": approach_reward,
        "velocity_penalty": vel_penalty,
        "angle_stability": angle_stability,
        "landing_quality": landing_reward,
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: `landing_quality` mean=0.0 (active_rate=0%)，20 条轨迹全部 truncated，平均长度 1000，无成功着陆。  
- **behavior**: agent 以接近 0.2 的速度向目标靠近，但从未做到双腿同时触地，缺乏着陆梯度导致无法完成最终接触。  
- **signal**: 着陆信号完全缺失（二进制 `both_contact`），没有中间态奖励来引导接触行为。  
- **level**: Level 2（数学结构变换）—— 将稀疏二值接触条件替换为连续接触因子 `(contactL+contactR)/2`。  
- **hypothesis**: 连续的接触梯度能使 agent 在单腿触地时获得正向信号，从而学习调节姿态完成双腿着陆。  
- **risk**: 单腿触地后可能形成局部最优，但随距离和速度因子衰减，最终仍需两条腿同时触地才能获得最大奖励。
