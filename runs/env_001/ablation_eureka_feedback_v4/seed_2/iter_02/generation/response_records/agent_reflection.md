# Response Record

# 设计理由
我将 `soft_landing` 组件内部的惩罚从无界二次惩罚改为 **hinge 形式**，并让接触奖励从二值乘积改为 **连续和**。原因：当前 gate 区域内的二次速度/角度惩罚在 agent 接近目标但未充分减速时产生巨大负值，与 `progress` 的正奖励冲突，导致靠近目标的行为被压制，着陆失败。新设计仅在速度或角度超出安全阈值时才产生适度惩罚，把接触信号从稀有事件变成连续梯度，使 agent 可以在着陆阶段逐步学会减速、正姿和双足接触。系数按主进展信号每步 ~0.85 的量级校准，惩罚预期值控制在 0.1~0.15 范围。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前状态
    x, y = obs[0], obs[1]
    # 下一状态
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_lc = next_obs[6]
    n_rc = next_obs[7]

    # 距离
    dist_old = (x**2 + y**2)**0.5
    dist_new = (nx**2 + ny**2)**0.5

    # 1. 目标进展
    w_progress = 1.0
    progress = w_progress * (dist_old - dist_new)

    # 2. 软着陆：靠近目标时激活，鼓励低速、正姿态、双足接触
    landing_threshold = 0.5
    gate = max(0.0, 1.0 - dist_new / landing_threshold)

    w_contact = 0.5
    contact_signal = w_contact * (n_lc + n_rc)   # 连续梯度，最大 1.0

    safe_speed = 0.2
    w_speed = 0.5
    speed_mag = (nvx**2 + nvy**2)**0.5
    speed_penalty = w_speed * max(0.0, speed_mag - safe_speed)

    safe_angle = 0.1
    w_angle = 0.5
    angle_abs = abs(n_angle)
    angle_penalty = w_angle * max(0.0, angle_abs - safe_angle)

    soft_landing = gate * (contact_signal - speed_penalty - angle_penalty)

    # 3. 燃料效率
    w_fuel = 0.02
    fuel_cost = -w_fuel * float(action != 0)

    total_reward = progress + soft_landing + fuel_cost
    components = {
        "progress": progress,
        "soft_landing": soft_landing,
        "fuel_cost": fuel_cost
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: terminated=20/20, len=86.15, score=-285.17; progress=+0.85/step 但 soft_landing=-0.21（gate 内高速高角度惩罚压制靠近）
- **behavior**: agent 快速冲向目标，接近时速度过大、姿态失控，导致 crash
- **signal**: soft_landing 在着陆阶段产生巨幅负奖励，与 progress 冲突，无法引导减速和姿态稳定
- **level**: Level 2（数学结构变换：惩罚从二次→hinge，接触奖励从二值乘积→连续和）
- **hypothesis**: hinge 惩罚在安全区内完全消失、越界才逐步增强，使 gate 区域的净奖励由接触信号主导，agent 将学会在靠近时主动减速、扶正并寻求双足接触
- **risk**: 如果 safe_speed/safe_angle 设置过松，agent 可能以勉强可接受的速度着陆但仍失败；后续可根据 terminated 率进一步收紧阈值
