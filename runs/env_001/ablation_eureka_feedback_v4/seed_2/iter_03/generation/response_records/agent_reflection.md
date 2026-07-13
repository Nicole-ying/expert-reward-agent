# Response Record

# 设计理由
当前 `soft_landing` 组件使用 `contact_signal - speed_penalty - angle_penalty`，训练反馈显示 `soft_landing` 均值为负，意味着靠近目标时惩罚压制了接触奖励，导致 agent 在着陆阶段缺乏正向引导，全部 episode 都因失败终止。本修改将速度和角度惩罚改为正向奖励（奖励低速、正姿态），即 `max(0, safe_speed - speed_mag)` 等，使 `soft_landing` 始终非负，鼓励 agent 在接近目标时减速、摆正姿态并建立接触，同时不伤害已有的 `progress` 信号。系数保持原尺度（0.5），在 gate 激活区域（dist < 0.5）的 per‑step 正奖励远小于 2 倍 `progress`，不会淹没主要接近目标信号。

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
    speed_bonus = w_speed * max(0.0, safe_speed - speed_mag)

    safe_angle = 0.1
    w_angle = 0.5
    angle_abs = abs(n_angle)
    angle_bonus = w_angle * max(0.0, safe_angle - angle_abs)

    soft_landing = gate * (contact_signal + speed_bonus + angle_bonus)

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
- **evidence**: progress 均值 0.887，soft_landing 均值 -0.339，20/20 terminated，得分 -298。
- **behavior**: 飞行器持续向目标接近，但在着陆阶段无法减速/稳定姿态，最终坠毁或翻倒。
- **signal**: soft_landing 的惩罚形式使接近任务区的奖励塌缩为负值，缺乏正向着陆引导。
- **level**: Level 2，将速度和角度惩罚变换为低速/正姿态的正向奖励。
- **hypothesis**: 把着陆阶段的信号转为正向鼓励，可让 agent 学会减速并调整姿态，从而成功着陆并改善外部得分。
- **risk**: 若奖励过强，agent 可能在目标区附近徘徊而不立即接触；但 progress 会持续提供向心力，风险可控。
