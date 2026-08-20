# Response Record

`evidence`：Current score -107.99，terminated=20/20，所有episode均在~68.5步提前坠毁。soft_landing_bonus active_rate仅0.9%（product form塌缩），episode_sum_mean=0.82但几乎不触发。上一轮将sparse binary改为连续product未改善（score从-106.27跌至-107.99），product形式未解决信号稀疏问题。

`behavior_diagnosis`：Agent在所有episode中坠毁，无法学会着陆。progress_reward=2.28说明它确实在靠近目标，但缺乏减速和姿态控制引导。soft_landing_bonus的乘积形式导致三个bounded因子相乘后几乎恒为零，无法为着陆动作提供可学习梯度。

`signal_completeness`：progress信号（接近）和stability信号（轻约束）均完备且活跃。缺失的是**可达的着陆引导信号**——当前乘积形式要求三个条件同时满足才能获得有意义的奖励，active_rate 0.9%证实其本质上仍是稀疏信号而非稠密引导。

`selected_level`：Level 2 — 触发条件为`product_to_noncollapsing_joint`：三个[0,1]因子乘积导致奖励几乎恒塌缩至零，单因子改善无法产生有意义反馈。

`selected_intervention`：将soft_landing_bonus从乘积 `proximity_score * speed_score * angle_score * 2.0` 改为几何平均 `(proximity_score * speed_score * angle_score) ** (1.0/3.0) * 2.0`。保持三个bounded因子和系数不变。

`falsifiable_hypothesis`：几何平均保留"任一因子为零则整体为零"的AND语义，但对中等值不再塌缩（如三者均为0.5时，乘积=0.125，GM=0.5），应显著提高active_rate并让agent在靠近目标时获得减速/调姿的连续梯度。

`expected_next_round`：soft_landing_bonus的active_rate应显著上升（从0.9%升至可观测水平如5-20%），episode_sum_mean适度上升。外部score若仍无改善（持续-107附近），说明着陆引导本身不是瓶颈，而是agent无法在坠毁前到达着陆条件激活区域。

`main_risk`：几何平均对单一维度极低值仍有较强惩罚（如0.9, 0.9, 0.05 → GM≈0.34），可能导致agent偏向将所有维度提升到勉强及格而非精进任一维度。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 主学习信号：distance-based progress（只奖励接近，不惩罚远离） ----
    prev_distance = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_distance = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = prev_distance - next_distance
    progress_reward = max(progress, 0.0) * 2.0

    # ---- 稳定/安全约束：轻量级惩罚 ----
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]

    stability_penalty_value = (
        0.01 * abs(vx)
        + 0.01 * abs(vy)
        + 0.01 * abs(angle)
        + 0.005 * abs(angular_vel)
    )
    stability_penalty = -stability_penalty_value

    # ---- 连续着陆逼近信号：几何平均替代乘积，防止塌缩 ----
    distance_to_pad = next_distance
    speed = (vx ** 2 + vy ** 2) ** 0.5
    angle_abs = abs(angle)

    proximity_score = max(0.0, 1.0 - distance_to_pad / 1.0)
    speed_score = max(0.0, 1.0 - speed / 0.5)
    angle_score = max(0.0, 1.0 - angle_abs / 0.3)

    # 几何平均：保留AND语义（任一为零则整体为零），但对中等值不再塌缩
    soft_landing_bonus = (proximity_score * speed_score * angle_score) ** (1.0 / 3.0) * 2.0

    # ---- 总奖励 ----
    total_reward = progress_reward + soft_landing_bonus + stability_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }

    return float(total_reward), components
```
