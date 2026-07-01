# Response Record

## 诊断与修改理由

**问题1：stability_penalty 严重主导（ratio_to_progress = -6.38）**
惩罚信号是 progress 信号的 6 倍多，agent 为避免惩罚而不敢行动，导致 episode 极短（74步，100%提前终止）。需要大幅降低惩罚系数（至少10倍）。

**问题2：soft_landing_proxy 几乎从不触发（nonzero_rate=0.46%）**
四个硬性条件同时满足的概率极低，无法提供有效引导。改为连续乘积形式，每个因子用 bounded 函数。

**修改方案：**
1. 将 stability_penalty 的三个子项系数各降低 10 倍（0.3→0.03, 0.2→0.02, 0.1→0.01）
2. 将 soft_landing_proxy 从硬性 if 条件改为连续乘积：`proximity_factor * low_speed_factor * upright_factor * both_contact_factor`，每个因子用 `max(0, 1 - x/threshold)` 形式 bounded 在 [0,1]

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward ==========
    # 目标位置假设为 (0, 0)，因为 obs[0] 和 obs[1] 是相对于目标着陆平台的坐标
    # 计算当前距离平方和下一时刻距离平方
    current_dist_sq = obs[0] ** 2 + obs[1] ** 2
    next_dist_sq = next_obs[0] ** 2 + next_obs[1] ** 2
    
    # progress_delta: 正数表示更接近目标
    progress_delta = current_dist_sq - next_dist_sq
    
    # 缩放因子，使奖励值在合理范围
    progress_scale = 2.0
    progress_delta_reward = progress_scale * progress_delta
    
    # ========== 稳定/安全约束：stability_penalty ==========
    # 惩罚高速、大姿态角和大角速度
    # 使用 next_obs 因为动作效果体现在下一状态
    vel_x = next_obs[2]
    vel_y = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # 速度惩罚：鼓励减速接近目标（系数降低10倍，从0.3→0.03）
    speed = (vel_x ** 2 + vel_y ** 2) ** 0.5
    speed_penalty_weight = 0.03
    speed_penalty = -speed_penalty_weight * speed
    
    # 姿态角惩罚：鼓励保持水平姿态（系数降低10倍，从0.2→0.02）
    angle_penalty_weight = 0.02
    angle_penalty = -angle_penalty_weight * abs(body_angle)
    
    # 角速度惩罚：鼓励稳定姿态（系数降低10倍，从0.1→0.01）
    angular_vel_penalty_weight = 0.01
    angular_vel_penalty = -angular_vel_penalty_weight * abs(angular_vel)
    
    stability_penalty = speed_penalty + angle_penalty + angular_vel_penalty
    
    # ========== 任务完成 proxy：soft_landing_proxy ==========
    # 改为连续乘积形式，每个因子 bounded 在 [0,1]，提供密集梯度引导
    # 距离因子：距离越近越接近1
    current_dist = current_dist_sq ** 0.5
    near_target_threshold = 0.5
    proximity_factor = max(0.0, 1.0 - current_dist / near_target_threshold)
    
    # 速度因子：速度越低越接近1
    low_speed_threshold = 0.3
    low_speed_factor = max(0.0, 1.0 - speed / low_speed_threshold)
    
    # 姿态角因子：角度越小越接近1
    stable_angle_threshold = 0.2
    upright_factor = max(0.0, 1.0 - abs(body_angle) / stable_angle_threshold)
    
    # 双接触因子：两个接触传感器都激活时接近1
    both_contact_factor = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    both_contact_factor = 1.0 if both_contact_factor else 0.0
    
    # 连续乘积：只有所有条件都满足时才接近1，但每个因子单独提供梯度
    soft_landing_proxy = proximity_factor * low_speed_factor * upright_factor * both_contact_factor
    
    # ========== 组合总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy
    
    # ========== 构建 components dict ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```
