## 诊断与修改理由

**问题分析：**
1. **stability_penalty 主导**：ratio_to_progress = -0.64（绝对值 > 0.5），惩罚信号主导了总奖励。agent 为避免惩罚而倾向于不行动，导致 episode 短（74步）且 score 差（-112）。
2. **soft_landing_proxy 稀疏**：nonzero_rate = 0.46%（< 1%），几乎从未触发。虽然当前代码已改为连续乘积形式，但 `both_contact_factor` 是硬性二值（0或1），导致整个乘积为0，没有梯度。
3. **历史对比**：当前代码相比历史最佳（iter 1），stability_penalty 从 -0.33 降到 -0.033（系数降了10倍），但 soft_landing_proxy 从 0.005 降到 0.003，说明连续乘积中的二值接触因子是瓶颈。

**修改方案（仅改一个组件）：**
- 将 `soft_landing_proxy` 中的 `both_contact_factor` 从硬性二值改为连续形式：`max(0.0, 1.0 - (1.0 - next_obs[6]) - (1.0 - next_obs[7]))` 或更简单的 `min(next_obs[6], next_obs[7])`，使接触信号提供连续梯度。
- 同时将 `proximity_factor` 的阈值从 0.5 放宽到 1.0，让梯度更早出现。

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
    # 距离因子：距离越近越接近1（放宽阈值到1.0，让梯度更早出现）
    current_dist = current_dist_sq ** 0.5
    near_target_threshold = 1.0  # 从0.5放宽到1.0
    proximity_factor = max(0.0, 1.0 - current_dist / near_target_threshold)
    
    # 速度因子：速度越低越接近1
    low_speed_threshold = 0.3
    low_speed_factor = max(0.0, 1.0 - speed / low_speed_threshold)
    
    # 姿态角因子：角度越小越接近1
    stable_angle_threshold = 0.2
    upright_factor = max(0.0, 1.0 - abs(body_angle) / stable_angle_threshold)
    
    # 双接触因子：改为连续形式，使用两个接触传感器的最小值
    # 这样当只有一个接触时也能提供部分梯度，而不是硬性0
    both_contact_factor = min(next_obs[6], next_obs[7])
    
    # 连续乘积：所有因子都在[0,1]区间，提供密集梯度
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