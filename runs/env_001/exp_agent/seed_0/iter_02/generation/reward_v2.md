## 诊断

从训练反馈看，问题非常清晰：

1. **stability_penalty 主导**：ratio_to_progress = -3.37，绝对值远大于 0.5。惩罚信号完全压倒了正向信号，agent 被吓到不敢动，episode 短（74步），score 差（-115）。

2. **soft_landing_proxy 几乎不触发**：nonzero_rate = 0.5%，二值条件太苛刻，无法提供学习引导。

3. **progress_delta_reward 正常**：nonzero_rate ≈ 100%，但被惩罚淹没。

**修改方案：**

**层次 1：改系数 + 改表达式**

- **stability_penalty**：系数削 10 倍（speed_penalty_weight 0.1→0.01，angle_penalty_weight 0.05→0.005，angular_vel_penalty_weight 0.02→0.002），并添加距离门控（远处不罚，只在靠近目标时约束）。
- **soft_landing_proxy**：从二值条件改为连续乘积形式，每个因子用 max(0, 1 - x/threshold)，提供密集梯度。
- **progress_delta_reward**：保持，但 scale 从 2.0 降到 1.0 避免与 bounded 信号冲突。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 诊断与修改说明：
    # 1. stability_penalty 的 ratio_to_progress = -3.37，完全主导训练，
    #    导致 agent 不敢动。系数削 10 倍 + 距离门控（远处不罚）。
    # 2. soft_landing_proxy 的 nonzero_rate = 0.5%，二值条件无梯度，
    #    改为连续乘积形式 max(0, 1-x/threshold) 提供密集信号。
    # 3. progress_delta_reward 正常但 scale 略降，避免与 bounded 信号冲突。
    # ============================================================
    
    # 当前距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    # 下一步距离
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    
    # ============================================================
    # 1. 主学习信号: progress_delta_reward
    #    基于距离变化引导接近目标
    # ============================================================
    progress_delta = current_dist - next_dist
    progress_scale = 1.0  # 略降，避免与 bounded 信号冲突
    progress_delta_reward = progress_delta * progress_scale

    # ============================================================
    # 2. 稳定/安全约束: stability_penalty（距离门控 + 系数削10倍）
    #    远处允许大动作探索，只在靠近目标时约束稳定性
    # ============================================================
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle = abs(next_obs[4])
    angular_vel = abs(next_obs[5])
    
    # 距离门控因子：远处不罚，靠近时逐渐生效
    # gate_radius = 2.0（初始距离约 3-4，取 50% 左右）
    gate_radius = 2.0
    distance_gate = max(0.0, 1.0 - next_dist / gate_radius)
    
    # 系数削 10 倍
    speed_penalty_weight = 0.01
    angle_penalty_weight = 0.005
    angular_vel_penalty_weight = 0.002
    
    speed_penalty = -speed_penalty_weight * speed * distance_gate
    angle_penalty = -angle_penalty_weight * angle * distance_gate
    angular_vel_penalty = -angular_vel_penalty_weight * angular_vel * distance_gate
    
    stability_penalty = speed_penalty + angle_penalty + angular_vel_penalty

    # ============================================================
    # 3. 任务完成proxy: soft_landing_proxy（连续乘积形式）
    #    用 max(0, 1-x/threshold) 替代二值条件，提供密集梯度
    # ============================================================
    # 距离因子：距离 < 0.5 时接近 1，> 1.0 时接近 0
    dist_factor = max(0.0, 1.0 - next_dist / 1.0)
    # 速度因子：速度 < 0.3 时接近 1，> 0.6 时接近 0
    speed_factor = max(0.0, 1.0 - speed / 0.6)
    # 姿态角因子：角度 < 0.2 时接近 1，> 0.4 时接近 0
    angle_factor = max(0.0, 1.0 - angle / 0.4)
    # 接触因子：双支撑接触的连续版本（>0.5 视为接触）
    contact_factor = min(1.0, (next_obs[6] + next_obs[7]) / 2.0)
    
    # 连续乘积，自动 bounded 在 [0,1]
    landing_bonus_weight = 2.0
    soft_landing_proxy = landing_bonus_weight * dist_factor * speed_factor * angle_factor * contact_factor

    # ============================================================
    # 4. 动作代价: energy_penalty（小权重）
    # ============================================================
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty_weight = 0.01
    energy_penalty = -energy_penalty_weight * engine_use

    # ============================================================
    # 总奖励
    # ============================================================
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy + energy_penalty

    # 组件字典
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components
```