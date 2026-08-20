## 诊断

**核心问题：stability_penalty 严重主导，ratio_to_progress = -3.45，是 progress 的 3.45 倍。** agent 为了避免惩罚而选择"不动"，导致 episode 极短（71.9步），外部得分极低（-111.68）。这是典型的 penalty_dominance 症状。

**次要问题：soft_landing_bonus 的 nonzero_rate = 0.54%，几乎从不触发。** 二值 if 条件太苛刻，无法提供梯度引导。需要改为连续乘积形式。

**修改计划（每次只改一个信号）：**

本轮先修 **stability_penalty**（最严重的问题）：
1. 大幅降低系数（至少10倍），从 -0.5*speed - 0.3*angle - 0.2*angular_vel 改为 -0.05*speed - 0.03*angle - 0.02*angular_vel
2. 同时加入距离门控：只在靠近目标（dist < 2.0）时生效，远处不约束

**不修改其他组件**，让下一轮反馈单独告诉我们这个改动是否有效。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward ==========
    # 计算到目标(0,0)的距离变化
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist  # 正数表示更接近目标
    progress_delta_reward = 10.0 * progress_delta  # 权重10，鼓励每一步都更接近

    # ========== 稳定/安全约束：stability_penalty（已大幅削弱 + 距离门控） ==========
    # 诊断：上一轮 stability_penalty ratio=-3.45，严重主导，agent 不敢动
    # 修复1：系数降低10倍（从0.5/0.3/0.2 → 0.05/0.03/0.02）
    # 修复2：加入距离门控，只在靠近目标（dist<2.0）时生效，远处不约束
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = abs(next_obs[4])  # 姿态角偏离0的绝对值
    angular_vel_penalty = abs(next_obs[5])  # 角速度绝对值
    
    # 距离门控因子：dist<2.0 时逐渐生效，dist>=2.0 时完全关闭
    gate_radius = 2.0
    distance_gate = max(0.0, 1.0 - next_dist / gate_radius)
    
    # 大幅降低系数（10倍），并应用距离门控
    stability_penalty = -distance_gate * (0.05 * speed + 0.03 * angle_penalty + 0.02 * angular_vel_penalty)

    # ========== 任务完成 proxy：soft_landing_proxy（本轮不修改） ==========
    # 当飞行器接近目标、速度低、姿态稳定且两个支撑接触时给予小奖励
    near_target = (next_dist < 0.5)
    low_speed = (speed < 0.3)
    stable_angle = (abs(next_obs[4]) < 0.2)
    both_contact = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0

    # ========== 动作代价：energy_penalty（小权重，本轮不修改） ==========
    engine_use = 0.0
    if action == 1 or action == 3:
        engine_use = 0.5
    elif action == 2:
        engine_use = 1.0
    energy_penalty = -0.1 * engine_use

    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + energy_penalty

    # ========== 组件字典 ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components
```