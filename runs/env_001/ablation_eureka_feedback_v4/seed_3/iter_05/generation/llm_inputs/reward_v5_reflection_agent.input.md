# Search objective
- target_score: 200.000000
- current_score: -24.928910
- gap_to_target: 224.928910
- target_achievement_ratio: -12.464%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -24.928910）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from next observation
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # A. Goal proximity shaping (unchanged)
    dist = (x**2 + y**2) ** 0.5
    proximity = 2.71828 ** (-dist)

    # B. Continuous landing quality (replaces landing_success)
    # Combines spatial proximity, safe posture, and contact bonus
    speed_sq = vx**2 + vy**2
    angle_sq = angle**2
    spatial_factor = 2.71828 ** (-2.0 * x**2) * 2.71828 ** (-0.5 * abs(y))
    safety_factor = 2.71828 ** (-10.0 * angle_sq) * 2.71828 ** (-5.0 * speed_sq)
    contact_bonus = 1.0 + left_contact + right_contact
    landing_quality = spatial_factor * safety_factor * contact_bonus

    # C. Energy efficiency (unchanged)
    energy_penalty = -0.01 if action != 0 else 0.0

    # D. Terminal velocity penalty (unchanged)
    vel_penalty = 0.0
    if y < 0.05 and abs(x) < 0.1:
        if vy < -0.3:
            vel_penalty = -0.5 * max(0.0, -vy - 0.3)

    total_reward = 1.0 * proximity + 1.0 * landing_quality + energy_penalty + vel_penalty

    components = {
        "proximity": proximity,
        "landing_quality": landing_quality,
        "energy_penalty": energy_penalty,
        "terminal_velocity_penalty": vel_penalty
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback (EUREKA-style)

## Final-policy outcome

score=-24.928910, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-55.543987, 14.434790]

## Reward component values (mean per episode)
- landing_quality: 687.880941
- proximity: 645.999266
- energy_penalty: -10.000000
- terminal_velocity_penalty: 0.000000

# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个 2D 飞行器着陆/停靠任务。飞行器起始于视野上方中央区域，受到随机初速度扰动。主要目标是尽快抵达中央的目标垫（target pad）并稳定停靠（settle），以最小的引擎推力消耗完成。飞行器需要学会靠近目标、降低速度、保持姿态平稳，并实现安全的双足（左右支撑）接触，最终停在目标垫上。避免与目标垫以外的任何部位发生碰撞、飞出视口边界或长时间不稳定晃动。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断，因为 Box 默认 float32）
- obs[0]: x_position（相对于目标垫的水平坐标），含义：飞行器在 x 方向上偏离目标垫中心的距离，可用于奖励接近目标；reward_usable: true
- obs[1]: y_position（相对于目标垫高度的垂直坐标），含义：飞行器高度与目标垫高度之差，可用于奖励下降/靠近垫面；reward_usable: true
- obs[2]: x_velocity（水平线速度），含义：横向移动速度，可用于惩罚过大侧向速度或奖励静止；reward_usable: true
- obs[3]: y_velocity（垂直线速度），含义：竖直方向速度，可用于惩罚硬着陆（大负值）或奖励稳定；reward_usable: true
- obs[4]: body_angle（机体角），含义：飞行器倾斜角度，可用于奖励保持平正姿态；reward_usable: true
- obs[5]: angular_velocity（角速度），含义：机体旋转速率，可用于奖赏姿态稳定；reward_usable: true
- obs[6]: left_support_contact（左支撑腿接触标志，1.0 或 0.0），含义：左腿是否与目标垫良好接触，可用于奖励双足着陆；reward_usable: true
- obs[7]: right_support_contact（右支撑腿接触标志，1.0 或 0.0），含义：右腿是否与目标垫良好接触，可用于奖励双足着陆；reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine（不操作）—— 不做任何推力/姿态调整，依赖当前动量。
- action 1: left_orientation_engine（左姿态引擎）—— 点火一个姿态引擎，产生逆时针/顺时针力矩以调整机体角度。
- action 2: main_engine（主引擎）—— 点火主引擎，产生主要推力（可能方向固定，对 body 坐标系）。
- action 3: right_orientation_engine（右姿态引擎）—— 点火相反的另一个姿态引擎。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled —— 机体不再活跃或已稳定停靠。这可能意味着飞行器已处于静止状态且（通常）两个支撑腿接触目标垫。高风险：如果发生于 crash 后也可能导致不再活跃，因此该终止条件不能单方面被认定为成功。需要结合接触标志、位置、速度等信号综合判断。
- failure-like termination: 
  - crash_or_body_contact —— 身体某部位（非支撑腿）发生碰撞，可能是撞击地面或目标垫以外的区域。
  - horizontal_position_outside_viewport —— 水平方向飞出视口边界。
- ambiguous termination: 无其他。
- truncation: 无显式时间截断（step 源码未显示 max_steps，默认无截断，由 gym wrapper 决定，但环境无）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （empty dict，无可用字段）
- forbidden_or_uncertain_info_fields: 所有未在源中列出的信息字段（none）

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1]) 以及 next_obs 对应值。
- velocity: x_velocity (obs[2]), y_velocity (obs[3])。
- orientation: body_angle (obs[4]), angular_velocity (obs[5])。
- contact: left_support_contact (obs[6]), right_support_contact (obs[7])。
- action/engine: 当前动作可以用于奖励/惩罚引擎使用（鼓励使用 no_engine）。
- other: 差分信号，如位置变化、速度变化、角度变化，均可从 obs 和 next_obs 构建。

# Compact expert route context
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + proximity + soft_landing + terminal_velocity_penalty | -16.95 | -16.95 | 0.00 | 1000.00 | energy_penalty=-0.007 proximity=0.695 soft_landing=0.402 terminal_velocity_penalty=-0.000 | new_best |
| 2 | energy_penalty + proximity + soft_landing + terminal_velocity_penalty | 115.51 | 115.51 | 0.00 | 725.70 | energy_penalty=-0.008 proximity=0.679 soft_landing=0.761 terminal_velocity_penalty=-0.000 | new_best |
| 3 | energy_penalty + landing_success + proximity + terminal_velocity_penalty | -17.09 | 115.51 | -132.61 | 1000.00 | energy_penalty=-0.008 landing_success=0.844 proximity=0.657 terminal_velocity_penalty=-0.000 | no_meaningful_improvement |
| 4 | energy_penalty + landing_quality + proximity + terminal_velocity_penalty | -24.93 | 115.51 | -140.44 | 1000.00 | energy_penalty=-0.008 landing_quality=1.205 proximity=0.649 terminal_velocity_penalty=-0.000 | no_meaningful_improvement |
