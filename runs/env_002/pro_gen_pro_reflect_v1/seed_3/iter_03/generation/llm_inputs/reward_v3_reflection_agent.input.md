# 1. Search objective
- target_score: 300.000000
- current_score: 284.141664
- gap_to_target: 15.858336
- target_achievement_ratio: 94.714%

# 2. 上一轮奖励函数代码（该轮得分: 284.141664）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Main learning signal: forward velocity, gated by posture health
    forward_vel = next_obs[2]                 # positive = moving forward
    hull_angle = next_obs[0]                  # radians, 0 = vertical
    health = 1.0 / (1.0 + 10.0 * (hull_angle ** 2))   # posture gate
    forward_progress = 1.0 * forward_vel * health

    # Stability: keep torso upright (secondary penalty, kept as safety net)
    tilt_penalty = -5.0 * (hull_angle ** 2)

    # Stability: suppress rotational jitter
    hull_angvel = next_obs[1]                 # rad/s
    angular_velocity_penalty = -1.0 * (hull_angvel ** 2)

    # Efficiency: light action cost
    avg_action_power = (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2) / 4.0
    action_cost = -0.05 * avg_action_power

    total_reward = forward_progress + tilt_penalty + angular_velocity_penalty + action_cost

    components = {
        'forward_progress': forward_progress,
        'tilt_penalty': tilt_penalty,
        'angular_velocity_penalty': angular_velocity_penalty,
        'action_cost': action_cost
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 1010.20 | 217.05 | ✅ |
| 2 | 将前进奖励与姿态健康直接乘积耦合，迫使 agent 为获得高速奖励而主动维持直立，从而在速度与稳定性之间自动平衡，... | 将前进奖励与姿态健康直接乘积耦合，迫使 agent 为获得高速奖励而主动维持直立，从而在速度与稳定性之间自动平衡，... | 885.10 | 284.14 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=284.141664, len=885.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[24.020388, 311.154853]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 471.509274 | 95.3% | 95.3% | 100.0% |
| action_cost | -15.938158 | -3.2% | 3.2% | 100.0% |
| tilt_penalty | -6.837885 | -1.4% | 1.4% | 100.0% |
| angular_velocity_penalty | -0.333029 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一项 2D 双足移动任务。智能体需要控制两条腿的髋、膝共四个关节的连续力矩，驱动身体在不同地形上尽可能远、尽可能快地向前行走，同时尽量减少能量消耗。机器人必须保持直立，避免摔倒，因为摔倒将直接终止当前回合。次级目标包括：保持平稳步态、不出现过多不必要的跳跃或抖动、使用合理的关节力矩。该任务不能混淆为定点到达或生存平衡：核心驱动信号是前进速度，成功条件是到达地形尽头，而非存活时长。

## 3. 观察空间 observation_space
- type: Box  
- shape: (24,)  
- dtype: float32（连续值）  
- obs[0]: hull_angle – 本体与垂线的夹角（弧度），可用于惩罚倾斜，保持直立。reward_usable: true  
- obs[1]: hull_angular_velocity – 本体角速度，用于抑制旋转抖动。reward_usable: true  
- obs[2]: horizontal_velocity – 水平线速度（正向为前进），是主进度信号。reward_usable: true  
- obs[3]: vertical_velocity – 垂直线速度，可用于限制跳跃/过重落地。reward_usable: true  
- obs[4]: hip1_angle – 腿1髋关节角度。reward_usable: potentially  
- obs[5]: hip1_speed – 腿1髋关节角速度。reward_usable: potentially  
- obs[6]: knee1_angle – 腿1膝关节角度。reward_usable: potentially  
- obs[7]: knee1_speed – 腿1膝关节角速度。reward_usable: potentially  
- obs[8]: leg1_contact – 腿1是否接触地面（1.0/0.0），可用于步态协调。reward_usable: true  
- obs[9]: hip2_angle – 腿2髋关节角度。reward_usable: potentially  
- obs[10]: hip2_speed – 腿2髋关节角速度。reward_usable: potentially  
- obs[11]: knee2_angle – 腿2膝关节角度。reward_usable: potentially  
- obs[12]: knee2_speed – 腿2膝关节角速度。reward_usable: potentially  
- obs[13]: leg2_contact – 腿2是否接触地面。reward_usable: true  
- obs[14]~obs[23]: lidar_0~lidar_9 – 10个前方激光雷达测距值，反映地形轮廓，早期可不依赖。reward_usable: true，但不强制使用。

## 4. 动作空间 action_space
- type: Box  
- shape: (4,)  
- continuous: true  
- bounds: 每个关节力矩限制在 [-1.0, 1.0]  
- action[0]: hip_torque_leg1 – 腿1髋关节力矩  
- action[1]: knee_torque_leg1 – 腿1膝关节力矩  
- action[2]: hip_torque_leg2 – 腿2髋关节力矩  
- action[3]: knee_torque_leg2 – 腿2膝关节力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain – 到达地形尽头，视为成功完成。  
- failure-like termination: body_fallen_over – 机器人摔倒，视为失败。  
- ambiguous termination: 无。  
- truncation: 环境中无内部步数截断（外部训练循环可能限定最大步数）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （step 返回 info 为空字典）  
- explicit_failure_flag_available: false （同上）  
- allowed_info_fields: [] （空字典，无任何允许字段）  
- forbidden_or_uncertain_info_fields: 所有 info 字段均为非法，因为环境中不提供任何信息字段。  
注意：奖励函数无法访问 `terminated` 或 `done`，必须完全基于 `obs`、`next_obs`、`action` 推断状态好坏。

## 7. 可用于奖励函数的信号
- position: 无直接位移坐标，可通过对 horizontal_velocity 在时间上的积分近似，但奖励中通常直接使用速度信号。  
- velocity: next_obs[2]（或 obs[2]）horizontal_velocity 作为前进速度；obs[3]/next_obs[3] vertical_velocity 可用于惩罚跳跃。  
- orientation: next_obs[0] hull_angle 衡量姿态倾斜；next_obs[1] hull_angular_velocity 抑制旋转。  
- contact: next_obs[8] leg1_contact, next_obs[13] leg2_contact，用于步态协调。  
- action/engine: 本次动作 action[0:4] 可用于力矩惩罚（能耗）；也可使用动作变化量（需结合上一步动作，但接口未提供，可放弃）。  
- other: lidar 信号（next_obs[14:24] 或 obs[14:24]），可能用于感知前方地形，不强制使用。

# 6. Formula switching guide
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 7. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | action_cost + angular_velocity_penalty + forward_progress + tilt_penalty | 217.05 | 217.05 | 0.00 | 1010.20 | action_cost=-0.025 angular_velocity_penalty=-0.001 forward_progress=0.160 tilt_penalty=-0.088 | new_best |
| 2 | action_cost + angular_velocity_penalty + forward_progress + tilt_penalty | 284.14 | 284.14 | 0.00 | 885.10 | action_cost=-0.024 angular_velocity_penalty=-0.001 forward_progress=0.169 tilt_penalty=-0.078 | new_best |
