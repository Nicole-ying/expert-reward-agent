# 1. Search objective
- target_score: 300.000000
- current_score: 88.341858
- gap_to_target: 211.658142
- target_achievement_ratio: 29.447%

# 2. 上一轮奖励函数代码（该轮得分: 88.341858）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ====== 主学习信号：前进速度奖励（平方形式，凸化激励） ======
    # 使用 forward_vel * |forward_vel| 保持方向，同时平方增强高速激励
    forward_vel = next_obs[2]
    w_forward = 2.0
    forward_reward = w_forward * forward_vel * abs(forward_vel)

    # ====== 竖直/姿态稳定性惩罚 ======
    hull_angle = obs[0]
    hull_angvel = obs[1]
    vertical_vel = obs[3]
    w_angle = 0.5
    w_angvel = 0.01
    w_vertical = 0.01
    stability_penalty = -(
        w_angle * hull_angle**2 +
        w_angvel * hull_angvel**2 +
        w_vertical * vertical_vel**2
    )

    # ====== 能量效率惩罚 ======
    w_energy = 0.001
    energy_penalty = -w_energy * (
        action[0]**2 + action[1]**2 +
        action[2]**2 + action[3]**2
    )

    total_reward = forward_reward + stability_penalty + energy_penalty
    components = {
        'forward_reward': forward_reward,
        'stability_penalty': stability_penalty,
        'energy_penalty': energy_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 930.35 | 290.58 | ✅ |
| 2 | 凸化后的前进奖励将为高速行为提供更强的梯度激励，使 agent 突破当前中速瓶颈，向更高分数冲刺。 | 凸化后的前进奖励将为高速行为提供更强的梯度激励，使 agent 突破当前中速瓶颈，向更高分数冲刺。 | 489.95 | 88.34 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=88.341858, len=489.950000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-42.783334, 323.139958]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 359.000305 | 98.2% | 98.2% | 100.0% |
| stability_penalty | -5.785282 | -1.6% | 1.6% | 100.0% |
| energy_penalty | -0.610927 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个二维双足身体在崎岖地形上稳定行走，尽可能走得更远、更快（最大化前进距离与速度），同时最小化关节力矩带来的能量消耗。身体必须保持直立，摔倒或到达地形终点会终止回合。核心是“快速前进”而不是“仅保持平衡”；能量消耗是重要的次要指标，但不能牺牲前进速度。

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (24,)
- dtype: float32
- 各维含义列表：

| index | 字段名                      | 物理含义                         | reward_usable |
|-------|-----------------------------|----------------------------------|---------------|
| 0     | hull_angle                  | 躯干相对于竖直方向的倾角         | true          |
| 1     | hull_angular_velocity       | 躯干角速度                       | true          |
| 2     | horizontal_velocity         | 前进/后退线速度                  | true          |
| 3     | vertical_velocity           | 上下线速度                       | true          |
| 4     | hip1_angle                  | 腿1髋关节角度                    | true          |
| 5     | hip1_speed                  | 腿1髋关节角速度                  | true          |
| 6     | knee1_angle                 | 腿1膝关节角度                    | true          |
| 7     | knee1_speed                 | 腿1膝关节角速度                  | true          |
| 8     | leg1_contact                | 腿1触地标志（1.0=触地，0.0=离地） | true          |
| 9     | hip2_angle                  | 腿2髋关节角度                    | true          |
| 10    | hip2_speed                  | 腿2髋关节角速度                  | true          |
| 11    | knee2_angle                 | 腿2膝关节角度                    | true          |
| 12    | knee2_speed                 | 腿2膝关节角速度                  | true          |
| 13    | leg2_contact                | 腿2触地标志（1.0=触地，0.0=离地） | true          |
| 14    | lidar_0                     | 前方第1个激光雷达测距值          | true          |
| 15    | lidar_1                     | 前方第2个激光雷达测距值          | true          |
| 16    | lidar_2                     | 前方第3个激光雷达测距值          | true          |
| 17    | lidar_3                     | 前方第4个激光雷达测距值          | true          |
| 18    | lidar_4                     | 前方第5个激光雷达测距值          | true          |
| 19    | lidar_5                     | 前方第6个激光雷达测距值          | true          |
| 20    | lidar_6                     | 前方第7个激光雷达测距值          | true          |
| 21    | lidar_7                     | 前方第8个激光雷达测距值          | true          |
| 22    | lidar_8                     | 前方第9个激光雷达测距值          | true          |
| 23    | lidar_9                     | 前方第10个激光雷达测距值         | true          |

## 4. 动作空间 action_space
- type: Box (连续)
- shape: (4,)
- 范围: 每个动作 ∈ [-1.0, 1.0]（关节力矩缩放值）
- action_dim 0: hip_torque_leg1 – 腿1的髋关节扭矩
- action_dim 1: knee_torque_leg1 – 腿1的膝关节扭矩
- action_dim 2: hip_torque_leg2 – 腿2的髋关节扭矩
- action_dim 3: knee_torque_leg2 – 腿2的膝关节扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（到达地形尽头，可能视为成功完成前进任务）
- failure-like termination: body_fallen_over（身体摔倒，通常因倾角过大）
- ambiguous termination: 无
- truncation: 未提供，默认无时间截断或未明确

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false (info 为空，无法在奖励函数中获取到达终点标志)
- explicit_failure_flag_available: false (info 为空，无法直接获取摔倒标志)
- allowed_info_fields: 当前 info 字典为空，reward 函数内部不可使用任何 info 字段
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，包括任何隐含的 "success"/"failure"/"termination_reason" 等

## 7. 可用于奖励函数的信号
- **位置/前进**:
  - next_obs[2] (horizontal_velocity)：直接反映前进速度大小与方向，是核心前进信号
  - obs[2] 用于变化量，但通常直接用 next_obs 的速度更直接
- **姿态/平衡**:
  - obs[0] (hull_angle)：偏离竖直的角度，应趋向0
  - obs[1] (hull_angular_velocity)：稳定状态应较小
  - obs[3] (vertical_velocity)：稳定行走时接近0（无跳跃）
- **接触/步态**:
  - obs[8], obs[13] (leg1_contact, leg2_contact)：可促进交替支撑或避免完全腾空
- **关节运动/能耗**:
  - action[0..3]：直接控制力矩，平方和代表能耗
  - obs[5,7,10,12]：关节角速度，可用于惩罚猛烈摆动（另一种能耗表征）
- **地形信息**:
  - obs[14..23] (lidar readings)：反映前方地形高度起伏，可用于期望步态调整，但直接用于奖励需谨慎
- **next_obs 对应量**也可用于构造变化量（如速度变化、倾角变化）

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
| 1 | energy_penalty + forward_reward + stability_penalty | 290.58 | 290.58 | 0.00 | 930.35 | energy_penalty=-0.002 forward_reward=0.290 stability_penalty=-0.011 | new_best |
| 2 | energy_penalty + forward_reward + stability_penalty | 88.34 | 290.58 | -202.23 | 489.95 | energy_penalty=-0.002 forward_reward=0.253 stability_penalty=-0.016 | no_meaningful_improvement |
