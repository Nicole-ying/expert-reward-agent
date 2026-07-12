# 1. Search objective
- target_score: 300.000000
- current_score: 28.667325
- gap_to_target: 271.332675
- target_achievement_ratio: 9.556%

# 2. 上一轮奖励函数代码（该轮得分: 28.667325）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 中提取关键信号（动作结果）
    hull_angle_next = next_obs[0]          # 躯干倾角，越小越稳
    hull_ang_vel_next = next_obs[1]        # 躯干角速度
    horizontal_velocity_next = next_obs[2] # 前向速度，正值向前

    # ---------- 健康门控：当躯干倾斜过大时自动衰减前向奖励 ----------
    danger_angle = 0.8   # 接近摔倒的阈值（~45°）
    max_angle = 1.2      # 完全关闭主奖励的阈值（~69°）
    # 线性衰减门：在 [0, danger_angle] 恒为 1，在 [danger_angle, max_angle] 从 1 线性降到 0
    gate = max(0.0, min(1.0, (max_angle - abs(hull_angle_next)) / (max_angle - danger_angle)))

    # ---------- 主学习信号：被门控的前向速度 ----------
    velocity_gated = gate * horizontal_velocity_next

    # ---------- 稳定性约束：温和地惩罚角速度，减少晃动 ----------
    w_ang_vel = 0.1  # 回退到合理范围，避免抑制必要动态
    stability_penalty = -w_ang_vel * (hull_ang_vel_next ** 2)

    # ---------- 动作效率：惩罚动作力矩平方和，降低能耗 ----------
    w_action = 0.1  # 约 0.1/step，不到主信号的 20%
    action_efficiency = -w_action * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    # ---------- 总奖励与组件字典 ----------
    total_reward = velocity_gated + stability_penalty + action_efficiency
    components = {
        "velocity_gated": velocity_gated,
        "stability_penalty": stability_penalty,
        "action_efficiency": action_efficiency
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 911.20 | 294.82 | ✅ |
| 2 | 提升稳定性惩罚系数 10 倍（0.05→0.5）将引导策略减少无效晃动，步态更平滑，从而提高行走效率，接近目标 3... | 提升稳定性惩罚系数 10 倍（0.05→0.5）将引导策略减少无效晃动，步态更平滑，从而提高行走效率，接近目标 3... | 803.15 | 243.23 | ❌ |
| 3 | 加动作惩罚后，策略被迫学习低功耗、高续航的步态，在不牺牲速度的前提下延长行走距离 → 同时提升主奖励和步长 → 分... | 加动作惩罚后，策略被迫学习低功耗、高续航的步态，在不牺牲速度的前提下延长行走距离 → 同时提升主奖励和步长 → 分... | 461.85 | 28.67 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=28.667325, len=461.850000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-45.527834, 138.847966]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_gated | 181.581473 | 90.5% | 90.5% | 100.0% |
| action_efficiency | -18.890303 | -9.4% | 9.4% | 100.0% |
| stability_penalty | -0.085291 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个 2D 双足行走任务。智能体控制一个拥有两条腿（各含髋、膝两个关节）的身体，在不平坦地形上向前行走。核心目标是 **走得尽可能远、尽可能快**，同时 **最小化能量消耗**。要获得好成绩，必须通过协调四肢关节力矩生成稳定、高效的双足步态；一旦身体倾斜过度、摔倒，回合即告失败。任务**不存在明确的到达点**，过程终止于身体摔倒（失败）或到达地形末端（可视为成功完成路程）。

## 3. 观察空间 observation_space
- type: Box  
- shape: (24,)  
- dtype: float（具体为 float32/64，下同）
- 各分量含义及 reward_usable 标记：

| index | name | meaning | reward_usable |
|-------|------|---------|---------------|
| 0 | hull_angle | 躯干相对于竖直方向的倾角（弧度） | true |
| 1 | hull_angular_velocity | 躯干的角速度 | true |
| 2 | horizontal_velocity | 前方的线速度（正值表示向前） | true |
| 3 | vertical_velocity | 上下方向线速度 | true |
| 4 | hip1_angle | 腿1髋关节角度 | true |
| 5 | hip1_speed | 腿1髋关节角速度 | true |
| 6 | knee1_angle | 腿1膝关节角度 | true |
| 7 | knee1_speed | 腿1膝关节角速度 | true |
| 8 | leg1_contact | 腿1触地标志（1.0 触地，0.0 离地） | true |
| 9 | hip2_angle | 腿2髋关节角度 | true |
|10 | hip2_speed | 腿2髋关节角速度 | true |
|11 | knee2_angle | 腿2膝关节角度 | true |
|12 | knee2_speed | 腿2膝关节角速度 | true |
|13 | leg2_contact | 腿2触地标志（1.0 触地，0.0 离地） | true |
|14 | lidar_0 | LIDAR 距离传感器 0（前方地形距离） | true（但需谨慎使用） |
|15 | lidar_1 | LIDAR 距离传感器 1 | true |
|16 | lidar_2 | LIDAR 距离传感器 2 | true |
|17 | lidar_3 | LIDAR 距离传感器 3 | true |
|18 | lidar_4 | LIDAR 距离传感器 4 | true |
|19 | lidar_5 | LIDAR 距离传感器 5 | true |
|20 | lidar_6 | LIDAR 距离传感器 6 | true |
|21 | lidar_7 | LIDAR 距离传感器 7 | true |
|22 | lidar_8 | LIDAR 距离传感器 8 | true |
|23 | lidar_9 | LIDAR 距离传感器 9 | true |

> 注：所有字段在理论上都可参与奖励计算。LIDAR 读数可用于感知前方地形起伏，但作为连续控制任务，直接奖励地形平整度可能引入噪声；除非明确需要地形自适应，否则奖励函数不一定要使用它们。

## 4. 动作空间 action_space
- type: Box  
- shape: (4,)  
- dtype: float  
- 各维度含义：

| index | name | meaning |
|-------|------|---------|
| 0 | hip_torque_leg1 | 施加到腿1髋关节的力矩，范围 [-1, 1] |
| 1 | knee_torque_leg1 | 施加到腿1膝关节的力矩，范围 [-1, 1] |
| 2 | hip_torque_leg2 | 施加到腿2髋关节的力矩，范围 [-1, 1] |
| 3 | knee_torque_leg2 | 施加到腿2膝关节的力矩，范围 [-1, 1] |

动作是连续的关节力矩，智能体需生成协调的关节动作以形成稳定步态。

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**：`reached_end_of_terrain` — 身体抵达地形终点，说明已走完全程。由于无明确成功标志，该终止仅暗示路径被完整覆盖，必须谨慎使用，不能直接作为奖励。
- **failure-like termination**：`body_fallen_over` — 躯干倾斜过度导致摔倒，这是明确的失败。
- **ambiguous termination**：无其它模式。
- **truncation**：源码中仅返回 `terminated`，无 `truncated` 标记，因此默认所有终止都视为 episode 结束（可能由最大步数截断，但未指明，需假设环境可能包含基于步数的截断）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**  
  （无 `info` 输出，无法直接得知 `reached_end_of_terrain` 的具体值）
- explicit_failure_flag_available: **false**  
  （同样，`body_fallen_over` 没有作为独立标志提供给 reward 函数）
- allowed_info_fields: 无（`info` 为 `{}`）
- forbidden_or_uncertain_info_fields: 任何需要从 `info` 读取的字段均禁止

> 奖励函数**必须**基于观测序列隐式推断失败风险（例如通过 `hull_angle` 绝对值过大），**绝对不能**依赖 `done` 或任何终止标志。

## 7. 可用于奖励函数的信号

- **姿态/平衡**：
  - `hull_angle`（obs[0]）：躯干偏离竖直的程度，越小越平衡。
  - `hull_angular_velocity`（obs[1]）：倾斜变化快慢，可用于惩罚剧烈晃动。
- **推进速度**：
  - `horizontal_velocity`（obs[2]）：正向速度，越大前进越快。
  - `vertical_velocity`（obs[3]）：可辅助检测颠簸或跳跃，但不一定是主要奖励源。
- **关节状态**：
  - 髋/膝角度与角速度（obs[4..7], obs[9..12]）：可用于限制关节极限、惩罚过大动作或鼓励平滑运动。
- **接触状态**：
  - `leg1_contact`, `leg2_contact`（obs[8], [13]）：反映脚是否着地，可用于检测步态交替或防止双腿同时离地。
- **动作/能量**：
  - `action` 四维力矩：直接反映控制能量，平方和或绝对值和可用来惩罚低效动作。
- **地形感知**：
  - LIDAR 传感器（obs[14..23]）：10个距离读数，描述前方地形轮廓，可用于奖励对地形的适应性（如避免过陡坡），但使用需谨慎，会大幅增加奖励设计复杂度。

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
| 1 | stability_penalty + velocity_gated | 294.82 | 294.82 | 0.00 | 911.20 | stability_penalty=-0.000 velocity_gated=0.331 | new_best |
| 2 | stability_penalty + velocity_gated | 243.23 | 294.82 | -51.59 | 803.15 | stability_penalty=-0.001 velocity_gated=0.340 | no_meaningful_improvement |
| 3 | action_efficiency + stability_penalty + velocity_gated | 28.67 | 294.82 | -266.15 | 461.85 | action_efficiency=-0.099 stability_penalty=-0.001 velocity_gated=0.234 | no_meaningful_improvement |
