# 1. Search objective
- target_score: 2000.000000
- current_score: 2.524524
- gap_to_target: 1997.475476
- target_achievement_ratio: 0.126%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 2. 上一轮奖励函数代码（该轮得分: 2.524524）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]
    vx = next_obs[13]
    vy = next_obs[14]

    # 姿态健康因子
    pose_error = quat_x**2 + quat_y**2 + quat_z**2
    posture_factor = 2.718281828 ** (-10.0 * pose_error)

    # 高度健康门控：从危险区 (0.25m) 到安全区 (0.55m) 线性提升
    height_gate = 0.0
    if body_z >= 0.55:
        height_gate = 1.0
    elif body_z <= 0.25:
        height_gate = 0.0
    else:
        height_gate = (body_z - 0.25) / 0.3

    # 前进奖励（基础量 × 姿态门控 × 高度门控）
    base_forward = max(0.0, vx)
    forward_reward = 1.0 * base_forward * posture_factor * height_gate

    # 侧向速度惩罚
    lateral_penalty = -0.5 * (vy ** 2)

    # 动作幅度惩罚
    action_penalty = -0.1 * sum(a ** 2 for a in action)

    # 低高度惩罚（hinge）：身体高度低于 0.3m 时施加温和惩罚
    height_penalty = -2.0 * max(0.0, 0.3 - body_z)

    total_reward = forward_reward + lateral_penalty + action_penalty + height_penalty

    components = {
        "forward_reward": forward_reward,
        "lateral_penalty": lateral_penalty,
        "action_penalty": action_penalty,
        "height_penalty": height_penalty
    }
    return float(total_reward), components
```

# 3. 上一轮你自己的诊断（risk / hypothesis）
- **hypothesis**: 增加温和的高度惩罚能警告 agent 远离 0.2 终止边界，从而延长生存、使前向奖励持续生效并改善表现。
- **risk**: 惩罚系数若过大可能抑制低高度区域的探索，但当前尺度（~0.1‑0.2/步）仅起引导作用，副作用低。
(第 1 步 Q3 必须回检：本轮数据是否验证了 risk？)

# 4. 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=2.524524, len=13.750000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-28.387182, 17.529615]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 12.016889 | 59.3% | 59.3% | 70.5% |
| action_penalty | -4.989419 | -24.6% | 24.6% | 100.0% |
| lateral_penalty | -3.253387 | -16.1% | 16.1% | 100.0% |
| height_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
任务主目标是控制一个3D四足机器人稳定地向前行走或奔跑（持续前进运动），同时保持身体直立并确保身体高度始终处于健康范围（0.2m–1.0m）。  
次要目标可能是降低能耗（动作幅度控制在合理范围内）以及维持平稳的运动姿态（避免剧烈翻滚或抖动）。  
不应将单纯保持静止直立或仅避免摔倒作为核心目标——必须持续产生正向的前进速度。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: float（通常float64或float32，不指定具体）
- 所有维度均可用于奖励设计（reward_usable: true），除非特别说明。

逐维含义：

| 索引 | 名称                    | 含义                                | reward_usable | 备注 |
|------|-------------------------|-------------------------------------|---------------|------|
| 0    | body_z                  | 机体垂直高度                        | true          | 用于生存/高度约束 |
| 1    | quat_w                  | 机体姿态四元数实部 w                | true          | 用于计算直立程度 |
| 2    | quat_x                  | 四元数虚部 x                        | true          | 用于计算直立程度 |
| 3    | quat_y                  | 四元数虚部 y                        | true          | 用于计算直立程度 |
| 4    | quat_z                  | 四元数虚部 z                        | true          | 用于计算直立程度 |
| 5    | joint_1_angle           | 髋关节1角度                         | true          | 可不直接用于奖励 |
| 6    | joint_2_angle           | 踝关节1角度                         | true          | 可不直接用于奖励 |
| 7    | joint_3_angle           | 髋关节2角度                         | true          | 可不直接用于奖励 |
| 8    | joint_4_angle           | 踝关节2角度                         | true          | 可不直接用于奖励 |
| 9    | joint_5_angle           | 髋关节3角度                         | true          | 可不直接用于奖励 |
| 10   | joint_6_angle           | 踝关节3角度                         | true          | 可不直接用于奖励 |
| 11   | joint_7_angle           | 髋关节4角度                         | true          | 可不直接用于奖励 |
| 12   | joint_8_angle           | 踝关节4角度                         | true          | 可不直接用于奖励 |
| 13   | body_x_velocity         | 世界坐标系下机体前向速度 (x)        | true          | 主前进信号 |
| 14   | body_y_velocity         | 世界坐标系下机体侧向速度 (y)        | true          | 方向偏离惩罚 |
| 15   | body_z_velocity         | 机体垂直速度                        | true          | 稳定性惩罚 |
| 16   | body_roll_velocity      | 机体滚转角速度                      | true          | 稳定性惩罚 |
| 17   | body_pitch_velocity     | 机体俯仰角速度                      | true          | 稳定性惩罚 |
| 18   | body_yaw_velocity       | 机体偏航角速度                      | true          | 方向控制/偏航惩罚 |
| 19   | joint_1_velocity        | 髋关节1角速度                       | true          | 运动平滑/能量惩罚 |
| 20   | joint_2_velocity        | 踝关节1角速度                       | true          | 同上 |
| 21   | joint_3_velocity        | 髋关节2角速度                       | true          | 同上 |
| 22   | joint_4_velocity        | 踝关节2角速度                       | true          | 同上 |
| 23   | joint_5_velocity        | 髋关节3角速度                       | true          | 同上 |
| 24   | joint_6_velocity        | 踝关节3角速度                       | true          | 同上 |
| 25   | joint_7_velocity        | 髋关节4角速度                       | true          | 同上 |
| 26   | joint_8_velocity        | 踝关节4角速度                       | true          | 同上 |

## 4. 动作空间 action_space
- type: Box
- shape: (8,)
- 连续值，每维范围 [-1.0, 1.0]，代表关节扭矩。

| 维度索引 | 名称             | 含义                   | 备注 |
|----------|------------------|------------------------|------|
| 0        | hip_1_torque     | 第1髋关节扭矩          |      |
| 1        | ankle_1_torque   | 第1踝关节扭矩          |      |
| 2        | hip_2_torque     | 第2髋关节扭矩          |      |
| 3        | ankle_2_torque   | 第2踝关节扭矩          |      |
| 4        | hip_3_torque     | 第3髋关节扭矩          |      |
| 5        | ankle_3_torque   | 第3踝关节扭矩          |      |
| 6        | hip_4_torque     | 第4髋关节扭矩          |      |
| 7        | ankle_4_torque   | 第4踝关节扭矩          |      |

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 不存在明确的成功终止，任务无到达目标、无指定终点。
- failure-like termination:
  1. 身体高度超出健康范围 (0.2, 1.0)，包括摔倒（过低）或异常跃起（过高）。
  2. 任何状态量变为 NaN 或 inf。（物体飞出、数值不稳定）
- ambiguous termination: 无。
- truncation: 达到时间限制（time_limit_reached）截断，非成功非失败。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
  （环境仅返回 terminated boolean，但该变量不传入奖励函数；info 字典为空，无法从中读取成功或失败标志。）
- allowed_info_fields: [] （info 为空，无可用字段）
- forbidden_or_uncertain_info_fields:
  - official reward terms (reward_forward, reward_ctrl, reward_contact, reward_survive) —— 已明确屏蔽
  - x_position, y_position, distance_from_origin —— 不可用
  - 任何接触信息 —— 不可用

# 6. Compact expert route context
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
| 1 | forward_reward + height_penalty + lateral_penalty + posture_penalty | -2.50 | -2.50 | 0.00 | 15.70 | forward_reward=0.632 height_penalty=-0.026 lateral_penalty=-0.340 posture_penalty=-1.948 | new_best |
| 2 | forward_reward + height_penalty + lateral_penalty | -14.41 | -2.50 | -11.90 | 938.85 | forward_reward=0.552 height_penalty=-0.001 lateral_penalty=-0.093 | no_meaningful_improvement |
| 3 | action_penalty + forward_reward + height_penalty + lateral_penalty | 18.41 | 18.41 | 0.00 | 20.85 | action_penalty=-0.278 forward_reward=0.336 height_penalty=-0.012 lateral_penalty=-0.157 | new_best |
| 4 | action_penalty + forward_reward + lateral_penalty | 8.92 | 18.41 | -9.50 | 17.45 | action_penalty=-0.311 forward_reward=0.333 lateral_penalty=-0.181 | unsolved_improving_continue_from_best |
| 5 | action_penalty + forward_reward + height_penalty + lateral_penalty | 2.52 | 18.41 | -15.89 | 13.75 | action_penalty=-0.323 forward_reward=0.298 height_penalty=-0.032 lateral_penalty=-0.218 | no_meaningful_improvement |
