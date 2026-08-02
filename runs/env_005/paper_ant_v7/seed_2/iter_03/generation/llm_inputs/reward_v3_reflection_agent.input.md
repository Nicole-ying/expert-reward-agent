# 1. Search objective
- target_score: 2000.000000
- current_score: -722.255458
- gap_to_target: 2722.255458
- target_achievement_ratio: -36.113%

# 2. 上一轮奖励函数代码（该轮得分: -722.255458）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 主学习信号：前进速度 × 直立门控 ----
    forward_velocity = next_obs[13]

    # 身体直立程度（1=完全竖直，<=0=翻倒）
    quat_x, quat_y = next_obs[2], next_obs[3]
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_gate = max(0.0, body_up_z)  # 倒立时门控为 0，抹掉前进奖励

    forward_reward = 2.0 * forward_velocity * upright_gate

    # ---- 稳定/健康约束：身体高度安全区间 ----
    body_height = next_obs[0]
    lower_safe = 0.3
    upper_safe = 0.9
    height_penalty = (
        -5.0 * max(0.0, lower_safe - body_height) +
        -5.0 * max(0.0, body_height - upper_safe)
    )

    # ---- 辅助约束：侧向漂移抑制 ----
    lateral_velocity = next_obs[14]
    lateral_penalty = -0.5 * lateral_velocity**2

    # ---- 效率约束：动作能量代价 ----
    action_energy = sum(a**2 for a in action)
    energy_penalty = -0.01 * action_energy

    total_reward = forward_reward + height_penalty + lateral_penalty + energy_penalty

    components = {
        "forward_velocity_reward": forward_reward,
        "height_health_penalty": height_penalty,
        "lateral_drift_penalty": lateral_penalty,
        "action_energy_penalty": energy_penalty
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 218.10 | -102.08 | ✅ |
| 2 | 将直立约束改为软门控后，奖励结构从对抗变为协同——agent 只有保持直立才能收获前进奖励，不再因直立问题而被巨量... | 将直立约束改为软门控后，奖励结构从对抗变为协同——agent 只有保持直立才能收获前进奖励，不再因直立问题而被巨量... | 552.60 | -722.26 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=-722.255458, len=552.600000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[-1851.000051, 4.358282]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_velocity_reward | 357.219844 | 73.0% | 77.1% | 43.9% |
| lateral_drift_penalty | -42.794668 | -8.7% | 8.7% | 52.8% |
| height_health_penalty | -39.911247 | -8.2% | 8.2% | 47.6% |
| action_energy_penalty | -29.244398 | -6.0% | 6.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 4/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个连续控制运动任务。控制一个 3D 四足机器人（8 个扭矩关节）在保证身体高度处于健康区间、保持直立姿态的前提下，以尽可能快的速度稳定前进。核心目标是稳定的前进运动（**前进速度最大化**），维持身体高度和直立姿态是避免提前终止的必要条件，但不是任务本身的核心优化目标。不允许依赖任何官方奖励项（info 被清空，official reward masked）。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: 推断为 float32（来自 continuous locomotion）
- obs[0] (body_z)：身体重心垂直高度。reward_usable: true（可构建高度健康奖励/惩罚）
- obs[1] (quat_w)：身体姿态四元数实部 w。reward_usable: true（用于计算直立程度）
- obs[2] (quat_x)：姿态四元数虚部 x。reward_usable: true
- obs[3] (quat_y)：姿态四元数虚部 y。reward_usable: true
- obs[4] (quat_z)：姿态四元数虚部 z。reward_usable: true
- obs[5] (joint_1_angle)：第一 hip 关节角度。reward_usable: true（可用来约束关节范围、平滑动作）
- obs[6] (joint_2_angle)：第一 ankle 关节角度。reward_usable: true
- obs[7] (joint_3_angle)：第二 hip 关节角度。reward_usable: true
- obs[8] (joint_4_angle)：第二 ankle 关节角度。reward_usable: true
- obs[9] (joint_5_angle)：第三 hip 关节角度。reward_usable: true
- obs[10] (joint_6_angle)：第三 ankle 关节角度。reward_usable: true
- obs[11] (joint_7_angle)：第四 hip 关节角度。reward_usable: true
- obs[12] (joint_8_angle)：第四 ankle 关节角度。reward_usable: true
- obs[13] (body_x_velocity)：身体在世界系 x 方向的前进速度。reward_usable: **true（核心前进信号）**
- obs[14] (body_y_velocity)：身体横向速度（世界 y）。reward_usable: true（可用于惩罚侧向漂移）
- obs[15] (body_z_velocity)：身体垂直速度。reward_usable: true（可用于惩罚剧烈上下颠簸）
- obs[16] (body_roll_velocity)：滚转角速度。reward_usable: true
- obs[17] (body_pitch_velocity)：俯仰角速度。reward_usable: true
- obs[18] (body_yaw_velocity)：偏航角速度。reward_usable: true
- obs[19] (joint_1_velocity)：第一 hip 关节角速度。reward_usable: true（用于平滑或能耗惩罚）
- obs[20] (joint_2_velocity)：第一 ankle 关节角速度。reward_usable: true
- obs[21] (joint_3_velocity)：第二 hip 关节角速度。reward_usable: true
- obs[22] (joint_4_velocity)：第二 ankle 关节角速度。reward_usable: true
- obs[23] (joint_5_velocity)：第三 hip 关节角速度。reward_usable: true
- obs[24] (joint_6_velocity)：第三 ankle 关节角速度。reward_usable: true
- obs[25] (joint_7_velocity)：第四 hip 关节角速度。reward_usable: true
- obs[26] (joint_8_velocity)：第四 ankle 关节角速度。reward_usable: true

## 4. 动作空间 action_space
- type: Box（连续）
- shape: (8,)
- 范围：[-1.0, 1.0] per joint（扭矩归一化值）
- action_dim 0 (hip_1_torque)：第一 hip 关节扭矩
- action_dim 1 (ankle_1_torque)：第一 ankle 关节扭矩
- action_dim 2 (hip_2_torque)：第二 hip 关节扭矩
- action_dim 3 (ankle_2_torque)：第二 ankle 关节扭矩
- action_dim 4 (hip_3_torque)：第三 hip 关节扭矩
- action_dim 5 (ankle_3_torque)：第三 ankle 关节扭矩
- action_dim 6 (hip_4_torque)：第四 hip 关节扭矩
- action_dim 7 (ankle_4_torque)：第四 ankle 关节扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: **无明显成功终止**。episode 如果一直保持安全状态直到被截断（truncation）则可能被视为成功完成一次稳定的前进回合。
- failure-like termination: 身体高度低于 0.2（跌倒）或高于 1.0（过度起跳）；任何状态值变为 NaN 或 inf（数值崩溃）。
- ambiguous termination: 无。
- truncation: 达到环境预设的最大仿真步数（时间限制），此时 episode 直接结束，无特殊终止标志。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false（terminated 信号在 step 外部返回，但 reward 函数接口无法直接获取 terminated 标志）
- allowed_info_fields: **无**（info 字典被清空，接口声明禁止使用任何 info 字段）
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin 等全部官方奖励或定位数据（明确禁止）

## 7. 可用于奖励函数的信号
- position: body_z（高度），关节角度（可通过与目标姿态的偏差设计奖励）
- velocity: body_x_velocity（前进速度，核心），body_y_velocity（侧向），body_z_velocity（垂直速度），各关节角速度
- orientation: body_up_z = 1 - 2*(quat_x² + quat_y²) 量化直立程度（0~1，1 为完全竖直）
- contact: 无直接接触力，本环境版本无接触信息
- action/engine: action 本身（扭矩可构成能量/平滑惩罚），action 变化量（需自行维护上次动作，但奖励函数无状态，故无法直接计算 delta；可以惩罚 action 的绝对大小）
- other: 关节角度偏离正常范围（如设定目标关节位置）可用作风格约束

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
| 1 | action_energy_penalty + forward_velocity_reward + height_health_penalty + lateral_drift_penalty + upright_orientation_penalty | -102.08 | -102.08 | 0.00 | 218.10 | action_energy_penalty=-0.042 forward_velocity_reward=0.745 height_health_penalty=-0.063 lateral_drift_penalty=-0.291 upright_orientation_penalty=-1.511 | new_best |
| 2 | action_energy_penalty + forward_velocity_reward + height_health_penalty + lateral_drift_penalty | -722.26 | -102.08 | -620.18 | 552.60 | action_energy_penalty=-0.049 forward_velocity_reward=0.464 height_health_penalty=-0.068 lateral_drift_penalty=-0.170 | no_meaningful_improvement |
