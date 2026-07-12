# 1. Search objective
- target_score: 2000.000000
- current_score: 370.674682
- gap_to_target: 1629.325318
- target_achievement_ratio: 18.534%

# 2. 上一轮奖励函数代码（该轮得分: 370.674682）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    body_z = obs[0]
    quat_x = obs[2]
    quat_y = obs[3]
    body_x_velocity = obs[13]
    body_y_velocity = obs[14]
    torque_penalty = sum([a**2 for a in action])

    # ---------- 身体直立投影 ----------
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- 高度安全门控 ----------
    low_factor = max(0.0, (body_z - 0.2) / 0.1)
    high_factor = max(0.0, (1.0 - body_z) / 0.1)
    height_factor = min(low_factor, high_factor)
    height_reward = 0.1 * height_factor

    # ---------- 高度边界前兆惩罚（保留，虽然已接近失效但仍作为安全网） ----------
    low_penalty = -0.3 * max(0.0, (0.25 - body_z) / 0.05)
    high_penalty = -0.3 * max(0.0, (body_z - 0.95) / 0.05)
    height_boundary_penalty = low_penalty + high_penalty

    # ---------- 主学习信号：凸化前进速度 ----------
    forward_reward = 1.0 * body_x_velocity + 0.5 * max(0.0, body_x_velocity)**2

    # ---------- 横向速度惩罚（新组件，替换僵尸 upright_penalty） ----------
    lateral_velocity_penalty = -0.2 * abs(body_y_velocity)

    # ---------- 力矩效率约束 ----------
    action_cost = -0.05 * torque_penalty

    # ---------- 总奖励 ----------
    total_reward = (forward_reward + height_reward + height_boundary_penalty
                    + lateral_velocity_penalty + action_cost)

    components = {
        "forward_reward": forward_reward,
        "height_reward": height_reward,
        "height_boundary_penalty": height_boundary_penalty,
        "lateral_velocity_penalty": lateral_velocity_penalty,
        "action_cost": action_cost
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 80.60 | -12.04 | ✅ |
| 2 | 乘性高度门使速度奖励与生存强绑定，agent 会主动保持安全高度以获取前进回报，生存时间延长，累积 speed 得... | 乘性高度门使速度奖励与生存强绑定，agent 会主动保持安全高度以获取前进回报，生存时间延长，累积 speed 得... | 731.50 | -73.71 | ❌ |
| 3 | 将全时二次姿态惩罚替换为 hinge（只在大倾斜时激活），agent 在正常步态中不再受罚，可以自由探索力矩空间学... | 将全时二次姿态惩罚替换为 hinge（只在大倾斜时激活），agent 在正常步态中不再受罚，可以自由探索力矩空间学... | 392.30 | -54.87 | ❌ |
| 4 | 将 upright_penalty 系数缩至 0.3 后，步均净回报转正，agent 在生存压力和高回报激励下会主... | 将 upright_penalty 系数缩至 0.3 后，步均净回报转正，agent 在生存压力和高回报激励下会主... | 421.00 | 260.75 | ✅ |
| 5 | 凸化前进奖励使高速步态获得超线性回报，在不破坏现有生存策略的前提下，逐步推高 agent 的前进速度，从而提升总得分。 | 凸化前进奖励使高速步态获得超线性回报，在不破坏现有生存策略的前提下，逐步推高 agent 的前进速度，从而提升总得分。 | 437.70 | 407.46 | ✅ |
| 6 | 加入轻量 hinge 惩罚在高度接近 0.2 或 1.0 时提供“远离边界”的梯度，能在不完全牺牲速度的前提下降低... | 加入轻量 hinge 惩罚在高度接近 0.2 或 1.0 时提供“远离边界”的梯度，能在不完全牺牲速度的前提下降低... | 621.95 | 662.91 | ✅ |
| 7 | 每步都给出姿态偏差的微小惩罚，能使策略主动维持高度直立，减少倾斜累积导致的终止，从而提高存活率同时保持速度。 | 每步都给出姿态偏差的微小惩罚，能使策略主动维持高度直立，减少倾斜累积导致的终止，从而提高存活率同时保持速度。 | 404.20 | 27.15 | ❌ |
| 8 | 去掉无处不在的姿态压力后，agent 将重新获得速度提升空间，前进奖励会驱动生存与高效步态，使总得分恢复到第6轮水... | 去掉无处不在的姿态压力后，agent 将重新获得速度提升空间，前进奖励会驱动生存与高效步态，使总得分恢复到第6轮水... | 702.90 | 1414.47 | ✅ |
| 9 | 对 body_y_velocity 施加轻度惩罚后，agent 将调整步态减少侧向摆动，用相同力矩获得更高 bod... | 对 body_y_velocity 施加轻度惩罚后，agent 将调整步态减少侧向摆动，用相同力矩获得更高 bod... | 792.10 | 370.67 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=370.674682, len=792.100000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[-264.678559, 606.290917]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 1930.053090 | 78.3% | 79.7% | 100.0% |
| height_reward | 265.243034 | 10.8% | 10.8% | 100.0% |
| action_cost | -142.729251 | -5.8% | 5.8% | 100.0% |
| lateral_velocity_penalty | -92.072981 | -3.7% | 3.7% | 100.0% |
| height_boundary_penalty | -0.090425 | -0.0% | 0.0% | 0.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个连续控制运动任务。一个3D四足机器人拥有四条腿和八个力矩控制关节，必须向前行走或奔跑，同时保持身体直立。机器人的身体高度必须保持在健康区间内（最低高度至最高高度之间），一旦高度低于最低值（趴下）或高于最高值（弹飞）则回合立刻终止。智能体的核心目标是产生持续、稳定的向前运动，而不是仅仅维持站立不倒下。次要目标包括减小非必要的关节力矩以及维持机身姿态平稳。

## 3. 观察空间 observation_space
- type: Box
- shape: [27]
- dtype: float
- 各维度含义：
  - obs[0]: body_z，身体垂直高度，reward_usable: true
  - obs[1]: quat_w，身体朝向四元数 w 分量，reward_usable: true (用于计算直立程度)
  - obs[2]: quat_x，身体朝向四元数 x 分量，reward_usable: true
  - obs[3]: quat_y，身体朝向四元数 y 分量，reward_usable: true
  - obs[4]: quat_z，身体朝向四元数 z 分量，reward_usable: true
  - obs[5]: joint_1_angle，第一个髋关节角度，reward_usable: false (一般不用)
  - obs[6]: joint_2_angle，第一个踝关节角度，reward_usable: false
  - obs[7]: joint_3_angle，第二个髋关节角度，reward_usable: false
  - obs[8]: joint_4_angle，第二个踝关节角度，reward_usable: false
  - obs[9]: joint_5_angle，第三个髋关节角度，reward_usable: false
  - obs[10]: joint_6_angle，第三个踝关节角度，reward_usable: false
  - obs[11]: joint_7_angle，第四个髋关节角度，reward_usable: false
  - obs[12]: joint_8_angle，第四个踝关节角度，reward_usable: false
  - obs[13]: body_x_velocity，身体世界x方向速度（前进速度），reward_usable: true (主要目标信号)
  - obs[14]: body_y_velocity，身体世界y方向速度（横向速度），reward_usable: true (可用，但非主要)
  - obs[15]: body_z_velocity，身体垂直速度，reward_usable: true (可用于平稳性)
  - obs[16]: body_roll_velocity，滚转角速度，reward_usable: true (姿态稳定性)
  - obs[17]: body_pitch_velocity，俯仰角速度，reward_usable: true
  - obs[18]: body_yaw_velocity，偏航角速度，reward_usable: true (非必须，但可约束)
  - obs[19]: joint_1_velocity，第一个髋关节角速度，reward_usable: false (很少用)
  - obs[20]: joint_2_velocity，第一个踝关节角速度，reward_usable: false
  - obs[21]: joint_3_velocity，第二个髋关节角速度，reward_usable: false
  - obs[22]: joint_4_velocity，第二个踝关节角速度，reward_usable: false
  - obs[23]: joint_5_velocity，第三个髋关节角速度，reward_usable: false
  - obs[24]: joint_6_velocity，第三个踝关节角速度，reward_usable: false
  - obs[25]: joint_7_velocity，第四个髋关节角速度，reward_usable: false
  - obs[26]: joint_8_velocity，第四个踝关节角速度，reward_usable: false

注意：前进方向对应obs[13]，且info完全不可访问，所有必须的奖励信号只能来自obs和action。

## 4. 动作空间 action_space
- type: Box
- shape: [8]
- dtype: float
- 连续动作空间，每个分量范围[-1.0, 1.0]，代表关节力矩。
  - action_dim 0: hip_1_torque，第一个髋关节力矩
  - action_dim 1: ankle_1_torque，第一个踝关节力矩
  - action_dim 2: hip_2_torque，第二个髋关节力矩
  - action_dim 3: ankle_2_torque，第二个踝关节力矩
  - action_dim 4: hip_3_torque，第三个髋关节力矩
  - action_dim 5: ankle_3_torque，第三个踝关节力矩
  - action_dim 6: hip_4_torque，第四个髋关节力矩
  - action_dim 7: ankle_4_torque，第四个踝关节力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功终止，只有达到时间上限truncated才算“存活满时间”。
- failure-like termination:
  - body_height_outside_healthy_range：高度低于0.2或高于1.0（跌倒/弹飞），视为失败。
  - state_value_outside_finite_range：任何状态值变为NaN或inf，视为失败。
- ambiguous termination: 无
- truncation: time_limit_reached，表示达到最大步数，并非失败，应该视为中性或轻微正向。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false (只能通过terminated推断，但官方不允许直接使用terminated标记作为奖励信号，除非我们从状态判断)
- allowed_info_fields: []  (info为空)
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin 等全部被禁止

## 7. 可用于奖励函数的信号
- position: body_z (obs[0]), quaternion (obs[1:5]) 可计算直立投影 (body_up_z = 1 - 2*(quat_x²+quat_y²))
- velocity: body_x_velocity (obs[13]), body_y_velocity (obs[14]), body_z_velocity (obs[15]), body_roll_velocity (obs[16]), body_pitch_velocity (obs[17]), body_yaw_velocity (obs[18])
- orientation: quat可直接用于惩罚翻滚/俯仰过大
- contact: 无
- action/engine: action 力矩本身可用于平滑性惩罚
- other: 身体高度是否在健康区间内 (可根据obs[0]判定濒死区域)

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
| 1 | action_cost + forward_reward + height_penalty + upright_penalty | -12.04 | -12.04 | 0.00 | 80.60 | action_cost=-0.043 forward_reward=0.461 height_penalty=-0.021 upright_penalty=-2.311 | new_best |
| 2 | action_cost + forward_reward + upright_penalty | -73.71 | -12.04 | -61.67 | 731.50 | action_cost=-0.042 forward_reward=0.888 upright_penalty=-3.588 | no_meaningful_improvement |
| 3 | action_cost + forward_reward + height_reward + upright_penalty | -54.87 | -12.04 | -42.83 | 392.30 | action_cost=-0.199 forward_reward=0.271 height_reward=0.187 upright_penalty=-0.824 | no_meaningful_improvement |
| 4 | action_cost + forward_reward + height_reward + upright_penalty | 260.75 | 260.75 | 0.00 | 421.00 | action_cost=-0.191 forward_reward=0.643 height_reward=0.244 upright_penalty=-0.145 | new_best |
| 5 | action_cost + forward_reward + height_reward + upright_penalty | 407.46 | 407.46 | 0.00 | 437.70 | action_cost=-0.199 forward_reward=1.829 height_reward=0.243 upright_penalty=-0.132 | new_best |
| 6 | action_cost + forward_reward + height_boundary_penalty + height_reward + upright_penalty | 662.91 | 662.91 | 0.00 | 621.95 | action_cost=-0.217 forward_reward=1.538 height_boundary_penalty=-0.001 height_reward=0.208 upright_penalty=-0.192 | new_best |
| 7 | action_cost + forward_reward + height_boundary_penalty + height_reward + upright_penalty | 27.15 | 662.91 | -635.76 | 404.20 | action_cost=-0.200 forward_reward=1.102 height_boundary_penalty=-0.001 height_reward=0.225 upright_penalty=-0.691 | no_meaningful_improvement |
| 8 | action_cost + forward_reward + height_boundary_penalty + height_reward + upright_penalty | 1414.47 | 1414.47 | 0.00 | 702.90 | action_cost=-0.205 forward_reward=2.172 height_boundary_penalty=-0.001 height_reward=0.221 upright_penalty=-0.288 | new_best |
| 9 | action_cost + forward_reward + height_boundary_penalty + height_reward + lateral_velocity_penalty | 370.67 | 1414.47 | -1043.79 | 792.10 | action_cost=-0.194 forward_reward=0.579 height_boundary_penalty=-0.001 height_reward=0.173 lateral_velocity_penalty=-0.098 | no_meaningful_improvement |
