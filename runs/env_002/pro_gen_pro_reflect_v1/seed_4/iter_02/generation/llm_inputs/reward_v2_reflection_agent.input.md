# 1. Search objective
- target_score: 300.000000
- current_score: 212.370151
- gap_to_target: 87.629849
- target_achievement_ratio: 70.790%

# 2. 上一轮奖励函数代码（该轮得分: 212.370151）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------- 核心观测提取 --------------------
    hull_angle     = float(obs[0])   # 身体倾角 (rad)
    hull_ang_vel   = float(obs[1])   # 身体角速度 (rad/s)
    horiz_vel      = float(obs[2])   # 前向速度 (m/s)
    vert_vel       = float(obs[3])   # 竖直速度 (m/s)

    # -------------------- 主进展奖励 (核心驱动力) --------------------
    # 只奖励正的前进速度，负值不惩罚也不奖励
    forward_reward = 1.0 * max(0.0, horiz_vel)

    # -------------------- 姿态与稳定性约束 --------------------
    # 倾角 hinge 惩罚：只在 |angle| > 0.5 rad 时生效，避免安全范围内全时惩罚
    angle_threshold = 0.5
    angle_penalty = -0.5 * max(0.0, abs(hull_angle) - angle_threshold)

    # 角速度二次惩罚：始终有轻量梯度，抑制剧烈摇摆
    angvel_penalty = -0.001 * (hull_ang_vel ** 2)

    # -------------------- 竖直速度抑制（防跳跃） --------------------
    # 步行中正常竖直速度波动 ≤0.5 m/s，超过部分轻罚，避免 agent 通过跳跃刷分
    vert_threshold = 0.5
    vert_penalty = -0.1 * max(0.0, abs(vert_vel) - vert_threshold)

    # -------------------- 总奖励 --------------------
    total_reward = forward_reward + angle_penalty + angvel_penalty + vert_penalty

    components = {
        "forward_reward": forward_reward,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "vert_penalty": vert_penalty,
    }

    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=212.370151, len=632.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-30.337260, 318.220843]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 404.485881 | 99.8% | 99.8% | 99.5% |
| angle_penalty | -0.841930 | -0.2% | 0.2% | 1.3% |
| angvel_penalty | -0.000814 | -0.0% | 0.0% | 99.8% |
| vert_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
环境要求一个双腿身体在不平坦地形上尽可能快、尽可能远地向前行走，同时最小化能量消耗。主体必须协调两侧髋、膝关节力矩，产生稳定的双足步态。**摔倒将导致回合提前终止**，因此保持身体平衡是隐含的刚性前提。次要是节省能量、获得平滑的关节动作，但它们绝不能以牺牲前进和平衡为代价。不应混淆“生存”与“到达终点”：虽然走过整个地形也算终止，但核心奖励必须源于持续的向前位移，而不是一次性的终点信号。

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32（假定）
- obs[0]: hull_angle — 身体相对竖直方向的倾斜角度，reward_usable: true（防摔倒）
- obs[1]: hull_angular_velocity — 身体角速度，reward_usable: true（姿态稳定性）
- obs[2]: horizontal_velocity — 前进/后退线速度，reward_usable: true（**主前进信号**）
- obs[3]: vertical_velocity — 上下线速度，reward_usable: true（可辅助检测跳跃或不稳定）
- obs[4]: hip1_angle — 腿1髋关节角度，reward_usable: true（姿态约束）
- obs[5]: hip1_speed — 腿1髋关节角速度，reward_usable: true（关节平滑性）
- obs[6]: knee1_angle — 腿1膝关节角度，reward_usable: true（姿态约束）
- obs[7]: knee1_speed — 腿1膝关节角速度，reward_usable: true（关节平滑性）
- obs[8]: leg1_contact — 腿1着地标志，1.0=接触，0.0=无接触，reward_usable: true（步态节律/接触）
- obs[9]: hip2_angle — 腿2髋关节角度，reward_usable: true（对称/姿态）
- obs[10]: hip2_speed — 腿2髋关节角速度，reward_usable: true（平滑）
- obs[11]: knee2_angle — 腿2膝关节角度，reward_usable: true（姿态）
- obs[12]: knee2_speed — 腿2膝关节角速度，reward_usable: true（平滑）
- obs[13]: leg2_contact — 腿2着地标志，reward_usable: true（步态节律/接触）
- obs[14..23]: lidar_0..9 — 10 路激光测距值（前方地形高度轮廓），reward_usable: 不确定，通常不直接用于简单前进奖励，可预留为高级地形感知，但基础奖励设计中不建议使用，防止引入不必要复杂度。

所有观测均可用，但是否用于奖励需依据职责。

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- 范围: 每维 [-1.0, 1.0]
- action_dim 0: hip_torque_leg1 — 腿1髋关节力矩
- action_dim 1: knee_torque_leg1 — 腿1膝关节力矩
- action_dim 2: hip_torque_leg2 — 腿2髋关节力矩
- action_dim 3: knee_torque_leg2 — 腿2膝关节力矩

说明：动作是连续力矩控制，无离散选项。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（走完全程）
- failure-like termination: body_fallen_over（身体摔倒）
- ambiguous termination: 无
- truncation: 未在描述中提及，假定有最大步数限制（通用环境），但不作为奖励信号依据。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 字典为空，无法直接获得 reached_end_of_terrain 标志）
- explicit_failure_flag_available: false（同样无法直接获得摔倒标志）
- allowed_info_fields: 无（当前 step 返回 info={}）
- forbidden_or_uncertain_info_fields: 所有的 info 字段均不存在，因此不能依赖任何外部成功/失败标记。奖励函数只能基于观测和动作构建生存和前进信号。

## 7. 可用于奖励函数的信号
- position: 无直接位置观测，但可通过积分水平速度推算（但由于基础，不宜复杂，建议仅用瞬时速度）
- velocity: obs[2] horizontal_velocity（主信号），obs[3] vertical_velocity（辅助），obs[1] hull_angular_velocity（姿态稳定）
- orientation: obs[0] hull_angle（倾斜惩罚）
- contact: obs[8] leg1_contact, obs[13] leg2_contact（步态节律/双脚支撑）
- action/engine: action 四个维度的幅值（能耗惩罚），action 的前后变化量（平滑惩罚）
- other: lidar 测距阵列（obs[14:23]），不建议用于基础奖励；关节角度/角速度可用于姿态正则化。

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
| 1 | angle_penalty + angvel_penalty + forward_reward + vert_penalty | 212.37 | 212.37 | 0.00 | 632.50 | angle_penalty=-0.004 angvel_penalty=-0.000 forward_reward=0.358 vert_penalty=0.000 | new_best |
