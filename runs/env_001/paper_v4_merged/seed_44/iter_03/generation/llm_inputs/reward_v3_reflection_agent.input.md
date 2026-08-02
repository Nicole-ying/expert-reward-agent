# 1. Search objective
- target_score: 200.000000
- current_score: 144.304309
- gap_to_target: 55.695691
- target_achievement_ratio: 72.152%

# 2. 上一轮奖励函数代码（该轮得分: 144.304309）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前位置与目标垫距离
    x_curr = obs[0]
    y_curr = obs[1]
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5

    x_next = next_obs[0]
    y_next = next_obs[1]
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    # 主进展信号：到目标的欧氏距离递减
    progress = dist_curr - dist_next

    # 下一步速度（用于原有速度约束）
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]

    # 速度硬约束（保留但可能保持僵尸）
    x_speed_viol = max(0.0, abs(x_vel_next) - 0.8)
    y_speed_viol = max(0.0, -y_vel_next - 0.8)
    speed_penalty = x_speed_viol + y_speed_viol

    # 姿态与角速度稳定约束
    body_angle_next = next_obs[4]
    angular_vel_next = next_obs[5]
    angle_penalty = body_angle_next ** 2 + angular_vel_next ** 2

    # 替换 contact_bonus：连续软着陆引导
    # 距离因子：越近奖励越大（指数衰减，半衰距离0.5）
    dist_factor = 2.718281828 ** (-dist_next / 0.5)
    # 速度因子：水平与垂直线速度绝对值和越小奖励越大（线性衰减，1.0处归零）
    speed_factor = max(0.0, 1.0 - (abs(x_vel_next) + abs(y_vel_next)) / 1.0)
    landing_reward = dist_factor * speed_factor

    # 加权组合
    total = (
        10.0 * progress
        - 1.0 * speed_penalty
        - 0.5 * angle_penalty
        + 0.01 * landing_reward
    )

    components = {
        "progress": 10.0 * progress,
        "speed_penalty": -1.0 * speed_penalty,
        "angle_penalty": -0.5 * angle_penalty,
        "landing_reward": 0.01 * landing_reward
    }

    return float(total), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 1000.00 | -18.89 | ✅ |
| 2 | 骨架变化: angle_penalty + landing_reward + progress + speed_ | — | 1000.00 | 144.30 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=144.304309, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[101.342861, 171.587967]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 13.960004 | 56.7% | 60.3% | 100.0% |
| landing_reward | 8.034063 | 32.7% | 32.7% | 98.5% |
| angle_penalty | -1.697453 | -6.9% | 6.9% | 100.0% |
| speed_penalty | -0.042040 | -0.2% | 0.2% | 0.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Score=144.3, all episodes truncated (len=1000). Progress (56.7% signed share) and landing_reward (32.7%) dominate reward; angle penalty negative (-6.9%); speed penalty dead (0.2% share, 0.1% active).

**Component Anomalies**: speed_penalty nearly dead (active 0.1%, negligible share). landing_reward active 94.3% but per-step mean tiny (0.0067). No component self-cancelling or >70% share.

**Training Dynamics**: No temporal snapshots provided; cannot assess trend.

**Signal Quality**: Dead gate: speed_penalty. angle_penalty always active (100% nonzero) so penalty never zero. No terminations, suggesting missing attractor for landing behavior.

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主体是一个 2D 飞行器/着陆器，从顶部中央附近随机施加初始力开始，必须尽可能快地运动到屏幕中央的目标垫上，并稳定、安全地停泊（软着陆）；同时要尽量减少发动机使用。智能体的核心挑战在于：接近目标、减速、保持姿态水平、双腿同时轻柔接触垫面，并避免坠毁、出界或长时间悬停。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- obs[0]: x_position – 相对目标垫的水平坐标，可奖励用
- obs[1]: y_position – 相对目标垫高度的垂直坐标，可奖励用
- obs[2]: x_velocity – 水平线速度，可奖励用
- obs[3]: y_velocity – 垂直线速度，可奖励用
- obs[4]: body_angle – 机体朝向角，可奖励用
- obs[5]: angular_velocity – 角速度，可奖励用
- obs[6]: left_support_contact – 左支撑腿是否接触目标垫（1/0），可奖励用
- obs[7]: right_support_contact – 右支撑腿是否接触目标垫（1/0），可奖励用

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine – 无推力
- action 1: left_orientation_engine – 点燃左方向发动机（产生姿态/侧向控制）
- action 2: main_engine – 点燃主发动机（提供向上推力）
- action 3: right_orientation_engine – 点燃右方向发动机

## 5. step 与终止条件分析
### 5.1 终止模式
- success‑like termination: 机体已稳定/不活跃（body_not_awake_or_settled），可能对应成功软着陆（双腿接触垫面、速度与角度极小）
- failure‑like termination: 坠毁/机体与地面或障碍物接触（crash_or_body_contact）；水平位置超出视口（horizontal_position_outside_viewport）
- ambiguous termination: 所有终止条件在源码中未区分成功/失败，需从观测信号间接推断
- truncation: 无显式说明

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info = {}）
- forbidden_or_uncertain_info_fields: 所有未在观察空间中

# 7. Formula switching guide
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2`，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |
| 缺少灾难性失败信号 | 终止率高且失败回合 reward 非负 | terminal_event | 从观测推断失败状态，加入硬覆盖惩罚 |
| 缺少任务完成信号 | agent 持续前进但 episode 在无摔倒情况下终止 | terminal_event 或 improvement_delta | 用位置 delta 做正向奖励，或在确认可达终点时加入软完成 bonus |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angle_penalty + contact_bonus + progress + speed_penalty | -18.89 | -18.89 | 0.00 | 1000.00 | angle_penalty=-0.027 contact_bonus=1.048 progress=0.029 speed_penalty=-0.009 | new_best |
| 2 | angle_penalty + landing_reward + progress + speed_penalty | 144.30 | 144.30 | 0.00 | 1000.00 | angle_penalty=-0.015 landing_reward=0.007 progress=0.025 speed_penalty=-0.008 | new_best |
