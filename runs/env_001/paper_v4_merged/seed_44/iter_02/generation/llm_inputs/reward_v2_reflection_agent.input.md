# 1. Search objective
- target_score: 200.000000
- current_score: -18.891319
- gap_to_target: 218.891319
- target_achievement_ratio: -9.446%

# 2. 上一轮奖励函数代码（该轮得分: -18.891319）
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

    # 下一步速度（用于约束）
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]

    # 速度硬约束：过快水平移动或过快下降
    x_speed_viol = max(0.0, abs(x_vel_next) - 0.8)
    # 假定 y_vel 向上为正，下降速度为 -y_vel，限制下降速度 ≤ 0.8
    y_speed_viol = max(0.0, -y_vel_next - 0.8)
    speed_penalty = x_speed_viol + y_speed_viol

    # 姿态与角速度稳定约束
    body_angle_next = next_obs[4]
    angular_vel_next = next_obs[5]
    angle_penalty = body_angle_next ** 2 + angular_vel_next ** 2

    # 软着陆接触奖励：双腿着垫且速度平稳时给予一次性正反馈
    left_next = next_obs[6]
    right_next = next_obs[7]
    contact_bonus = 0.0
    if left_next > 0.5 and right_next > 0.5 and abs(x_vel_next) < 0.3 and abs(y_vel_next) < 0.3:
        contact_bonus = 1.0

    # 加权组合
    total = (
        10.0 * progress
        - 1.0 * speed_penalty
        - 0.5 * angle_penalty
        + 2.0 * contact_bonus
    )

    components = {
        "progress": 10.0 * progress,
        "speed_penalty": -1.0 * speed_penalty,
        "angle_penalty": -0.5 * angle_penalty,
        "contact_bonus": 2.0 * contact_bonus
    }

    return float(total), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-18.891319, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-44.197166, 17.370737]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 12.089526 | 73.1% | 84.0% | 100.0% |
| angle_penalty | -2.650961 | -16.0% | 16.0% | 100.0% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |
| speed_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


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
