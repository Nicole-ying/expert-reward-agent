# 1. Search objective
- target_score: 200.000000
- current_score: -70.915446
- gap_to_target: 270.915446
- target_achievement_ratio: -35.458%

# 2. 上一轮奖励函数代码（该轮得分: -70.915446）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current observation
    x = obs[0]
    y = obs[1]
    # vx, vy not directly used for reward, but could be in future iterations
    # angle, angular_vel, contacts could be used, but here we use next_obs versions

    # Unpack next observation (state after action)
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_angle = next_obs[4]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]

    # ------------------  Main progress signal (improvement_delta)  ------------------
    # Reward distance reduction to the target pad (0,0)
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    w_progress = 1.0
    progress = (dist - next_dist)  # positive when moving toward the target

    # -----------  Contact bonus: soft proxy for task completion  -----------
    # Sparse reward when at least one leg touches the pad and the lander is close to center
    w_contact = 10.0
    contact_threshold = 0.3
    contact_bonus = 0.0
    if (next_left_contact + next_right_contact) >= 1 and next_dist < contact_threshold:
        contact_bonus = w_contact

    # -------------------  Health constraint: body angle -------------------
    # Penalize extreme tilt that could lead to a crash (hinge form)
    w_angle = 0.5
    safe_angle = 0.5          # radians
    angle_error = abs(next_angle) - safe_angle
    angle_penalty = -w_angle * angle_error if angle_error > 0 else 0.0

    # -------------------  Total reward  -------------------
    total_reward = w_progress * progress + contact_bonus + angle_penalty

    components = {
        "progress_reward": w_progress * progress,
        "contact_bonus": contact_bonus,
        "angle_penalty": angle_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-70.915446, len=985.250000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-135.695817, -31.716770]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 0.712424 | 51.6% | 90.8% | 100.0% |
| angle_penalty | -0.127609 | -9.2% | 9.2% | 0.3% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个2D飞行器从顶部出发，尽快且尽可能少地用引擎推力降落到中央目标垫上，并稳定停靠。主体要求是到达并安全着陆，附属要求是推进效率和姿态平稳。不要把纯粹的时间最短或燃料最少当成独立主目标，它们只是附属优化。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float（推测）
- obs[0]: x_position, 相对目标垫的水平坐标，reward_usable: true
- obs[1]: y_position, 相对目标垫高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity, 水平线速度，reward_usable: true
- obs[3]: y_velocity, 垂直线速度，reward_usable: true
- obs[4]: body_angle, 机体倾斜角，reward_usable: true
- obs[5]: angular_velocity, 角速度，reward_usable: true
- obs[6]: left_support_contact, 左支撑腿接触标志（0或1），reward_usable: true
- obs[7]: right_support_contact, 右支撑腿接触标志（0或1），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不点火，无推力
- action 1: left_orientation_engine — 点燃左姿态引擎，产生顺时针转动效果（具体方向取决于坐标系）
- action 2: main_engine — 点燃主引擎，通常产生向上推力以减速或提供升力
- action 3: right_orientation_engine — 点燃右姿态引擎，产生逆时针转动效果

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled 极可能表示机体已稳定停靠并进入休眠，结合观测中位置接近原点、速度极小、至少一个支撑接触时，可判定为成功着陆。
- failure-like termination: crash_or_body_contact（可能与障碍物或地面猛烈碰撞）、horizontal_position_outside_viewport（水平出界）
- ambiguous termination: 如果 body_not_awake_or_settled 发生时位置偏离目标垫或姿态异常，则为失败（如侧翻冻住）。需通过观测信号区分。
- truncation: 无明确最大步数截断说明，但可能存在时间上限；该截断不属于任务成功或失败。

### 5.2 success/failure

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
| 1 | angle_penalty + contact_bonus + progress_reward | -70.92 | -70.92 | 0.00 | 985.25 | angle_penalty=-0.017 contact_bonus=4.073 progress_reward=0.005 | new_best |
