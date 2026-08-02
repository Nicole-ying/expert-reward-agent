# 1. Search objective
- target_score: 200.000000
- current_score: -102.509641
- gap_to_target: 302.509641
- target_achievement_ratio: -51.255%

# 2. 上一轮奖励函数代码（该轮得分: -102.509641）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next_obs dimensions per environment card
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Primary progress signal: dense quadratic penalty on position and velocity errors.
    #    Encourages the lander to move toward (0,0) with zero speed.
    pos_sq_error = x_pos**2 + y_pos**2
    vel_sq_error = x_vel**2 + y_vel**2
    progress = -0.05 * pos_sq_error - 0.1 * vel_sq_error

    # 2. Stability constraint: quadratic penalty on body angle and angular velocity.
    #    Keeps the lander upright and prevents excessive spinning.
    pose_penalty = -5.0 * (body_angle**2) - 0.5 * (angular_vel**2)

    # 3. Soft landing bonus: a task-completion proxy active when both legs are grounded.
    #    The bonus is large only when touchdown is gentle (low speed, nearly vertical).
    both_leg_grounded = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    speed_magnitude = abs(x_vel) + abs(y_vel)
    speed_factor = 1.0 / (1.0 + 5.0 * speed_magnitude)        # bounded signal: decays with speed
    angle_factor = 1.0 / (1.0 + 20.0 * abs(body_angle))      # bounded signal: decays with tilt
    landing_bonus = 10.0 * both_leg_grounded * speed_factor * angle_factor

    total_reward = progress + pose_penalty + landing_bonus

    components = {
        'progress': progress,
        'pose_penalty': pose_penalty,
        'landing_bonus': landing_bonus
    }
    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-102.509641, len=530.150000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-183.071485, 101.872854]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | -19.285949 | -56.8% | 56.8% | 100.0% |
| pose_penalty | -9.726772 | -28.6% | 28.6% | 100.0% |
| landing_bonus | 4.961779 | 14.6% | 14.6% | 1.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个 2D 飞行器（着陆器）轨迹优化任务。主体从视口顶部中心附近出发，初始受到随机扰动。核心目标是 **尽快、平稳地降落到中央目标平台上，并保持机身竖直稳定**。次要目标是 **尽量节省主引擎燃料**，即少用主推力。  
Agent 需要学会：向目标平台逼近、适时减速、保持小角度、最终实现低冲击的安全着陆。  
不要把“平稳着陆”与单纯的“位置到达”混淆，着陆质量（速度、姿态、接触）与燃料效率不可忽略，但到达目标是第一优先级。

## 3. 观察空间 observation_space
- **type**: `Box`
- **shape**: `[8]`
- **dtype**: `float32` (推断)
- **obs[0]**: `x_position` —— 相对于目标平台中心的水平坐标（正右方向），reward_usable: true
- **obs[1]**: `y_position` —— 相对于目标平台着陆面高度的垂直坐标（疑似上为正，平台面为 0），reward_usable: true
- **obs[2]**: `x_velocity` —— 水平线速度，reward_usable: true
- **obs[3]**: `y_velocity` —— 垂直线速度，reward_usable: true
- **obs[4]**: `body_angle` —— 机身倾角（很可能以弧度表示，0 为竖直），reward_usable: true
- **obs[5]**: `angular_velocity` —— 角速度，reward_usable: true
- **obs[6]**: `left_support_contact` —— 左支撑脚接地标志（1.0 表示接触），reward_usable: true
- **obs[7]**: `right_support_contact` —— 右支撑脚接地标志，reward_usable: true

## 4. 动作空间 action_space
- **type**: `Discrete`
- **n**: 4
- **动作清单**：
  - **action 0**: `no_engine` —— 所有引擎关闭
  - **action 1**: `left_orientation_engine` —— 启动左姿态引擎（产生角力矩，主要用于调整机头方向）
  - **action 2**: `main_engine` —— 启动主引擎（向下喷气，产生向上的推力，同时可能带来微小角力矩）
  - **action 3**: `right_orientation_engine` —— 启动右姿态引擎（与左姿态引擎相反方向）

## 5. step 与终止条件分析
### 5.1 终止模式
- **crash_or_body_contact**：机体与地面或平台发生强烈接触（可能包含坠毁或非常粗糙的着陆），触发终止。
- **horizontal_position_outside_viewport**：水平位置超出有效视野/世界边界，任务失败终止。
- **body_not_awake_or_settled**：机体进入稳定/不活跃状态（如着陆后静止），终止发生。此极可能是成功着陆后的正常终止。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
- **explicit_failure_flag_available**: false  
- **allowed_info_fields**: 无（环境 step 返回的 `info` 为空字典 `{}`）  
- **forbidden_or_uncertain_info_fields**: 所有可能的终止原因、成败标记、elapsed steps 等均不可直接使用  
- 尽管如此，成功着陆的迹象可通过 **next_obs** 间接推断：
  - 位置接近目标原点 `(0,0)`，速度接近 0，角度接近 0，且左右支撑接地标志同时为 1。  
  - 该推断路径记为 **derived_possible**，可在奖励设计中使用，但不可作为绝对成功判决。

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
| 1 | landing_bonus + pose_penalty + progress | -102.51 | -102.51 | 0.00 | 530.15 | landing_bonus=2.237 pose_penalty=-0.109 progress=-0.066 | new_best |
