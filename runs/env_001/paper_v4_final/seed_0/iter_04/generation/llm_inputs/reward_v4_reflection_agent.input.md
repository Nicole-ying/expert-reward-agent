# 1. Search objective
- target_score: 200.000000
- current_score: -81.431904
- gap_to_target: 281.431904
- target_achievement_ratio: -40.716%

# 2. 上一轮奖励函数代码（该轮得分: -81.431904）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D lander goal-reaching task.
    Drives the agent to reach the target pad and settle gently.
    v3: removed contact_ok from soft_landing so agent gets positive signal
        when approaching the target zone, not only after both feet touch.
    """
    # ---------- constants ----------
    PROGRESS_WEIGHT = 1.0
    LANDING_WEIGHT = 0.2
    ANGLE_PENALTY_WEIGHT = 0.01
    ANGULAR_VELOCITY_PENALTY_WEIGHT = 0.02

    PROXIMITY_THRESHOLD = 0.5
    ANGLE_THRESHOLD = 0.5
    VELOCITY_THRESHOLD = 0.5
    ANGULAR_VELOCITY_THRESHOLD = 0.5

    # ---------- unpack observations ----------
    x_o, y_o, x_v_o, y_v_o, angle_o, angvel_o, left_o, right_o = tuple(obs)
    x_n, y_n, x_v_n, y_v_n, angle_n, angvel_n, left_n, right_n = tuple(next_obs)

    # ---------- 1) progress to target ----------
    R_obs = (x_o ** 2 + y_o ** 2) ** 0.5
    R_next = (x_n ** 2 + y_n ** 2) ** 0.5
    progress_reward = PROGRESS_WEIGHT * (R_obs - R_next)

    # ---------- 2) soft landing incentive (now without contact requirement) ----------
    proximity = max(0.0, 1.0 - R_next / PROXIMITY_THRESHOLD)
    angle_ok = max(0.0, 1.0 - abs(angle_n) / ANGLE_THRESHOLD)
    vel_ok = max(0.0, 1.0 - (abs(x_v_n) + abs(y_v_n)) / VELOCITY_THRESHOLD)

    soft_landing = LANDING_WEIGHT * proximity * angle_ok * vel_ok

    # ---------- 3) light angular penalty ----------
    angle_penalty = -ANGLE_PENALTY_WEIGHT * (angle_n ** 2)

    # ---------- 4) angular velocity hinge penalty ----------
    angular_velocity_penalty = (
        -ANGULAR_VELOCITY_PENALTY_WEIGHT
        * max(0.0, abs(angvel_n) - ANGULAR_VELOCITY_THRESHOLD)
    )

    # ---------- aggregate ----------
    total_reward = progress_reward + soft_landing + angle_penalty + angular_velocity_penalty

    components = {
        "progress_reward": progress_reward,
        "soft_landing": soft_landing,
        "angle_penalty": angle_penalty,
        "angular_velocity_penalty": angular_velocity_penalty
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 68.45 | -111.81 | ✅ |
| 2 | 通过新增加速速度惩罚，agent 将学会在接近目标时控制姿态稳定性，减少 crash 终止，延长 episode ... | 通过新增加速速度惩罚，agent 将学会在接近目标时控制姿态稳定性，减少 crash 终止，延长 episode ... | 68.45 | -109.32 | ✅ |
| 3 | 移除 `contact_ok` 后，agent 在靠近目标且姿态/速度良好时即可获得正向奖励，学会"飞到目标上方"... | 移除 `contact_ok` 后，agent 在靠近目标且姿态/速度良好时即可获得正向奖励，学会"飞到目标上方"... | 69.50 | -81.43 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-81.431904, len=69.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-108.776750, -45.881498]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 1.120410 | 89.6% | 92.9% | 100.0% |
| angular_velocity_penalty | -0.045280 | -3.6% | 3.6% | 0.9% |
| soft_landing | 0.042450 | 3.4% | 3.4% | 0.7% |
| angle_penalty | -0.000933 | -0.1% | 0.1% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 19/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主体是一个2D飞行器/着陆器类型的轨迹优化问题。智能体从视口顶部中央附近以随机初始力出发，目标是以**最短时间**到达中央目标垫（target pad）并稳定停靠，同时**尽可能减少引擎推力**的使用。  
成功意味着智能体轻柔地接触目标垫、保持直立姿态、速度接近于零，并稳定下来。失败来源于坠毁、超出水平边界或无法稳定。  
注意：不能将“快速”和“低能耗”视为等权核心目标——核心是“到达并稳定停靠”，速度与能耗是附属优化目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32 (推断，因未明确给出但通常为 float)
- obs[0]: x_position — 相对于目标垫的水平坐标（归一化/缩放未知，但语义为横向误差）。reward_usable: true
- obs[1]: y_position — 相对于目标垫高度的垂直坐标。reward_usable: true
- obs[2]: x_velocity — 水平线速度。reward_usable: true
- obs[3]: y_velocity — 垂直线速度。reward_usable: true
- obs[4]: body_angle — 机体方向角。reward_usable: true
- obs[5]: angular_velocity — 角速度。reward_usable: true
- obs[6]: left_support_contact — 左支撑脚接触标志（1.0 表示接触）。reward_usable: true
- obs[7]: right_support_contact — 右支撑脚接触标志（1.0 表示接触）。reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0 (no_engine): 不点火
- action 1 (left_orientation_engine): 左侧姿态引擎点火
- action 2 (main_engine): 主引擎（向下推力）点火
- action 3 (right_orientation_engine): 右侧姿态引擎点火

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  “body_not_awake_or_settled” 可能表示机体进入休眠/稳定的状态。当同时满足位置接近目标、速度很低、姿态垂直且存在支撑接触时，极可能对应成功着陆。  
- failure-like termination:  
  “crash_or_body_contact”（与地面/物体不安全碰撞）、  
  “horizontal_position_outside_viewport”（水平出界）属于明确的失败类终止。
- ambiguous termination:  
  “body_not_awake_or_settled” 在远离目标垫时也可能触发（如坠毁后静止），需结合其他观测信号才能判定成功与否。
- truncation: 未提及。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无 (info dict 为空 {})
- forbidden_or_uncertain_info_fields: 全部 info 字段均不存在且禁止依赖。

## 7. 可用于奖励函数的信号
- position: x_position, y_position — 可直接计算到目标的欧氏距离及变化量。
- velocity: x_velocity, y_velocity — 可用于评估接近平稳、能量或冲击。
- orientation: body_angle — 保持竖直（接近0）的约束信号。
- contact: left_support_contact, right_support_contact — 用于判断是否安全接触垫子。
- action/engine: 离散动作可转化为引擎使用代价。
- derived_possible（间接推断）:
  - 成功着陆推断：若 episode 因 body_not_awake_or_settled 终止，且在终止前的最后一步 next_obs 中 (|x_position| 小, |y_position| 小, 速度低, |body_angle| 小, 至少一个支撑接触) → 可视为成功事件，用于终端奖励。
  - 坠毁推断：若 episode 终止且 body_angle 巨变、y_position 突变、接触信号异常激活 → 可推导失败。
  - 注意：这些推断不能在 compute_reward 中直接访问 done 标志，但可以通过检查 next_obs 的状态组合来实现隐式终端奖励——step 执行后调用 compute_reward 时若 next_obs 恰好是终态，则可用该状态判定。

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
| 1 | angle_penalty + progress_reward + soft_landing | -111.81 | -111.81 | 0.00 | 68.45 | angle_penalty=-0.000 progress_reward=0.016 soft_landing=0.001 | new_best |
| 2 | angle_penalty + angular_velocity_penalty + progress_reward + soft_landing | -109.32 | -109.32 | 0.00 | 68.45 | angle_penalty=-0.000 angular_velocity_penalty=-0.001 progress_reward=0.016 soft_landing=0.001 | new_best |
| 3 | angle_penalty + angular_velocity_penalty + progress_reward + soft_landing | -81.43 | -81.43 | 0.00 | 69.50 | angle_penalty=-0.000 angular_velocity_penalty=-0.001 progress_reward=0.016 soft_landing=0.001 | new_best |
