# 1. Search objective
- target_score: 200.000000
- current_score: 98.815158
- gap_to_target: 101.184842
- target_achievement_ratio: 49.408%

# 2. 上一轮奖励函数代码（该轮得分: 98.815158）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------- unpack observations -------------------
    x,  y  = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle      = obs[4]
    angvel     = obs[5]
    left_leg   = obs[6]
    right_leg  = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle  = next_obs[4]
    n_angvel = next_obs[5]
    n_left   = next_obs[6]
    n_right  = next_obs[7]

    # ------------------- helper quantities -------------------
    dist      = (x**2  + y**2)  ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    vel_abs       = (vx**2 + vy**2) ** 0.5
    next_vel_abs  = (nvx**2 + nvy**2) ** 0.5

    # ------------------- thresholds & weights -------------------
    w_progress = 1.0
    w_proximity = 10.0   # 大幅提升 soft_landing 继承者的权重

    th_angle  = 0.5    # radians, about 30 degrees
    th_vel    = 1.0    # speed magnitude
    th_angvel = 2.0    # rad/s
    th_dist   = 0.5    # proximity to target

    gate_min = 0.1     # for progress gate
    gate_min_stab = 0.2  # for stability factors to prevent collapse

    # ------------------- 1. progress signal (distance delta) -------------------
    delta_dist = max(0.0, dist - next_dist)

    gate_angle  = max(gate_min, 1.0 - abs(angle)  / th_angle)
    gate_vel    = max(gate_min, 1.0 - vel_abs      / th_vel)
    gate_angvel = max(gate_min, 1.0 - abs(angvel)  / th_angvel)
    gate = gate_angle * gate_vel * gate_angvel

    progress_gated = w_progress * delta_dist * gate

    # ------------------- 2. proximity + stability reward (replaces soft_landing) -------------------
    # proximity: how close to target (0,0)
    prox_factor = max(0.0, 1.0 - next_dist / th_dist)

    # stability after the step
    a_stab  = max(gate_min_stab, 1.0 - abs(n_angle)  / th_angle)
    v_stab  = max(gate_min_stab, 1.0 - next_vel_abs   / th_vel)
    av_stab = max(gate_min_stab, 1.0 - abs(n_angvel)  / th_angvel)
    stab = a_stab * v_stab * av_stab

    # contact gives a 1.5x multiplier, but is not required
    contact_flag = 1.0 if (n_left + n_right) >= 1.0 else 0.0
    contact_mult = 1.0 + 0.5 * contact_flag

    proximity_stability_reward = w_proximity * prox_factor * stab * contact_mult

    # ------------------- total reward -------------------
    total_reward = progress_gated + proximity_stability_reward

    components = {
        'progress_gated':   progress_gated,
        'proximity_stability': proximity_stability_reward
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 68.45 | -110.63 | ✅ |
| 2 | 密集的 proximity_stability 信号（预期 ~2.0/step）将对抗环境惩罚，引导 agent ... | 密集的 proximity_stability 信号（预期 ~2.0/step）将对抗环境惩罚，引导 agent ... | 372.45 | 98.82 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=98.815158, len=372.450000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-68.669016, 254.618698]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_stability | 2211.304918 | 100.0% | 100.0% | 74.4% |
| progress_gated | 0.417642 | 0.0% | 0.0% | 70.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Automatic fallback after 5 turns without submit. Raw data: [inspect_component_dynamics]: (no monitor snapshots — training may not have completed)
[inspect_training_feedback]: # Training Feedback

## Final-policy outcome
score=98.815158, len=372.450000, termin

**Component Anomalies**: Subagent exhausted turns without explicit submission.

**Training Dynamics**: No temporal analysis available.

**Signal Quality**: No signal quality assessment available.

**Evidence Confidence**: `low`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
任务核心是控制一个具有两条支撑腿的 2D 飞行器（启动时带有随机初始扰动），从视口顶部中央附近出发，安全、平稳地降落到画面中央的水平目标平台上，并稳定停靠。主要目标是抵达目标位置并实现“软着陆”（低速、姿态竖直、支撑腿接触），尽量减少发动机使用量（燃料消耗），同时鼓励快速完成。附属目标为姿态保持、节能及时间效率，但不应与安全降落冲突，也不应被误认为单纯的点对点导航或纯粹的平衡维持任务。

## 3. 观察空间 observation_space
- **type**: Box
- **shape**: (8,)
- **dtype**: float32（根据 Box 推断）
- **各维含义与 reward_usable 属性**：
  - **obs[0]**: x_position — 水平坐标，相对于目标平台的水平偏移，reward_usable: **true**
  - **obs[1]**: y_position — 垂直坐标，相对于目标平台高度的偏移，reward_usable: **true**
  - **obs[2]**: x_velocity — 水平线速度，reward_usable: **true**
  - **obs[3]**: y_velocity — 垂直线速度，reward_usable: **true**
  - **obs[4]**: body_angle — 机体倾斜角度，reward_usable: **true**
  - **obs[5]**: angular_velocity — 机体角速度，reward_usable: **true**
  - **obs[6]**: left_support_contact — 左支撑腿接触标志（1.0 表示接触，0.0 表示未接触），reward_usable: **true**
  - **obs[7]**: right_support_contact — 右支撑腿接触标志，reward_usable: **true**

所有观测字段均可直接用于奖励计算。

## 4. 动作空间 action_space
- **type**: Discrete
- **n**: 4
- **具体动作与含义**：
  - **action 0**: no_engine — 不启动任何引擎，自由滑行
  - **action 1**: left_orientation_engine — 启动左侧姿态引擎，产生使机体逆时针（或对应方向）旋转的力矩
  - **action 2**: main_engine — 启动主引擎，产生垂直向上的推力（减速或悬停）
  - **action 3**: right_orientation_engine — 启动右侧姿态引擎，产生与左引擎反向的力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**：身体稳定停靠（body_not_awake_or_settled）且至少有一只支撑腿接触地面，且没有发生 crash 或出界。这是期望的成功状态，表现为速度极小、姿态接近竖直、接触信号为 1，但无法从 info 直接读取，必须通过观测信号间接推断。
- **failure-like termination**：
  - crash_or_body_contact：身体主体（非支撑腿）接触地面或其他碰撞导致坠毁，通常与高速、大角度撞击有关。
  - horizontal_position_outside_viewport：水平位置超出可显示边界，即机体飞离有效区域。
- **ambiguous termination**：body_not_awake_or_settled 但左右支撑腿均未接触——可能代表机体已倒地且静止，本质上属于失败。
- **truncation**：未提及显式 step 限制，但可能存在隐式最大步数（环境未披露），此时 info 为空字典，无法直接识别。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false （info 为空字典，无任何成功标志）
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 无（info 为空）
- **forbidden_or_uncertain_info_fields**: 所有通常可能存在于 info 中的字段如 "success"、"failure"、"termination_reason"、"reward_components" 等均不存在，且不得假设它们可用。终止条件只能通过观测组合（位置、速度、角度、接触）以及是否在达到稳定/边界时 episode 结束来间接推断，标记为 **derived_possible**。

## 7. 可用于奖励函数的信号
- **位置信号**：`obs[0] x_position`、`obs[1] y_position`、`next_obs[0]`、`next_obs[1]`。可用于计算到目标 (0,0) 的距离、高度误差等。
- **速度信号**：`obs[2] x_velocity`、`obs[3] y_velocity`、`next_obs` 中对应项。可用于惩罚高速撞击或奖励低速软着陆。
- **姿态信号**：`obs[4] body_angle`、`obs[5] angular_velocity`、`next_obs` 对应项。可用于鼓励竖直姿态和减少旋转。
- **接触信号**：`obs[6] left_support_contact`、`obs[7] right_support_contact`、`next_obs` 对应项。可用于奖励支撑腿接触，表示着陆成功。
- **动作/引擎信号**：`action` 取值可用于计算燃料消耗（若 action ≠ 0 则为引擎启用）。
- **衍生推断信号（derived_possible）**：
  - 邻近成功：当 `next_obs` 中支撑腿接触为 1，且 `next_obs` 的 `x_velocity`、`y_velocity`、`body_angle` 接近 0，`y_position` 接近 0，可推断为成功软着陆。虽然无法从 info 获得标识，但在连续奖励中可通过组合条件给出额外奖励。
  - 坠毁推断：`next_obs` 中 `body_angle` 突然大幅偏离 0 或 `y_position` 突变（被重置），可间接推测崩溃，但不要用于奖励，仅用于诊断。
  - 出界推断：`x_position` 超出合理范围（如 >1 或 <-1），可用于惩罚，但此时环境已终止，一般不需要奖励。

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
| 1 | progress_gated + soft_landing | -110.63 | -110.63 | 0.00 | 68.45 | progress_gated=0.002 soft_landing=0.008 | new_best |
| 2 | progress_gated + proximity_stability | 98.82 | 98.82 | 0.00 | 372.45 | progress_gated=0.001 proximity_stability=3.735 | new_best |
