# 1. Search objective
- target_score: 200.000000
- current_score: 144.813937
- gap_to_target: 55.186063
- target_achievement_ratio: 72.407%

# 2. 上一轮奖励函数代码（该轮得分: 144.813937）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observations
    x = obs[0]
    y = obs[1]
    vx = obs[2]
    vy = obs[3]
    angle = obs[4]
    angvel = obs[5]
    # left_contact, right_contact from obs (not used)
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_vx = next_obs[2]
    next_vy = next_obs[3]
    next_angle = next_obs[4]
    next_angvel = next_obs[5]
    next_left = next_obs[6]
    next_right = next_obs[7]

    # 1. Progress towards center (0,0)
    dist = (x**2 + y**2)**0.5 + 1e-6
    next_dist = (next_x**2 + next_y**2)**0.5 + 1e-6
    progress_delta = dist - next_dist

    # 2. Orientation stability penalty (hinge)
    angle_threshold = 0.3
    angvel_threshold = 0.5
    angle_violation = max(0.0, abs(next_angle) - angle_threshold)
    angvel_violation = max(0.0, abs(next_angvel) - angvel_threshold)
    orientation_penalty = -0.1 * angle_violation - 0.05 * angvel_violation

    # 3. Speed safety penalty (hinge)
    speed_threshold = 0.5
    vx_violation = max(0.0, abs(next_vx) - speed_threshold)
    vy_violation = max(0.0, abs(next_vy) - speed_threshold)
    speed_penalty = -0.05 * (vx_violation + vy_violation)

    # 4. NEW: contact encouragement (dense reward for feet on ground)
    contact_reward = 0.1 * (next_left + next_right)

    total_reward = progress_delta + orientation_penalty + speed_penalty + contact_reward

    components = {
        'progress_delta': progress_delta,
        'orientation_penalty': orientation_penalty,
        'speed_penalty': speed_penalty,
        'contact_reward': contact_reward
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 1000.00 | 128.47 | ✅ |
| 2 | 骨架变化: contact_reward + orientation_penalty + progress_de | — | 960.10 | 144.81 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=144.813937, len=960.100000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[73.665443, 183.494761]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 140.110002 | 98.7% | 98.7% | 77.2% |
| progress_delta | 1.372766 | 1.0% | 1.0% | 100.0% |
| orientation_penalty | -0.188116 | -0.1% | 0.1% | 1.9% |
| speed_penalty | -0.149573 | -0.1% | 0.1% | 3.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: score=144.8, ep_len=960.1, term_rate=5% (1/20). Contact_reward dominates with 98.7% signed share (episode sum mean=140.1). Other components negligible.

**Component Anomalies**: contact_reward >70% share (dominating). orientation_penalty and speed_penalty nearly dead (active rates 1.9%, 3.3%). progress_delta 100% active but only 1.0% share.

**Training Dynamics**: No checkpoint snapshots provided; temporal trends unknown.

**Signal Quality**: Dead gates: orientation/speed penalties rarely triggered. Contact_reward overshadows progress_delta, which provides tiny signal. Missing attractor: low termination suggests agent optimizes contact survival without successful landing.

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本任务是一个 2D 载体轨迹优化问题。主体从一个随机初始位置（上部中央附近）开始，受随机初始推力影响。主要目标是使主体到达并稳定停靠在中央目标平台上，同时尽可能少地使用发动机推力（省燃料）。主体需要学会：精确接近目标、降低线速度与角速度、保持姿态稳定、安全接触平台。次要目标是快速完成和省燃料，但不能与安全着陆冲突。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断）
- 字段含义：
  - obs[0] `x_position`：相对目标垫中心的水平坐标，单位未指定，奖励可用 true
  - obs[1] `y_position`：相对目标垫高度的垂直坐标（正向可能代表高于垫），单位未指定，奖励可用 true
  - obs[2] `x_velocity`：水平线速度，奖励可用 true
  - obs[3] `y_velocity`：垂直线速度，奖励可用 true
  - obs[4] `body_angle`：主体朝向角（弧度，0为直立），奖励可用 true
  - obs[5] `angular_velocity`：角速度，奖励可用 true
  - obs[6] `left_support_contact`：左支撑脚接触标志（1.0=接触，0.0=未接触），奖励可用 true
  - obs[7] `right_support_contact`：右支撑脚接触标志（1.0=接触，0.0=未接触），奖励可用 true

所有维度均可直接或间接用于奖励函数。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：
  - 0：`no_engine` — 不点火任何发动机，无推力
  - 1：`left_orientation_engine` — 点燃左侧姿态发动机，产生向左水平推力及/或旋转力矩（具体推力方向可影响水平速度与姿态角）
  - 2：`main_engine` — 点燃主发动机，产生垂直向上推力（对抗重力），同时可能产生微小力矩
  - 3：`right_orientation_engine` — 点燃右侧姿态发动机，产生向右水平推力及/或旋转力矩

注意：动作空间未描述精确力矩，但结合`body_angle`和`angular_velocity`，左右发动机可能同时影响水平加速度和角加速度。

## 5. step 与终止条件分析
### 5.1 终止模式
- **crash_or_body_contact**：主体非支撑部分撞击地面或与平台碰撞过猛导致坠毁（如角速度/速度过大）
- **horizontal_position_outside_viewport**：水平坐标超出视口范围（视为出界失败）
- **body_not_awake_or_settled**：主体进入“静止”或“稳定着陆”状态（可能包含成功着陆或长期静止）——这可能是成功着陆的主要终止触发器

没有显式的成功或失败标志。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**
- explicit_failure_flag_available: **false**
- allowed_info_fields: {}（终端返回空的info字典）
- forbidden_or_uncertain_info_fields: 所有info字段均不可用。推断成功/失败只能通过观测信号组合与终止事件进行（derived_possible）：
  - 推断成功：终止时 `left_support_contact == 1 and right_support_contact == 1`，同时 `|x_position|` 和 `|y_position|` 接近0，`|x_velocity|`、`|y_velocity|`、`|body_angle|`、`|angular_velocity|` 均低于较小阈值。
  - 推断失败：终止时上述条件不满足，例如水平出界、或仅单脚接触、或角度/速度过大等。

## 7. 可用于奖励函数的信号
- **position**：x_position, y_position（相对目标垫中心坐标，可直接计算到目标(0,0)的距离）
- **velocity**：x_velocity, y_velocity
- **orientation**：body_angle, angular_velocity
- **contact**：left_support_contact, right_support_contact
- **action/engine**：当前 action（可用于燃料消耗惩罚，但无法知道推力大小，只能视为开关）
- **其他**：可从 next_obs 与 obs 构造差值（如 delta 位置、速度变化、角度变化），推断稳定性。

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
| 1 | orientation_penalty + progress_delta + speed_penalty | 128.47 | 128.47 | 0.00 | 1000.00 | orientation_penalty=-0.001 progress_delta=0.002 speed_penalty=-0.002 | new_best |
| 2 | contact_reward + orientation_penalty + progress_delta + speed_penalty | 144.81 | 144.81 | 0.00 | 960.10 | contact_reward=0.129 orientation_penalty=-0.001 progress_delta=0.002 speed_penalty=-0.001 | new_best |
