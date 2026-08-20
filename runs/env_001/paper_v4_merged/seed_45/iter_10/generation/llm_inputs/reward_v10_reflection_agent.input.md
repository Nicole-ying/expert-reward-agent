# 1. Search objective
- target_score: 200.000000
- current_score: -45.068935
- gap_to_target: 245.068935
- target_achievement_ratio: -22.534%

# 2. 上一轮奖励函数代码（该轮得分: -45.068935）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, angvel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nleft, nright = next_obs

    # 到目标中心距离
    dist = (x**2 + y**2)**0.5 + 1e-8
    next_dist = (nx**2 + ny**2)**0.5 + 1e-8

    # 基准进展信号（delta）
    progress = 5.0 * (dist - next_dist)

    # 接触奖励（引导脚触地）
    contact_reward = 0.2 * (nleft + nright)

    # 完成因子（各子条件连续映射到[0,1]）
    proximity_factor = max(0.0, 1.0 - next_dist / 0.3)            # 距中心<0.3
    velocity_factor  = max(0.0, 1.0 - (abs(nvx) + abs(nvy)) / 0.3)  # 合速度<0.3
    angle_factor     = max(0.0, 1.0 - abs(nangle) / 0.15)         # 倾角<0.15 rad
    angvel_factor    = max(0.0, 1.0 - abs(nangvel) / 0.2)         # 角速度<0.2 rad/s
    contact_factor   = (nleft + nright) / 2.0                     # 双脚接触程度

    # min‑joint completion：只有最差条件改善总分才提高
    completion = 10.0 * min(proximity_factor, velocity_factor, angle_factor, angvel_factor, contact_factor)

    # 安全阈值惩罚（降低阈值使约束可感知）
    speed_penalty    = -0.5 * (max(0.0, abs(nvx) - 0.4) + max(0.0, abs(nvy) - 0.4))
    angle_penalty    = -1.0 * max(0.0, abs(nangle) - 0.15)
    angvel_penalty   = -0.3 * max(0.0, abs(nangvel) - 0.3)
    boundary_penalty = -2.0 * max(0.0, abs(nx) - 0.8)   # 水平出界预警

    total_reward = (progress + contact_reward + completion +
                    speed_penalty + angle_penalty + angvel_penalty + boundary_penalty)

    components = {
        'progress': progress,
        'contact_reward': contact_reward,
        'completion': completion,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty,
        'boundary_penalty': boundary_penalty
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 1000.00 | 128.47 | ✅ |
| 2 | 骨架变化: contact_reward + orientation_penalty + progress_de | — | 960.10 | 144.81 | ✅ |
| 3 | 骨架变化: contact_reward + landing_progress + orientation_pe | — | 897.70 | 141.28 | ❌ |
| 4 | 骨架变化: angle_penalty + angvel_penalty + completion_proxy  | — | 754.65 | 195.26 | ✅ |
| 5 | 骨架变化: angle_penalty + angvel_penalty + completion_proxy  | — | 750.35 | 186.99 | ❌ |
| 6 | 骨架变化: angle_penalty + angvel_penalty + completion_proxy  | — | 68.45 | -111.41 | ❌ |
| 7 | 骨架变化: angle_penalty + angvel_penalty + boundary_warning  | — | 885.15 | 142.74 | ➖ |
| 8 | 骨架变化: angle_penalty + angvel_penalty + completion_bonus  | — | 1000.00 | -58.91 | ❌ |
| 9 | 骨架变化: angle_penalty + angvel_penalty + boundary_penalty  | — | 1000.00 | -45.07 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-45.068935, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-76.752602, -8.359576]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 4.788842 | 42.2% | 57.1% | 100.0% |
| angle_penalty | -3.265621 | -28.7% | 28.7% | 3.8% |
| speed_penalty | -1.538650 | -13.5% | 13.5% | 2.0% |
| contact_reward | 0.050000 | 0.4% | 0.4% | 0.0% |
| angvel_penalty | -0.015091 | -0.1% | 0.1% | 0.2% |
| boundary_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| completion | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Mean eval reward -45, all episodes truncated (1000 steps). Progress has largest signed share (42.2%) but is canceled by angle/speed penalties. Score highly negative despite positive progress.

**Component Anomalies**: Angle_penalty has high magnitude (28.7% signed share) yet active only 3.8% of steps—sparse large penalties. Completion and boundary_penalty dead (0 sum, 0% active). Contact_reward virtually dead in final eval.

**Training Dynamics**: No temporal monitor snapshots provided; cannot assess component growth/decay or checkpoint drift.

**Signal Quality**: Contact_reward active in training (mean 0.19, 50.2% nonzero) but zero in final eval—policy likely avoids contacts. Progress and penalty components self-cancelling, leaving no net progress towards completion. Completion metric never triggered.

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
| 3 | contact_reward + landing_progress + orientation_penalty + progress_delta + speed_penalty | 141.28 | 144.81 | -3.53 | 897.70 | contact_reward=0.135 landing_progress=0.075 orientation_penalty=-0.001 progress_delta=0.002 speed_penalty=-0.001 | no_meaningful_improvement |
| 4 | angle_penalty + angvel_penalty + completion_proxy + progress_delta + speed_penalty | 195.26 | 195.26 | 0.00 | 754.65 | angle_penalty=-0.003 angvel_penalty=-0.001 completion_proxy=0.492 progress_delta=0.016 speed_penalty=-0.007 | new_best |
| 5 | angle_penalty + angvel_penalty + completion_proxy + progress_delta + speed_penalty | 186.99 | 195.26 | -8.27 | 750.35 | angle_penalty=-0.002 angvel_penalty=-0.001 completion_proxy=0.591 progress_delta=0.012 speed_penalty=-0.005 | no_meaningful_improvement |
| 6 | angle_penalty + angvel_penalty + completion_proxy + engine_penalty + progress_delta + speed_penalty | -111.41 | 195.26 | -306.67 | 68.45 | angle_penalty=-0.001 angvel_penalty=-0.004 completion_proxy=0.005 engine_penalty=-0.006 progress_delta=0.081 | no_meaningful_improvement |
| 7 | angle_penalty + angvel_penalty + boundary_warning + contact_reward + landing_bonus + progress_delta | 142.74 | 195.26 | -52.53 | 885.15 | angle_penalty=-0.005 angvel_penalty=-0.001 boundary_warning=-0.018 contact_reward=0.360 landing_bonus=0.279 | unsolved_high_achievement_continue_from_best |
| 8 | angle_penalty + angvel_penalty + completion_bonus + progress + speed_penalty | -58.91 | 195.26 | -254.18 | 1000.00 | angle_penalty=-0.008 angvel_penalty=-0.001 completion_bonus=4.515 progress=0.017 speed_penalty=-0.014 | no_meaningful_improvement |
| 9 | angle_penalty + angvel_penalty + boundary_penalty + completion + contact_reward + progress | -45.07 | 195.26 | -240.33 | 1000.00 | angle_penalty=-0.027 angvel_penalty=-0.004 boundary_penalty=-0.001 completion=2.269 contact_reward=0.190 | no_meaningful_improvement |
