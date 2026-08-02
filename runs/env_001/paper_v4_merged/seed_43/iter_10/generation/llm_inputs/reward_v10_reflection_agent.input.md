# 1. Search objective
- target_score: 200.000000
- current_score: -24.047127
- gap_to_target: 224.047127
- target_achievement_ratio: -12.024%

# 2. 上一轮奖励函数代码（该轮得分: -24.047127）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # unpack observations
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nl_contact, nr_contact = next_obs

    # distances to target (0,0)
    dist = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5

    # 1. progress delta: positive when approaching target
    delta_dist = dist - dist_next
    progress = 1.0 * delta_dist

    # 2. angle gate: linearly decay progress when body angle exceeds safe range
    safe_angle = 0.5  # radians
    gate_angle = max(0.3, 1.0 - abs(nangle) / safe_angle)

    # 3. contact factor: encourage both legs on ground
    if nl_contact == 1 and nr_contact == 1:
        contact_factor = 1.0
    elif nl_contact == 1 or nr_contact == 1:
        contact_factor = 0.7
    else:
        contact_factor = 0.4

    # shaped progress: main learning signal with safety and contact modulation
    shaped_progress = progress * gate_angle * contact_factor

    # 4. speed penalty near ground to promote gentle landing
    close_threshold = 0.5
    speed_penalty = 0.0
    if dist_next < close_threshold:
        speed_norm = abs(nvx) + abs(nvy)
        speed_penalty = -0.1 * speed_norm

    # 5. continuous success proxy using geometric mean of proximity, stability and contact
    proximity = 1.0 / (1.0 + 10.0 * dist_next)
    speed_norm_eucl = (nvx**2 + nvy**2) ** 0.5
    stability = 1.0 / (1.0 + 3.0 * speed_norm_eucl + 3.0 * abs(nangle))
    contact_quality = (nl_contact + nr_contact) / 2.0  # in [0,1]
    # geometric mean to avoid product collapse; add tiny epsilon for zero case
    product = proximity * stability * contact_quality
    eps = 1e-6
    success_factor = (max(product, eps)) ** (1.0 / 3.0)
    success_bonus = 5.0 * success_factor

    # 6. action cost: small penalty for any engine use
    action_cost = -0.01 if action != 0 else 0.0

    total_reward = shaped_progress + success_bonus + speed_penalty + action_cost

    components = {
        "progress": progress,
        "gate_angle": gate_angle,
        "contact_factor": contact_factor,
        "shaped_progress": shaped_progress,
        "speed_penalty": speed_penalty,
        "success_bonus": success_bonus,
        "action_cost": action_cost
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 68.30 | -117.88 | ✅ |
| 2 | 骨架变化: action_cost + angle_hinge + danger_penalty + progr | — | 68.35 | -117.48 | ✅ |
| 3 | 骨架变化: action_cost + angle_hinge + landing_contact_reward | — | 68.30 | -122.17 | ❌ |
| 4 | 骨架变化: action_cost + landing_contact_reward + landing_spe | — | 143.70 | -87.19 | ✅ |
| 5 | 骨架变化: action_cost + landing_contact_reward + progress_sh | — | 143.70 | -87.19 | ❌ |
| 6 | 骨架变化: action_cost + angle_hinge_penalty + landing_contac | — | 68.35 | -114.35 | ❌ |
| 7 | 骨架变化: action_cost + angle_hinge_penalty + landing_contac | — | 71.20 | -105.53 | ❓ |
| 8 | 骨架变化: angle_penalty + fuel_cost + progress_reward + soft | — | 84.45 | -124.39 | ❌ |
| 9 | 骨架变化: action_cost + contact_factor + gate_angle + progre | — | 980.75 | -24.05 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-24.047127, len=980.750000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-127.219130, 24.581447]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gate_angle | 916.943191 | 65.1% | 65.1% | 100.0% |
| contact_factor | 398.390000 | 28.3% | 28.3% | 100.0% |
| success_bonus | 75.372604 | 5.4% | 5.4% | 100.0% |
| action_cost | -9.765000 | -0.7% | 0.7% | 99.6% |
| speed_penalty | -5.777554 | -0.4% | 0.4% | 77.6% |
| progress | 1.191460 | 0.1% | 0.1% | 100.0% |
| shaped_progress | 0.410168 | 0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Mean eval reward -24.0, terminated 1/20, len 980.8. Reward dominated by gate_angle (916.9,65.1%) and contact_factor (398.4,28.3%); progress negligible (1.19,0.1%). Agent survives but fails task.

**Component Anomalies**: gate_angle+contact_factor >93% share, over-incentivize posture. progress/shaped_progress dead (<0.1%).

**Training Dynamics**: No checkpoint data; final policy exploits angle/contact without target approach.

**Signal Quality**: shaped_progress collapsed (tiny progress); success_bonus (75.4 sum) insufficient. No dead gates but reward fails to guide to goal.

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个 2D 飞行器/着陆器从视口上方出发，尽快降落到画面中央的水平目标垫上并稳定停靠。主目标是精确到达并停稳在目标垫中心（位置误差趋于零，速度接近零，两支撑脚着垫）。次要目标是尽量减少引擎使用（节能），快速完成任务。注意不要与此类任务可能混淆的纯飞行姿态控制、单纯前进速度优化或仅存活不要求停稳的任务混淆。

## 3. 观察空间 observation_space
- **type:** Box  
- **shape:** (8,)  
- **dtype:** 通常为 float64（环境默认），可视为连续浮点数。  

各索引含义：  
- `obs[0]`：`x_position`，飞行器相对目标垫中心的水平距离（向右为正），reward_usable: true  
- `obs[1]`：`y_position`，飞行器相对目标垫高度的垂直距离（向上为正，0 表示与垫面等高），reward_usable: true  
- `obs[2]`：`x_velocity`，水平线速度，reward_usable: true  
- `obs[3]`：`y_velocity`，垂直线速度，reward_usable: true  
- `obs[4]`：`body_angle`，机身倾角（弧度，0 为水平），reward_usable: true  
- `obs[5]`：`angular_velocity`，角速度，reward_usable: true  
- `obs[6]`：`left_support_contact`，左侧支撑脚触地标志（0.0 或 1.0），reward_usable: true  
- `obs[7]`：`right_support_contact`，右侧支撑脚触地标志（0.0 或 1.0），reward_usable: true

## 4. 动作空间 action_space
- **type:** Discrete  
- **n:** 4  
- **动作说明：**  
  - `action 0`：“no_engine” — 所有引擎关闭，无推力。  
  - `action 1`：“left_orientation_engine” — 点燃左侧姿态引擎，产生偏航/旋转力矩。  
  - `action 2`：“main_engine” — 点燃主引擎，产生主体推力（通常向上或沿机身轴线）。  
  - `action 3`：“right_orientation_engine” — 点燃右侧姿态引擎，产生反方向旋转力矩。

## 5. step 与终止条件分析
### 5.1 终止模式
根据 `terminated = crash_or_body_contact or horizontal_position_outside_viewport or body_not_awake_or_settled`，三种触发情景：
- **crash_or_body_contact**：飞行器主体（非支撑脚）与地面或环境障碍碰撞，通常表示失败。  
- **horizontal_position_outside_viewport**：飞行器水平超出视口范围，失败。  
- **body_not_awake_or_settled**：物理体进入休眠状态或因稳定停靠而“settled”。根据任务目标，在目标垫上稳定停靠后应触发此条件，属于成功结果；但也可能因坠毁后体僵硬休眠触发，因此需要结合其他观测才能确定是成功还是失败。  

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available:** false  
- **explicit_failure_flag_available:** false  
- **allowed_info_fields:** `info` 当前为空字典 `{}`，无法直接获得任何结果标志。  
- **forbidden_or_uncertain_info_fields:** 任何未声明的字段（如 `success`、`failure`、`termination_reason` 等）均不可信。  

成功/失败只能通过 **derived_possible** 方式从观测序列中推断：  
- 成功终端（目标垫稳定停靠）：`episode` 结束时，`x_position`≈0, `y_position`≈0, `|x_velocity|` 和 `|y_velocity|` 很小，`left_support_contact`==1, `right_support_contact`==1，且未发生 `horizontal_out` 现象。  
- 坠毁终端：`episode` 结束时，倾角 `|body_angle|` 很大，或 `y_position` 异常低（地面以下），或只有一只脚接触物且位置远离目标垫。  
- 出界终端：`episode` 结束时，`x_position` 绝对值超出合理范围（范围需通过环境运行中观测到的边界估计，如 |x| > 1.5，或从 rollouts 中统计）。

## 7. 可用于奖励函数的信号
- **位置相关：**  
  - `x_position`, `y_position`（可直接计算到目标垫中心的欧氏距离 `dist = sqrt(x² + y²)`）  
  - 可衍生：`dist_to_target`，上一时刻距离与当前距离之差（delta progress）：`progress = dist(obs) - dist(next_obs)`，正值表示靠近。  
- **速度相关：**  
  - `x_velocity`, `y_velocity` 可用于惩罚接近时的剩余动能，或构建稳定条件。  
- **姿态相关：**  
  - `body_angle` 可用于 hinge penalty（防止倾斜过大）；`angular_velocity` 用于抑制快速旋转。  
- **接触信号：**  
  - `left_support_contact`, `right_support_contact` 可判断双脚是否着垫，是成功停靠的必要条件。  
- **动作相关：**  
  - `action` 值可用于计算动作成本（action ≠ 0 时轻微惩罚）。  
- **衍生信号（derived_possible，需与环境边界参数拟合）：**  
  - **终端成功事件：** 当 `terminated` 且 `dist_to_target` 小于阈值 (如 0.1)，速度幅值低于阈值，且 `left_support_contact` 和 `right_support_contact` 均为 1。  
  - **坠毁事件：** 当 `terminated` 且不满足成功条件，同时 `|body_angle|` 过大或 `y_position` 偏离过大。  
  - **出界事件：** 当 `terminated` 且 `x_position` 超出可靠运行范围。

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
| 1 | action_cost + angle_hinge + progress_shaping | -117.88 | -117.88 | 0.00 | 68.30 | action_cost=-0.001 angle_hinge=-0.001 progress_shaping=0.015 | new_best |
| 2 | action_cost + angle_hinge + danger_penalty + progress_shaping | -117.48 | -117.48 | 0.00 | 68.35 | action_cost=-0.002 angle_hinge=-0.001 danger_penalty=-0.002 progress_shaping=0.015 | new_best |
| 3 | action_cost + angle_hinge + landing_contact_reward + progress_shaping | -122.17 | -117.48 | -4.69 | 68.30 | action_cost=-0.001 angle_hinge=-0.001 landing_contact_reward=0.003 progress_shaping=0.015 | no_meaningful_improvement |
| 4 | action_cost + landing_contact_reward + landing_speed_gate + progress_shaping + shaped_progress | -87.19 | -87.19 | 0.00 | 143.70 | action_cost=-0.002 landing_contact_reward=0.007 landing_speed_gate=0.879 progress_shaping=0.014 shaped_progress=0.011 | new_best |
| 5 | action_cost + landing_contact_reward + progress_shaping + shaped_progress | -87.19 | -87.19 | 0.00 | 143.70 | action_cost=-0.002 landing_contact_reward=0.007 progress_shaping=0.014 shaped_progress=0.011 | no_meaningful_improvement |
| 6 | action_cost + angle_hinge_penalty + landing_contact_reward + progress_shaping + shaped_progress | -114.35 | -87.19 | -27.16 | 68.35 | action_cost=-0.001 angle_hinge_penalty=-0.000 landing_contact_reward=0.003 progress_shaping=0.015 shaped_progress=0.012 | no_meaningful_improvement |
| 7 | action_cost + angle_hinge_penalty + landing_contact_reward + progress_shaping + shaped_progress | -105.53 | -87.19 | -18.34 | 71.20 | action_cost=-0.001 angle_hinge_penalty=-0.000 landing_contact_reward=0.003 progress_shaping=0.008 shaped_progress=0.007 | unsolved_stagnation_fresh_restart |
| 8 | angle_penalty + fuel_cost + progress_reward + soft_landing_bonus + speed_penalty | -124.39 | -87.19 | -37.20 | 84.45 | angle_penalty=-0.002 fuel_cost=-0.005 progress_reward=0.009 soft_landing_bonus=0.004 speed_penalty=-0.015 | no_meaningful_improvement |
| 9 | action_cost + contact_factor + gate_angle + progress + shaped_progress + speed_penalty | -24.05 | -24.05 | 0.00 | 980.75 | action_cost=-0.008 contact_factor=0.696 gate_angle=0.747 progress=0.003 shaped_progress=0.001 | new_best |
