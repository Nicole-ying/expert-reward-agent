# 1. Search objective
- target_score: 200.000000
- current_score: 215.224738
- gap_to_target: -15.224738
- target_achievement_ratio: 107.612%

# 2. 上一轮奖励函数代码（该轮得分: 215.224738）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # obs / next_obs : [x, y, vx, vy, angle, ang_vel, left_contact, right_contact]
    # goal is at origin (0,0); x,y relative to target pad
    dist_old = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_new = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    delta_distance = dist_old - dist_new  # positive when approaching

    # Soft landing progress: encourage closeness, low speed, low angle,
    # boosted when both legs contact the target platform
    near_goal = 1.0 / (1.0 + 5.0 * dist_new)
    speed_sq = next_obs[2]**2 + next_obs[3]**2
    low_speed = 1.0 / (1.0 + 10.0 * speed_sq)
    abs_angle = abs(next_obs[4])
    low_angle = 1.0 / (1.0 + 20.0 * abs_angle)
    # contact factor: 1.0 when none or one leg, 1.0+2.0=3.0 when both legs contact
    contact_bonus = 1.0 + 2.0 * (next_obs[6] * next_obs[7])
    soft_progress = near_goal * low_speed * low_angle * contact_bonus

    # Engine usage penalty: penalize any thrust action (discrete actions 1,2,3)
    engine_penalty = 1.0 if action != 0 else 0.0

    # Weights
    w_dist = 10.0
    w_soft = 2.0
    w_engine = 0.01

    total = (w_dist * delta_distance +
             w_soft * soft_progress -
             w_engine * engine_penalty)

    components = {
        'distance_delta': w_dist * delta_distance,
        'soft_landing_progress': w_soft * soft_progress,
        'engine_penalty': -w_engine * engine_penalty,
    }
    return float(total), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 554.20 | 216.19 | ✅ |
| 2 | 骨架变化: distance_delta + engine_penalty + soft_landing_pro | — | 356.30 | 243.73 | ✅ |
| 3 | 骨架变化: distance_delta + engine_penalty + soft_landing_pro | — | 557.25 | 215.22 | ❓ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=215.224738, len=557.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[183.970636, 261.090338]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_progress | 497.166490 | 96.2% | 96.2% | 100.0% |
| distance_delta | 13.857126 | 2.7% | 2.8% | 97.5% |
| engine_penalty | -5.106000 | -1.0% | 1.0% | 91.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: mean score 215.2, 100% termination, ep len 557.3. soft_landing_progress ep_sum_mean 497.2 (96.2% signed share); distance_delta 13.9 (2.7%); engine_penalty -5.1 (-1.0%).

**Component Anomalies**: soft_landing_progress dominates reward (>70% share). No dead components (active rates ≥91.6%). Engine penalty negligible in magnitude (1.0% magnitude share).

**Training Dynamics**: No temporal checkpoint data; trends across training unknown.

**Signal Quality**: Soft_landing_progress near-monopoly dilutes distance and penalty signals. Engine penalty active but tiny (ep_sum -5.1). No dead gates detected; coupling of sub-signals in soft_progress not observed.

**Evidence Confidence**: `low`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
任务目标是控制一个受随机初始力作用的二维飞行器，使其从视野顶部中央出发，尽可能快地飞抵并稳定停靠于中央目标平台上。核心目标是“到达并停靠”（reaching and settling），附属优化目标包括“尽可能少用引擎推力”和“保持姿态稳定、安全接触”。不应将其混淆为单纯的存活任务、无限制漫游任务或多目标博弈任务。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float64 (推测连续量)
- obs[0]: x_position (水平方向相对于目标平台的坐标)，reward_usable: true
- obs[1]: y_position (垂直方向相对于平台高度的坐标)，reward_usable: true
- obs[2]: x_velocity (水平线速度)，reward_usable: true
- obs[3]: y_velocity (垂直线速度)，reward_usable: true
- obs[4]: body_angle (机体朝向角)，reward_usable: true
- obs[5]: angular_velocity (角速度)，reward_usable: true
- obs[6]: left_support_contact (左支撑腿接触标志, 0/1)，reward_usable: true
- obs[7]: right_support_contact (右支撑腿接触标志, 0/1)，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine (无推力)，即不激活任何引擎
- action 1: left_orientation_engine (左姿态引擎)，产生顺时针或逆时针力矩中的一种；具体方向需在交互中推断，但用于调整朝向
- action 2: main_engine (主引擎)，沿机体纵轴提供推力，用于平移/减速/抵抗重力
- action 3: right_orientation_engine (右姿态引擎)，产生与左姿态引擎相反的力矩，用于反方向姿态修正

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功终止；任务期望通过“到达并稳定停靠”后 episode 结束，这很可能通过 **timeout/truncation** 或在目标区域达到低速度、双腿接触、小角度等条件后被环境内部判定为 settled 而终止。
- failure-like termination:
  - *crash_or_body_contact*: 机体部分（非支撑腿）碰撞地面/平台以外区域，或姿态严重偏离导致翻倒。
  - *horizontal_position_outside_viewport*: 机体飞出水平边界，视为严重失控。
  - *body_not_awake_or_settled*: 可能是检测到速度/加速度极小但未达成着陆条件，或进入睡眠状态的超时机制。
- ambiguous termination: 支撑腿接触目标平台但未满足所有稳定条件，被 terminated 可能属于部分成功/硬着陆，不能直接视为完美成功。
- truncation: 任务可能包含 episode 长度上限，届时会直接截断。该截断不携带成功/失败固有语义。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 根据 source，step 返回空字典 `{}` ，因此 `info` 无任何可用字段。
- forbidden_or_uncertain_info_fields: info字典为空，无字段可用；不得依赖任何隐式 info 键。

## 7. 可用于奖励函数的信号
- position: `next_obs[0:2]` 表示相对于目标平台的位置。可计算当前距离、距离变化量。
- velocity: `next_obs[2:4]` 线速度。可用于接近速度、稳定着陆时趋零、水平漂移控制。
- orientation: `next_obs[4]` 机体角度。可用于姿态维护、着陆时接近水平的奖励/惩罚。
- contact:
  - `next_obs[6]` 左支撑腿接触
  - `next_obs[7]` 右支撑腿接触
  - derived_possible: 双腿同时接触（legs_contact = left & right）是成功着陆的关键条件，可直接从观测构造。
- action/engine: `action` 可以用于对引擎使用施加惩罚。
- other:
  - angular_velocity `next_obs[5]` 可用于控制姿态抖动的阻尼惩罚。
  - derived_possible: settled 成功事件可间接推断：如果 episode 未因 crash/越界终止而截断，且最后几步保持双腿接触、低速度、小角度，则很可能为成功着陆。可在最终奖励中使用 sparse terminal success bonus，但必须标注为 derived_possible，且需在策略中小心处理以避免误判。

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
| 1 | angle_penalty + distance_delta + engine_penalty + soft_landing_progress | 216.19 | 216.19 | 0.00 | 554.20 | angle_penalty=-0.001 distance_delta=0.031 engine_penalty=-0.007 soft_landing_progress=0.728 | target_solved_new_best |
| 2 | distance_delta + engine_penalty + soft_landing_progress | 243.73 | 243.73 | 0.00 | 356.30 | distance_delta=0.029 engine_penalty=-0.007 soft_landing_progress=0.767 | target_solved_new_best |
| 3 | distance_delta + engine_penalty + soft_landing_progress | 215.22 | 243.73 | -28.50 | 557.25 | distance_delta=0.032 engine_penalty=-0.007 soft_landing_progress=1.634 | target_solved_no_improvement |
