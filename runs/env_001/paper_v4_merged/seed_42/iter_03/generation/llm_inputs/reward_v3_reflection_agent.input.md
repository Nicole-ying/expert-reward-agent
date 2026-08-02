# 1. Search objective
- target_score: 200.000000
- current_score: 194.868331
- gap_to_target: 5.131669
- target_achievement_ratio: 97.434%

# 2. 上一轮奖励函数代码（该轮得分: 194.868331）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 航向进展：距离目标越近越好（improvement_delta）
    d_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    d_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = d_prev - d_next
    goal_progress = 1.0 * progress

    # 2. 稳定停靠奖励：靠近目标时鼓励低速、竖直、双腿接触
    proximity_thresh = 0.5
    proximity_gate = max(0.0, 1.0 - d_next / proximity_thresh)

    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    vel_thresh = 0.2
    velocity_bonus = 0.5 * max(0.0, 1.0 - speed / vel_thresh)

    angle_thresh = 0.1
    angle_bonus = 0.2 * max(0.0, 1.0 - abs(next_obs[4]) / angle_thresh)

    contact_bonus = 1.0 * next_obs[6] * next_obs[7]

    stable_bonus = proximity_gate * (velocity_bonus + angle_bonus + contact_bonus)

    # 3. 燃料效率惩罚
    fuel_penalty = -0.01 if action != 0 else 0.0

    # 4. 密集距离奖励：越接近目标奖励越大（连续有界）
    approach_reward = 0.1 / (1.0 + d_next)

    total_reward = goal_progress + stable_bonus + fuel_penalty + approach_reward
    components = {
        'goal_progress': float(goal_progress),
        'stable_bonus': float(stable_bonus),
        'fuel_penalty': float(fuel_penalty),
        'approach_reward': float(approach_reward)
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 68.75 | -92.75 | ✅ |
| 2 | 骨架变化: approach_reward + fuel_penalty + goal_progress + s | — | 661.90 | 194.87 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=194.868331, len=661.900000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[68.690992, 250.993146]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stable_bonus | 281.108735 | 81.7% | 81.7% | 82.8% |
| approach_reward | 55.277598 | 16.1% | 16.1% | 100.0% |
| fuel_penalty | -6.307000 | -1.8% | 1.8% | 95.3% |
| goal_progress | 1.298210 | 0.4% | 0.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: mean_eval_reward=194.9, terminated=16/20, ep_len=661.9. Reward dominated by stable_bonus (81.7% signed share).

**Component Anomalies**: stable_bonus dominates (81.7% share, 82.8% active); approach_reward active 100% but only 16.1% share; fuel_penalty negative but negligible magnitude.

**Training Dynamics**: No temporal monitor data; only final-policy composition.

**Signal Quality**: stable_bonus sparse (proximity gate, contacts) yields high reward when active; goal_progress negligible (0.4%); original_env_reward mean negative (-0.012); coupling: fuel_penalty only with non-zero action, effect small.

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个 2D 飞行器从视口顶部出发，以最短时间和最少推力消耗到达并稳定停靠在画面中心的目标平台上。  
要求同时满足：水平与垂直位置均收敛至平台原点、速度趋近于零、身体姿态保持稳定、左右支撑腿同时与平台接触，且过程中避免坠毁、翻倾或飞出边界。  
任务核心是精准导航‑停靠；附属优化是燃料经济与快速性，两者不应混淆为主要目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32 (隐含，所有分量均为连续值或 0/1 浮点数)

各维度含义：

- obs[0]: x_position — 飞行器相对于目标平台中心的水平坐标，reward_usable: true  
- obs[1]: y_position — 飞行器相对于平台高度的垂直坐标，reward_usable: true  
- obs[2]: x_velocity — 水平线速度，reward_usable: true  
- obs[3]: y_velocity — 垂直线速度，reward_usable: true  
- obs[4]: body_angle — 身体朝向角，reward_usable: true  
- obs[5]: angular_velocity — 角速度，reward_usable: true  
- obs[6]: left_support_contact — 左支撑腿是否与表面接触 (1.0 接触，0.0 未接触)，reward_usable: true  
- obs[7]: right_support_contact — 右支撑腿是否与表面接触 (1.0 接触，0.0 未接触)，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4

各动作含义：

- action 0: no_engine — 不启动任何引擎，只靠惯性运动  
- action 1: left_orientation_engine — 点燃左侧姿态引擎，用于调整角度/旋转  
- action 2: main_engine — 点燃主引擎，通常产生沿身体某方向的推力（可能包含垂直方向的一次性推力）  
- action 3: right_orientation_engine — 点燃右侧姿态引擎，与左引擎反向旋转

动作选择直接影响燃料消耗和姿态变化，奖励设计中需要跟踪动作计数来估计燃料/推力使用。

## 5. step 与终止条件分析
### 5.1 终止模式
- crash_or_body_contact — 飞行器主体发生不应有的碰撞或坠毁，通常视为失败  
- horizontal_position_outside_viewport — 水平位置超出画面边界，视为失败  
- body_not_awake_or_settled — 身体进入沉睡状态或判定为已稳定停靠，可能是成功，但源码中未区分是否为正常着陆成功

无任何显式成功/失败标志传入 info 字典，因此需要**通过观测信号间接推断**终止原因。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: (空字典，没有任何字段)  
- forbidden_or_uncertain_info_fields: 禁止使用 info 读取任何字段，因为环境不提供额外信息

终止判断：
- 若 episode 终止 (not truncated) 且最终观测满足 `距离目标近、双腿接触、速度低、姿态角小`，则推断为**成功停靠** (derived_possible)  
- 若最终状态中出现任意一条不满足（如位置严重偏离、未接触、速度极大、角度过大），则推断为**失败**（坠毁、出界等）  
- 由于 termination 函数已混合了成功与失败条件，无法直接从环境获取标签，所以成功奖励必须通过 derived 推断给出

## 7. 可用于奖励函数的信号
可直接使用的观测信号：
- position (相对于目标): `obs[0]` (x), `obs[1]` (y)  
- velocity: `obs[2]` (vx), `obs[3]` (vy)  
- orientation: `obs[4]` (angle), `obs[5]` (angular_vel)  
- contact: `obs[6]` (left contact), `obs[7]` (right contact)  
- action/engine: 当前动作 `action`，可用于检测引擎使用

间接可用信号（从观测推导）：
- distance_to_target: ‖(obs[0], obs[1])‖  
- is_crashed_or_oob: 由最终状态的位置、速度突变、接触缺失推断 (derived_possible)  
- is_successful_landing: 距离近、双腿均接触、速度低、角速度低 (derived_possible)  
- fuel_usage: 动作 1、2、3 视为消耗燃料（可加权计数）

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
| 1 | fuel_penalty + goal_progress + stable_bonus | -92.75 | -92.75 | 0.00 | 68.75 | fuel_penalty=-0.002 goal_progress=0.016 stable_bonus=0.022 | new_best |
| 2 | approach_reward + fuel_penalty + goal_progress + stable_bonus | 194.87 | 194.87 | 0.00 | 661.90 | approach_reward=0.078 fuel_penalty=-0.007 goal_progress=0.002 stable_bonus=0.714 | new_best |
