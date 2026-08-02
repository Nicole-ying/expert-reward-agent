# 1. Search objective
- target_score: 200.000000
- current_score: -110.682953
- gap_to_target: 310.682953
- target_achievement_ratio: -55.341%

# 2. 上一轮奖励函数代码（该轮得分: -110.682953）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 计算当前和下一步到目标中心的欧氏距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    
    # 1. 距离缩短奖励（靠近目标为正）
    approach_delta = current_dist - next_dist
    approach_reward = 8.0 * approach_delta
    
    # 2. 成功着陆软奖励：联合条件代理（几何平均防塌缩）
    f_dist = max(0.0, 1.0 - next_dist / 0.3)               # 距离因子
    speed_mag = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    f_speed = max(0.0, 1.0 - speed_mag / 0.5)              # 速度因子
    f_angle = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)      # 姿态角因子
    f_contact = (next_obs[6] + next_obs[7]) / 2.0          # 接触因子
    success_proxy = (f_dist * f_speed * f_angle * f_contact) ** 0.25
    success_reward = 3.0 * success_proxy
    
    # 3. 燃料消耗惩罚（离散动作：非零动作即惩罚）
    fuel_penalty = -0.05 if action != 0 else 0.0
    
    total = approach_reward + success_reward + fuel_penalty
    
    components = {
        'approach_delta': float(approach_delta),
        'success_proxy': float(success_proxy),
        'fuel_penalty': float(fuel_penalty)
    }
    return (float(total), components)
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 68.75 | -92.75 | ✅ |
| 2 | 骨架变化: approach_reward + fuel_penalty + goal_progress + s | — | 661.90 | 194.87 | ✅ |
| 3 | 骨架变化: approach_reward + fuel_penalty + goal_progress + s | — | 896.45 | 163.45 | ❌ |
| 4 | 骨架变化: fuel_penalty + shaping | — | 68.40 | -119.85 | ❌ |
| 5 | 骨架变化: angular_penalty + fuel_penalty + shaping | — | 91.40 | -91.37 | ➖ |
| 6 | 骨架变化: angular_vel_penalty + approach_delta + fuel_penalt | — | 68.35 | -117.48 | ❌ |
| 7 | 骨架变化: approach_delta + fuel_penalty + hinge_angle + hing | — | 68.40 | -116.69 | ❌ |
| 8 | 骨架变化: angle_improvement + approach_delta + fuel_penalty  | — | 87.45 | -103.69 | ➖ |
| 9 | 骨架变化: approach_delta + fuel_penalty + success_proxy | — | 69.70 | -110.68 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-110.682953, len=69.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-205.796452, 12.522001]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_delta | 1.118792 | 69.9% | 72.4% | 100.0% |
| success_proxy | 0.273736 | 17.1% | 17.1% | 0.5% |
| fuel_penalty | -0.167500 | -10.5% | 10.5% | 4.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 18/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


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
| 3 | approach_reward + fuel_penalty + goal_progress + stable_bonus | 163.45 | 194.87 | -31.42 | 896.45 | approach_reward=0.079 fuel_penalty=-0.007 goal_progress=0.002 stable_bonus=0.537 | no_meaningful_improvement |
| 4 | fuel_penalty + shaping | -119.85 | 194.87 | -314.72 | 68.40 | fuel_penalty=-0.003 shaping=0.098 | no_meaningful_improvement |
| 5 | angular_penalty + fuel_penalty + shaping | -91.37 | 194.87 | -286.24 | 91.40 | angular_penalty=-0.002 fuel_penalty=-0.005 shaping=0.092 | unsolved_high_achievement_continue_from_best |
| 6 | angular_vel_penalty + approach_delta + fuel_penalty + stable_proxy | -117.48 | 194.87 | -312.34 | 68.35 | angular_vel_penalty=-0.004 approach_delta=0.016 fuel_penalty=-0.002 stable_proxy=0.011 | no_meaningful_improvement |
| 7 | approach_delta + fuel_penalty + hinge_angle + hinge_x + success_proxy | -116.69 | 194.87 | -311.56 | 68.40 | approach_delta=0.016 fuel_penalty=-0.007 hinge_angle=0.004 hinge_x=0.000 success_proxy=0.024 | no_meaningful_improvement |
| 8 | angle_improvement + approach_delta + fuel_penalty + landing_proxy + speed_improvement + x_penalty | -103.69 | 194.87 | -298.56 | 87.45 | angle_improvement=-0.002 approach_delta=0.015 fuel_penalty=-0.007 landing_proxy=0.004 speed_improvement=0.005 | unsolved_high_achievement_continue_from_best |
| 9 | approach_delta + fuel_penalty + success_proxy | -110.68 | 194.87 | -305.55 | 69.70 | approach_delta=0.016 fuel_penalty=-0.004 success_proxy=0.004 | no_meaningful_improvement |
