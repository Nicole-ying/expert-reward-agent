# Search objective
- target_score: 300.000000
- current_score: 231.235900
- gap_to_target: 68.764100
- target_achievement_ratio: 77.079%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 231.235900）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. forward_progress: 主学习信号 - 鼓励向前行走
    #    signal: horizontal_velocity (obs[2])
    #    operator: dense_state_signal (线性正奖励)
    #    使用线性形式而非凸化，因为v1阶段需要鼓励agent探索不同速度
    # ============================================================
    forward_velocity = obs[2]
    forward_reward = 1.0 * forward_velocity  # 系数1.0，使速度值直接作为奖励
    
    # ============================================================
    # 2. balance_maintenance: 健康约束 - 保持身体直立
    #    signals: hull_angle (obs[0]), hull_angular_velocity (obs[1])
    #    operator: dense_state_signal (hinge penalty)
    #    使用hinge形式：只在角度超过安全阈值时惩罚，避免过度约束
    #    安全阈值设为0.5弧度(~28度)，超过后线性增加惩罚
    # ============================================================
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    
    # 角度惩罚：只在超过0.5弧度时生效
    angle_threshold = 0.5
    angle_penalty = -2.0 * max(0.0, abs(hull_angle) - angle_threshold)
    
    # 角速度阻尼：轻微惩罚角速度，防止快速旋转
    angular_vel_penalty = -0.1 * hull_angular_vel ** 2
    
    balance_penalty = angle_penalty + angular_vel_penalty
    
    # ============================================================
    # 3. soft_health_gate: 将平衡状态作为门控乘到前进奖励上
    #    当身体角度接近危险时，前进奖励逐渐衰减
    #    使用倒数门形式：1 / (1 + k * abs(angle))
    #    这样在角度较小时门控接近1，角度增大时平滑衰减
    # ============================================================
    gate_k = 3.0  # 控制衰减速度
    health_gate = 1.0 / (1.0 + gate_k * abs(hull_angle))
    
    # 应用门控：前进奖励乘以健康门控
    gated_forward_reward = forward_reward * health_gate
    
    # ============================================================
    # 4. 组合最终奖励
    #    total_reward = 门控前进奖励 + 平衡惩罚
    #    门控前进奖励是主要驱动力，平衡惩罚是辅助约束
    # ============================================================
    total_reward = gated_forward_reward + balance_penalty
    
    # ============================================================
    # 组件记录
    # ============================================================
    components = {
        'forward_reward': forward_reward,          # 原始前进奖励（未门控）
        'health_gate': health_gate,                # 健康门控值
        'gated_forward_reward': gated_forward_reward,  # 门控后的前进奖励
        'angle_penalty': angle_penalty,            # 角度越界惩罚
        'angular_vel_penalty': angular_vel_penalty, # 角速度阻尼
        'balance_penalty': balance_penalty         # 总平衡惩罚
    }
    
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=231.235900, len=794.200000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-8.630315, 304.936297]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| health_gate | 739.805773 | 46.9% | 46.9% | 100.0% |
| forward_reward | 433.975669 | 27.5% | 27.5% | 100.0% |
| gated_forward_reward | 403.783431 | 25.6% | 25.6% | 100.0% |
| balance_penalty | -0.059792 | -0.0% | 0.0% | 99.9% |
| angular_vel_penalty | -0.034990 | -0.0% | 0.0% | 99.9% |
| angle_penalty | -0.024802 | -0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主目标是控制一个双足身体在崎岖地形上尽可能远、尽可能快地向前行走，同时最小化能量消耗。次目标是保持身体平衡不摔倒。不应混淆的目标是：没有明确的“到达目标点”或“抓取物体”要求，核心是持续前进与平衡维持。

## 3. 观察空间 observation_space
- type: Box
- shape: (24,)
- dtype: float32 (推测)
- obs[0]: hull_angle，身体相对于竖直方向的角度，reward_usable: true
- obs[1]: hull_angular_velocity，身体角速度，reward_usable: true
- obs[2]: horizontal_velocity，前后线速度，reward_usable: true
- obs[3]: vertical_velocity，上下线速度，reward_usable: true
- obs[4]: hip1_angle，腿1髋关节角度，reward_usable: true
- obs[5]: hip1_speed，腿1髋关节角速度，reward_usable: true
- obs[6]: knee1_angle，腿1膝关节角度，reward_usable: true
- obs[7]: knee1_speed，腿1膝关节角速度，reward_usable: true
- obs[8]: leg1_contact，腿1地面接触标志(1.0=接触,0.0=无接触)，reward_usable: true
- obs[9]: hip2_angle，腿2髋关节角度，reward_usable: true
- obs[10]: hip2_speed，腿2髋关节角速度，reward_usable: true
- obs[11]: knee2_angle，腿2膝关节角度，reward_usable: true
- obs[12]: knee2_speed，腿2膝关节角速度，reward_usable: true
- obs[13]: leg2_contact，腿2地面接触标志(1.0=接触,0.0=无接触)，reward_usable: true
- obs[14..23]: lidar_0..9，10个LIDAR测距仪沿前方地形的距离测量值，reward_usable: true

## 4. 动作空间 action_space
- type: Box
- shape: (4,)
- dtype: float32 (推测)
- action[0]: hip_torque_leg1，腿1髋关节施加的扭矩，范围[-1.0, 1.0]
- action[1]: knee_torque_leg1，腿1膝关节施加的扭矩，范围[-1.0, 1.0]
- action[2]: hip_torque_leg2，腿2髋关节施加的扭矩，范围[-1.0, 1.0]
- action[3]: knee_torque_leg2，腿2膝关节施加的扭矩，范围[-1.0, 1.0]

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain — 成功到达地形终点，属于成功终止
- failure-like termination: body_fallen_over — 身体摔倒，属于失败终止
- ambiguous termination: 无
- truncation: 无（step返回truncated=False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info为空字典{}，无显式成功标志）
- explicit_failure_flag_available: false（info为空字典{}，无显式失败标志）
- allowed_info_fields: 无（info为空字典）
- forbidden_or_uncertain_info_fields: 所有info字段均不可用

## 7. 可用于奖励函数的信号
- position: 无直接位置信号，可通过horizontal_velocity积分或LIDAR间接推断
- velocity: horizontal_velocity (obs[2]), vertical_velocity (obs[3]), hull_angular_velocity (obs[1])
- orientation: hull_angle (obs[0])
- contact: leg1_contact (obs[8]), leg2_contact (obs[13])
- action/engine: action[0..3] 扭矩值
- other: lidar_0..9 (obs[14..23]) 地形距离测量

# Compact expert route context
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angle_penalty + angular_vel_penalty + balance_penalty + forward_reward + gated_forward_reward + health_gate | 231.24 | 231.24 | 0.00 | 794.20 | angle_penalty=-0.004 angular_vel_penalty=-0.000 balance_penalty=-0.005 forward_reward=0.275 gated_forward_reward=0.243 | new_best |
