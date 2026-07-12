# ⚠️ 上一版代码验证失败
错误信息：Reward v4 failed validation: 出现未允许的 obs/next_obs 切片 (record: runs/env_002/chat_gen_pro_reflect_v1/seed_1/iter_04/generation/validations/reward_v4.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
# 设计理由
当前angle_penalty触发率为0，说明摔倒不以持续角度超限的形式出现；问题更可能是在崎岖路段缺乏速度抑制导致失控。我在gated_forward_reward的gate中引入地形粗糙度因子：用10个LIDAR读数的标准差度量地形崎岖度，乘以 `1/(1 + lidar_std)` 衰减速度奖励。这样崎岖时前进信号变弱，鼓励agent减速以保持稳定，但平坦时与原来一致。系数选择让粗糙度因子在平坦段接近1，在std=1时降至0.5，保持适度信号。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 信号提取
    horizontal_velocity = obs[2]
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]
    hip1_torque = action[0]
    knee1_torque = action[1]
    hip2_torque = action[2]
    knee2_torque = action[3]
    lidar = obs[14:24]  # 10个值

    # 组件1: 前进奖励（线性，仅正向）
    forward_reward = 1.0 * max(horizontal_velocity, 0.0)

    # 组件2: 平衡+地形门控
    angle_factor = 1.0 / (1.0 + 2.0 * abs(hull_angle))
    angular_velocity_factor = 1.0 / (1.0 + 0.5 * abs(hull_angular_velocity))

    # 地形粗糙度因子: 基于LIDAR标准差
    lidar_mean = sum(lidar) / 10.0
    lidar_var = sum((x - lidar_mean) ** 2 for x in lidar) / 10.0
    lidar_std = lidar_var ** 0.5
    terrain_factor = 1.0 / (1.0 + lidar_std)  # std=0时=1.0, std增大时缓慢衰减

    # 综合门控: 取最保守值
    balance_gate = min(angle_factor, angular_velocity_factor)
    gate = min(balance_gate, terrain_factor)
    gated_forward_reward = forward_reward * gate

    # 组件3: 能耗惩罚
    energy_penalty_weight = 0.01
    energy_penalty = -energy_penalty_weight * (
        hip1_torque**2 + knee1_torque**2 + hip2_torque**2 + knee2_torque**2
    )

    # 组件4: 额外平衡惩罚（保留，但历史触发率为0，不影响）
    angle_safety_threshold = 0.8
    angle_penalty_weight = 1.0
    angle_penalty = -angle_penalty_weight * max(0.0, abs(hull_angle) - angle_safety_threshold)

    total_reward = gated_forward_reward + energy_penalty + angle_penalty

    components = {
        "gated_forward_reward": gated_forward_reward,
        "energy_penalty": energy_penalty,
        "angle_penalty": angle_penalty,
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: 20条episode全部terminated，长度均值913，得分范围[-58,300]，angle_penalty触发率0%，gated_forward_reward占signed_share 96.6%，说明某些episode摔倒导致大幅负分。
- **behavior**: agent在崎岖路段可能因未减速而失控摔倒，拉低平均分数。
- **signal**: 缺乏对前方地形的感知信号，无法在危险路段提前抑制速度。
- **level**: Level 2（数学结构变换，在门控中引入地形粗糙度因子）
- **hypothesis**: 地形粗糙度门控使agent在崎岖处减速，降低摔倒概率，从而提升最低分并提高平均得分。
- **risk**: 速度过度抑制可能导致探索缓慢，损失极致速度得分；若LIDAR标准差常态偏高，整体速度下降会拉低平均分。

# Search objective
- target_score: 300.000000
- current_score: 213.705914
- gap_to_target: 86.294086
- target_achievement_ratio: 71.235%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 213.705914）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 信号提取 ==========
    horizontal_velocity = obs[2]
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]
    hip1_torque = action[0]
    knee1_torque = action[1]
    hip2_torque = action[2]
    knee2_torque = action[3]

    # ========== 组件 1: 前进奖励 (线性，仅正向) ==========
    # 使用线性速度，避免平方对高速的过度奖励
    forward_reward = 1.0 * max(horizontal_velocity, 0.0)

    # ========== 组件 2: 平衡约束 (soft_health_gate) ==========
    angle_factor = 1.0 / (1.0 + 2.0 * abs(hull_angle))
    angular_velocity_factor = 1.0 / (1.0 + 0.5 * abs(hull_angular_velocity))
    balance_gate = min(angle_factor, angular_velocity_factor)
    gated_forward_reward = forward_reward * balance_gate

    # ========== 组件 3: 能耗惩罚 (quadratic_penalty) ==========
    energy_penalty_weight = 0.01
    energy_penalty = -energy_penalty_weight * (
        hip1_torque**2 + knee1_torque**2 + hip2_torque**2 + knee2_torque**2
    )

    # ========== 组件 4: 额外平衡惩罚 (hinge) ==========
    angle_safety_threshold = 0.8
    angle_penalty_weight = 1.0
    angle_penalty = -angle_penalty_weight * max(0.0, abs(hull_angle) - angle_safety_threshold)

    # ========== 总奖励 ==========
    total_reward = gated_forward_reward + energy_penalty + angle_penalty

    components = {
        "gated_forward_reward": gated_forward_reward,
        "energy_penalty": energy_penalty,
        "angle_penalty": angle_penalty,
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=213.705914, len=913.000000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-58.217024, 300.779514]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_forward_reward | 400.811413 | 96.6% | 96.6% | 99.3% |
| energy_penalty | -14.212596 | -3.4% | 3.4% | 100.0% |
| angle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主目标是控制一个双足身体在崎岖地形上尽可能远、尽可能快地向前行走，同时最小化能量消耗。次目标是保持身体平衡不摔倒。不应混淆的目标是：任务不是到达某个特定目标点，也不是单纯保持平衡，而是持续前进与能耗之间的权衡。

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
- continuous: true
- bounds: [-1.0, 1.0] per joint
- action[0]: hip_torque_leg1，腿1髋关节施加的扭矩
- action[1]: knee_torque_leg1，腿1膝关节施加的扭矩
- action[2]: hip_torque_leg2，腿2髋关节施加的扭矩
- action[3]: knee_torque_leg2，腿2膝关节施加的扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（成功到达地形终点）
- failure-like termination: body_fallen_over（摔倒，任务失败）
- ambiguous termination: 无
- truncation: 无（truncated=False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（terminated=True 但无法区分是成功还是失败）
- explicit_failure_flag_available: false（同上）
- allowed_info_fields: 无（info 字典为空 {}）
- forbidden_or_uncertain_info_fields: 无（但无法从 terminated 中区分成功/失败）

## 7. 可用于奖励函数的信号
- position: 无直接位置信号，但可通过 LIDAR 间接推断前进距离
- velocity: horizontal_velocity (obs[2]), vertical_velocity (obs[3])
- orientation: hull_angle (obs[0]), hull_angular_velocity (obs[1])
- contact: leg1_contact (obs[8]), leg2_contact (obs[13])
- action/engine: action[0..3]（扭矩值，可用于能耗惩罚）
- other: 关节角度/速度 (obs[4..7], obs[9..12])，LIDAR (obs[14..23])