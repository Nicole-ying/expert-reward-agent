# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 信号提取 ==========
    # 主学习信号：水平速度
    horizontal_velocity = obs[2]
    
    # 平衡信号：身体角度和角速度
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]
    
    # 能耗信号：动作扭矩
    hip1_torque = action[0]
    knee1_torque = action[1]
    hip2_torque = action[2]
    knee2_torque = action[3]
    
    # ========== 组件 1: 前进奖励 (主学习信号) ==========
    # 使用 dense_state_signal 的线性正奖励形式
    # 直接奖励水平速度，鼓励 agent 向前移动
    forward_reward_weight = 2.0
    forward_reward = forward_reward_weight * horizontal_velocity
    
    # ========== 组件 2: 平衡约束 (soft_health_gate) ==========
    # 使用 soft_health_gate 将平衡状态作为门控乘到前进奖励上
    # 当身体角度偏离竖直方向或角速度过大时，门控因子衰减
    # 这样 agent 在平衡恶化时不会获得前进奖励，而不是额外惩罚
    angle_threshold = 0.5  # 弧度，约28度
    angular_velocity_threshold = 2.0  # 弧度/秒
    
    # 计算角度偏离惩罚因子 (0~1之间)
    angle_factor = 1.0 / (1.0 + 2.0 * abs(hull_angle))
    # 计算角速度惩罚因子 (0~1之间)
    angular_velocity_factor = 1.0 / (1.0 + 0.5 * abs(hull_angular_velocity))
    
    # 组合门控因子：取最小值确保最差情况主导
    balance_gate = min(angle_factor, angular_velocity_factor)
    
    # 应用门控：前进奖励乘以平衡门控
    gated_forward_reward = forward_reward * balance_gate
    
    # ========== 组件 3: 能耗惩罚 (quadratic_penalty) ==========
    # 使用 quadratic_penalty 惩罚动作扭矩平方和
    # 权重较小，避免压制前进动力
    energy_penalty_weight = 0.01
    energy_penalty = -energy_penalty_weight * (
        hip1_torque**2 + knee1_torque**2 + hip2_torque**2 + knee2_torque**2
    )
    
    # ========== 组件 4: 额外平衡惩罚 (dense_state_signal hinge形式) ==========
    # 当身体角度超过安全阈值时，给予额外惩罚
    # 使用 hinge 形式，只在越界时生效
    angle_safety_threshold = 0.8  # 弧度，约46度
    angle_penalty_weight = 1.0
    angle_penalty = -angle_penalty_weight * max(0.0, abs(hull_angle) - angle_safety_threshold)
    
    # ========== 总奖励 ==========
    total_reward = gated_forward_reward + energy_penalty + angle_penalty
    
    # ========== 组件记录 ==========
    components = {
        "gated_forward_reward": gated_forward_reward,
        "energy_penalty": energy_penalty,
        "angle_penalty": angle_penalty,
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 任务画像与角色选择

- **task_family**: locomotion_continuous_control
- **dynamics_subtype**: planar_bipedal_gait
- **selected reward roles**:
  1. **forward_progress** (mandatory) - 主学习信号
  2. **balance_maintenance** (mandatory) - 健康约束
  3. **energy_efficiency** (mandatory) - 效率约束

## role_to_signal_mapping

| role_id | usable signals | selected formula operator |
|---|---|---|
| forward_progress | horizontal_velocity (obs[2]) | dense_state_signal (线性正奖励) |
| balance_maintenance | hull_angle (obs[0]), hull_angular_velocity (obs[1]) | soft_health_gate + dense_state_signal (hinge) |
| energy_efficiency | action[0..3] | quadratic_penalty |

## 每个 role 的 formula operator 选择理由

### 1. forward_progress → dense_state_signal (线性正奖励)
- 水平速度是直接的任务目标信号，每步都有梯度
- 使用线性形式而非凸化形式，因为 v1 阶段 agent 需要先学会"前进"这个基本行为，凸化可能让 agent 在低速时梯度太小
- 权重 2.0 确保在平衡门控衰减后仍有足够信号

### 2. balance_maintenance → soft_health_gate + hinge penalty
- **soft_health_gate**: 这是核心设计选择。根据 expert_reward_context.md 的触发证据，高 terminated 率（摔倒）且前进奖励在失败回合仍为正时，应该使用 gate 而非独立惩罚。门控因子在身体角度偏离或角速度增大时平滑衰减前进奖励，让 agent 学会"在保持平衡的前提下前进"
- **hinge penalty**: 作为安全网，当身体角度超过 0.8 弧度（约46度）时给予额外惩罚。使用 hinge 形式确保在安全范围内不干扰正常动作
- 两个信号（角度和角速度）组合成 gate 的最小值，确保最差情况主导

### 3. energy_efficiency → quadratic_penalty
- 使用动作扭矩平方和作为能耗代理
- 权重 0.01 较小，避免压制前进动力
- v1 阶段能耗优化是次要目标，主要让 agent 先学会前进和平衡

## excluded_roles 及原因

| excluded role | reason |
|---|---|
| explicit_success_reward | explicit_success_flag_available=false，无法区分成功/失败 |
| distance_traveled | 无绝对位置观测，无法计算前进距离 |
| gait_regularity | 条件职责，v1 阶段不引入，避免过早约束步态模式 |
| terrain_adaptation | 条件职责，需要复杂 LIDAR 理解，v1 阶段不引入 |

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- explicit_success_flag_available=false，terminated 无法区分成功（到达终点）和失败（摔倒）
- 使用 terminal 信号会导致 agent 无法区分"应该继续前进"和"应该避免摔倒"
- 所有引导都通过稠密信号实现，避免稀疏 terminal 信号带来的学习困难

## 留到后续迭代的职责

1. **gait_regularity**: 当观察到步态不对称或效率低下时引入，使用关节角度和接触信号
2. **terrain_adaptation**: 当 LIDAR 显示地形变化剧烈时引入，使用 LIDAR 信号
3. **动作平滑惩罚**: 如果出现动作抖动，使用 action difference penalty
4. **curriculum_weighting**: 如果早期探索和后期精细控制冲突明显

## 训练后应观察的 failure modes

| failure_mode | evidence | possible intervention |
|---|---|---|
| 过早摔倒 | 训练早期 terminated 频繁，horizontal_velocity 低 | 增加 balance_gate 的衰减系数，或降低 forward_reward_weight |
| 原地踏步 | horizontal_velocity 接近 0，但未摔倒 | 增加 forward_reward_weight，或检查 balance_gate 是否过严 |
| 能耗过高 | action 值接近边界，关节速度大 | 增加 energy_penalty_weight，或引入动作平滑惩罚 |
| 步态不对称 | 一条腿接触时间远多于另一条 | 引入 gait_regularity 职责 |
| 地形适应不良 | 在 LIDAR 显示地形变化时摔倒 | 引入 terrain_adaptation 职责 |