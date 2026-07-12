# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. forward_progress: 主学习信号
    #    使用 horizontal_velocity (obs[2]) 作为前进激励
    #    采用凸化形式 (signal**2) 打破低速平台，鼓励更快行走
    # ============================================================
    horizontal_vel = obs[2]
    forward_progress_reward = 1.0 * (horizontal_vel ** 2)

    # ============================================================
    # 2. balance_maintenance: 稳定/安全约束
    #    使用 hull_angle (obs[0]) 和 hull_angular_velocity (obs[1])
    #    采用 quadratic_penalty 形式，对倾斜角度和角速度施加轻量惩罚
    #    权重较小，避免压制探索
    # ============================================================
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    balance_penalty = -0.5 * (hull_angle ** 2) - 0.1 * (hull_angular_vel ** 2)

    # ============================================================
    # 3. gait_cycle_encouragement: 步态周期鼓励
    #    使用 leg1_contact (obs[8]) 和 leg2_contact (obs[13])
    #    鼓励双腿交替接触地面，惩罚双腿同时离地
    #    采用 binary_state_signal 组合形式
    # ============================================================
    leg1_contact = obs[8]
    leg2_contact = obs[13]
    # 鼓励至少一条腿接触地面（避免跳跃/蹦跳）
    contact_sum = leg1_contact + leg2_contact
    # 当双腿同时离地时给予惩罚
    both_off_ground_penalty = -0.2 * (1.0 - contact_sum) if contact_sum < 0.5 else 0.0
    # 鼓励交替接触（当一条腿接触时给予小奖励）
    alternating_bonus = 0.1 * (leg1_contact * (1.0 - leg2_contact) + leg2_contact * (1.0 - leg1_contact))

    # ============================================================
    # 组合总奖励
    # ============================================================
    total_reward = forward_progress_reward + balance_penalty + both_off_ground_penalty + alternating_bonus

    # ============================================================
    # 组件记录
    # ============================================================
    components = {
        "forward_progress_reward": forward_progress_reward,
        "balance_penalty": balance_penalty,
        "both_off_ground_penalty": both_off_ground_penalty,
        "alternating_bonus": alternating_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 任务画像与角色选择

- **task_family**: locomotion_continuous_control
- **dynamics_subtype**: planar_bipedal_gait
- **selected reward roles**: forward_progress (mandatory), balance_maintenance (mandatory), gait_cycle_encouragement (mandatory)

## role_to_signal_mapping

| role_id | usable signals | formula operator | 具体实现 |
|---|---|---|---|
| forward_progress | horizontal_velocity (obs[2]) | dense_state_signal (凸化) | `1.0 * vel^2`，凸化形式打破低速平台 |
| balance_maintenance | hull_angle (obs[0]), hull_angular_velocity (obs[1]) | quadratic_penalty | `-0.5 * angle^2 - 0.1 * angular_vel^2`，轻量约束 |
| gait_cycle_encouragement | leg1_contact (obs[8]), leg2_contact (obs[13]) | binary_state_signal + hinge | 惩罚双腿同时离地，奖励交替接触 |

## excluded_roles 及原因

- **energy_efficiency**: 条件职责，v1 阶段优先建立基础行走能力，过早加入能耗惩罚可能导致 agent 不敢用力，无法前进。留到后续迭代。
- **terrain_adaptation**: 条件职责，LIDAR 信号维度高（10维），需要合理降维或特征提取，v1 暂不加入。留到后续迭代。
- **smooth_motion**: 条件职责，v1 阶段动作抖动问题不突出，且可能限制必要的快速调整。留到后续迭代。
- **explicit_success_reward**: 环境无 success flag（info 始终为空），信号缺失，禁用。
- **distance_to_goal**: 无绝对位置观测，信号缺失，禁用。
- **foot_clearance**: 无足部高度或位置观测，信号缺失，禁用。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- **explicit_success_flag_available**: false，info 字典始终为空，无法区分成功终止和失败终止。
- **explicit_failure_flag_available**: false，info 字典始终为空，无法获取 failure 标志。
- 因此无法使用 terminal 奖励，所有奖励必须基于每步观测信号。

## 留到后续迭代的职责

1. **energy_efficiency**: 当基础行走能力建立后，加入动作扭矩的二次惩罚，优化能耗。
2. **terrain_adaptation**: 利用 LIDAR 观测提取地形特征，调整步态以适应不规则地形。
3. **smooth_motion**: 当动作抖动明显时，加入动作变化率或关节加速度惩罚。
4. **soft_health_gate**: 如果训练中出现"先冲后死"模式（高 terminated 率但前进奖励仍为正），可将 balance_maintenance 改为 gate 乘到 forward_progress 上。

## 训练后应观察的 failure modes

| failure_mode | 观察证据 | 可能干预 |
|---|---|---|
| 原地踏步/不前进 | horizontal_velocity 均值接近 0，但 leg_contact 交替正常 | 增加 forward_progress 权重，或改用线性形式 |
| 向前倾倒行走 | hull_angle 持续偏向一侧，horizontal_velocity 高但角速度大 | 增加 balance_penalty 权重，或加入 hull_angular_velocity 惩罚 |
| 单腿跳跃/蹦跳 | 双腿同时离地时间长，both_off_ground_penalty 频繁触发 | 增加 both_off_ground_penalty 权重 |
| 动作过大/抖动 | action 值频繁接近 ±1 | 后续迭代加入 energy_efficiency 或 smooth_motion |
| 过早摔倒 | 回合长度短，hull_angle 快速增大 | 检查 balance_penalty 是否足够，或加入 soft_health_gate |