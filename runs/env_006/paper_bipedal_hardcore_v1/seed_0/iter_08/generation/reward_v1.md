# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # =========================================================================
    # 1. 提取观测信号
    # =========================================================================
    # obs 索引 (共24维)
    hull_angle = obs[0]           # 躯干倾角 (rad)
    hull_ang_vel = obs[1]         # 躯干角速度 (rad/s)
    horizontal_speed = obs[2]     # 水平速度 (m/s)
    vertical_speed = obs[3]       # 垂直速度 (m/s)
    # obs[4..11]: 关节角度和角速度，暂不直接使用
    leg_1_contact = obs[12]       # 腿1触地 (0/1)
    leg_2_contact = obs[13]       # 腿2触地 (0/1)
    # obs[14..23]: lidar, 不可用于奖励

    # next_obs 索引
    next_hull_angle = next_obs[0]
    next_hull_ang_vel = next_obs[1]
    next_horizontal_speed = next_obs[2]
    next_vertical_speed = next_obs[3]
    next_leg_1_contact = next_obs[12]
    next_leg_2_contact = next_obs[13]

    # =========================================================================
    # 2. 前向速度奖励 (主学习信号)
    #    role: forward_velocity_reward
    #    formula: dense_state_signal (线性), 直接鼓励保持正水平速度
    #    scale: 1.0 使每步贡献约 0.8~2.0
    # =========================================================================
    forward_speed = max(0.0, horizontal_speed)  # 负向速度不奖励
    forward_reward = 1.0 * forward_speed

    # =========================================================================
    # 3. 姿态稳定门 (soft_health_gate)
    #    role: upright_penalty 变形为 gate, 替代独立惩罚
    #    rationale: 尝试过的独立 tilt penalty 效果不佳 (score -49~-62).
    #               采用 soft gate 在姿态恶化时直接衰减 forward_reward,
    #               避免 agent 在倾斜时仍因高速获得大量奖励。
    #    formula: soft_health_gate (线性衰减)
    #    gate = 1.0 当 hull_angle 在安全范围内,
    #           线性衰减至 0.0 当 hull_angle 接近危险阈值
    # =========================================================================
    # 设定安全区和衰退区间
    tilt_safe_bound = 0.3          # rad, 近似17°, 正常行走摆动范围
    tilt_danger_bound = 0.7        # rad, 近似40°, 接近摔倒临界 (经验阈值 ~0.8)
    tilt_margin = tilt_danger_bound - tilt_safe_bound  # 0.4 rad 衰退区间

    abs_tilt = abs(hull_angle)
    if abs_tilt <= tilt_safe_bound:
        tilt_gate = 1.0
    elif abs_tilt >= tilt_danger_bound:
        tilt_gate = 0.0
    else:
        tilt_gate = 1.0 - (abs_tilt - tilt_safe_bound) / tilt_margin

    # 角速度惩罚: 当躯干快速旋转时进一步收紧 gate, 捕捉突然失去平衡的前兆
    # 在 tilt_gate 基础上再乘一个角速度衰减因子
    ang_vel_thresh = 2.0           # rad/s, 正常步态摆动通常 < 1.5
    ang_vel_margin = 4.0           # 2.0~6.0 rad/s 区间衰减
    abs_ang_vel = abs(hull_ang_vel)
    if abs_ang_vel <= ang_vel_thresh:
        ang_vel_factor = 1.0
    elif abs_ang_vel >= ang_vel_thresh + ang_vel_margin:
        ang_vel_factor = 0.3       # 不归零, 保留微弱梯度以防止完全丧失学习信号
    else:
        ang_vel_factor = 1.0 - 0.7 * (abs_ang_vel - ang_vel_thresh) / ang_vel_margin

    stability_gate = tilt_gate * ang_vel_factor

    # =========================================================================
    # 4. 能量效率惩罚 (轻量)
    #    role: energy_penalty
    #    formula: action_efficiency (L2 范数)
    #    scale: 极小权重, 仅在主任务已驱动后提供效率偏好
    # =========================================================================
    action_sq_sum = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = 0.005 * action_sq_sum  # 步贡献约 -0.005~-0.02

    # =========================================================================
    # 5. 组合并返回
    # =========================================================================
    # 核心思路: 主奖励通过稳定性 gate 衰减, 再减去轻量能耗
    gated_forward = forward_reward * stability_gate
    total_reward = gated_forward - energy_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_gate": stability_gate,
        "gated_forward": gated_forward,
        "energy_penalty": -energy_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 设计假设与历史对比

**关键假设**：之前尝试的主流结构是"门控速度 + 独立姿态惩罚"。这些尝试（gated_forward_speed + stability_tilt_hinge_penalty 等）best_score 为 -49.540 到 -57.470，未能有效学习。

失败的可能原因：
- 独立 penalty 在 tilt 越过阈值后才施加，缺少"恶化过程"中的衰减梯度，导致 agent 在越界后直接进入惩罚域而无缓冲。
- Hinge 惩罚在阈值附近梯度突变（0 → -w），可能破坏价值函数平滑性。
- Gate 已经衰减了 forward_reward，再加独立惩罚可能使"倾斜状态"的净奖励过负，agent 不敢尝试动态步态。

**本设计差异**：
1. **仅用 soft_health_gate 约束姿态**，不加任何独立 tilt penalty。Gate 本质是"当你倾斜时，我不奖励你的速度"，而非"我要惩罚你"。这在早期保留了探索动力，同时防止 agent 在摔倒前通过高速刷分。
2. Gate 形状连续可微（分段线性），在安全区→危险区的整个区间内提供梯度。
3. 引入角速度衰减因子：捕捉突然失衡的前兆（如脚底打滑导致躯干猛转）。双因子乘法结构使 gate 对"倾斜 + 急转"的组合状态格外敏感。

## selected roles 与信号映射

| role | signal | formula operator | rationale |
|---|---|---|---|
| forward_velocity_reward | `horizontal_speed` (obs[2]) | dense_state_signal (linear) | 直接代理前进距离，步数稠密 |
| upright_penalty → stability_gate | `hull_angle` (obs[0]), `hull_ang_vel` (obs[1]) | soft_health_gate (双因子线性衰减) | 姿态恶化时衰减主奖励，而非加惩罚 |
| energy_penalty | `action` (4维) | action_efficiency (L2) | 极小权重，提供效率偏好但不压制探索 |

## excluded roles 及原因

| role_id | 排除原因 |
|---|---|
| foot_contact_pattern_reward | 需要与前进状态耦合，设计不当易奖励原地踏步；当前阶段主目标是建立稳定前进步态，接触模式奖励会引入额外噪声 |
| alive_bonus | 在速度奖励已有较稠密信号时，存活奖励可能引发刷步数（站立不动）问题 |
| lidar_based_terrain_reward | 明确禁止：lidar 是感知信号，不应作为奖励 |
| goal_reaching_reward | 缺少显式位置/距离信号，无法构造可靠到达检测 |
| smooth_gait_reward | 当前阶段关节平滑度不是瓶颈，过早加入可能抑制必要的高扭矩探索 |

## 为什么没有 terminal_success_reward / terminal_failure_penalty

- `explicit_success_flag_available=false`，`explicit_failure_flag_available=false`。
- info 为空，无法读取终止原因。
- 虽然可以从观测推断摔倒（hull_angle > 0.8, 接触消失等），但在 v1 版本中，连续的 stability_gate 已提供了摔倒前兆的稠密梯度信号，硬覆盖的 terminal_event 反而不连续。若后续观察到摔倒率仍高且 gate 无法拦截，再考虑加入基于 next_obs 推断的 terminal_penalty。

## 留在后续迭代的职责

- terminal_failure_penalty（如果摔倒率仍高）
- foot_contact_pattern_reward（步态质量，用于优化步行姿态和步态对称性）
- 动态调整 gate 阈值（curriculum scheduling）
- 基于垂直速度/弹跳检测的额外惩罚

## 训练后应观察的 failure modes

| failure mode | 监测指标 | 可能干预 |
|---|---|---|
| 频繁摔倒 | episode_length_mean < 100, 大量回合 hull_angle 超过 0.8 | 收紧 gate 衰退区间（降低 tilt_danger_bound），或引入 terminal_penalty |
| 缓慢挪动、能耗高 | horizontal_speed 均值 < 0.5, energy_penalty 不降 | 提高 forward_reward 系数，或降低 energy_penalty 权重 |
| 原地踏步 | horizontal_speed 极低但 episode 长 | 引入 foot_contact 相关步态奖励，或对 zero-velocity 状态施加负奖励 |
| 弹跳前进 | vertical_speed 波动大 | 引入 vertical_activity 惩罚，或约束 hull 高度变化 |
| 步态僵硬但得分稳定 | 关节 torque 方差低，步态周期长 | 后续迭代加入动作平滑惩罚或步态多样性奖励 |