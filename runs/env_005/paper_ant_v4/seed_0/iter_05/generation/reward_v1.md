# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 提取观测 ---
    body_z = obs[0]
    # 四元数分量
    quat_w = obs[1]
    quat_x = obs[2]
    quat_y = obs[3]
    quat_z = obs[4]
    # 速度
    body_x_vel = obs[13]   # 世界前向速度
    body_y_vel = obs[14]   # 侧向速度
    # 角速度
    roll_vel  = obs[16]
    pitch_vel = obs[17]
    yaw_vel   = obs[18]

    # --- 健康门控因子 ---
    # 1. 高度健康：当 body_z >= 0.5 时视为完全健康，0.5 -> 0.2 线性衰减至 0
    if body_z >= 0.5:
        health_gate = 1.0
    elif body_z <= 0.2:
        health_gate = 0.0
    else:
        health_gate = (body_z - 0.2) / 0.3   # [0.2,0.5] 线性

    # 2. 直立健康：使用 body_up_z = 1 - 2*(quat_x^2 + quat_y^2)
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    # body_up_z 通常在 1(完全直立) 到 -1(倒立) 之间，要求在 0.5 以上才有正向门控
    if body_up_z >= 0.5:
        upright_gate = 1.0
    else:
        # 0.5 以下线性衰减，0 时门控为 0
        upright_gate = max(0.0, body_up_z / 0.5)

    # 组合门控
    combined_gate = health_gate * upright_gate

    # --- 奖励组件 ---
    # 主前向速度奖励，由门控调节
    w_fwd = 1000.0
    forward_reward = w_fwd * body_x_vel * combined_gate

    # 侧向漂移惩罚（平方惩罚柔和，避免绝对值死区）
    w_lat = 2.0
    lat_penalty = w_lat * body_y_vel**2

    # 角速度惩罚（惩罚剧烈的身体旋转）
    w_ang = 0.5
    ang_penalty = w_ang * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)

    # 总奖励
    total_reward = forward_reward - lat_penalty - ang_penalty

    # 组件记录
    components = {
        'forward_reward': forward_reward,
        'lat_penalty': -lat_penalty,
        'ang_penalty': -ang_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像与动态类型
- **任务族**：locomotion_continuous_control（连续运动控制）
- **动力学子类型**：四足机器人 3D 前向行走/奔跑，需保持身体直立与高度健康，避免摔倒终止。
- **目标分数**：2000.0，当前最高原始分数仅 26.07，说明现有设计未能赋予策略足够的前进驱动力。

## 2. 本次选定的奖励角色（reward roles）
基于环境卡片中的职责分解，v1 聚焦于：
- **主学习信号**：`forward_progress` —— 通过 body_x_velocity（obs[13]）提供持续的稠密梯度。
- **稳定/安全约束（0-2个）**：
  - `lateral_stability` —— 惩罚侧向线速度，抑制漂移。
  - `angular_stability` —— 惩罚身体角速度，抑制翻滚/旋转。
- **软健康门控（soft health gate）**：将身体高度（body_z）和直立程度（body_up_z）作为主奖励的乘积门控，而非直接惩罚项。这避免了之前版本中 height_penalty 和 upright_penalty 可能压制探索的问题。

## 3. 职责-信号映射与公式算子
| 职责                 | 信号                   | 算子形式                                                                 |
|----------------------|------------------------|--------------------------------------------------------------------------|
| forward_progress     | obs[13] body_x_velocity | `w_fwd * signal * health_gate * upright_gate`（软门控连续增益）         |
| lateral_stability    | obs[14] body_y_velocity | quadratic_penalty：`-w_lat * signal^2`                                   |
| angular_stability    | obs[16:19] 角速度       | quadratic_penalty：`-w_ang * (roll^2 + pitch^2 + yaw^2)`                |
| height survival      | obs[0] body_z           | soft_health_gate：线性 ramp [0.2,0.5] → [0,1]                           |
| upright posture      | 四元数 obs[1:5]         | soft_health_gate：通过 body_up_z 线性 ramp [0,0.5] → [0,1]               |

- 主奖励使用 **乘法门控**：`forward_reward = w_fwd * body_x_vel * combined_gate`。这样确保只有机器人在健康、直立的状态下前进才能获得大量奖励，摔倒或严重倾斜时奖励几乎为零。相比直接添加高度/直立惩罚项，门控方式更平滑且不带走梯度。
- 侧向和角速度惩罚采用 **quadratic_penalty**，温和抑制漂移和混乱旋转。

## 4. 排除的角色及其原因
- **terminal_success_reward**：环境无显式成功标志 (`explicit_success_flag_available=false`)。
- **terminal_failure_penalty**：虽存在高度失败终止，但 info 不可用，无法在奖励函数内感知失败事件。
- **height_penalty / upright_penalty（直接惩罚项）**：已用更平滑的 soft health gate 替代，避免过度惩罚导致 agent 不敢行动。
- **action_efficiency / action_smoothness**：v1 阶段优先使 agent 学会前进，后续再考虑节能。
- **curriculum_weighting / dynamic_curriculum**：v1 无训练进度依赖，不做阶段课程。

## 5. 与过往尝试的结构式差异
此前多次尝试的结构均为 `angular_penalty + height_penalty + lateral_penalty + upright_penalty + velocity_reward`，其中 velocity_reward 已包含基于高度的门控，但仍保留了 height_penalty 和 upright_penalty 作为独立惩罚项。本次设计**彻底移除直接的高度和直立惩罚**，将它们完全吸收为前向速度奖励的乘法门控，从而从根本上改变了奖励的时间语义：不良姿态不再削减总奖励（通过负项），而是使主进展奖励失效，形成一种更自然的“无进展即惩罚”机制，而非显式扣分。

## 6. 权重选择与尺度平衡
- `w_fwd = 1000`：预期最大全身速度约 1 m/s，故前向奖励贡献可达 1000 量级，结合数百步的 episode，总回报有望接近 2000 目标。
- `w_lat = 2`、`w_ang = 0.5`：使侧向/角速度惩罚在典型漂移（0.2 m/s 侧向、0.5 rad/s 角速度）时贡献约 -0.1 ~ -0.5，远小于前向奖励，仅作引导。
- 无界的前向速度经由乘法门控约束，避免刷分。

## 7. 训练后应关注的典型失败模式
- 机器人可能通过剧烈侧向摆动或旋转获得暂时高速度（因为门控允许一些倾斜），但可能摔倒。若摔倒，episode 终止，自然亏损。
- 若 `w_fwd` 过大，可能导致策略优先选择短期高速冲刺而后摔倒，此时应增加角速度或侧向惩罚权重。
- 若机器人始终缓慢爬行，可能需要降低健康门控的阈值（如将 height gate 上限从 0.5 降到 0.4）以降低直立难度，或尝试提升 `w_fwd`。
- 后续迭代可考虑引入前进距离增量（如使用动作与关节速度的相关性）或步态周期性，但目前 v1 先构建稳定前向驱动力。