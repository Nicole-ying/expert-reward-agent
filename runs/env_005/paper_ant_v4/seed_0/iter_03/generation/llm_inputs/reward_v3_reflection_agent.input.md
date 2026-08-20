# Search objective
- target_score: 2000.000000
- current_score: 13.818725
- gap_to_target: 1986.181275
- target_achievement_ratio: 0.691%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 13.818725）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== Core Forward Velocity Reward (mandatory) ==========
    forward_velocity = next_obs[13]
    velocity_reward = 1.0 * forward_velocity

    # ========== Height Safety Penalty (mandatory) ==========
    body_height = next_obs[0]
    height_error = body_height - 0.6
    height_penalty = 10.0 * (height_error ** 2)   # 系数从 0.2 提升到 10.0

    # ========== Upright Pose Penalty (mandatory) ==========
    quat_w = next_obs[1]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]
    body_up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)
    upright_error = 1.0 - body_up_z
    upright_penalty = 0.15 * (upright_error ** 2)

    # ========== Lateral Drift Penalty (conditional) ==========
    lateral_velocity = abs(next_obs[14])
    lateral_penalty = 0.1 * (lateral_velocity ** 2)

    # ========== Angular Velocity Penalty (conditional) ==========
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    angular_penalty = 0.05 * (roll_vel ** 2 + pitch_vel ** 2)

    # ========== Soft Health Gate for Extreme Heights ==========
    health_factor = 1.0
    if body_height < 0.35:
        health_factor = max(0.0, (body_height - 0.2) / 0.15)
    elif body_height > 0.85:
        health_factor = max(0.0, (1.0 - body_height) / 0.15)

    velocity_reward = velocity_reward * health_factor

    # ========== Total Reward Assembly ==========
    total_reward = (
        velocity_reward
        - height_penalty
        - upright_penalty
        - lateral_penalty
        - angular_penalty
    )

    # ========== Component Dictionary ==========
    components = {
        'velocity_reward': velocity_reward,
        'height_penalty': height_penalty,
        'upright_penalty': upright_penalty,
        'lateral_penalty': lateral_penalty,
        'angular_penalty': angular_penalty
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=13.818725, len=29.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-39.863111, 35.704146]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_reward | 18.785118 | 34.5% | 52.8% | 96.6% |
| angular_penalty | 11.156708 | 20.5% | 20.5% | 100.0% |
| height_penalty | 9.893848 | 18.2% | 18.2% | 100.0% |
| lateral_penalty | 4.601561 | 8.4% | 8.4% | 100.0% |
| upright_penalty | 0.102423 | 0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标  
本环境要求控制一个 3D 四足机器人，使用 8 个连续扭矩关节产生稳定的前向步态（行走或奔跑）。机器人的身体高度必须维持在健康范围（0.2 ~ 1.0）内，超出范围立即触发 episode 终止（摔倒或弹跳过高）。核心任务是在保持直立和高度约束的前提下，持续获得尽可能大的前向线速度。次要目标包括抑制横向漂移、减小能量消耗与动作振荡。本任务不应被混淆为纯平衡存活任务——即使机器人站着不动，只要高度合规就不终止，但仅站立无法获得有效的前进奖励。

## 3. 观察空间 observation_space  
- type: Box  
- shape: (27,)  
- dtype: float32（默认）  
- 详细字段（索引 0–26）：  

### 3.1 位置部分（13 维）  
| 索引 | 名称 | 含义 | 可用于奖励 | 备注 |
|------|------|------|-------------|------|
| 0 | `body_z` | 身体（基座）在世界坐标系中的高度 | 是 | 直接用于高度健康约束 |
| 1 | `quat_w` | 姿态四元数实部 w | 是 | 用于计算直立程度 |
| 2 | `quat_x` | 姿态四元数 x 分量 | 是 | 同上 |
| 3 | `quat_y` | 姿态四元数 y 分量 | 是 | 同上 |
| 4 | `quat_z` | 姿态四元数 z 分量 | 是 | 同上 |
| 5 | `joint_1_angle` | 第 1 髋关节角度 | 可选 | 姿态/步态相关 |
| 6 | `joint_2_angle` | 第 1 踝关节角度 | 可选 | 同上 |
| 7 | `joint_3_angle` | 第 2 髋关节角度 | 可选 | 同上 |
| 8 | `joint_4_angle` | 第 2 踝关节角度 | 可选 | 同上 |
| 9 | `joint_5_angle` | 第 3 髋关节角度 | 可选 | 同上 |
| 10 | `joint_6_angle` | 第 3 踝关节角度 | 可选 | 同上 |
| 11 | `joint_7_angle` | 第 4 髋关节角度 | 可选 | 同上 |
| 12 | `joint_8_angle` | 第 4 踝关节角度 | 可选 | 同上 |

### 3.2 速度部分（14 维）  
| 索引 | 名称 | 含义 | 可用于奖励 | 备注 |
|------|------|------|-------------|------|
| 13 | `body_x_velocity` | 世界坐标系 x 方向线速度（前进方向） | **是（核心）** | 主要任务信号 |
| 14 | `body_y_velocity` | 世界坐标系 y 方向线速度（横向） | 是 | 抑制横向漂移 |
| 15 | `body_z_velocity` | 身体垂直速度 | 是 | 防止剧烈跳跃 |
| 16 | `body_roll_velocity` | 滚转角速度 | 是 | 惩罚翻滚 |
| 17 | `body_pitch_velocity` | 俯仰角速度 | 是 | 惩罚栽倒 |
| 18 | `body_yaw_velocity` | 偏航角速度 | 可选 | 若需保持航向 |
| 19 | `joint_1_velocity` | 第 1 髋关节角速度 | 可选 | 平滑性/能耗 |
| 20 | `joint_2_velocity` | 第 1 踝关节角速度 | 可选 | 同上 |
| 21 | `joint_3_velocity` | 第 2 髋关节角速度 | 可选 | 同上 |
| 22 | `joint_4_velocity` | 第 2 踝关节角速度 | 可选 | 同上 |
| 23 | `joint_5_velocity` | 第 3 髋关节角速度 | 可选 | 同上 |
| 24 | `joint_6_velocity` | 第 3 踝关节角速度 | 可选 | 同上 |
| 25 | `joint_7_velocity` | 第 4 髋关节角速度 | 可选 | 同上 |
| 26 | `joint_8_velocity` | 第 4 踝关节角速度 | 可选 | 同上 |

- 所有字段均可直接用于奖励，但具体是否采纳取决于职责设计。

## 4. 动作空间 action_space  
- type: Box  
- shape: (8,)  
- bounds: [-1.0, 1.0] 每关节  
- continuous: true  
- 每个动作维度的含义（与关节对应）：  

| 索引 | 名称 | 含义 |
|------|------|------|
| 0 | `hip_1_torque` | 第一个髋关节扭矩指令 |
| 1 | `ankle_1_torque` | 第一个踝关节扭矩指令 |
| 2 | `hip_2_torque` | 第二个髋关节扭矩指令 |
| 3 | `ankle_2_torque` | 第二个踝关节扭矩指令 |
| 4 | `hip_3_torque` | 第三个髋关节扭矩指令 |
| 5 | `ankle_3_torque` | 第三个踝关节扭矩指令 |
| 6 | `hip_4_torque` | 第四个髋关节扭矩指令 |
| 7 | `ankle_4_torque` | 第四个踝关节扭矩指令 |

- 动作是直接施加在关节上的力矩，可用于构建动作平滑或能耗惩罚。

## 5. step 与终止条件分析  
### 5.1 终止模式  
- **success-like termination**：无显式成功判断。环境本身不给出“到达目标/完成任务”的终止，只有时间截断（truncation）。从任务语义看，若能保持高度、存活到 episode 结束并获得持续前进速度，可视为成功的表现形式。  
- **failure-like termination**：
  - `body_height_outside_healthy_range`：身体高度 ≤ 0.2（摔倒）或 ≥ 1.0（猛烈弹跳），属于明确的失败。
  - `state_value_outside_finite_range`：任意状态变为 NaN 或无穷大，属异常/失败。
- **ambiguous termination**：无。
- **truncation**：时间限制到达，episode 自然结束。对于该任务，本身不算失败；多数成功训练下，时间截断即代表存活且前进了一段距离。

### 5.2 success/failure 信号可用性  
- explicit_success_flag_available: **false**  
- explicit_failure_flag_available: **true**（通过 terminated 标志可推断为失败，但需自行判断具体原因）  
- allowed_info_fields: **[]**（为空列表）  
- forbidden_or_uncertain_info_fields:  
  - `reward_forward`、`reward_ctrl`、`reward_contact`、`reward_survive`、`x_position`、`y_position`、`distance_from_origin` 等字段明确禁止使用。  
  - 不允许使用任何未在 `allowed_info_fields` 中声明的 info 字段。

## 7. 可用于奖励函数的信号  
- **前进速度**：`next_obs[13]`（body_x_velocity），直接体现任务核心目标。  
- **高度保持**：`next_obs[0]`（body_z）处于 (0.2, 1.0) 内。  
- **直立姿态**：通过 `next_obs[1:5]`（四元数）计算身体 z 轴与世界 z 轴投影 `body_up_z = 1 - 2*(quat_x² + quat_y²)`，接近 1 表示完全竖直。  
- **横向/垂直速度抑制**：`next_obs[14]`（body_y_velocity）、`next_obs[15]`（body_z_velocity）。  
- **角速度惩罚**：`next_obs[16:18]`（roll, pitch 角速度）。  
- **动作平滑/能耗**：`action` 本身（扭矩绝对值或平方）、`obs[19:27]` 关节速度（可用于关节运动惩罚）。  
- **步态发育辅助信号**：可选用关节位置 `next_obs[5:13]` 或速度 `next_obs[19:27]`，避免僵直或过度振荡。

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angular_penalty + height_penalty + lateral_penalty + upright_penalty + velocity_reward | -27.22 | -27.22 | 0.00 | 409.80 | angular_penalty=0.258 height_penalty=0.013 lateral_penalty=0.055 upright_penalty=0.265 velocity_reward=0.205 | new_best |
| 2 | angular_penalty + height_penalty + lateral_penalty + upright_penalty + velocity_reward | 13.82 | 13.82 | 0.00 | 29.25 | angular_penalty=0.316 height_penalty=0.623 lateral_penalty=0.065 upright_penalty=0.245 velocity_reward=0.171 | new_best |
