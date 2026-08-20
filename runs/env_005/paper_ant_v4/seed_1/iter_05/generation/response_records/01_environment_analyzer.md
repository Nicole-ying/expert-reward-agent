# Response Record

# 匿名环境理解卡片

## 1. 任务目标
训练一个 3D 四足机器人，通过扭矩控制 8 个关节实现稳定、快速的前向行走或奔跑，并始终保持身体高度在健康范围内 (0.2 ~ 1.0)。**核心目标**是持续前向运动，而不是单纯站立不倒；次要目标包括保持直立姿态、抑制侧向漂移、降低能耗并避免翻倒或崩溃。

## 2. 任务类型选择
**selected_route_id**: `locomotion_continuous_control`  
**confidence**: high  
**reason**: 任务核心是让四足机器人沿着一个方向（默认世界 x 轴）持续前进，附属有高度稳定、姿态保持等约束，符合“持续前进通过地形”的定义，不涉及目标点到达，也不仅仅为了平衡存活。

## 3. 观察空间 observation_space
- **type**: `Box`  
- **shape**: `[27]`  
- **dtype**: 未显式声明，通常为 `float32`（连续动力学环境）  
- 各维度含义及可用于奖励的说明：

| 索引 (index) | 名称 | 含义 | reward_usable |
|---|---|---|---|
| 0 | body_z | 身体中心垂直高度 | true（高度健康约束） |
| 1 | quat_w | 身体朝向四元数 w | true（用于计算 body_up_z） |
| 2 | quat_x | 身体朝向四元数 x | true |
| 3 | quat_y | 身体朝向四元数 y | true |
| 4 | quat_z | 身体朝向四元数 z | true |
| 5 | joint_1_angle | 第 1 髋关节角度 | true（关节姿态平滑/约束） |
| 6 | joint_2_angle | 第 1 踝关节角度 | true |
| 7 | joint_3_angle | 第 2 髋关节角度 | true |
| 8 | joint_4_angle | 第 2 踝关节角度 | true |
| 9 | joint_5_angle | 第 3 髋关节角度 | true |
| 10 | joint_6_angle | 第 3 踝关节角度 | true |
| 11 | joint_7_angle | 第 4 髋关节角度 | true |
| 12 | joint_8_angle | 第 4 踝关节角度 | true |
| 13 | body_x_velocity | 身体在世界 x 方向的前向速度 | true（前向速度奖励） |
| 14 | body_y_velocity | 身体在世界 y 方向的侧向速度 | true（侧滑惩罚） |
| 15 | body_z_velocity | 身体垂直速度 | true（跳跃/振颤抑制） |
| 16 | body_roll_velocity | 身体滚转角速度 | true（稳定性惩罚） |
| 17 | body_pitch_velocity | 身体俯仰角速度 | true |
| 18 | body_yaw_velocity | 身体偏航角速度 | true |
| 19 | joint_1_velocity | 第 1 髋关节角速度 | true |
| 20 | joint_2_velocity | 第 1 踝关节角速度 | true |
| 21 | joint_3_velocity | 第 2 髋关节角速度 | true |
| 22 | joint_4_velocity | 第 2 踝关节角速度 | true |
| 23 | joint_5_velocity | 第 3 髋关节角速度 | true |
| 24 | joint_6_velocity | 第 3 踝关节角速度 | true |
| 25 | joint_7_velocity | 第 4 髋关节角速度 | true |
| 26 | joint_8_velocity | 第 4 踝关节角速度 | true |

**重要**：环境未提供绝对世界坐标 (`x_position`, `y_position`) 和接触力信息。所有可用于奖励的信号均来自上述 27D 向量，`info` 字典为空且禁止使用任何内部拆解的奖励项。

## 4. 动作空间 action_space
- **type**: `Box` (连续)  
- **shape**: `[8]`  
- **dtype**: 通常为 `float32`，范围 `[-1.0, 1.0]` per joint  
- 每个动作维度对应一个关节的目标扭矩：

| 动作维度 | 名称 | 含义 |
|---|---|---|
| 0 | hip_1_torque | 第 1 髋关节扭矩 |
| 1 | ankle_1_torque | 第 1 踝关节扭矩 |
| 2 | hip_2_torque | 第 2 髋关节扭矩 |
| 3 | ankle_2_torque | 第 2 踝关节扭矩 |
| 4 | hip_3_torque | 第 3 髋关节扭矩 |
| 5 | ankle_3_torque | 第 3 踝关节扭矩 |
| 6 | hip_4_torque | 第 4 髋关节扭矩 |
| 7 | ankle_4_torque | 第 4 踝关节扭矩 |

扭矩范围归一化到 [-1, 1]，每个关节的加速度/力由底层模拟器转换成实际力矩。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 无明确成功终止条件。没有目标到达或存活时长判定。
- **failure-like termination**: 
  - `body_height_outside_healthy_range`: 身体高度 `body_z` ≤ 0.2（崩溃/倒地）或 ≥ 1.0（过度跳跃/弹射）时环境结束，属于失败。
  - `state_value_outside_finite_range`: 任何状态值变为 NaN 或 Inf，属于失败。
- **ambiguous termination**: 无。
- **truncation**: `time_limit_reached` 触发截断，不表示成功或失败。

### 5.2 success/failure 信号可用性
- `explicit_success_flag_available`: **false**  
  观察信息中不存在任何“成功”标志。
- `explicit_failure_flag_available`: **false**  
  无法在奖励函数内部直接获取 `terminated` 或 `truncated`，只能在回合结束后从外部日志推断。任何 `info` 字段均不可用。
- `allowed_info_fields`: **无**（`info = {}`，严禁使用）
- `forbidden_or_uncertain_info_fields`: 所有 `info` 字段均禁止使用，尤其禁止以下已知但被隐藏的字段：
  - `reward_forward`, `reward_ctrl`, `reward_contact`, `reward_survive`
  - `x_position`, `y_position`, `distance_from_origin`

**结论**：奖励函数 **不能** 依赖成功/失败标志来施加终止奖励或分段奖励，所有引导必须来自于观测值（`obs`）和动作（`action`）本身。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

- **允许使用**：
  - `obs`：27D 当前观测
  - `action`：8D 动作
  - `next_obs`：27D 下一步观测
  - `info` 中 **明确允许的字段**（当前无任何允许字段，故 `info` 不可用）
  - `training_progress` **仅在 prompt 明确允许时才可使用**，当前未提及，一概禁止使用。

- **禁止使用**：
  - `original_reward`（官方奖励被遮盖）
  - `official_reward` 或任何从 `info` 中拼凑出的隐含奖励项
  - 未明确声明的 `info` 字段（如 `x_position`, `y_position`）
  - 未声明的 `obs` 切片（下标 0-26 均已在观察空间中声明，不得使用超出范围的索引）
  - 依靠全局状态或跨 Episode 累积量（函数必须是无状态的）

## 7. 可用于奖励函数的信号
- **位置/姿态**：`body_z`；四元数 `quat_w, quat_x, quat_y, quat_z` → 可计算 `body_up_z = 1 - 2*(quat_x² + quat_y²)`；8 个关节角度。
- **速度**：`body_x_velocity`、`body_y_velocity`、`
