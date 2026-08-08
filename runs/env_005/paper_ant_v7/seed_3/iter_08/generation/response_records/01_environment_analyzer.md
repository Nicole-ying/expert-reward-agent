# Response Record

```markdown
# 匿名环境理解卡片

## 1. 任务目标
这是一个 3D 四足机器人的连续控制运动任务，要求机器人通过 8 个转矩控制关节产生稳定的向前行走或奔跑步态。
核心目标是**持续、稳定地向前移动**（通过 body_x_velocity 体现），同时必须满足两个硬性生存约束：
- 身体高度保持在健康区间 `(0.2, 1.0)` 内，低于下限（塌倒）或超过上限（跃起）都会立刻终止；
- 所有状态值必须为有限实数，不允许出现 NaN 或无穷大。
次要/隐含目标是保持身体直立（减少侧倾和翻转）、动作平滑节能、避免关节超限或剧烈抖动。**不要误将其理解为纯平衡任务**：只保持不倒而不前进是不满足任务目标的。

## 2. 任务类型选择
**selected_route_id**: `locomotion_continuous_control`  
**confidence**: high  
**reason**: 核心目标是利用四足身体在连续动作空间中产生长时间的前向运动，前进速度是主要性能指标；保持高度和方向只是生存前提，不是最终目的。这完全符合“持续前进通过地形”类型的描述，没有越障或复杂操控，也没有多个无法区分权重的目标。

## 3. 观察空间 observation_space
- **type**: Box  
- **shape**: [27]  
- **dtype**: 浮点型（通常 float32）  
- 各维度 index 含义及是否可用于奖励函数：

| index | name | meaning | reward_usable |
|---|---|---|---|
| 0 | body_z | 身体（质心）垂直高度 | true |
| 1 | quat_w | 朝向四元数实部 w | true |
| 2 | quat_x | 朝向四元数虚部 x | true |
| 3 | quat_y | 朝向四元数虚部 y | true |
| 4 | quat_z | 朝向四元数虚部 z | true |
| 5 | joint_1_angle | 第一髋关节角度 | true |
| 6 | joint_2_angle | 第一踝关节角度 | true |
| 7 | joint_3_angle | 第二髋关节角度 | true |
| 8 | joint_4_angle | 第二踝关节角度 | true |
| 9 | joint_5_angle | 第三髋关节角度 | true |
| 10 | joint_6_angle | 第三踝关节角度 | true |
| 11 | joint_7_angle | 第四髋关节角度 | true |
| 12 | joint_8_angle | 第四踝关节角度 | true |
| 13 | body_x_velocity | 世界坐标系下身体前向线速度 | true |
| 14 | body_y_velocity | 身体横向线速度 | true |
| 15 | body_z_velocity | 身体垂直线速度 | true |
| 16 | body_roll_velocity | 绕 x 轴（翻滚）角速度 | true |
| 17 | body_pitch_velocity | 绕 y 轴（俯仰）角速度 | true |
| 18 | body_yaw_velocity | 绕 z 轴（偏航）角速度 | true |
| 19 | joint_1_velocity | 第一髋关节角速度 | true |
| 20 | joint_2_velocity | 第一踝关节角速度 | true |
| 21 | joint_3_velocity | 第二髋关节角速度 | true |
| 22 | joint_4_velocity | 第二踝关节角速度 | true |
| 23 | joint_5_velocity | 第三髋关节角速度 | true |
| 24 | joint_6_velocity | 第三踝关节角速度 | true |
| 25 | joint_7_velocity | 第四髋关节角速度 | true |
| 26 | joint_8_velocity | 第四踝关节角速度 | true |

补充：`body_up_z = 1 - 2*(quat_x² + quat_y²)` 可以直接作为直立程度指标（1.0 为完全直立），可安全用于奖励函数。

## 4. 动作空间 action_space
- **type**: Box  
- **shape**: [8]  
- **范围**: 每维 `[-1.0, 1.0]`  
- 各维含义：

| action_dim | name | meaning |
|---|---|---|
| 0 | hip_1_torque | 第一髋关节转矩（归一化或按比例缩放） |
| 1 | ankle_1_torque | 第一踝关节转矩 |
| 2 | hip_2_torque | 第二髋关节转矩 |
| 3 | ankle_2_torque | 第二踝关节转矩 |
| 4 | hip_3_torque | 第三髋关节转矩 |
| 5 | ankle_3_torque | 第三踝关节转矩 |
| 6 | hip_4_torque | 第四髋关节转矩 |
| 7 | ankle_4_torque | 第四踝关节转矩 |

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 无显式成功终止。唯一“正面”的结果是活到时间上限（truncated），但截断不是成功，不能用作奖励信号。
- **failure-like termination**:
  - `body_height_outside_healthy_range`: `body_z` 离开 `(0.2, 1.0)` 区间。这是典型的失败（摔倒或弹起过高）。
  - `state_value_outside_finite_range`: 任何状态出现 NaN 或无限大，通常是动力学崩溃。
- **ambiguous termination**: 无。
- **truncation**: `time_limit_reached`，表示达到最大步数，属于中性截断，不应赋予奖励正或负含义。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
- **explicit_failure_flag_available**: false （但通过终止条件可以推断失败）  
- **allowed_info_fields**: 无（返回的 info 为空字典），即不能使用任何 info 字段。  
- **forbidden_or_uncertain_info_fields**: 以下真实环境字段被明确禁止使用：  
  `reward_forward`, `reward_ctrl`, `reward_contact`, `reward_survive`, `x_position`, `y_position`, `distance_from_origin`。任何未在上述 allowed_info_fields 出现的字段均不可用。

**重要说明**：虽然观察中包含 `body_x_velocity` 可以直接作为前进速度信号，但没有全局位置或距起点距离。奖励函数是**无状态的（stateless）**，不可跨步累积位移信息。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用**：
- `obs`：上一步观察（形状 [27]）
- `action`：应用于最后一步的动作（形状 [8]）
- `next_obs`：当前步观察（形状 [27]）
- `info`：当前环境下为空字典，**不得使用**任何 info 字段。
- `training_progress`：仅当环境 prompt 明确要求时效性奖励调度时使用，此处没有被要求，默认不使用。

**禁止使用**：
- `original_reward`：官方奖励被屏蔽，严禁直接或间接使用。
- 任何禁止的 info 字段（如 `reward_forward`, `x_position` 等）。
- 任何未在观察空间中声明的物理量（如接触力、关节力矩、地面反作用力等）。
- 任何跨步累积的状态（如积分器），因为 reward 函数必须无状态。

## 7. 可用于奖励函数的信号
以下信号均可直接从 obs/next_obs 或 action 得到，可作为奖励设计基础：

- **位置/高度**：`body_z`（高度），可通过 `obs[0]` 和 `next_obs[0]` 获取。
- **速度**：
  - `body_x_velocity`（前向速度）：`obs[13]` 或 `next_obs[13]`，直接从观察获得。
  - `body_y_velocity`（横向速度）：`obs[14]`，可用于惩罚侧滑。
  - `body_z_velocity`（垂向速度）：`obs[15]`，可用来防止突然摔落或弹跳。
  - 身体角速度：`obs[16:19]`，可抑制翻滚和俯仰。
  - 关节角速度：`obs[19:27]`，可用于动作平滑性。
- **方向**：四元数 `obs[1:5]`，可计算 `body_up_z` 来表示直立程度。
- **关节角度**：`obs[5:13]`，可监测关节超限，但无明确限位，需谨慎使用。
- **动作**：`action` 向量 [8]，直接用于惩罚大力矩或动作变化率（与 `obs` 中上一动作相比）。但注意这里 `action` 是刚刚执行完的动作，可以结合 next_obs 设计平滑项。

## 8. 不确定或不可用的信号
- **接触力**：无任何脚底接触或地面反力信息。
- **关节力矩/电流**：不可见，只有命令力矩 action。
- **全局位置**：`x_position`, `y_position` 不在 obs 中，info 也被禁止，无法获得。
- **距离/里程**：无法，因为无位置积分。
- **目标速度/参考轨迹**：不存在。
- **关节角度限位**：未提供关节限位信息，无法设计精确的关节限位惩罚，只能猜测大致范围（如 ±π）或完全不使用。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: quadruped
  actuator_type: torque-controlled_joints (8 DoF)
  contact_structure: ground_contact_without_force_feedback
primary_objectives:
  - sustain_forward_velocity (via body_x_velocity)
  - maintain_healthy_body_height (z between 0.2 and 1.0)
secondary_objectives:
  - stay_upright (body_up_z close to 1)
  - suppress_lateral_velocity
  - smooth_actuation_and_low_energy
  - avoid_joint_limit_crossing (inferred)
main_failure_risks:
  - collapse (body_z < 0.2)
  - launch (body_z > 1.0)
  - overturning (roll/pitch leading to collapse)
  - numerical divergence (NaN/inf states)
  - low or zero forward speed (standstill)
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id**: `forward_progress`  
  **purpose**: 鼓励机器人产生正向的前进速度，使 body_x_velocity 持续为正且尽可能高（但不过度失控）。  
  **why_required**: 这是整个任务的核心目标，没有这个职责策略可能只学会原地站立保持高度。  
  **usable_signals**: `next_obs[13]` (body_x_velocity)  
  **risks**: 若权重过高，可能使机器人以极端姿态（如低头冲刺）获得速度，容易摔倒。

- **role_id**: `survival_height`  
  **purpose**: 惩罚身体高度偏离健康区间边界，尤其是防止过低（摔倒）和过高（弹跳）。  
  **why_required**: 高度超出区间直接终止，这是生存硬约束；必须通过 reward 梯度引导策略维持在安全区内。  
  **
