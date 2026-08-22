# Response Record

# 匿名环境理解卡片

## 1. 任务目标
双足智能体需要在崎岖地形（梯子、树桩、坑洼等）上尽可能远且高效地前进。  
**主目标**：通过学习稳定步行穿越复杂地形，尽可能走得更远并到达地形远端（避免摔倒）。  
**次要目标**：在前进过程中最小化关节力矩消耗，实现节能运动。  
注意：不要将“到达终点”视为唯一的导航类点对点任务，因为地形是连续变化的，核心仍是步行能力；也不要把力矩最小化升格为主目标。

## 2. 任务类型选择
- selected_route_id: `locomotion_continuous_control`
- confidence: high
- reason: 核心目标是持续、稳定地前进通过崎岖地形，没有指定的姿态或抓取操作；到达终点是前进的自然结果，而非显式的目标坐标导航。因此属于连续运动控制下的移动任务族。

动力学子类型进一步细化为：
- dynamics_subtype: `planar_bipedal_gait`
（双足平面步态，在含有高度变化的纵平面内前进）

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: 假定为 float32
- 每个索引的含义与奖励可用性：
  - obs[0]: **hull_angle** – 躯干倾角 (rad)，可用于检测摔倒风险。 reward_usable: true
  - obs[1]: **hull_angular_velocity** – 躯干角速度 (rad/s)。 reward_usable: true
  - obs[2]: **horizontal_speed** – 质心水平（前进方向）速度 (m/s)。 reward_usable: true
  - obs[3]: **vertical_speed** – 质心垂直速度 (m/s)。 reward_usable: true
  - obs[4]: **joint_0_angle** (hip_1) – 髋关节1角度。 reward_usable: true
  - obs[5]: **joint_0_speed** (hip_1) – 髋关节1角速度。 reward_usable: true
  - obs[6]: **joint_1_angle** (knee_1) – 膝关节1角度。 reward_usable: true
  - obs[7]: **joint_1_speed** (knee_1) – 膝关节1角速度。 reward_usable: true
  - obs[8]: **joint_2_angle** (hip_2) – 髋关节2角度。 reward_usable: true
  - obs[9]: **joint_2_speed** (hip_2) – 髋关节2角速度。 reward_usable: true
  - obs[10]: **joint_3_angle** (knee_2) – 膝关节2角度。 reward_usable: true
  - obs[11]: **joint_3_speed** (knee_2) – 膝关节2角速度。 reward_usable: true
  - obs[12]: **leg_1_ground_contact** – 腿1触地指示 (0.0 or 1.0)。 reward_usable: true (离散)
  - obs[13]: **leg_2_ground_contact** – 腿2触地指示。 reward_usable: true (离散)
  - obs[14]~obs[23]: **lidar_1~lidar_10** – 前方地形高度测距值。 reward_usable: true（可辅助预判，但难以直接量化成奖励，属于条件可用的感知信号）

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- bounds: [-1.0, 1.0] (归一化转矩)
- 各动作维度含义：
  - action_dim 0: **hip_1_torque** – 髋关节1转矩
  - action_dim 1: **knee_1_torque** – 膝关节1转矩
  - action_dim 2: **hip_2_torque** – 髋关节2转矩
  - action_dim 3: **knee_2_torque** – 膝关节2转矩

所有动作维度均可用于力矩惩罚或动作平滑性奖励。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `reached_end_of_terrain`（到达地形远端）。环境中将此视为一次成功完成。
- **failure-like termination**: `body_fallen_over`（躯干摔倒）。判定标准未在观察中直接给出，但可以从躯干角度突变、垂直速度骤降或触地信号异常推断。
- **ambiguous termination**: 无。任务描述未提到最大步数限制（time limit），故推测不存在 `truncation`。若实际存在未声明的步数上限，则未到达终点且未摔倒的截断属于不明确的终止。
- **truncation**: 文档中未提及，假定为无。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false** (info 为空，无显式成功标记)
- explicit_failure_flag_available: **false** (info 为空，无显式失败标记)
- allowed_info_fields: [] (空)
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用

**推断路径（derived_possible）**：
- 摔倒推断：当终止发生时，若满足 `|hull_angle|` 超过较大阈值（如 0.4 rad）、或 `|hull_angular_velocity|` 极高、或垂直速度突然负向极大，则可以认为发生了摔倒。
- 成功到达终点推断：当终止发生时，若未检测到摔倒信号（hull 角度正常、触地状态持续），则可以推断是到达了终点，此时可给予终端奖励。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- `obs` 和 `next_obs` 的全部 24 个维度
- `action`（4维动作）
- `info` 中唯一允许的字段：无（`info` 为空，禁止使用）
- `training_progress`：本次允许使用，但需谨慎，仅可在奖励被允许调整时使用（本任务未明确禁止，可作可选项）

禁止使用：
- `original_reward`（官方奖励已被遮掩）
- 任何未声明的 `info` 字段
- 任何未在 observation_space 中定义的额外状态

## 7. 可用于奖励函数的信号
- **前进速度**：`horizontal_speed`（obs[2]）可以直接奖励。
- **躯干稳定**：`hull_angle` (obs[0])、`hull_angular_velocity` (obs[1])，用于惩罚倾斜。
- **接触状态**：`leg_1_ground_contact` (obs[12])、`leg_2_ground_contact` (obs[13])，用于鼓励稳定触地或防止抬脚过久。
- **关节力矩**：`action` 的四个维度可直接用于惩罚大转矩（能耗）。
- **垂直速度**：`vertical_speed` (obs[3]) 可用于惩罚剧烈起跳，但在凹凸地形中需小心使用。
- **激光雷达**：`lidar_i` (obs[14:23]) 感知前方地形，可用于条件性奖励（例如预测即将出现的障碍并提前鼓励调整步态），但由于缺乏直接映射，只能作为辅助信号。
- **摔倒检测 derived_possible**：基于 `hull_angle`、`hull_angular_velocity`、`vertical_speed` 的组合。
- **成功到达终点 derived_possible**：基于“终止且非摔倒”的推断。

## 8. 不确定或不可用的信号
- 绝对位置 / 前进距离：观测中不含 x 坐标，无法直接奖励位移。
- 地形类型标签：无法得知当前踏在何种障碍物上。
- 接触力：只有二值触地指示，无接触力大小。
- 能耗/机械功率：无法直接测量，只能通过扭矩间接惩罚。
- 任务是否真正成功/失败：没有显式标志，必须依赖推断。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: bipedal with rigid hull
  actuator_type: torque-controlled hip and knee joints (2 legs × 2 joints)
  contact_structure: binary foot-ground contact indicators (leg_1, leg_2)
primary_objectives:
  - maximize forward progress (distance) on rough terrain while avoiding falling
secondary_objectives:
  - minimize joint torque / energy consumption
  - maintain stable gait with limited unnecessary motions
main_failure_risks:
  - hull tilt exceeds critical angle -> fall
  - getting stuck in deep pits or unable to step over high obstacles (stumps/ladders)
  - oscillation or excessive joint torque leading to unstable gait
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: forward_progress**
  purpose: 鼓励智能体持续前进  
  why_required: 主任务是走得尽可能远，没有前进速度奖励将无法驱动步态形成  
  usable_signals: [horizontal_speed (obs[2])]  
  risks: 单纯最大化速度可能导致失控摔倒；必须与稳定性惩罚结合

- **role_id: survival_balance**
  purpose: 防止摔倒，保持躯干直立  
  why_required: 一旦摔倒任务结束，本体必须学会稳定平衡
