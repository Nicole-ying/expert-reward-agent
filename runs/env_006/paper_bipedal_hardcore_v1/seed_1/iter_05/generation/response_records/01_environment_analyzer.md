# Response Record

# 匿名环境理解卡片

## 1. 任务目标
该环境要求一个双足智能体在不规则地形（包含梯子、树桩、坑洼、不平表面）上尽可能远、尽可能高效地向前移动。  
主目标：持续稳定地向前推进，最大化累计前进距离，同时抑制不必要的关节力矩（能量消耗）。  
次目标：利用前方激光雷达扫描数据预判地形变化，主动调整步态以避免摔倒并适应障碍。  
不应混淆的目标：环境中**没有指定的坐标点到达任务**，到达地形末端只是自然边界（可视为成功，但不需单独奖励学习）。核心是 locomotion 连续控制，不是稀疏目标导航。

## 2. 任务类型选择
selected_route_id: **locomotion_continuous_control**  
confidence: high  
reason: 任务描述明确要求向前移动越过连续粗糙地面，没有固定目标点，重要的是前进距离与稳定性，属于典型的连续运动控制问题。观测中包含关节状态与地形传感，动作是关节力矩，符合 locomotion 属性。

**动力学子类型（dynamics_subtype）**：planar_bipedal_gait  
（二维平面双足步行，双腿各有一个髋关节和一个膝关节，需要生成交替步态以在不平地上前进）

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32（推测，所有传感器和关节测量均为浮点）
- obs[0]: hull_angle, 身体俯仰角；可用于稳定性奖励，reward_usable: true
- obs[1]: hull_angular_velocity, 身体俯仰角速度；可用于惩罚快速倾斜，reward_usable: true
- obs[2]: horizontal_speed, 质心水平速度；用于前进奖励，reward_usable: true
- obs[3]: vertical_speed, 质心垂直速度；可用于跌落或弹跳检测，reward_usable: true
- obs[4]: joint_0_angle (hip_1)，髋关节1角度；用于动作平滑或姿态约束，reward_usable: true
- obs[5]: joint_0_speed (hip_1_speed)，髋关节1角速度；reward_usable: true
- obs[6]: joint_1_angle (knee_1)，膝关节1角度；reward_usable: true
- obs[7]: joint_1_speed (knee_1_speed)，reward_usable: true
- obs[8]: joint_2_angle (hip_2)，reward_usable: true
- obs[9]: joint_2_speed (hip_2_speed)，reward_usable: true
- obs[10]: joint_3_angle (knee_2)，reward_usable: true
- obs[11]: joint_3_speed (knee_2_speed)，reward_usable: true
- obs[12]: leg_1_ground_contact, 腿1接地标志(0/1)；可用于步态接触奖励，reward_usable: true
- obs[13]: leg_2_ground_contact, 腿2接地标志；reward_usable: true
- obs[14]–obs[23]: lidar_1 至 lidar_10, 激光测距读数（前方地形高度信息）；可用于前瞻性步态调整，但直接作为奖励信号较难，reward_usable: true (但需配合使用)

**全部信号均可用于奖励函数**，但不推荐直接使用 lidar 原始值作为奖励，而更适合作为状态特征。

## 4. 动作空间 action_space
- type: Box (连续)
- shape: [4]
- bounds: [-1.0, 1.0] (归一化力矩)
- action_dim 0: hip_1_torque, 施加在第一个髋关节的力矩（归一化）
- action_dim 1: knee_1_torque, 施加在第一个膝关节的力矩
- action_dim 2: hip_2_torque, 施加在第二个髋关节的力矩
- action_dim 3: knee_2_torque, 施加在第二个膝关节的力矩

每个动作维度直接控制一个关节力矩，没有离散动作分支。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success‑like termination**: `reached_end_of_terrain` – 智能体到达地形末端，视为任务成功完成。
- **failure‑like termination**: `body_fallen_over` – 身体摔倒（俯仰角过大或接触地面等内部判定），视为失败。
- **ambiguous termination**: 无。终止仅由以上两个条件之一触发。
- **truncation**: 源码中 `truncated` 硬编码为 `False`，因此没有时间截断；若环境实际有最大步数限制（常见于封装），则会被环境外层截断，但 step 调用不返回该信息。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false** (info 为空，无 `info["success"]` 等字段)
- explicit_failure_flag_available: **false**
- allowed_info_fields: [] （info 始终为空）
- forbidden_or_uncertain_info_fields: 任何 `info` 字段均不可用
- 终止原因无法直接从返回数据中读得，但可以通过**间接推断**获得：
  - **摔倒推断**：若 `terminated=True` 且 `next_obs[0]` (hull_angle) 绝对值较大、`next_obs[3]` (vertical_speed) 出现大幅负值、`next_obs[1]` (hull_angular_velocity) 突发变化，且双腿接触标志可能同时为 0，则极有可能是摔倒。
  - **到达终点推断**：若 `terminated=True` 且上述摔倒特征均未出现，hull_angle 保持较小、vertical_speed 接近 0、hull_angular_velocity 平缓，同时可能双腿接地，可推断为成功到达终点。注意，这种推断存在一定误判风险，需谨慎使用。

因此，奖励函数中**不可依赖显式成功/失败标志**，可将推断结果作为条件信号使用（见后文 `role_to_signal_mapping` 中的 `derived_possible` 标注）。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- `obs`: 上一步的观测（24维）
- `action`: 上一步执行的动作（4维力矩）
- `next_obs`: 当前步的观测（24维）
- `info` 中**明确的允许字段**（本例允许字段为空，故 info 不可用）
- `training_progress` 仅当 prompt 明确允许时才可用（此处未明确允许避免使用，但若需要也可作为辅助变量）。

禁止使用：
- `original_reward`
- `official_reward`
- 任何未声明的 info 字段
- 未声明的 obs 切片

## 7. 可用于奖励函数的信号
- **position**: 无绝对位置，但可通过积分水平速度近似前进距离。
- **velocity**: `horizontal_speed` (obs[2]) – 正向速度，用于前进奖励；`vertical_speed` (obs[3]) – 用于检测摔倒、异常跳跃；`hull_angular_velocity` (obs[1]) – 用于惩罚快速翻滚。
- **orientation**: `hull_angle` (obs[0]) – 身体俯仰，惩罚倾斜。
- **contact**: `leg_1_ground_contact` (obs[12]), `leg_2_ground_contact` (obs[13]) – 用于步态模式奖励（鼓励交替接触）。
- **action/engine**: 动作力矩 `action[0..3]` – 用于能量效率惩罚（平方和）。
- **other**:
  - 关节角度/速度 (obs[4..11]) – 可用于关节限位或平滑性惩罚。
  - lidar 读数 (obs[14..23]) – 原始地形高度信息，难以直接作为标量奖励，但可结合 `next_obs` 的稳定性变化形成衍生信号（实际很少直接使用）。
  - **推断终止成功** (derived_possible): 基于 `next_obs` 中 hull_angle、vertical_speed、contact 等特征判断是否正常到达终点，可给出一次性终止奖励。

## 8. 不确定或不可用的信号
- 显式 `success` 标志：不可用
- 显式 `failure` 标志：不可用
- 绝对坐标（x 位置）：不可用
- 累计前进距离（除非环境返回，实际无）：不可用
- 地形全局信息（如剩余长度）：不可用
- 其他 info 字段（如 reward components）：不可用
- 原始奖励 `original_reward`：禁止使用
- 训练进度 `training_progress`：未明确要求使用，谨慎使用

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: bipedal_walker (two legs, each with hip and knee joints)
  actuator_type: torque_controlled_motors (normalised range -1..1)
  contact_structure: binary foot contacts (two feet)
primary_objectives:
  - maximise forward progression (integrated horizontal velocity)
  - maintain upright stability (minimise hull tilt)
  - minimise joint torque usage (energy efficiency)
secondary_objectives:
  - adapt gait to rough terrain using forward lidar scans
  - achieve smooth, alternating step pattern
main_failure_risks:
  - falling over due to terrain obstacles or poor balance
  - high energy, jerky actions that destabilise the robot
  - failure to learn terrain-aware foot placement (lidar under‑
