# Response Record

# 匿名环境理解卡片

## 1. 任务目标
这是一个 3D 四足机器人的连续控制行走任务。机器人拥有四条腿、八个力矩控制的关节，需要在保持身体直立的条件下持续向前（世界 x 方向）移动。身体高度必须维持在健康区间（0.2 m 到 1.0 m）内，一旦超出该范围（摔倒或向上弹飞）或状态出现非法值，episode 立即终止。策略必须学会稳定、高效的前进步态，而不能只是被动维持在安全高度而不前进。

副产品不得被当作核心目标（例如最小化能耗、横向位移等虽有价值，但在成功前进面前属于次要）。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 核心目标是使得四足机器人在连续动作控制下产生持续向前移动；生存（高度、姿态）是为前进服务的健康约束，而非任务主目标。动态特征表现为多足身体沿指定方向行进，属于 locomotion_continuous_control 族。

额外判断的动力学细节：
dynamics_subtype: multi_legged_body_locomotion  
（多足躯体沿目标方向前进，高度和姿态约束作为健康条件）

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: float32（隐含连续值）
- 各维度含义：

| 索引 | 名称                    | 含义                                        | reward 可用 |
|------|------------------------|---------------------------------------------|-------------|
| 0    | body_z                 | 身体中心垂直高度（m）                        | 是          |
| 1    | quat_w                 | 身体朝向四元数实部 w                         | 是（参与计算直立） |
| 2    | quat_x                 | 身体朝向四元数虚部 x                         | 是（参与计算直立） |
| 3    | quat_y                 | 身体朝向四元数虚部 y                         | 是（参与计算直立） |
| 4    | quat_z                 | 身体朝向四元数虚部 z                         | 是（参与计算直立） |
| 5    | joint_1_angle          | 髋关节 1 角度                               | 是（可选姿势成本） |
| 6    | joint_2_angle          | 踝关节 1 角度                               | 是（可选姿势成本） |
| 7    | joint_3_angle          | 髋关节 2 角度                               | 是（可选姿势成本） |
| 8    | joint_4_angle          | 踝关节 2 角度                               | 是（可选姿势成本） |
| 9    | joint_5_angle          | 髋关节 3 角度                               | 是（可选姿势成本） |
| 10   | joint_6_angle          | 踝关节 3 角度                               | 是（可选姿势成本） |
| 11   | joint_7_angle          | 髋关节 4 角度                               | 是（可选姿势成本） |
| 12   | joint_8_angle          | 踝关节 4 角度                               | 是（可选姿势成本） |
| 13   | body_x_velocity        | 身体在世界 x 方向的速度（前进速度）          | 是（关键前进信号） |
| 14   | body_y_velocity        | 身体在世界 y 方向的速度（横向漂移）          | 是（惩罚漂移）   |
| 15   | body_z_velocity        | 身体垂直方向速度                             | 是（惩罚剧烈跳动） |
| 16   | body_roll_velocity     | 翻滚角速度                                  | 是（惩罚旋转）   |
| 17   | body_pitch_velocity    | 俯仰角速度                                  | 是（惩罚旋转）   |
| 18   | body_yaw_velocity      | 偏航角速度                                  | 是（惩罚旋转）   |
| 19   | joint_1_velocity       | 关节 1 角速度                               | 是（可选动作平滑成本） |
| 20   | joint_2_velocity       | 关节 2 角速度                               | 是（可选动作平滑成本） |
| 21   | joint_3_velocity       | 关节 3 角速度                               | 是（可选动作平滑成本） |
| 22   | joint_4_velocity       | 关节 4 角速度                               | 是（可选动作平滑成本） |
| 23   | joint_5_velocity       | 关节 5 角速度                               | 是（可选动作平滑成本） |
| 24   | joint_6_velocity       | 关节 6 角速度                               | 是（可选动作平滑成本） |
| 25   | joint_7_velocity       | 关节 7 角速度                               | 是（可选动作平滑成本） |
| 26   | joint_8_velocity       | 关节 8 角速度                               | 是（可选动作平滑成本） |

- 注：世界 x 方向固定（任务方向），无需从 observation 中读取 x/y 位置。

## 4. 动作空间 action_space
- type: Box
- shape: (8,)
- continuous: true
- 每维取值范围：[-1.0, 1.0]
- 各动作维度对应：

| 维度 | 名称               | 含义                             |
|------|--------------------|----------------------------------|
| 0    | hip_1_torque       | 第一条腿的髋关节力矩命令         |
| 1    | ankle_1_torque     | 第一条腿的踝关节力矩命令         |
| 2    | hip_2_torque       | 第二条腿的髋关节力矩命令         |
| 3    | ankle_2_torque     | 第二条腿的踝关节力矩命令         |
| 4    | hip_3_torque       | 第三条腿的髋关节力矩命令         |
| 5    | ankle_3_torque     | 第三条腿的踝关节力矩命令         |
| 6    | hip_4_torque       | 第四条腿的髋关节力矩命令         |
| 7    | ankle_4_torque     | 第四条腿的踝关节力矩命令         |

- 所有动作都表示归一化力矩指令，施加于对应的关节。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 不存在明确成功终止（例如达到目标位置）；唯一的“正常结束”是时间截断（truncated）。
- failure-like termination:
  - body_z 小于 0.2 或大于 1.0 → 跌倒或向上猛烈弹射，视为失败。
  - 状态值包含 NaN 或无穷 → 模拟不稳定，失败。
- ambiguous termination: 无。
- truncation: 达到 episode 时间上限（细节未提供，但典型 MuJoCo 任务有步骤上限）→ 截断，不能直接视为成功或失败，但高度存活且有一定前进距离时可认为表现良好。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 中无任何字段）
- explicit_failure_flag_available: false（info 为空）
- allowed_info_fields: 无（明确 info 字典中不提供任何字段以供奖励使用）
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin 等均禁止；此外无法获得接触力。禁止依赖任何 info 内容。

## 6. reward 函数接口契约

函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 上一时刻观测 (27D)
- action: 刚刚执行的动作 (8D)
- next_obs: 当前时刻观测 (27D)，注意实际可能希望用 next_obs 来评估状态结果
- info: 空字典，**禁止使用**
- training_progress: 仅在 prompt 明确要求时使用，当前上下文未给予，**建议不使用**

禁止使用：
- original_reward: 被 masked，禁止使用
- official_reward 或任何从 info 中获取的原始奖励分量
- info 中任何未声明的字段（目前为全部字段）
- obs 中未声明的维度（例如未定义的 x/y 位置）

函数等价于接收当前步的转换元组 (obs, action, next_obs) 并返回一个标量奖励值。

## 7. 可用于奖励函数的信号
- **position**:
  - body_z (obs[0], next_obs[0]): 身体高度，可用于惩罚偏离健康高度。
  - quaternion (obs[1:5], next_obs[1:5]): 身体朝向，可推导 body_up_z = 1 - 2*(quat_x² + quat_y²)，用于奖励直立。
  - joint angles (obs[5:13], next_obs[5:13]): 可用于惩罚关节超出合理范围或过大偏移，但难以无接触力直接定义合理姿态。
- **velocity**:
  - body_x_velocity (obs[13], next_obs[13]): 前进速度，核心前进奖励。
  - body_y_velocity (obs[14], next_obs[14]): 横向速度，可用于惩罚漂移。
  - body_z_velocity (obs[15], next_obs[15]): 垂直速度，可用于惩罚猛跳。
  - angular velocities (obs[16:19], next_obs[16:19]): 翻滚、俯仰、偏航角速度，可用于惩罚非直立体转。
  - joint velocities (obs[19:27], next_obs[19:27]): 关节速度，可用于惩罚剧烈抖动或动作能量消耗。
- **contact**: 无（观测中没有脚底接触力或触地信号）
- **action/engine**:
  - action (8D): 动作大小，可基于 L2 惩罚力矩平方和鼓励平滑、低功耗。
- **other**:
  - 可使用 next_obs[0] 与目标高度的差值实现有界的身体高度奖励。
  - 可从四元数计算 tilt_angle = arccos(body_up_z)，用于精细倾斜惩罚。

## 8. 不确定或不可用的信号
- **x/y 位置**: 被明确禁止，无法获得世界平面坐标，因此不能给予基于绝对位移的奖励，只能依赖身体速度。
- **接触力/触地信号**: 观测中不包含，无法判断哪条腿着地，无法构建精确的步态奖励（如对称触地）。
- **success/failure flag**: info 为空，无法根据终止原因给出额外奖励或惩罚。
- **重心速度方向与躯干朝向偏差**: 可间接用 body x/y velocity 和 quaternion 推导，但不一定准确。
- **能量消耗精确建模**: 无法获取实际扭矩与速度下的功，只能从动作幅度与关节速度近似。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: 3D_quadruped (四足，整体躯干)
  actuator_type: torque_controlled_joints (8个力矩驱动器)
  contact_structure: feet_ground_contact (隐藏，无法直接观测)
primary_objectives:
  - forward_locomotion: 最大化世界 x 方向前进速度（body_x_velocity）
secondary_objectives:
  - upright_posture: 维持 body_up_z ≈ 1.0，避免躯干倾斜
  - height_regulation: 维持 body_z 在安全范围（接近理想值，如 0.5–0.7），远离上下边界
  - lateral_stability: 最小化横向速度 body_y_velocity
  - smooth_actuation: 降低动作幅度/力矩、关节速度，提倡节能与少抖动
main_failure_risks:
  - 跌倒（body_z 低于 0.2 m，躯干砸地）
  - 弹飞（body_z 超过 1.0 m，躯干猛烈上冲）
  - 原地踏步或倒退（body_x_velocity ≤ 0）
  - 过度横向漂移（body_y_velocity 幅值过大）
  - 姿态发散，旋转过快，最终翻倒
  - 动作剧烈振荡，导致关节速度过大或数值不稳定
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_velocity_reward
  purpose: 鼓励机器人向前移动，核心前进信号。
  why_required: 没有此奖励，策略可能选择停滞并只维持高度/直立，难以完成 locomotion 根本目标。
  usable_signals: [body_x_velocity (next_obs[13])]
  risks: 过高的前进奖励可能导致牺牲姿态或高度安全而摔倒；可配合有界信号（如 tanh）或终止惩罚缓解。

- role_id: upright_and_orientation_penalty
  purpose: 惩罚躯干偏离直立姿态，保证行走时姿态稳定。
  why_required: 若缺少直立约束，策略可能通过极端倾斜获得虚假的前进速度，但实际会加剧摔倒风险。
  usable_signals: [quaternion (next_obs[1:5]) 计算 body_up_z]
  risks: 过度惩罚微小倾斜可能影响急转弯等必要
