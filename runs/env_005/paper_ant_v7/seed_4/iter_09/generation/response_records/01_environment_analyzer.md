# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本任务要求控制一个3D四足机器人，在 8 个力矩驱动关节的作用下，持续、稳定地向前行走或奔跑，同时  
保持直立姿态与健康身体高度（不得低于 0.2 米或高于 1.0 米）。**核心目标是稳定前向移动**，而不是  
仅仅维持不倒。附属目标包括减少不必要的侧向漂移、维持平稳姿态、降低关节能耗，但这些都属于辅助  
优化，不应混淆为主目标。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 任务的核心是“持续前进通过平坦地形”，前向速度是主要驱动指标，高度约束仅作为安全性  
生存边界。不存在等权的多目标冲突，唯一的主线就是稳定位移，因此归入 locomotion_continuous_control。

## 3. 观察空间 observation_space
- type: Box  
- shape: (27,)  
- dtype: 连续浮点（通常为 float32）  

各维度含义及可用于奖励与否：

| index | name              | meaning（已匿名）                                  | reward_usable |
|-------|-------------------|----------------------------------------------------|---------------|
| 0     | body_z            | 主干身体垂直位置（高度）                            | true          |
| 1     | quat_w            | 身体姿态四元数实部 w                               | true          |
| 2     | quat_x            | 四元数虚部 x                                       | true          |
| 3     | quat_y            | 四元数虚部 y                                       | true          |
| 4     | quat_z            | 四元数虚部 z                                       | true          |
| 5     | joint_1_angle     | 第 1 髋关节角度                                    | true          |
| 6     | joint_2_angle     | 第 1 踝关节角度                                    | true          |
| 7     | joint_3_angle     | 第 2 髋关节角度                                    | true          |
| 8     | joint_4_angle     | 第 2 踝关节角度                                    | true          |
| 9     | joint_5_angle     | 第 3 髋关节角度                                    | true          |
| 10    | joint_6_angle     | 第 3 踝关节角度                                    | true          |
| 11    | joint_7_angle     | 第 4 髋关节角度                                    | true          |
| 12    | joint_8_angle     | 第 4 踝关节角度                                    | true          |
| 13    | body_x_velocity   | 身体在世界坐标系 x 轴（正前方）的速度               | true          |
| 14    | body_y_velocity   | 身体侧向速度                                       | true          |
| 15    | body_z_velocity   | 身体垂向速度                                       | true          |
| 16    | body_roll_velocity  | 身体绕 x 轴的角速度                              | true          |
| 17    | body_pitch_velocity | 身体绕 y 轴的角速度                              | true          |
| 18    | body_yaw_velocity   | 身体绕 z 轴的角速度                              | true          |
| 19    | joint_1_velocity  | 第 1 髋关节角速度                                  | true          |
| 20    | joint_2_velocity  | 第 1 踝关节角速度                                  | true          |
| 21    | joint_3_velocity  | 第 2 髋关节角速度                                  | true          |
| 22    | joint_4_velocity  | 第 2 踝关节角速度                                  | true          |
| 23    | joint_5_velocity  | 第 3 髋关节角速度                                  | true          |
| 24    | joint_6_velocity  | 第 3 踝关节角速度                                  | true          |
| 25    | joint_7_velocity  | 第 4 髋关节角速度                                  | true          |
| 26    | joint_8_velocity  | 第 4 踝关节角速度                                  | true          |

所有观察均可被奖励函数读取。注意：无接触力、无绝对 x/y 位置，仅可从速度推断位移趋势。

## 4. 动作空间 action_space
- type: Box  
- shape: (8,)  
- dtype: 连续（float），范围为 [-1.0, 1.0] 每个动作  

| dim | name              | 含义 (关节力矩)                                      |
|-----|-------------------|------------------------------------------------------|
| 0   | hip_1_torque      | 第 1 髋关节输出扭矩                                  |
| 1   | ankle_1_torque    | 第 1 踝关节输出扭矩                                  |
| 2   | hip_2_torque      | 第 2 髋关节输出扭矩                                  |
| 3   | ankle_2_torque    | 第 2 踝关节输出扭矩                                  |
| 4   | hip_3_torque      | 第 3 髋关节输出扭矩                                  |
| 5   | ankle_3_torque    | 第 3 踝关节输出扭矩                                  |
| 6   | hip_4_torque      | 第 4 髋关节输出扭矩                                  |
| 7   | ankle_4_torque    | 第 4 踝关节输出扭矩                                  |

动作是直接力矩命令，策略需通过这 8 个输出来同步四条腿的运动。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 时间限制截断（truncation）。如果机器人一直在健康高度、无 NaN，  
  坚持到 episode 时间上限，可视为一次成功的“持续运动”。  
- **failure-like termination**:  
  - body_height_outside_healthy_range：高度 < 0.2（跌倒）或 > 1.0（弹飞），任意一条触发即终止；  
  - state_value_outside_finite_range：任一状态值变为 NaN/inf，系统崩溃。  
- **ambiguous termination**: 无。  
- **truncation**: 仅 time_limit_reached，不属于终止条件，但作为截断信号。  
  注：truncation 本身不代表失败，但也不携带“走得多远”的绝对路径信息。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: [] （没有任何 info 字段可供使用）  
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin 等所有被明确标为 forbidden 的字段均不可用。  

因此，无法通过 info 直接获知“本次达到成功”，只能从截断或终止条件自行推断（如 truncated=true 视为存活完成，terminated=true 且高度超出范围视为失败）。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs (当前观察，shape (27,))  
- action (当前动作，shape (8,))  
- next_obs (下一观察)  
- info （但该环境中 allowed_info_fields 为空，故只能使用空的 info，实质不可依赖任何 info 字段）  
- training_progress（仅在明确允许时可用，本任务无此要求，通常忽略）

禁止使用：
- original_reward（已 masked）  
- 任何在 forbidden_info_fields 中列出的字段（如 reward_forward 等）  
- 任何未在观察空间中声明的信息，例如机体绝对坐标、接触力等  

## 7. 可用于奖励函数的信号
- **position**:  
  - body_z (obs[0])：可直接作为高度信号  
  - 四元数 (obs[1:5])：可计算身体直立度 `body_up_z = 1 - 2*(quat_x² + quat_y²)`，值域约 [-1,1]，1.0 表示完全竖直  
  - 关节角度 (obs[5:13])：可用于模仿参考姿态或惩罚异常关节位置  
- **velocity**:  
  - body_x_velocity (obs[13])：前向速度，核心推进指标  
  - body_y_velocity (obs[14])：侧向速度，可惩罚侧滑  
  - body_z_velocity (obs[15])：垂向速度，可结合高度变化设计稳定性奖励  
  - 身体角速度 (obs[16:19])：可用于惩罚晃动  
  - 关节角速度 (obs[19:27])：可惩罚剧烈转动  
- **orientation**: 四元数间接提供身体倾斜信息，同 position 部分  
- **contact**: 无（无接触力、无足端压力）  
- **action/engine**:  
  - 动作 torque (action[0:8])：可用于惩罚过大力矩、平滑度（与上一动作的差异）  

## 8. 不确定或不可用的信号
- 绝对 x 坐标 / y 坐标：禁止；无法判断实际走了多远，只能通过速度积累的间接估计（但奖励函数 stateless，不能跨 step 积分）  
- 接触信息：无传感器，无法知道哪条腿着地  
- 官方奖励项：全部遮蔽，不得重建或使用  
- 成功率/失败标签：info 无相应字段  
- 环境内部动力学参数（摩擦、质量等）：不可见  

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: quadruped_torso_with_four_eleven_limbs
  actuator_type: torque_controlled_joints
  contact_structure: four_leg_ground_contacts_no_force_sensor
primary_objectives:
  - generate stable forward velocity without falling
  - maintain body_z within [0.2, 1.0] and keep upright posture
secondary_objectives:
  - suppress lateral velocity and excessive joint torques
  - achieve natural-looking, energy-efficient gait
main_failure_risks:
  - body_z falls below 0.2 (leg collapse / fall)
  - body_z exceeds 1.0 (launch / extreme jump)
  - loss of upright orientation leading to imbalance
  - NaN in states due to simulation instability
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1
