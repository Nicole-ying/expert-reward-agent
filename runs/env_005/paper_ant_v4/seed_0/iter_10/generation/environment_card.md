# 匿名环境理解卡片

## 1. 任务目标
本任务是一个三维四足机器人的连续控制问题。机器人拥有四条腿和八个力矩控制关节，需要向前行走或奔跑，全程保持身体高度在健康范围内，并维持基本直立的姿态。主目标是实现稳定、高效的前进运动；次要目标是抑制侧向漂移、过度抖动以及非必要的能量消耗。任务不要求到达特定位置，只关心前行速度和存活状态。不应混淆的目标包括：保持原地静止（只求平衡）或单纯避免跌倒而不前进。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 核心目标是让四足机器人在连续力矩控制下持续向前运动，符合持续移动通过环境（平坦地面）的特征，无明确终止目标位置，属于经典的移动连续控制。附属的速度、姿态、能耗等要求是辅助性的，不构成多目标冲突。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: float (连续)
- 各维度含义与奖励可用性：

| 索引 | 名称                        | 含义                                            | reward_usable |
|------|-----------------------------|-------------------------------------------------|---------------|
| 0    | body_z                      | 躯体垂直高度                                      | true          |
| 1    | quat_w                      | 躯体朝向四元数 w 分量                             | true（辅助计算直立） |
| 2    | quat_x                      | 躯体朝向四元数 x 分量                             | true          |
| 3    | quat_y                      | 躯体朝向四元数 y 分量                             | true          |
| 4    | quat_z                      | 躯体朝向四元数 z 分量                             | true          |
| 5    | joint_1_angle               | 第1髋关节角度                                    | true（用于平滑度量） |
| 6    | joint_2_angle               | 第1踝关节角度                                    | true          |
| 7    | joint_3_angle               | 第2髋关节角度                                    | true          |
| 8    | joint_4_angle               | 第2踝关节角度                                    | true          |
| 9    | joint_5_angle               | 第3髋关节角度                                    | true          |
| 10   | joint_6_angle               | 第3踝关节角度                                    | true          |
| 11   | joint_7_angle               | 第4髋关节角度                                    | true          |
| 12   | joint_8_angle               | 第4踝关节角度                                    | true          |
| 13   | body_x_velocity             | 世界坐标系下躯体前向速度                          | true（核心信号） |
| 14   | body_y_velocity             | 世界坐标系下躯体横向速度                          | true          |
| 15   | body_z_velocity             | 躯体垂直速度                                    | true          |
| 16   | body_roll_velocity          | 躯体滚动角速度                                  | true          |
| 17   | body_pitch_velocity         | 躯体俯仰角速度                                  | true          |
| 18   | body_yaw_velocity           | 躯体偏航角速度                                  | true          |
| 19   | joint_1_velocity            | 第1髋关节角速度                                  | true          |
| 20   | joint_2_velocity            | 第1踝关节角速度                                  | true          |
| 21   | joint_3_velocity            | 第2髋关节角速度                                  | true          |
| 22   | joint_4_velocity            | 第2踝关节角速度                                  | true          |
| 23   | joint_5_velocity            | 第3髋关节角速度                                  | true          |
| 24   | joint_6_velocity            | 第3踝关节角速度                                  | true          |
| 25   | joint_7_velocity            | 第4髋关节角速度                                  | true          |
| 26   | joint_8_velocity            | 第4踝关节角速度                                  | true          |

- body_up_z 可从 quat 推导：`1 - 2*(quat_x^2 + quat_y^2)`，可用于衡量直立程度（1.0 完全直立，0 为水平，负值为倒置）。

## 4. 动作空间 action_space
- type: Box
- shape: (8,)
- dtype: float, 连续
- 范围: [-1.0, 1.0] 每个维度
- 每个维度的含义：

| 维度 | 动作名            | 含义                       |
|------|-------------------|----------------------------|
| 0    | hip_1_torque      | 第1髋关节扭矩指令           |
| 1    | ankle_1_torque    | 第1踝关节扭矩指令           |
| 2    | hip_2_torque      | 第2髋关节扭矩指令           |
| 3    | ankle_2_torque    | 第2踝关节扭矩指令           |
| 4    | hip_3_torque      | 第3髋关节扭矩指令           |
| 5    | ankle_3_torque    | 第3踝关节扭矩指令           |
| 6    | hip_4_torque      | 第4髋关节扭矩指令           |
| 7    | ankle_4_torque    | 第4踝关节扭矩指令           |

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无明确的成功终止条件；存活至截断可视为隐式成功。
- failure-like termination:
  - body_height_outside_healthy_range: body_z 落在 (0.2, 1.0) 之外（倒地或过度腾空）。
  - state_value_outside_finite_range: 任何状态值变为 NaN 或 Inf。
- ambiguous termination: 无。
- truncation: 达到预设最大时间步数（time_limit_reached），视为正常截断，不应被惩罚。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 中无任何成功标志）。
- explicit_failure_flag_available: false（终止原因不通过 info 暴露）。
- allowed_info_fields: []（空的，禁止使用任何 info 字段）。
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin 等均被屏蔽，不可用。
- 备注：终止条件只能在环境交互层面感知（如训练循环），`compute_reward` 内部无法直接获取 `terminated` 标志（除非通过 `next_obs` 中的无效值推断，但不建议依赖）。奖励设计必须依赖观测信号（如当前身体高度、速度）来预测并避免失败区域。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`：当前观测（27维数组）
- `action`：当前执行的动作（8维数组）
- `next_obs`：下一时刻观测（27维数组）
- `info` 中明确允许的字段（本环境中无允许字段，禁止访问任何键）
- `training_progress`：仅在 prompt 明确允许时可用（本任务未提及，因此禁止使用）

禁止使用：
- `original_reward`（已被屏蔽）
- `official_reward` 或任何隐藏在 info 中的原有奖励项
- `info` 的任何字段（包括可能存在的 success, failure, termination_reason 等）
- `x_position`, `y_position`, `distance_from_origin` 等被屏蔽的位置信息
- 未声明的观测维度切片（例如，不得假设未来观测中存在绝对位置）

## 7. 可用于奖励函数的信号
- position:
  - body_z：身体高度，可构建高度安全奖励/惩罚。
  - quat：可计算 body_up_z，反映直立程度。
- velocity:
  - body_x_velocity：前进速度（核心正奖励信号）。
  - body_y_velocity：横向速度（可用作方向保持惩罚）。
  - body_z_velocity：垂直速度（可抑制跳跃倾向）。
  - body_roll/pitch/yaw_velocity：角速度（稳定性惩罚）。
- joint state:
  - 所有关节角度和角速度（可用于动作平滑性、关节限位惩罚）。
- action/engine:
  - action（扭矩向量）：可构建能耗惩罚或动作幅度正则化。
- other:
  - 没有接触力、足底触地状态等信号。

## 8. 不确定或不可用的信号
- 绝对位置（世界坐标 x, y）：不可用，无法计算距离进度。
- 接触力或接触标志：不可用，无法奖励正确的着地模式。
- 官方奖励项（forward, ctrl, contact, survive）：被屏蔽，禁止使用。
- 终止标志、成功/失败标签：info 不可访问，不能作为奖励依据。
- 环境内部状态（如质心位置、关节力矩限制等）：未暴露。
- 训练进度参数：未允许使用。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: 三维四足躯体，8个旋转关节（髋+踝）
  actuator_type: 力矩直接控制，无齿轮或位置伺服模拟
  contact_structure: 足端与地面接触（接触力不可观测）
primary_objectives:
  - 持续向前高速移动（最大化 body_x_velocity 正方向分量）
  - 保持躯体直立（body