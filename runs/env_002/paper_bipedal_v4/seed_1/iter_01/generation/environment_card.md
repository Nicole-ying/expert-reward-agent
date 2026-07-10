# 匿名环境理解卡片

## 1. 任务目标
主体是一个双足躯干，需在起伏不平的二维地形上持续向前行走。  
- 主目标：尽可能快速且远地前进（最大化前进距离和前进速度）。  
- 附属目标：最小化能量消耗（关节力矩使用的代价）。  
- 不应混淆：不是简单到达一个定点，而是持续通过地形，直到走完整段路程或摔倒。摔倒直接终止且视为失败，到达地形末端可能视为更优的完成状态，但不提供显式成功标志。  

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 核心行为是在连续动作空间下通过关节力矩协调步态，以连续前进通过起伏地形为主要目标。次要的能耗优化不改变任务族。无平衡存活等专项目标，故远离survival_balance；无多点到达，故非navigation_goal_reaching；无抓取操作，故非manipulation_grasping；无驾驶安全约束，故非autonomous_driving_safety；目标持续且非稀疏事件，故非sparse_exploration；仅存在一个明确主目标，不构成多目标冲突，故非multi_objective_task。

动力学子类型 dynamics_subtype: planar_bipedal_gait  

## 3. 观察空间 observation_space
- type: Box  
- shape: (24,)  
- dtype: float32（推断，在环境卡片中不严格要求）  

各维度含义与奖励可用性：
| index | name | meaning | reward_usable |
|-------|------|---------|---------------|
| 0 | hull_angle | 躯干相对竖直方向的倾角 | true |
| 1 | hull_angular_velocity | 躯干角速度 | true |
| 2 | horizontal_velocity | 前行/后退线速度 | true |
| 3 | vertical_velocity | 上下线速度 | true |
| 4 | hip1_angle | 腿1髋关节角度 | true |
| 5 | hip1_speed | 腿1髋关节角速度 | true |
| 6 | knee1_angle | 腿1膝关节角度 | true |
| 7 | knee1_speed | 腿1膝关节角速度 | true |
| 8 | leg1_contact | 腿1接地指示（1.0=着地，0.0=离地） | true |
| 9 | hip2_angle | 腿2髋关节角度 | true |
| 10 | hip2_speed | 腿2髋关节角速度 | true |
| 11 | knee2_angle | 腿2膝关节角度 | true |
| 12 | knee2_speed | 腿2膝关节角速度 | true |
| 13 | leg2_contact | 腿2接地指示（1.0=着地，0.0=离地） | true |
| 14–23 | lidar_0..9 | 前方沿地形的10路激光雷达距离测量 | potentially useful, but see notes |

说明：lidar 信息可用于感知前方地形起伏，但在基础前进奖励中并非必需，可作为条件使用或暂不使用。

## 4. 动作空间 action_space
- type: Box (continuous)  
- shape: (4,)  
- bounds: [-1.0, 1.0] per dimension  

动作维度含义：
- action[0]: hip_torque_leg1——施加到腿1髋关节的力矩  
- action[1]: knee_torque_leg1——施加到腿1膝关节的力矩  
- action[2]: hip_torque_leg2——施加到腿2髋关节的力矩  
- action[3]: knee_torque_leg2——施加到腿2膝关节的力矩  

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（完成地形穿越，可能视为成功，但无明确成功标签）  
- failure-like termination: body_fallen_over（躯干摔倒，显然为失败）  
- ambiguous termination: 无  
- truncation: 未提及最大步数截断，但通常环境有预设上限（此处暂不列入主终止条件）  

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: none (step 返回的 info 为空字典 {})  
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，因环境未提供任何额外信息。禁止依赖类似于 `info["success"]`、`info["failure"]`、`info["termination_reason"]` 等未出现的信号。  

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观察向量，shape (24,)）
- action（当前动作向量，shape (4,)）
- next_obs（下一观察向量，shape (24,)）
- info 中明确允许的字段（当前环境中 info 为空，因此实际上无可用的 info 字段）
- training_progress 本提示未明确允许使用，因此**禁止使用**；若未来明确说明可用，才可酌情用于调整温度或衰减等。

禁止使用：
- original_reward（官方奖励被屏蔽，严禁利用）
- official_reward（同义）
- 任何未在观察空间说明书中指明的 obs 切片或字段
- 任何未在 info 中出现的字段（如 success、failure、reward 组成等）

## 7. 可用于奖励函数的信号
- position: 不直接提供世界坐标，但可通过水平速度积分得到前进距离估计（非直接观测）。  
- velocity:  
  - hull_angular_velocity (obs[1])  
  - horizontal_velocity (obs[2])  ← 关键前进速度  
  - vertical_velocity (obs[3])  
- orientation: hull_angle (obs[0])  
- contact:  
  - leg1_contact (obs[8])  
  - leg2_contact (obs[13])  
- action/engine:  
  - 四个关节力矩 (action[0..3]) 可用于能耗度量  
- other:  
  - lidar_0..9 (obs[14..23])：前方地形扫描，可用于地形感知相关奖励，但非进度主目标必需  

## 8. 不确定或不可用的信号
- 无世界坐标 x 位置，故无法直接奖励绝对前进距离；只能通过速度推断。
- 无明确 “healthy” 或 “alive” 标志。
- 无明确力矩与能耗的直接数值（需从动作自行计算，如力矩平方和）。
- 无脚下地形高度或摩擦力反馈（仅有 lidar 前方测距）。
- 无重心高度或触地反力等额外信息。
- info 为空，故无法获得成功/失败标志、接触力、总能耗统计等辅助信号。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: single_segment_torso
  actuator_type: torque_controlled_hip_and_knee_joints
  contact_structure: two_point_leg_contacts_with_ground_flag
primary_objectives:
  - maximize_forward_horizontal_velocity
secondary_objectives:
  - minimize_energy_consumption (sum of squared joint torques)
  - maintain_stability (avoid large hull angle or high angular velocity)
main_failure_risks:
  - falling_over due to loss of balance
  - rigid or uncoordinated gait leading to zero forward motion
  - excessive energy waste without progress
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress
  purpose: 推动智能体产生向前的水平速度。
  why_required: 任务主目标是“走得快、走得远”，必须直接奖励前进速度。
  usable_signals: [obs[2] horizontal_velocity], 也可用 next_obs[2] 的前后差分。
  risks: 若只奖励速度，可能鼓励大幅前倾或不稳定步态，需与其他职责配合。

- role_id: energy_penalty
  purpose: 抑制多余力矩输出，减少能耗。
  why_required: 任务描述明确要求 minimizing energy consumption，必须包含能耗成本。
  usable_signals: [action[0..3] hip/knee torques]
  risks: 过高的惩罚系数可能导致步幅极小甚至停止不动；需要与前进奖励平衡。

- role_id: upright_stability
  purpose: 避免躯干倾斜过大或快速旋转，防止摔倒。
  why_required: 摔倒直接终止且没有进度，维持平衡是健康步态的前提。
  usable_signals: [obs[0] hull_angle, obs[1] hull_angular_velocity]
  risks: 过分限制角度变化可能阻碍正常步态中的轻微身体摆动，应采用轻度二次惩罚或边界外惩罚。

### 10.2 条件职责 conditional_roles
- role_id: contact_regularization
  condition_to_use: 当观察到步态中出现拖腿、双足同时离地或单腿持续不着地时，可考虑引入以鼓励交替接触。
  usable_signals: [obs[8] leg1_contact, obs[13] leg2_contact]
  risks: 错误设计可能奖励“原地踏步”而牺牲前进；仅应在基础前进奖励形成有效步态后作为微调工具。

- role_id: terrain_aware_adaptation（谨慎推荐）
  condition_to_use: 若训练后期发现智能体在复杂地形段缺乏适应性速度调节，可利用 lidar 信号给予小量辅助奖励。
  usable_signals: [obs[14..23] lidar readings]
  risks: 直接奖励 lidar 数据可能导致策略聚焦于传感器读数而非真实行走，且 lidar 的解释性弱，易引入噪声。初期应禁用。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: absolute_distance_reward
  reason: 环境中没有提供 x 坐标，无法直接计算真实前进距离；通过速度积分来模拟距离奖励可能累积漂移，且环境可能相对坐标系重置，故不建议直接使用。

- role_id: alive_bonus
  reason: 环境已由摔倒终止，不必再提供存活时间奖励；存活奖励可能鼓励保守静止，与“走得快又远”冲突。

- role_id: goal_reaching_bonus
  reason: 虽然有 reached_end_of_terrain 终止条件，但无显式成功标志且无法在 episode 中区分；在终止时给予大额奖励需要访问终止标志，而 info 为空导致不可靠，故暂时禁用。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress | obs[2] horizontal_velocity (or next_obs[2]) | absolute x position | linear_velocity_reward, bounded_signal | 小心负速度惩罚；可结合躯干角度修正速度方向 |
| energy_penalty | action[0..3] | direct power/energy measurement | quadratic_penalty(sum_of_squares) | 常用 torque^2 惩罚 |
| upright_stability | obs[0] hull_angle, obs[1] hull_angular_velocity | | quadratic_penalty, bounded_abs_penalty | 可以分开角度和角速度惩罚 |
| contact_regularization | obs[8], obs[13] | ground reaction force | desired_contact_ratio_reward, alternating_contact_bonus | 需定义期望的接触模式 |
| terrain_aware_adaptation | lidar_0..9 (obs[14..23]) | fine-grained height map | function of lidar variance or clearance | 目前建议作为预留，初始训练不用 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 快速摔倒 | high hull_angle, high hull_angular_velocity，episode_length 很短 | 加强 upright_stability 惩罚，降低速度奖励权重，或加入角度-速度联合惩罚 |
| 完全不进（零或负速度） | mean horizontal_velocity ≈ 0 或负 | 检查 energy_penalty 是否过强抑制动作；可引入最小速度阈值奖励或降低能耗项 |
| 原地踏步但稳定 | mean horizontal_velocity ≈ 0，hull_angle 小，episode_length 大但无进度 | 增大 forward_progress 权重，调整 contact_regularization 以鼓励迈步，必要时启用交替接触奖励 |
| 颤动高能耗低速度 | action 幅度大且变化剧烈，quadratic_penalty 高但 velocity 低 | 增加 action 平滑或变化率惩罚，或降低力矩惩罚指数 |
| 仅适用于平坦地形 | lidar 方差小时表现好，处理起伏地形差 | 后期可考虑加入地形感知小奖励或 curriculum 地形难度 |

---