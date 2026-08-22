# 匿名环境理解卡片

## 1. 任务目标

本环境是一个匿名 2D 双足地形行走任务。主目标是控制双足身体在崎岖地形上尽可能远、尽可能快地向前行走，同时保持稳定步态，不发生摔倒。次目标是最小化能量消耗，并保持髋、膝关节动作协调平滑。  
需要注意：该任务不是稀疏导航到唯一目标点的任务，也不是纯粹的平衡存活任务；能量消耗和动作平滑是辅助目标，不应与“向前行走”这一主目标混合成多目标问题。

## 2. 任务类型选择

selected_route_id: locomotion_continuous_control

confidence: high

reason: 核心目标是持续通过地形并前进，动作是连续关节力矩，终止条件包含摔倒。虽然存在 reached_end_of_terrain，但任务描述强调的是“尽可能远、尽可能快地向前走”，而不是到达一个明确稀疏目标点，因此属于连续运动控制中的步态前进任务。

## 3. 观察空间 observation_space

- type: Box
- shape: (24,)
- dtype: float32（任务 spec 未显式写出，但观测均为连续标量，按连续控制环境惯例默认）
- obs[0]: hull_angle，主躯干相对竖直方向的倾角，reward_usable: true
- obs[1]: hull_angular_velocity，主躯干角速度，reward_usable: true
- obs[2]: horizontal_velocity，前进/后退线速度，reward_usable: true
- obs[3]: vertical_velocity，上下线速度，reward_usable: true
- obs[4]: hip1_angle，腿1髋关节角度，reward_usable: true
- obs[5]: hip1_speed，腿1髋关节角速度，reward_usable: true
- obs[6]: knee1_angle，腿1膝关节角度，reward_usable: true
- obs[7]: knee1_speed，腿1膝关节角速度，reward_usable: true
- obs[8]: leg1_contact，腿1触地标志，1.0 表示触地，0.0 表示未触地，reward_usable: true
- obs[9]: hip2_angle，腿2髋关节角度，reward_usable: true
- obs[10]: hip2_speed，腿2髋关节角速度，reward_usable: true
- obs[11]: knee2_angle，腿2膝关节角度，reward_usable: true
- obs[12]: knee2_speed，腿2膝关节角速度，reward_usable: true
- obs[13]: leg2_contact，腿2触地标志，1.0 表示触地，0.0 表示未触地，reward_usable: true
- obs[14]: lidar_0，前方地形第1个激光测距值，reward_usable: true
- obs[15]: lidar_1，前方地形第2个激光测距值，reward_usable: true
- obs[16]: lidar_2，前方地形第3个激光测距值，reward_usable: true
- obs[17]: lidar_3，前方地形第4个激光测距值，reward_usable: true
- obs[18]: lidar_4，前方地形第5个激光测距值，reward_usable: true
- obs[19]: lidar_5，前方地形第6个激光测距值，reward_usable: true
- obs[20]: lidar_6，前方地形第7个激光测距值，reward_usable: true
- obs[21]: lidar_7，前方地形第8个激光测距值，reward_usable: true
- obs[22]: lidar_8，前方地形第9个激光测距值，reward_usable: true
- obs[23]: lidar_9，前方地形第10个激光测距值，reward_usable: true

## 4. 动作空间 action_space

- type: Box
- shape: (4,)
- dtype: float32（连续关节力矩）
- action_dim 0: hip_torque_leg1，腿1髋关节力矩，连续范围 [-1.0, 1.0]
- action_dim 1: knee_torque_leg1，腿1膝关节力矩，连续范围 [-1.0, 1.0]
- action_dim 2: hip_torque_leg2，腿2髋关节力矩，连续范围 [-1.0, 1.0]
- action_dim 3: knee_torque_leg2，腿2膝关节力矩，连续范围 [-1.0, 1.0]

## 5. step 与终止条件分析

### 5.1 终止模式

- success-like termination: reached_end_of_terrain，表示到达地形终点。这很可能意味着完成了向前行走任务，但当前接口未暴露成功标志。
- failure-like termination: body_fallen_over，表示身体摔倒，任务失败。
- ambiguous termination: 在 step source 中 terminated = body_fallen_over or reached_end_of_terrain，未返回终止原因。因此无法从 info 区分是成功完成还是摔倒。
- truncation: step source 返回 truncated=False，未显式暴露时间截断或最大步数截断。外部环境可能存在时间限制，但当前接口不可见，不能假设。

### 5.2 success/failure 信号可用性

- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []（step source 返回 info={}，没有允许使用的 info 字段）
- forbidden_or_uncertain_info_fields: 任何 info 字段，例如 `success`、`failure`、`termination_reason`、`is_success` 等，均禁止使用；original_reward 也禁止使用。

## 6. reward 函数接口契约

函数签名：

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：

- obs
- action
- next_obs
- info 中明确允许的字段：当前环境 info 为空，因此没有可用 info 字段
- training_progress：只有 prompt 明确允许时才可用；本任务 spec 未明确允许，因此默认不使用

禁止使用：

- original_reward
- official_reward
- 未声明的 info 字段
- 未声明的 obs 切片或未知数组元素
- 假设存在 success、failure、termination_reason 等终止原因字段

## 7. 可用于奖励函数的信号

- position: 没有绝对 x 位置；可利用 next_obs[2] horizontal_velocity 与时间步长近似累计前进量，但不能依赖真实里程计。
- velocity: horizontal_velocity obs[2]，vertical_velocity obs[3]，关节角速度 obs[5]、obs[7]、obs[10]、obs[12]。
- orientation: hull_angle obs[0]，hull_angular_velocity obs[1]。
- contact: leg1_contact obs[8]，leg2_contact obs[13]。
- action/engine: action[0..3] 四维连续髋/膝力矩，可用作能量消耗估计与动作平滑惩罚。
- other: lidar_0..9 obs[14:24] 可感知前方地形起伏；关节角度 obs[4]、obs[6]、obs[9]、obs[11] 可用于关节姿态/限位判断。

## 8. 不确定或不可用的信号

- 绝对前进位移 / x 坐标：观察中不存在。
- explicit success / failure / termination_reason：info 为空，不可用。
- official_reward / original_reward：被 masked，禁止使用。
- 真实机械能量消耗：没有功率传感器，只能用 torque 与关节角速度近似。
- 足端高度或离地间隙：观察中不存在。
- 身体高度：无绝对高度，只有 vertical_velocity。
- 地形终点的距离或进度百分比：无法从观测直接获取。

## 9. 专家任务画像 expert_task_profile

```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: 2D bipedal torso with two articulated legs
  actuator_type: continuous torque-controlled hip and knee joints
  contact_structure: two leg/foot contact points on uneven terrain
primary_objectives:
  - maximize forward displacement across uneven terrain
  - maintain a stable bipedal gait without falling
secondary_objectives:
  - minimize energy consumption
  - coordinate smooth hip and knee torques
main_failure_risks:
  - falling over from torso pitch instability
  - stopping or oscillating in place without forward speed
  - high-frequency or high-torque inefficient gait
  - terrain-induced stumbling or getting stuck
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles

- role_id: forward_progress
  purpose: 鼓励持续向前移动，特别是保持正的水平速度。
  why_required: 主任务是“尽可能远、尽可能快地向前行走”，前进是核心目标。
  usable_signals:
    - obs[2] horizontal_velocity
    - next_obs[2] horizontal_velocity
  risks:
    - 只看速度可能被原地摆动或小步抖动利用。
    - 若过强，可能迫使 agent 使用不适合地形的激进步态。

- role_id: torso_stability
  purpose: 保持躯干接近竖直、抑制剧烈旋转，从而防止摔倒。
  why_required: 稳定躯干是产生稳定双足步态、避免终止的必要健康约束。
  usable_signals:
    - obs[0] hull_angle
    - obs[1] hull_angular_velocity
    - next_obs[0] hull_angle
    - next_obs[1] hull_angular_velocity
  risks:
    - 过强的稳定性惩罚会降低探索性，导致 agent 不敢动。
    - 若只惩罚角度而忽略角速度，可能仍在稳定姿态附近高频抖动。

### 10.2 条件职责 conditional_roles

- role_id: energy_regularization
  condition_to_use: 当基础前进步态已出现，或动作幅度、关节速度持续过大时加入；也可在后期作为效率优化项使用。
  usable_signals:
    - action[0..3]
    - obs[5]、obs[7]、obs[10]、obs[12] 关节角速度
  risks:
    - 早期加能量惩罚可能过度惩罚有效动作，抑制探索。
    - 若使用 torque 平方和，需要防止 agent 只选择小幅动作导致停止。

- role_id: contact_patterning
  condition_to_use: 当步态表现为单腿拖行、只靠一条腿接触或明显非交替步态时加入。
  usable_signals:
    - obs[8] leg1_contact
    - obs[13] leg2_contact
  risks:
    - 在崎岖地形上强制接触模式可能与地形冲突。
    - 该信号只能表达是否触地，无法表达足端高度或步态阶段。

### 10.3 慎用/禁用职责 avoid_roles

- role_id: sparse_goal_reward
  reason: 环境没有暴露 success 标志，也没有绝对位置，无法实现可靠的稀疏终点奖励。
  forbidden_or_missing_signals:
    - explicit success flag
    - absolute x position
    - termination_reason

- role_id: official_reward_matching
  reason: original_reward 和 official_reward 被 masked，禁止重建或拟合官方奖励。
  forbidden_or_missing_signals:
    - original_reward
    - official_reward
    - official_reward 内部结构信息

- role_id: foot_clearance_reward
  reason: 观测中没有足端高度或离地间隙，无法设计可靠的地面间隙奖励。
  forbidden_or_missing_signals:
    - foot height
    - foot clearance
    - foot z position

- role_id: lidar_obstacle_penalty
  reason: lidar 提供地形测距，但没有障碍物标签，也没有地形高度映射；直接作为障碍惩罚容易引入错误信号。仅可在极强先验下作为辅助信息使用。
  forbidden_or_missing_signals:
    - obstacle label
    - terrain height profile
    - lidar ground-truth interpretation

## 11. role_to_signal_mapping

| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress | obs[2], next_obs[2] | absolute x, terminal success | dense_state_signal, bounded_signal | 可以使用正水平速度主导，但要避免被高频摆动利用 |
| torso_stability | obs[0], obs[1], next_obs[0], next_obs[1] | fall flag | quadratic_penalty, bounded_signal | 惩罚躯干偏离竖直与角速度，核心稳定性项 |
| energy_regularization | action[0..3], obs[5], obs[7], obs[10], obs[12] | true mechanical power | quadratic_penalty | 能量消耗只能近似，建议条件使用 |
| contact_patterning | obs[8], obs[13] | foot height, contact phase | bounded_signal, sparse_event_signal | 若步态出现单腿拖行再使用 |
| sparse_goal_reward | none | success flag, absolute x, termination_reason | none | 禁用：缺乏成功信号 |
| official_reward_matching | none | original_reward, official_reward | none | 禁用：不得重建官方奖励 |
| foot_clearance_reward | none | foot clearance, foot z | none | 禁用：缺少足端高度 |
| lidar_obstacle_penalty | obs[14:24] | obstacle label, terrain height map | bounded_signal、quadratic_penalty | 慎用：lidar 语义不明确，不适合直接当障碍惩罚 |

## 12. 初始训练后应观察的 failure modes

| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 很早就摔倒，几乎不移动 | hull_angle 很大、hull_angular_velocity 高、水平速度低，episode 很快结束 | 加强 torso_stability，必要时降低 forward_progress 强度 |
| 原地站立或几乎不动 | horizontal_velocity 接近 0，关节速度低，episode 较长且不摔倒 | 提高 forward_progress 权重，降低稳定性惩罚 |
| 高能耗或高频抖动 | action 幅度接近 ±1，关节角速度很大，步态无明显前进 | 加入 energy_regularization 或动作平滑惩罚 |
| 前后摆动、平均前进速度低 | horizontal_velocity 正负交替，关节规律但无净前进 | 使用正水平速度主导的 forward_progress，惩罚负/零前进 |
| 单腿拖行或单腿触地 | 一个 contact 长时间为 1，另一个长时间为 0 | 条件性加入 contact_patterning，或惩罚长时间单侧接触 |
| 利用躯干起伏而非腿部前进 | vertical_velocity 波动大，horizontal_velocity 低 | 惩罚 vertical_velocity 或增加对水平速度的正向激励 |
| 地形适应差、频繁卡住 | lidar 分布变化少，水平速度骤降 | 可试探使用 lidar 辅助的平滑项，但避免地形障碍惩罚误导 |