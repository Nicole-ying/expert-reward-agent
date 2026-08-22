# Response Record

# 匿名环境理解卡片

## 1. 任务目标
控制一个二维双足身体在崎岖地形上稳定行走，尽可能走得更远、更快（最大化前进距离与速度），同时最小化关节力矩带来的能量消耗。身体必须保持直立，摔倒或到达地形终点会终止回合。核心是“快速前进”而不是“仅保持平衡”；能量消耗是重要的次要指标，但不能牺牲前进速度。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 核心目标是在二维地形上持续向前移动，速度/距离是主指标，双足结构仅是实现方式；没有多目标冲突达到无法区分主次的程度。

- dynamics_subtype: planar_bipedal_gait

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (24,)
- dtype: float32
- 各维含义列表：

| index | 字段名                      | 物理含义                         | reward_usable |
|-------|-----------------------------|----------------------------------|---------------|
| 0     | hull_angle                  | 躯干相对于竖直方向的倾角         | true          |
| 1     | hull_angular_velocity       | 躯干角速度                       | true          |
| 2     | horizontal_velocity         | 前进/后退线速度                  | true          |
| 3     | vertical_velocity           | 上下线速度                       | true          |
| 4     | hip1_angle                  | 腿1髋关节角度                    | true          |
| 5     | hip1_speed                  | 腿1髋关节角速度                  | true          |
| 6     | knee1_angle                 | 腿1膝关节角度                    | true          |
| 7     | knee1_speed                 | 腿1膝关节角速度                  | true          |
| 8     | leg1_contact                | 腿1触地标志（1.0=触地，0.0=离地） | true          |
| 9     | hip2_angle                  | 腿2髋关节角度                    | true          |
| 10    | hip2_speed                  | 腿2髋关节角速度                  | true          |
| 11    | knee2_angle                 | 腿2膝关节角度                    | true          |
| 12    | knee2_speed                 | 腿2膝关节角速度                  | true          |
| 13    | leg2_contact                | 腿2触地标志（1.0=触地，0.0=离地） | true          |
| 14    | lidar_0                     | 前方第1个激光雷达测距值          | true          |
| 15    | lidar_1                     | 前方第2个激光雷达测距值          | true          |
| 16    | lidar_2                     | 前方第3个激光雷达测距值          | true          |
| 17    | lidar_3                     | 前方第4个激光雷达测距值          | true          |
| 18    | lidar_4                     | 前方第5个激光雷达测距值          | true          |
| 19    | lidar_5                     | 前方第6个激光雷达测距值          | true          |
| 20    | lidar_6                     | 前方第7个激光雷达测距值          | true          |
| 21    | lidar_7                     | 前方第8个激光雷达测距值          | true          |
| 22    | lidar_8                     | 前方第9个激光雷达测距值          | true          |
| 23    | lidar_9                     | 前方第10个激光雷达测距值         | true          |

## 4. 动作空间 action_space
- type: Box (连续)
- shape: (4,)
- 范围: 每个动作 ∈ [-1.0, 1.0]（关节力矩缩放值）
- action_dim 0: hip_torque_leg1 – 腿1的髋关节扭矩
- action_dim 1: knee_torque_leg1 – 腿1的膝关节扭矩
- action_dim 2: hip_torque_leg2 – 腿2的髋关节扭矩
- action_dim 3: knee_torque_leg2 – 腿2的膝关节扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（到达地形尽头，可能视为成功完成前进任务）
- failure-like termination: body_fallen_over（身体摔倒，通常因倾角过大）
- ambiguous termination: 无
- truncation: 未提供，默认无时间截断或未明确

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false (info 为空，无法在奖励函数中获取到达终点标志)
- explicit_failure_flag_available: false (info 为空，无法直接获取摔倒标志)
- allowed_info_fields: 当前 info 字典为空，reward 函数内部不可使用任何 info 字段
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，包括任何隐含的 "success"/"failure"/"termination_reason" 等

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs (当前观测)
- action (本次执行的动作)
- next_obs (下一步的观测)
- training_progress (若 prompt 明确允许，否则禁用)

禁止使用：
- original_reward (已被掩码，严禁解构或逼近)
- 任何未出现在观测/动作声明中的变量
- info 中的任何字段（当前环境 info 为空）
- 任何隐含的“终止原因”或“episode成功标记”，因环境未显式提供

## 7. 可用于奖励函数的信号
- **位置/前进**:
  - next_obs[2] (horizontal_velocity)：直接反映前进速度大小与方向，是核心前进信号
  - obs[2] 用于变化量，但通常直接用 next_obs 的速度更直接
- **姿态/平衡**:
  - obs[0] (hull_angle)：偏离竖直的角度，应趋向0
  - obs[1] (hull_angular_velocity)：稳定状态应较小
  - obs[3] (vertical_velocity)：稳定行走时接近0（无跳跃）
- **接触/步态**:
  - obs[8], obs[13] (leg1_contact, leg2_contact)：可促进交替支撑或避免完全腾空
- **关节运动/能耗**:
  - action[0..3]：直接控制力矩，平方和代表能耗
  - obs[5,7,10,12]：关节角速度，可用于惩罚猛烈摆动（另一种能耗表征）
- **地形信息**:
  - obs[14..23] (lidar readings)：反映前方地形高度起伏，可用于期望步态调整，但直接用于奖励需谨慎
- **next_obs 对应量**也可用于构造变化量（如速度变化、倾角变化）

## 8. 不确定或不可用的信号
- 身体绝对位置/位移不显式提供，只能用速度估计前进距离（但可累积）
- 能量实际消耗值（功率）未给出，必须通过力矩动作近似
- 步态周期相位、双足支撑比等信息需额外推断，环境未提供
- episode 是否“成功”无法在奖励函数内部可靠获取（info 为空）
- 地形高度差的具体物理量未提供，仅雷达距离

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: rigid main body with two legs
  actuator_type: direct torque at hip and knee joints (2 joints per leg)
  contact_structure: point-foot ground contact with binary contact flags per leg
primary_objectives:
  - maximize forward distance (long-term integrated horizontal velocity)
  - maximize average forward speed during the episode
secondary_objectives:
  - minimize energy consumption (joint torque magnitude)
  - maintain upright torso orientation (small hull_angle)
  - maintain stable motion (low vertical/angular oscillations)
main_failure_risks:
  - falling over due to large hull tilt
  - inefficient limping or crawling instead of bipedal gait
  - excessive joint power causing early exhaustion or unrealistic motion
  - failure to adapt to uneven terrain (lidar information underutilized)
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_velocity_reward
  purpose: 鼓励身体尽快向前移动，将 horizontal_velocity 直接映射为奖励
  why_required: 任务核心是“走得又远又快”，没有前进信号将导致原地踏步或后退
  usable_signals: [next_obs[2] (horizontal_velocity)]
  risks: 若奖励过大会导致过度加速而摔跤，需配合平衡惩罚；注意 backward velocity 会得到负奖励，有利于抑制后退

- role_id: upright_stability_penalty
  purpose: 维持躯干直立，避免倾倒，抑制角速度和垂直跳动
  why_required: 摔倒直接导致回合失败，且长期倾斜会降低前进效率
  usable_signals: [obs[0] (hull_angle), obs[1] (hull_angular_velocity), obs[3] (vertical_velocity)]
  risks: 过于严苛的惩罚可能抑制必要的身体摆动，应使用平缓的二次型或温和的约束

- role_id: energy_efficiency_penalty
  purpose: 降低关节力矩幅值，体现能耗目标
  why_required: 明确的任务次级指标，防止无谓的暴力高力矩动作
  usable_signals: [action[0..3]]
  risks: 若惩罚过重会阻碍探索有力的步伐，初期训练需要适度放松

### 10.2 条件职责 conditional_roles
- role_id: alternating_contact_encouragement
  purpose: 鼓励双腿交替触地、避免同时离地或同时触地过久，促进自然的双足步态
  condition_to_use: 当 agent 已能稳定前行但步态出现单腿跳跃或拖腿时加入；早期强加可能干扰平衡
  usable_signals: [obs[8] (leg1_contact), obs[13] (leg2_contact)]
  risks: 过多人为步态约束可能与物理特性冲突，导致不稳定，效果需根据实验决定

### 10.3 慎用/禁用职责 avoid_roles
- role_id: episodic_success_bonus
  reason: info 为空，无法可靠获取“到达终点”事件，强行推测会引入噪声且违反信息约束
  forbidden_or_missing_signals: [any success flag in info]

- role_id: lidar_terrain_adaptation_reward
  reason: 雷达信息更适合作为状态的输入而非奖励项；在奖励中直接使用可能扭曲目标，且环境已提供 raw 观测即可用作特征
  forbidden_or_missing_signals: 不可直接当作奖励项，但可用作观测学习

- role_id: imitation_of_masked_reward
  reason: 明确禁止重建 official reward；任何试图逼近 masked reward 的做法都违反规范

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_velocity_reward | next_obs[2] | 无 | dense_state_signal, linear_reward, bounded_signal (clamp) | 通常直接返回 velocity 值，或 velocity*dt；负值自然惩罚后退 |
| upright_stability_penalty | obs[0], obs[1], obs[
