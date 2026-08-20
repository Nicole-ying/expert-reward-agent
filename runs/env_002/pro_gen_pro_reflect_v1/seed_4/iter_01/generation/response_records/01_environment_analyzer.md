# Response Record

# 匿名环境理解卡片

## 1. 任务目标
环境要求一个双腿身体在不平坦地形上尽可能快、尽可能远地向前行走，同时最小化能量消耗。主体必须协调两侧髋、膝关节力矩，产生稳定的双足步态。**摔倒将导致回合提前终止**，因此保持身体平衡是隐含的刚性前提。次要是节省能量、获得平滑的关节动作，但它们绝不能以牺牲前进和平衡为代价。不应混淆“生存”与“到达终点”：虽然走过整个地形也算终止，但核心奖励必须源于持续的向前位移，而不是一次性的终点信号。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 任务核心是“持续向前通过（不平坦）地形”，主体是双腿形态，动作是连续的关节力矩。目标为延长前进距离和速度，并附带能耗/平滑要求，完全匹配 locomotion_continuous_control 族。无多个冲突严重的等价目标，因此不做 multi_objective_task。

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32（假定）
- obs[0]: hull_angle — 身体相对竖直方向的倾斜角度，reward_usable: true（防摔倒）
- obs[1]: hull_angular_velocity — 身体角速度，reward_usable: true（姿态稳定性）
- obs[2]: horizontal_velocity — 前进/后退线速度，reward_usable: true（**主前进信号**）
- obs[3]: vertical_velocity — 上下线速度，reward_usable: true（可辅助检测跳跃或不稳定）
- obs[4]: hip1_angle — 腿1髋关节角度，reward_usable: true（姿态约束）
- obs[5]: hip1_speed — 腿1髋关节角速度，reward_usable: true（关节平滑性）
- obs[6]: knee1_angle — 腿1膝关节角度，reward_usable: true（姿态约束）
- obs[7]: knee1_speed — 腿1膝关节角速度，reward_usable: true（关节平滑性）
- obs[8]: leg1_contact — 腿1着地标志，1.0=接触，0.0=无接触，reward_usable: true（步态节律/接触）
- obs[9]: hip2_angle — 腿2髋关节角度，reward_usable: true（对称/姿态）
- obs[10]: hip2_speed — 腿2髋关节角速度，reward_usable: true（平滑）
- obs[11]: knee2_angle — 腿2膝关节角度，reward_usable: true（姿态）
- obs[12]: knee2_speed — 腿2膝关节角速度，reward_usable: true（平滑）
- obs[13]: leg2_contact — 腿2着地标志，reward_usable: true（步态节律/接触）
- obs[14..23]: lidar_0..9 — 10 路激光测距值（前方地形高度轮廓），reward_usable: 不确定，通常不直接用于简单前进奖励，可预留为高级地形感知，但基础奖励设计中不建议使用，防止引入不必要复杂度。

所有观测均可用，但是否用于奖励需依据职责。

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- 范围: 每维 [-1.0, 1.0]
- action_dim 0: hip_torque_leg1 — 腿1髋关节力矩
- action_dim 1: knee_torque_leg1 — 腿1膝关节力矩
- action_dim 2: hip_torque_leg2 — 腿2髋关节力矩
- action_dim 3: knee_torque_leg2 — 腿2膝关节力矩

说明：动作是连续力矩控制，无离散选项。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（走完全程）
- failure-like termination: body_fallen_over（身体摔倒）
- ambiguous termination: 无
- truncation: 未在描述中提及，假定有最大步数限制（通用环境），但不作为奖励信号依据。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 字典为空，无法直接获得 reached_end_of_terrain 标志）
- explicit_failure_flag_available: false（同样无法直接获得摔倒标志）
- allowed_info_fields: 无（当前 step 返回 info={}）
- forbidden_or_uncertain_info_fields: 所有的 info 字段均不存在，因此不能依赖任何外部成功/失败标记。奖励函数只能基于观测和动作构建生存和前进信号。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：当前观测数组（24维），全部维度都可读取，但应仅使用与职责对应的维度。
- action：当前动作数组（4维）。
- next_obs：下一时刻观测数组（24维），用于计算变化量（如速度、姿态变化）。
- info：当前环境返回的 info 字典（本环境为空，不允许使用）。
- training_progress：仅在 prompt 明确允许时才使用，本描述未要求使用，因此禁用。

禁止使用：
- original_reward：被屏蔽的官方奖励，严禁依赖。
- 未声明的 info 字段（无可用字段，因此 info 完全禁用）。
- 未声明的 obs 切片（虽然已声明全部，但 lidar 不推荐用作基础奖励）。
- 任何隐式的成功或失败标记（如`done`标志），因为接口中不提供。

## 7. 可用于奖励函数的信号
- position: 无直接位置观测，但可通过积分水平速度推算（但由于基础，不宜复杂，建议仅用瞬时速度）
- velocity: obs[2] horizontal_velocity（主信号），obs[3] vertical_velocity（辅助），obs[1] hull_angular_velocity（姿态稳定）
- orientation: obs[0] hull_angle（倾斜惩罚）
- contact: obs[8] leg1_contact, obs[13] leg2_contact（步态节律/双脚支撑）
- action/engine: action 四个维度的幅值（能耗惩罚），action 的前后变化量（平滑惩罚）
- other: lidar 测距阵列（obs[14:23]），不建议用于基础奖励；关节角度/角速度可用于姿态正则化。

## 8. 不确定或不可用的信号
- 终点成功标志：无直接信号，无法奖励“到达终点”。
- 水平位置：没有任何位置坐标，无法计算累积位移，只能依赖瞬时水平速度。
- 着地力的大小：只有二值接触标志（1.0/0.0），不能得出冲击力。
- 地形信息：lidar 存在，但地形崎岖程度难以量化到标量奖励中，容易引入噪声。
- 外部终止原因：无法区分摔倒与到达终点，因此不能根据回合结束给予稀疏奖励。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: bilateral_two_legs
  actuator_type: torque_motors_at_hip_and_knee
  contact_structure: bilateral_foot_ground_contact
primary_objectives:
  - maximize_forward_progress：最大化前向速度/位移
  - maintain_upright_balance：保持身体竖直（低倾斜角与角速度）
secondary_objectives:
  - minimize_energy：关节力矩尽量小
  - smooth_control：相邻动作变化小，减少震颤
main_failure_risks:
  - 身体倾斜过大、角速度发散造成摔倒
  - 步态不协调导致原地打转或后退
  - 过度追求低能耗而丧失前进动力
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress
  purpose: 鼓励持续前进，是核心优化方向。
  why_required: 任务明确要求“走得尽量远、尽量快”，没有该职责将导致智能体呆立或摔倒。
  usable_signals: [obs[2] horizontal_velocity, obs[3] vertical_velocity（可选抑制跳跃）]
  risks: 仅使用瞬时速度可能鼓励快速而忽略平衡，需配合生存职责。

- role_id: upright_balance_survival
  purpose: 防止摔倒，保持身体竖直。
  why_required: 摔倒即终止，影响学习效率；姿态稳定是有效前进的前提。
  usable_signals: [obs[0] hull_angle, obs[1] hull_angular_velocity]
  risks: 惩罚过重会惧动，导致智能体不敢迈步。

### 10.2 条件职责 conditional_roles
- role_id: energy_efficiency
  condition_to_use: 当智能体已能稳定前进一段时间（如训练中后期）或前进速度已较高时逐步引入，权重由低升高。
  usable_signals: [action 各维度的二次型或绝对值]
  risks: 如果过早或过重引入，会抑制探索，使智能体选择不动作以最小化惩罚。

- role_id: action_smoothness
  condition_to_use: 当观察到关节震颤、高频抖动，或关节速度变化剧烈时加入。
  usable_signals: [action_t - action_{t-1} 的范数]（需要在奖励函数内部缓存上一个动作）。
  risks: 过度平滑会降低响应速度，阻碍快速迈步。

- role_id: contact_gait_rhythm
  condition_to_use: 当步态明显异常（例如双脚长期同时离地，或长期单脚支撑导致旋转）时尝试加入，用于引导交替支撑模式。
  usable_signals: [obs[8] leg1_contact, obs[13] leg2_contact]
  risks: 设计不当会奖励奇怪的接触序列；该环境非必须，可列为可选。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: terrain_awareness_from_lidar
  reason: lidar 数据维度高，抽取可用标量奖励困难；本环境的基础前进无需预知地形起伏，避免复杂化。
  forbidden_or_missing_signals: 无直接可用的地形崎岖度或障碍物标签。
  
- role_id: terminal_sparse_success
  reason: 缺少 reached_end_of_terrain 显式标志，无法可靠奖励终点。
  forbidden_or_missing_signals: info 为空，终止原因不可知。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress | obs[2] horizontal_velocity, (可选)obs[3] vertical_velocity | 水平位置 | dense_state_signal, bounded_linear_reward, velocity_clip | 直接奖励正速度，可加竖直速度惩罚防跳跃 |
| upright_balance_survival | obs[0] hull_angle, obs[1] hull_angular_velocity | 无 | quadratic_penalty, bounded_signal (angle_limit) | 角度、角速度二次惩罚或设定生存阈值 |
| energy_efficiency | action[0..3] | 无 | quadratic_penalty, l2_norm, normalized_action_cost | 力矩平方和惩罚，系数逐步加大 |
| action_smoothness | (需要上一步动作缓存) action_diff | 无 | l2_norm(action_t - action_{t-1}) | 缓存动作向量，使用时注意初始化 |
| contact_gait_rhythm | obs[8] leg1_contact, obs[13] leg2_contact | 多接触力细节 | binary_contact_pattern_reward | 例如交替接触奖励，或惩罚双脚同时离地 |
| terminal_sparse_success | (None) | 终止分类标志 | — | 本环境不可用，禁用 |
| terrain_awareness_from_lidar | obs[14:23] | 无地形高度解释器 | — | 复杂、易过拟合，禁用 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 智能体不敢动或全零动作 | 回合极短，前进速度始终≈0，动作幅值极小 | 降低能量/平滑惩罚权重，确保前进奖励主导 |
| 快速摔倒（身体倾斜过大） | avg hull_angle 很大，或角速度均值高，episode length 小 | 增强 upright_balance 惩罚，或增加姿态角约束（如 >0.5 rad 严重惩罚） |
| 原地踏步、位移为零但动作不停 | horizontal_velocity 均值接近零但动作幅度不小 | 奖励只用正速度，可能需要增加“最小起身”或禁止后退；考虑使用接触节奏奖励促交替迈步 |
| 身体颤抖、动作高频震荡 | action 序列的高频分量明显，joint speed 波动大 | 引入 action_smoothness 职责，或对动作变化做惩罚 |
| 走一段后原地打转或后退 | horizontal_velocity 出现较大负值，hull_angle 周期性摆动 | 增加旋转/横向速度惩罚（若有对应观测
