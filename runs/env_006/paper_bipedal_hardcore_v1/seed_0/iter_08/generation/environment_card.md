# 匿名环境理解卡片

## 1. 任务目标
控制一个两足机器人在崎岖地面上尽可能远地前进，同时尽量高效（少用力矩）并保持稳定（避免摔倒）。  
辅助目标包括：利用10维激光雷达感知前方地形高度，提前调整步态；到达地形尽头时终止。  
**不应混淆**：该任务不是纯粹平衡任务，也不是精确到达某个目标点；主目标是在摔倒之前跑得尽可能远，能耗和姿态为次要优化。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 核心是通过平坦/不平坦地面的速度与距离推进，即使有效率与平衡要求，但它们从属于前进这一主目标；动作空间为连续扭矩，观察包含速度和关节信息，符合行走类连续控制。

## 3. 观察空间 observation_space
- type: Box  
- shape: [24]  
- dtype: 推测为 float32（匿名描述未显式说明）  
各维度含义与奖励可用性：
- obs[0]: hull_angle，本体躯干倾角，**reward_usable: true**（用于检测摔倒/姿态偏离）
- obs[1]: hull_angular_velocity，躯干角速度，reward_usable: true（姿态变化速率）
- obs[2]: horizontal_speed，质心水平速度，**reward_usable: true**（前进距离的主信号）
- obs[3]: vertical_speed，质心垂直速度，reward_usable: true（辅助判断起跳/坠落）
- obs[4]: joint_0_angle，髋关节1角度，reward_usable: true（可结合动作做能耗/姿态约束）
- obs[5]: joint_0_speed，髋关节1角速度，reward_usable: true
- obs[6]: joint_1_angle，膝关节1角度，reward_usable: true
- obs[7]: joint_1_speed，膝关节1角速度，reward_usable: true
- obs[8]: joint_2_angle，髋关节2角度，reward_usable: true
- obs[9]: joint_2_speed，髋关节2角速度，reward_usable: true
- obs[10]: joint_3_angle，膝关节2角度，reward_usable: true
- obs[11]: joint_3_speed，膝关节2角速度，reward_usable: true
- obs[12]: leg_1_ground_contact，腿1接触地面指示（0/1），**reward_usable: true**（用于步态/触地奖励）
- obs[13]: leg_2_ground_contact，腿2接触地面指示（0/1），reward_usable: true
- obs[14]~[23]: lidar_1 ~ lidar_10，前方10个测距传感器，测量地形高度，**reward_usable: false**（地形感知信号，但无法直接映射为奖励；只能由agent在策略中使用，reward function不应直接依赖，除非通过间接方式推导惩罚，但不适合作为奖励项）

## 4. 动作空间 action_space
- type: Box  
- shape: [4]  
- bounds: [-1.0, 1.0]  
维度含义：
- action_dim 0: hip_1_torque，左（或右）髋关节扭矩
- action_dim 1: knee_1_torque，同侧膝关节扭矩
- action_dim 2: hip_2_torque，另一侧髋关节扭矩
- action_dim 3: knee_2_torque，另一侧膝关节扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**：到达地形尽头（reached_end_of_terrain），此时终止且无失败标记，可视为一种成功完成。
- **failure-like termination**：身体摔倒（body_fallen_over），显然不希望发生。
- **ambiguous termination**：两种终止均通过 terminated=True 返回，且 info 为空，因此无法直接从终止标志区分成功或失败。
- **truncation**：不存在显式截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: “info 为空，无可用字段”
- forbidden_or_uncertain_info_fields: 所有 info 字段均禁止使用
- **推断成功/失败的可能路径**：
  - 摔倒可通过 hull_angle 绝对值突然增大、vertical_speed 剧烈下降、leg contact 消失等综合判断，视为 derived_possible。
  - 到达尽头可通过横向速度持续且 episode 终止时 hull_angle 未超过阈值，再结合 legs 接触等推断，但仍不确定。实际 reward 设计中不应依赖成功终止信号。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
- 允许使用：obs（当前观察），action（执行的动作），next_obs（下一状态向量），以及 info 中明确声明的字段（当前无），training_progress（仅在 prompt 明确允许时使用）。
- 禁止使用：original_reward（被屏蔽）、official_reward、未声明的 obs 切片（如不能使用 lidar 作奖赏输入，除非下游明确允许“观察所有维度”但任务语义上 lidar 不应作为奖励信号）、以及任意未声明的 info 字段。

## 7. 可用于奖励函数的信号
- **position/velocity**：horizontal_speed（可直接用于正向进度奖励），vertical_speed（用于辅助惩罚异常弹跳或坠落），hull_angular_velocity（用于惩罚剧烈摇晃）。
- **orientation**：hull_angle，直接反映躯干倾斜，可惩罚非直立姿态。
- **contact**：leg_1_ground_contact，leg_2_ground_contact，可用于步态奖励（如双脚交替触地、避免无接触时间过长）。
- **action/engine**：action 本身（[hip_1_torque, knee_1_torque, hip_2_torque, knee_2_torque]），可用于能耗惩罚（平方或绝对值）。
- **other**：关节角度/角速度可与动作结合做平滑度或正常运动范围约束。
- **derived_possible**：hull_angle 阈值（如 > 0.8 rad）可推断摔倒，用作终止前的高惩罚；horizontal_speed 在某阈值以上且 episode 终止可尝试作为到达终点的奖励，但不稳定。

## 8. 不确定或不可用的信号
- 明确不可用：lidar 数据（10维），不能作为奖励信号，因为其作用是辅助决策而非直接评定任务完成度。
- 无直接可用：地形信息、终点距离等。
- 不可靠：info 为空，无法获取距离或成功标志。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: biped (two legs, rigid hull)
  actuator_type: torque-controlled hip and knee joints (4 DOF)
  contact_structure: two foot contact points with ground, capable of sensing contact
primary_objectives:
  - maximize forward distance traveled before episode ends
secondary_objectives:
  - avoid falling over (maintain upright posture)
  - minimize joint torque usage (energy efficiency)
main_failure_risks:
  - falling due to steep terrain or incorrect step timing
  - getting stuck in low-speed, high-energy gaits
  - overshooting joints leading to loss of balance
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_velocity_reward
  purpose: 鼓励 agents 保持正向水平速度前进
  why_required: 核心目标是“走得更远”，速度是最直接的代理
  usable_signals: [horizontal_speed]
  risks: 若权重过高，agent 可能牺牲稳定性以追求速度，导致频繁摔倒

- role_id: upright_penalty
  purpose: 惩罚躯干严重倾斜，降低摔倒概率
  why_required: 摔倒导致提前终止，且多数摔倒前 hull_angle 会增大
  usable_signals: [hull_angle, hull_angular_velocity]
  risks: 若惩罚太强，可能抑制 agent 学习必要的步态摆动，导致行走僵硬

- role_id: energy_penalty
  purpose: 降低关节扭矩总和，提升能效
  why_required: 任务明确希望“最小化不必要的关节力矩”
  usable_signals: [action]
  risks: 过于严厉的惩罚会降低探索所需的机动性，导致 agent 不敢用力迈步

### 10.2 条件职责 conditional_roles
- role_id: foot_contact_pattern_reward
  condition_to_use: 当观察到双脚触地状态频繁切换且能走出稳定步态时，可作为辅助
  usable_signals: [leg_1_ground_contact, leg_2_ground_contact]
  risks: 设计不当可能奖励原地踏步；需结合前进速度判断

- role_id: alive_bonus
  condition_to_use: 每个时间步存活（未摔倒）给予小额奖励，加速早期探索，但必须在摔倒时扣除所有后清零，注意勿与速度奖励叠加导致刷步数
  usable_signals: [hull_angle 未超过阈值、info 为空但可用 survived 推断]
  risks: 若结算不当会导致 agent 故意不前进以赚取 alive 奖励

### 10.3 慎用/禁用职责 avoid_roles
- role_id: lidar_based_terrain_reward
  reason: LIDAR 测量仅在策略输入中可用，reward 函数无法直接从地形高度评估任务完成度，且语义上不应将感知信号作为奖赏，易导致不可解释的奖励变化。
  forbidden_or_missing_signals: [lidar_* 数据不应参与 reward 计算]

- role_id: goal_reaching_reward
  reason: 虽然存在到达终点的成功终止，但无显式位置或距离信号，info 为空，无法可靠检测何时成功，盲目假设到达终点会导致错误正反馈。
  forbidden_or_missing_signals: [缺少 x 坐标、终点距离或终点触发标志]

- role_id: smooth_gait_reward
  reason: 可通过关节速度和加速度的连续性实现，但当前信号（关节角速度、扭矩）足够，但需谨慎以免过度平滑导致奇怪步态；暂不列为必须。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_velocity_reward | horizontal_speed | none | dense_state_signal (linear) | 可用 clamped_linear 防过大 |
| upright_penalty | hull_angle, hull_angular_velocity | none | quadratic_penalty, bounded_signal | 当角度超过阈值时惩罚，亦可对变化率惩罚 |
| energy_penalty | action | none | L2_norm_penalty | 求和平方或绝对值 |
| foot_contact_pattern_reward | leg_1_ground_contact, leg_2_ground_contact | none | event_detection (toggle) | 需结合 horizontal_speed 生成奖励，不宜单独使用 |
| alive_bonus | hull_angle (阈值)、episode未终止 | explicit terminated cause | constant_per_step | 仅当训练早期且性能很差时使用，注意终止时回退 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 频繁摔倒（hull_angle过大） | 平均 episode 长度短，hull_angle 频繁超限 | 增强 upright_penalty 并适当降低速度奖励，也可能需改变步态教学 |
| 缓慢挪动、能耗高 | horizontal_speed 低而 energy_penalty 高 | 降低 energy_penalty 权重、增加速度奖励权重，或引入步态密度奖励 |
| 原地踏步以赚取 alive 奖励 | horizontal_speed 极低但 episode 持续性正常 | 移除 alive_bonus 或改为步数惩罚，确保速度主奖励足够 |
| 无法泛化至复杂地形 | 平坦地形成功，但出现台阶或坑洼时摔倒 | 检查是否充分使用了 LIDAR 输入（策略端），并考虑在 reward 中加强姿势稳定性惩罚 |