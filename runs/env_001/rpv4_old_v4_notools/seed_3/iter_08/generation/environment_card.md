# 匿名环境理解卡片

## 1. 任务目标
主体是一个 2D 简化的飞行器/着陆器，起始于视口顶部中央附近，并被赋予随机初始速度（随机力）。核心目标是**尽快到达画面中央的目标平台并稳定停靠**。过程中需尽可能少用推力（燃料优化），并保持稳定的姿态角度与低速着陆。该任务属于典型的**目标导向到达 + 软着陆控制**，不包含多目标冲突（燃料、速度、姿态均为从属优化）。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务的核心是到达指定目标位置（中央平台）并停靠，附属目标为节省燃料、快速到达、姿态平稳，这些附属项不构成权重相当的冲突目标。动力学子类型进一步符合 goal_approach_and_soft_contact（接近目标并低速、稳定接触/停靠）。

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: float32 (推测)  
- obs[0]: x_position – 相对目标平台的水平坐标，reward_usable: true  
- obs[1]: y_position – 相对平台高度的垂直坐标，reward_usable: true  
- obs[2]: x_velocity – 水平线速度，reward_usable: true  
- obs[3]: y_velocity – 垂直线速度，reward_usable: true  
- obs[4]: body_angle – 身体朝向角度，reward_usable: true  
- obs[5]: angular_velocity – 角速度，reward_usable: true  
- obs[6]: left_support_contact – 左支撑接触标志（1.0 表示接触），reward_usable: true  
- obs[7]: right_support_contact – 右支撑接触标志（1.0 表示接触），reward_usable: true  

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- action 0: no_engine – 不点燃任何引擎  
- action 1: left_orientation_engine – 点燃一个姿态引擎（顺时针或逆时针？推测会使身体旋转一个方向）  
- action 2: main_engine – 点燃主引擎（产生向上的推力）  
- action 3: right_orientation_engine – 点燃另一个姿态引擎（产生与 1 相反的旋转）  

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled – 身体进入“休眠”状态，通常表示已稳定停靠在平台且速度极低，可视为成功着陆。
- failure-like termination: crash_or_body_contact – 可能是身体撞击地面/山体/边界，或任何不应发生的接触；horizontal_position_outside_viewport – 水平位置超出视口范围。
- ambiguous termination: 以上两种 fate 的判断依赖环境内部物理逻辑，但从任务描述可以推断成功率由 body_not_awake_or_settled 定义，其余均为失败。
- truncation: 无明确最大步数截断，但一般由环境外部控制。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空，没有 success 字段）
- explicit_failure_flag_available: false
- allowed_info_fields: {} （空字典，无任何额外字段）
- forbidden_or_uncertain_info_fields: info 中没有任何字段可用，因此不能依赖 info 去辨别成功/失败。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：当前观测（8维）
- action：当前步选择的动作（0-3）
- next_obs：下一步观测（8维）
- info：当前空字典，无可用字段
- training_progress：当前训练进度（0.0~1.0），仅当 prompt 明确允许时使用（此处未明确要求，故谨慎使用）

禁止使用：
- original_reward：官方奖励被遮蔽，不得使用
- 任何未在观测中声明的变量或 info 字段
- 任何基于官方奖励的重建或近似

## 7. 可用于奖励函数的信号
- position: obs[0] (x), obs[1] (y)，及其在 next_obs 中的对应值
- velocity: obs[2], obs[3]，next_obs[2], next_obs[3]
- orientation: obs[4]（角度）, obs[5]（角速度），next_obs 中亦然
- contact: obs[6]（左接触）, obs[7]（右接触），next_obs[6], next_obs[7]
- action/engine: 动作值（0:无, 1:左姿态, 2:主引擎, 3:右姿态），可用于能耗或推力惩罚
- other: 无其他可用信号

## 8. 不确定或不可用的信号
- 成功标志：没有显式 success/failure 标志，不能直接奖励“胜利”
- 平台中心绝对坐标：只有相对位置，无绝对目标坐标
- 燃料剩余/推力强度：未知，仅知道有 3 种引擎动作
- 碰撞细节：crash_or_body_contact 是什么物体，不可知

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body (2D)
  actuator_type: single_main_thruster + two_side_orientation_thrusters
  contact_structure: two_leg_support_contacts
primary_objectives:
  - arrive_at_target_pad_center
  - settle_with_low_velocity
secondary_objectives:
  - minimize_main_engine_usage
  - minimize_orientation_angle_and_angular_velocity
  - (implied) reach as fast as possible
main_failure_risks:
  - crashing_into_ground_or_boundary
  - overuse_of_thrust_leading_to_high_velocity_crash
  - losing_control_of_orientation_during_descent
  - failing_to_align_contacts_on_pad
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity  
  purpose: 鼓励主体向目标平台中心靠近。减少水平与垂直距离。  
  why_required: 任务是到达目标，必须驱动位置接近。  
  usable_signals: obs[0], obs[1] 或 next_obs[0], next_obs[1]  
  risks: 若仅依赖距离奖励，可能忽略减速和软着陆，导致高速坠毁。

- role_id: soft_contact_success  
  purpose: 奖励成功稳定停靠：双腿接触 + 速度极低。是实现“settled”终止的关键信号。  
  why_required: 任务明确要求停靠，仅靠距离到达无法保证任务完成。  
  usable_signals: next_obs[6], next_obs[7] (接触), next_obs[2], next_obs[3] (速度)  
  risks: 稀疏，早期难以获得；需要引入渐进奖励（如接近目标时接触奖励权重加大）。

### 10.2 条件职责 conditional_roles
- role_id: energy_efficiency  
  purpose: 惩罚不必要的燃料消耗，尤其避免无意义的主引擎持续喷射。  
  condition_to_use: 当 agent 尚未接近目标平台（例如距离较远）时，可使用轻微惩罚；当接近时，可加大惩罚以促进精细控制。  
  usable_signals: action（是否为 2）  
  risks: 过强惩罚会抑制探索，导致无法抵达；宜配合距离衰减权重。

- role_id: stability_penalty  
  purpose: 惩罚大的身体角度与角速度，促使姿态平稳着陆。  
  condition_to_use: 全局施加小权重；可在接近地面时（y 较小）增大权重，保证着陆阶段姿态稳定。  
  usable_signals: obs[4], obs[5] 或 next_obs[4], next_obs[5]  
  risks: 过大权重会妨碍必要的转向机动；需平衡。

- role_id: fast_approach  
  purpose: 奖励朝向目标的速度投影，满足“尽快到达”的从属目标。  
  condition_to_use: 只在未靠近地面且无接触风险时激活，例如 y > 某个阈值且速度未指向地面碰撞方向。  
  usable_signals: 利用当前位置方向与速度计算投影。  
  risks: 极易导致高速坠毁或 overshoot，必须配合减速约束。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: survival_balance_coherence  
  reason: 本任务无平衡存活需求，主体不会因倾斜而直接失败（除非撞击），因此不适用保持存活类职责。  
- role_id: direct_success_reward  
  reason: info 无成功标志，不能直接奖励成功。  
- role_id: action_smoothing  
  reason: 离散动作空间未提供平滑切换要求，且惩罚动作变化可能阻碍快速响应，不建议使用。  

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | next_obs[0], next_obs[1] | – | distance_to_target, dense_state_signal, bounded_signal | 可用欧氏距离或 -distance 作为密集奖励 |
| soft_contact_success | next_obs[6], next_obs[7], next_obs[2], next_obs[3] | info.success | product_of_contacts, threshold_logical, scaled_bonus | 当双腿接触且速度接近零时给予较大奖励 |
| energy_efficiency | action (==2) | fuel_consumption_value | binary_penalty, scaled_penalty | 每次点燃主引擎施加负奖励 |
| stability_penalty | next_obs[4], next_obs[5] | – | absolute_value, quadratic_penalty | 惩罚 abs(angle) 和 abs(angular_vel) |
| fast_approach | velocity projection toward target (using next_obs[0:3]) | – | projection_reward, conditional_gating | 需结合高度阈值与减速约束使用 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 无法到达目标区域 | 平均最终距离大，奖励不上升 | 调整 goal_proximity 权重，或引入潜力塑形 |
| 高速坠毁/出界 | 最终速度高，平局终止原因多为 crash | 强化 soft_contact_success 和 stability_penalty，减小 fast_approach 权重 |
| 长时间盘旋不降落 | y 速度近零，x 抖动 | 增大接近时的 forward progress 奖励（谨慎），或利用接触标志触发 bonus |
| 过度使用主引擎 | 动作中 2 比例过高 | 增加 energy_efficiency 权重或引入渐进惩罚 |
| 姿态失控（角速度大） | 角度和角速度波动大 | 增大 stability_penalty 权重 |
| 单腿接触反复弹跳 | 接触标志频繁切换 | 奖励双腿同时接触且持续接触的状态，可加入持续时间阈值 |