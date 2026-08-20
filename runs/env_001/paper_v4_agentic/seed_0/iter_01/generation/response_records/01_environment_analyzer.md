# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本环境要求一个 2D 飞行器（带主引擎和两个姿态引擎）从视口顶部中心附近出发，以随机初始速度开始，尽快到达视口中心的着陆平台，并以极低的速度、稳定的姿态安全接触并停稳。主目标是**精确到达目标位置并稳定停靠**；次要目标是**快速完成**和**尽可能少地使用引擎推力**。任务的核心是导航与精确着陆，不应与纯粹的生存或持续前进任务混淆。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务的核心目标是到达一个指定的空间位置（中央平台），并稳定停止。其他目标（快速、省燃料）是典型的附属优化指标，不作为权重相当的冲突主目标，因此不属于 multi_objective_task。环境观测是连续的，动作是离散的，整体属于导航任务族。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推测）
- 各维含义：
  - obs[0]（x_position）：水平坐标，相对于目标着陆点的水平偏移，reward_usable: true
  - obs[1]（y_position）：垂直坐标，相对于平台高度的垂直偏移（平台高度处为 0），reward_usable: true
  - obs[2]（x_velocity）：水平线速度，reward_usable: true
  - obs[3]（y_velocity）：垂直线速度，reward_usable: true
  - obs[4]（body_angle）：机体朝向角（弧度），reward_usable: true
  - obs[5]（angular_velocity）：角速度，reward_usable: true
  - obs[6]（left_support_contact）：左侧支撑杆与地面/平台的接触标志（1.0 表示接触），reward_usable: true
  - obs[7]（right_support_contact）：右侧支撑杆接触标志（1.0 表示接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作列表：
  - action 0: no_engine，不开启任何引擎
  - action 1: left_orientation_engine，开启左姿态引擎（产生角加速度，可能向左旋转）
  - action 2: main_engine，开启主引擎（产生向上的推力）
  - action 3: right_orientation_engine，开启右姿态引擎（产生相反方向的角加速度）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  机体在平台上稳定停靠，触发 **body_not_awake_or_settled**（休眠/静止）。此时通常同时满足：两腿接触 flag 均为 1，x_position 和 y_position 接近 0，线速度与角速度均很小，且未发生 crash 或出界。
- failure-like termination:  
  - **horizontal_position_outside_viewport**：机体水平飞出视口边界，直接失败。  
  - **crash_or_body_contact**（非着陆接触）：机体以过大速度、过大角度或接触到非平台区域（如地面以外）触发终止，属于失败。需要结合接触标志和速度判断。
- ambiguous termination:  
  **crash_or_body_contact** 在某些情况下也可能是成功着陆，因为着陆时也会发生身体接触并可能触发该条件。需要进一步通过双腿是否都接触、速度是否低、是否在目标附近来区分。  
  **body_not_awake_or_settled** 也可能是碰撞后卡住不动导致的静止，但碰撞后通常接触标志不会全为 1 且位置会偏离目标，因此可通过位置与接触标志排除模糊性。
- truncation:  
  无显式截断（源码中返回的 truncated 恒为 False）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 始终为空字典）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用
- 推断成功/失败的间接路径（derived_possible）：
  - **成功**：episode 终止且满足以下条件 → (left_support_contact == 1.0) and (right_support_contact == 1.0) and (|x_position| 很小) and (|y_position| 很小) and (|body_angle| 很小) and 线速度/角速度很低。  
  - **失败（crash）**：episode 终止但上述条件不成立（例如双腿未同时接触、位置大幅偏离、角度或速度很大）。  
  - **出界**：可通过终止时 |x_position| 显著大于视口半宽推断。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs（当前步观测）
- action（当前步选择的动作）
- next_obs（执行动作后的观测，在终止步也有效）
- info 中已声明可用的字段（当前无可用字段）

禁止使用：
- original_reward（原官方奖励）
- 任何 info 字段（因未声明，且环境返回空字典）
- training_progress（本 prompt 未明确许可使用）

明确说明：
- 即使环境终止，next_obs 仍可作为奖励计算的依据，其接触标志、位置、速度可用来判断着陆成功或失败，并给予相应奖惩。
- 动作信息只能用于衡量能耗、动作连续性等，不能直接奖励选择某个动作本身。

## 7. 可用于奖励函数的信号
- position: x_position, y_position（均可直接获得，表示相对于目标的位置）
- velocity: x_velocity, y_velocity, angular_velocity
- orientation: body_angle
- contact: left_support_contact, right_support_contact
- action/engine: 离散动作 id 可映射为推力状态（无推力、左旋、主推、右旋）；可用于估计燃料消耗、避免无用点火
- other: 从上述信号可派生的距离（euclidean distance, |x|+|y| 等）、接近速度、朝向对齐程度、双腿是否均接触、是否在目标附近等

所有信号均为可直接从 obs 或 next_obs 读取的数值，无量纲但具有物理意义。

## 8. 不确定或不可用的信号
- 明确的 success/failure 布尔标志：不可用，必须从 next_obs 推断。
- 环境的时间戳或剩余步数：不可用。
- 平台中心的世界绝对坐标：不可用，需依赖相对位置。
- 引擎的剩余燃料量：不可用。
- 任何源自 info 字段的辅助信息：均不可用。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2d_rigid_body_with_two_legs
  actuator_type: one_main_engine_and_two_orientation_engines
  contact_structure: two_leg_support
primary_objectives:
  - reach_target_pad (minimize horizontal and vertical offset)
  - stabilize_at_target (near-zero velocity, near-zero angle)
secondary_objectives:
  - minimize_time_to_land (fast terminal approach once stable)
  - minimize_engine_usage (energy efficiency)
main_failure_risks:
  - crashing into ground or non-pad surface
  - flying out of horizontal viewport bounds
  - landing too fast or at large angle causing leg break
  - only one leg touching or tipping over
  - never approaching target (hovering or drifting away)
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: distance_to_target  
  purpose: 奖励飞行器逐渐靠近目标平台（减小位置偏移）  
  why_required: 核心目标是导航至目标，缺失此职责 agent 可能不向目标移动  
  usable_signals: [x_position, y_position]  
  risks: 仅奖励位置靠近可能忽略速度控制，导致 agent 快速撞向平台

- role_id: landing_softness  
  purpose: 在接近目标时要求低速、小角度、双腿同时接触，以安全着陆  
  why_required: 必须将碰撞式的抵达转变为受控着陆，否则任务从未真正成功  
  usable_signals: [x_velocity, y_velocity, angular_velocity, body_angle, left_support_contact, right_support_contact, x_position, y_position]  
  risks: 若全局施加可能减缓学习前期接近，建议在距离较近时逐步激活或作为最终步奖励

- role_id: crash_prevention  
  purpose: 惩罚撞毁或飞出边界的工况  
  why_required: 安全是任务强约束，失败直接终止  
  usable_signals: 可从 next_obs 的接触标志、速度、位置组合推断 crash；飞出边界可由 |x_position| 过大推断  
  risks: 若推断逻辑不够准确，可能漏判或误判，需要仔细设计触发条件

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency  
  purpose: 鼓励用尽可能少的引擎推力完成着陆  
  condition_to_use: 仅在与目标距离较近且姿态较稳时施加，避免在早期强迫 Agent 不推力而无法学习导航  
  usable_signals: [action] （通过动作映射推力消耗量）  
  risks: 过早或不恰当
