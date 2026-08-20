# Response Record

# 匿名环境理解卡片

## 1. 任务目标
任务主体是一个2D飞行器/着陆器，初始从视野上方中央区域出发并带有随机初速。  
**主目标**：尽快飞到中央目标平台并稳定停靠（settle），要求位置精准、速度近乎为零、姿态稳定，并且两个支撑脚均安全接触地面。  
**次目标**：在此过程中尽可能减少引擎使用（节省燃料），并隐含要求快速完成（时间效率）。  
**不应混淆的目标**：燃料效率与快速到达是一对次要权衡，但不应喧宾夺主；安全平稳着陆优先于速度与能耗。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是到达并稳定于一个固定的空间位置（目标平台），姿态和速度要求都是为实现“到达并安全停靠”服务，属于典型的导航目标到达类，只是附带了软着陆的动力学约束。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32（推断）  
- 各维度定义与可用性：
  - obs[0]: x_position，水平相对目标平台的坐标，reward_usable: true  
  - obs[1]: y_position，垂直相对平台高度的坐标，reward_usable: true  
  - obs[2]: x_velocity，水平线速度，reward_usable: true  
  - obs[3]: y_velocity，垂直线速度，reward_usable: true  
  - obs[4]: body_angle，机体倾角，reward_usable: true  
  - obs[5]: angular_velocity，角速度，reward_usable: true  
  - obs[6]: left_support_contact，左支撑脚接地标志（1.0/0.0），reward_usable: true  
  - obs[7]: right_support_contact，右支撑脚接地标志（1.0/0.0），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 动作含义：
  - action 0: no_engine (不做任何推进)  
  - action 1: left_orientation_engine (启动左侧姿态调整引擎)  
  - action 2: main_engine (启动主引擎，提供主要推力)  
  - action 3: right_orientation_engine (启动右侧姿态调整引擎)

## 5. step 与终止条件分析
### 5.1 终止模式
- **crash_or_body_contact**：出现碰撞或非预期的机体接触（典型失败）。  
- **horizontal_position_outside_viewport**：水平位置出界（失败）。  
- **body_not_awake_or_settled**：机体不再活跃/已稳定停靠。此条件**可能代表成功**（如果在目标位置附近、速度低、双足接地），但也可能因为其他原因“静默”失败（如卡在视口外、姿态错误而静止）。需结合状态判断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（info 返回 {}，不提供任何额外字段）  
- forbidden_or_uncertain_info_fields: info 内所有假设字段（如 “success”, “failure”, “termination_reason” 等）均不存在且不可用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
**允许使用：**
- obs（当前观测）
- action（本次动作）
- next_obs（下一步观测）
- info（当前已知为空，不可使用任何字段）
- training_progress（仅在任务明确允许时使用）

**禁止使用：**
- original_reward 或任何形式的官方奖励
- 未声明的 info 字段
- 未在观测空间中声明的维度或假设的隐藏状态

## 7. 可用于奖励函数的信号
全部来自观测：
- **位置**：x_position, y_position（目标参考零点）  
- **速度**：x_velocity, y_velocity  
- **姿态与角速度**：body_angle, angular_velocity  
- **接触信号**：left_support_contact, right_support_contact  
- **动作**：action（离散值 0-3）

## 8. 不确定或不可用的信号
- **显式成功/失败标志**：无  
- **时间步数 / elapsed steps**：未提供，不可直接奖励“快速”  
- **燃料消耗量**：未提供，不能直接奖励低能耗，只能通过动作选择间接引导  
- **接触类型细节**：仅有支撑脚接触标志，无法区分安全着陆与碰撞  
- **目标距离的精确度量**：可通过位置计算，但官方未给出额外信号

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: planar_lander  # 2D着陆器外形
  actuator_type: discrete_thrusters  # 一个主推力和两个姿态调整推力
  contact_structure: two_leg_support_with_contact_flags
primary_objectives:
  - 到达目标位置 (x=0, y=0)
  - 稳定停靠（速度≈0，姿态稳定）
  - 双支撑脚同时接地
secondary_objectives:
  - 节省引擎使用（燃料效率）
  - 尽快速达（时间效率）
main_failure_risks:
  - 撞击目标区域（高速/错误姿态导致 crash）
  - 水平漂移出界
  - 到达目标但姿态倾斜导致翻倒或无法稳定
  - 提前“假静默”而未正确着陆
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: target_approach**  
  purpose: 鼓励减小与目标位置的欧氏距离  
  why_required: 核心到达任务  
  usable_signals: [x_position, y_position]  
  risks: 单纯距离奖励会导致高速撞击，需与速度控制耦合。

- **role_id: soft_landing_speed**  
  purpose: 在接近目标时抑制速度，尤其是垂直接触速度  
  why_required: 实现“settle”的低速接触要求  
  usable_signals: [x_velocity, y_velocity, x_position, y_position]（联合使用，按距离加权）  
  risks: 早期过度惩罚正常机动速度，应渐进激活。

- **role_id: stable_orientation**  
  purpose: 维持机体接近竖直，抑制角速度  
  why_required: 安全接地与姿态稳定  
  usable_signals: [body_angle, angular_velocity]  
  risks: 可能限制必要的机动，需在着陆末段加强。

- **role_id: contact_completion**  
  purpose: 奖励双足同时接地  
  why_required: 标志着陆成功、稳定停靠  
  usable_signals: [left_support_contact, right_support_contact]  
  risks: 过早奖励地面接触可能导致在平台外着陆，需结合位置条件使用。

### 10.2 条件职责 conditional_roles
- **role_id: fuel_efficiency**  
  condition_to_use: 当 agent 已经接近目标且速度较低时，鼓励选择 no_engine 动作  
  usable_signals: [action, x_position, y_position, x_velocity, y_velocity]  
  risks: 若条件过严，可能阻碍必要的机动或主引擎减速。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: time_minimization**  
  reason: 缺少步数或计时信号，无法准确奖励快速到达；强行奖励可能导致危险加速  
  forbidden_or_missing_signals: [elapsed_steps, episode_length]（环境未提供）

- **role_id: explicit_collision_penalty**  
  reason: 无碰撞细节信号，终止环境时已触发失败，不适合在奖励函数中重复惩罚  
  forbidden_or_missing_signals: [collision_intensity, impact_force]

## 11. role_to_signal_mapping
| role_id              | usable signals                                           | missing signals | candidate formula operators        | notes                                                                 |
|----------------------|----------------------------------------------------------|-----------------|------------------------------------|-----------------------------------------------------------------------|
| target_approach      | x_position, y_position                                   | -               | distance_to_target, negative_distance | 可采用平滑有界惩罚，如 -sqrt(x²+y²) 或 -tanh(distance)              |
| soft_landing_speed   | x_velocity, y_velocity, x_position, y_position           | -               | speed_magnitude, distance_gated_penalty | 速度惩罚可用距离门控：距离小权重增大，或使用 -speed*exp(-dist²) 等形式 |
| stable_orientation   | body_angle, angular_velocity                             | -               | orientation_error, angular_speed_penalty | 可使用角度偏差二次惩罚和角速度二次惩罚                               |
| contact_completion   | left_support_contact, right_support_contact, x_position, y_position | -               | logical_and, dense_contact_reward       | 仅当接近目标且速度低时奖励双接触，避免过早奖励                         |
| fuel_efficiency      | action, x_position, y_position, x_velocity, y_velocity   | -               | action_cost, conditional_idle_bonus    | 动作0可获得小额奖励，但必须在安全接近阶段启用                         |

## 12. 初始训练后应观察的 failure modes
| failure_mode                      | evidence_to_check                                                                     | possible_intervention                                                                 |
|-----------------------------------|---------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| 高速撞击目标区域导致 crash        | 最后几步观测中 x_position、y_position 接近0但 speed 很高，且 episode 终止于 crash     | 加强 soft_landing_speed 职责，尤其是在接近目标时的速度惩罚权重或渐进因子              |
| 漂移出界                          | 水平坐标绝对值持续增大，未收敛到0                                                     | 强化 target_approach 奖励，或加入边界警告（但仍只能用观测）                         |
| 着陆姿态倾斜导致非稳定接地        | body_angle 较大，或双足接触不平衡，但 episode 因 body_not_awake 终止而位置不佳        | 增强 stable_orientation 与 contact_completion 的协同，提高姿态精度的敏感度           |
| 引擎过度使用                      | 大量主引擎或姿态引擎动作，燃料效率低                                                 | 在安全接近阶段适当引入 fuel_efficiency 条件奖励，设置合理的距离阈值                 |
| “假着陆” – 在错误位置静默         | episode 因 body_not_awake 终止，但 x_position 或 y_position 远离零点，且接触标志可能为真 | 必须将 contact_completion 与位置/速度条件强绑定，拒绝在远离目标时的 ground contact 奖励 |
