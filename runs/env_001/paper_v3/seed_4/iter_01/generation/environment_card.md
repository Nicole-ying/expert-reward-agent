# 匿名环境理解卡片

## 1. 任务目标
该匿名环境是一个 2D 飞行器轨迹优化任务。飞行器从视口顶部中央附近出发，初始带有随机作用力。任务核心目标是使飞行器尽快到达并稳定停靠在中央目标平台上，同时尽可能少地使用引擎推力。智能体需要在接近目标的过程中降低速度、保持姿态稳定，并以低速度、小角度实现双腿安全触地。次要目标是节省燃料，避免不必要的引擎动作。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务的核心是到达指定目标平台并稳定停靠，具有明确的空间目标（目标垫中央），而保持姿态、节油等均为辅助要求，并非并列或多个冲突目标。尽管有接触条件，但本质仍是状态空间中的目标到达问题，因此属于导航/目标到达族。

动力学子类型：goal_approach_and_soft_contact（接近目标并低速、稳定接触/停靠）

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: 推断为 float32（未明确给出，但通常如此）
- 各维度含义：
  - obs[0]: x_position，飞行器相对于目标垫的水平坐标，reward_usable: true
  - obs[1]: y_position，飞行器相对于垫面高度的垂直坐标，reward_usable: true
  - obs[2]: x_velocity，水平线速度，reward_usable: true
  - obs[3]: y_velocity，垂直线速度，reward_usable: true
  - obs[4]: body_angle，机体姿态角，reward_usable: true
  - obs[5]: angular_velocity，角速度，reward_usable: true
  - obs[6]: left_support_contact，左支撑腿接触标志（0.0 或 1.0），reward_usable: true
  - obs[7]: right_support_contact，右支撑腿接触标志（0.0 或 1.0），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：
  - action 0: no_engine，不做任何推进
  - action 1: left_orientation_engine，点燃左侧姿态引擎
  - action 2: main_engine，点燃主引擎
  - action 3: right_orientation_engine，点燃右侧姿态引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- **body_not_awake_or_settled**：飞行器因稳定停靠而进入休眠或被判定为已安定，环境终止。此条件是**可能的成功终止**，但需要根据终止时的状态（位置接近目标、双腿接触、低速度、小角度）进一步判别。
- **crash_or_body_contact**：飞行器与地面或障碍物发生剧烈碰撞（可能倾覆或过猛接触），视为**失败终止**。
- **horizontal_position_outside_viewport**：飞行器水平飞出可视区域，视为**失败终止**。

（注：未给出 truncation 条件，无最大步数截断信息，但可以预见训练中可能设置时间上限，但环境源代码中未体现。）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 空（step 返回 info 为 {}，无可用的额外字段）
- forbidden_or_uncertain_info_fields: 禁止使用任何 info 字段，因为均为空。

成功与否必须完全从观测序列和终止时的状态（next_obs）中推断。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
**允许使用：**
- `obs`：当前观测，8 维数值向量
- `action`：当前执行的动作，0~3 整数
- `next_obs`：下一观测，结构同 obs
- `info`：但当前环境 info 为空字典，实际不可用任何字段
- `training_progress`：仅在 prompt 明确允许或训练框架要求时使用，当前环境描述未明确允许，保持谨慎（可声明为“若允许则可使用”）

**禁止使用：**
- `original_reward`（已遮蔽，禁止重构）
- 任何未声明的 info 字段
- 任何未声明的 obs 切片或外部状态

## 7. 可用于奖励函数的信号
- **位置信号**：obs[0] (x_position), obs[1] (y_position)；next_obs 对应维度。
- **速度信号**：obs[2] (x_velocity), obs[3] (y_velocity)；next_obs 对应维度。
- **姿态信号**：obs[4] (body_angle), obs[5] (angular_velocity)；next_obs 对应维度。
- **接触信号**：obs[6] (left_support_contact), obs[7] (right_support_contact)；next_obs 对应维度。
- **动作/引擎使用信号**：action 值本身（0 为无推力，其他为使用引擎）。
- **其他**：终止信号 `terminated` 未直接进入 reward 函数，但可通过 `next_obs` 后是否结束来间接感知（训练循环外部，通常 reward 函数不接收 terminated 标志）。若环境提供 `training_progress`，可用于调度课程。

## 8. 不确定或不可用的信号
- 任何形如 `info["success"]`、`info["failure"]`、`info["landing_quality"]` 等官方成功/失败标签：**不可用**（info 为空）。
- 剩余燃料量、引擎推力强度等内部物理量：**未在观测中提供**，不可直接获取。
- 全局时间步数或剩余时间：**未提供**（除非通过 training_progress 间接获得）。
- 环境重置时的随机初始力大小：**不可使用**（仅在初始时影响，奖励函数不应依赖初始状态）。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete (4 actions)
morphology:
  body_type: rigid_body_with_two_legs
  actuator_type: impulse_based_main_and_orientation_engines
  contact_structure: left_and_right_leg_contact_sensors
primary_objectives:
  - 尽快到达目标平台中央并稳定停靠
  - 实现安全软着陆（低速度、小角度、双腿触地）
secondary_objectives:
  - 最小化引擎使用（省油）
main_failure_risks:
  - 飞行器硬撞击地面或倾覆
  - 水平漂出视野范围
  - 未能及时减速导致越标或错过平台
  - 姿态角过大导致触地后倾翻
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: goal_reaching_and_safe_contact**
  - purpose: 激励飞行器靠近目标平台并最终在满足成功条件时稳定停靠。
  - why_required: 这是任务的核心目标，无此奖励则学习无方向。
  - usable_signals: x_position, y_position, left_support_contact, right_support_contact, velocity/angle（用于判定成功状态）。
  - risks: 若只考虑位置而不考虑速度/接触，可能导致高速撞击；需与着陆质量奖励配合。

- **role_id: soft_landing_quality**
  - purpose: 惩罚触地时过大的速度、过大的姿态角以及角速度，鼓励双腿同时接触。
  - why_required: 保证安全着陆，避免硬着陆和倾翻，是任务的关键质量要求。
