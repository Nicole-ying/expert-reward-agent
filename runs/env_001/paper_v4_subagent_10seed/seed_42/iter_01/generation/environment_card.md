# 匿名环境理解卡片

## 1. 任务目标
本任务是一个 2D 飞行器/车辆类似的任务：智能体从视口顶部中心附近受初始随机力开始，需要尽可能快地飞抵并稳定停靠在中央目标平台上，同时尽量减少发动机推力使用。核心目标是 **快速、稳定地完成着陆（到达并停留）**，次要目标是 **节省燃料、保持姿态平稳**。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**  
confidence: high  
reason: 核心问题是“是否离目标更近了？”，需要到达指定位置（目标平台）并最终静止，虽有省燃料和姿态要求，但它们明显从属于到达目的，不属于多目标冲突场景。动力学子类型需要精细控制减速、稳定接触，属于目标接近与软着陆一类。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32 (推断)
- obs[0]: `x_position` – 相对于目标平台中心的水平坐标，越小越接近，reward_usable: true
- obs[1]: `y_position` – 相对于目标平台高度的垂直坐标，reward_usable: true
- obs[2]: `x_velocity` – 水平线速度，reward_usable: true
- obs[3]: `y_velocity` – 垂直线速度，reward_usable: true
- obs[4]: `body_angle` – 身体倾斜角度（假设水平为 0），reward_usable: true
- obs[5]: `angular_velocity` – 角速度，reward_usable: true
- obs[6]: `left_support_contact` – 左支撑腿是否接触（0 或 1），reward_usable: true
- obs[7]: `right_support_contact` – 右支撑腿是否接触（0 或 1），reward_usable: true

（注意：所有字段均可用，但需小心接触信号的语义，任务目标中“接触”指的是安全着陆在目标平台，而非与地面或障碍物的碰撞）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine` – 不点火，依靠当前动量漂移
- action 1: `left_orientation_engine` – 点燃左姿态发动机（调整姿态或水平推力）
- action 2: `main_engine` – 点燃主发动机（主要提供垂直或前进推力）
- action 3: `right_orientation_engine` – 点燃右姿态发动机

## 5. step 与终止条件分析
### 5.1 终止模式
根据掩码源码，存在三种终止触发：
- `crash_or_body_contact` – 坠毁或部分身体接触（可能包括与地面/障碍物的不当接触）
- `horizontal_position_outside_viewport` – 水平位置超出视口边界（失败）
- `body_not_awake_or_settled` – 身体不再活跃或已经稳定（可能为成功，若发生在目标平台上）

成功意义上的终止并没有显式分离，只能通过观测状态间接判别：当智能体接近目标( x ≈ 0, y ≈ 0 )，速度极小，且两侧支撑腿接触（可能），触发 `body_not_awake_or_settled` 可视为 soft landing success；而其他终止条件（crash、出界）则对应失败。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {}（info 为空字典，无额外字段）
- forbidden_or_uncertain_info_fields: 所有未在 `observation_space` 中列出的字段均不可用（包括 `terminated` 标记、`success` 等）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`（8维数组）
- `action`（整数 0-3）
- `next_obs`（8维数组）
- `info`（固定空字典，无任何可依赖字段）
- `training_progress`（仅当 prompt 明确允许时，本例不允许使用）

严格禁止使用：
- `original_reward`（被掩码的官方奖励，不可读取）
- `official_reward` 或其他未声明的奖励值
- 任何未出现在 observation_space 中的信号（如外界风扇、风力等）
- 任何依赖 `info` 中隐藏内容的逻辑

## 7. 可用于奖励函数的信号
- **位置**：`x_position`，`y_position`（相对目标，直接表征进度）
- **速度**：`x_velocity`，`y_velocity`（绝对值或矢量和可用于判断稳定、能耗）
- **姿态**：`body_angle`，`angular_velocity`（衡量晃动，违反稳定着陆）
- **接触**：`left_support_contact`，`right_support_contact`（区分接触/非接触，可用于软着陆推断）
- **动作/引擎**：动作类别 0-3，可用于燃油消耗惩罚（action != 0 视为使用引擎）
- **其他衍生信号**：
  - 距离目标：`dist = sqrt(x^2 + y^2)`（可直接计算）
  - 速度大小：`speed = sqrt(vx^2 + vy^2)`
  - 距离减少：`delta_dist = dist_prev - dist_next`
  - 姿态偏离：`angle_deviation`（假设水平为0）

## 8. 不确定或不可用的信号
- **显式成功/失败标志**：不存在，info 为空，不能直接获取
- **环境内部奖励**：`original_reward` 被明确禁用
- **终止原因**：无法从任何可用信息中可靠读取出 `terminated` 的原因
- **风速/外力**：未声明，不可使用
- **目标坐标绝对位置**：已知目标垫为相对原点，但无法获取其在全局中的绝对坐标（相对坐标已足够）
- **精确碰撞类型**：无法区分坠毁与着陆，只能通过位置、速度、姿态间接推测

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body_2d_vehicle_with_legs
  actuator_type: main_engine_plus_two_orientation_thrusters
  contact_structure: two_support_legs (left/right contact sensors)
primary_objectives:
  - reach_target_position (x≈0, y≈0) with minimal overshoot
  - stabilize_on_target (small velocity, safe orientation, supported legs)
secondary_objectives:
  - minimize_fuel_usage (action is not free; no_engine preferred when possible)
  - maintain_stable_orientation (keep body_angle close to 0)
main_failure_risks:
  - crashing_into_ground_or_obstacle (excessive speed, bad angle)
  - drifting_out_of_bounds (horizontal overshoot)
  - hovering_or_not_settling (agent stays near target but never lands)
  - fuel_exhaustion (too many engine ignitions causing inefficiency)
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: progress_to_target**
  purpose: 驱动智能体持续减少与目标平台的距离，形成向目标移动的基本梯度。
  why_required: 任务本质是到达（goal reaching），没有此信号无法形成向目标移动的清晰奖励。使用 delta distance 避免静态悬停取巧（proximity-based reward 会让 agent 停在靠近但不着陆的位置）。
  usable_signals: [obs[:2], next_obs[:2]]
  risks: 如果 agent 已经非常接近但速度仍然很大，单纯 delta distance 可能鼓励以高速冲向目标，可能导致坠毁或错过着陆。需要由后续安全保障角色约束。

### 10.2 条件职责 conditional_roles
- **role_id: soft_landing_guidance**
  condition_to_use: 当智能体已进入目标附近区域（dist < 阈值，如 0.2）时激活。
  usable_signals: [speed (from next_obs[2:4]), dist (from next_obs[0:2])]
  risks: 如果阈值过大，可能过早抑制正常接近速度，延长训练时间；阈值过小则帮助不大。可作为一个可选的辅助项，在训练中后期加入。

- **role_id: fuel_efficiency**
  condition_to_use: 任务明确要求“尽可能少地使用引擎