# 匿名环境理解卡片

## 1. 任务目标
这是一个 2D 飞行器/着陆器任务。主体从画面顶部中央附近出发，受到随机初始力作用。**主要目标**是尽快飞到并稳定在中央的目标着陆垫上，同时尽可能少用引擎推力。智能体必须学会：平滑接近目标、减小速度、保持水平姿态，并以安全接触方式着陆。**次要目标**是节约燃料（即少用引擎）。不应将目标分解为纯导航或纯生存，到达并稳定着陆是本环境的唯一核心目标，燃料效率是附加在相同轨迹上的性能优化。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务的核心是到达指定的目标位置（着陆垫）并稳定，属于导航到达型，附加有速度/姿态要求及能耗优化，但达到目标是主目标，省燃料为附属优化，不属于多目标任务。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32（推测）
- obs[0]: `x_position_relative_to_target`，水平坐标（相对目标垫），reward_usable: true
- obs[1]: `y_position_relative_to_pad_height`，垂直坐标（相对垫高度），reward_usable: true
- obs[2]: `x_velocity`，水平线速度，reward_usable: true
- obs[3]: `y_velocity`，垂直线速度，reward_usable: true
- obs[4]: `body_angle`，机体方向角（弧度？），reward_usable: true
- obs[5]: `angular_velocity`，角速度，reward_usable: true
- obs[6]: `left_support_contact`，左侧支撑脚接触标志（1.0 接触，0.0 未接触），reward_usable: true
- obs[7]: `right_support_contact`，右侧支撑脚接触标志（1.0 接触，0.0 未接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作 0: `no_engine` —— 所有引擎关闭，不施加推力。
- 动作 1: `left_orientation_engine` —— 点燃左姿态引擎，产生改变姿态的力。
- 动作 2: `main_engine` —— 点燃主引擎（可能产生向上的主推力）。
- 动作 3: `right_orientation_engine` —— 点燃右姿态引擎，产生反向改变姿态的力。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 无明确的成功终止类型，但 `body_not_awake_or_settled` 可能对应于成功着陆并稳定在目标垫上，亦可能在到达垫之前就休眠造成失败，需结合位置判断。
- **failure-like termination**: `crash_or_body_contact`（硬碰撞或非目标垫的接触）和 `horizontal_position_outside_viewport`（水平超出画布）显然是失败终止。
- **ambiguous termination**: `body_not_awake_or_settled` 本身不能直接区分成功或失败。
- **truncation**: 未提供任何截断条件。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空）
- explicit_failure_flag_available: false （info 为空）
- allowed_info_fields: 无（info 是空字典 `{}`）
- forbidden_or_uncertain_info_fields: 无（所有 info 字段均不可用）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用**：
- `obs`
- `action`
- `next_obs`
- `info` 中明确允许的字段（当前为空，无字段可用）
- `training_progress` 仅在提示明确允许时使用（此处未允许，不使用）

**禁止使用**：
- `original_reward`
- `official_reward`
- 任何未声明的 `info` 字段
- 任何未声明的 `obs` 切片

## 7. 可用于奖励函数的信号
- **位置**：`obs[0]`（x）, `obs[1]`（y），以及对应的 next_obs 值，可构造距离、接近速度等。
- **速度**：`obs[2]`（vx）, `obs[3]`（vy），可用于惩罚过快着陆或水平漂移。
- **姿态与角速度**：`obs[4]`（角度）, `obs[5]`（角速度），可用于鼓励水平姿态和稳定性。
- **接触标志**：`obs[6]`, `obs[7]`，可判断是否与垫接触（但无法区分是目标垫还是其他表面，只能结合位置估计成功着陆）。
- **动作/引擎使用**：`action` 本身，0 为无推力，1/2/3 表示使用了引擎，可用于惩罚或奖励节油。
- **其他**：无。

## 8. 不确定或不可用的信号
- **explicit success / failure flags**：不存在。
- **接触的语义**（目标垫 vs 地面）不可直接获取，需通过位置推断。
- **目标垫的位置范围**未知，仅知观测给出相对于垫的偏移，假设目标坐标 (0,0) 时，垫位于该位置附近。
- **燃料量**不可用，只能通过引擎使用次数间接衡量。
- **轨迹存储**未提供，无法回溯历史状态。
- **环境内在奖励**被屏蔽，绝不可依赖。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body (vehicle-like lander)
  actuator_type: main engine (vertical thrust) + orientation thrusters (left/right)
  contact_structure: two support contact points (left leg, right leg)
primary_objectives:
  - Reach the target pad (minimize horizontal and vertical offset)
  - Land safely with low velocity and near-zero orientation
secondary_objectives:
  - Minim