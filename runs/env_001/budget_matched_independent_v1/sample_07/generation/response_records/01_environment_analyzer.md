# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本任务是一个 2D 飞行器轨迹优化问题。一个刚体从视口顶部中心附近出发，带有初始随机扰动。
核心目标是**到达并稳定停留在中心目标平台上（settle）**，要求飞行器最终停在目标垫上，
速度趋于零，姿态稳定，并且两个接触标志均为有效接触。
次要目标是最小化到达时间和发动机推力使用量，即“尽快、尽量省燃料”。
任务**不能混淆**为单纯推进（locomotion）或仅保持平衡，着陆成功是最终判定标准。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**  
confidence: high  
reason: 核心目标是到达指定目标位置（目标垫）并稳定停驻，附属有时间/能效等优化但不改变主要目的。该任务不属于持续行走、操作、多目标冲突等类别。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32（连续量 + 接触标志 0.0/1.0）  
- 各分量含义与可奖励性：  

| index | name | meaning | reward_usable |
|-------|------|---------|---------------|
| 0 | x_position | 飞行器质心相对于目标垫中心的水平偏移量 | true |
| 1 | y_position | 飞行器质心相对于目标垫高度的垂直偏移量 | true |
| 2 | x_velocity | 水平线速度 | true |
| 3 | y_velocity | 垂直线速度 | true |
| 4 | body_angle | 身体倾斜角 | true |
| 5 | angular_velocity | 角速度 | true |
| 6 | left_support_contact | 左支撑接触标志（1.0 有接触，0.0 无） | true |
| 7 | right_support_contact | 右支撑接触标志（1.0 有接触，0.0 无） | true |

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 动作含义：

| action | name | meaning |
|--------|------|---------|
| 0 | no_engine | 不启动任何引擎，仅依靠惯性、重力、风等 |
| 1 | left_orientation_engine | 启动左侧姿态引擎，提供偏航/扭矩 |
| 2 | main_engine | 启动主引擎，提供主要推力（通常向下或向后） |
| 3 | right_orientation_engine | 启动右侧姿态引擎，与左引擎反向，调节姿态 |

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  `body_not_awake_or_settled` — 身体已经静止或安定。若此时飞行器位于目标垫上方（x、y 接近 0）、速度极低、双接触点有效，且姿态水平，则很可能成功着陆。该模式是唯一可能代表成功的终止，但无法直接确认。
- **failure-like termination**:  
  - `crash_or_body_contact` — 身体与地面或非目标区域发生碰撞（例如机腹触地、侧翻等）。  
  - `horizontal_position_outside_viewport` — 水平位置超出可视范围，飞行器脱离可控制区域。
- **ambiguous termination**:  
  上述 `body_not_awake_or_settled` 也可在不安全状态下触发（如悬停在空中但静止，或只有一个接触点触地），因此为模糊终止。
- **truncation**:  
  源步骤中 `return` 的第四个参数为 `False`，表示无截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**  
- explicit_failure_flag_available: **false**  
- allowed_info_fields: 无（`info` 为 `{}`）  
- forbidden_or_uncertain_info_fields: 任何依赖 `info` 的 `success`、`failure`、`termination_reason` 等字段均不可用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`（当前观测）
- `action`（离散动作索引）
- `next_obs`（下一步观测）
- `info` 中明确允许的字段（这里 `info` 为空，**无可用的额外信息**）
- `training_progress` 仅在 prompt 明确允许时使用（当前未开放，不建议依赖）

禁止使用：
- `original_reward`（官方奖励已掩膜）
- 任何未明确声明的 `obs` 切片含义外的原始值
- 假定的 `info` 字段（如 `success`、`landing_quality` 等）

## 7. 可用于奖励函数的信号
- 位置：`obs[0]` (x), `obs[1]` (y)，均为相对目标垫中心的偏移。  
- 速度：`obs[2]` (vx), `obs[3]` (vy)。  
- 姿态：`obs[4]` (angle), `obs[5]` (angular_vel)。  
- 接触：`obs[6]` (left contact), `obs[7]` (right contact)。  
- 动作：`action`（0~3 离散值，可判断是否用引擎）。  
- 可在 `next_obs` 中获取以上信号的变化，例如速度变化、位置变化等。

## 8. 不确定或不可用的信号
- 官方奖励：完全掩膜，不可获得。  
- 着陆成功 / 失败标识：无。  
- 剩余燃料 / 时间步计数：无法从当前接口获取。  
- 目标垫的绝对坐标或尺寸：未知，但 obs 已提供相对位置，足够。  
- `info` 中的任何字段：均为空，不可用。  
- `training_progress` 的具体含义与范围未知，当前不应依赖。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: single_rigid_body
  actuator_type:
    - main_engine (垂直 / 倾斜推力)
    - two_orientation_engines (左右旋转控制)
  contact_structure:
    - two_support_contacts (left, right)
primary_objectives:
  - reach_and_settle_on_target_pad (到达目标垫并稳定停驻)
secondary_objectives:
  - minimize_time_to_settle (最小化着陆时间)
  - minimize_engine_usage (最小化引擎使用)
main_failure_risks:
  - crash_into_ground (与地面/非目标碰撞)
  - out_of_horizontal_bound (水平漂移出界)
  - unstable_or_incomplete_landing (着陆时翻滚、单脚着地、速度未消除等)
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
这些职责直接服务于核心目标，必须在所有训练阶段存在或占据主导。

- **role_id**: `approach_to_target`  
  purpose: 引导飞行器朝目标垫移动，使相对位置 (x, y) 趋近于 (0,0)。  
  why_required: 不朝目标移动则永远无法抵达。  
  usable_signals: [obs[0] x_position, obs[1] y_position, next_obs[0], next_obs[1]]  
  risks: 过分简单的负距离奖励可能导致震颤；需要使用平滑、有界的信号。

- **role_id**: `soft_landing`  
  purpose: 在接近目标时促使
