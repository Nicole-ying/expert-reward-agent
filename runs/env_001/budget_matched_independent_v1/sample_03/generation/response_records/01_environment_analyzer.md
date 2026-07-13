# Response Record

# 匿名环境理解卡片

## 1. 任务目标
主要目标：控制一个2D飞行器从顶部中心区域出发，最终平稳降落在画面中央的目标平台上。要求平台接触发生在两条支撑腿上、身体方向水平、速度趋近于零并且不发生 crash（如身体其他部位触碰地面或墙壁）。  
次要目标：在达成安全着陆的前提下，尽可能缩短飞行时间，并尽量减少引擎推力（节省燃料）。  
不应混淆的目标：任务的核心是精准软着陆，而非单纯穿越地形或保持平衡；快速与节能只是锦上添花，不能以牺牲安全降落为代价。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是到达指定的目标位置（平台中心）并稳定驻留，属于典型的终点导航/到达任务；附属的时间、燃料优化是次要约束，不构成权重相当的冲突目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推测默认）
- obs[0]: x_position，含义：相对于目标平台中心的水平坐标，reward_usable: true
- obs[1]: y_position，含义：相对于目标平台高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity，含义：水平线速度，reward_usable: true
- obs[3]: y_velocity，含义：垂直线速度，reward_usable: true
- obs[4]: body_angle，含义：身体朝向角度（可能以“水平为0”），reward_usable: true
- obs[5]: angular_velocity，含义：角速度，reward_usable: true
- obs[6]: left_support_contact，含义：左支撑腿是否接触平台（1.0接触，0.0未接触），reward_usable: true
- obs[7]: right_support_contact，含义：右支撑腿是否接触平台（1.0接触，0.0未接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine，含义：不点火，仅靠惯性运动
- action 1: left_orientation_engine，含义：启动左侧姿态发动机，产生某个方向的力矩/推力以调整朝向
- action 2: main_engine，含义：启动主引擎，提供向上的推力
- action 3: right_orientation_engine，含义：启动右侧姿态发动机，产生与左侧发动机相反的力矩/推力

（引擎的具体推力方向、大小未给出，但可以从名称推断：主引擎用于减速/悬浮，姿态引擎用于旋转身体。）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` —— 当身体静止/稳定时触发。极可能对应成功着陆并停稳的场景，因为在平台上稳定后不再移动，环境认为任务结束。
- **failure-like termination**:  
  - `crash_or_body_contact` —— 身体某些非支撑部位（如底部、侧边）发生碰撞，通常意味着坠毁。  
  - `horizontal_position_outside_viewport` —— 水平坐标超出可视范围，表示飞行器飞离并丢失。
- **ambiguous termination**: `body_not_awake_or_settled` 理论上也可能因为卡在某处不动而触发，但在该环境的典型设计中它大概率与成功绑定。
- **truncation**: 当前 step source 未展示时间截断，但实际环境可能存在 episode 长度限制（unknown）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info = {}，禁止使用任何 info 字段）
- forbidden_or_uncertain_info_fields: info 所有字段均不可用；original_reward 被屏蔽，禁止使用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- `obs`（当前观测）
- `action`（当前动作）
- `next_obs`（下一观测）
- `info` 目前为空，不得依赖任何内容
- `training_progress` 仅在 prompt 明确要求时使用，否则禁止

禁止使用：
- `original_reward`（官方奖励，已被屏蔽）
- 任何未声明或凭空出现的 `info` 字段
- 未在 observation_space 中声明的 obs 切片

## 7. 可用于奖励函数的信号
- **位置信号**：`obs[0]`（x偏移）、`obs[1]`（y偏移）及其变化量（可由 `next_obs - obs` 近似）
- **速度信号**：`obs[2]`（x速度）、`obs[3]`（y速度）
- **方向信号**：`obs[4]`（身体角度）、`obs[5]`（角速度）
- **接触信号**：`obs[6]`（左腿接触）、`obs[7]`（右腿接触）
- **动作/引擎信号**：`action`（0/1/2/3），可间接反映燃料消耗
- **其他**：可通过位置/速度组合计算距离、靠近速率等；时间代价可隐式用每步常数惩罚表示。

## 8. 不确定或不可用的信号
- **明确的 crash 标志**：无。仅能从 `obs` 中推测（如位置极度异常、速度突然反向等），但边界未知，初期不可靠。
- **精确的视野边界**：未提供值，无法预设阈值。
- **真实剩余燃料或推力大小**：观测空间中无此类信号，只能通过动作近似。
- **时间戳**：未提供步数或剩余时间，无法构建时间直接惩罚（但可间接用常数惩罚）。
- **身体其他部位碰撞信息**：除两条支撑腿外，无其它接触传感器，因此无法精确判断 crash_or_body_contact，只能靠经验间接推断。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D lander/flyer
  actuator_type: main engine (vertical thrust) + two lateral orientation engines
  contact_structure: two symmetric support legs for touchdown
primary_objectives:
  - Reach the central target pad (x≈0, y≈0)
  - Achieve soft touchdown with both legs contacting and low velocities
  - Keep body angle near zero (horizontal)
secondary_objectives:
  - Minimize flight time
  - Minimize engine usage (fuel)
main_failure_risks:
  - Crashing body into ground/walls
  - Exceeding horizontal boundaries
  - Touching down with non‑leg parts
  - Bouncing
