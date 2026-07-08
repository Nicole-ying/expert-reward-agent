# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
控制一个2D飞行器从上方区域出发，尽快飞向并稳定降落在画面中央的目标平台上。  
在此过程中，需要尽可能减少引擎推力，同时保持合理姿态与安全接触。  
主要目标是快速、平滑、安全地到达目标平台并稳定（settle）。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务的核心是到达指定的中央目标平台并稳定（goal reaching），尽管也包含节省燃料、稳定姿态等子目标，但这些属于到达目标后的效率与质量约束，并不改变“到达目标”这一本质。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32 (推测，接触标志为 0.0/1.0)  
- obs[0]: x_position —— 相对于目标平台中心点的水平坐标  
- obs[1]: y_position —— 相对于目标平台高度的垂直坐标  
- obs[2]: x_velocity —— 水平方向线速度  
- obs[3]: y_velocity —— 垂直方向线速度  
- obs[4]: body_angle —— 飞行器朝向角  
- obs[5]: angular_velocity —— 角速度  
- obs[6]: left_support_contact —— 左支撑（腿）是否接触平台（0/1 浮点数）  
- obs[7]: right_support_contact —— 右支撑（腿）是否接触平台（0/1 浮点数）

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- action 0: `no_engine` —— 不做任何操作  
- action 1: `left_orientation_engine` —— 启动一个转向引擎（使飞行器向左旋转）  
- action 2: `main_engine` —— 启动主引擎（产生向上的推力）  
- action 3: `right_orientation_engine` —— 启动另一个转向引擎（使飞行器向右旋转）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  - `body_not_awake_or_settled` 可能是成功的一种形式，即飞行器在平台上方或平台附近稳定（settled）并停止活动。但这需要结合位置、速度等条件进一步判断，不能直接作为成功标志。
- **failure-like termination**:  
  - `crash_or_body_contact` —— 通常表示坠毁或与地面等不期望的物体发生碰撞。  
  - `horizontal_position_outside_viewport` —— 水平位置超出视野边界，明显是失败。
- **ambiguous termination**:  
  - `body_not_awake_or_settled` 也可能发生在非目标区域的静止状态，因此仅凭该条件无法确知是否成功。
- **truncation**:  
  - 环境中未出现 `Truncated` 信息（完整 mask 中的 `return` 只有 `terminated, False`），因此环境不存在基于步数或时间的截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（info 为空字典，且环境未声明任何额外字段）  
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用；`success`、`failure`、`termination_reason` 等均不存在且不可假设。

**重要提醒**：奖励函数中不能依赖任何 info 中的成功/失败标志，必须基于观测、动作和终止状态自行推断。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`  —— 当前步骤的观测向量（含 8 个维度）
- `action` —— 所执行的动作索引 (0-3)
- `next_obs` —— 执行动作后下一时刻的观测向量（同 8 维）
- `info` 中明确允许的字段（本例中不存在）
- `training_progress` **不得使用**（任务描述未明确允许，遵循“只有 prompt 明确允许时才用”的原则）

禁止使用：
- `original_reward`（原始奖励已被屏蔽，不可重建或依赖）
- 任何未声明的 info 字段
- 任何未明确列在观测空间或禁止字段列表之外的辅助量

## 7. 可用于奖励函数的信号
- signal name: x_position
  - source: obs[0] / next_obs[0]
  - meaning: 飞行器相对于目标平台中心的水平距离
  - availability: available

- signal name: y_position
  - source: obs[1] / next_obs[1]
  - meaning: 飞行器相对于目标平台高度的垂直距离（目标平台高度处为 0）
  - availability: available

- signal name: x_velocity
  - source: obs[2] / next_obs[2]
  - meaning: 水平方向线速度
  - availability: available

- signal name: y_velocity
  - source: obs[3] / next_obs[3]
  - meaning: 垂直方向线速度
  - availability: available

- signal name: body_angle
  - source: obs[4] / next_obs[4]
  - meaning: 飞行器朝向角
  - availability: available

- signal name: angular_velocity
  - source: obs[5] / next_obs[5]
  - meaning: 角速度
  - availability: available

- signal name: left_contact
  - source: obs[6] / next_obs[6]
  - meaning: 左支撑腿是否接触平台（1.0 表示接触）
  - availability: available

- signal name: right_contact
  - source: obs[7] / next_obs[7]
  - meaning: 右支撑腿是否接触平台（1.0 表示接触）
  - availability: available

- signal name: action_main_engine_used
  - source: action == 2
  - meaning: 是否使用了主引擎（可作为推力/燃料消耗的惩罚依据）
  - availability: available

- signal name: action_orientation_engine_used
  - source: action in {1, 3}
  - meaning: 是否使用了转向引擎（同样可视为能耗或控制惩罚）
  - availability: available

## 8. 不确定或不可用的信号
- 官方原始奖励 (original_reward) —— 被明确屏蔽，不可用
- 任何 info 字段（包括 success, failure, episode_length 等）—— 不可用
- training_progress —— 未声明允许，不可用
- 对外部环境知识的假设（例如精确物理参数、目标区域形状、燃料量等）—— 不可用
- 显式的成功/失败标志 —— 不存在，不可用
