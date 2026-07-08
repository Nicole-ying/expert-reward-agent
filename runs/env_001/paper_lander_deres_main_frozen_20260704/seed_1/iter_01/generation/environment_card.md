# Env_001 环境理解卡片

## 1. 任务目标
控制一辆 2D 飞行器（或类似车辆）从画面顶部中央区域出发（初始有随机作用力），在**尽可能短的时间内**到达并**稳定停留在中央目标垫**上，同时尽量**少使用引擎推力**。  
智能体需要学会：平滑接近目标、减速、保持垂直方向稳定、与地面安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（默认，例如 `np.float32`）
- obs[0]: x_position — 相对于目标垫中心的水平坐标
- obs[1]: y_position — 相对于目标垫高度的垂直坐标
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 机体朝向角（弧度）
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左侧支撑腿/触地标志（0.0/1.0）
- obs[7]: right_support_contact — 右侧支撑腿/触地标志（0.0/1.0）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine — 不点燃任何引擎
- action 1: left_orientation_engine — 点燃左方向引擎（用于调整方向）
- action 2: main_engine — 点燃主引擎（提供主要推力）
- action 3: right_orientation_engine — 点燃右方向引擎（与左方向相反）

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**:  
  可能为 `body_not_awake_or_settled` 且位置/速度符合目标垫稳定标准。  
  *注意：由于 info 为空且 masked 奖励不可见，无法直接确认是否为成功。只可视为“可能成功”的信号，不能直接作为奖励中的成功标志。*
- **failure-like termination**:  
  `crash_or_body_contact`（身体与地面以外物体碰撞或超出正常接触）  
  `horizontal_position_outside_viewport`（水平飞出视口）
- **ambiguous termination**:  
  `body_not_awake_or_settled` 可能发生在远离目标垫的位置，这时不算成功；也可能发生在目标垫上且速度接近零。该条件**不可直接用于奖励**，因为缺少对是否在目标垫上的判断。
- **truncation**:  
  从 step 源码看 `truncated` 恒为 `False`，不存在时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: （无，info 为空字典）  
- forbidden_or_uncertain_info_fields: `info` 的全部内容（目前无任何字段）；`original_reward` / `official_reward` 严格禁止使用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- `obs`（当前状态）
- `action`（所执行的动作）
- `next_obs`（下一状态）
- 此任务中 `info` 为空，**无任何可用字段**

禁止使用：
- `original_reward`
- 任何未在观察空间中明确定义的内部状态
- `training_progress`（仅在 prompt 明确允许时可用，本例未允许）
- `info` 中任何字段（目前为空，若有未声明的字段亦禁止）

## 7. 可用于奖励函数的信号
- **position**: `next_obs[0]`（x 偏差）、`next_obs[1]`（y 偏差）—— 用于奖励接近目标垫中心。
- **velocity**: `next_obs[2]`、`next_obs[3]` —— 用于惩罚高速接近或奖励软着陆。
- **orientation**: `next_obs[4]`（角度）、`next_obs[5]`（角速度）—— 用于鼓励竖直稳定。
- **contact**: `next_obs[6]`、`next_obs[7]`（左右支撑腿触地）—— 用于鼓励安全接触或判断稳定着陆。
- **action/engine**: `action`（0~3）—— 可直接惩罚引擎使用次数或推力强度（例如 action 1/2/3 视为推力消耗）。

## 8. 不确定或不可用的信号
- 是否**成功到达目标垫**：`terminated` 中的 `body_not_awake_or_settled` 没有指明是否在目标范围内，不可靠。
- 碰撞严重程度：只有终止标志，无连续伤害值。
- 能耗真实数值（如燃料消耗量）：masked 奖励中隐藏，不能使用。
- 任何 info 字段（如 `success`, `failure`, `distance` 等）：均未暴露，必须视为不存在。