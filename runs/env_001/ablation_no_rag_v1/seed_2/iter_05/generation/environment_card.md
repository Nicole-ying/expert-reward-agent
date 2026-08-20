# 匿名环境理解卡片

## 1. 任务目标
本环境是一个 2D 飞行器轨迹优化任务。  
初始时飞行器出现在视口顶部中央，并带有随机初始力。  
核心目标是 **安全、快速地降落在中央的降落平台上，并保持稳定**。  
理想行为应包含：接近目标、降低速度、保持姿态稳定、轻柔接触。  
附加考虑是 **尽量少用主引擎推力**（节能导向），但到达平台是首要任务。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box（连续值）
- shape: (8,)
- dtype: float32
- obs[0]: x_position — 相对于降落平台的水平坐标（负值为左，正值为右）
- obs[1]: y_position — 相对于降落平台高度的垂直坐标（负值在平台下方，正值在上方）
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 机体倾角（方向角）
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左支撑接触标志（1.0 接触，0.0 未接触）
- obs[7]: right_support_contact — 右支撑接触标志（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine — 什么也不做（无推力、无姿态控制）
- action 1: left_orientation_engine — 启动左侧姿态发动机（产生一个方向力矩）
- action 2: main_engine — 启动主发动机（向下推力，产生上升力并影响姿态）
- action 3: right_orientation_engine — 启动右侧姿态发动机（产生相反方向力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- crash_or_body_contact：可能指飞行器主体与地面或障碍物发生致命碰撞（如侧翻、撞击）。此为**失败终止**。
- horizontal_position_outside_viewport：水平移出视口边界。此为**失败终止**。
- body_not_awake_or_settled：机体进入非活跃状态或已稳定停靠。当飞行器即将静止或已停止运动时触发。可能包含**成功**（停在平台上且姿态稳定）或**失败**（停在平台外的地面上、悬停在空中但不再运动等）。因此属于**模糊终止**。
- 此外可能存在时间截断（truncation），来自 `terminated=False` 以外的逻辑，但源码未给出，可暂认为无显式超时。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 源码返回空 dict `{}`，即无可直接使用的 info 字段。
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用（包括任何可能的 success、failure、termination_reason 等），因为这些字段未在 step 源码中声明。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用：**
- obs：上一步的观测（8 维数组）
- action：上一步执行的动作（0~3）
- next_obs：当前步的观测（8 维数组）
- info 中明确允许的字段：无（info 为空）
- training_progress：仅当环境描述明确说明需要时才使用，当前任务未说明，故**不应使用**。

**禁止使用：**
- original_reward（掩码的官方奖励）
- 任何未在 info 中声明的字段
- 任何未在 obs 中声明的切片或衍生变量

## 7. 可用于奖励函数的信号
以下信号均可从 `obs` 或 `next_obs` 中获取，供奖励塑造：
- position：`obs[0]`(x)、`obs[1]`(y)，相对于目标平台
- velocity：`obs[2]`(vx)、`obs[3]`(vy)
- orientation：`obs[4]`(angle)、`obs[5]`(angular velocity)
- contact：`obs[6]`(left contact)、`obs[7]`(right contact)
- action/engine：可从 action 值推断是否使用了主引擎或姿态引擎

## 8. 不确定或不可用的信号
- 官方奖励（`original_reward`） **禁用**
- 任何明确标记“success”或“failure”的标记位 **不可用** （info 为空）
- 环境内部时间步、推力大小等物理变量 **未暴露**，不可直接使用
- 超时截断信号（truncation flag） 未在 step 返回中提供，若存在也属于隐藏信息