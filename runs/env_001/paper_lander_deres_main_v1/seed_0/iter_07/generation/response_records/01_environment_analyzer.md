# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
这是一个二维飞行器样式的轨迹规划任务。飞行器初始时位于画面上方中央区域，并受到随机的初始力。目标是在尽可能短的时间内飞抵画面中央的目标垫并稳定停靠，同时尽量少使用主引擎或方向引擎推力。智能体需要学会接近目标地点、减小速度、保持平稳姿态，并以安全接触的方式着陆到目标垫上。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推测，Box 默认为 float）
- obs[0]: x_position — 飞行器相对于目标垫中心的水平坐标
- obs[1]: y_position — 飞行器相对于目标垫高度的垂直坐标
- obs[2]: x_velocity — 飞行器水平方向线速度
- obs[3]: y_velocity — 飞行器垂直方向线速度
- obs[4]: body_angle — 飞行器机身的姿态角（弧度）
- obs[5]: angular_velocity — 飞行器角速度
- obs[6]: left_support_contact — 左侧支撑腿是否接触目标垫（1.0 接触，0.0 未接触）
- obs[7]: right_support_contact — 右侧支撑腿是否接触目标垫（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine — 什么也不做，不施加推力
- action 1: left_orientation_engine — 启动左侧方向引擎（用于调整姿态）
- action 2: main_engine — 启动主引擎（产生推力，改变速度）
- action 3: right_orientation_engine — 启动右侧方向引擎（用于调整姿态）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（休眠/稳定），当飞行器在目标垫附近且姿态稳定时很可能表示成功着陆并稳定停留
- failure-like termination: crash_or_body_contact（机身非着陆部位发生碰撞）；horizontal_position_outside_viewport（水平位置超出画面边界）
- ambiguous termination: body_not_awake_or_settled 也可能在远离目标垫的位置发生，例如因撞击后静止，因此不能直接等同于成功，需要结合位置信息判断
- truncation: 未在源文件中出现，但一般由时间步上限引发

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（step 返回的 info 固定为空字典 {}）
- forbidden_or_uncertain_info_fields: 所有 info 字段均禁止使用（包括任何可能隐含的终止原因字段）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：当前观测
- action：当前执行的动作
- next_obs：下一步观测
- info 中明确允许的字段（当前无任何字段被明确允许）
- training_progress（仅当提示明确要求使用时才可安全使用，否则应视为不可用）

禁止使用：
- original_reward（原始奖励信号已被屏蔽，严禁依赖）
- official_reward 或任何形式的官方奖励
- 未声明的 info 字段（info 完全为空，使用任何字段均违背契约）
- 未在环境卡片中声明的 obs 切片含义

## 7. 可用于奖励函数的信号
- position: obs[0]（水平位置相对目标）、obs[1]（垂直相对高度）
- velocity: obs[2]（水平速度）、obs[3]（垂直速度）
- orientation: obs[4]（姿态角）
- angular velocity: obs[5]
- contact: obs[6]（左支撑腿接触）、obs[7]（右支撑腿接触）
- action/engine: 动作编号本身可反映使用何种引擎，用于衡量推力使用量

## 8. 不确定或不可用的信号
- info 字典：完全为空，无法提供任何额外终止原因或成功标记
- original_reward：已被屏蔽，数值不可知，不允许读取或拟合
- 机身碰撞详情：crash_or_body_contact 的具体碰撞类型在观测中不可见
- 全局时间步或剩余时间：未在观测中提供
- 目标垫绝对坐标：只能通过相对位置推知接近程度
