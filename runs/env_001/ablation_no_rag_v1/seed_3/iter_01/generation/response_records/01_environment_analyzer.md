# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个 2D 车辆型轨迹优化任务。主体从视口顶部中央附近以随机初始力开始运动。目标是尽快到达并稳定停靠在中央的着陆垫上，同时尽可能少地使用引擎推力。智能体需要学会向目标接近、降低速度、维持稳定姿态并安全触垫。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32（推测，连续观测）
- obs[0]: x_position — 相对于着陆垫的水平坐标
- obs[1]: y_position — 相对于垫面高度的垂直坐标
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 机体朝向角度
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左侧支撑触地标志（1.0 接触，0.0 未接触）
- obs[7]: right_support_contact — 右侧支撑触地标志（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete
- 动作数量: 4
- action 0: no_engine — 不执行任何推力
- action 1: left_orientation_engine — 点燃一个姿态发动机（产生旋转推力）
- action 2: main_engine — 点燃主发动机（产生向上的推力，通常用于减速/悬停）
- action 3: right_orientation_engine — 点燃另一个姿态发动机（产生反向旋转推力）

## 5. step 与终止条件分析

### 5.1 终止模式
- success-like termination: 当机体成功稳定在着陆垫上时（可能体现为 `body_not_awake_or_settled` 且双支撑足接触，且位置靠近垫中心），视为成功着陆。
- failure-like termination: 发生坠毁或身体其他部位接触地面（`crash_or_body_contact`）、水平位置超出视口边界（`horizontal_position_outside_viewport`）均属于失败终止。
- ambiguous termination: 机体在垫外静止或倾斜不稳但仍未触发 crash/越界，此时由 `body_not_awake_or_settled` 触发终止，需结合位置和接触状态判断是否成功，但环境未提供显式成功/失败标志。
- truncation: 未显示时间截断信息，当前 step 源未提及 `truncated` 返回，仅返回 `False`。因此该环境不因步数截断，只可能由上述物理条件终止。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空字典 `{}`，未提供任何额外字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用（因为未定义）。不可使用 `info['success']`、`info['failure']` 等不存在字段。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：当前步观测（8 维向量）
- action：当前步动作（0～3 整数）
- next_obs：下一步观测（8 维向量）
- info：必须为空（不可依赖任何字段）
- training_progress：仅在 prompt 明确允许时使用，默认禁用

禁止使用：
- original_reward（环境原始奖励，已屏蔽）
- official_reward（等同于 original_reward）
- 未声明的 info 字段（info 本身唯一允许就是空字典）
- 未声明的 obs 切片（仅限上述 8 维含义）

## 7. 可用于奖励函数的信号
- position: `next_obs[0]`（距垫水平距离），`next_obs[1]`（距垫垂直高度）
- velocity: `next_obs[2]`（水平速度），`next_obs[3]`（垂直速度）
- orientation: `next_obs[4]`（姿态角），`next_obs[5]`（角速度）
- contact: `next_obs[6]`（左腿触地），`next_obs[7]`（右腿触地）
- action/engine: `action` 值可用来惩罚引擎使用（如 main engine 点火的能耗）

## 8. 不确定或不可用的信号
- 原始奖励 `original_reward`：不可用。
- 任何来自 `info` 的成功/失败标志：不可用。
- 引擎推力大小、风扰等物理参数：观测空间中未暴露，不可用。
- 着陆垫准确坐标：已知相对坐标通过 obs[0,1] 可得，但着陆垫的具体尺寸、容许误差需通过奖励塑造间接处理，无直接标志。
