# 匿名环境理解卡片

## 1. 任务目标
控制一个 2D 飞行器从初始位置（接近视口顶部中心）出发，  
**尽可能快地到达并稳定停在中央的目标着陆垫上**，  
同时尽量减少引擎推力消耗。  
需要学习接近目标、降低速度、保持稳定姿态并安全接触垫面。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（连续值，最后两个分量为 0.0 或 1.0）
- obs[0]: x_position — 相对于目标垫中心的水平坐标
- obs[1]: y_position — 相对于垫面高度的垂直坐标
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 机体朝向角
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左支撑腿接触标志（1.0=接触，0.0=未接触）
- obs[7]: right_support_contact — 右支撑腿接触标志（1.0=接触，0.0=未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine — 不启动任何引擎，仅靠惯性
- action 1: left_orientation_engine — 启动左侧姿态引擎
- action 2: main_engine — 启动主引擎（向下推力）
- action 3: right_orientation_engine — 启动右侧姿态引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（机体稳定/沉睡，表明已安全着陆）
- failure-like termination: crash_or_body_contact（机身非支撑部位触碰地面）、horizontal_position_outside_viewport（水平位置超出视口边界）
- ambiguous termination: 三个条件通过 `or` 逻辑连接，没有独立 success/failure 标志
- truncation: 由环境外部步数限制处理，此处 `step` 不返回截断信号（return 的第四个值为 False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {}（info 为空字典，无任何可用字段）
- forbidden_or_uncertain_info_fields: 全部 info 字段均不可用；不能假设存在 success/failure 等键

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- obs（当前观察）
- action（当前动作）
- next_obs（下一时刻观察）
- info 中明确允许的字段（本环境 info 为空，因此不可使用任何 info 内容）
- training_progress（本 prompt 未明确允许使用，默认禁止）

禁止使用：
- original_reward（被掩蔽的官方奖励）
- official_reward
- 任何未声明的 info 字段
- 任何其他未声明的环境内部变量（如发动机推力、风等）

## 7. 可用于奖励函数的信号
- position: next_obs[0] (x 相对目标), next_obs[1] (y 相对垫高)
- velocity: next_obs[2], next_obs[3]
- orientation: next_obs[4] (机体角度), next_obs[5] (角速度)
- contact: next_obs[6], next_obs[7]（支撑腿是否触垫）
- action/engine: 动作选择，可对引擎使用（action 1,2,3）施加负代价
- 相邻状态变化：如位置/速度/角度的变化量，可计算趋近或平稳性

## 8. 不确定或不可用的信号
- 官方原始奖励（被掩蔽）
- 任何 info 字典内容（空的）
- 引擎推力大小、燃料消耗量等内部物理量（未被观察空间提供）
- 任务是否成功/失败的显式布尔值