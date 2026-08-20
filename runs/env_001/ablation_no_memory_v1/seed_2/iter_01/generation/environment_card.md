# 匿名环境理解卡片

## 1. 任务目标
控制一个 2D 飞行器从初始位置（靠近视口顶部中心）尽快飞向中央的目标着陆垫，并在垫上稳定停靠。学习过程中要求尽量少使用主引擎，同时保持姿态平稳、安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- obs[0]: x 坐标（相对于目标垫中心的水平距离）
- obs[1]: y 坐标（相对于目标垫高度的垂直距离）
- obs[2]: x 方向线速度
- obs[3]: y 方向线速度
- obs[4]: 机体角度（姿态朝向）
- obs[5]: 角速度
- obs[6]: 左侧支撑接触标志（0 或 1）
- obs[7]: 右侧支撑接触标志（0 或 1）

## 4. 动作空间 action_space
- type: Discrete (4 个动作)
- action 0: 无引擎（不做任何推进）
- action 1: 左侧姿态引擎（启动左侧转向引擎，产生力矩）
- action 2: 主引擎（启动主推进引擎）
- action 3: 右侧姿态引擎（启动右侧转向引擎，产生反方向力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
终止由以下任一条件触发（terminated = True）：
1. 机体碰撞地面或其他物体 crash_or_body_contact
2. 水平位置超出视口 horizontal_position_outside_viewport
3. 机体不再活跃或已稳定睡眠 body_not_awake_or_settled

- success-like termination: 无明显唯一成功标志；可能体满足“到达并在垫上稳定”时 body_not_awake_or_settled 为真，但也可能在其他位置提前睡眠，故不能直接当作成功。
- failure-like termination: crash_or_body_contact 和 horizontal_position_outside_viewport 可能为失败终止。
- ambiguous termination: body_not_awake_or_settled 可能对应成功着陆稳定，也可能对应非目标位置的过早睡眠。
- truncation: 无，truncated 始终为 False。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {}（info 为空字典）
- forbidden_or_uncertain_info_fields: info 无任何可用字段，不能从中提取成功/失败/原因。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- obs：当前观测数组
- next_obs：下一时刻观测数组
- action：当前采取的动作索引
- info 中明确允许的字段（此处 info 为空，所以无可使用字段）

禁止使用：
- original_reward（官方奖励）
- official_reward
- 未声明的 info 字段
- 未声明的 obs 切片含义
- training_progress（除非后续说明允许，默认为禁止）

## 7. 可用于奖励函数的信号
- position: obs[0] 相对水平距离、obs[1] 相对垂直距离
- velocity: obs[2] 和 obs[3] 的线速度大小
- orientation: obs[4] 机体角度、obs[5] 角速度
- contact: obs[6]、obs[7] 左右支撑接触标志（着陆垫信号）
- action/engine: 动作选择，尤其可以惩罚主引擎使用（action == 2）

## 8. 不确定或不可用的信号
- 官方奖励 original_reward 被禁止，不可还原
- info 无内容，不能获得 success/failure 标志，也不能获得任何评估提示
- 终止原因无显式返回，无法从 step 返回值区分具体终止类型
- 无外界提供的速度/加速度参考值，需从观测自行推导奖励项