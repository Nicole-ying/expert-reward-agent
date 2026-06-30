# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
这是一个二维车辆/飞行器式控制任务。智能体需要接近中央目标区域，并尽量稳定地停在目标附近。次要目标包括减少发动机使用和更高效完成任务。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 核心目标是到达并稳定接近目标区域，属于导航/到达目标类任务；稳定、接触和燃料是附加约束，不改变主要任务类型。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32
- obs[0]: x_position，水平位置，相对目标区域
- obs[1]: y_position，垂直位置/高度，相对目标区域
- obs[2]: x_velocity，水平速度
- obs[3]: y_velocity，垂直速度
- obs[4]: body_angle，机体角度
- obs[5]: angular_velocity，角速度
- obs[6]: left_contact，左支撑接触标志，0/1
- obs[7]: right_contact，右支撑接触标志，0/1

## 4. 动作空间 action_space
- type: Discrete
- action 0: no_engine，不喷气
- action 1: left_orientation_engine，左/侧向姿态发动机
- action 2: main_engine，主发动机
- action 3: right_orientation_engine，右/侧向姿态发动机

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 可能存在“稳定/停止活动”类终止，但没有明确 success flag 传入 reward 函数。
- failure-like termination: 可能包括碰撞、越界、机体接触等，但没有明确 failure flag 传入 reward 函数。
- ambiguous termination: done/terminated 只有二值终止时，不能直接判断成功还是失败。
- truncation: 如果存在时间截断，也不能当作成功或失败。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []
- forbidden_or_uncertain_info_fields: success, failure, termination_reason, official_reward, original_reward

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs
- next_obs
- action
- training_progress 只有明确需要课程奖励时才用

禁止使用：
- original_reward
- official_reward
- info["success"] / info.get("success")
- info["failure"] / info.get("failure")
- info["termination_reason"]
- 未声明的 obs 切片，例如 obs[0:3]

## 7. 可用于奖励函数的信号
- position: obs[0], obs[1], next_obs[0], next_obs[1]
- velocity: obs[2], obs[3], next_obs[2], next_obs[3]
- orientation: obs[4], obs[5], next_obs[4], next_obs[5]
- contact: obs[6], obs[7], next_obs[6], next_obs[7]
- action/engine: action 可以反映是否使用发动机，但能耗项建议后续迭代再加

## 8. 不确定或不可用的信号
- explicit success flag
- explicit failure flag
- termination reason
- official reward
