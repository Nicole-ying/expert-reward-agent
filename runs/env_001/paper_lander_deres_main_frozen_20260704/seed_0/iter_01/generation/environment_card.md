# Env_001 环境理解卡片

## 1. 任务目标
环境模拟一个 2D 飞行器轨迹优化问题。一个刚体从视口顶部中心附近出发，带有随机初始扰动力。目标是**尽可能快地飞到中央目标平台并平稳降落**，同时尽量少用引擎推力。智能体需要学会靠近目标、减小速度、保持稳定姿态，并安全接触平台。

简化为一句话：以最小的燃料消耗和最短时间，精准降落到固定的目标平台上并稳定下来。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (8,)
- dtype: float64 或 float32 (具体取决于底层实现，均为浮点)
- obs[0]: x_position —— 飞行器水平位置相对于目标平台的水平偏移
- obs[1]: y_position —— 飞行器垂直位置相对于平台高度
- obs[2]: x_velocity —— 水平线速度
- obs[3]: y_velocity —— 垂直线速度
- obs[4]: body_angle —— 机体朝向角
- obs[5]: angular_velocity —— 机体旋转角速度
- obs[6]: left_support_contact —— 左侧支撑点是否接触（1.0 接触，0.0 未接触）
- obs[7]: right_support_contact —— 右侧支撑点是否接触（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine —— 不点火，无推力
- action 1: left_orientation_engine —— 点燃左侧姿态调节发动机（产生旋转力矩）
- action 2: main_engine —— 点燃主发动机（产生与机体角度相关的推力，通常用于减速/悬浮）
- action 3: right_orientation_engine —— 点燃右侧姿态调节发动机（产生相反旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  `body_not_awake_or_settled` —— 机体速度与角速度极低且可能已接触，视为稳定着陆。
- failure-like termination:  
  `horizontal_position_outside_viewport` —— 水平位置超出边界，飞行器丢失或失控。
- ambiguous termination:  
  `crash_or_body_contact` —— 字面上包含“撞击”和“身体接触”，可能包含不安全的碰撞（失败）或刚好接触平台但未稳定（失败）甚至安全接触（成功）。由于无法从返回值直接区分，除非结合其他状态判断，否则不可作为干净的成败信号。
- truncation:  
  无。step 返回元组中 truncated 恒为 False，说明没有时间步上限截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 恒为空字典 `{}`）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不存在，不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 当前观察（8 维数组）
- action: 执行的离散动作（0~3）
- next_obs: 下一时刻观察（8 维数组）
- info 中明确允许的字段：无（info 恒为空，禁止访问其中任何字段）
- training_progress: 仅当 prompt 明确允许时才可使用，本次未授权

禁止使用：
- original_reward: 已被掩码的官方奖励，严禁读取或模仿
- official_reward: 同 original_reward
- 未声明的 info 字段
- 未声明的 obs 切片含义（只能使用本卡片第 3 节列出的索引含义）

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]) 和 y_position (obs[1])，可计算到目标的距离
- velocity: x_velocity (obs[2]) 和 y_velocity (obs[3])，可评估降落平稳性
- orientation: body_angle (obs[4]) 和 angular_velocity (obs[5])，可衡量姿态稳定性
- contact: left_support_contact (obs[6]) 和 right_support_contact (obs[7])，可判断是否已经接触、是否双足着地
- action/engine: 当前动作是否使用引擎（0为非零推力，2为主推，1、3为姿态推），可评估燃料消耗

## 8. 不确定或不可用的信号
- original_reward: 被彻底掩码，不可用
- info 字典内任何信号：不可用（因为恒为空）
- crash_or_body_contact 的准确含义：无法区分成功/失败，不能作为直接的奖励来源，仅可在终止时结合其他状态推测
- 外界风速或其他干扰力：在 step 源代码中被省略，无法确定其存在及影响