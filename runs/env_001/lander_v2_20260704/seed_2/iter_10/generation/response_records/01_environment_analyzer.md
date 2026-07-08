# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
这是一个 2D 飞行器着陆任务。飞行器从靠近视口顶部中央的位置出发，受到初始随机力。目标是最快速度到达中央目标着陆区并平稳停稳，同时尽可能少使用引擎推力。智能体需要学会接近目标、减速、保持稳定姿态并安全着陆（接触着陆区）。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float64（按一般 Box 推测）
- obs[0]: x_position —— 飞行器相对于目标着陆区中心的水平坐标
- obs[1]: y_position —— 飞行器相对于着陆区高度的垂直坐标（越接近 0 表示降落到着陆面高度）
- obs[2]: x_velocity —— 水平线速度
- obs[3]: y_velocity —— 垂直线速度
- obs[4]: body_angle —— 飞行器朝向角（可能相对于竖直方向，0 为竖直向上）
- obs[5]: angular_velocity —— 角速度
- obs[6]: left_support_contact —— 左侧支撑接触标志（1.0 接触，0.0 未接触）
- obs[7]: right_support_contact —— 右侧支撑接触标志（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine —— 不点火，无推力
- action 1: left_orientation_engine —— 点燃一个方向引擎（产生侧向/旋转推力，可能用于调整姿态）
- action 2: main_engine —— 点燃主引擎（产生主要推力，通常用于减速或悬停）
- action 3: right_orientation_engine —— 点燃另一个方向引擎（与 action 1 相对）

## 5. step 与终止条件分析
### 5.1 终止模式
terminated 在以下任一条件为真时成立：
1. **crash_or_body_contact**：可能代表机体某部位与地面/障碍物发生危险碰撞（非着陆区接触），通常为失败。
2. **horizontal_position_outside_viewport**：水平位置超出视口边界，通常为失败。
3. **body_not_awake_or_settled**：机体处于非唤醒状态（可能已稳定停止）。这一条件可能包含两种情况：① 成功着陆并稳定；② 机体掉出视口后进入睡眠。因此该终止状态本身**不直接指示成功或失败**。

- success-like termination: 当 `body_not_awake_or_settled` 为真，且同时满足：位置接近目标 (x ≈ 0, y ≈ 0)，速度接近 0，至少一侧接触标志为 1.0 时，很可能为成功着陆。但无法仅从该 flag 直接判定。
- failure-like termination: `crash_or_body_contact`、`horizontal_position_outside_viewport`。
- ambiguous termination: `body_not_awake_or_settled`（需要结合观测才能推断是否为成功）。
- truncation: 代码中未见时间截断，终止全部由 terminated 驱动。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 没有可用的 info 字段（step 返回的 info 为空字典 `{}`）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，因为 info 为空

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观测）
- action（动作）
- next_obs（下一观测）
- info 中没有可用字段，所以实质上不可使用 info

禁止使用：
- original_reward（原始奖励被屏蔽，严禁引入）
- official_reward（任何形式的官方奖励）
- training_progress（prompt 未明确允许）
- info 中未声明的字段（全部禁止）
- 任何未在观测空间中声明的 obs 切片（全部 8 个索引均已声明，可使用）

## 7. 可用于奖励函数的信号
- position: obs[0] (x), obs[1] (y) —— 相对目标的水平/垂直位置，可用于接近奖励
- velocity: obs[2] (vx), obs[3] (vy) —— 可用于速度惩罚，鼓励软着陆
- orientation: obs[4] (angle), obs[5] (angular vel) —— 可用于姿态惩罚，鼓励稳定竖直
- contact: obs[6], obs[7] —— 着陆接触标志，可用于判断是否成功着地
- action/engine: action ∈ {0,1,2,3} —— 可用于燃料消耗惩罚（非零动作施加负奖励）
- 下一观测 next_obs 中的对应信号也可使用（如位置变化、速度变化等）

## 8. 不确定或不可用的信号
- 显式的成功/失败标志（info 中无此字段）
- crash 具体类型（无法区分是碰撞还是单纯躯体接触）
- 原始的 official_reward（被屏蔽，不可用）
- 任何未在观测空间中定义的变量（如环境内部物理量）
