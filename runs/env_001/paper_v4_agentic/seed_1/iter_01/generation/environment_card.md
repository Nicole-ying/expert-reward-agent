# 匿名环境理解卡片

## 1. 任务目标

任务目标：控制一个2D飞行器从顶部初始位置尽快且节省燃料地降落到中央目标垫上，并稳定停靠（settle）。  
次目标：使用尽可能少的引擎推力（省燃料），同时保持飞行器姿态稳定（小角度）、降低相对速度、让两条支撑腿都安全接触目标垫。  
不应混淆的目标：不应追求原地悬停、单纯加速、或与垫子发生刚性碰撞。

## 2. 任务类型选择

selected_route_id: **navigation_goal_reaching**  
confidence: **high**  
reason: 核心任务是到达指定的唯一目标位置（目标垫），并且要稳定停靠在目标上；不存在多个等价目标或持续行进地形，不是单纯存活或探索。附属燃料优化和时间优化属于次要指标，不构成多目标公平冲突。

动力学子类型：**goal_approach_and_soft_contact**

## 3. 观察空间 observation_space

- type: Box
- shape: [8]
- dtype: float32（推断）
- obs[0]: x_position，水平坐标相对目标垫（目标垫为原点），reward_usable: true
- obs[1]: y_position，垂直坐标相对垫高度，reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，机体倾角，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左支撑腿接触标志（1.0/0.0），reward_usable: true
- obs[7]: right_support_contact，右支撑腿接触标志（1.0/0.0），reward_usable: true

## 4. 动作空间 action_space

- type: Discrete
- n: 4
- action 0: no_engine — 不点火，仅靠惯性运动
- action 1: left_orientation_engine — 点燃左侧姿态引擎（产生旋转力矩）
- action 2: main_engine — 点燃主引擎（产生推力，主要向上或向前）
- action 3: right_orientation_engine — 点燃右侧姿态引擎（产生反向旋转力矩）

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` — 当飞行器稳定停靠在目标垫上且处于休眠/静止状态时触发，这可能对应成功settle。
- **failure-like termination**: `crash_or_body_contact`（机体与障碍物或危险接触导致坠落/碰撞），`horizontal_position_outside_viewport`（水平位置超出视野范围，出界）。
- **ambiguous termination**: `body_not_awake_or_settled` 也可能是由于空中静止（hover）造成的，需结合位置、速度、接触信号区分成功与悬停。
- **truncation**: 未明确提及最大步数截断，但通常存在环境上限（不可用于奖励）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**
- explicit_failure_flag_available: **false**
- allowed_info_fields: `{}`（空字典，无任何可用字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，因为 step 返回空 `{}`。

> **间接推断路径**（derived_possible）：
> - 成功着陆：episode 被 `body_not_awake_or_settled` 终止，且此时 obs 满足 `|x_position|很小，|y_position|很小，|body_angle|小，left_contact==1 且 right_contact==1`。
> - 坠毁：`crash_or_body_contact` 触发，或 x_position 骤变伴随异常接触。
> - 出界：`horizontal_position_outside_viewport` 触发，或 x_position 绝对值超过阈值。
> 以上推断可在 reward 中利用 obs 信号构建成功/失败的密集奖励代理，但**不可**直接使用 done 标志或 info 中的显式 flag。

## 6. reward 函数接口契约

函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`（当前观测，numpy array）
- `action`（当前执行的动作，整数 0-3）
- `next_obs`（下一时刻观测，numpy array）
- `info` 中明确允许的字段（此处 info 为空，实际不可用）
- `training_progress`：仅当 prompt 明确允许时才使用，**本环境未允许，禁用**

禁止使用：
- `original_reward`（官方奖励，已屏蔽）
- 任何未声明的 info 字段（info 为空）
- 未声明的 obs 切片（以第