# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
控制一个近似车辆的刚体，从画面顶部中央附近出发，尽快到达并稳定降落在正中央的目标垫上，同时尽量少用引擎推力。  
智能体需要学会：
- 飞向目标区域；
- 降低速度、保持平稳姿态；
- 安全触垫并最终静止。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 要求移动到指定的目标垫位置并稳定停留，核心为导航型目标达成任务。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 推测为 float32（具体由环境内部实现确定）
- obs[0]: x_position — 相对目标垫中心的水平距离
- obs[1]: y_position — 相对垫上方某参考高度的垂直距离
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 机体方向角
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左支撑/接触标志（1.0 表示接触，0.0 表示未接触）
- obs[7]: right_support_contact — 右支撑/接触标志（1.0 表示接触，0.0 表示未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: 无引擎 — 不点火，仅靠惯性运动
- action 1: 左姿态引擎 — 点燃一个方向调整引擎（提供姿态修正推力）
- action 2: 主引擎 — 点燃主推进引擎（通常提供向上的推力）
- action 3: 右姿态引擎 — 点燃另一个方向调整引擎（与左姿态引擎相反）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 机体稳定且不再活动（`body_not_awake_or_settled` 为真），且未发生 crash 或越界。暗含左右接触点可能均已触垫，并处于静止状态。
- failure-like termination: 发生 crash 或与地面/非垫区域接触（`crash_or_body_contact`），或水平位置超出视口（`horizontal_position_outside_viewport`）。
- ambiguous termination: 机体休眠但可能并非在目标垫上（例如飘离很远后因外力停止），仅靠终止信号无法区分。
- truncation: 本环境不返回截断信号（step 返回 `False`），无基于步数的超时终止。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: [] (step 返回的 info 为空字典)
- forbidden_or_uncertain_info_fields: 任何 info 键均不允许使用；试图从 info 中读取 success/failure/termination_reason 等字段均不可靠，源码中不存在。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
参数约束：
- 允许使用：
  - `obs`：当前观察向量（8 维）
  - `action`：刚执行的动作索引（0~3）
  - `next_obs`：下一观察向量（8 维）
  - `info`：空字典，禁止依赖任何字段
  - `training_progress`：仅在 prompt 明确要求时才使用
- 禁止使用：
  - `original_reward`（即被遮盖的官方奖励值）
  - 任何未在 `obs`/`next_obs` 中定义的维度
  - info 的任意键
  - 环境内部隐藏状态或真实终止原因标志

## 7. 可用于奖励函数的信号
- 位置：x_position, y_position（可直接鼓励接近目标区域，如惩罚距离）
- 速度：x_velocity, y_velocity（可用于奖励减速、降低动能）
- 姿态：body_angle（鼓励保持竖直/水平的稳定姿态）
- 角速度：angular_velocity（抑制旋转）
- 接触：left_support_contact, right_support_contact（奖励安全触垫、惩罚单侧或过早接触）
- 动作/引擎：action 类型（可对燃料消耗进行惩罚，如对引擎点火的额外代价）
- 此外，`next_obs` 可与 `obs` 对比获得相邻状态变化。

## 8. 不确定或不可用的信号
- **明确的成功/失败标志**：info 中不存在，不可用；终止时为自主推导。
- **机体的“休眠”状态**：虽然作为终止条件之一，但无法在奖励函数中直接获取。只能通过观察向量中静止特征（速度≈0，接触标志为1）间接推测。
- **外部风的干扰**：源码中提到的 wind 已剔除，无法感知。
- **相对目标垫的精确距离**：obs 中给出的是坐标，未提供直接的 L2 距离，需自行计算。
- **游戏物理参数**：质量、推力大小、重力等均不可见。
