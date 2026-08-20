# 匿名环境理解卡片

## 1. 任务目标
这是一个 2D 飞行器（类似着陆器）轨迹优化任务。主体从接近画面顶部中心的位置开始，并带有一个随机初始力。任务是**尽快到达画面中央的固定目标平台并稳定停驻**，同时**尽量少使用引擎推力**。智能体需要学会：朝目标接近、减速、保持稳定朝向、安全接触并稳定在平台上。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (8,)
- dtype: float32
- obs[0]: x_position – 飞行器相对于目标平台中心的水平坐标
- obs[1]: y_position – 飞行器相对于平台高度（pad height）的垂直坐标
- obs[2]: x_velocity – 水平线速度
- obs[3]: y_velocity – 垂直线速度
- obs[4]: body_angle – 机体朝向角度
- obs[5]: angular_velocity – 角速度
- obs[6]: left_support_contact – 左侧支撑脚接触标志（0.0 或 1.0）
- obs[7]: right_support_contact – 右侧支撑脚接触标志（0.0 或 1.0）

## 4. 动作空间 action_space
- type: Discrete (4 个动作)
- action 0: no_engine – 不启动任何引擎，仅凭惯性运动
- action 1: left_orientation_engine – 点燃一侧的姿态调整引擎（产生角力矩）
- action 2: main_engine – 点燃主引擎（产生向下的推力）
- action 3: right_orientation_engine – 点燃另一侧的姿态调整引擎（相反方向角力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: **body_not_awake_or_settled** 可能表示飞行器稳定停驻在目标平台上（如果位置在目标附近且速度很低），但需要结合观测判断，无显式成功标记。
- failure-like termination: **crash_or_body_contact**（非平台且因撞击导致终止）、**horizontal_position_outside_viewport**（水平飞出边界）都极可能是失败。
- ambiguous termination: body_not_awake_or_settled 本身不区分停在平台上还是停在平台外，需要观测才能确定。
- truncation: 可能由最大步数截断，但源码中未体现，info 为空，无法区分。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 在 step 返回中为空字典 `{}`）
- forbidden_or_uncertain_info_fields: 任何 info 键均不可用（因返回 `{}`）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs ：当前状态（terminate 前一步的观测，但注意调用时机可能为 executed action 之后）
- action：执行的离散动作（0-3）
- next_obs：执行动作后的下一状态（即 step 返回的 observation）
- info 中目前无任何字段可用

禁止使用：
- original_reward（被屏蔽的官方奖励）
- 任何未声明的 info 字段
- training_progress（任务描述未明确允许使用）

## 7. 可用于奖励函数的信号
- position: obs[0] x_position（相对目标水平偏移）、obs[1] y_position（相对平台高度）
- velocity: obs[2] x_velocity, obs[3] y_velocity（速度项，可用于惩罚高速或奖励低速靠近）
- orientation: obs[4] body_angle（朝向角度，希望尽量竖直）
- angular velocity: obs[5] angular_velocity（平稳性）
- contact: obs[6] left_support_contact, obs[7] right_support_contact（接触指示，可能用于判断着陆状态）
- action/engine: action 所代表的引擎使用情况（可惩罚使用主引擎或姿态引擎）

## 8. 不确定或不可用的信号
- 接触信号（obs[6], obs[7]）不能直接等同于成功停驻，因为飞行器可能接触非平台的表面或被卡住，无法区分“正确平台上的双足接触”。
- 没有明确的“目标是否达到”的布尔标志；信息中无 success/failure。
- 无法访问官方的原始奖励或组成它的中间信号（如 shaping 项）。