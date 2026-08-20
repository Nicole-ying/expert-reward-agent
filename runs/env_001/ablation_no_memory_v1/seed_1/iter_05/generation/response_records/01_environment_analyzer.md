# Response Record

# 匿名环境理解卡片

## 1. 任务目标
这是一个 2D 飞行器（或类似着陆器）轨迹优化任务。飞行器从视野上方偏向中心的位置出发，带有随机的初始作用力。目标是**尽快降落到中央的目标平台上，并在此过程中保持姿态稳定、速度轻微，实现安全着陆**，同时**尽可能少地使用主发动机推力**。换句话说，核心是到达指定平台并安全停稳，附属优化是减少燃料消耗和快速完成。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: 推断为 float32（位置/速度连续值，接触标志为 0.0/1.0 浮点）
- obs[0] (x_position): 飞行器水平坐标，相对于目标平台中心的偏移量
- obs[1] (y_position): 飞行器垂直坐标，相对于平台表面的高度
- obs[2] (x_velocity): 水平线速度
- obs[3] (y_velocity): 垂直线速度
- obs[4] (body_angle): 机身朝向角度（rad 或归一化角度）
- obs[5] (angular_velocity): 角速度
- obs[6] (left_support_contact): 左侧支撑接触标志，接触时为 1.0，否则 0.0
- obs[7] (right_support_contact): 右侧支撑接触标志，接触时为 1.0，否则 0.0

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0 (no_engine): 不启动任何发动机，飞行器仅受当前动量和重力/物理影响
- action 1 (left_orientation_engine): 启动左侧姿态发动机，产生旋转力矩（改变角速度/角度）
- action 2 (main_engine): 启动主发动机，产生向上的推力（影响 y 方向速度）
- action 3 (right_orientation_engine): 启动右侧姿态发动机，产生相反的旋转力矩

## 5. step 与终止条件分析

### 5.1 终止模式
根据代码和描述，终止条件为以下三种逻辑值的 **或**：
- **crash_or_body_contact**：飞行器主体发生碰撞（可能包括猛烈着陆或超出安全阈值的接触）
- **horizontal_position_outside_viewport**：水平位置超出视野范围
- **body_not_awake_or_settled**：飞行器“休眠”或已稳定停住（物理引擎判定速度/角速度极小）

具体成功/失败对应关系如下：
- success-like termination: 仅在 **body_not_awake_or_settled** 发生时，且同时满足**所有接触腿恰好着陆在平台上、速度/姿态处于安全范围内**的场景（但代码中未显式给出“安全”判定，仅能从任务描述和目标推断）。通常成功停稳后才触发该终止。
- failure-like termination: **crash_or_body_contact**（如果着陆时速度过高或姿态错误而导致碰撞）或 **horizontal_position_outside_viewport** 都会导致失败终止。需要注意的是，crash_or_body_contact 也可能包含“安全着陆后仍在接触状态”的情形，但由于安全着陆后很快就会进入 not_awake 状态，所以成功路径通常不会由该条件单独触发；但单纯基于终止条件的划分，**仅有 not_awake 而没有任何接触也可能属于成功（接触腿可能在平台之外的不安全情况？）**，这需要额外信息，因此直接使用终止条件判断成功与否并不完全可靠。
- ambiguous termination: 仅凭 terminated=true 和三个原始条件无法可靠区分成功与失败，因为**没有显式的 success 或 failure 标志**。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空字典 `{}`，没有任何 success 字段）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空）
- forbidden_or_uncertain_info_fields: 禁止使用任何 info 字段（如 info["success"]、info["failure"]、info["termination_reason"]）——它们不存在。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`：动作执行前的 8 维观测（numpy 数组或类似容器）
- `action`：执行的离散动作 id（int）
- `next_obs`：动作执行后的 8 维观测
- `info` 中存在的字段：当前 info 为空字典，**没有任何可用字段**（不允许假设添加）
- `training_progress`：除非明确提示允许使用，否则**不应使用**

禁止使用：
- `original_reward`（官方奖励已被屏蔽，禁止利用其还原或计算）
- 任何未在观测空间中声明的隐式信号（如内部碰撞类型、燃料量等）
- 未明确提供的 info 字段（包括但不限于 success, failure, termination_reason 等）

## 7. 可用于奖励函数的信号
- **position**: `obs[0]` (x 相对位置)、`obs[1]` (y 相对高度) 以及相应 `next_obs` 值
- **velocity**: `obs[2]` (vx)、`obs[3]` (vy) 以及变化值
- **orientation**: `obs[4]` (机体角度)、`obs[5]` (角速度)
- **contact**: `obs[6]` (左腿接触)、`obs[7]` (右腿接触) —— 可用于判断着陆状态
- **action/engine**: `action` 值可用于惩罚主发动机使用（action == 2）或姿态发动机使用，以鼓励节省燃料

## 8. 不确定或不可用的信号
- 官方原始奖励 (`original_reward`)：完全禁止使用
- 碰撞严重程度或碰撞类型：观测中无法获得（虽有 crash_or_body_contact 触发终止，但未传入观测或 info）
- 燃料剩余量：无相关观测
- 目标平台中心是否存在标志符：无
- 成功标志或失败原因（如是否软着陆、可着陆区域是否合法）：info 为空，禁止使用
- 任务的真实名称或已知基准（Lunar Lander 等）：禁止猜测或使用任何外部先验
