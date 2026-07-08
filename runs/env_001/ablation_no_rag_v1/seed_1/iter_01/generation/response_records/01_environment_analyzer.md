# Response Record

# 匿名环境理解卡片

## 1. 任务目标
一个 2D 飞行器从视口顶部中心附近随机受力启动，需要尽快并安全地降落到中央目标平台（target pad）上并保持稳定。任务鼓励以最快的速度到达目标，同时尽量节约主引擎推力，并维持合理的姿态与轻柔的触地。核心目标是到达并稳定在目标平台上。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: 浮点数（实际可能为 float32），其中索引 6、7 为二值标志（0.0 / 1.0）
- obs[0]: x_position — 相对于目标平台的水平距离
- obs[1]: y_position — 相对于目标平台高度的垂直距离
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 机体方向角
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左支撑腿接触标志（1.0 表示接触，0.0 表示未接触）
- obs[7]: right_support_contact — 右支撑腿接触标志（1.0 表示接触，0.0 表示未接触）

## 4. 动作空间 action_space
- type: Discrete (4 个动作)
- action 0: no_engine — 不点火，依靠惯性滑行
- action 1: left_orientation_engine — 点火一个方向舵引擎，通常产生逆时针旋转或平移分量
- action 2: main_engine — 点火主引擎，产生向上或沿机体方向的推力
- action 3: right_orientation_engine — 点火另一个方向舵引擎，通常产生顺时针旋转或平移分量

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 身体不再醒来或处于稳定状态（body_not_awake_or_settled），且大概率双腿均已接触平台（left_support_contact==1.0 且 right_support_contact==1.0），位置和速度都接近零。但没有显式的 success 标志。
- **failure-like termination**:
  1. crash_or_body_contact — 机体与其他区域（如地面、山体）发生碰撞；由于合法接触仅有双腿接触平台，任何其他身体接触均视为失败。
  2. horizontal_position_outside_viewport — 水平方向超出视口边界。
- **ambiguous termination**: body_not_awake_or_settled 但缺少双腿接触或位置偏差过大，可能是未成功着陆的静止状态（如漂浮在空中或仅部分接触平台达到力学平衡）。
- **truncation**: 代码中未见时间步截断，但训练时可能存在 max_episode_steps，此处不展开。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（step 返回 info 为空字典）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用；不能依赖任何隐含的 success / failure 标志。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs: 形状 [8] 的当前观测（或任意合法切片）
- action: 当前执行的动作索引 (0~3)
- next_obs: 下一时刻观测 [8]
- info: 只允许使用明确声明的字段（目前为空，不推荐使用任何 info 内容）
- training_progress: 只有 prompt 明确允许时才使用（本任务未声明，应禁止）

禁止使用：
- original_reward — 原始奖励被屏蔽，严禁直接或间接使用
- official_reward 或任何等价物
- 任何未在 observation_space 中声明的 obs 切片
- 任何未在 info 中明确声明并允许的字段

## 7. 可用于奖励函数的信号
- **position**: 可用 obs[0] (x 相对距离) 和 obs[1] (y 相对高度)，以及对应的 next_obs 值；可以用来衡量接近目标的进度。
- **velocity**: obs[2], obs[3] 的线速度，可用于鼓励低速着陆。
- **orientation**: obs[4] (角度) 和 obs[5] (角速度)，可用于惩罚倾斜或旋转。
- **contact**: obs[6], obs[7] 的左右支撑腿接触标志，可用于判断是否到达平台，并奖励同时接触。
- **action/engine**: 动作索引本身可用于计算引擎使用惩罚，例如对 main_engine (action 2) 或姿态调节引擎 (action 1, 3) 设置负奖励以节约燃料。

## 8. 不确定或不可用的信号
- 没有官方的 success / failure 终止标记，不能在 reward 中直接依赖 “成功” 或 “失败” 的布尔值。
- info 字段完全为空，任何尝试读取 info 内部键的行为都是违规的。
- 原始的 official_reward 被完全屏蔽，不应尝试推测或恢复。
- body_not_awake_or_settled 是终止条件的一部分，但无法直接从 obs 中获取，不能用于 reward 计算。
