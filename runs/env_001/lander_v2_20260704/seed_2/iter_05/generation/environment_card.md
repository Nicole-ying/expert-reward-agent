# Env_001 环境理解卡片

## 1. 任务目标
这是一个 2D 飞行器/着陆器轨迹优化任务。主体从画面顶部中央附近出发，带有随机初始速度。  
核心任务：**尽快到达并稳定著陆在画面中央的目标着陆垫上**，同时**尽量减少主引擎推力消耗**。  
代理需学会向目标靠近、减速、保持正确姿态并安全接触垫面。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（其中 contact 标志为 0.0 或 1.0）
- obs[0]: x 坐标，相对于目标着陆垫中心的水平位置
- obs[1]: y 坐标，相对于着陆垫高度的垂直位置
- obs[2]: 水平线速度 vx
- obs[3]: 垂直线速度 vy
- obs[4]: 机身朝向角（弧度）
- obs[5]: 角速度
- obs[6]: 左支撑腿接触标志（1.0 表示接触，0.0 未接触）
- obs[7]: 右支撑腿接触标志（1.0 表示接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: `no_engine` – 不點火，仅靠重力/惯性漂移
- action 1: `left_orientation_engine` – 点燃左侧姿态調整引擎
- action 2: `main_engine` – 点燃主引擎（向下推力，消耗燃料）
- action 3: `right_orientation_engine` – 点燃右侧姿态調整引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- **成功型终止**：`body_not_awake_or_settled`（机体静止/镇定）—— 当它发生且左右支撑腿均接触垫面时，可视为成功着陆。但单独出现时也可能是中途静止或悬空失败，需结合接触信号确认。
- **失败型终止**：
  - `crash_or_body_contact`：机体除着陆腿以外的部分碰撞地面或障碍物，或发生破坏性碰撞。
  - `horizontal_position_outside_viewport`：机体水平方向飞出视野/安全区。
- **模糊终止**：无特殊额外模式，上述三种为唯一终止触发条件。
- **truncation**：未指定最大步长，但通常会有时间上限（若存在，算作截断而非成功/失败）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （无，`info` 恒为空字典 `{}`）
- forbidden_or_uncertain_info_fields: 任何 `info` 字段均不可用；官方原始奖励被屏蔽，不可使用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- `obs`：上一步观察（shape (8,)）
- `action`：刚刚执行的动作（0～3）
- `next_obs`：当前步观察（shape (8,)）
- `info`：已确认恒为空，不提供有效信息
- `training_progress`：仅在明确指示时才可使用，本任务中未要求，**默认禁止**

禁止使用：
- `original_reward`（官方奖励被屏蔽，严禁泄露）
- 任何未在上述清单中声明的 `info` 字段
- 任何未在观察空间说明中包含的 `obs` 切片
- 直接或间接还原官方奖励

## 7. 可用于奖励函数的信号
- 位置：`obs[0]`（x）、`obs[1]`（y），表示与目标垫的相对位置，理想为目标点 (0, 0)
- 线速度：`obs[2]`（vx）、`obs[3]`（vy），理想着陆时接近 0
- 姿态：`obs[4]`（倾角），理想接近 0 （竖直）
- 角速度：`obs[5]`，理想接近 0
- 接触：`obs[6]`、`obs[7]`，左右腿是否着地，两腿同时接触可判断稳定着陆
- 动作/引擎使用：`action` 是否为 2（主引擎）可作为燃料消耗的惩罚信号；动作 1,3 可引导姿态修正

## 8. 不确定或不可用的信号
- 官方原始奖励被完全屏蔽，无法参考
- `info` 为空，不提供任何成功/失败/碰撞类型等辅助信息
- “crash_or_body_contact” 终止条件的具体判定逻辑隐藏，只能通过接触标志与是否终止间接推断：如果终止时 `obs[6]` 或 `obs[7]` 为 1 但 `crash` 为真，可能指示错误接触；若同时满足 `body_not_awake_or_settled` 且双腿接触，则大概率成功
- 环境内部的重力、推力大小、视野边界、物理参数等细节不可直接观测，只能通过位置/速度变化间接感知