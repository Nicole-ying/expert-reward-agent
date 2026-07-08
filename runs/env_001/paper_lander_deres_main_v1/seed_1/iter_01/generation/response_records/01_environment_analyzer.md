# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
本环境是一个2D飞行器/着陆器轨迹优化任务。飞行器初始位于画面顶部中央附近，带有随机初始作用力。  
**核心目标**：飞行器应尽快且平稳地降落在场地中央的“目标垫”上，并保持稳定的姿态与相对静止。  
**附属约束**：尽可能少地使用引擎推力，但这不是与核心目标冲突的多目标优化任务，而是围绕“高效着陆”的自然偏好。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: 推断为 float (位置、速度、角度、角速度为连续值，接触标志用 1.0/0.0 表示)  
- obs[0]: x_position — 飞行器相对于目标垫中心的水平距离 (接近0表示水平对齐)  
- obs[1]: y_position — 飞行器相对于目标垫高度的垂直距离 (接近0表示正好在垫面高度)  
- obs[2]: x_velocity — 水平线速度  
- obs[3]: y_velocity — 垂直线速度  
- obs[4]: body_angle — 机体倾斜角 (接近0表示竖直)  
- obs[5]: angular_velocity — 角速度  
- obs[6]: left_support_contact — 左支撑脚触地标志 (1.0 触碰, 0.0 未触碰)  
- obs[7]: right_support_contact — 右支撑脚触地标志 (1.0 触碰, 0.0 未触碰)

## 4. 动作空间 action_space
- type: Discrete (4)  
- action 0: no_engine — 不点火，无推力输出  
- action 1: left_orientation_engine — 侧向引擎喷火，产生侧向推力（用于调整机头朝向/水平速度）  
- action 2: main_engine — 主引擎喷火，产生向上的主要推力（用于减速/悬停）  
- action 3: right_orientation_engine — 与左侧引擎对称的侧向喷火

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  `body_not_awake_or_settled` 触发时，若飞行器双支撑脚均触地 (`left_support_contact == 1.0` 且 `right_support_contact == 1.0`)、水平/垂直位置接近 0、速度很小、机体角度近乎竖直，则很可能是一次成功着陆。  
- failure-like termination:  
  - `crash_or_body_contact` — 机体与地面非目标区域剧烈接触（如翻倒、撞击地面），通常视为失败。  
  - `horizontal_position_outside_viewport` — 飞出水平边界，视为失败。  
- ambiguous termination:  
  `body_not_awake_or_settled` 也可能出现在非成功场景（例如飞行器靠某种方式静止在错误位置），因此单靠终止信号无法完全确定成功与否，需要结合观察信号自行判定。  
- truncation: 源信息中未提及 episode 步长上限，因此无显式 truncation；但实际使用中可能存在外部限制（如最大步数），非环境内建。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: (空) — 该环境 `info` 返回空字典 `{}`，因此 `info` 不提供任何额外字段。  
- forbidden_or_uncertain_info_fields: `info` 所有字段均不可用，`info["success"]`、`info["termination_reason"]` 等均不存在。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- `obs` (上一个状态的完整8维观察)
- `action` (当前执行的动作)
- `next_obs` (动作执行后的新状态)
- **无** — `info` 为空字典，禁止假定任何字段存在。
禁止使用：
- `original_reward` — 被遮蔽，不可用。
- 任何 `info` 字段。
- `training_progress` — 未授权，不可用。
- 任何未声明的 `obs` / `next_obs` 切片维度外信息。

## 7. 可用于奖励函数的信号
从 `next_obs` 中可直接提取的、可量化的信号包括：
- **位置**：`next_obs[0]` (x), `next_obs[1]` (y) — 表示与目标垫的相对距离，可用于塑造接近/居中的奖励。  
- **速度**：`next_obs[2]` (vx), `next_obs[3]` (vy) — 可用于惩罚过高触地速度。  
- **姿态**：`next_obs[4]` (角度) — 可用于奖励竖直姿态（接近0）。  
- **角速度**：`next_obs[5]` — 可用于惩罚快速旋转。  
- **接触状态**：`next_obs[6]`, `next_obs[7]` — 双脚同时触地可作为成功着陆的有力判据，也可用于塑造中间奖励。  
- **动作/引擎使用**：`action` 为非零（使用引擎）时，可考虑付出燃油成本。

## 8. 不确定或不可用的信号
- `info` 全部内容（为空）  
- 任何显式的成功/失败标记  
- 官方的 `original_reward`（已遮蔽，禁止重建）  
- 超出 8 维的观察值  
- 真实的连续时间步序号或进度信息（除非外部提供，但未被环境声明可用）
