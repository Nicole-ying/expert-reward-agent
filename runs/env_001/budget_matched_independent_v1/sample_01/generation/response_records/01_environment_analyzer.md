# Response Record

# 匿名环境理解卡片

## 1. 任务目标
这是一个 2D 飞行器着陆轨迹优化任务。飞行器从视野上方的随机初始位置出发，受到初始随机力扰动。核心目标是**控制飞行器安全、平稳地降落在画面中央的目标平台上**。具体而言：
- **主要目标**：使飞行器到达平台正上方并实现稳定着陆（水平与垂直速度趋于零、姿态竖直、双腿接触平台）。
- **附属目标**：尽可能快地完成着陆，同时**最小化发动机推力使用**（节省燃料）。
- **不可混淆**：这不是纯粹的导航任务——着陆后的稳定性（速度、姿态、接触）是任务成功的关键组成部分；不是典型的持续运动控制任务，因为它以到达并停止在固定目标点为终点。

## 2. 任务类型选择
- **selected_route_id**: `navigation_goal_reaching`
- **confidence**: high
- **reason**: 任务的核心是到达指定的固定目标位置（目标平台）并稳定接触，附属有速度、姿态、能耗要求。这与“核心是到达指定目标位置”的导航‑目标到达族最为吻合，且后续动力学子类型进一步细化了“软接触”特性。

**动力学子类型 dynamics_subtype**: `goal_approach_and_soft_contact`

## 3. 观察空间 observation_space
- **type**: Box
- **shape**: (8,)
- **dtype**: float32 （连续值，接触标志为 0.0/1.0 浮点）
- 各索引含义：
  - `obs[0]`: `x_position` —— 飞行器相对于目标平台中心的水平距离，reward_usable: true
  - `obs[1]`: `y_position` —— 飞行器相对于目标平台高度的垂直距离，reward_usable: true
  - `obs[2]`: `x_velocity` —— 水平线速度，reward_usable: true
  - `obs[3]`: `y_velocity` —— 垂直线速度，reward_usable: true
  - `obs[4]`: `body_angle` —— 飞行器躯干朝向与竖直方向的夹角，reward_usable: true
  - `obs[5]`: `angular_velocity` —— 角速度，reward_usable: true
  - `obs[6]`: `left_support_contact` —— 左支撑腿与平台或地面发生接触的标志（1.0 接触 / 0.0 未接触），reward_usable: true
  - `obs[7]`: `right_support_contact` —— 右支撑腿接触标志，reward_usable: true

## 4. 动作空间 action_space
- **type**: Discrete
- **n**: 4
- 动作含义：
  - `action 0`：无引擎 —— 不施加任何推力，仅靠惯性运动，reward_usable: true（可用于燃料惩罚）
  - `action 1`：左姿态发动机 —— 向左上方喷射，产生逆时针旋转力矩，reward_usable: true
  - `action 2`：主发动机 —— 向正下方喷射，产生向上的推力，同时也是消耗燃料的主要动作，reward_usable: true
  - `action 3`：右姿态发动机 —— 向右上方喷射，产生顺时针旋转力矩，reward_usable: true

## 5. step 与终止条件分析
### 5.1 终止模式
终止条件组合：`crash_or_body_contact or horizontal_position_outside_viewport or body_not_awake_or_settled`
- **success‑like termination**：`body_not_awake_or_settled` 在飞行器双腿接触目标平台且速度、姿态足够稳定时，由物理引擎自动进入休眠/稳定状态触发。这很可能对应成功着陆。
- **failure‑like termination**：
  - `crash_or_body_contact`：飞行器主体与地面或平台以外物体发生了不安全的接触（如坠毁、侧翻等）。
  - `horizontal_position_outside_viewport`：飞行器水平飞出视口边界。
- **ambiguous termination**：`body_not_awake_or_settled` 也可能由其他原因（如翻倒后无动能）触发，无法从终止本身直接区分成功与失败。
- **truncation**：未提到 episode 长度限制。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false
- **explicit_failure_flag_available**: false
- **allowed_info_fields**：从 masked step source 可知 `info` 为一个空字典 `{}`，因此奖励函数内部 **不可依赖 info**
- **forbidden_or_uncertain_info_fields**：`success`, `failure`, `termination_reason`, 任何与环境成功/失败有关的预定义标志均不存在或不可靠

## 6. reward 函数接口契约
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
**允许使用**：
- `obs`：上一步观测（np.ndarray）
- `action`：刚执行的动作（int）
- `next_obs`：新观测（np.ndarray）
- `training_progress` 仅在 prompt 明确允许时方可使用（当前无此类说明，**禁用**）
- `info`：仅可使用任务说明中明确声明的字段（当前为空，故不可用）

**禁止使用**：
- `original_reward`（官方奖励已掩蔽，不得参照或泄露）
- 任何未在观测空间说明中声明的 obs 切片（环境可能返回额外噪声，但不应依赖）
- 任何未明确允许的 info 字段

## 7. 可用于奖励函数的信号
所有信号均从 `obs` 或 `next_obs`（以及 `action`）中获得。

- **position**：
  - `x_pos = next_obs[0]` （或 `obs[0]`）
  - `y_pos = next_obs[1]`
  - 可构造到目标点的**二维距离**：`dist = sqrt(x_pos² + y_pos²)`
- **velocity**：
  - `x_vel = next_obs[2]`
  - `y_vel = next_obs[3]`
  - 可用于速度惩罚、软着陆条件
- **orientation**：
  - `angle = next_obs[4]`
  - 可构造角度惩罚（目标接近 0）
- **angular velocity**：
  - `ang_vel = next_obs[5]`
  - 可辅助评估旋转稳定性
- **contact**：
  - `left_contact = next_obs[6]`
  - `right_contact = next_obs[7]`
  - 用于判断是否已接触平台/地面，作为稳定着陆的标志
- **action/engine**：
  - `action` 值可用于**燃料消耗惩罚**：动作 1/2/3 表示使用发动机，2（主发动机）消耗最大。

## 8. 不确定或不可用的信号
- **明确的成功/失败标志**：不存在，无法用于 sparse 奖励。
- **接触类型**：`left_contact`、`right_contact` 无法区分是与目标平台接触还是与地面/障碍物接触。因此仅靠接触标志不能完全保证“正确着陆”。
- **地形/地面高度**：观测不包含地面绝对位置，
