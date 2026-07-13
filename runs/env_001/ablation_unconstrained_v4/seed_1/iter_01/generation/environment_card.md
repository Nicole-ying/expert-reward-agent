# 匿名环境理解卡片

## 1. 任务目标
这是一个二维飞行器轨迹优化任务。智能体控制一个带有主引擎和左右姿态引擎的飞行器，从视口顶部中心附近出发。主目标是**以尽可能快的速度到达并稳定降落到中央目标垫上**，同时**尽可能少地使用引擎推力**（节省燃料）。次要目标包括保持姿态稳定、在接触时速度尽可能小（软着陆）、避免主体碰撞或飞出边界。  
**不应混淆的目标**：单纯追求速度而不计燃料，或不关心姿态稳定只求触碰目标垫。

## 2. 任务类型选择
- **selected_route_id:** `navigation_goal_reaching`  
- **confidence:** `high`  
- **reason:** 核心目标是导航到固定的中心目标垫位置并稳定停靠，附属优化（速度、燃料、姿态）不改变任务根本类别。

## 3. 观察空间 observation_space
- **type:** `Box`
- **shape:** `[8]`
- **dtype:** `float32`（推测，因为标志位也以浮点数 1.0 / 0.0 表示）
- **各维度含义与奖励可用性：**

| 索引 | 名称 | 含义 | 奖励可用 |
|------|------|------|----------|
| 0 | `x_position` | 相对于目标垫中心的水平坐标，目标值为 0 | true |
| 1 | `y_position` | 相对于垫子高度的垂直坐标，目标值为 0 | true |
| 2 | `x_velocity` | 水平线速度 | true |
| 3 | `y_velocity` | 垂直线速度（正值向上，负值向下） | true |
| 4 | `body_angle` | 机体朝向角度 | true |
| 5 | `angular_velocity` | 角速度 | true |
| 6 | `left_support_contact` | 左支撑腿接触标志（1.0 或 0.0） | true |
| 7 | `right_support_contact` | 右支撑腿接触标志（1.0 或 0.0） | true |

## 4. 动作空间 action_space
- **type:** `Discrete`
- **n:** 4
- **动作含义：**

| 动作编号 | 名称 | 含义 |
|----------|------|------|
| 0 | `no_engine` | 什么都不做（无推力） |
| 1 | `left_orientation_engine` | 启动左姿态引擎（产生逆时针旋转力矩） |
| 2 | `main_engine` | 启动主引擎（产生向上的推力，抵消重力） |
| 3 | `right_orientation_engine` | 启动右姿态引擎（产生顺时针旋转力矩） |

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination:**  
  由 `body_not_awake_or_settled` 触发，表示机体已稳定并且不再活跃。若此时满足着陆条件（x≈0, y≈0, 两腿接触, 速度和角速度极低），则可判定为成功着陆。  
- **failure-like termination:**  
  - `crash_or_body_contact`：机体主体或非腿部部分触地/碰到障碍物，视为坠毁。  
  - `horizontal_position_outside_viewport`：水平位置超出视口边界，视为飞出任务区域。  
- **ambiguous termination:**  
  - 理论上 `body_not_awake_or_settled` 若发生在非目标位置（如悬停不动但并未着陆），虽然罕见，但被视作失败（因为未完成到达目标垫的任务）。  
- **truncation:**  
  未提及，此处未使用。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available:** `false`（`info` 为空，无明确成功标志）
- **explicit_failure_flag_available:** `false`
- **allowed_info_fields:** 无（`info = {}`，任何字段均不可用）
- **forbidden_or_uncertain_info_fields:** `success`, `failure`, `termination_reason` 等任何未在 step 源码中出现的字段均禁止使用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
- **允许使用：**  
  - `obs`（当前观测，shape [8]）  
  - `action`（离散动作 0-3）  
  - `next_obs`（下一观测，shape [8]，可用于检测终止状态或接触变化）  
  - `info` 中仅限 `{}`，实际无可用字段  
  - `training_progress` 未被明确允许，因此**不建议使用**  
- **禁止使用：**
  - `original_reward`
  - 任何 `official_reward`、`env.reward_range` 等
  - 未在 obs 中声明的信号

## 7. 可用于奖励函数的信号
- **位置信号:** `x_position` (goal=0), `y_position` (goal=0)  
- **速度信号:** `x_velocity`, `y_velocity`（希望着陆时接近 0）  
- **姿态与角速度:** `body_angle`, `angular_velocity`（着陆时应接近 0）  
- **接触信号:** `left_support_contact`, `right_support_contact`（两条腿都接触才满足软着陆）  
- **动作信号:** 离散动作编号，可用于推力消耗惩罚  
- **其他:** 可以从 `next_obs` 推断终止时的着陆成功（见 12 节）

## 8. 不确定或不可用的信号
- **直接的成功/