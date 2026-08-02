# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本任务是一个 2D 类飞行器着陆优化问题。智能体初始位于画面顶部中央附近，并受到随机扰动。核心目标是尽快让飞行器安全、稳定地降落在中央的目标垫上，同时尽量减少发动机推力消耗。智能体需要学会向目标靠近、在靠近过程中减速、保持稳定姿态，并在双脚平稳接触目标垫后安全停止（settle）。次要但明确的目标是“快速”与“低能耗”，但这些只作为附属优化，不能取代主目标——必须成功安全着陆。

## 2. 任务类型选择
- selected_route_id: navigation_goal_reaching  
- confidence: high  
- reason: 任务明确要求“到达并稳定在中央目标垫上”，存在明确的固定目标位置，其他要求（快速、节能）是辅助性的，不构成等权重多目标冲突。因此核心任务是目标到达，匹配 navigation_goal_reaching。

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: 可能是 float32（具体由环境决定，假定额定 float）  
- obs[0]: `x_position`，相对目标垫的水平坐标。reward_usable: true  
- obs[1]: `y_position`，相对目标垫高度的垂直坐标。reward_usable: true  
- obs[2]: `x_velocity`，水平线速度。reward_usable: true  
- obs[3]: `y_velocity`，垂直线速度。reward_usable: true  
- obs[4]: `body_angle`，机体朝向角度。reward_usable: true  
- obs[5]: `angular_velocity`，角速度。reward_usable: true  
- obs[6]: `left_support_contact`，左脚接触标志（0.0 或 1.0）。reward_usable: true  
- obs[7]: `right_support_contact`，右脚接触标志（0.0 或 1.0）。reward_usable: true

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- action 0: `no_engine` – 不点火  
- action 1: `left_orientation_engine` – 点燃左侧姿态发动机（产生旋转力矩）  
- action 2: `main_engine` – 点燃主发动机（产生向上的推力，同时可能附带微小姿态影响）  
- action 3: `right_orientation_engine` – 点燃右侧姿态发动机

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  `body_not_awake_or_settled` – 当机体停止运动并满足“settled”状态时触发。这通常是成功着陆的标志（稳定在目标垫上）。
- failure-like termination:  
  `crash_or_body_contact` – 机体与地面发生非支撑脚接触（硬碰撞）触发。  
  `horizontal_position_outside_viewport` – 水平位置超出视野边界触发。
- ambiguous termination: 无明确不分明的情况。
- truncation: 从源码看 `step()` 返回的 `truncated` 为 `False`，因此不考虑最大步数截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，无显式标志）  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（只允许空的 `{}` 且不应依赖任何字段）  
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用
- 备注：  
  成功/失败可通过终止时的**观测信号**间接推断：  
  - **成功推断链** (`derived_possible`)：  
    若单步后的 `terminated == True`，且 `next_obs` 满足：  
    `|x_position| < 极小阈值`，`|y_position| < 极小阈值`，  
    `|x_velocity| < 极小值`，`|y_velocity| < 极小值`，  
    `|body_angle| < 极小值`，  
    `left_support_contact == 1.0` 且 `right_support_contact == 1.0`，  
    则可判定为“成功着陆”。  
  - **越界推断链**：若 `|x_position|` 超出某合理上限（如 1.0 或 1.5）且 terminated，则判定为出界。  
  - **碰撞推断链**：若 terminated 但不满足成功条件，且 x_position 未明显越界，则判定为碰撞/失控。  
  这些推断基于观测空间中的连续信号，可在奖励函数中安全使用，但需标注 `derived_possible`。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- `obs`（当前观测）
- `action`（当前动作）
- `next_obs`（下一步观测）
- `info` 中确认允许的字段（本环境 info 为空，无可用字段）
- `training_progress` 仅在 prompt 明确允许时使用（本任务未要求，不建议使用）

禁止使用：
- `original_reward`（官方奖励已屏蔽）
- 任何未在信息源中声明的 `info` 字段
- 任何基于环境内部状态、游戏物理引擎私有变量等的信息

## 7. 可用于奖励函数的信号
- **位置**：  
  - `x_position`（相对目标），`y_position`（相对目标垫高度），用于计算到目标垫的欧氏距离/水平垂直误差。  
- **速度**：  
  - `x_velocity`, `y_velocity`，用于衡量飞行器运动状态、准备着陆的减速情况。  
- **姿态**：  
  - `body_angle`（俯仰/倾斜），用于确保安全姿态（接近 0）。  
  - `angular_velocity`，可辅助抑制快速旋转。  
- **接触**：  
  - `left_support_contact`，`right_support_contact`，直接反映双脚是否着垫，是判断成功 settling 的关键。  
- **动作/发动机**：  
  - 动作编号（0/1/2/3），可用于计算燃料消耗惩罚。  
- **衍生/推断信号** (`derived_possible`)：  
  - 到达目标垫的欧氏距离：`sqrt(x_position^2 + y_position^2)`（基于相对坐标）。  
  - 接近速度：利用距离差分 `distance(obs) - distance(next_obs)`（可选）。  
  - 成功着陆检测：由 `next_obs` 在 terminated 时满足位置、速度、角度、双脚触地条件推断。  
  - 越界检测：由 `|x_position|` 超出阈值推断。  
  - 碰撞失败：排除上述两类的终止推断。

## 8. 不确定或不可用的信号
- `original_reward`：明确禁止。  
- info 字段：info 恒为空字典，无法使用任何成功/失败标志或额外奖励权重。  
- 真实的物理碰撞标志：无法直接获取，只能通过观测信号间接推断。  
- 时间/步数：没有提供 episode 内时间戳，不可直接使用。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: "2D 着陆器/飞行器，具有主推进器和
