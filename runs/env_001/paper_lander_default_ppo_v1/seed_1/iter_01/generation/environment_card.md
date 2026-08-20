# 匿名环境理解卡片

## 1. 任务目标
本环境是一个 2D 飞行器着陆任务。飞行器从视口上部中心附近出发，受随机初始力影响。目标是**尽快平稳地降落在中央的着陆垫上**，同时**尽可能少地使用引擎推力**。智能体需要学习：向目标靠近、减速、保持机身姿态稳定、实现安全触垫。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断）
- obs[0]: x位置（相对着陆垫的水平坐标）
- obs[1]: y位置（相对垫高度的垂直坐标）
- obs[2]: x方向线速度
- obs[3]: y方向线速度
- obs[4]: 机身角度（俯仰角）
- obs[5]: 角速度
- obs[6]: 左支撑接触标志（1.0 接触，0.0 未接触）
- obs[7]: 右支撑接触标志（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: 无引擎（不施加推力）
- action 1: 左侧姿控引擎点火
- action 2: 主引擎点火（向下推力）
- action 3: 右侧姿控引擎点火

## 5. step 与终止条件分析
### 5.1 终止模式
Termination 由以下三种条件触发（任一为真即结束）：
1. `crash_or_body_contact`：坠毁或身体与地面/其他物体发生接触（大概率是失败）。
2. `horizontal_position_outside_viewport`：水平位置超出边界（失败）。
3. `body_not_awake_or_settled`：机体进入休眠/稳定状态。当飞行器完全静止、不再受物理唤醒时触发。若此时已处于着陆垫上且支撑腿接触，则通常表示成功着陆；若在坠毁后静止，则被前面的 crash 条件先行触发。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（step 返回的 info 为空，无显式标记）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空字典 `{}`，不可使用任何字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用。由于源码显式返回 `{}`，不能假设存在 `success`、`termination_reason` 等键。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- `obs`：当前观察向量（完整 8 维）
- `action`：当前执行的动作（0,1,2,3）
- `next_obs`：下一时刻观察向量（完整 8 维）
- `info`：空字典（实际无可用信息）
- `training_progress`：禁止使用（除非 prompt 明确允许）

禁止使用：
- `original_reward`（官方奖励被遮蔽，不可重建或依赖）
- `official_reward` 或其它变体
- 未声明的 `info` 字段（例如 `success`、`failure`、`termination_reason` 均不存在）
- 任何未在上述“允许使用”中列出的外部信号

## 7. 可用于奖励函数的信号
- **位置**：相对着陆垫的 x, y 坐标（obs[0], obs[1]），可直接用于评估靠近/悬停/触垫。
- **速度**：x, y 线速度（obs[2], obs[3]），可用于惩罚高速撞击或奖励稳定减速。
- **姿态**：机身角度（obs[4]），角速度（obs[5]），用于鼓励保持直立姿态。
- **接触**：左右支撑接触标志（obs[6], obs[7]），可用于检测是否着陆成功、是否平稳（双脚着垫）。
- **动作/引擎使用**：动作编号本身，可以用于惩罚推力使用（动作 1,2,3 为引擎点火）以鼓励省油。

## 8. 不确定或不可用的信号
- **明确成功标志**：info 为空，无法获得。
- **燃料消耗量**：环境未提供燃料量字段，只能从动作序列中自行累加推断。
- **着陆垫位置**：未单独提供绝对坐标，仅通过相对坐标隐含在观察中。
- **是否在着陆垫正上方**：必须结合 y 位置和接触标志间接推断。
- **终止原因**：没有 termination_reason，无法直接区分是成功着陆还是坠毁。只能从最后时刻的观察（如接触标志、位置、速度）逻辑推断。