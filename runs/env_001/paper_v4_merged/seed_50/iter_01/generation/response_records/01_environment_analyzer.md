# Response Record

# 匿名环境理解卡片

## 1. 任务目标
这是一个二维飞行器/着陆器轨迹优化任务。agent 从视窗上方中央附近以随机初始力开始，需要尽快、省油地**到达视窗中央的目标着陆平台，并以安全姿态稳定接触**（即实现软着陆）。  
核心是导航到目标并实现 safe and stable contact，附属优化是节省发动机推力（能量效率）和缩短耗时，但不改变核心目标。  
**不可混淆**：任务不是持续前行（没有前进方向），也不是纯粹的存活（没有存活计时器），而是**定点到达 + 停稳**。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**  
confidence: high  
reason: 任务的核心问题是“到达并停稳在目标点”，到达目标位置是主目标，节省燃料和快速是附属优化。不属于 locomotion（无持续前进轴）、不属于 survival（目标不是一直活着）、不属于 sparse exploration（有明显目标距离信号），也不存在多个权重相等且冲突的核心目标（如既要快速又要非常省油但快和省油都是可量化的副目标，到达目标是严格必要条件），因此不划为 multi_objective_task。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32（推测）  

维度说明（索引从 0 开始，均为可用信号，reward_usable 均为 true）：

- **obs[0]**: `x_position` — 飞行器质心相对目标着陆平台的水平坐标（单位未知，相对值）。reward_usable: true  
- **obs[1]**: `y_position` — 飞行器质心相对平台高度的垂直坐标（下正？待确认方向；通常上正，但可通过初始位置和下降过程推断方向）。rewars_usable: true  
- **obs[2]**: `x_velocity` — 水平线速度。reward_usable: true  
- **obs[3]**: `y_velocity` — 垂直线速度。reward_usable: true  
- **obs[4]**: `body_angle` — 机体方向角（弧度）。reward_usable: true  
- **obs[5]**: `angular_velocity` — 角速度。reward_usable: true  
- **obs[6]**: `left_support_contact` — 左支撑腿是否接触平台（布尔化 float: 1.0/0.0）。reward_usable: true  
- **obs[7]**: `right_support_contact` — 右支撑腿是否接触平台。reward_usable: true

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  

动作含义：
- **action 0**: `no_engine` — 不启动任何发动机（滑行）。
- **action 1**: `left_orientation_engine` — 点燃左定向发动机，产生侧向/旋转力矩，可改变机体角度并小幅移动。
- **action 2**: `main_engine` — 点燃主发动机，产生主要推力（推测在机体坐标系向上或向下，结合角度影响水平和垂直速度）。
- **action 3**: `right_orientation_engine` — 点燃右定向发动机，与左对称，改变旋转和侧向移动。

## 5. step 与终止条件分析
### 5.1 终止模式
环境给出三个终止条件，经抽象后为：
- `crash_or_body_contact` — 飞行器坠毁或身体其他部分（非支撑腿）接触地面/平台，属于 likely failure。
- `horizontal_position_outside_viewport` — 水平坐标超出视窗范围，显然为 failure。
- `body_not_awake_or_settled` — 飞行器“休眠”或已经稳定停靠，这**很可能对应成功软着陆**（双腿接触且速度、角度足够小后触发）。由于任务目标是到达并 settle，该条件可作为 success-like termination。

当前 info 字典为空，无任何 explicit success/failure flag。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**
- explicit_failure_flag_available: **false**
- allowed_info_fields: 无（info 为 {}）
- forbidden_or_uncertain_info_fields: 所有 info 字段（不可用）

**成功推断路径**（derived_possible）：  
当 episode 终止（terminated=True）且最后一次观测满足：  
 `left_support_contact == 1.0 && right_support_contact == 1.0`  
 并将 `abs(body_angle)`、`|x_velocity|`、`|y_velocity|` 控制在很小阈值内，且 `x_position` 和 `y_position` 接近零，则可认为发生了成功软着陆。  
**失败推断路径**：  
若终止时 `abs(x_position)` 很大（出界），或存在坠毁迹象（极端 body_angle 突变、两腿未接触），可判断为失败。由于缺少身体接触传感器，无法直接获得碰撞信号，角度过陡、速度冲击可作为间接证据。

## 6. reward 函数接口契约
函数签名（由调用方约定）：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs` (8维np.ndarray)  
- `action` (int)  
- `next_obs` (8维np.ndarray)  
- `info` **中明确允许的字段** → 当前无允许字段（info为空），故**禁止使用任何 info 字段**  
- `training_progress` **仅在任务描述或 prompt 明确允许时可用** → 当前未明确允许，故**禁止使用**  

禁止使用：
- `original_reward`  
- 任何官方奖励内部变量  
- 任何未在上述允许清单中出现的数据（包括未声明的 obs 切片、未经允许的环境内部状态）

## 7. 可用于奖励函数的信号
由于 info 不可用，reward 只能依赖 `obs`、`action` 和 `next_obs`。

- **位置**：`obs[0], obs[1]` 和 `next_obs[0], next_obs[1]`  
- **速度**：`obs[2], obs[3]` 和 `next_obs[2], next_obs[3]`  
- **姿态**：`obs[4]` (body_angle) 和 `next_obs[4]`  
- **角速度**：`obs[5]` 和 `next_obs[5]`  
- **接触**：`obs[6], obs[7]` 和 `next_obs[6], next_obs[7]` (双腿接触标志)  
- **动作**：`action` 值（离散 0-3），可用于动作效率惩罚/奖励

**可从观测间接推断的衍生信号**（derived_possible）：  
- 成功率线索：两腿接触 + 小速度 + 小倾角 + 接近零位置 → 可推断成功着陆  
- 坠毁线索：倾角突然超过安全阈值（如 abs(angle)>某一临界值）或速度骤变 → 可推断碰撞

## 8. 不确定或不可用的信号
- 任何显式的 `success`、`failure`、`termination_reason` 标志（info 为空）  
- 身体其他部分接触传感器（只有支撑腿接触）  
- 燃料消耗量/剩余燃料（未提供）  
- 时间步计数或已消耗时间（未在观测或 info 中给出）  
- 平台的实际坐标（相对位置已给出，但绝对坐标可能未知）  
- 发动机推力大小（动作是离散的，推力效果隐藏在动力学中，无法直接获知）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid-body lander
  actuator_type: two side orientation thrusters + one main thruster (discrete on/off)
  contact_structure: two landing legs with contact flags (left/right)
primary_objectives:
  - 到达目标位置：将 (x_position, y_position) 基本降至 (0, 0) 附近
  - 软着陆：接触时速度近乎零 (x_velocity, y_velocity 接近 0)
  - 安全姿态：着陆时 body_angle 接近 0（竖直向上或与平台垂直）
  - 双支撑稳定：着陆时 left_support_contact==1 且 right_support_contact==1
secondary_objectives:
  - 快速完成（隐含时间压力，但无时间指标可用，只能通过奖励塑造中的速度导向间接影响）
  - 节能：尽量减少推力使用（特别是主发动机），动作越少、越零越好
main_failure_risks:
  - 高速撞击平台或地面（速度过大着陆）
  - 机体倾翻（angle 过大导致侧翻或头部触地）
  - 水平飞出视窗（x 超出范围）
  - 悬停过度
