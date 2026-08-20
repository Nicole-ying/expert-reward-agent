# Response Record

# 匿名环境理解卡片

## 1. 任务目标
该任务是一个 2D 车辆/着陆器的轨迹优化问题。智能体从视口上方中央附近以随机初始推力出发，必须驾驶自身到达画面中央的目标平台，并在其上稳定停靠（settle）。核心目标是**快速到达并安全停在目标点**，次要目标是**尽量少用引擎推力**以节省燃料。智能体需要学会平滑减速、保持直立姿态，并用两个支撑脚平稳接触平台。任何碰撞、飞出视口或不稳定的翻滚都应避免。不要将此任务与纯加速度最小化或多目标探索混淆，其核心明确是以“精准停靠”为成功的导航到达任务。

## 2. 任务类型选择
- **selected_route_id**: `navigation_goal_reaching`
- **confidence**: high
- **reason**: 任务描述的核心是“到达并停靠在中心目标平台”，这是典型的导航到达任务。尽管有燃料节俭和姿态稳定等附属目标，但成功与否由是否到达并稳定停靠决定，没有其他与到达目标同等权重的冲突目标，因此归入导航到达类。

## 3. 观察空间 observation_space
- **type**: Box (连续)
- **shape**: (8,)
- **dtype**: float32 (推断)
- **各维度含义**:

| 索引 | 名称 | 含义 | 奖励中可用 |
|------|------|------|------------|
| 0 | x_position | 相对目标平台的水平坐标 | true |
| 1 | y_position | 相对目标平台高度的垂直坐标 | true |
| 2 | x_velocity | 水平线速度 | true |
| 3 | y_velocity | 垂直线速度 | true |
| 4 | body_angle | 机体方向角 | true |
| 5 | angular_velocity | 角速度 | true |
| 6 | left_support_contact | 左支撑脚接触标志 (0/1) | true |
| 7 | right_support_contact | 右支撑脚接触标志 (0/1) | true |

## 4. 动作空间 action_space
- **type**: Discrete
- **n**: 4
- **动作含义**:

| 动作值 | 名称 | 含义 |
|--------|------|------|
| 0 | no_engine | 无推进，所有引擎关闭 |
| 1 | left_orientation_engine | 点燃左侧姿态控制引擎（产生顺时针力矩） |
| 2 | main_engine | 点燃主引擎（产生向上的推力） |
| 3 | right_orientation_engine | 点燃右侧姿态控制引擎（产生逆时针力矩） |

- 这些动作直接影响水平/垂直速度和角速度，通过多次交互控制轨迹。燃料消耗与是否点火相关，可通过动作值间接测量。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` 中的 **settled** 部分 —— 当机体在目标平台上稳定停靠且不再运动时触发，暗示成功。
- **failure-like termination**: `crash_or_body_contact` （非平台接触的碰撞，如撞击地面/边界）以及 `horizontal_position_outside_viewport` （水平飞出视口）。
- **ambiguous termination**: `body_not_awake_or_settled` 同样包括“机体失去活动能力（未唤醒）”的情况，可能是体力耗尽、卡住等不成功状态，需结合其他指标判断。
- **truncation**: 无显式最大步数截断，但环境可能隐式包含。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 无（step 返回的 info 为空字典）
- **forbidden_or_uncertain_info_fields**: 任何假设的 `info["success"]` 或 `info["failure"]` 均不存在，禁止在奖励函数中使用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- `obs`  (当前观察的 8 维向量)
- `action` (当前选择的离散动作 0~3)
- `next_obs` (下一观察，可用于差分或接触检测)
- `info` 中只能使用该环境允许的字段，但本环境中 `info` 恒为空 `{}`，**不得依赖任何 `info` 字段**
- `training_progress` 仅当 prompt 明确允许时可用，当前未声明，默认为不可用

禁止使用：
- `original_reward`：已完全遮蔽，禁止访问
- 任何未声明的 `info` 字段
- 任何未在观察空间中声明的 `obs` 切片（如越界索引）
- 任何对环境内部状态（如燃料量、真实坐标）的硬编码假设

## 7. 可用于奖励函数的信号
- **位置**：`obs[0]`(x_position), `obs[1]`(y_position) → 可计算到目标 (0,0) 的距离、逼近项
- **速度**：`obs[2]`(x_velocity), `obs[3]`(y_velocity) → 可用于速度惩罚、减速奖励
- **姿态**：`obs[4]`(body_angle), `obs[5]`(angular_velocity) → 可用来奖励直立、角速度抑制
- **接触**：`obs[6]`(left_support_contact), `obs[7]`(right_support_contact) → 可探测着陆触地、双脚稳定接触
- **动作/引擎**：`action` 本身可用来惩罚非零推力（动作0为无推力），间接衡量燃料消耗
- **其他**：`next_obs` 可用于差分（如速度变化、接触状态变化）以增强奖励，但需注意其并非独立信号，而是 `obs` 的下一帧。

## 8. 不确定或不可用的信号
- **显式任务成功标志**：info 中无 `success`
- **显式失败原因**：info 中无 `failure_reason`
- **精确燃料量**：观察中无燃料计；只能通过 `action` 是否为 0 来粗略衡量推力是否被使用
- **目标位置绝对坐标**：观察已相对化，目标总是 (0,0)，无需额外信号
- **平台位置或关键点**：只通过接触箱信号间接反映，无法得到精确的平台边界

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D single rigid body (lander / vehicle)
  actuator_type:
    - main_engine (vertical thrust)
    - left_orientation_engine (clockwise torque)
    - right_orientation_engine (counterclockwise torque)
    - no_engine (passive)
  contact_structure: two support feet (left, right) with binary contact flags
primary_objectives:
  - Reach the target pad (position (0,0) )
  - Settle stably with both feet in contact and near-zero velocity
secondary_objectives:
  - Minimize fuel usage (avoid unnecessary engine firings)
  - Maintain upright orientation (body_angle near 0)
  - Achieve fast arrival (implicit time pressure from sparse reward)
main_failure_risks:
  - Crashing into ground or walls
  - Flying outside horizontal bounds
  - Rotating uncontrollably (tumbling)
  - Hovering indefinitely without landing
  - Wasting thrust and never approaching
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: `goal_proximity`**
  **purpose**: 鼓励智能体向目标 (0,0) 靠近。
  **why_required**: 核心导航需求，没有接近奖励智能体无法学习方向。
  **usable_signals**: `obs[0]`, `obs[1]` → 欧氏/曼哈顿距离；`next_obs[0:2]` 差分。
  **risks**: 距离奖励设计不当可能导致高速冲过目标，需配合减速项。

- **role_id: `safe_landing`**
  **purpose**: 奖励在低速、双支撑接触且姿态良好时完成停靠。
  **why_required**: 真实目标为“settle”，仅位置接近不够，必须平稳接触并停止。
  **usable_signals**: `obs[2]`, `obs[3]` (速度), `obs[4]` (倾角), `obs[6]`, `obs[7]` (接触标志)；可用 `next_obs` 验证接触状态变化。
  **risks**: 该奖励为稀疏事件，需与密集奖励配合，否则不易收敛；阈值设置需谨慎。

- **role_id: `attitude_stability`**
  **purpose**: 惩罚大倾角和角速度，维持直立姿态。
  **why_required**: 过大的倾角可能导致碰撞或无法正确接触支撑脚；直立姿态是成功停靠的重要前提。
  **usable_signals**: `obs[4]`, `obs[5]`.
  **risks**: 与目标接近冲突时（如需要倾斜来减速）可能过度干扰，应使用较小的系数或仅在接近地面时激活。

- **role_id: `fuel_efficiency`**
  **purpose**: 惩罚任何推力动作（动作1,2,3），鼓励使用无推力滑行。
  **why_required**: 任务明确要求“尽可能少用引擎推力”，且离散动作允许直接衡量。
  **usable_signals**: `action` (是否为 0).
  **risks**: 单纯的推力惩罚会让智能体不敢使用引擎，必须与其他奖励平衡，尤其在需要主引擎减速时。

### 10.2 条件职责 conditional_roles
- **role_id: `soft_landing_surge_penalty`**
  **condition_to_use**: 当智能体历史表现出高速触地倾向时启用；或根据 `training_progress` 在后期引入。
  **usable_signals**: `obs[2]`, `obs[3]` (速度), `obs[1]` (高度) → 靠近地面时若垂直速度过大多施加惩罚。
  **risks**: 过早引入会阻碍探索，可通过 `training_progress` 控制强度。

- **role_id: `approach_velocity_adaptation`**
  **condition_to_use**: 当训练出现“悬停不决”或“高速冲撞”两种极端时。
  **usable_signals**: `obs[0]`, `obs[1]` (距离), `obs[2]`, `obs[3]` (速度) → 期望速度随距离减小而降低。
  **risks**: 强加约束可能造成非最小相位动态，需要平滑过渡。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: `time_pressure`**
  **reason**: 任务描述提及“尽可能快”，但时间信息在标准 MDP 中不可直接观测（无步数惩罚），且若通过外部 `training_progress` 调权可能破坏自然探索。当前环境未提供剩余时间信号，不宜作为显式奖励项，改为依赖稀疏成功奖励的 decay 效应。
  **forbidden_or_missing_signals**: 缺少实时时间或剩余步数。

- **role_id: `contact_only_reward`**
  **reason**: 仅依赖 `obs[6]`, `obs[7]` 而不考虑位置和速度容易产生“悬停并用脚轻触”的投机行为，不能保证真正停靠。
  **forbidden_or_missing_signals**: 虽然接触信号可用，但缺少平台受力或稳定指示器，因此不宜单独作为主要成功判据。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | obs[0], obs[1], next_obs[0:2] | - | negative_distance, quadratic_distance, bounded_difference | 常用负欧氏距离或指数衰减，注意避免奖励密度过高 |
| safe_landing
