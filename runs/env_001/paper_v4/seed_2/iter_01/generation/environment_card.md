# 匿名环境理解卡片

## 1. 任务目标
该环境是一个 2D 飞行器着陆任务。飞行器从目标着陆平台上方区域出发，受到随机初始扰动力。核心目标是引导飞行器尽快接近并稳定地停在中央着陆平台上，并最大限度减少引擎使用。成功的着陆应满足：位置靠近平台中心、相对垂直速度很小、身体姿态平稳、且两个支撑部件都与平台接触。附属目标是降低燃料消耗（即减少引擎推力动作的使用）并缩短耗时。任务**不**是简单的平衡或存活，而是以到达指定位置为根本驱动。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**
confidence: **high**
reason: 主目标明确为“到达并稳定在终点平台”，附属目标（节能、快速）不构成同等权重的冲突优化项，核心属于目标到达任务族。

动力学子类型进一步判断为 **goal_approach_and_soft_contact**，它精准刻画了“接近目标→减速→保持姿态→安全软接触”的着陆动力学过程。

## 3. 观察空间 observation_space
- type: `Box`
- shape: `[8]`
- dtype: 默认 `float32`（其中 contact 标志为 0.0 / 1.0）
- 每一维含义及奖励可用性：

| 维度索引 | 名称 | 含义 | reward_usable |
|-----------|------|------|---------------|
| 0 | `x_position` | 飞行器相对于着陆平台的水平坐标 | true |
| 1 | `y_position` | 飞行器相对于平台高度的垂直坐标 | true |
| 2 | `x_velocity` | 水平线速度 | true |
| 3 | `y_velocity` | 垂直线速度 | true |
| 4 | `body_angle` | 机体倾斜角 | true |
| 5 | `angular_velocity` | 角速度 | true |
| 6 | `left_support_contact` | 左支撑腿是否接触地面（0/1） | true |
| 7 | `right_support_contact` | 右支撑腿是否接触地面（0/1） | true |

所有维度均可安全用于奖励函数计算，不存在需禁用的字段。

## 4. 动作空间 action_space
- type: `Discrete`
- n: 4
- 动作含义：

| 动作索引 | 名称 | 含义 |
|----------|------|------|
| 0 | `no_engine` | 无任何推力输出 |
| 1 | `left_orientation_engine` | 开启左侧转向引擎（产生逆时针旋转力矩及少量推力） |
| 2 | `main_engine` | 开启主引擎（产生强大的向下推力，抵消重力） |
| 3 | `right_orientation_engine` | 开启右侧转向引擎（产生顺时针旋转力矩及少量推力） |

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` —— 当飞行器身体不再活跃（进入休眠）且已基本静止时触发。通常由稳定着陆引发。
- **failure-like termination**: `crash_or_body_contact` —— 飞行器与地面/平台发生不当碰撞（如侧面触地、强烈冲击），视为坠毁；`horizontal_position_outside_viewport` —— 水平位置超出视觉窗口，表示偏离过大。
- **ambiguous termination**: 无。所有终止条件均可明确区分为成功或失败。
- **truncation**: 环境描述未提及最大步数截断，但实际部署时可能存在；truncation 不能被直接当作成功/失败信号。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false** （`info` 字典为空，无直接标志位）
- explicit_failure_flag_available: **false** （同上）
- allowed_info_fields: **无** 额外允许字段（`info = {}`）
- forbidden_or_uncertain_info_fields: 所有未在 step source 中出现的 `info` 键均不可用，包括但不限于 `success`、`failure`、`termination_reason`。

因此，奖励函数**必须**仅依赖 `obs`、`action`、`next_obs` 以及终止状态来隐式推断任务成果，不能依赖外部成功/失败标记。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
**允许使用：**
- `obs`：当前观测（8维数组）
- `action`：当前动作（int 或 one-hot，具体由环境决定；此处为 int）
- `next_obs`：下一时刻观测
- `info`：仅允许使用明确声明的字段（当前为空，故不可用）
- `training_progress`：当且仅当 prompt 明确允许时才能使用

**严格禁止使用：**
- `original_reward` / `official_reward`（已被屏蔽）
- 任何未在观察空间说明中出现的 `obs`/`next_obs` 切片
- 任何未在 `info` 中明确声明的字段

## 7. 可用于奖励函数的信号
- **位置信号**: `x_position`, `y_position` （相对目标，可直接计算距离）
- **速度信号**: `x_velocity`, `y_velocity` （衡量平缓着陆的关键）
- **姿态信号**: `body_angle` （水平姿态维稳）
- **角速度**: `angular_velocity`
- **接触信号**: `left_support_contact`, `right_support_contact` （软着陆的最后确认）
- **动作/引擎信号**: 动作索引可识别是否使用主引擎或转向引擎，用于燃料惩罚。
- **其他潜在信号**: 可从 `obs` 和 `next_obs` 差分得到加速度等，但需谨慎引入。

## 8. 不确定或不可用的信号
- 明确的目标位置坐标：由于观测是**相对于目标平台**的坐标，目标为 (0,0) 可自然推导。
- 明确的成功/失败标记：不存在。
- 剩余步数、燃料量等额外状态：不可用。
- 风力扰动等外部信息：被屏蔽，不可观测。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body with two support legs
  actuator_type: one main downward thruster + two lateral orientation thrusters
  contact_structure: two foot contacts on landing platform
primary_objectives:
  - 将飞行器运动到目标平台中心（x≈0, y≈0）
  - 在接触时保持极低的线速度和角速度（软着陆）
  - 着陆后两腿均触地且姿态基本水平
secondary_objectives:
  - 最小化燃料消耗（减少主引擎及转向引擎使用）
  - 尽可能快速完成着陆（隐式奖励通过效率导向）
main_failure_risks:
  - 坠毁或硬着陆（垂直速度过大导致 crash_or_body_contact）
  - 水平漂移出视口
  - 长时间盘旋消耗燃料无法完成着陆
  - 仅单腿着陆导致姿态不稳
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- **role_id: `position_proximity`**
  purpose: 鼓励飞行器向目标中心靠近。
  why_required: 到达目标是根本任务，无此职责 agent 不会学习定向移动。
  usable_signals: [`x_position`, `y_position`]
  risks: 单纯距离奖励可能造成高速撞击；需配合着陆职责。

- **role_id: `soft_landing_velocity`**
  purpose: 惩罚接近地面时的高线速度，尤其是垂直接触速度，促使轻缓落地。
  why_required: 防止以高速 crash 终止，是安全着陆的直接约束。
  usable_signals: [`y_position`（用于激活阈值）, `x_velocity`, `y_velocity`]
  risks: 若在高中空即施加强惩罚，会阻碍下降；需要结合高度门控。

- **role_id: `stable_orientation`**
  purpose: 抑制机体倾斜，要求着陆时姿态接近水平。
  why_required: 倾斜着陆极易单脚先触地导致 crash 或姿势失衡。
  usable_signals: [`body_angle`]
  risks: 角度惩罚过强可能使 agent 害怕使用转向引擎，影响水平位置控制。

- **role_id: `contact_completion`**
  purpose: 迫使者陆后两腿均与平台接触（仅接触地面但单腿不算完美）。
  why_required: 任务明确需要“安全接触”，双腿触地才是稳定终止条件的主要前兆。
  usable_signals: [`left_support_contact`, `right_support_contact`]（可与终止时或着陆后 next_obs 结合使用）
  risks: 提前给予接触奖赏可能导致虚假接触行为，必须与位置、速度、姿态强耦合。

- **role_id: `fuel_efficiency`**
  purpose: 惩罚使用引擎推力，降低燃料消耗。
  why_required: 明确要求“as little engine thrust as possible”，是任务描述的一部分。
  usable_signals: [`action`]（动作索引 1,2,3 对应不同引擎）
  risks: 惩罚过大会使 agent 选择无推理漂移，放弃着陆；惩罚太小则无法体现节能要求。

### 10.2 条件职责 conditional_roles
- 无额外条件职责。任务目标已由上述主职责完全覆盖。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: `speed_penalty_early`**
  reason: 在所有高度下均惩罚高速度可能严重阻碍下降进程，应改为仅在接近地面（y 较小）时激活。
  forbidden_or_missing_signals: 无（就是使用方式问题）

- **role_id: `force_fast_landing`（时间惩罚）**
  reason: 任务确有“as fast as possible”要求，但**缺失剩余时间或步数信号**，无法直接构建可靠的时间效率奖励，强行使用步数倒扣容易导致过早坠落或投机行为。建议避免。
  forbidden_or_missing_signals: 步数/剩余时间不可用

## 11. role_to_signal_mapping

| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| `position_proximity` | `x_position`, `y_position` | 无 | `dense_state_signal` (negative distance / exponential decay) | 可直接用欧几里得距离的相反数或高斯式强度 |
| `soft_landing_velocity` | `y_position`, `y_velocity`, `x_velocity` | 无 | `bounded_signal` (高度门控 → quadratic_penalty) | 仅当 y 低于阈值时才激活速度惩罚项 |
| `stable_orientation` | `body_angle` | 无 | `quadratic_penalty` 或 `abs_penalty` | 需避免与转向引擎惩罚冲突 |
| `contact_completion` | `left_support_contact`, `right_support_contact` | 显式的“成功着陆”标志 | `binary_bonus` / `joint_contact_condition` | 宜在终止步或最后几步给予，且必须配合位置/速度检验 |
| `fuel_efficiency` | `action` (0,1,2,3) | 无 | `discrete_action_cost` | 可为主引擎动作分配更高惩罚，转向引擎较低 |

## 12. 初始训练后应观察的 failure modes

| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 着陆速度过高导致 crash | 终止前几步 `y_velocity` 绝对值很大；终止于 crash | 增大 `soft_landing_velocity` 角色权重，或降低高度触发阈值 |
| 悬停漂浮不下降 | `y_position` 长时间保持较高，动作长期选择 0 或仅转向 | 调整 `position_proximity` 奖励陡度，强调 y 接近 0 的收益；或轻度放松燃料惩罚 |
| 单脚着陆不稳定 | 终止时仅一条腿 contact=1，另一条=0 | 强化 `contact_completion` 对双腿同时接触的要求；结合 `body_angle` 做耦合约束 |
| 水平漂移出视口 | 终止于 `horizontal_position_outside_viewport` | 增大 `position_proximity` 中 x 方向分量的惩罚，或加入 x 速度在远距离时的阻尼项 |
| 过度使用转向引擎导致旋转失控 | `angular_velocity` 持续很大，角度来回振荡 | 引入对 `angular_velocity` 的额外惩罚（与 `stable_orientation` 联合） |
| 为节省燃料完全不使用主引擎 | 任务长期无下降，终止于超时或 crash | 动态调整 `fuel_efficiency` 权重，或仅在接近地面时收紧引擎惩罚 |

此环境