# Response Record

# 匿名环境理解卡片

## 1. 任务目标
控制一个2D着陆器从视野顶部中央区域出发，借助初始随机速度，以最短时间和最少燃料消耗完成向中央着陆垫的精确软着陆。要求最终稳定停靠在目标垫上：位置接近垫中心，速度趋近于零，姿态保持垂直，且两条支撑腿与地面发生安全接触。避免任何形式的撞击、侧翻、越界或持续不稳定振荡。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 核心目标是让着陆器到达并稳定在指定的固定目标垫上，属于导航-目标到达任务族。燃料最小化和快速到达是性能优化，非冲突性多目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float64 (默认，观测中 contact 是 0.0/1.0)
- obs[0]: `x_position` – 相对于目标垫的水平距离，reward_usable: true
- obs[1]: `y_position` – 相对于着陆垫高度（垫平面）的垂直距离，reward_usable: true
- obs[2]: `x_velocity` – 水平线速度，reward_usable: true
- obs[3]: `y_velocity` – 垂直线速度，reward_usable: true
- obs[4]: `body_angle` – 机体相对于垂直方向的偏转角度，reward_usable: true
- obs[5]: `angular_velocity` – 机体角速度，reward_usable: true
- obs[6]: `left_support_contact` – 左支撑腿接触标志 (1.0 表示接触)，reward_usable: true
- obs[7]: `right_support_contact` – 右支撑腿接触标志 (1.0 表示接触)，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine` – 无引擎推力，仅靠惯性飞行
- action 1: `left_orientation_engine` – 点燃左侧姿态调节引擎，产生顺时针力矩和少量侧推力
- action 2: `main_engine` – 点燃主发动机，产生向上推力，同时消耗燃料
- action 3: `right_orientation_engine` – 点燃右侧姿态调节引擎，产生逆时针力矩和少量侧推力

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled` – 着陆器进入静止/休眠状态通常意味着两条腿牢固着陆且已稳定，可视为成功着陆。
- failure-like termination:
  - `crash_or_body_contact` – 机体任何非支撑腿部分触地、猛烈撞击或侧翻，导致坠毁。
  - `horizontal_position_outside_viewport` – 水平漂移超出可接受边界（离开视口），代表任务失败。
- ambiguous termination: 无。
- truncation: 返回 `False`，无额外截断限制；但实际环境中可能存在最大步数限制，但 env 未透露，本卡片不采用。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {} （空字典）
- forbidden_or_uncertain_info_fields: `original_reward`, `official_reward`, `success`, `failure`, 任何未在 info 中声明的字段

注意：不能根据 `terminated` 的真假直接判断成功/失败，因为有两种失败和一种成功都会触发终止，但 `terminated` 本身不区分原因。必须从 `next_obs` 和 `done` 中推断，或者整合观测信号（如两条腿是否接触、速度大小等）来构建奖励。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- `obs` – 当前步观测数组 (8,)
- `action` – 当前执行的动作 (int)
- `next_obs` – 下一步观测数组 (8,)
- `info` – 仅限空字典，不得依赖任何字段

禁止使用：
- `original_reward` （官方奖励被遮蔽，严禁以任何方式引用或重构）
- `training_progress` 除非本提示明确声明允许（此处未声明）
- 任何未在 observation_space 描述中明确列出的 `obs` 切片
- 任何未在 info 约束中列出的字段

## 7. 可用于奖励函数的信号
- position: `x_position`, `y_position` （均为相对目标垫）
- velocity: `x_velocity`, `y_velocity`
- orientation: `body_angle`, `angular_velocity`
- contact: `left_support_contact`, `right_support_contact` （0.0/1.0 浮点）
- action/engine: 动作编号（0～3），可据此构造燃料消耗罚项或推力鼓励
- other: 可通过 `next_obs` 观察变化量（如速度变化、角度变化），但每一步间隔固定，近似微分可用

## 8. 不确定或不可用的信号
- 任何显式成功/失败标志 (info 中无) 。
- 燃料余量或耗量 （观察空间中未提供，只能通过动作使用情况间接推断）。
- 目标位置绝对坐标 （因为观测本身就是相对于目标垫的偏移量，但目标垫位置固定未给出，只能假设垫中心为 (0,0) 参考系）。
- 地形高度、风力扰动等环境隐藏变量。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: symmetric two-legged lander (lander)
  actuator_type: thrusters (main engine + 2 orientation engines)
  contact_structure: two support legs with binary contact sensors at the feet
primary_objectives:
  - Achieve soft landing on the target pad: x_pos ≈ 0, y_pos ≈ 0, low velocity, near-upright orientation, both legs in contact.
secondary_objectives:
  - Minimize engine usage (fuel) and time to landing.
main_failure_risks:
  - Crashing by high vertical speed or body part contact.
  - Drifting out of horizontal bounds.
  - Over-rotating and failing to stabilize.
  - Inefficient hover or oscillation leading to never settling.
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: `goal_approach_and_soft_landing`
  purpose: 鼓励着陆器靠近目标垫、减速、保持直立，并最终用双腿接触地面。
  why_required: 这是任务的核心成功条件，没有它智能体无法学习到达并稳定着陆。
  usable_signals:
    - 位置误差：`x_position`, `y_position`（越小越好）
    - 速度幅值：`x_velocity`, `y_velocity`（越小越好）
    - 姿态误差：`|body_angle|` 或 `body_angle^2`
    - 接触奖励：`left_support_contact` 与 `right_support_contact` 同时为 1 时给予奖励
  risks:
    - 单纯位置接近奖励可能导致高速撞击；必须配合速度惩罚。
    - 姿态奖励需要 careful 设计，以免过早将着陆器锁死在直立状态而妨碍必要的倾角调整。

### 10.2 条件职责 conditional_roles
- role_id: `fuel_efficiency`
  condition_to_use: 可在任何阶段加入，但权重需平衡，避免在降落关键阶段过度抑制主发动机使用。
  usable_signals:
    - 当前动作是否为 `main_engine` (动作2) 或姿态引擎 (动作1/3) 触发惩罚；或者给予恒定的位置无关奖励，并对任何非零推力动作施加小惩罚。
  risks:
    - 过度惩罚燃料使用可能导致智能体拒绝点火，永远无法着陆; 建议用小的负奖励或在成功着陆后给予更大的一次性奖励来抵消。
    - 若结合位置速度惩罚，燃料惩罚需适度。

- role_id: `time_pressure` (快速到达)
  condition_to_use: 如果希望在有限时间内完成任务，可加入每一步微小的负奖励，但会加剧燃料惩罚压力。通常不需要显式实现，因为步数限制自然产生压力。
  usable_signals: 每步恒定负值（如-0.05），但需谨慎。
  risks: 可能导致匆忙撞击，必须伴随强力安全约束。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: `explicit_termination_reward`
  reason: 环境不提供 info 中的成功/失败标志，且 `terminated` 无法区分成功与失败，若直接根据 `terminated` 给予大奖励极其危险（可能把失败也当作成功奖赏）。任何依赖终止原因分发的奖励都不可用。
  forbidden_or_missing_signals: 缺失 `success`/`failure` 字段。

- role_id: `shaping_based_on_original_reward`
  reason: original_reward 被禁止使用，不能作为参考或差值奖励。
  forbidden_or_missing_signals: original_reward 被遮蔽。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_approach_and_soft_landing | x_position, y_position, x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact | 无 | `bounded_signal( (x_pos^2 + y_pos^2) )`, `quadratic_penalty(velocity)`, `cosine_proximity(angle)`, `logical_and(left_contact, right_contact)` | 接触信号可作为步骤内奖励，但应仅在两条腿都接触并且速度都接近0时给予大奖励，防止提前奖励。 |
| fuel_efficiency | action (0-4) | 燃料消耗量 | `discrete_action_penalty([0, -0.03, -0.3, -0.03])` 或类似 | 主发动机的惩罚应显著高于姿态引擎，因为它的脉冲更大。注意平衡。 |
| time_pressure | （每步常数） | 无 | `stepwise_constant_penalty(-0.005)` | 可以省略，由环境截断时间自然施压。 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 高速撞击坠毁 | 训练曲线奖励不升，episode 长度短，最终 `y_velocity` 很大负值且终止 | 增加速度惩罚权重，特别是在低高度时；引入高度相关速度上限惩罚。 |
| 悬停不降或无限等待 | episode 长度达到环境最大值但不终止成功，位置接近但仍有速度，双腿未同时接触 | 增加高度奖励（对接近地面给予小奖励）或引入 soft landing bonus，仅在双腿接触且速度极小时给予较大奖励。 |
| 越界漂出视口 | 水平位置超出边界导致终止，x_position 绝对值很大 | 加重水平偏差惩罚，或者在奖励中施加平方惩罚，让智能体更用力修正。 |
| 过早起火导致燃料耗尽 | 每步燃料惩罚已存在但智能体仍大量使用主发，最终熄灭后坠落 | 检查燃料惩罚是否过小，或引入燃料总预算感知（但观测无燃料信息），可模拟采用动作熵惩罚或增加主发惩罚。 |
| 振荡不稳，永远不 sleep | 身体角速度持续非零，腿交替接触但不稳定 | 增加角速度惩罚，或者接触后给予小量额外固定奖励以鼓励快速镇定。 |
| 只有一条腿接触就停 | 左或右腿接触为1而另一个为0，终止由于body_not_awake? 需确认是否只有完整着陆才触发 sleep | 确保接触奖励要求双腿都接触，避免单腿接触产生奖励。 |
