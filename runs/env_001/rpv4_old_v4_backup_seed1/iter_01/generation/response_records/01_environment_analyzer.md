# Response Record

# 匿名环境理解卡片

## 1. 任务目标
该任务要求控制一个具有两条支撑腿的 2D 刚体飞行器，从视口顶部中央附近开始（带有随机初始力），**快速、低燃料地到达并稳定停靠在中央目标垫上**。核心是到达指定目标位置并实现安全软着陆，附属目标是最小化发动机使用次数（燃料）与总时间。代理人必须学会平滑接近目标、减速、保持竖直姿态并产生双腿的稳定接触。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**  
confidence: **high**  

reason: 任务的唯一核心是到达明确的空间目标（中央目标垫），其他如速度、姿态、燃料等都属于保障性约束或效率优化，不存在权重相当的冲突目标。因此，粗粒度任务族属于导航目标到达类。进一步，由于要求“减速、保持稳定方向、安全接触”，其动力学表现为接近目标后的柔和减速与接触，故动力学子类型为 **goal_approach_and_soft_contact**。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float64
- obs[0]: `x_position` — 飞行器相对于目标垫中心的水平坐标。可用于奖励对靶心的逼近。reward_usable: true
- obs[1]: `y_position` — 飞行器相对于目标垫表面的垂直坐标。可用于奖励下降与接近。reward_usable: true
- obs[2]: `x_velocity` — 水平线速度。可用于惩罚大水平漂移。reward_usable: true
- obs[3]: `y_velocity` — 垂直线速度。控制软着陆冲击，极其重要。reward_usable: true
- obs[4]: `body_angle` — 刚体倾角。竖直向上时接近 0。奖励维持竖直。reward_usable: true
- obs[5]: `angular_velocity` — 角速度。鼓励姿态稳定。reward_usable: true
- obs[6]: `left_support_contact` — 左腿接触标志（0/1）。表示左腿已与地面接触。reward_usable: true
- obs[7]: `right_support_contact` — 右腿接触标志（0/1）。表示右腿已与地面接触。reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine` — 不开启任何引擎，仅受重力/惯性影响。
- action 1: `left_orientation_engine` — 启动左侧姿态引擎，主要产生逆时针力矩（可能附带微小侧向力），用于校正右倾。
- action 2: `main_engine` — 启动主引擎，产生向上的推力，是主要的减速和高度控制手段。
- action 3: `right_orientation_engine` — 启动右侧姿态引擎，主要产生顺时针力矩，校正左倾。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 当 agent 以低速度、接近竖直的姿态通过双腿接触目标垫，且不再运动（`body_not_awake_or_settled`）时触发。这通常意味着安全着陆成功。
- **failure-like termination**:
  - `crash_or_body_contact`：飞行器任意部位（特别是非腿部）与地面或其它物体发生硬碰撞，导致坠毁。
  - `horizontal_position_outside_viewport`：飞行器水平漂出窗口，视为失控。
- **ambiguous termination**: 若仅因 `body_not_awake_or_settled` 终止，但未满足成功条件（例如悬浮在空中静止，或偏离目标垫静止），则不属于成功；但这类情形在合理动力学下罕见。
- **truncation**: 任务描述中未出现时间截断条件，但实际训练中可能存在未声明的最大步数截断；当前 step 源无 `truncated` 信息，奖励函数中不应依赖该假设。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
- **explicit_failure_flag_available**: false  
  step 返回的 `info` 固定为 `{}`，没有任何成功/失败标志或终止原因字段。奖励函数必须完全从 `next_obs` 的物理状态（位置、速度、接触）中推断是否成功着陆或发生硬碰撞。
- **allowed_info_fields**: 无（info 为空）。
- **forbidden_or_uncertain_info_fields**: `info["success"]`, `info["failure"]`, `info["termination_reason"]` 等未声明字段均禁止使用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用的参数：
- `obs`：合法，上一时间步的观测。
- `action`：合法，刚刚执行的离散动作。
- `next_obs`：合法，动作执行后的下一时刻观测，是奖励的核心依据。
- `info`：仅 `{}`，无可用字段，禁止从中提取任何信息。
- `training_progress`：若 prompt 未明确允许，默认禁止使用。

禁止使用的参数：
- `original_reward`：已掩盖，严禁引用任何形式的官方奖励。
- 任何未在观察空间中声明的 obs 切片。
- 任何未在 info 中声明的字段。

## 7. 可用于奖励函数的信号
- **position**: `next_obs[0]`（水平差距）, `next_obs[1]`（垂直高度）。两者可直接用于构建距离/接近奖励。
- **velocity**: `next_obs[2]`, `next_obs[3]`。尤其垂向速度对软着陆至关重要。
- **orientation**: `next_obs[4]`（倾角）, `next_obs[5]`（角速度）。可用于姿态稳定性奖励。
- **contact**: `next_obs[6]`, `next_obs[7]`。表示双腿是否触地，是判断着陆成功的关键二值特征。
- **action/engine**: 可利用 `action` 实现燃料惩罚（例如对 1,2,3 动作添加负奖励）。
- **other**: 无其他活跃信号。

## 8. 不确定或不可用的信号
- **显式成功/失败标志**：不存在。
- **步数或时间**：未观察到任何时间信号，不能直接构造“尽快完成”奖励，但可通过每步微小负奖励模拟时间惩罚，需谨慎使用。
- **腿部力量或接触力**：仅有 0/1 接触指示，无连续力/压强信息。
- **视口外惩罚细节**：不清楚水平超界时的具体位置值（可能被截断）, 奖励设计时不依赖该事件自身。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body with two support legs
  actuator_type: single main vertical thruster + two lateral orientation thrusters
  contact_structure: two point contacts, each binary collision flag
primary_objectives:
  - reach and rest at center target pad (zero relative horizontal position, zero altitude over pad)
  - achieve stable, upright landing at near-zero linear and angular velocities
secondary_objectives:
  - minimize main and orientation engine usage (fuel efficiency)
  - implicitly minimize episode length (fast settling), though no explicit timer
main_failure_risks:
  - hard crash from excessive vertical speed or body contact other than legs
  - drifting out of horizontal bounds
  - landing upside down or with excessive tilt, causing crash
  - failure to reach target pad or remaining airborne until possible truncation
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id**: `goal_proximity`  
  purpose: 引导 agent 持续缩短与目标垫的水平距离和垂直距离。  
  why_required: 核心导航目标，必须始终施加。  
  usable_signals: `next_obs[0]`, `next_obs[1]`  
  risks: 若权重大于安全信号，可能导致直接高速坠落接触。

- **role_id**: `soft_landing_velocity_penalty`  
  purpose: 在 agent 接近垫面时，严厉惩罚垂向和水平速度，促使减速。  
  why_required: 实现“软接触”和稳定停靠，是避免坠毁的关键。  
  usable_signals: `next_obs[2]`, `next_obs[3]`  
  risks: 可能造成悬停或延长停留时间；需与 `goal_proximity` 平衡。

- **role_id**: `upright_orientation`  
  purpose: 维持身体倾角接近 0，并抑制角速度。  
  why_required: 倾斜过大会导致单腿先触地、另一腿悬空或身体侧面碰撞地面，直接触发 `crash_or_body_contact`。  
  usable_signals: `next_obs[4]`, `next_obs[5]`  
  risks: 过强的角度约束可能妨碍必要的姿态调整。

- **role_id**: `contact_success`  
  purpose: 当双腿同时接触垫面（`left_contact=1` and `right_contact=1`）时，给予大规模正奖励，标记着陆成功。  
  why_required: 这是唯一明确的“着陆完成”信号；结合速度几乎为零时，即可确信安全着陆。  
  usable_signals: `next_obs[6]`, `next_obs[7]`  
  risks: 可能诱导 agent 在目标垫外强行伸出双腿触地，因此必须与位置奖励联合使用。

### 10.2 条件职责 conditional_roles
- **role_id**: `fuel_efficiency`  
  condition_to_use: 当训练进入中后期，基本着陆能力已稳定时逐步引入。  
  usable_signals: `action` (对 1,2,3 给予小负奖励)  
  risks: 早期引入会严重限制探索，导致 agent 不敢使用主引擎。

- **role_id**: `angular_velocity_damping_near_surface`  
  condition_to_use: 当 `next_obs[1]`（高度）低于某个阈值时增强，防止接地瞬间旋转。  
  usable_signals: `next_obs[5]`  
  risks: 若阈值过高，可能干扰下降过程中的姿态校准。

- **role_id**: `per_step_time_penalty`  
  condition_to_use: 可选，用于鼓励快速完成；但可能引发冒险行为，仅在基本安全着陆率高时谨慎加微负值。  
  usable_signals: 可直接返回常数负值，不依赖任何观测。  
  risks: 过早使用会导致 agent 优先速度而非安全。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: `explicit_success_flag_reward`  
  reason: 因为 info 为空，没有可用的显式成功标志。  
  forbidden_or_missing_signals: `info["success"]` 等所有 info 字段。

- **role_id**: `terrain_awareness` 或 `force_based_penalty`  
  reason: 没有连续接触力测量，仅二值接触，无法构建精细接触力惩罚。  
  forbidden_or_missing_signals: 连续力/压强数据。

- **role_id**: `trajectory_style`（例如最小作用量或平滑性）  
  reason: 当前高维观测已足够，且离散动作下平滑性很难有效量化，暂不引入额外复杂度。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | `next_obs[0]`, `next_obs[1]` | – | `euclidean_distance`, `bounded_signal` | 可直接使用水平、垂直偏差的平方和或绝对值。 |
| soft_landing_velocity_penalty | `next_obs[2]`, `next_obs[3]` | – | `quadratic_penalty`, `dense_state_signal` | 主用垂向速度（`next_obs[3]`），水平速度次之。 |
| upright_orientation | `next_obs[4]`, `next_obs[5]` | – | `absolute_penalty`, `quadratic_penalty` | 角度和角速度可独立惩罚。 |
| contact_success | `next_obs[6]`, `next_obs[7]` | – | `logical_and` + `large_bonus` | 仅当两腿同时触地时发放高分。 |
| fuel_efficiency (conditional) | `action` (0,1,2,3) | – | `sparse_action_penalty` | 对 1,2,3 动作给予小额负奖励。 |
| angular_velocity_damping_near_surface (conditional) | `next_obs[5]`, `next_obs[1]` (for gating) | – | `gated_quadratic` | 以高度阈值激活。 |
| per_step_time_penalty (conditional) | 无（常数） | – | `constant_negative` | 极小的常数负值，需后期启用。 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 高速撞击目标垫 | `next_obs[3]` 绝对值很大，终止时双腿虽接触但奖励可能过早高分。 | 增加垂向速度惩罚权重，或缩小 `contact_success` 激活时的速度阈值，仅在低速接触时给予高分。 |
| 悬停不降 | 高度长期不变，速度接近零，但未接触垫子；若无时间截断则不会终止。 | 引入微小的高度-时间惩罚或势能函数，或在 `goal_proximity` 中增加垂向迫近力；也可添加逐步减弱悬停奖励的机制。 |
| 过度倾斜着陆 | `body_angle` 远离 0，最终只有单腿接触，可能触发 `crash_or_body_contact`。 | 加强 `upright_orientation` 权重，并在接近地面时增加角度惩罚。 |
| 在目标垫外触地 | 双腿接触但 `x_position` 偏离 0 较大。 | 在 `contact_success` 发放时乘以与 `x_position` 相关的高斯衰减项，或在接触时仍保留强烈的水平位置惩罚。 |
| 过度使用引擎 | 连续启用主引擎或姿态引擎，尽管已稳定着陆或悬停。 | 逐步引入 `fuel_efficiency` 惩罚，并确保着陆后 agent 学会执行 `no_engine`。 |
| 学习到“牺牲一腿”策略 | 始终保持大倾角，仅让一条腿接触，可能仍能触发 `body_not_awake_or_settled` 而误判为成功？ | `contact_success` 必须要求双腿同时接触，否则不给予成功奖励；同时加强倾角惩罚。 |
