# 匿名环境理解卡片

## 1. 任务目标
本环境为一个二维飞行器轨迹优化任务。飞行器从视口上方中心附近出发并带有随机初速度，目标是**尽可能快地安全降落在画面中央的着陆平台上**，同时**尽量减少引擎推力**。着陆时必须满足：位置接近目标点、速度极低、保持近似水平的姿态、且两条支撑腿同时接触平台（软着陆）。次要目标是降低燃料消耗和缩短飞行时间。任务的本质是“导航至目标点 + 软接触”类问题，切勿与纯平衡或纯探索任务混淆。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**  
confidence: **high**  
reason: 核心目标是驱动飞行器到达指定的目标垫位置（x≈0, y≈0）并稳定着陆。燃料节省与姿态保持都是附属要求，不构成同等权重、彼此冲突的多目标，因此不属于multi_objective_task或survival_balance_task。

## 3. 观察空间 observation_space
- type: `Box`
- shape: `[8]`
- dtype: float32（默认，每个维度均为数值）
- obs[0]: **x_position** – 飞行器相对于目标垫的水平坐标，reward_usable: true  
- obs[1]: **y_position** – 飞行器相对于垫面高度的垂直坐标，reward_usable: true  
- obs[2]: **x_velocity** – 水平线速度，reward_usable: true  
- obs[3]: **y_velocity** – 垂直线速度，reward_usable: true  
- obs[4]: **body_angle** – 身体朝向角（0 为水平？根据任务描述，应鼓励接近 0°），reward_usable: true  
- obs[5]: **angular_velocity** – 角速度，reward_usable: true  
- obs[6]: **left_support_contact** – 左支撑腿接触标志（0/1），reward_usable: true  
- obs[7]: **right_support_contact** – 右支撑腿接触标志（0/1），reward_usable: true  

> 以上 8 维全部可用于奖励函数构造。info 字典为空，无任何额外信号。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: **no_engine** – 不点火，悬空/惯性滑行  
- action 1: **left_orientation_engine** – 点火左姿态引擎，产生逆时针力矩（减小角度）  
- action 2: **main_engine** – 点火主引擎，产生沿机身向上的推力（用于减速上升/下降）  
- action 3: **right_orientation_engine** – 点火右姿态引擎，产生顺时针力矩（增大角度）  

> 主引擎提供纵向力，方向发动机提供转矩，任务需学习推力与姿态的组合。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination:**  
  - `body_not_awake_or_settled` 看似是休眠/稳定着陆，但该条件是否伴随安全接触未知。必须结合观测中的双接触标志、位置和速度才能推断是否为成功。
  - `crash_or_body_contact` 中可能包含“通过两条腿接触垫面”的情况，但同样需上下文判断。
- **failure-like termination:**  
  - `crash_or_body_contact` 中的“crash”（身体侧面撞击、速度过高导致非腿部接触）很可能构成失败。
  - `horizontal_position_outside_viewport`（水平飞出边界）明确是失败。
- **ambiguous termination:**  
  - 单一的 `crash_or_body_contact` 或 `body_not_awake_or_settled` 无法区分成功/失败，因缺少内部标签。
- **truncation:** 未显式声明，可能由环境内部最大步数触发，但不提供额外信号。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**  
- explicit_failure_flag_available: **false**  
- allowed_info_fields: `{}`（空字典，不允许使用任何字段）  
- forbidden_or_uncertain_info_fields: 所有未在源码中出现的字段（如 `success`、`failure`、`terminal_reason`）均不可用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用：**
- `obs` – 当前步状态向量
- `action` – 当前步动作（标量 int）
- `next_obs` – 下一步状态向量
- `info` – 仅允许空字典 `{}`，不得假设任何键存在
- `training_progress` – 在本任务描述中未被允许，故**禁止使用**

**禁止使用：**
- `original_reward`（被屏蔽的官方奖励）
- `official_reward` 或任何内部奖励信号
- 未声明的 `info` 字段（如 success、failure 等）
- 未在观察空间中声明的任何隐式特征

## 7. 可用于奖励函数的信号
直接从 obs / next_obs / action 提取：
- **位置类：** `x_position`，`y_position`（距目标垫的距离）
- **速度类：** `x_velocity`，`y_velocity`（线速度，尤其垂直速度对软着陆重要）
- **姿态类：** `body_angle`，`angular_velocity`
- **接触类：** `left_support_contact`，`right_support_contact`（布尔标志，可用于判断双腿着陆）
- **动作/引擎：** 离散动作值（可施加动作惩罚以鼓励燃料节省）
- **其他：** 无

## 8. 不确定或不可用的信号
- 明确的成功/失败标志（info 中不存在）
- 燃料消耗的直接度量（只能通过动作惩罚近似）
- 任务耗时（缺少 time stamp 或 step 计数可用，且 training_progress 禁止）
- 随机初始力信息（不可观测）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body (vehicle-like) with two legs
  actuator_type: main vertical thruster, two opposing orientation thrusters
  contact_structure: two sparse leg contacts (left/right, binary)
primary_objectives:
  - reach target pad (x ≈ 0, y ≈ 0) and settle safely
  - both legs contact, near zero velocity, near zero angle
secondary_objectives:
  - minimize engine thrust (reduce fuel consumption)
  - minimize flight duration (encourage fast landing)
main_failure_risks:
  - crashing (body‑side impact, leg‑only contacts absent)
  - horizontal drift out of viewport
  - landing too fast or at large angle → crash
  - prolonged hovering without progress
  - unstable oscillation around target
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: goal_proximity_reward**  
  purpose: 鼓励飞行器向目标垫靠近  
  why_required: 到达目标点是核心目标，需持续引导  
  usable_signals: [x_position, y_position]  
  risks: 距离为零后可能不再提供激励，需配合其他约束  

- **role_id: soft_landing_velocity_penalty**  
  purpose: 在接近目标时抑制线速度，特别是垂直速度  
  why_required: 安全着陆要求速度极小，否则即使位置正确也会触发 crash  
  usable_signals: [x_velocity, y_velocity, 可通过距离做加权]  
  risks: 全局抑制速度会阻碍前期快速接近，宜用距离门控  

- **role_id: orientation_stability_penalty**  
  purpose: 惩罚身体倾斜，鼓励保持水平姿态  
  why_required: 倾斜过大会导致着陆失败或单侧接触  
  usable_signals: [body_angle]  
  risks: 轻微倾斜不影响着陆，可设计容忍区间  

- **role_id: contact_reward**  
  purpose: 鼓励双腿接触地面，作为软着陆成功的前兆  
  why_required: 双接触是完成任务的关键条件  
  usable_signals: [left_support_contact, right_support_contact]  
  risks: 若过早给予高额奖励，飞行器可能未靠近地面就“虚拟接触”。需与位置/速度条件结合  

- **role_id: action_efficiency_penalty**  
  purpose: 惩罚引擎使用，降低燃料消耗  
  why_required: 次要目标，且避免无意义抖动  
  usable_signals: [action]  
  risks: 过度惩罚可能导致 agent 不点火而坠毁，需合理权重  

### 10.2 条件职责 conditional_roles
- **role_id: terminal_success_bonus**  
  condition_to_use: episode 结束时，根据最终观测判断是否成功着陆（位置接近目标、速度‖<阈值、角度<阈值、双接触=1）  
  usable_signals: [next_obs 的所有维度]  
  risks: 阈值设定不当会漏判或产生虚假奖励；需注意接触可能在终止前一步已达成  

- **role_id: early_failure_penalty**  
  condition_to_use: episode 结束时，若离开安全区域或发生非双腿接触 crash，给予惩罚  
  usable_signals: [next_obs 的 x_position 是否超出视口，或出现单侧接触但位置遥远／速度过大]  
  risks: 可能错误地将“停在半空”判定为失败，需谨慎设定条件  

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: time_penalty_or_progress_bonus**  
  reason: 缺少时间信号，training_progress 被禁止，无法构造可靠的时间激励  
  forbidden_or_missing_signals: [step count, elapsed time]  

- **role_id: info_based_success_failure**  
  reason: info 为空，不可使用  
  forbidden_or_missing_signals: [info.success, info.failure]  

- **role_id: dense_energy_consumption_measure**  
  reason: 无法直接测量引擎推力的数值（只有离散动作），只能通过动作频率间接估计  
  forbidden_or_missing_signals: [instantaneous thrust magnitude]  

## 11. role_to