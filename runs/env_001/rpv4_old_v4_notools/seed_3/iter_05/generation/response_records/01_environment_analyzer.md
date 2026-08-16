# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个 2D 刚体飞行/着陆任务。一个身体从视口顶部中心附近被随机初始冲量射出，目标是以**最快的速度**到达并稳定留在**中心目标平台**上，同时**尽可能少用引擎推力**。智能体必须学会接近目标、降低速度、保持合适身体姿态、最终让双脚安全接触平台并静止。

**主任务**：导航到目标位置并稳定着陆（到达 + 软接触）。  
**次任务**：减少燃料消耗（少用推进器）、缩短飞行时间。  
**不应混淆的目标**：单纯追求高速撞击、持续在空中徘徊、仅保持姿态不移动。

## 2. 任务类型选择
selected_route_id: `navigation_goal_reaching`  
confidence: high  
reason: 核心目标是到达并稳定在一个空间固定目标点（平台）上，具有明确的到达与接触完成条件。附属优化（省燃料、求快）不构成权重冲突的第二个独立目标，因此属于导航目标到达族。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float64 (假设)
- obs[0]: `x_position` 相对目标平台的水平坐标，reward_usable: true（可用于距离/误差信号）
- obs[1]: `y_position` 相对平台高度的垂直坐标（上方为正），reward_usable: true（可用于高度/加速度控制）
- obs[2]: `x_velocity` 水平线速度，reward_usable: true（速度惩罚、减速信号）
- obs[3]: `y_velocity` 垂直线速度，reward_usable: true（下降速度控制、接近软着陆）
- obs[4]: `body_angle` 身体绝对倾斜角，reward_usable: true（姿态稳定，保持直立）
- obs[5]: `angular_velocity` 角速度，reward_usable: true（姿态剧烈变化惩罚）
- obs[6]: `left_support_contact` 左支撑接触标志（0/1），reward_usable: true（接触奖励、着陆成功条件）
- obs[7]: `right_support_contact` 右支撑接触标志（0/1），reward_usable: true（同上）

所有观测均可直接用于奖励设计。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine` — 不点火，依靠惯性运动
- action 1: `left_orientation_engine` — 点燃左侧姿态发动机，产生旋转力矩（可能是逆时针？）
- action 2: `main_engine` — 点燃主发动机，产生指向主要方向的推力（通常向下反推）
- action 3: `right_orientation_engine` — 点燃右侧姿态发动机，产生相反旋转力矩

连续控制维度：无，纯离散。

## 5. step 与终止条件分析
### 5.1 终止模式
根据 source 中 `terminated = crash_or_body_contact or horizontal_position_outside_viewport or body_not_awake_or_settled`：
- `crash_or_body_contact`：身体主体与地面或障碍物碰撞（非预期的强接触），典型失败终止。
- `horizontal_position_outside_viewport`：水平位置超出屏幕边界，失败终止。
- `body_not_awake_or_settled`：身体进入休眠或已稳定（速度/角速度极低，接触平台）。可能代表成功着陆，但也可能出现在空中或远离目标提前停止（如游戏超时或因某种冻结），因此是**模糊终止**，必须结合位置和接触条件判断。
- 没有显式的 `truncation` 标志，环境未提供截断信息。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空，无 `"success"` 字段）
- explicit_failure_flag_available: false （同上）
- allowed_info_fields: `{}` （无可用字段）
- forbidden_or_uncertain_info_fields: `{"success": "缺失", "failure": "缺失", "termination_reason": "缺失"}`
- 因此所有成功/失败判定必须从观测中自行组合逻辑，例如：
  - 成功候选：`body_not_awake_or_settled` 且 双脚接触 `(obs[6]==1 and obs[7]==1)` 且 相对位置在允许范围内。
  - 失败候选：`crash_or_body_contact` 或 `horizontal_position_outside_viewport`，或错误的静止情况。

## 6. reward 函数接口契约
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- `obs` 当前观测 (shape [8])，全部维度可用
- `action` 当前动作 (0..3)
- `next_obs` 下一观测
- `info` 中目前为空字典，不可依赖
- `training_progress` 除非 prompt 明确要求，否则**不应使用**

禁止使用：
- `original_reward` 或任何形式的官方奖励
- 未声明的 `info` 字段
- 未声明的观测切片外的值

## 7. 可用于奖励函数的信号
- **position**: `x_position`, `y_position` — 距目标平台误差、高度控制
- **velocity**: `x_velocity`, `y_velocity` — 速度量级、下降速率
- **orientation**: `body_angle`, `angular_velocity` — 姿态稳定
- **contact**: `left_support_contact`, `right_support_contact` — 着陆/接触状态
- **action/engine**: 动作选择可用于燃料消耗惩罚（如 `action != 0` 为使用引擎）
- **other**: 无额外信号

## 8. 不确定或不可用的信号
- 无明确的“得分”或“完成”指示器
- 环境不提供平台范围或具体接触法向量
- 没有外部定位系统（如 landmark）
- 没有引擎推力大小或燃料剩余量（仅知动作是否使用引擎）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body with bilateral supports (two legs/pads)
  actuator_type: discrete thrusters (main engine + two orientation engines)
  contact_structure: two-point contact (left & right support feet)
primary_objectives:
  - Reach the target platform (minimize relative horizontal/vertical position)
  - Achieve stable, safe contact (both support contacts active, low velocity, upright attitude)
secondary_objectives:
  - Minimize fuel usage (reduce engine actuation count/intensity)
  - Complete the task quickly (implicit via episode length, not explicit reward required)
main_failure_risks:
  - Crashing into ground/obstacle due to excessive downward speed or bad orientation
  - Drifting outside allowed horizontal range
  - Premature "settled" state far from target or without proper contact, causing early termination
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id**: `progress_toward_target`  
  purpose: 鼓励智能体沿最短路径向目标平台移动，缩短位置误差  
  why_required: 任务是到达目标，无进度信号则无引导  
  usable_signals: `x_position`, `y_position`（相对误差），`next_obs` 的相同维度  
  risks: 若只给正奖励，可能诱导高速冲撞目标，需配合减速职责

- **role_id**: `soft_landing_and_contact`  
  purpose: 奖励在接近目标且满足接触条件时以低速、小姿态着“陆”，确保稳定停靠  
  why_required: 仅到达不足以成功，需要双脚接触且无反弹/侧翻  
  usable_signals: `x_velocity`, `y_velocity`, `body_angle`, `angular_velocity`, `left_support_contact`, `right_support_contact`, `x_position`, `y_position`  
  risks: 过于严格的减速要求可能使学习前期难获得奖励，需分区间或结合课程

- **role_id**: `crash_and_boundary_penalty`  
  purpose: 阻止智能体因失控导致提前失败终止（碰撞、出界）  
  why_required: 终止事件会结束回合并切断学习信号，必须明确惩罚  
  usable_signals: 由终止模式触发（但无显式标志），可通过对 `next_obs` 状态推测（如速度突变、位置飞速变化）或在 episode 结束处基于 `terminated` 与位置/速度判断，但准确度受限。建议设计时用**临界值警告**：对过大的垂直速度、水平位置超阈提供密集惩罚。  
  risks: 在缺失显式失败标志时可能不精确，但密集惩罚可弥补。

### 10.2 条件职责 conditional_roles
- **role_id**: `fuel_efficiency`  
  condition_to_use: 在整个 episode 中启用，但权重不宜过高，避免智能体选择完全不动作。  
  usable_signals: `action`（通过 action != 0 判断用引擎），可加和并除以步数或直接惩罚每次点火  
  risks: 强惩罚可能导致智能体永远不点燃主引擎而无法到达目标，需与进度职责平衡；初始学习阶段可减轻权重。

- **role_id**: `orientation_stability`  
  condition_to_use: 全程使用，但严格倾斜角度限制可在接近目标时加强。  
  usable_signals: `body_angle`, `angular_velocity`  
  risks: 在远处过度要求零倾角会限制机动，可用宽松规范 + 靠近目标时收紧。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: `velocity_maximization`  
  reason: 任务需要快速到达，但简单的速度最大化会鼓励冲撞和不稳定行为；速度应作为“在安全范围内的适当快速”而不是目标。该职责与软着陆冲突，禁用。  
  forbidden_or_missing_signals: 缺乏显式的“安全速度上限”，需从接触动能反向推导，不可直接设高速度奖励。

- **role_id**: `direct_success_flag_reward`  
  reason: 无显式 `success` 标志，不能以该 flag 发放大型奖励。  
  forbidden_or_missing_signals: `info["success"]` 不存在。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| progress_toward_target | `x_position`, `y_position` 及其变化量 | 无 | `distance_to_goal`, `negative_l2_norm`, `delta_distance` | 越靠近越好，可用连续负值惩罚距离 |
| soft_landing_and_contact | 速度、角度、角速度、双脚接触、与目标的接近度 | 无 | `bounded_signal` (速度阈值), `indicator` (接触), `quadratic_penalty` (倾角) | 接近目标且双脚接触时给予大正奖励，否则严格惩罚高速、大倾角 |
| crash_and_boundary_penalty | 推测的越界条件（位置、速度）或 episode 结束判断 | 显式 fail flag | `threshold_indicator` (|x|>max_x, vy < -danger_limit) | 当成密集约束使用，比如当 y_vel 过负时持续惩罚 |
| fuel_efficiency | action (0=no_engine) | 剩余燃料量 | `episode_action_count`, `per_step_penalty` if action!=0 | 轻柔惩罚，以免抑制必要动作 |
| orientation_stability | `body_angle`, `angular_velocity` | 无 | `quadratic_cost` (angle), `abs_penalty` (angvel) | 后期可随接近目标增加权重 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 盘旋不下降、过度节约燃料 | 平均 episode 长度很长，y_position 初始后几乎不变，不落地 | 增加对高度维持的惩罚（鼓励减少 y_pos），或增加进度奖励中高度的权重 |
| 高速垂直冲击导致 crash | y_velocity 终末巨大负值，随即 episode 结束（crash） | 引入下降速度上限的密集惩罚，并对 soft_landing 奖励放松距离条件，先奖轻微接触 |
| 姿态不稳翻滚出界 | angular_velocity 持续高值，body_angle 大幅振荡，x_position 很快越界 | 加强姿态稳定惩罚，尤其是接近目标时；也可添加对 angular_velocity 的 L1/L2 惩罚 |
| 仅一脚接触便静止（body_not_awake 但只有一侧接触） | 一条腿接触，另一条腿悬空，速度极低，agent 误以为成功 | 明确 rewarding 必须双脚同时接触，单脚接触给予正奖励但不足以终止训练，同时检查 episode 结束时的接触状态 |
| 过度依赖一种方向引擎导致轨迹弯曲 | 动作分布极不均衡，x_position 偏移过大 | 增大 x_position 偏离的惩罚，必要时对动作分布加熵奖励或课程引导平衡控制 |
