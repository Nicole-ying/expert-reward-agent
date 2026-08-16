# Response Record

# 匿名环境理解卡片

## 1. 任务目标
Agent 控制一个 2D 飞行器/着陆器，从画面顶部附近出发，带有随机初始推力。核心目标是在最短时间内平稳降落到画面中央的目标着陆垫上，同时尽可能少地使用引擎。Agent 必须学会：快速接近目标、在靠近时主动减速、保持姿态稳定、实现安全的软接触着陆。附属目标（时间最优、燃料最优）服务于主目标，不构成多目标冲突。

## 2. 任务类型选择
selected_route_id: `navigation_goal_reaching`  
confidence: high  
reason: 核心任务是到达并稳定在指定目标位置（中央着陆垫），虽然伴随时间和能耗约束，但它们是从属优化项；动作空间用于控制推进和姿态，是为了实现精准到达，因此归属导航目标到达家族。

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (8,)
- dtype: float32 (推断)
- 字段说明：

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|---------------|
| 0 | x_position | 相对于目标着陆垫的水平坐标 | true |
| 1 | y_position | 相对于垫面高度的垂直坐标（向上为正） | true |
| 2 | x_velocity | 水平线速度 | true |
| 3 | y_velocity | 垂直线速度（向上为正） | true |
| 4 | body_angle | 机身倾角（弧度） | true |
| 5 | angular_velocity | 角速度 | true |
| 6 | left_support_contact | 左支撑腿接触标志（1.0 接触，0.0 非接触） | true |
| 7 | right_support_contact | 右支撑腿接触标志（1.0 接触，0.0 非接触） | true |

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：

| 动作索引 | 名称 | 含义 |
|----------|------|------|
| 0 | no_engine | 不启动任何引擎，无推力/姿态力矩 |
| 1 | left_orientation_engine | 启动左侧姿态引擎，产生负方向（逆时针）力矩 |
| 2 | main_engine | 启动主引擎，产生沿机身法向的推力（通常提供向上的力） |
| 3 | right_orientation_engine | 启动右侧姿态引擎，产生正方向（顺时针）力矩 |

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled`（机身静止或稳定）若同时满足：位置接近目标垫中心、两个支撑腿都接触垫面、垂向速度很小。此类终止最可能对应成功着陆。
- **failure-like termination**:  
  - `crash_or_body_contact`：除目标着陆垫之外的任何碰撞（比如机身撞地、单侧腿接触非目标区域等）。  
  - `horizontal_position_outside_viewport`：水平坐标超出边界，意味着飞离安全区域。  
- **ambiguous termination**: `crash_or_body_contact` 如果包含在目标垫上的接触，则会与成功情况混淆，但根据任务描述“make safe contact”且终止条件独立列出，推测着陆垫上的安全接触被归为 `body_not_awake_or_settled`，而 `crash_or_body_contact` 特指危险接触。  
- **truncation**: 未在代码或描述中提供，可能不存在时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: none （step 返回 info 为空字典）  
- forbidden_or_uncertain_info_fields: info 中任何未声明的字段均不可用（如 success、failure、termination_reason 等）。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`（当前观测向量，shape=(8,)，含义如上）
- `action`（当前执行的动作索引 0~3）
- `next_obs`（下一时刻观测向量）
- `info` 中明确允许的字段（当前为空，**禁止使用任何 info 字段**）

禁止使用：
- `original_reward`（官方奖励被完全屏蔽，不可访问）
- `training_progress`（本任务描述未明确授权使用，应视为不可用）
- 任何未在 observation_space 中声明的观测分量或 info 附加信息。

## 7. 可用于奖励函数的信号
- **position**:  
  - `next_obs[0]` (x_position) —— 目标：趋近 0  
  - `next_obs[1]` (y_position) —— 目标：趋近 0（或某个负值区域代表接触垫面）  
  可直接用于距离奖励。
- **velocity**:  
  - `next_obs[2]` (x_velocity) —— 鼓励接近目标时趋向 0  
  - `next_obs[3]` (y_velocity) —— 鼓励着陆时趋近 0 或小负值（软着陆）  
  可用于速度惩罚或接近目标时的阻尼奖励。
- **orientation**:  
  - `next_obs[4]` (body_angle) —— 目标：保持接近 0（垂直），可奖励角度小  
  - `next_obs[5]` (angular_velocity) —— 可惩罚角速度大小
- **contact**:  
  - `next_obs[6]` (left_support_contact)  
  - `next_obs[7]` (right_support_contact)  
  两者均为 1.0 时可能表示完全着陆，可结合位置、速度来判断成功。
- **action/engine**:  
  - `action` 本身：鼓励少用引擎（动作惩罚），可对主引擎（2）和姿态引擎（1,3）分别惩罚。
- **other**:  
  - 可构造接近终端的“着陆区”概念：当 x_position 小、y_position 接近垫面时，进一步强化软着陆条件。

## 8. 不确定或不可用的信号
- 无明确的“成功”或“失败”标志。
- 无法从 info 获得任何额外终止原因。
- 无法获得 body 是否在垫面上的精确逻辑（只能通过接触标志和位置间接推断）。
- 无法获得离散事件（如“刚刚接触垫面”）的直接脉冲，需要从连续观测变化中提取。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body with two extendable legs
  actuator_type: discrete thrusters (main engine + left/right orientation engines)
  contact_structure: two contact sensors on legs, binary flags
primary_objectives:
  - reach target landing pad (x→0, y→0 or designated pad level)
  - land softly (minimal impact velocity)
secondary_objectives:
  - save fuel (minimize engine usage)
  - land quickly (implicit time optimality)
  - maintain stable upright orientation during descent
main_failure_risks:
  - crashing into ground outside pad
  - drifting out of horizontal bounds
  - excessive vertical impact causing failure
  - never settling (hovering) and triggering truncation if exists
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: approach_target**  
  purpose: 鼓励 agent 快速移向目标着陆点区域  
  why_required: 这是任务的根本，不接近目标永远无法成功  
  usable_signals: [next_obs[0] (x), next_obs[1] (y)]  
  risks: 若奖励过强，agent 可能忽略速度控制，导致高速撞地

- **role_id: soft_landing**  
  purpose: 在接近目标时抑制垂直/水平速度，防止硬撞击  
  why_required: 成功的必要条件是低冲击着陆，但单独的位置奖励无法保证  
  usable_signals: [next_obs[2] (vx), next_obs[3] (vy), next_obs[0,1] 作为条件门控]  
  risks: 过早激活会拖慢接近；需依赖与目标的距离动态调节权重

- **role_id: fuel_efficiency**  
  purpose: 惩罚不必要的引擎使用，降低总推力消耗  
  why_required: 任务明确要求“as little engine thrust as possible”，且无其他惩罚时 agent 会无意义喷射  
  usable_signals: [action (0-3)]  
  risks: 若惩罚过重，可能使 agent 不敢使用必要的推力，导致坠落；建议只对非必要动作（如悬停时持续喷射）加重惩罚

### 10.2 条件职责 conditional_roles
- **role_id: orientation_stability**  
  condition_to_use: 当 agent 已经进入目标着陆区（x 小、y 接近垫面）时激活  
  usable_signals: [next_obs[4] (angle)]  
  risks: 在远离目标时要求姿态垂直会干扰最优飞行路径（可能需要倾斜来平移）。仅在着陆末段使用

- **role_id: two_leg_contact_reward**  
  condition_to_use: 当两条腿接触标志同时为 1.0，且位置接近原点，且垂直速度接近 0 时，可给予终点成功奖励  
  usable_signals: [next_obs[6], next_obs[7], next_obs[0, 1, 3]]  
  risks: 若仅基于接触标志而不管位置，可能将侧翻误判为成功。必须结合位置和速度条件

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: time_penalty**  
  reason: 虽然没有明确的 `training_progress` 可用，且无法简单地在全 episode 上施加负向常数（会与 fuel 惩罚混淆）。可通过进度奖励间接实现时间最优，不加独立的时间惩罚项更稳健。  
  forbidden_or_missing_signals: episode_step_count 未暴露，无法可靠使用

- **role_id: explicit_success_bonus**  
  reason: 由于 info 中没有 success 标志，而且我们只能推测终止原因，不适合使用绝对的“成功”奖励。采用条件接触奖励 + 位置速度阈值组合更安全。  
  forbidden_or_missing_signals: info 空，termination 无标志

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_target | `x_position`, `y_position` | None | `dense_state_signal`, `bounded_signal`, `euclidean_distance_penalty` | 可使用负的 L2 距离或分段线性，注意 y 方向定义 |
| soft_landing | `x_velocity`, `y_velocity`, `x_position`, `y_position`（用于门控） | None | `quadratic_penalty`, `gated_penalty` | 建议在距离目标较近时打开，惩罚速度平方 |
| fuel_efficiency | `action` | None | `action_penalty`, `multiplicative_discount_on_action` | 可对 main engine (2) 和 orientation (1,3) 设置不同权重 |
| orientation_stability | `body_angle` (cos²或绝对值) | None | `abs_penalty`, `quadratic` 或 `cosine_signal` | 门控激活，避免远端倾斜受到惩罚 |
| two_leg_contact_reward | `left_contact`, `right_contact`, `x`, `y`, `vy` | explicit landing event | `threshold_bonus`, `product_condition` | 仅当位置和速度符合着陆区条件时给予正奖励 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 悬停不降落 | episode长度稳定但短，reward停滞；y_position 不再减小 | 加强 position 下降方向的奖励，或引入时间压力（如逐渐增大的 penalty） |
| 高速撞垫 | 大量 episode 以 crash 终止，y_velocity 负值很大 | 加强软着陆项在低空时的权重，开启 velocity 的二次惩罚 |
| 过度使用主引擎（浪费燃料） | 动作 2 频率极高，但变化不大 | 增加 action 惩罚权重，或仅惩罚非必要连续使用 |
| 倾斜过度导致侧翻 | body_angle 偏离 0 较大后 episode 结束，left_contact 和 right_contact 不一致 | 在高度较低时加强 orientation_stability 条件角色 |
| 漂出水平边界 | x_position 绝对值持续增大，episode 以 out of viewport 结束 | 提高 x_position 惩罚，或添加水平速度阻尼项 |
| 单侧接触假性着陆 | 只有一条腿接触就终止，但位置和速度条件不满足 | 确保 two_leg_contact_reward 要求双边均接触且位置速度达标；否则不奖励 |
