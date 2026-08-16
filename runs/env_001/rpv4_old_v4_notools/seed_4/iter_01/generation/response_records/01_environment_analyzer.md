# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个2D飞行器轨迹优化任务。主体从视口顶部中心附近以随机初始力开始，核心目标是**尽快到达中心的目标垫板并稳定停靠**（位置对齐、速度归零、姿态平稳、安全接触）。次要目标是在满足到达的前提下**最小化引擎推力使用**（节省燃料）。任务不能混淆为纯粹的生存平衡，因为主体必须移动到指定地点，而不是单纯保持不倒。

## 2. 任务类型选择
- **selected_route_id**: navigation_goal_reaching
- **confidence**: high
- **reason**: 任务的最终成功明确依赖于“到达并稳定在目标垫板上”，属于典型的带软着陆的导航目标到达，节油和快速只是附属优化，不构成冲突的多目标问题。

## 3. 观察空间 observation_space
- **type**: Box
- **shape**: (8,)
- **dtype**: float32（典型）
- **各维度含义与奖励可用性**：
  - `obs[0]`: **x_position** – 相对于目标垫板的水平坐标，单位未知。奖励可用：是。
  - `obs[1]`: **y_position** – 相对于垫板高度的垂直坐标（越接近0表示越靠近垫板表面）。奖励可用：是。
  - `obs[2]`: **x_velocity** – 水平线速度。奖励可用：是。
  - `obs[3]`: **y_velocity** – 垂直线速度。奖励可用：是。
  - `obs[4]`: **body_angle** – 机体朝向角度。奖励可用：是。
  - `obs[5]`: **angular_velocity** – 角速度。奖励可用：是。
  - `obs[6]`: **left_support_contact** – 左支撑腿是否接触地面的二进制标志（1.0/0.0）。奖励可用：是。
  - `obs[7]`: **right_support_contact** – 右支撑腿是否接触地面的二进制标志（1.0/0.0）。奖励可用：是。

所有8维观测均可用于奖励函数设计。

## 4. 动作空间 action_space
- **type**: Discrete
- **n**: 4
- **动作含义**：
  - `0` (no_engine)：无推力输出，仅靠惯性运动。
  - `1` (left_orientation_engine)：点燃左侧姿态引擎，产生力矩使机体逆时针旋转（或按物理定义）。
  - `2` (main_engine)：点燃主引擎，提供正Y方向的推力（向上的力，用于减速下降/调整高度）。
  - `3` (right_orientation_engine)：点燃右侧姿态引擎，产生反向力矩，使机体顺时针旋转。

动作离散，适合进行使用惩罚（fuel efficiency），但无法直接获得推力大小的连续信号。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success‑like termination**:
  - `body_not_awake_or_settled`：机体进入休眠或判定为已稳定（速度、角速度极低并可能接触垫板）。这通常表示安全着陆任务完成。
- **failure‑like termination**:
  - `crash_or_body_contact`：机体的主体部分或非支撑腿部位发生碰撞，视为坠毁。
  - `horizontal_position_outside_viewport`：水平位置完全离开可见区域，相当于飞出任务区。
- **ambiguous termination**:
  - （无）
- **truncation**:
  - 所提供的 `step` 源码中 `truncated` 固定返回 `False`，因此没有时间截断，但实际包装可能隐含有最大步数，此处不作假设。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false（`info` 字典为空，无法直接获得成功/失败标志）。
- **explicit_failure_flag_available**: false。
- **allowed_info_fields**: 无（`info` 始终为 `{}`）。
- **forbidden_or_uncertain_info_fields**: 所有未声明的字段均禁止使用，特别是 `success`、`termination_reason`、`failure` 等。

## 6. reward 函数接口契约
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
**允许使用**：
- `obs` – 当前观测（8维）
- `action` – 当前动作（离散 0‑3）
- `next_obs` – 下一观测（8维）
- `info` – 只允许访问明确声明的字段（当前无）
- `training_progress` – 仅当后续 prompt 明确声明可用时才允许使用

**禁止使用**：
- `original_reward`
- 任何未声明的 `info` 字段
- 任何未文档化的观测切片或假设

## 7. 可用于奖励函数的信号
- **位置相关**：`x_position`、`y_position` – 可直接测量与目标垫板的水平和垂直偏差。
- **速度相关**：`x_velocity`、`y_velocity` – 衡量是否接近静止。
- **姿态相关**：`body_angle`、`angular_velocity` – 衡量机体是否水平稳定。
- **接触相关**：`left_support_contact`、`right_support_contact` – 判断是否双支撑安全落地。
- **动作/引擎**：`action` – 离散动作，可用于计算推力消耗（action ≠ 0 视为消耗燃料）。
- **其他**：无。

## 8. 不确定或不可用的信号
- 无明确的成功/失败标志（`info` 为空）。
- 无碰撞强度或接触力测量。
- 无剩余燃料量。
- 无精确时间步计数。
- 无法获得风扰或其他外部干扰的显式信号。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D vehicle-like body with bilateral landing legs
  actuator_type: discrete thrusters (main engine + two orientation engines)
  contact_structure: two support pads (left/right) that can contact the ground
primary_objectives:
  - reach the target pad (minimize x, y deviation)
  - settle stably (low velocity, low angular velocity, upright orientation)
  - achieve safe two-leg contact on the pad
secondary_objectives:
  - time-to-target (as fast as possible)
  - fuel saving (minimize engine usage)
main_failure_risks:
  - crashing with body parts other than legs
  - drifting out of the horizontal viewport
  - excessive oscillations that prevent settling
  - premature contact without controlled approach (hard landing)
  - fuel waste from constant thruster usage
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: goal_proximity**
  - purpose: 驱使机体向目标垫板靠近，并最终贴齐垫板表面。
  - why_required: 到达目标是任务根本，必须将位置误差驱动至零。
  - usable_signals: `x_position`, `y_position`, `x_velocity`, `y_velocity`
  - risks: 若纯用位置惩罚，可能导致高速冲击目标；需配合稳定性约束。

- **role_id: settling_stability**
  - purpose: 抑制接近目标时的速度、角速度和不必要摇摆，保证机体能够平稳降落。
  - why_required: 不稳定的接近会导致接触失败或触发碰撞终止。
  - usable_signals: `x_velocity`, `y_velocity`, `angular_velocity`, `body_angle`
  - risks: 若强度过高，可能阻止快速接近，需要在接近目标时逐渐加强。

### 10.2 条件职责 conditional_roles
- **role_id: safe_landing_contact**
  - condition_to_use: 当机体已充分接近目标垫板（如 `|x_position|<threshold` 且 `|y_position|<threshold`）且速度较低时激活。
  - purpose: 鼓励双支撑腿同时接触垫板，完成“软着陆”。
  - usable_signals: `left_support_contact`, `right_support_contact`
  - risks: 无条件激活可能导致 agent 在远处故意摔腿或撞击地面，必须配合 gate 条件。

- **role_id: fuel_efficiency**
  - condition_to_use: 作为轻度惩罚持续存在，但不应抑制必要的探索。可在训练后期增大权重。
  - purpose: 最小化引擎使用次数（每个非零动作计为消耗）。
  - usable_signals: `action`
  - risks: 过强的燃料惩罚会使 agent 完全不点火，无法完成导航。

- **role_id: upright_orientation**
  - condition_to_use: 仅在机体接近目标或速度较小时激活。
  - purpose: 保持机体角度接近 0，避免倾斜着陆。
  - usable_signals: `body_angle`
  - risks: 若全程强调角度，可能干扰大角度机动，减慢学习。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: explicit_success_bonus**
  - reason: 环境中没有可访问的 success flag，终止原因不可用于奖励函数，强行使用会导致不可知。
  - forbidden_or_missing_signals: `info["success"]` (不存在)。

- **role_id: sparse_reward**
  - reason: 奖励函数无法获取终止信息，仅靠终止事件作为稀疏奖励不可行；任务需要密集引导信号。
  - forbidden_or_missing_signals: 结束标志不可访问。

- **role_id: collision_penalty**
  - reason: 没有碰撞强度或碰撞事件的可观测信号，唯一碰撞感知来自终止，不能用于奖励内部。
  - forbidden_or_missing_signals: 碰撞传感器不存在。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|--------|----------------|----------------|----------------------------|-------|
| goal_proximity | `x_position`, `y_position`, `x_velocity`, `y_velocity` | - | `dense_state_signal`, `distance_to_target`, `velocity_to_target` | 组合位置与速度的 progress 奖励；可单独对横向/纵向偏差加权。 |
| settling_stability | `x_velocity`, `y_velocity`, `angular_velocity`, `body_angle` | - | `bounded_signal`, `quadratic_penalty`, `weighted_sum` | 速度惩罚可用平方或门控，接近目标时权重升高。 |
| safe_landing_contact | `left_support_contact`, `right_support_contact` | - | `gated_signal`, `(contact_left AND contact_right) * gate_progress` | 需与进度或低速门结合，防止误奖励。 |
| fuel_efficiency | `action` | exact thrust magnitude (离散动作) | `discrete_action_penalty`, `(action != 0) * penalty` | 轻微负奖励，可进行调度（如 curriculum）。 |
| upright_orientation | `body_angle` | - | `bounded_signal`, `abs_penalty` | 可配合接触条件使用，避免倒立着陆。 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|-------------|-------------------|----------------------|
| **hover without approaching** | `y_position` 不下降，`y_velocity ≈ 0`，agent 不点火主引擎或用错方向 | 增强 goal_proximity 的 y 方向权重，减小燃料惩罚，引入高度下降的 increment reward |
| **fly out of horizontal bounds** | `x_position` 持续增大，超出视口导致终止 | 增加横向偏差的惩罚，检查 orientation roles 是否过度鼓励旋转，可能需要姿态限幅 |
| **crash at high speed** | `|y_velocity|` 或 `|x_velocity|` 在接触前仍然很大，接触后立即 crash | 加强 settling_stability（尤其 y_velocity），在接近垫板时使用指数级增强的速度惩罚 |
| **waste excessive fuel** | 动作日志中非零动作频率极高，但位置无明显改良 | 适当增加 fuel_efficiency 惩罚，或采用“只有无动作时奖励”的保守策略；可能需配合探索 |
| **oscillate without settling** | `body_angle` 或 `x_position` 在垫板附近来回摆动，无法触发 body_not_awake | 提升稳定性的角速度/角度的惩罚权重，或将 settling 相关奖励设为在低速度区域内才最大化 |
| **always no‑op and fall** | agent 学习为 0 动作以避免燃料惩罚，但直接掉落导致 crash | 动态调整 fuel_efficiency 惩罚强度，确保 survival 相关奖励（如下降速度控制）远大于燃料惩罚 |
