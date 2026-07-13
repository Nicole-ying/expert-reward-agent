# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个 2D 载具式着陆任务。主体从视口上方中央附近开始，带有随机初速扰动。**核心目标**是在尽可能短的时间内安全抵达画面中央的目标着陆垫并稳定停驻，同时尽可能减少引擎推力消耗。智能体必须学会精确控制位置与速度，在接近着陆垫时减速、保持竖直姿态，并实现两条支撑腿的平稳接触。

**次要目标**：节约引擎燃料；快速完成任务。  
**不应混淆的目标**：不存在与到达目标同等权重的冲突目标（如“保持高速”或“探索未知区域”），燃料节省仅为附属要求。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**  
confidence: high  
reason: 任务描述明确要求“到达并安定在中央目标垫”（reaching a target pad），位置目标占绝对主导，速度、姿态、能耗均为辅助条件。选择此项符合主目标单一性的基本判断。

动力学子类型（dynamics_subtype）：**goal_approach_and_soft_contact**  
理由：要求接近目标、减速、保持稳定姿态并最终实现支撑腿的安全接触，属于典型的靠近目标并软接触/停靠类动力学。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 推测为 float32（来自 Box 观察）
- 各维度含义：

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|---------------|
| 0 | x_position | 相对于目标垫的水平坐标 | true |
| 1 | y_position | 相对于目标垫高度的垂直坐标 | true |
| 2 | x_velocity | 水平线速度 | true |
| 3 | y_velocity | 垂直线速度 | true |
| 4 | body_angle | 主体姿态角（绝对值或从竖直偏移） | true |
| 5 | angular_velocity | 角速度 | true |
| 6 | left_support_contact | 左支撑腿是否触地（1.0 或 0.0） | true（谨慎） |
| 7 | right_support_contact | 右支撑腿是否触地（1.0 或 0.0） | true（谨慎） |

**注意**：左/右支撑接触标志可用于奖励，但需考虑它们可能与导致终止的“crash_or_body_contact”混淆风险（见下文分析）。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 各动作含义：

| 动作编号 | 名称 | 含义 |
|----------|------|------|
| 0 | no_engine | 不点火，仅靠惯性漂移 |
| 1 | left_orientation_engine | 点燃左侧姿态调节引擎，产生一个方向扭矩/推力 |
| 2 | main_engine | 点燃主引擎（推测向上推力，对抗重力） |
| 3 | right_orientation_engine | 点燃右侧姿态调节引擎，产生相反方向扭矩/推力 |

动作空间为离散选择，每次步只能执行四个动作之一。无连续控制。

## 5. step 与终止条件分析
### 5.1 终止模式
源码中的终止判断为：
```python
terminated = crash_or_body_contact or horizontal_position_outside_viewport or body_not_awake_or_settled
```
其中 **crash_or_body_contact**、**horizontal_position_outside_viewport** 和 **body_not_awake_or_settled** 均为复合条件（具体实现未暴露）。

- **success-like termination**：`body_not_awake_or_settled` 可能表示主体已安定在着陆垫上（无运动或规定时间内稳定），此类终止可视为成功。
- **failure-like termination**：`crash_or_body_contact`（如翻滚、身体直接碰撞）与 `horizontal_position_outside_viewport`（飞出边界）明显是失败。
- **ambiguous termination**：`body_not_awake_or_settled` 若发生在着陆垫之外，可能不算成功，但由于无额外信息，无法在 step 返回值中分辨。环境没有提供显式的“成功 / 失败”标志。
- **truncation**：源码中 `step` 未返回 `truncated`（第四个返回值为 `False`），因此无时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**
- explicit_failure_flag_available: **false**
- allowed_info_fields: `info` 固定为空字典 `{}`，无任何可用字段。
- forbidden_or_uncertain_info_fields: 不可假设存在 `info["success"]`、`info["failure"]`、`info["termination_reason"]` 等。`info` 完全不可用。

## 6. reward 函数接口契约
函数签名（必须遵循）：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```
**允许使用**：
- `obs`：当前观察（8 维向量）
- `action`：当前执行的动作（0~3 整数）
- `next_obs`：下一时刻观察（8 维向量）
- `info`：仅限已声明的字段（目前为空，实际不应使用）
- `training_progress`：当且仅当环境 prompt 明确允许使用时可使用（本环境未提及，故禁止使用，必须忽略）

**禁止使用**：
- `original_reward`（官方奖励，已屏蔽，严禁引用或反推）
- 任何未声明或不确定的 `info` 字段（如 success、failure、termination_reason）
- 观察中未明确记录的维度（如 8 维以外的索引）
- 环境内部变量、真实环境名称、benchmark 假设

## 7. 可用于奖励函数的信号
基于观察和动作的可用信号分类如下：

- **位置**：`x_position`（0）、`y_position`（1）—— 相对目标垫坐标，目标为 (0, 0)。
- **速度**：`x_velocity`（2）、`y_velocity`（3）—— 目标为零。
- **方向/姿态**：`body_angle`（4）、`angular_velocity`（5）—— 理想为零（竖直稳定）。
- **接触/着陆**：`left_support_contact`（6）、`right_support_contact`（7）—— 标志双腿是否触地，可用于安全着陆奖励，但需注意避免与 crash 混淆（见下节）。
- **动作/引擎**：采取的动作 `action`（0~3）用于推力惩罚（如约束不点火为主，或限制除主引擎外的浪费）。可设计动作惩罚。

## 8. 不确定或不可用的信号
- **成功标志**：不存在。
- **失败标志**：不存在。
- **终止原因类型**：无法在 `compute_reward` 内获取。
- **接触信号的安全性区分**：`left_support_contact` 和 `right_support_contact` 本身不能区分安全着陆与导致 crash 的非法接触（但源码中终止条件 `crash_or_body_contact` 可能已包含身体碰撞，支撑腿接触如果是安全的，则这些标志是正信号；但奖励设计者须明白单步奖励无法知晓是否就此终止，因此直接奖励接触存在风险）。环境未公布接触垫位置或着陆区边界，所以无法通过位置精确判断是否在垫上。
- **燃料消耗测量**：无可直接观测的剩余燃料或瞬时推力大小，只能间接通过动作计数。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body with two landing legs
  actuator_type: discrete thrusters (main + left/right orientation)
  contact_structure: two-point support (left/right leg contacts)
primary_objectives:
  - reach target pad (x_position → 0, y_position → 0)
  - achieve near-zero linear velocities and zero angular motion
  - maintain upright orientation (angle ≈ 0)
secondary_objectives:
  - minimize engine usage (sparse firing)
  - minimize time-to-landing (implicitly incentivise fast settling)
main_failure_risks:
  - crashing body parts onto ground (termination by crash_or_body_contact)
  - moving out of viewport horizontally
  - landing outside the pad but settling (ambiguous failure, no explicit signal)
  - excessive fuel consumption leading to suboptimal policy
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: goal_proximity_reward**  
  purpose: 鼓励智能体缩短与目标垫的距离，推动快速到达。  
  why_required: 到达目标是本任务第一优先级，缺乏位置奖励会导致无目标探索。  
  usable_signals: [x_position, y_position]（可构造距离或分量惩罚）  
  risks: 若权重过大可能导致过度加速、忽略减速阶段；需配合速度稳定职责。

- **role_id: stability_penalty**  
  purpose: 在接近目标区域时抑制速度和姿态振荡，确保平稳着陆。  
  why_required: 如果没有速度和姿态惩罚，智能体可能以高速撞击垫子、触发 crash 或无法安定。  
  usable_signals: [x_velocity, y_velocity, body_angle, angular_velocity]  
  risks: 全局惩罚可能过早抑制探索阶段的速度；建议使用位置门控（接近垫子时增强）。

- **role_id: fuel_thrust_penalty**  
  purpose: 抑制不必要的引擎点火，节省燃料。  
  why_required: 任务明确要求“使用尽可能少的引擎推力”，且离散动作使得每次非零动作都有代价。  
  usable_signals: [action]（可通过动作不等于 0 计数或对应掩码）  
  risks: 过重的推力惩罚可能导致智能体不敢点火，无法稳定姿态；需在降落阶段适当放松。

### 10.2 条件职责 conditional_roles
- **role_id: safe_contact_bonus**  
  condition_to_use: 当智能体已接近目标垫（例如距离或高度小于某阈值）且同时满足 `left_support_contact==1` 且 `right_support_contact==1` 时，给予额外奖励以鼓励双腿安全着陆。  
  若不满足接近条件则不触发。  
  usable_signals: [x_position, y_position, left_support_contact, right_support_contact]  
  risks: 如果在远离垫子时也奖励接触，可能导致智能体刻意用腿触地触发奖励但实际发生 crash（因终止判断可能包含 body_contact）。使用需极度谨慎并配合位置门控。若环境训练中出现异常，应临时禁用。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: termination_success_reward**  
  reason: 缺少显式成功标志，`info` 为空，无法在每步区分成功终止与平凡安定。  
  forbidden_or_missing_signals
