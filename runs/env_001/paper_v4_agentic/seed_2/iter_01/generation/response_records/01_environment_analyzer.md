# Response Record

# 匿名环境理解卡片

## 1. 任务目标
任务主目标是控制一台具有左右支撑腿和主/侧向引擎的 2D 着陆器，从视野顶部中央附近以随机初始推力开始，尽快到达场地中央的着陆垫，并稳定、安全地停靠在该垫上。次要目标是尽量减少引擎使用（燃料消耗），同时保持车身姿态稳定，实现软着陆。不应将单纯的快速到达或单纯省燃料作为独立核心目标，代理必须在安全着陆的前提下兼顾速度与能效。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是到达指定目标垫并停留，附属有速度、姿态、能耗约束，属于典型的导航目标达到任务。不存在其他同等权重且冲突的核心目标。

动力学子类型进一步判断：  
dynamics_subtype: goal_approach_and_soft_contact  
理由：需要朝着固定目标接近，并在低速、低角速度下通过支撑腿产生安全接触完成着陆。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32（根据环境惯例，具体从环境读取，但推测为 float）  
- obs[0]: x_position，相对目标垫的水平坐标，reward_usable: true  
- obs[1]: y_position，相对着陆垫高度的垂直坐标，reward_usable: true  
- obs[2]: x_velocity，水平线速度，reward_usable: true  
- obs[3]: y_velocity，垂直线速度，reward_usable: true  
- obs[4]: body_angle，车身俯仰/横滚角度，reward_usable: true  
- obs[5]: angular_velocity，角速度，reward_usable: true  
- obs[6]: left_support_contact，左支撑腿接触标志（0.0 或 1.0），reward_usable: true（需谨慎使用）  
- obs[7]: right_support_contact，右支撑腿接触标志（0.0 或 1.0），reward_usable: true（需谨慎使用）

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- action 0: no_engine，不点火，不做任何事情  
- action 1: left_orientation_engine，点燃左姿态引擎（产生方向性的力）  
- action 2: main_engine，点燃主引擎（通常向上推力）  
- action 3: right_orientation_engine，点燃右姿态引擎（与左引擎反向）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 当 `body_not_awake_or_settled` 为真，且未发生 crash/出界时，可推断为成功着陆并稳定。
- failure-like termination: `crash_or_body_contact`（猛烈碰撞或非法身体接触）和 `horizontal_position_outside_viewport`（水平飞出视野）均视为失败。
- ambiguous termination: 单独的 `body_not_awake_or_settled` 可能发生在成功着陆后不久，也可能是因摔落后静止，需结合位置、速度判断。
- truncation: 环境中未提供 truncation 信号（step 返回 `terminated, False`），因此不存在时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（step 返回 info={}）  
- forbidden_or_uncertain_info_fields: 任何 info 字典中的字段均不存在，不可使用

间接推断路径：
- 成功着陆可从 termination 时接近目标垫中心、垂直速度极低、角度接近 0 且至少一腿接触的条件组合推断（derived_possible）。
- crash 可从 termination 时 y_position 骤降至地面以下或水平位置远超边界等条件间接检测（derived_possible）。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs（当前观测，8维）
- action（当前动作，整数 0~3）
- next_obs（下一步观测，8维）
- info 中明确允许的字段（当前环境均为空，无可用字段）
- training_progress（本环境 prompt 未明确允许，默认不应使用）

禁止使用：
- original_reward（原始奖励被屏蔽）
- official_reward
- 未在观测空间中定义的 obs 切片
- info 中任何未声明的字段
- 假设终止原因字符串或 success 标志存在

## 7. 可用于奖励函数的信号
- position：
  - 相对垫的水平距离 `|x_position|`
  - 相对垫高度的垂直距离 `|y_position|`（当 y_position 为正代表高于垫，到达垫面时理想 y≈0）
- velocity：
  - 水平速度 `x_velocity`（软着陆要求接近0）
  - 垂直速度 `y_velocity`（负值代表下落，着陆瞬间需要小）
- orientation：
  - body_angle（理想接近0，可取其绝对值或二次惩罚）
  - angular_velocity（软着陆应接近0）
- contact：
  - left_support_contact / right_support_contact（至少一腿接触可能表明着陆成功，但需结合速度和位置，否则可能鼓励猛烈砸地）
- action/engine：
  - action 本身（可计算引擎使用惩罚，no_engine 时无推力）
- other：
  - 通过 next_obs 与 obs 的差值得出速度变化，可用于检测剧烈推力
  - 衍生信号：是否接近目标垫且速度降低（derived_possible）

## 8. 不确定或不可用的信号
- info["success"] 或 info["failure"]：不存在
- 真实燃料消耗量、引擎推力大小：环境抽象为离散动作，无直接推力值
- 外部风扰等不可观测因素：无相关观测
- 任务耗时或剩余时间：无此信息

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D lander with two support legs
  actuator_type: one main engine (upward thrust) and two lateral orientation engines
  contact_structure: two legs ground contact (binary flags)
primary_objectives:
  - reach target pad (x≈0, y≈0 relative to pad)
  - achieve soft landing (low vertical velocity, near-zero horizontal velocity, near-zero angle)
secondary_objectives:
  - minimize engine ignition count (fuel/thrust penalty)
  - fast arrival (through truncation pressure or sparse bonus)
main_failure_risks:
  - crash due to high vertical speed or flipped angle
  - flying out of bounds horizontally
  - hovering but never landing (inefficient)
  - slamming onto pad by relying on contact reward too early
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity
  purpose: 引导代理向目标垫中心移动
  why_required: 没有此职责代理无法得知目标方位
  usable_signals: [x_position, y_position, 派生距离]
  risks: 可能导致高速撞击目标垫，需与软着陆结合

- role_id: soft_landing_velocity
  purpose: 在接近目标时抑制水平和垂直速度，使着陆平稳
  why_required: 没有减速约束会导致硬着陆或撞毁
  usable_signals: [x_velocity, y_velocity, 接近目标时的位置]
  risks: 过早施加减速奖励可能阻碍接近阶段，需与 proximity 联合调节

- role_id: orientation_stability
  purpose: 保持车身水平（角度和角速度小）
  why_required: 大角度会导致支撑腿不能安全接触或增加 crash 风险
  usable_signals: [body_angle, angular_velocity]
  risks: 可能在飞行阶段抑制必要的姿态微调

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency
  purpose: 惩罚非必要的引擎使用，鼓励节省燃料
  condition_to_use: 在整个 episode 中启用，但权重应较低，并且当代理已接近 pad 且即将着陆时可适当降低惩罚以避免不点火而漂移
  usable_signals: [action]（encourage action 0）
  risks: 过强的燃料惩罚会导致代理不愿点火而无法控制姿态或减速

- role_id: soft_contact_bonus
  purpose: 当代理以低速、小角度并位于垫上方时腿接触给予小量正向奖励
  condition_to_use: 仅在 (|x_position| < 阈值) 且 (|y_velocity| 小) 且 (|x_velocity| 小) 且 (|body_angle| 小) 时启用
  usable_signals: [left_support_contact, right_support_contact, x_position, y_velocity, x_velocity, body_angle]
  risks: 不加条件直接奖励接触会鼓励摔落式砸地

### 10.3 慎用/禁用职责 avoid_roles
- role_id: success_termination_bonus
  reason: 环境未提供显式成功标志，无法在 episode 结束时可靠发放离散奖励；且用间接推断易出错
  forbidden_or_missing_signals: [info.success, explicit_success_flag]

- role_id: time_penalty
  reason: 无时间截断信号，无法按时间步惩罚；但可通过期望快速完成的压力用 future return 体现

- role_id: dense_orientation_penalty_in_early_phase
  reason: 初期随机推力可能引起大幅摆动，若过早严苛惩罚角度会抑制探索；应逐渐加强

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | x_position, y_position, derived_distance | — | dense_state_signal, l2_norm, exponential_decay | 距离越近奖励越高 |
| soft_landing_velocity | x_velocity, y_velocity, x_position (as condition) | — | bounded_signal, quadratic_penalty, conditional_gating | 在接近目标时激活速度惩罚 |
| orientation_stability | body_angle, angular_velocity | — | absolute_penalty, quadratic_penalty | 全 episode 使用，可随目标接进度动态调整权重 |
| fuel_efficiency | action | engine_thrust_metric | action_penalty (binary for non-zero) | 全 episode 启用 |
| soft_contact_bonus | left_support_contact, right_support_contact, x_position, y_velocity, x_velocity, body_angle | — | conditional_reward, logical_and | 仅当多条件满足时给予 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 代理悬停不动，不敢降落 | episode 长度长，位置保持在远处，动作多为 no_engine | 降低燃料惩罚权重，增加距离奖励的渐进压力 |
| 高速撞击目标垫 crash | 支撑腿接触时垂直速度过大，或 body_angle 大 | 增强 soft_landing_velocity 和 orientation_stability 权重，或在接触条件中要求速度<阈值 |
| 飞出水平边界 | x_position 超过某个极限导致终止 | 增加距离中心越远越重的惩罚，并可小量水平位移惩罚 |
| 过早依赖引擎侧推导致抖振 | 姿态频繁切换，angular_velocity 大 | 追加角度变化率的惩罚，或在早期使用较小 action_penalty |
| 左右腿接触奖励被错误奖励摔落 | 接触次数多但 episode 短且 crash 率高 | 收紧 soft_contact_bonus 的条件门，尤其严格限制速度、角度 |
