# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
控制一个平面单腿关节体，使其向前连续跳跃前进。  
要求尽可能保持身体直立，同时高效使用关节扭矩。  
一旦身体高度、躯干倾角或任何状态值超出健康范围，回合立即终止。  
主要目标是持续向前移动，而非仅仅保持平衡不倒。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (11,)
- dtype: float（推断）
- obs[0]: torso_height —— 躯干（主体）的垂直高度
- obs[1]: torso_angle —— 躯干的倾斜角
- obs[2]: upper_joint_angle —— 大腿关节角度
- obs[3]: lower_joint_angle —— 小腿关节角度
- obs[4]: foot_joint_angle —— 足部关节角度
- obs[5]: forward_velocity —— 躯干水平速度（前进方向）
- obs[6]: vertical_velocity —— 躯干垂直速度
- obs[7]: torso_angular_velocity —— 躯干角速度
- obs[8]: upper_joint_speed —— 大腿关节角速度
- obs[9]: lower_joint_speed —— 小腿关节角速度
- obs[10]: foot_joint_speed —— 足部关节角速度

## 4. 动作空间 action_space
- type: Box (连续)
- shape: (3,)
- 范围: 每个关节扭矩 ∈ [-1.0, 1.0]
- action 0: upper_joint_torque —— 施加在大腿关节的扭矩
- action 1: lower_joint_torque —— 施加在小腿关节的扭矩
- action 2: foot_joint_torque —— 施加在足部关节的扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功终止；任务没有设定“到达终点”之类的条件，因此不存在成功终止信号。
- failure-like termination:  
  - 身体高度超出健康范围（跌倒或飞起）  
  - 躯干倾角超出健康范围（过度倾斜）  
  - 任意状态值变为非有限值（NaN / inf）
- ambiguous termination: 无
- truncation: 时间步到达上限（truncated）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []（info 字典为空）
- forbidden_or_uncertain_info_fields:  
  - reward_forward  
  - reward_ctrl  
  - reward_survive  
  - x_position  
  - y_position  
  - z_distance_from_origin  
  （任何未声明的 info 字段均禁止使用）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观测，11维）
- action（3维连续动作）
- next_obs（下一时刻观测，11维）
- info 中仅可包含空字典（无可用字段）

禁止使用：
- original_reward（官方奖励被屏蔽）
- official_reward 或任何 masked reward
- info 中任何字段（即使环境内部有，对外也视作不存在）
- training_progress（本 prompt 未允许使用）

## 7. 可用于奖励函数的信号
- position:  
  - torso_height (obs[0])  
  - torso_angle (obs[1])  
  - 各关节角度 (obs[2]~obs[4])
- velocity:  
  - forward_velocity (obs[5]) —— 可直接用于鼓励前进  
  - vertical_velocity (obs[6])  
  - torso_angular_velocity (obs[7])  
  - 各关节角速度 (obs[8]~obs[10])
- orientation: torso_angle, 各关节角度
- contact: 无直接观测
- action/engine: 动作扭矩本身 (action[0]~action[2])，可度量控制代价

## 8. 不确定或不可用的信号
- 世界坐标位置（x, y, z）被排除在观测之外，不可用
- 接触或足底受力信息未提供
- 任何原本可能出现在 info 中的奖励分量（如前进奖励、控制惩罚、生存奖励）均不可用
- 没有成功标志，没有直接指示“向前移动了多少米”的绝对位移信号（仅有瞬时 forward_velocity）



# expert_reward_context.md

# 专家奖励知识上下文（RAG 压缩版）

这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。
以下骨架是任务相关的设计原语、风险提示和参考起点，不构成封闭候选集合。可直接采用、组合、变形，或基于环境事实提出新的数学结构。

## 1. 任务路由摘要
- locomotion_continuous_control：任务目标是稳定前进通过地形。重点观察 fast_then_fail / hover_or_stand_still / over_conservative_policy。

## 2. 相关奖励骨架摘要（按任务路由检索）

以下骨架由任务路由检索推荐。是否使用某个骨架取决于：
1. 该骨架所需信号是否在环境接口中实际可用；
2. 是否与该任务阶段匹配（v1 优先设计核心学习信号，效率/安全类后续迭代加入）。

### forward_progress_reward
- 角色: 前进方向引导
- 数学形态: lambda_fwd * forward_velocity
- 需要信号: forward velocity component
- 使用说明: 奖励沿前进方向的速度。适合 locomotion 类任务。
- 风险: 快速前进但容易摔倒。
- 后续迭代: 若 fast_then_fail，配合稳定性约束。

### terminal_failure_penalty
- 角色: 失败惩罚
- 数学形态: -R_failure * I[failure]
- 需要信号: 显式 failure flag 或 termination_reason
- 使用说明: 当环境提供显式 failure flag 时可用。若 explicit_failure_flag_available=false，不可使用。
- 风险: 误判终止原因。
- 后续迭代: 当能区分失败终止后再加。

### energy_penalty
- 角色: 动作/能耗约束
- 数学形态: -lambda_action * engine_use(action)
- 需要信号: action
- 使用说明: 惩罚动作幅度/能耗。先完成任务再加入，v1 太早加入可能不敢动。
- 风险: agent_afraid_to_move。
- 后续迭代: 能完成任务并稳定后再优化能耗。

### alive_bonus
- 角色: 存活激励
- 数学形态: lambda_alive * I[not_done]
- 需要信号: done flag
- 使用说明: 每步给予小额存活奖励，鼓励 agent 不提前终止。适合 locomotion/balance 类任务。
- 风险: hover_or_stand_still（原地不动来获取存活奖励）。
- 后续迭代: 若 agent 不动，减小权重或配合前向奖励。

### action_smoothness_penalty
- 角色: 动作平滑约束
- 数学形态: -lambda_smooth * |action - previous_action|
- 需要信号: previous action or action history
- 使用说明: 惩罚动作的剧烈变化。适合连续控制任务。
- 风险: 离散动作空间不可用（无历史信息）。
- 后续迭代: 若动作抖动，增大权重。

### stability_penalty
- 角色: 轻量稳定约束
- 数学形态: -lambda_v*|velocity| - lambda_a*|angle| - lambda_w*|angular_velocity|
- 需要信号: next_obs[2], next_obs[3], next_obs[4], next_obs[5]
- 使用说明: 抑制高速、大角度或高角速度。适合需要稳定运动或姿态控制的任务。
- 风险: 过强会保守或不敢动。
- 后续迭代: 若高速失稳，增大权重。

## 3. reward_v1 生成要求
- 直接生成 reward_v1.py，不再生成 reward_design_plan.json。
- 使用 role-based component budget：每个组件必须有明确角色，不能为了显得完整而堆叠。
- 将上述骨架作为思考面而非允许列表；最终设计由任务目标、可用信号和预期策略行为决定，不要求组件对应已有 skeleton_id。
- 如果 success/failure 显式信号不存在，不要使用 terminal_success_reward / terminal_failure_penalty。
- 效率类骨架（energy_penalty、time_penalty）和复杂门控（gated_reward）默认后续迭代再加入。
- 每个组件的设计要考虑可利用风险：agent 可能找到哪些捷径？条件信号是否容易被 exploit？
- 返回格式建议为 return float(total_reward), components；components 必须是 dict。



# Fresh Restart Evidence

- target_score: 3800.000
- best_score_so_far: 2663.090

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| forward_stability_reward + stability_penalty | 2 | 2663.090 | 2663.090 | unsolved |
| forward_stability_reward + stability_penalty + vertical_activity | 1 | 1338.040 | 1338.040 | unsolved |
| forward_stability_reward + stability_penalty + vertical_pushoff | 1 | 380.750 | 380.750 | unsolved |

## Previous interventions

- iter 3 (score=1338.040, structure=forward_stability_reward + stability_penalty + vertical_activity): `selected_level`：Level 2（必要信号缺失）。跳跃任务的垂直振荡动态完全没有被奖励函数表达，仅靠 forward_vel * upright_factor 无法区分跳跃步态与贴地滑行，需要在数学结构上增加一个垂直活动组件来弥补信号缺口。 | `selected_intervention`：新增 `vertical_activity` 组件——对垂直速度的绝对值进行奖励，并用同一 upright_factor 门控，确保只在直立姿态下鼓励跳跃的升降动态。系数设为 0.2，使其成为辅助信号而不会压倒主前进目标。
- iter 4 (score=380.750, structure=forward_stability_reward + stability_penalty + vertical_pushoff): 4. `selected_level`：Level 2 —— `abs(vertical_vel)` 的数学形态直接导致 proxy 与外部任务错位（奖励坠落破坏稳定），证据明确否定该形态，不是单纯尺度问题。 | 5. `selected_intervention`：将 `abs(vertical_vel)` 替换为 `max(0, vertical_vel)`，仅奖励上升阶段（主动推离），不再奖励下降阶段（被动坠落）；系数从 0.2 降至 0.15 以匹配新值域（信号范围减半，保守起步）。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
