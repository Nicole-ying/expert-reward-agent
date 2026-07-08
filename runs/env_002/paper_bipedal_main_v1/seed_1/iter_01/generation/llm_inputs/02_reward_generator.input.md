# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
控制一个双足机器人在不平坦的二维地形上行走。主要目标是让机器人尽可能远、尽可能快地向前移动，同时尽量降低能量消耗。机器人需要通过协调两条腿的髋关节和膝关节来产生稳定的步态。如果机器人身体倒地则立即终止，若能到达地形尽头也会终止。任务鼓励在到达终点前走得更远、更快，并保持低能耗与身体直立。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box
- shape: (24,)
- dtype: float32
- obs[0]: hull_angle —— 身体相对于直立方向的角度 (弧度)
- obs[1]: hull_angular_velocity —— 身体的角速度
- obs[2]: horizontal_velocity —— 向前/向后线速度
- obs[3]: vertical_velocity —— 向上/向下线速度
- obs[4]: hip1_angle —— 腿1髋关节角度
- obs[5]: hip1_speed —— 腿1髋关节角速度
- obs[6]: knee1_angle —— 腿1膝关节角度
- obs[7]: knee1_speed —— 腿1膝关节角速度
- obs[8]: leg1_contact —— 腿1接触地面标志 (1.0=接触, 0.0=未接触)
- obs[9]: hip2_angle —— 腿2髋关节角度
- obs[10]: hip2_speed —— 腿2髋关节角速度
- obs[11]: knee2_angle —— 腿2膝关节角度
- obs[12]: knee2_speed —— 腿2膝关节角速度
- obs[13]: leg2_contact —— 腿2接触地面标志 (1.0=接触, 0.0=未接触)
- obs[14] ~ obs[23]: lidar_0 ~ lidar_9 —— 10个前方地形激光雷达测距值，表示到地形表面的距离或障碍物距离

## 4. 动作空间 action_space
- type: Box (连续)
- shape: (4,)
- bounds: 每个分量均在 [-1.0, 1.0]
- action 0: hip_torque_leg1 —— 施加到腿1髋关节的扭矩
- action 1: knee_torque_leg1 —— 施加到腿1膝关节的扭矩
- action 2: hip_torque_leg2 —— 施加到腿2髋关节的扭矩
- action 3: knee_torque_leg2 —— 施加到腿2膝关节的扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（到达地形尽头，episode 正常结束，可视为成功）
- failure-like termination: body_fallen_over（身体倒地，走不下去）
- ambiguous termination: 无
- truncation: 无（本任务没有设置时间步上限的截断，终止只来自上述两个条件）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {} （info 为空字典，没有任何可用字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不提供，禁止依赖 info 做奖励判断

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs （当前观察）
- action （当前动作）
- next_obs （下一观察）
- info 中明确允许的字段：无（info 为空，不可用任何字段）
- training_progress：本任务 prompt 未明确允许使用，应避免依赖

禁止使用：
- original_reward （被屏蔽的官方奖励）
- official_reward 或任何以“official”命名的变量
- 未声明的 info 字段
- 未声明的 obs 切片或超出上述定义范围的维度

## 7. 可用于奖励函数的信号
- position / progress: 可通过 horizontal_velocity 累加推断水平移动距离（next_obs 与 obs 的速度积分），但没有直接的 x 坐标。
- velocity: horizontal_velocity（前进速度），vertical_velocity（用于惩罚不必要的跳起）。
- orientation: hull_angle（身体倾斜，倾向于惩罚偏离0），hull_angular_velocity（惩罚快速旋转）。
- contact: leg1_contact, leg2_contact（可用于步态对称性、防止同时离地等）。
- action / engine: 动作扭矩的平方和或绝对值可作为能耗惩罚。
- terrain sensing: lidar_0..9（可用来奖励对前方崎岖地形的预适应，但通常谨慎使用）。

## 8. 不确定或不可用的信号
- 没有显式的成功/失败标志（info 为空），不能直接用这些做奖励塑造。
- 没有直接的位置坐标（x, y），前进距离只能通过速度积分获取，可能受到积分误差影响。
- 没有地形的全局信息，只有局部雷达读数。
- 禁止使用 original_reward，因此不能通过监控其数值来获取基线。



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



# Validation repair

具体错误：Reward v1 failed validation: 函数体中使用了 original_reward (record: runs\env_002\paper_bipedal_main_v1\seed_1\iter_01\generation\validations\reward_v1.validation.json)

只修复代码合规问题，不重新分析环境，不改变原定奖励设计。直接输出完整合规的compute_reward函数。

# Invalid previous draft

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Computes the reward for the bipedal locomotion task.

    Args:
        obs: Current observation (24,).
        action: Current action (4,) in [-1, 1].
        next_obs: Next observation (24,).
        original_reward: Placeholder (not used).
        info: Empty dict (not used).
        training_progress: Placeholder (not used).

    Returns:
        total_reward: float, the total reward.
        components: dict, individual reward components.
    """
    # Extract relevant next observations
    hull_angle = next_obs[0]           # body tilt angle
    hull_angular_velocity = next_obs[1]  # angular velocity
    horizontal_velocity = next_obs[2]   # forward speed (positive forward)
    vertical_velocity = next_obs[3]     # vertical speed (up positive)

    # Weights (adjustable hyperparameters)
    w_forward = 1.0    # reward for forward progress
    w_angle = 1.0      # penalty for body tilt
    w_angvel = 0.2     # penalty for fast rotation
    w_vert = 1.0       # penalty for vertical motion (jumping)

    # Main learning signal: forward velocity
    forward_reward = w_forward * horizontal_velocity

    # Stability penalty: keep the torso upright and motion smooth, discourage unnecessary jumping
    stability_penalty = (
        -w_angle * abs(hull_angle) 
        - w_angvel * abs(hull_angular_velocity) 
        - w_vert * (vertical_velocity ** 2)
    )

    total_reward = forward_reward + stability_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 使用的奖励组件及其角色

- **forward_reward**（主学习信号）：直接奖励沿前进方向的速度（`next_obs[2]`）。给予每步稠密的梯度，引导机器人获得向前移动的策略。若速度为负（后退），该分量自然为负，抑制倒退行为。
- **stability_penalty**（稳定/安全约束）：惩罚身体倾斜角（`next_obs[0]`）、高角速度（`next_obs[1]`）和不必要的垂直速度（`next_obs[3]`）。其作用是让机器人学习在前进的同时保持躯干直立、动作平滑，降低摔倒风险（fast‑then‑fail）。该组件将多个稳定相关信号合并为一个，避免过度拆分。

V1 不引入额外的存活奖励、能耗惩罚或步态约束，因为核心目标只需要“稳定地前进”，更复杂的优化留给后续迭代。

## 2. 未使用 terminal_success_reward / terminal_failure_penalty 的原因

- `explicit_success_flag_available` 和 `explicit_failure_flag_available` 均为 false，环境没有提供任何可用的成功或失败标志（`info` 为空字典）。
- 即使知道“到达地形尽头”或“身体倒地”会终止 episode，我们也无法在奖励函数中可靠区分这两种终止原因，强行引入终止后奖励/惩罚会引入错误的梯度或无法正确解释终止信号。
- 因此 V1 完全依赖稠密的运动信号，通过稳定性惩罚间接抑制导致倒地的行为，不依赖稀疏的终止事件。

## 3. 未纳入 V1 但后续迭代可考虑的部分

- **energy_penalty**：惩罚动作扭矩的大小，降低能耗。V1 阶段优先学会稳定行走，过早加入能耗约束可能导致 agent 不敢动。
- **action_smoothness_penalty**：惩罚动作的突变，需要访问历史动作。当前接口未提供上一步动作，需在后续阶段由外部 wrapper 提供或使用差分 `action` 与 `obs` 的变化，复杂度较高，留待 V2/ V3。
- **步态对称性 / contact heuristic**：利用 `leg1_contact`、`leg2_contact` 鼓励交替支撑，例如惩罚双脚同时离地。此信号可作为 V2 的额外稳定/效率信号。
- **地形预适应（lidar）**：通过前方雷达读数引导避障或抬腿，V1 不引入以免过早复杂化。
- **gated / dynamic curriculum**：多阶段或自适应权重，明显超出 V1 范围。

## 4. 预期训练后需要观察的 failure mode

- **fast‑then‑fail**：如果 `forward_reward` 的权重相对稳定性惩罚过大，机器人可能倾向于快速前冲然后摔倒。此时应增大 `w_angle` 或 `w_vert`，或调整稳定性惩罚的数学形态（如使用乘积形式将速度奖励与直立因子耦合）。
- **hover / stand_still**：如果稳定性惩罚过强（尤其是对角速度的惩罚），机器人可能选择原地站立不动以获取零惩罚但几乎没有正向奖励。此时应适当减小 `w_angvel`，或确保 `forward_reward` 足够有吸引力。
- **过度跳跃**：若 `w_vert` 太小，机器人可能通过小幅跳跃获得额外前向速度，导致能耗高且步态不自然。可动态调整 `w_vert`。
- **后退逃逸**：由于 `forward_reward` 允许负值，策略暂时后退也能获得负奖励，但整体应导向前进。若出现长期后退，可能需要改为 max(0, velocity) 的形式，但 V1 先保持线性以便梯度更平滑。

所有权重目前为初始建议值，实际使用需根据环境的物理量级和训练曲线进行调整。