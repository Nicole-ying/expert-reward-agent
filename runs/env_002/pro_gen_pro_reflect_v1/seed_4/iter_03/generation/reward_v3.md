# 设计理由

## 为什么以前都失败了 / 为什么需要重建
虽然 iter‑2 已 `target_solved` (score=316.49)，但 reward 结构高度依赖单一信号：用 `stability_gate` 连续乘在 `forward_reward` 上，形成一个"加权和"骨架。

该骨架的风险在于
- **高度线性：** `forward_reward = max(0, horiz_vel)` 是纯正奖励。agent 可能倾向于"在姿态恶化前尽可能快跑"，即使速度较高但触地模式粗放。
- **缺乏步态节律信号：** 环境包含 `leg1_contact` / `leg2_contact`，但从未被使用。双腿机器人真正的核心学习目标是**形成稳定的接触-蹬地循环**，而不是纯粹的 forward speed maximization。
- **能量效率缺失：** 动作维度有 4 个连续力矩，却没有任何能耗惩罚。agent 可能依赖大扭矩暴力驱动以维持速度，长期稳定性存疑。

因此，虽然当前 score 达标，但其实现路径可能偏向 exploit（暴力前进+硬抗姿态），而非学习到稳健步态。重建要求我**避开"以 horiz_vel 为主骨架"的路径**，这正是前两轮迭代的共同特征。

## 新骨架选择：步态循环奖励（Gait Cycle Reward）
新骨架的**核心主信号**不再是 `horiz_vel`，而是**交替触地事件**。这利用了 `joint_condition_proxy` 思想：将步态拆解为连续的接触状态变化，每完成"一条腿从摆起到着地"的循环就给予进展奖励。

这样做和已尝试过的版本有本质不同：
- 过去：依赖 `horiz_vel` 作为唯一主信号，姿态作为软门控乘数。
- 现在：主信号来自**接触事件的周期性计数**，姿态降级为轻量约束，并引入动作能耗惩罚以抑制暴力驱动。

## 数学形式与系数校准

### 1. 主信号：步态循环奖励 (gait_cycle_reward)
- 信号：`leg1_contact` (obs[8]) 和 `leg2_contact` (obs[13])
- 逻辑：维护一个内部状态 `prev_contact_state`，记录上一次的接触组合（`(leg1_contact > 0.5, leg2_contact > 0.5)`）。
- 每当**任意一条腿的接触状态从 `0` 翻转为 `1`**，就认为完成了一段 swing→contact 的过渡，给予固定奖励 `+0.3`。
- 这是 `improvement_delta` 的离散版本：`contact_event = max(0, new_contact - prev_contact)`，每条腿独立触发。

**校准依据：**
- `len` 当前约 892 步，正常步态周期约 20–30 步一个循环，预期每集触发约 30–45 次。
- `episode_sum` 约 30×0.3 = 9 ~ 13.5。
- 主信号 per‑step ≈ (9~13.5)/892 ≈ 0.01~0.015。这是合理的低密度主信号，不会过度主导奖励。
- 符合 `forward reward` 完全移除后，新的正向驱动力；agent 必须学习交替触地以获取回报。

### 2. 姿态 soft hinge：倾角过大概率摔倒的前兆
- `hull_angle = obs[0]`
- 使用 hinge 形式：`-0.2 * max(0, abs(hull_angle) - 0.35)`
- 0.35 rad ≈ 20° 倾斜，是摔倒边界（通常 0.5–0.8 rad）的约 60–70%。
- 只在已经比较倾斜时开始施加轻微线性惩罚，不干扰正常微倾。

### 3. 动作能量惩罚：防止暴力扭矩
- `torque_penalty = -0.0005 * sum(action_i ** 2)`  for i=0..3
- 这是典型的 `quadratic_penalty`。
- 主信号 per‑step ≈ 0.01，所以处罚 per‑step ≈ 0.0005×4×1 ≈ 0.002，是主信号的 0.2x，完全在 0.3x 安全系数内。

### 4. 角速度 light 二次惩罚（保留但削弱）
- `angvel_penalty = -0.0002 * hull_ang_vel ** 2`
- 保持对高频振荡的轻微抑制，系数仅为上一轮的 1/5，避免成为主要主导项。

**总惩罚负担：** 步均 ≤ 0.0025，远小于主信号步均 0.01，符合 0.5x 上限。

现在，agent 必须通过**重复地着地-蹬地**才能累积奖励，不具备纯依赖高速前冲的漏洞。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------- 观测提取 --------------------
    hull_angle     = float(obs[0])
    hull_ang_vel   = float(obs[1])
    leg1_contact   = float(obs[8])
    leg2_contact   = float(obs[13])

    # -------------------- 步态循环奖励（主信号） --------------------
    # 使用闭包内状态记录上一次接触状况，确保每条腿的未接触→接触转换被奖励
    # 由于 compute_reward 在每步调用一次，我们通过函数属性存储状态
    if not hasattr(compute_reward, "prev_contacts"):
        compute_reward.prev_contacts = [False, False]  # [leg1, leg2]
    
    prev_leg1, prev_leg2 = compute_reward.prev_contacts
    curr_leg1 = leg1_contact > 0.5
    curr_leg2 = leg2_contact > 0.5
    
    # 每条腿的未接触→接触事件 (rising edge)
    event1 = 1.0 if (curr_leg1 and not prev_leg1) else 0.0
    event2 = 1.0 if (curr_leg2 and not prev_leg2) else 0.0
    
    gait_cycle_reward = 0.3 * (event1 + event2)

    # 更新状态为下一次调用准备
    compute_reward.prev_contacts = [curr_leg1, curr_leg2]

    # -------------------- 姿态安全边界 (hinge) --------------------
    tilt_penalty = -0.2 * max(0.0, abs(hull_angle) - 0.35)

    # -------------------- 动作能量惩罚 --------------------
    torque_penalty = -0.0005 * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    # -------------------- 角速度轻量化约束 --------------------
    angvel_penalty = -0.0002 * (hull_ang_vel ** 2)

    # -------------------- 总奖励 --------------------
    total_reward = gait_cycle_reward + tilt_penalty + torque_penalty + angvel_penalty

    components = {
        "gait_cycle_reward": gait_cycle_reward,
        "tilt_penalty": tilt_penalty,
        "torque_penalty": torque_penalty,
        "angvel_penalty": angvel_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 上一轮遗漏了接触/步态节律信号，过份依赖 horiz_vel 作为唯一进展，缺乏能量效率考量。
- **behavior**: agent 采取“高速奔跑+姿态门控”策略，虽然成活且达目标，但可能依赖暴力扭矩，非稳健步行。
- **signal**: 缺少步态循环（alternate contact）的正向进度信号；缺少能耗惩罚；vertical_velocity 惩罚实际未激活。
- **level**: Level 3
- **hypothesis**: 引入接触事件奖励将直接引导 agent 形成交替触地的步行周期，而不再仅优化 forward speed；能耗惩罚会抑制过大的关节力矩，促进更经济的步态。
- **risk**: 接触事件奖励密度较低，可能导致初期学习信号稀疏，延长训练时间。若 gait_cycle_reward 未能充分引导，agent 可能提前停滞。