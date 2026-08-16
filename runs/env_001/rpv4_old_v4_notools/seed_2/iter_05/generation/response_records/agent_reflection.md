# Response Record

# 诊断报告

**evidence**  
- 最终评估：score=29.6，episode 长度=1000（全部截断，terminated=0），说明策略从未成功着陆，只是在环境中存活到时间上限。  
- 组件占比：contact_guidance 的 signed_share 和 magnitude_share 都高达 96.6%，而 progress 仅占 1.2% 且 active_rate 100%。contact_guidance 的平均 episode sum 达 66.3，远超其他组件。  
- Checkpoint 趋势：contact_guidance 在训练中后期迅速膨胀（cp10% 0.2 → cp100% 59.6），但最终得分在 cp70% 达到峰值 80.4 后反而下跌到 43.2；progress 全程几乎停滞在 0.6~0.7。  
- 这些共同指向：agent 学会了一种 exploit——通过反复拾起/放下腿触发 `contact_improvement` 来获取大量 contact_guidance 奖励，而不是真正靠近并稳定着陆。该行为使 contact_guidance 成为主导信号，淹没了 progress 的引导，导致策略停在“腿接触奖励”的局部最优，无法完成着陆。

**behavior_diagnosis**  
当前策略是一个“接触奖励收集器”：它可能在平台上方反复微动腿部，保持某种间歇接触以获得高奖励，但实际并未减少与目标的距离或实现稳定停靠。这种行为不健康，因为它完全抛弃了任务目标（到达并稳定着陆），只追求高奖励部件。

**signal_completeness**  
必要职责缺失：缺少一个持续、无法被 exploit 的**全局接近驱动信号**。当前的 progress（增量式距离变化）信号太弱，不足以对抗 exploit。同时，contact_guidance 使用的“增量接触”形态正是 exploit 的根源，需要替换为直接鼓励接触状态但难以被重复触发的形式。此外，速度惩罚没有起到约束剧烈动作的作用，可能需要稍加强化。

**selected_level**  
L2 —— 需要改变 contact 组件和接近组件的数学形态，而不只是调权重。contact_guidance 的乘积门控和增量机制是 exploit 的根源，progress 的弱增量形式也无法提供足够的全局驱动。这两者必须从数学结构上修改。

**selected_intervention**  
- **删除** 原有的 `contact_guidance`（增量 × 接近门控 × 速度门控）。
- **新增** 简洁的 `contact_state` 奖励：直接奖励每条腿的接触状态（0 或 1），提供持续、非重复触发的正反馈。
- **改造** `progress`：同时提供基于负距离的持续全局压力（`-dist_next`）和一个较强增量式 shaping（`dist_prev - dist_next`），确保全程都有明确的“靠近目标”信号，且该信号不会被 contact 奖励淹没。
- **简化** 速度惩罚：去除 `altitude_factor` 等复杂门控，改为单一的全局速度平方惩罚，避免在特定阶段无效。
- **保留** 方向惩罚，系数微调以维持基本姿态约束。

**falsifiable_hypothesis**  
如果我的判断正确，接触 exploit 被消除后，agent 将不再能通过重复触发增量接触获取大量奖励；取而代之的是持续受 `-dist` 驱动向原点移动，并在靠近后因为 `contact_state` 奖励更倾向于放下双腿保持接触。由于 `-dist` 的绝对压力无法通过徘徊规避，agent 必须真正缩小距离，从而更可能过渡到稳定着陆行为。预计下一轮 episode_length 将开始缩短（出现 terminated episode），平均分数有望超过 80。

**expected_next_round**  
策略将从“接触奖励收集徘徊者”转变为“接近-减速-着陆者”：agent 会优先向目标移动（受 `-dist` 驱动），在接近时减速（受速度惩罚约束）并放下双腿（受 contact_state 奖励激励），从而更可能达成着陆条件。最终评估中应出现 terminated episode，score 提升至 >80，progress 组件的占比和绝对值显著上升，contact 组件 magnitude 大幅降低。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- continuous proximity signal (absolute distance penalty + incremental progress)
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # global pressure to reduce distance (cannot be evaded by local oscillation)
    proximity_pressure = -0.4 * dist_next

    # incremental progress – strong shaping towards the target
    progress = 2.0 * (dist_prev - dist_next)

    # --- simple velocity penalty (no altitude gating, applies everywhere)
    vx = next_obs[2]
    vy = next_obs[3]
    velocity_penalty = -0.02 * (vx ** 2 + vy ** 2)

    # --- orientation penalty (unchanged rationale, small tweak)
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.03 * (next_obs[5] ** 2)

    # --- contact state reward (not incremental → no exploit by lifting/placing repeatedly)
    # each leg contact gives a small constant positive reward
    contact_state = 0.3 * (float(next_obs[6]) + float(next_obs[7]))

    # --- combine
    total_reward = proximity_pressure + progress + velocity_penalty + orientation_penalty + contact_state

    components = {
        "proximity_pressure": proximity_pressure,
        "progress": progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_state": contact_state
    }

    return float(total_reward), components
```
