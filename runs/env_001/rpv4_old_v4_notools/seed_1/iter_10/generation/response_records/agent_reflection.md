# Response Record

# 诊断报告

## evidence
- 第9轮得分 -56.0，范围 [-85.8, -17.9]，所有评估集均被截断（truncated 20/20，len=1000），从未触发成功终止。
- 训练过程中分数从 -758 提升到 -50 附近后停滞，说明某些阶段的学习信号已饱和或缺失。
- `goal_progress` 组件 magnitude_share 高达 94.9%，主导了奖励；`attitude_penalty` 占 5.1%；`soft_velocity_penalty` 的 `ep_sum_mean` 与 `active_rate` 在全部检查点中始终保持 0——属于完全死组件。
- `soft_velocity_penalty` 的门控条件为 `y < 0.5`（高度），agent 可能从未进入该高度区域，或者进入后速度已很小，加上系数过低导致影响可忽略。
- 当前奖励缺少着陆末段的明确引导（减速、触地），导致 agent 学会靠近目标但不下落，徘徊直至步数耗尽。
- 检查 `read_environment_card()`（obs: [x, y, vx, vy, angle, angular_vel, left_leg_contact, right_leg_contact]），确认存在腿接触信号，此前未使用。
- 查阅 `read_past_reflection(8)` 等历史，无与本轮冲突的重复语义。

## behavior_diagnosis
agent 学到了“向目标移动并保持姿态”的策略，但缺乏着陆末段引导，因此在距离缩减到一定程度后不再进一步减速/下降，最终因步数耗尽被截断。此行为是不健康的，因为车辆无法真正完成任务（安全着陆）。

## signal_completeness
必要职责中存在缺口：
- **减速着陆信号**：原有的速度惩罚因门控失效完全死掉，等效缺失。
- **接触/触地信号**：环境包含腿接触状态，但从未被奖励，缺少鼓励车轮（腿）着地的正反馈。
- **姿态稳定**：现有，但量级太小。

## selected_level
**L2**——修改速度惩罚的触发逻辑（改为距离门控），并新增腿接触奖励，两项服务于同一语义“引导末段着陆”。

## selected_intervention
- 将 `soft_velocity_penalty` 改为 **距离门控**：当与目标的距离小于阈值（0.5）时，施加速度平方惩罚，迫使 agent 接近时减速。
- 新增 `leg_contact` 组件：对每个接触的支撑腿给予小的正奖励（0.2/腿），提供触地正反馈。
- 调整权重：速度惩罚系数增加到 0.5，接触奖励系数 0.2，保持原有 progress 和 attitude 不变，但总着陆相关信号量级提升。

## falsifiable_hypothesis
引入距离门控的速度惩罚和腿接触奖励后，agent 在接近目标时将被迫减速，同时被鼓励让腿触地，从而减少徘徊提前着陆，提高成功率并改善分数。

## expected_next_round
预计下一轮 agent 会在距离目标较近时大幅减速，并尝试让腿接触垫面，可能出现成功着陆的 episode，分数明显上升（预期至少提高至 0 以上），episode 长度可能缩短，terminated 比例增加。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2 reward: progressive distance reduction + attitude stabilisation +
    distance‑gated velocity penalty + leg‑contact bonus.
    """
    # coefficients
    w_progress = 1.0
    w_attitude = 0.05
    w_velocity = 0.5     # increased to be effective
    w_contact = 0.2      # per leg contact

    # ---------- goal progress (potential‑based delta distance) ----------
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    goal_progress = w_progress * (dist_current - dist_next)

    # ---------- upright stabilisation ----------
    attitude_penalty = -w_attitude * (next_obs[4] ** 2 + 0.1 * next_obs[5] ** 2)

    # ---------- distance‑gated velocity penalty ----------
    # only active near the landing pad to encourage gentle final approach
    vel_penalty = 0.0
    if dist_next < 0.5:
        speed_sq = next_obs[2] ** 2 + next_obs[3] ** 2
        vel_penalty = -w_velocity * speed_sq

    # ---------- leg contact bonus (touchdown feedback) ----------
    # obs[6] and obs[7] are left/right leg contact indicators (0/1)
    leg_contact = 0.0
    if next_obs[6] > 0.5:
        leg_contact += w_contact
    if next_obs[7] > 0.5:
        leg_contact += w_contact

    total_reward = goal_progress + attitude_penalty + vel_penalty + leg_contact

    components = {
        "goal_progress": goal_progress,
        "attitude_penalty": attitude_penalty,
        "velocity_penalty": vel_penalty,
        "leg_contact": leg_contact
    }
    return float(total_reward), components
```
