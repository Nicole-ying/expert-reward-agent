# Response Record

# 设计理由

**第一轮修改：将直立姿态约束从独立惩罚改为软门控因子**

当前最大问题是 `upright_orientation_penalty`（平均 -177.88，占 signed_share -40.9%）与 `forward_velocity_reward`（+159.08）量级相当、彼此抗衡，导致总奖励被严重压制甚至为负，agent 无法有效优化前进目标。85% 的 episode 以终止结束，说明机器人虽能产生一定前进速度，但经常失去平衡跌倒。

根据 Formula Switching Guide：「独立约束惩罚 + 高 terminated」的证据模式对应的算子变换是 **soft_health_gate**——把该状态做成门控乘到主奖励上，完全去除独立惩罚。这样直立姿态不再是“犯错就扣分”的罚项，而是“站得直才给前进奖励”的准入条件。agent 为了最大化前进奖励，必须主动维持直立，奖励结构从对抗性变为协同性。

**数学形式与系数校准**：
- `upright_gate = max(0, 1 - 2*(q_x² + q_y²))`（范围 0~1，1=完全直立）
- `forward_reward = 2.0 * forward_velocity * upright_gate`
- 删除原来的 `-1.0 * (1 - upright_gate)²`
- 主信号 per‑step 预估约 0.66（0.73 × 平均 gate ~0.9），总惩罚负担 ≤ 0.23，远低于 0.5× 主信号的上限。
- height、lateral、energy 惩罚暂时保持不变，它们的量级很小。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 主学习信号：前进速度 × 直立门控 ----
    forward_velocity = next_obs[13]

    # 身体直立程度（1=完全竖直，<=0=翻倒）
    quat_x, quat_y = next_obs[2], next_obs[3]
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_gate = max(0.0, body_up_z)  # 倒立时门控为 0，抹掉前进奖励

    forward_reward = 2.0 * forward_velocity * upright_gate

    # ---- 稳定/健康约束：身体高度安全区间 ----
    body_height = next_obs[0]
    lower_safe = 0.3
    upper_safe = 0.9
    height_penalty = (
        -5.0 * max(0.0, lower_safe - body_height) +
        -5.0 * max(0.0, body_height - upper_safe)
    )

    # ---- 辅助约束：侧向漂移抑制 ----
    lateral_velocity = next_obs[14]
    lateral_penalty = -0.5 * lateral_velocity**2

    # ---- 效率约束：动作能量代价 ----
    action_energy = sum(a**2 for a in action)
    energy_penalty = -0.01 * action_energy

    total_reward = forward_reward + height_penalty + lateral_penalty + energy_penalty

    components = {
        "forward_velocity_reward": forward_reward,
        "height_health_penalty": height_penalty,
        "lateral_drift_penalty": lateral_penalty,
        "action_energy_penalty": energy_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 直立约束以独立惩罚形式对抗前进奖励，导致总奖励被负值主导；缺少门控式安全信号。
- **behavior**: agent 能产生一定前进速度但频繁失去平衡，总奖励为负，终止率高达 85%。
- **signal**: upright_penalty 的 magnitude 与 forward_reward 相当，形成“前进越多就越容易倾斜→扣分越大”的错误反馈。
- **level**: Level 2（结构变换：独立惩罚 → 门控因子）
- **hypothesis**: 将直立约束改为软门控后，奖励结构从对抗变为协同——agent 只有保持直立才能收获前进奖励，不再因直立问题而被巨量罚分，前进优化信号得到干净释放。
- **risk**: 门控在轻微倾斜时也会削减前进奖励，可能使学习初期的速度探索变慢；但“站得直才有前进”是任务必要前提，这一代价可接受。
