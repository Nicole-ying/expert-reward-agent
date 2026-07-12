# 设计理由
## 为什么以前都失败了
- **iter 2**：乘性门控 `forward * gait_quality` 过于严苛，探索初期步态质量差就会直接把主奖励砍到零，长度从 1600 暴跌至 194，agent 还没学会平衡就被饿死。
- **iter 3–4**：退回加性构造后，`gait_bonus`（0.3 × gait_quality，全时发放）迅速成为压倒性收入源（episode_sum ≈ 360），而 `forward_progress` 只有 0.12，agent 退化为“原地交替踏步但绝对不走”的安全 exploit——姿态完美、不摔倒、不前进，最终 score 锁死在 -74～-95 区间。

根本原因是：
1. **主信号（前进）太弱**：`2.0 * vel²` 在速度极小时梯度 ≈ 0，agent 走一步拿不到正向反馈，自然选择更丰厚的 gait_bonus。
2. **副信号（步态）独立于前进发生**：可以在速度 = 0 时大量获得，形成局部最优。
3. **连续迭代只系数的修补不再有效**：iter 3、4 两次改动都未能打破这个 exploit 循环。

## 新骨架选了什么算子、和已尝试过的有什么本质不同
- **主信号重构**：采用 **dense_state_signal（线性 + 凸化）** 组合，`5.0 * vel + 2.0 * vel²`，使极低速度时也有线性梯度（~5/unit），同时高速时凸化激励，打破“没有梯度就原地踏步”的死锁。
- **步态信号重新绑定前进**：放弃独立的 `gait_bonus`，改用 **依赖速度的 gait_efficiency**：`0.5 * gait_quality * vel`。没有速度就没有步态奖励，彻底消除原地踏步 exploit。这不是 iter 2 的乘性门控——它不会遮断主奖励，只是在主奖励之上追加一个受步态调节的增量。
- **增加动作惩罚**：轻量的 **quadratic_penalty**（-0.01 Σa²），引导节能步态。其 per-step 负担 < 主信号 1%，不会阻碍探索，但会迫使 agent 用更少扭矩前进。
- **保持宽松平衡约束**：继续用 hinge 版兜底，只在大角度时介入，不惩罚正常前倾探索。
- **整体架构**：`前进(线性+凸化) + 步态效率奖励(绑定前进) + 平衡铰链惩罚 + 双脚离地惩罚 + 动作惩罚`。这本质上是一个**单一强目标（前进）驱动、多辅助约束**的骨架，与 iter 1–4 的“多个弱奖励互相竞争”模式根本不同。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取信号
    horizontal_vel = obs[2]
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    leg1_contact = obs[8]
    leg2_contact = obs[13]

    # --- 1. 主奖励：前进速度，线性+凸化 ---
    forward = max(0.0, horizontal_vel)
    forward_reward = 5.0 * forward + 2.0 * (forward ** 2)

    # --- 2. 步态效率：与前进速度绑定的交替奖励 ---
    gait_quality = 1.0 - abs(leg1_contact - leg2_contact)
    gait_efficiency = 0.5 * gait_quality * forward

    # --- 3. 平衡惩罚：铰链形式，允许 0.3 rad 自由 ---
    angle_dev = abs(hull_angle) - 0.3
    angle_penalty = -0.2 * (max(0.0, angle_dev) ** 2)
    angular_vel_penalty = -0.05 * (hull_angular_vel ** 2)
    balance_penalty = angle_penalty + angular_vel_penalty

    # --- 4. 双脚离地惩罚 ---
    contact_sum = leg1_contact + leg2_contact
    if contact_sum < 0.5:
        both_off_penalty = -0.1 * (1.0 - contact_sum)
    else:
        both_off_penalty = 0.0

    # --- 5. 动作能耗惩罚 ---
    action_penalty = -0.01 * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    total_reward = forward_reward + gait_efficiency + balance_penalty + both_off_penalty + action_penalty

    components = {
        "forward_reward": forward_reward,
        "gait_efficiency": gait_efficiency,
        "balance_penalty": balance_penalty,
        "both_off_penalty": both_off_penalty,
        "action_penalty": action_penalty,
        "gait_quality": gait_quality
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 终止条件是摔倒，有平衡惩罚兜底；目标是前进，新主信号直接给梯度；纯 gait_bonus 已成为 exploit 源头，本骨架将其删除并绑定到前进。
- **behavior**: 之前 agent 停在“完美交替、姿态良好但velocity≈0”的安全局部最优，靠 gait_bonus 活着，不走也不死。
- **signal**: 缺“走起来”的强梯度；gait 信号脱耦于前进。
- **level**: Level 3
- **hypothesis**: 大幅强化前进速度信号的线性梯度 + 将步态奖励绑定速度，会迫使 agent 必须真的前进才能获得正收益，从而打破原地踏步的 exploit；同时轻量动作惩罚推动节能步态。
- **risk**: 线性前进奖励可能导致 agent 满足于极慢蠕动（vel≈0.05 时 per‑step ≈ 0.25，仍远大于惩罚），需在后续迭代中观察是否需要进一步凸化或速度门槛。