# 设计理由

## 第 0 步：信号覆盖审计

**0.1 终止模式分析**：20/20 episode 均为 truncated（超时 1000 步），无 terminated。agent 存活了整整 1000 步但从未触发 success termination（`body_not_awake_or_settled`），也从未触发 failure termination。这说明 agent 学会了**生存**，但没学会**着陆**。

**0.2 观测使用扫描**：
- 已使用：`obs[0]` x, `obs[1]` y, `obs[4]` body_angle
- 未使用：`obs[2]` x_velocity, `obs[3]` y_velocity, `obs[5]` angular_velocity, `obs[6]` left_support_contact, `obs[7]` right_support_contact

关键的信号缺口在 `obs[6]` 和 `obs[7]`——支撑腿接触标志。当前奖励函数对"靠近着陆点"和"真正着陆"完全不加区分。landing_incentive 占奖励的 99.3%，但它只惩罚距离，不奖励接触。agent 的最优策略可能是：飞到靠近原点但悬空的位置，然后以最小代价维持在那里刷 landing_incentive。

**0.3 信号缺口判断**：**信号缺失**。未使用的 `obs[6]/obs[7]` 正好是区分"靠近目标垫"和"真正着陆"的唯一信号。没有这个信号，agent 没有动力完成最后一步着陆动作。

**0.4 僵尸组件检查**：`angle_penalty` active_rate = 0.3%——几乎不触发。在当前形式下无实质作用。

## 1. 行为诊断

agent 在 exploit landing_incentive：它学会了在低距离处悬停/盘旋，最大化 `1/(1+10d)` 收益，但从不完成着陆（无 terminated success）。进度信号 `progress_reward` 均值仅 1.4，说明它到达某个最优悬停位置后就停止前进了——或者在该位置附近 oscillate 以维持最小距离。

## 2. 干预层级

**Level 2 — 结构变换**：landing_incentive 是一个全局势场，它创造了 exploit 空间（低空悬停）。需要把它转变为**局部门控**：只有在真正接触着陆点、或至少在着陆区域时才能获得显著奖励。根据 Operator Library，"proxy 提高但外部分数不升 → proxy 对齐任务完成"。

**具体变换**：引入支撑腿接触信号 `(obs[6] + obs[7])` 作为 soft gate 乘到 landing incentive 上，或用接触信号作为二值开关。由于接触是二值的（0 或 1），直接乘法会塌缩为 0。因此使用 **soft gate**：`gate = 0.1 + 0.9 * leg_contact`，确保在无接触时仍有微弱势场引导（防止完全丧失信号），但有接触时奖励放大 10 倍。

同时，收紧角度惩罚使其真正起作用：将 threshold 从 0.5 降至 0.3 rad（约 17°），并对超过部分使用 hinge。

## 设计校准

- 主信号 per-step ≈ 0.23（landing_incentive 总和/1000 步）
- 新 gate 结构在有接触时放大奖励至 ~0.5/step（≤ 2x 主信号）
- 角度惩罚 per-step ≤ 0.02（为主信号的 0.09x，安全）
- 无腿接触时，landing_incentive 降至 0.05/step，仍然非零但不足以支撑 exploit

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next observation
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ------------------  Main progress signal: distance reduction  ------------------
    dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    w_progress = 1.0
    progress = (dist - next_dist)

    # -----------  Landing incentive with soft contact gate  -----------
    # Gate: 0.1 when no leg contact, 1.0 when at least one leg in contact
    # This makes landing ~10x more rewarding than hovering nearby
    leg_contact = 1.0 if (left_contact > 0.5 or right_contact > 0.5) else 0.0
    contact_gate = 0.1 + 0.9 * leg_contact

    # Continuous proximity bonus, gated by actual contact
    w_landing = 0.5
    landing_incentive = contact_gate * w_landing / (1.0 + next_dist * 5.0)

    # -------------------  Health constraint: body angle (tightened)  -------------------
    # Tightened safe_angle from 0.5 -> 0.3 rad (~17 degrees)
    # Makes the penalty actually engage before extreme angles
    w_angle = 0.5
    safe_angle = 0.3
    angle_error = abs(next_angle) - safe_angle
    angle_penalty = -w_angle * angle_error if angle_error > 0 else 0.0

    # -------------------  Total reward  -------------------
    total_reward = w_progress * progress + landing_incentive + angle_penalty

    components = {
        "progress_reward": w_progress * progress,
        "landing_incentive": landing_incentive,
        "angle_penalty": angle_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号缺失——支撑腿接触标志 obs[6]/obs[7] 未被使用，这是区分"靠近着陆点"和"真正着陆"的唯一信号
- **behavior**: agent exploit landing_incentive，低空悬停刷分但不完成着陆（0 terminated success, 20/20 truncated）
- **signal**: 缺对着陆完成的激励；landing_incentive 无条件发放使悬停成为最优策略；angle_penalty 几乎不触发（0.3% active_rate）
- **level**: Level 2
- **hypothesis**: 引入接触门控后，悬停收益降至 1/10，agent 有动力降低到足以触发支撑腿接触的高度并完成着陆。收紧角度惩罚使其在接近着陆姿态时提供有意义的引导
- **risk**: 门控使奖励在无接触时变稀疏，可能短期增加探索难度；但 progress_reward 仍提供向心引导，且 contact_gate 保留 0.1 基线防止完全零信号