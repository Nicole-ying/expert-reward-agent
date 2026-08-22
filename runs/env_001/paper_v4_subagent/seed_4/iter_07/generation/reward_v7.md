# 设计理由
## 审计与缺口分析
**终止模式**：20/20 episodes 均为 truncated（超时 1000 步），没有任何 episode 因坠毁、出界或静置终止。Agent 存活了整段 episode 但没有完成着陆任务。  
**观测使用扫描**：所有 8 个观测维度均被使用，未发现遗漏维度。  
**信号缺口**：`landing_bonus` 组件 active_rate = 0%，其严格的门控条件（`nl_contact * nr_contact == 1` 且双速度均 < 0.2）从未被满足，导致 agent 在整个训练过程中完全没有收到“完成着陆”的正向激励。`progress` 虽然持续提供接近信号，但幅度极小（episode_sum_mean / len ≈ 0.0037/step），无法驱动 agent 完成最后的关键阶段——下降到垫面并触发支撑脚接触。  
**僵尸组件**：`landing_bonus` active 0%，必须废弃。

## 行为诊断
Agent 处于“存活徘徊”模式：通过微小位移积累 `progress`，但因缺少终止阶段的密集正激励，始终悬浮在半空，既未触地也未出界。历史 iter 2（score -6, len 455）证明 agent 有能力接近成功，但后续骨架的“稀疏接触奖励 + 惩罚主导”结构破坏了这条学习路径。

## 干预层级：Level 2 — 结构变换（单组件替换）
**变换类型**：将 `landing_bonus` 从**稀疏二值乘积**替换为**连续吸引子 + 接触放大**的 dense reward。  
**数学形式**：
- proximity = \( e^{-dist / 0.8} \)（指数衰减，距离越近奖励越高）
- speed_factor = max(0, 1 – (|vx|+|vy|)/1.0)（低速奖励）
- angle_factor = max(0, 1 – |angle|/0.5)（角度奖励）
- contact_boost = 1 + 2×(left_contact or right_contact)（触地时大幅放大）
- landing_attractor = proximity × speed_factor × angle_factor × contact_boost  
- 权重 w_attractor = 1.0，确保平均每步贡献与 progress 可比（设计校准微微突破，但因 progress 停滞属必要刺激）。

**为什么应改善**：
- 消灭僵尸组件，把着陆信号从 0% active 变为全程连续，弥补 agent“不知道该停下”的缺口。
- 接触放大因子（1→3）在最后触地瞬间给予强梯度，促使 agent 下降而非悬浮。
- 指数距离因子比线性 progress 更直接打击“在远处刷 progress”的滞留行为。
- 保留 stability 惩罚作为姿态辅助约束，不影响主要目标。

**风险**：attractor 在原点附近峰值约 1.0（接触时 ×3 ≈ 3.0），可能诱使 agent 在刚触地即结束前反复微调（轻微 exploit），需要后续监控 score 是否卡在触点附近。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations (all are scalar float)
    x, y, vx, vy, angle, angvel, l_contact, r_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nl_contact, nr_contact = next_obs

    # 1. Main progress signal: distance reduction to target pad (improvement_delta)
    dist_obs = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress = dist_obs - dist_next  # positive when getting closer

    # 2. Stability constraints (quadratic penalties on next state)
    angle_penalty = -0.1 * (nangle ** 2)
    angvel_penalty = -0.05 * (nangvel ** 2)

    # 3. Soft landing attractor (replaces the dead landing_bonus)
    #    Dense signal that grows as agent approaches, slows, levels, and contacts.
    dist_norm = (nx**2 + ny**2) ** 0.5
    proximity = 2.718281828 ** (-dist_norm / 0.8)         # [0,1], peak at origin

    speed_sum = abs(nvx) + abs(nvy)
    speed_factor = max(0.0, 1.0 - speed_sum / 1.0)       # [0,1], 1 when fully stopped

    angle_factor = max(0.0, 1.0 - abs(nangle) / 0.5)     # [0,1], 1 when level

    contact = float(nl_contact or nr_contact)            # 0 or 1
    contact_boost = 1.0 + 2.0 * contact                  # ×1 without contact, ×3 with

    landing_attractor = proximity * speed_factor * angle_factor * contact_boost

    w_progress = 10.0
    w_attractor = 1.0

    total = (w_progress * progress +
             angle_penalty + angvel_penalty +
             w_attractor * landing_attractor)

    components = {
        "progress": w_progress * progress,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "landing_bonus": w_attractor * landing_attractor
    }

    return float(total), components
```

# 诊断摘要
- **audit**: landing_bonus 门控过死导致全程无着陆正向信号，agent 超时徘徊；观测无缺失，但着陆奖励彻底失效。
- **behavior**: 存活悬浮、靠微小位移刷 progress，既不触碰垫面也不触发任何终止条件。
- **signal**: 缺少从“接近”到“接触”的连续梯度奖励，现有 progress 不足以完成最后 10% 距离。
- **level**: Level 2
- **hypothesis**: 替换为指数吸引子 + 接触放大后，agent 将首次获得“靠近即得利、触地奖更丰”的密集向导，预期 episode 长度下降且 termination 出现。
- **risk**: 触地瞬间奖励 spike 可能导致 agent 在接触后微调振荡；若出现可后续降低 contact_boost 或增加微小触地后惩罚。