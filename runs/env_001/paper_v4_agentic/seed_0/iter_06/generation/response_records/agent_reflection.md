# Response Record

# 设计理由
**改动组件**：删除僵尸组件 `landing_contact_bonus`（active_rate=0%），新增 `landing_approach_bonus`。

**为什么**：
- 审计发现 `landing_contact_bonus` 因距离门控过于尖锐，在最终策略中完全无法触发，形式上是"既需要接近又需要接触"的死锁：agent 永远不会在接近时学会平和登陆，因为它的接触经验都来自失败撞击。
- `progress_reward` 是唯一有效的正向信号（100% active），这让 agent 只学会向目标移动（ep len=1000 全程存活），但始终不启动着陆行为，因为缺少对"在目标附近减速、摆正姿态"的正向引导。
- 将接触奖励替换为**不依赖接触条件**的 `landing_approach_bonus`，使用三个 soft 门控因子（距离接近度 × 线速度抑制 × 姿态/角速度抑制）的乘积。agent 只有在接近目标的同时降低速度、稳定姿态才能获得该奖励，从而学会"接近→减速→着陆"的完整链条。

**数学形式**：
\[
\text{prox} = \frac{1}{1 + 10 \cdot \text{dist\_next}}, \quad
\text{speed\_factor} = \frac{1}{1 + 5 \cdot (|v_x| + |v_y|)}, \quad
\text{angle\_factor} = \frac{1}{1 + 3 \cdot (|\theta| + |\dot{\theta}|)}
\]
\[
\text{approach\_bonus} = w \cdot \text{prox} \cdot \text{speed\_factor} \cdot \text{angle\_factor},\quad w = 2.0
\]
所有因子 ∈ (0, 1]，乘积不会塌缩到零（因为 soft 函数在条件不完美时也有正基值），在目标附近、低速、姿态良好的理想区域可达约 1.6~2.0，足以引导着陆行为。

**系数校准**：
- 最大单步奖励 2.0，远高于当前 `progress_reward` 的 per‑step average（≈0.0013），确保着陆行为能被学到。但它只在目标附近一小段轨迹内触发（`prox` 随距离迅速衰减），不会在全程支配奖励。
- 安全惩罚 `landing_safety_penalty` 当前 per‑step mean ≈0.0005 < 0.3×主信号，符合设计要求，本轮不调整（单组件修改限制）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前状态（用于距离计算）
    x, y = obs[0], obs[1]
    # 下一状态
    nx, ny = next_obs[0], next_obs[1]
    nx_v, ny_v = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_v = next_obs[5]

    # ---------- 1. 进度奖励：向目标 (0,0) 靠近 ----------
    dist_curr = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress = dist_curr - dist_next
    progress_reward = 1.0 * progress

    # ---------- 2. 着陆预备奖励：接近目标且速度低、姿态稳 ----------
    # soft 门控因子，不依赖接触标志
    prox = 1.0 / (1.0 + 10.0 * dist_next)
    speed_factor = 1.0 / (1.0 + 5.0 * (abs(nx_v) + abs(ny_v)))
    angle_factor = 1.0 / (1.0 + 3.0 * (abs(n_angle) + abs(n_ang_v)))

    approach_bonus = 2.0 * prox * speed_factor * angle_factor

    # ---------- 3. 着陆安全性惩罚（保持原系数，仅门控微调） ----------
    v_limit = 0.5
    vx_pen = max(0.0, abs(nx_v) - v_limit)
    vy_pen = max(0.0, abs(ny_v) - v_limit)
    vel_pen = vx_pen + vy_pen

    ang_limit = 1.0
    ang_pen = max(0.0, abs(n_ang_v) - ang_limit)

    tilt_pen = abs(n_angle)

    gate_safety = 1.0 / (1.0 + 5.0 * dist_next)
    landing_safety_penalty = (0.03 * vel_pen + 0.02 * ang_pen + 0.03 * tilt_pen) * gate_safety

    # ---------- 总奖励 ----------
    total_reward = progress_reward + approach_bonus - landing_safety_penalty

    components = {
        "progress_reward": float(progress_reward),
        "approach_bonus": float(approach_bonus),
        "landing_safety_penalty": float(landing_safety_penalty)
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**：接触奖励是僵尸组件（active_rate=0%），距离门控造成死锁；安全惩罚太弱但暂时保留；所有观测已被使用，问题在于门控设计而非信号缺失。
- **behavior**：agent 学会了向目标缓慢移动，全程存活（truncated at 1000），但从未尝试着陆。
- **signal**：缺少“接近目标后减速+调姿”的正向引导，导致 agent 没有动机启动着陆行为。
- **level**：Level 2（替换僵尸组件，用无接触条件的 soft 多因子乘积引导着陆预备）。
- **hypothesis**：新组件在接近目标时奖励低速和稳定姿态，agent 将学会先靠近再减速着陆，突破当前的徘徊死结。
- **risk**：如果 agent 在目标附近学会“缓慢移动以持续骗取奖励”而不着陆，可能延长 episode 而非终止；需后续监测是否成功触发 body_not_awake。
