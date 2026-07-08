# Response Record

## 分析

### 1. Agent 发生了什么？

**核心症状：高组件奖励、负总分、零终止。**

- **20/20 truncated**，0 terminated — agent 从未真正着陆，在 1000 步耗尽后截断。
- **组件总和 ~480**（landing_proxy 265 + proximity 206 + progress 12 + stability -3），但 score = **-24.2**。
- 约 504 的环境隐性惩罚（燃料消耗）被我们的密集正向奖励掩盖，agent 找到局部最优：**在目标附近悬停，部分满足 landing 条件，持续收割 proxy 奖励，永不完成着陆**。
- `proximity_reward` 激活率 100%，`landing_proxy` 65.8% — 验证了"靠近+部分着陆条件"的悬停策略。
- `progress_reward` 仅 2.5% 份额 — 对行为几乎无贡献。
- `stability_penalty` 仅 -0.6% — 惩罚太弱，未有效约束。

这是典型的 `state_to_improvement` 诊断：策略停留在奖励状态，而非完成任务。

### 2. 哪个组件最值得干预？

**proximity_reward** 是最大问题（100% 激活、42% 份额、恒定发放），它与 landing_proxy 联合创造了"靠近目标就可获利"的局部最优。landing_proxy 的 0.1 地板值（无接触也能拿 10%）和松弛阈值进一步放大了漏洞。

### 3. 上一轮改了什么？

从 best 到 current：将 `r_dist = 1/(1+d)` 拆成 `r_progress + r_proximity`，降低 stability 系数（0.01→0.005 等），提升 landing 系数（8→15）。结果是 proximity 仍占主导，landing proxy 更易被收割，总分更差。

### 修改方案

- **proximity_reward 从 0.3 降至 0.1** — 大幅削弱悬停收割的收益
- **progress_reward 改为对称式** `8.0*(old_dist - new_dist)` — 靠近奖励、远离惩罚、静止为零，消除 steady-state 获利
- **landing_proxy 收紧**：距离窗口 0.3、速度窗口 0.2、角度窗口 0.15（原 0.5/0.4/0.3），去除地板，要求至少单腿触地才激活，双腿满分、单腿 15%
- **stability_penalty 恢复 best 系数**（0.01/0.1/0.05），提供足够行为约束

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack states
    cx, cy = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    old_dist = (cx ** 2 + cy ** 2) ** 0.5
    new_dist = (nx ** 2 + ny ** 2) ** 0.5

    # 1. Symmetric progress: reward approach, penalize retreat, zero when stationary
    r_progress = 8.0 * (old_dist - new_dist)

    # 2. Weak proximity for gradient maintenance only
    r_proximity = 0.1 / (1.0 + new_dist)

    # 3. Stability penalty (restored best coefficients)
    r_stability = -(
        0.01 * (abs(nvx) + abs(nvy)) +
        0.1 * abs(body_angle) +
        0.05 * abs(ang_vel)
    )

    # 4. Strict landing proxy — tight thresholds, no floor, contact-gated
    proximity = max(0.0, 1.0 - new_dist / 0.3)
    stillness = max(0.0, 1.0 - (abs(nvx) + abs(nvy)) / 0.2)
    upright   = max(0.0, 1.0 - abs(body_angle) / 0.15)

    if left_contact > 0.5 and right_contact > 0.5:
        contact_factor = 1.0
    elif left_contact > 0.5 or right_contact > 0.5:
        contact_factor = 0.15
    else:
        contact_factor = 0.0

    r_landing = 30.0 * proximity * stillness * upright * contact_factor

    total_reward = r_progress + r_proximity + r_stability + r_landing

    components = {
        "progress_reward": r_progress,
        "proximity_reward": r_proximity,
        "stability_penalty": r_stability,
        "landing_proxy": r_landing
    }

    return float(total_reward), components
```
