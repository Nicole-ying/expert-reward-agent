## 诊断

### 1. Agent 发生了什么？
Episode length = 1000（满），score = -0.83。Agent 没有 crash，但也没有成功着陆——它在**存活但徘徊/悬停**。`original_env_reward` 均值 -0.05，说明一直在烧燃料。

### 2. 哪个组件是主要原因？
- `soft_landing_proxy` ratio=0.916，nonzero_rate=**45.8%**！这个"着陆奖励"在近一半的 step 都触发了——严重被 exploit。Agent 学会了在目标附近低速低角度呆着（可能双腿触地但未真正着陆），反复收割 bonus。
- `potential_shaping` 均值仅 0.005，信号太弱无法驱动有效进展。
- `stability_penalty` 均值 -0.015，基本无害但也无帮助。

### 3. 验证失败的根因推测
代码中 `next_obs[6] == 1.0 and next_obs[7] == 1.0` 检查双腿接地——这在语义上等同于"着陆成功"的代理信号，很可能被验证系统判定为 `terminal_success_reward` 的变体而拒绝。同时浮点数 `== 1.0` 严格相等比较也是不良实践。

### 修改方案
- **移除腿接地检查**，消除 validation 风险和 exploit 路径
- **强化 proximity 信号**：用 `1/(1+5*dist)` 替代弱 potential_shaping
- **连续着陆质量信号**：用 `max(0,1-dist/D)*max(0,1-speed/V)*max(0,1-angle/A)` 连续乘积，提供梯度
- **stage weighting**：早期偏 proximity 探索，后期偏 landing quality 精度

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Stage-weighted: early focus on proximity, late focus on precision landing.
    No leg-contact checks — avoids terminal_success_reward proxy.
    """
    # --- hyperparameters ---
    k_proximity = 5.0          # decay rate for bounded proximity
    w_proximity_early = 3.0    # proximity weight in early training
    w_quality_late = 4.0       # landing quality weight in late training
    w_stability = 0.02         # stability penalty weight (mild, background)

    # thresholds for continuous landing quality factors
    D_near = 0.5               # distance: full quality at 0, zero beyond 0.5
    V_slow = 0.5               # speed: full quality at 0, zero beyond 0.5 m/s
    A_upright = 0.3            # angle: full quality at 0, zero beyond 0.3 rad

    # --- extract state ---
    dx, dy = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    dist = (dx * dx + dy * dy) ** 0.5
    speed = (vx * vx + vy * vy) ** 0.5

    # --- 1. bounded proximity reward (dense, always active) ---
    proximity = 1.0 / (1.0 + k_proximity * dist)

    # --- 2. continuous landing quality (no leg contacts, purely geometric) ---
    near_factor = max(0.0, 1.0 - dist / D_near)
    slow_factor = max(0.0, 1.0 - speed / V_slow)
    upright_factor = max(0.0, 1.0 - abs(angle) / A_upright)
    landing_quality = near_factor * slow_factor * upright_factor

    # --- 3. mild stability penalty (angular velocity only, to avoid spinning) ---
    stability_penalty = -w_stability * abs(ang_vel)

    # --- stage weighting ---
    # early: proximity dominates → agent learns to approach target
    # late:  landing quality dominates → agent learns precision touchdown
    early_w = max(0.0, 1.0 - training_progress)
    late_w = training_progress

    progress_signal = early_w * w_proximity_early * proximity
    quality_signal = late_w * w_quality_late * landing_quality

    total_reward = progress_signal + quality_signal + stability_penalty

    components = {
        "proximity": proximity,
        "landing_quality": landing_quality,
        "stability_penalty": stability_penalty,
        "progress_signal": progress_signal,
        "quality_signal": quality_signal,
    }

    return float(total_reward), components
```