## 诊断

### 1. 这个 agent 发生了什么？
**得分 251.36，episode 375.7，零 crash。** 这是质的飞跃（上轮 -0.83）。Stage-weighted 策略生效了：早期 proximity 驱动靠近，后期 quality 驱动精准着陆。agent 已经学会在约 375 步内完成着陆。

### 2. 哪个组件需要改进？
`landing_quality` 用了**三因子乘积** `near_factor * slow_factor * upright_factor`。虽然 nonzero_rate 58% 说明 agent 进入了近区，但乘积形态有梯度消失问题：当任一因子接近 0 时，其他两个因子的梯度也被乘性压制，导致 agent 难以同时优化三个维度。mean=0.319 意味着各因子均值约 0.68，乘积把信号压缩到了 `0.68³ ≈ 0.32`。

### 3. 上轮改了什么？
上轮是完全重构（从 potential_shaping 换到 stage-weighted proximity + landing quality），得分从 -0.83 跳到 251。本轮只在 landing_quality 内部做数学形态改进。

### 修改方案
**只改 landing_quality 的数学形式**：从乘积改为均值 `(near_factor + slow_factor + upright_factor) / 3.0`。每个因子独立贡献梯度，不再互相压制。系数 `w_quality_late` 从 4.0 降到 3.0 以匹配新形态的量级（均值约为乘积的 2 倍）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Stage-weighted: early focus on proximity, late focus on precision landing.
    Changed landing_quality from product to mean — each factor gets independent gradient.
    """
    # --- hyperparameters ---
    k_proximity = 5.0          # decay rate for bounded proximity
    w_proximity_early = 3.0    # proximity weight in early training
    w_quality_late = 3.0       # landing quality weight in late training (reduced from 4.0 to match sum form)
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

    # --- 2. continuous landing quality (MEAN instead of product) ---
    # Product form: near * slow * upright → gradient vanishes when any factor ≈ 0
    # Mean form: each factor contributes independently, no gradient interference
    near_factor = max(0.0, 1.0 - dist / D_near)
    slow_factor = max(0.0, 1.0 - speed / V_slow)
    upright_factor = max(0.0, 1.0 - abs(angle) / A_upright)
    landing_quality = (near_factor + slow_factor + upright_factor) / 3.0

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