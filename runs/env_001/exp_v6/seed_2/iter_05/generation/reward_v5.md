## 诊断

### 1. Agent 发生了什么？
**Crash。** Episode 长度从 iter 3 的 1000 步骤降到 iter 4 的 85 步，60% 提前终止，分数从 -12 暴跌到 -60。agent 在快速死亡。

### 2. 哪个组件是主要原因？
- **Iter 3（最佳，-12.26，1000步）：** `approach_bonus = 2.711`，ratio_to_progress 巨大（~1355）。这是典型的 exploitation — agent 学会了在原点附近悬停不动，collect 稳定的 bonus，而不是真正降落。
- **Iter 4（当前，-60.67，85步）：** 上一次我做了两件事：① 加了 `max(0, progress_delta)` 门控，② 把系数从 5.0 砍到 2.0。结果是 `approach_bonus` 从 2.711 暴跌到 0.005 — **消失了**。而 `stability_penalty` ratio_to_progress = -0.55（> 0.5 阈值），这个负信号现在是主导力量。agent 失去了 iter 3 中那个（虽然 exploitative 但至少提供了稳定性引导的）bonus，只剩下惩罚，所以它 crash 了。

### 3. 我之前改了什么？
上一次把 approach_bonus 从"状态奖励"（product form，每步都给）改成了"进度门控奖励"（乘以 progress_delta）。这个方向对——状态奖励确实导致了 exploit——但数学形式错了：`progress_delta ~ 0.01` 乘进去之后信号直接消失。

### 修复方案
换用 **potential-based shaping**（`Φ(next) - Φ(obs)`）。这是唯一数学上保证不改变最优策略的塑形方式，天然解决"hover-and-collect"问题：停在原地 → ΔΦ = 0，没有奖励。只改 approach_bonus 的数学形式，其他组件不动。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- position and progress delta (main learning signal) ----
    px, py = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    d_obs = (px**2 + py**2) ** 0.5
    d_next = (nx**2 + ny**2) ** 0.5
    progress_delta = d_obs - d_next

    # ---- distance penalty (unchanged) ----
    distance_penalty = -0.005 * d_next

    # ---- stability penalty (unchanged) ----
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    stability_penalty = (
        -0.005 * abs(vx)
        - 0.005 * abs(vy)
        - 0.02 * abs(angle)
        - 0.02 * abs(ang_vel)
    )

    # ---- approach quality bonus: potential-based shaping ----
    # KEY CHANGE: switched from state-reward (product form → hover exploit)
    # to potential-based shaping (Φ(next) - Φ(obs)).
    # Φ = proximity * speed_ok * angle_ok — a bounded [0,1] potential that
    # captures "how good is the approach state".
    # Bonus = Φ(next) - Φ(obs) — only rewards IMPROVEMENT, zero when static.
    # This naturally prevents the hover exploit: staying still gives ΔΦ = 0.
    APPROACH_DIST = 2.0
    SPEED_THRESH = 2.0
    ANGLE_THRESH = 0.5

    # Potential at current state (obs)
    prox_now = max(0.0, 1.0 - d_obs / APPROACH_DIST)
    spd_now = max(0.0, 1.0 - (abs(obs[2]) + abs(obs[3])) / SPEED_THRESH)
    ang_now = max(0.0, 1.0 - abs(obs[4]) / ANGLE_THRESH)
    potential_now = prox_now * spd_now * ang_now

    # Potential at next state (next_obs)
    prox_next = max(0.0, 1.0 - d_next / APPROACH_DIST)
    spd_next = max(0.0, 1.0 - (abs(vx) + abs(vy)) / SPEED_THRESH)
    ang_next = max(0.0, 1.0 - abs(angle) / ANGLE_THRESH)
    potential_next = prox_next * spd_next * ang_next

    # γ = 1.0 (standard for potential-based shaping, per Ng 1999)
    approach_bonus = (potential_next - potential_now) * 2.0

    # ---- total ----
    total = progress_delta + distance_penalty + stability_penalty + approach_bonus

    components = {
        "progress_delta_reward": progress_delta,
        "distance_penalty": distance_penalty,
        "stability_penalty": stability_penalty,
        "approach_bonus": approach_bonus,
        "total_reward": total,
    }
    return float(total), components
```