## 诊断

### 1. Agent 状态
得分 197（历史最佳），episode 长 796（无 crash），0 early terminal。**Agent 正在学习且持续进步**（-109 → 149 → 197）。但进步速度在放缓：上次改连续 landing proxy 后得分 +32%，而 progress_delta 均值反而从 0.011 跌到 0.004。

### 2. 组件分析
- **soft_landing_proxy**: ratio=47.8, mean=0.192, nonzero_rate=56.6%。不是"偶尔触发的终端事件"（56% 触发率太高），而是**在充当持续奖励源**，贡献了总奖励的 85%（0.192/0.226）。
- **progress_delta**: mean=0.004。小是自然的（agent 大部分时间在目标附近），但 10×0.004=0.04 的贡献 vs 0.192 的 landing proxy，主学习信号被淹没了。
- **stability_penalty**: ratio=-1.61，健康。

### 3. 根因
上一轮把 landing proxy 从 binary → continuous product，这是正确的方向（得分 +32%），但系数 0.4 偏大，导致 landing proxy 从"引导信号"变成了"主导信号"。Agent 可能在 exploit 部分 landing credit 而非追求完美着陆（得分分布 124-284 跨度大也印证了这一点）。

### 修改方案
**层次 1 — 改系数**：把 `soft_landing_proxy` 系数从 0.4 降到 0.25。保留连续乘积形式（梯度质量已验证有效），但减弱其量级，让 progress_delta 重新成为主要驱动力。不改其他组件。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== Helper: distance to goal ==========
    def dist_to_goal(x, y):
        return (x**2 + y**2) ** 0.5

    # ========== 1. Progress delta reward (main learning signal) ==========
    d_current = dist_to_goal(obs[0], obs[1])
    d_next = dist_to_goal(next_obs[0], next_obs[1])
    progress_delta = d_current - d_next   # positive when getting closer

    # ========== 2. Stability penalty with distance gating ==========
    # Unchanged — well-balanced (ratio ~ -1.6)
    distance_gate = 1.0 / (1.0 + 5.0 * d_next)
    w_vel   = 0.03
    w_angle = 0.15
    w_omega = 0.03
    speed_penalty = w_vel * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = w_angle * abs(next_obs[4])
    omega_penalty = w_omega * abs(next_obs[5])
    stability_penalty = -distance_gate * (speed_penalty + angle_penalty + omega_penalty)

    # ========== 3. Continuous soft landing proxy (CHANGED: coefficient 0.4 → 0.25) ==========
    # Rationale: iter 3 showed continuous product works (score +32%), but ratio 47.8 means
    # landing proxy dominates total reward (85% contribution). Reducing coefficient to 0.25
    # brings expected ratio from ~48 down to ~30 while preserving gradient signal.

    prox_factor   = max(0.0, 1.0 - d_next / 0.5)                                # 0→1 as dist 0.5→0
    speed_factor  = max(0.0, 1.0 - (abs(next_obs[2]) + abs(next_obs[3])) / 0.5)  # 0→1 as total speed 0.5→0
    angle_factor  = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)                       # 0→1 as |angle| 0.3→0
    contact_factor = (next_obs[6] + next_obs[7]) / 2.0                           # 0 / 0.5 / 1.0

    soft_landing_proxy = 0.25 * prox_factor * speed_factor * angle_factor * contact_factor

    # ========== Total reward ==========
    total_reward = 10.0 * progress_delta + stability_penalty + soft_landing_proxy

    components = {
        "progress_delta": progress_delta,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components
```