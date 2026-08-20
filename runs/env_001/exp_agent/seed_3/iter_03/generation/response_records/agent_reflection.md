# Response Record

## 诊断

### 1. Agent 状态
Score 从 -109（iter 1，70步 crash）跳到 148（iter 2，745步存活）。进步巨大，agent 不再 crash，在正常探索。但 episode 偏长（745），可能近目标区域有徘徊。

### 2. 组件分析
- **progress_delta**：mean=0.011，贡献约 0.11/timestep — 主信号正常。
- **stability_penalty**：ratio=-1.06，已平衡，不再压制探索。✅
- **soft_landing_proxy**：nonzero_rate=8.7%，触发不算太低，但它是**二值信号**（0 or 0.2），没有梯度。搜索结果显示这匹配 `goal_near_oscillation` 模式：episode 长、landing proxy 触发率 < 10%。

### 3. 上轮改动回顾
上轮加了 distance gate 并降低 stability 系数，效果很好。本轮聚焦 soft_landing_proxy：从二值改为连续乘积形式，提供梯度引导。

### 修改方案
将 `soft_landing_proxy` 从二值 `if` 改为连续乘积因子（`bounded_continuous_proxy` 技法）：
- 每个条件用 `max(0, 1 - x/threshold)` 提供梯度
- 乘积形式天然要求所有条件同时满足
- 系数 0.4，完美状态下 max=0.4（比原来 0.2 高，但乘积很少达到 1.0）

只改这一个组件，progress_delta 和 stability_penalty 不动。

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
    # Unchanged from last iteration — already well-balanced (ratio ~ -1.06)
    distance_gate = 1.0 / (1.0 + 5.0 * d_next)
    w_vel   = 0.03
    w_angle = 0.15
    w_omega = 0.03
    speed_penalty = w_vel * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = w_angle * abs(next_obs[4])
    omega_penalty = w_omega * abs(next_obs[5])
    stability_penalty = -distance_gate * (speed_penalty + angle_penalty + omega_penalty)

    # ========== 3. Continuous soft landing proxy (CHANGED: binary → continuous product) ==========
    # Rationale: binary if-condition gives no gradient. Replaced with product of
    # bounded factors max(0, 1-x/T), each providing gradient toward the desired state.
    # The product naturally enforces "all conditions simultaneously" while still
    # giving partial credit when some (but not all) are met.

    prox_factor   = max(0.0, 1.0 - d_next / 0.5)                                # 0→1 as dist 0.5→0
    speed_factor  = max(0.0, 1.0 - (abs(next_obs[2]) + abs(next_obs[3])) / 0.5)  # 0→1 as total speed 0.5→0
    angle_factor  = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)                       # 0→1 as |angle| 0.3→0
    contact_factor = (next_obs[6] + next_obs[7]) / 2.0                           # 0 / 0.5 / 1.0

    soft_landing_proxy = 0.4 * prox_factor * speed_factor * angle_factor * contact_factor

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
