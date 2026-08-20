# Response Record

## 诊断

### 1. Agent 发生了什么？
Episode 长度 70，全部 early terminal，score -110 → **crash**。Agent 完全没学会靠近目标。

### 2. 哪个组件是主要原因？
- `soft_landing_proxy` 的 `nonzero_rate = 0.6%` — 几乎从不触发。四个因子乘积 (prox × speed × angle × contact)：只要 agent 离目标 > 0.5，`prox_factor = 0`，整个乘积归零，梯度链断裂。
- `stability_penalty` ratio = 0.69，相对 progress 偏强，在 agent 挣扎时进一步压制探索。
- Iter 3 曾用**完全相同代码**拿到 197 分，说明这个奖励函数**能工作但极度脆弱**——一旦策略稍有扰动偏离目标区域，landing proxy 立刻死寂，agent 再也恢复不了。

### 3. 我之前改了什么？
Iter 5 给 `contact_factor` 加了 `+0.05` epsilon，试图让乘积总有梯度。但这没用——当 `prox_factor = 0`（距离 > 0.5）时，乘积仍然是 0。根因不在 `contact_factor`，而在乘积形态本身。

**结论**：乘积形态是根本问题。改成加权和，让每个因子**独立**贡献梯度。这是层次 2 修改（数学形式），回退 epsilon hack。

同时检查 validation 失败原因：上轮代码在 `components` 中放了 `total_reward`，虽不致命但按规范只应放公式中直接出现的变量。严格遵循规范即可。

---

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
    # Proven balanced across iter 1-5, ratio ~0.69 relative to progress.
    distance_gate = 1.0 / (1.0 + 5.0 * d_next)
    w_vel   = 0.03
    w_angle = 0.15
    w_omega = 0.03
    speed_penalty = w_vel * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = w_angle * abs(next_obs[4])
    omega_penalty = w_omega * abs(next_obs[5])
    stability_penalty = -distance_gate * (speed_penalty + angle_penalty + omega_penalty)

    # ========== 3. Continuous soft landing proxy (CHANGED: product → weighted sum) ==========
    # Rationale: the product form prox*speed*angle*contact has zero gradient whenever
    # ANY single factor is zero — prox_factor=0 when d>0.5 kills everything.
    # This makes the reward landscape fragile: once the agent drifts outside the goal 
    # zone, it receives zero landing signal and cannot recover (iter 4-5 both crashed).
    #
    # New form: weighted sum. Each bounded factor contributes independently,
    # so gradient flows even when some factors are at zero. The sum still peaks
    # when ALL conditions are met (all factors→1), preserving the desired behavior.
    # contact_factor gets double weight to emphasize leg contact as the strongest cue.

    prox_factor   = max(0.0, 1.0 - d_next / 0.5)                                # 0→1 as dist 0.5→0
    speed_factor  = max(0.0, 1.0 - (abs(next_obs[2]) + abs(next_obs[3])) / 0.5)  # 0→1 as total speed 0.5→0
    angle_factor  = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)                       # 0→1 as |angle| 0.3→0
    contact_factor = (next_obs[6] + next_obs[7]) / 2.0                           # 0 / 0.5 / 1.0

    soft_landing_proxy = 0.1 * (prox_factor + speed_factor + angle_factor + 2.0 * contact_factor)

    # ========== Total reward ==========
    total_reward = 10.0 * progress_delta + stability_penalty + soft_landing_proxy

    components = {
        "progress_delta": progress_delta,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy
    }

    return float(total_reward), components
```
