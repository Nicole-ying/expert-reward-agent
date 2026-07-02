## 诊断

### 1. Agent 发生了什么？
Episodes 极短 (70.6步)，全部 early terminal，得分 -111。Agent 在崩溃——根本没有接近目标区域。这是从 iter 3 (得分 197，长度 796) 的完全回退。

### 2. 根因分析

核心问题出在 `soft_landing_proxy`：

- **nonzero_rate = 0.006**（仅 0.6% 的 step 有信号）。在 iter 3 成功时 agent 能到达着陆区，landing proxy 有密集信号；现在 agent 崩溃，根本进不了着陆区。
- **为什么进不了着陆区？** 连续乘积 `prox * speed * angle * contact` 有一个致命缺陷：当 `contact_factor = 0`（双腿不触地）时，**所有四个因子的梯度全部为零**（链式法则：∂/∂(prox) = speed×angle×contact = 0）。这意味着 agent 在接触地面前，无法获得任何"靠近目标"或"减速"的梯度引导。着陆区成了一个 agent 必须靠运气撞进去的盲区。
- iter 3 的成功可能是运气好撞进了着陆区，然后高系数 (0.4) 的强信号把它留住了。系数降到 0.25 后，信号太弱，agent 再也没有重新发现着陆区。
- `stability_penalty` 通过 `distance_gate` 在靠近目标时变强（ratio = -0.70），这进一步惩罚 agent 接近目标，形成"靠近就被罚"的局部最优。

### 3. 上一轮改了什么？
上次把 landing proxy 系数从 0.4 降到 0.25，理由是"ratio 47.8 太高"。但这个诊断本身有问题——完成型奖励 ratio 天然偏高不是 bug（原则 1）。降系数导致崩溃。

### 修复方案

**单一改动：给 `contact_factor` 加 epsilon (0.05)，恢复系数到 0.4。**

- `contact_factor = (next_obs[6] + next_obs[7]) / 2.0 + 0.05`：确保即使双腿不触地，product 也有非零梯度，所有四个因子的梯度都能回流。
- 系数回到 0.4（iter 3 验证过的值）。
- progress_delta 和 stability_penalty 不动——它们在各轮迭代中表现稳定。

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
    # Unchanged — well-balanced across all iterations
    distance_gate = 1.0 / (1.0 + 5.0 * d_next)
    w_vel   = 0.03
    w_angle = 0.15
    w_omega = 0.03
    speed_penalty = w_vel * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = w_angle * abs(next_obs[4])
    omega_penalty = w_omega * abs(next_obs[5])
    stability_penalty = -distance_gate * (speed_penalty + angle_penalty + omega_penalty)

    # ========== 3. Continuous soft landing proxy (CHANGED: epsilon on contact_factor) ==========
    # Rationale: the product form prox * speed * angle * contact has zero gradient for ALL
    # factors when contact_factor = 0 (chain rule: derivative involves product of other three).
    # This means the agent gets NO gradient toward the goal until it accidentally touches a leg.
    # Adding epsilon=0.05 to contact_factor ensures gradient always flows through the product,
    # while leg contact still provides a strong boost (0.55 or 1.05 vs baseline 0.05).
    # Coefficient restored to 0.4 (validated in iter 3, score 197).

    prox_factor   = max(0.0, 1.0 - d_next / 0.5)                                # 0→1 as dist 0.5→0
    speed_factor  = max(0.0, 1.0 - (abs(next_obs[2]) + abs(next_obs[3])) / 0.5)  # 0→1 as total speed 0.5→0
    angle_factor  = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)                       # 0→1 as |angle| 0.3→0
    contact_factor = (next_obs[6] + next_obs[7]) / 2.0 + 0.05                    # 0.05 / 0.55 / 1.05

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