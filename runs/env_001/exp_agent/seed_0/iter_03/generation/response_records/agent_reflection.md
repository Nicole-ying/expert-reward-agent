# Response Record

## 诊断分析

### 1. Agent 发生了什么？
- Episode 长度 74，100% 提前终止（<150 steps + score<-50）→ **agent 在 crash**。
- 得分 -111，和 iter 1 的 -107 相比无改善。

### 2. 哪个组件是主因？
- **`stability_penalty` ratio_to_progress = -3.576**，惩罚强度是进度信号的 3.5 倍。agent 被"不动比动安全"的梯度绑架，无法有效探索。
- `soft_landing_proxy` nonzero_rate = 0.49%，即使改成连续乘积，阈值仍太紧，对 crash 中的 agent 形同虚设。
- episode_length 74 说明 agent 在远处就失控了，距离门控 `proximity_gate` 根本没机会激活（agent 到不了目标附近）。

### 3. 我之前改了什么？
- Iter 1→2：给 stability 加距离门控 + 把 soft_landing 从二值改为连续乘积。得分从 -107 变成 -111（更差）。距离门控对 crash 中的 agent 无效——它根本到不了"近处"。
- **同一骨架家族已 2 轮无改善，且最佳得分 -107 远低于可接受水平。历史 iter 3 用 `potential_shaping` 骨架拿到 242 分（已解决）。**

### 决策：换骨架 → `potential_based_shaping`

根据原则"层次 3：换骨架"——同一骨架迭代 2 轮无改善 + 最佳得分远低于 target 的 25%，应停止在系数/形式上打转。同时，历史记忆显示 `potential_shaping + soft_landing_proxy + angular_vel_penalty + energy_penalty` 曾解出 242 分，这是已验证可行的方向。

`potential_shaping` 的优势：Φ = -(distance + speed + |angle|)，势能差天然同时引导"接近 + 减速 + 稳定"，一个信号完成三个目标，不存在 stability_penalty 和 progress 互相打架的问题。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── 提取当前状态 ──
    x, y = obs[0], obs[1]
    curr_vx, curr_vy = obs[2], obs[3]
    curr_angle = obs[4]

    # ── 提取下一时刻状态（反映动作后果）──
    next_x, next_y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 距离 ──
    dist = (x**2 + y**2) ** 0.5
    next_dist = (next_x**2 + next_y**2) ** 0.5

    # ── 速率标量 ──
    speed = (vx**2 + vy**2) ** 0.5
    curr_speed = (curr_vx**2 + curr_vy**2) ** 0.5

    # ── 主学习信号：potential-based shaping ──
    # Φ = -(distance + α*speed + β*|angle|)
    # F = γ * Φ(next) - Φ(curr)，天然引导接近+减速+姿态稳定
    alpha = 0.1
    beta = 0.2
    gamma = 0.99

    phi_curr = -(dist + alpha * curr_speed + beta * abs(curr_angle))
    phi_next = -(next_dist + alpha * speed + beta * abs(angle))
    potential_shaping = gamma * phi_next - phi_curr

    # ── 软着陆 proxy（连续乘积，阈值放宽）──
    # 用 bounded max(0, 1-x/threshold) 提供连续梯度
    prox_factor = max(0.0, 1.0 - next_dist / 1.0)               # 距离 < 1.0
    vel_factor = max(0.0, 1.0 - speed / 0.5)                    # 速率 < 0.5
    angle_factor = max(0.0, 1.0 - abs(angle) / 0.3)             # 倾角 < 0.3
    ang_vel_factor = max(0.0, 1.0 - abs(ang_vel) / 0.3)         # 角速度 < 0.3
    contact_factor = min(left_contact, right_contact)            # 双脚触地

    soft_landing_proxy = (
        prox_factor * vel_factor * angle_factor * ang_vel_factor * contact_factor
    )

    # ── 轻量辅助惩罚 ──
    # 角速度惩罚：抑制剧烈旋转
    angular_vel_penalty = -0.01 * abs(ang_vel)
    # 能量惩罚：鼓励高效移动
    energy_penalty = -0.01 * (abs(vx) + abs(vy))

    # ── 总奖励 ──
    total_reward = (
        1.0 * potential_shaping +
        2.0 * soft_landing_proxy +
        1.0 * angular_vel_penalty +
        1.0 * energy_penalty
    )

    components = {
        "potential_shaping": potential_shaping,
        "soft_landing_proxy": soft_landing_proxy,
        "angular_vel_penalty": angular_vel_penalty,
        "energy_penalty": energy_penalty,
    }

    return float(total_reward), components
```
