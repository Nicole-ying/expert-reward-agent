## 诊断

让我逐一分析三个问题：

**1. Agent 发生了什么？**
Episode 长度只有 72 步，10/10 全部提前终止（crash）。得分 -108.75，极差。这是典型的 crash 模式。

**2. 哪个组件是主因？**
看 `ratio_to_progress`（以 potential_shaping 为基准）：
- `potential_shaping`：+0.021（主进度信号）
- `stability_penalty`：**-0.031**（惩罚项 magnitude 比进度信号大 1.5 倍！）

根据 `coefficient_scale_awareness` 原则：**progress 信号的 abs_mean 应该至少是惩罚项的 10 倍**。当前惩罚项反而比进度信号还大。Agent 收到的净信号是负的（total_reward=-0.001），它每走一步都在被惩罚，自然会 crash。

`soft_landing_proxy` 触发率仅 0.48%，完全来不及起作用。

**3. 我之前改了什么？**
上一轮（iter 7）从 `approach_bonus` 骨架切换到 `potential_shaping` 骨架。问题在于 `potential_shaping = gamma*phi_next - phi_obs` 没有系数放大，每步信号量级受物理步长限制，天生微弱（~0.02）。而 stability_penalty 的系数没相应调整，导致惩罚喧宾夺主。

**修复方案：** 给 potential_shaping 加一个系数 `w_potential = 10.0`，使其成为真正的主导信号。这在理论上等价于使用势能函数 `Φ = -10*dist`，仍然是最优策略不变的塑形。只改这一个组件，让下一轮可归因。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Parameters
    gamma = 0.995
    w_potential = 10.0  # amplify potential shaping to dominate penalty
    w_v = 0.02  # velocity penalty weight
    w_a = 0.1   # angle penalty weight
    w_w = 0.05  # angular velocity penalty weight
    landing_bonus = 2.0
    # Thresholds for soft landing proxy
    dist_thresh = 0.3
    speed_thresh_x = 0.1
    speed_thresh_y = 0.15
    angle_thresh = 0.15

    # 1. Potential based shaping:
    #    Phi(s) = - distance_to_target
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    phi_obs = -dist_obs
    phi_next = -dist_next
    potential_shaping = gamma * phi_next - phi_obs  # = -gamma*dist_next + dist_obs

    # 2. Stability penalty (encourages low speed, small angle, low angular velocity)
    stability_penalty = - (
        w_v * (abs(next_obs[2]) + abs(next_obs[3])) +
        w_a * abs(next_obs[4]) +
        w_w * abs(next_obs[5])
    )

    # 3. Soft landing proxy (indicates successful touchdown without explicit success flag)
    soft_landing_proxy = 0.0
    if (dist_next < dist_thresh
            and abs(next_obs[2]) < speed_thresh_x
            and abs(next_obs[3]) < speed_thresh_y
            and abs(next_obs[4]) < angle_thresh
            and next_obs[6] == 1.0
            and next_obs[7] == 1.0):
        soft_landing_proxy = landing_bonus

    total_reward = w_potential * potential_shaping + stability_penalty + soft_landing_proxy

    components = {
        "potential_shaping": potential_shaping,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

**预期效果：** potential_shaping 均值从 0.021 提升到 ~0.21，而 stability_penalty 保持在 -0.031，比例从 0.67x 反转为 **~7x**，进度信号成为主导。Agent 应该不再原地崩溃，开始向目标移动。