## 诊断

### 1. Agent 发生了什么？
- **得分 -110.31，episode 长度 72（100% 早终止）→ Agent 在 crash。**
- 对比 iter 3（最佳：-12.26，len=1000，成功着陆），iter 5 完全退化了。

### 2. 哪个组件是主要原因？
追踪历史：

| iter | approach_bonus mean | score | len | 
|------|---------------------|-------|-----|
| 3 (best) | **2.711** | -12.26 | 1000 |
| 4 | 0.005 | -60.67 | 85 |
| 5 (current) | **0.009** | -110.31 | 72 |

关键发现：**iter 3 的 state-reward 乘积形态成功引导了着陆**，approach_bonus=2.711 是主导信号。而我上一轮把它改成了 potential-based shaping（`Φ(next)-Φ(obs)`）——这个差分量级天然极小（相邻帧的势能变化微乎其微），导致 approach_bonus 均值只有 0.009，完全失去了引导能力。

### 3. 我之前改了什么？
上一轮从 **state-reward 乘积** → **potential-based shaping**，以为能解决"hover exploit"。但 iter 3 证明 state-reward 并没有 hover exploit 问题（agent 成功着陆了）。这个改动是方向性错误。

### 根因：乘积形态的脆弱性 + 错误的 potential-based shaping

但 iter 3→4 的退化也暴露了纯乘积形态的一个问题：当任一因子为零且处于 flat 区域时（如 angle > ANGLE_THRESH 导致 `max(0, 1-angle/THRESH)=0` 且导数为 0），**所有因子的梯度都被杀死**。Agent 一旦初始化到 bad angle，就再也收不到 approach 信号了。

**修复方案**：保持 state-reward 形态，但将 `speed_ok * angle_ok` 改为 `(speed_ok + angle_ok) / 2` —— 这样即使 angle 因子为零，speed 因子仍独立贡献梯度，反之亦然。proximity 保持乘法门控（距离远时整个 bonus 归零，这是合理的设计）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- position and progress delta (main learning signal) ----
    px, py = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    d_obs = (px**2 + py**2) ** 0.5
    d_next = (nx**2 + ny**2) ** 0.5
    progress_delta = d_obs - d_next

    # ---- distance penalty ----
    distance_penalty = -0.005 * d_next

    # ---- stability penalty ----
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    stability_penalty = (
        -0.005 * abs(vx)
        - 0.005 * abs(vy)
        - 0.02 * abs(angle)
        - 0.02 * abs(ang_vel)
    )

    # ---- approach quality bonus (state-reward, additive sub-factors) ----
    # KEY CHANGE: replaced speed_ok * angle_ok (product kills all gradients
    # when ANY factor is zero in flat region) with (speed_ok + angle_ok)/2.
    # Each attitude factor now independently contributes gradient, even if
    # the other is saturated at zero.  proximity remains multiplicative as
    # a distance gate — bonus is naturally zero when far from origin.
    APPROACH_DIST = 2.0
    SPEED_THRESH = 2.0
    ANGLE_THRESH = 0.5

    proximity = max(0.0, 1.0 - d_next / APPROACH_DIST)
    speed_ok = max(0.0, 1.0 - (abs(vx) + abs(vy)) / SPEED_THRESH)
    angle_ok = max(0.0, 1.0 - abs(angle) / ANGLE_THRESH)

    # Additive combination of attitude factors → independent gradients
    # Scale 5.0 matches iter-3 best (where product*5.0 gave mean≈2.71)
    approach_bonus = proximity * (speed_ok + angle_ok) / 2.0 * 5.0

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