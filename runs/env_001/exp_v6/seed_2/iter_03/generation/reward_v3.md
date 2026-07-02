## 诊断

### 1. Agent 发生了什么？
Episode 长度 72 步，100% 提前终止，score=-98。这基本等于随机坠落——agent 完全没有学会任何控制。`original_env_reward` 均值 -1.48/步，远超 `generated_reward` 的 0.011/步（不到环境信号的 1%），生成的奖励只是噪音。

### 2. 哪个组件是主要原因？
- **`soft_landing_bonus` nonzero_rate = 0.65%** → 几乎从不触发。`both_legs_contact` 二值门在 agent 还在 crash 的阶段等于把整个组件的梯度完全切断。乘积形式让本就稀少的触发信号更弱。
- **`stability_penalty` ratio = -0.55** → 和 progress 信号几乎等量级，在早期探索阶段惩罚速度会直接抑制学习控制所需的动作。
- **`progress_delta` mean = 0.016** → 信号太弱，被环境惩罚淹没。

### 3. 我之前改了什么？
上一轮把 stability_penalty 和 distance_penalty 降低了 10 倍，并把 landing 从二值改为连续乘积（但保留了 `both_legs_contact` 门）。得分从 -108 升到 -98，方向对但幅度远远不够。问题根因是 `both_legs_contact` 这个二值门让整个 landing 信号在 99.35% 的步数里为 0。

### 修改方案
**层次 2：改数学形式**。聚焦 `soft_landing_bonus`——移除 `both_legs_contact` 二值门，改为纯连续 proximity-gated approach bonus。proximity 本身就是天然的距离门控：远距离时 bonus=0，靠近时自动获得梯度。速度因子和角度因子保留为连续乘积，让 agent 在接近目标区域时获得"慢下来、保持直立"的梯度引导。其他组件完全不动。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- position and progress delta (main learning signal, unchanged) ----
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

    # ---- continuous approach bonus (REMOVED both_legs_contact binary gate) ----
    # proximity provides natural distance-gating: zero when far, gradient when close.
    # speed and angle factors give gradient toward "slow + upright" approach.
    APPROACH_DIST = 2.0      # bonus activates within this radius
    SPEED_THRESH = 2.0       # combined |vx|+|vy| threshold
    ANGLE_THRESH = 0.5       # tilt threshold

    proximity = max(0.0, 1.0 - d_next / APPROACH_DIST)
    speed_ok = max(0.0, 1.0 - (abs(vx) + abs(vy)) / SPEED_THRESH)
    angle_ok = max(0.0, 1.0 - abs(angle) / ANGLE_THRESH)

    # Product of 3 bounded [0,1] factors, scaled so a good approach gives
    # meaningful reward.  No binary gate — gradient flows at all distances < APPROACH_DIST.
    approach_bonus = proximity * speed_ok * angle_ok * 5.0

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

### 预期效果
- `approach_bonus` 的 nonzero_rate 应该从 0.65% 跃升到 30-60%（取决于 agent 在 APPROACH_DIST=2.0 范围内的时间比例）。
- 乘积形式天然提供梯度：靠近一点 → proximity 增大 → bonus 增大，慢一点 → speed_ok 增大 → bonus 增大，直一点 → angle_ok 增大 → bonus 增大。
- 如果下一轮 approach_bonus 的 ratio_to_progress 过高（>3），下一轮只需调系数；如果还是太低（nonzero_rate 仍然 <5%），说明 APPROACH_DIST 太小，需要放宽。