## 诊断

### 1. Agent 发生了什么？
Score 从 iter-1 的 -107 跃升到 124，episode length 873（长），说明 agent 学会了靠近目标并保持在那里。但 episode 偏长且 progress_reward 均值仅 0.002，说明 agent 在目标附近"悬停"而非高效着陆——它靠 approach_quality 拿分，progress 信号几乎被淹没。

### 2. 哪个组件是主要原因？
**approach_quality_reward 的 ratio 高达 280**，远超 progress_reward。虽然 proxy 类信号 ratio 天然偏大，但 280 远超合理范围（通常 2~20），说明它完全主宰了学习信号。agent 学到的是"靠近 + 慢速 + 直立"就能拿高分，而非"到达并着陆"。attitude_penalty 太小（ratio -0.21），几乎无作用。

### 3. 我之前改了什么？
Iter-1→2：从 binary-contact-gated `landing_quality_reward` 换成 dense 的 `approach_quality_reward`。Score 从 -107 跃升到 124——这个改动方向是对的，但 proxy 权重太高导致悬停。

### 验证失败分析
回顾上一版代码，最可能的验证失败原因是 **components 与 total_reward 不一致**：
- `total_reward = progress + w_approach * approach_quality + attitude_penalty`
- 但 `components["approach_quality_reward"] = approach_quality`（裸因子，未乘权重）
- 验证器很可能检查 components 各分量之和与 total_reward 的关系

修复：让 `approach_quality_reward` 存储加权后的实际贡献值 `w_approach * approach_quality`，同时适当降低 w_approach 来缓解悬停问题。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current and next states
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_angle = next_obs[4]

    # 1. Core learning signal: progress towards target (0,0)
    dist_current = (x ** 2 + y ** 2) ** 0.5
    dist_next = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = dist_current - dist_next  # positive when approaching

    # 2. Continuous approach quality shaping
    #    Three bounded factors [0,1], product ensures all must be satisfied
    prox_factor = 1.0 / (1.0 + 5.0 * dist_next)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    speed_factor = 1.0 / (1.0 + speed)
    angle_factor = 1.0 / (1.0 + 5.0 * abs(next_angle))
    approach_quality = prox_factor * speed_factor * angle_factor  # [0, 1], dense

    # 3. Orientation penalty (gentle, keeps the craft upright)
    attitude_penalty = -0.01 * abs(next_angle)

    # Combine — w_approach reduced from 0.3 to 0.15 to curb hovering
    # (ratio was 280:1 vs progress, now targeting ~140:1 as first step)
    w_approach = 0.15
    approach_quality_reward = w_approach * approach_quality
    total_reward = progress + approach_quality_reward + attitude_penalty

    components = {
        "progress_reward": progress,
        "approach_quality_reward": approach_quality_reward,
        "attitude_penalty": attitude_penalty,
    }

    return float(total_reward), components
```

**修改说明：**
- **修复验证错误**：`approach_quality_reward` 现在存储 `w_approach * approach_quality`（实际参与 total_reward 的项），而非裸 `approach_quality` 因子。components 各分量现在与 total_reward 公式一致。
- **层次 1 微调**：`w_approach` 从 0.3 降到 0.15，缓解 approach_quality 对 progress 的压倒性优势（280:1 → ~140:1），让 agent 更关注实际前进而非悬停。这是唯一修改的系数。