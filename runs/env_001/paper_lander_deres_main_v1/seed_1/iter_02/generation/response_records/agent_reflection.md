# Response Record

## 诊断

### 0. 环境理解
这是一个 2D 着陆器任务：飞行器需从顶部中央飞到目标垫中央，双脚平稳触地、竖直、低速。当前奖励 = `progress_reward`（距离缩减量）+ `landing_quality_bonus`（双脚触地时三因子乘积质量分）。

### 1. Agent 发生了什么？

| 指标 | 值 | 解读 |
|------|-----|------|
| score | 241.4 / max 298.0 | 能着陆，但未到上限 |
| episode_length | 394 | **偏长**（着陆任务理想应在 200 步内） |
| landing_quality_bonus nonzero_rate | 54.19% | 约 213 步双脚着地，在垫上"磨" |
| progress_reward mean | 0.004 | **极弱**，每步几乎无飞行引导 |
| landing_quality_bonus ratio | 89.5x progress | 质量奖励绝对主导 |

**结论**：agent 能着陆但效率低。飞行阶段缺乏有效梯度（progress 太弱），着地后长时间在垫上微调以榨取质量分。不是 crash，不是 exploit，是**徘徊/低效**。

### 2. 哪个组件是主要原因？

- `progress_reward`：mean=0.004，episode_sum=1.27。飞行阶段几乎"盲飞"，agent 完全靠 landing_quality_bonus 的远期信号驱动——这导致下降轨迹低效。
- `landing_quality_bonus`：nonzero_rate=54%（健康，>5%），ratio 虽高（89x）但按原则「proxy ratio 偏高不是 bug」，不应削弱它。

**根因：progress_reward 太弱，飞行阶段无有效梯度。**

### 3. 我之前改了什么？
这是第一轮，没有历史修改。当前代码没有对 progress_reward 做任何缩放。

### 修改方案
**层次 1：改系数。** 将 `progress_reward` 乘以 30，使其从每步 ~0.004 提升到 ~0.12，让飞行阶段有足够梯度引导 agent 快速接近目标。landing_quality_bonus 保持不变（不削弱正在工作的 proxy）。

预期：progress 增强后，agent 会更积极地下降，减少空中徘徊；episode_length 应下降，在更短步数内完成着陆，给质量微调留出合理而非过量的时间。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === Progress reward (main learning signal) ===
    # Scale up by 30x: raw distance delta is ~0.004/step, too weak to guide flight.
    # At 30x, per-step progress ~0.12, providing meaningful gradient during descent
    # while still letting landing quality bonus (~0.37/step active) dominate terminal phase.
    x_prev, y_prev = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]

    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress_reward = 30.0 * (dist_prev - dist_next)  # positive when getting closer

    # === Landing quality bonus (soft proxy for successful landing) ===
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    both_contact = (left_contact > 0.5) and (right_contact > 0.5)

    landing_quality_bonus = 0.0
    if both_contact:
        # position error from center of the pad
        pos_err = dist_next
        # velocity magnitude
        vx, vy = next_obs[2], next_obs[3]
        vel = (vx ** 2 + vy ** 2) ** 0.5
        # absolute body angle
        angle = abs(next_obs[4])

        # exponential-based quality score (max ~1.0 for perfect landing)
        # temperatures control how quickly the score decays with imperfection
        temp_pos = 0.2
        temp_vel = 0.5
        temp_angle = 0.1
        quality = (2.718281828 ** (-pos_err / temp_pos)) * \
                  (2.718281828 ** (-vel / temp_vel)) * \
                  (2.718281828 ** (-angle / temp_angle))
        landing_quality_bonus = quality

    total_reward = progress_reward + landing_quality_bonus

    components = {
        "progress_reward": progress_reward,
        "landing_quality_bonus": landing_quality_bonus
    }
    return float(total_reward), components
```
