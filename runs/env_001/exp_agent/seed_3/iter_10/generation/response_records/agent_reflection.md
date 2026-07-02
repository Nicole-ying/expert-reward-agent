# Response Record

## 诊断

### 1. Agent 发生了什么？

Episode length = 1000（满时长），score = -2.34。不是 crash（没有 early terminal），而是在**徘徊**。`progress_delta_reward` 均值仅 0.010——delta 信号在 agent 不持续朝原点移动时会正负抵消趋近于零。与此同时 `proximity_bonus` 均值 1.077，是 progress 信号的 **110 倍**，完全主导了总奖励。Agent 学到的策略是"待在原地就能拿 1.0 左右的奖励"，根本没有被驱动走向原点。

### 2. 哪个组件是主因？

**progress_delta_reward**。它的数学形态决定了：old_dist - new_dist 在随机或震荡移动时均值为零，无法提供"你应该去原点"的方向性梯度。相比之下，iter 8 的 `distance_reward = -1.0 * dist`（均值 -0.552）明确告诉 agent "离原点越远惩罚越大"，因此拿到了 94.80 分。

### 3. 我上一轮改了什么？

从 iter 8 → iter 9：把 `distance_reward` 换成了 `progress_delta_reward`。这是错误的方向——`distance_reward` 正在工作（94.80 分），换成 delta 形式后丧失了所有方向信息。**本轮应该回退到 distance_reward。**

搜索确认：`progress_delta_reward` 骨架文档明确指出"目标附近易震荡（delta 正负交替）"，这正是我们看到的——agent 全程震荡，progress 均值为零。

### 修改方案

回退到 iter 8 的 `distance_reward + proximity_bonus + stability_penalty` 结构。这是唯一稳定有效的组合（94.80 分），只改一个信号：progress_delta → distance_reward。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 到原点 (0,0) 的距离
    dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 主信号：负距离——提供指向原点的密集、单调、方向性梯度
    # dist=2 → -2.0, dist=0 → 0.0，每一步都有明确信号
    # 回退 iter 9 的 progress_delta（均值 0.010，无方向信息）
    distance_reward = -1.0 * dist

    # 辅助信号：连续接近奖励——近距离时提供额外吸引力
    # bounded [0, 2]，dist=0 时=2.0，dist=1 时≈0.33
    proximity_bonus = 2.0 / (1.0 + 5.0 * dist)

    # 稳定约束——保持系数不变，单独归因本轮改动
    vel_penalty = 0.1 * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = 0.2 * abs(next_obs[4])
    angvel_penalty = 0.05 * abs(next_obs[5])
    stability_penalty = -vel_penalty - angle_penalty - angvel_penalty

    total_reward = distance_reward + proximity_bonus + stability_penalty

    components = {
        'distance_reward': distance_reward,
        'proximity_bonus': proximity_bonus,
        'stability_penalty': stability_penalty,
        'total_reward': total_reward
    }

    return float(total_reward), components
```
