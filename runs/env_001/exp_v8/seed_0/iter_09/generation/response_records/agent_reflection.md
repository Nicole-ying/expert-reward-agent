# Response Record

## 诊断

### 1. Agent 状态：表现良好，正在改进中

Score=245.28，episode_length=374.9，无 crash，无 early termination。从 iter 7（137.61, len=1000）到 iter 8（245.28, len=375），得分 +78%，episode 缩短了 62%——agent 正在学会更快地完成任务。

### 2. 组件分析

| 组件 | ratio | nonzero | 判断 |
|------|-------|---------|------|
| `soft_landing_proxy` | 14.14x | 52% | 健康。proxy 天然高 ratio，52% nonzero 说明四个因子同时满足的频率很高，不是 bug |
| `center_bonus` | 4.15x | 73% | 正常。引导 agent 留在目标附近，是 score 从 137→245 的关键改动 |
| `progress_reward` | 1.00x | 99.9% | **偏弱**。均值仅 0.028，在 total_reward（0.534）中只占 ~5% |

### 3. 上轮改了什么

上次把 `stability_penalty` 换成了 `center_bonus`，得分从 137 跳到 245——这个改动有效。现在保持这个方向，强化主信号。

### 对齐检查

`original_env_reward` ratio=-2.78（负），`progress_reward` 为正。这说明环境在给旅程中的燃料消耗等发负向信号，而我们的 shaped reward 在引导接近目标。这不代表 misaligned——在着陆任务中，旅程中环境 reward 为负（燃料惩罚）但最终完成任务时会给大正向奖励是正常的。abs_mean=1.606 远大于 mean=-0.077，说明有正有负的大幅值事件。

### 修改方案：层次 1（改系数）

`progress_reward` 作为主学习信号，均值只有 0.028，在 total 中仅占 5%。骨架推荐 scale 5~20，当前 8.0 偏保守。将 scale 从 8.0 提升到 15.0，预计 progress 均值升至 ~0.052，ratio 从 14x 降到 ~7.5x。不改 proxy（它在工作）、不改 center_bonus（上次刚加且有效）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    cx, cy = float(obs[0]), float(obs[1])
    nx, ny = float(next_obs[0]), float(next_obs[1])

    d_curr = (cx ** 2 + cy ** 2) ** 0.5
    d_next = (nx ** 2 + ny ** 2) ** 0.5

    # 1. 主学习信号：scale 从 8.0 → 15.0，强化进度引导
    #    骨架推荐 5~20，当前 8.0 下 progress 仅占总 reward ~5%，偏弱
    progress_reward = (d_curr - d_next) * 15.0

    # 2. 软着陆近似信号：连续乘积（保持 best 路线不变）
    vx, vy = float(next_obs[2]), float(next_obs[3])
    speed = (vx ** 2 + vy ** 2) ** 0.5
    angle = abs(float(next_obs[4]))

    proximity_factor = max(0.0, 1.0 - d_next / 0.3)
    speed_factor = max(0.0, 1.0 - speed / 0.5)
    angle_factor = max(0.0, 1.0 - angle / 0.3)
    contact_factor = (
        max(0.0, float(next_obs[6])) * max(0.0, float(next_obs[7]))
    )

    soft_landing_proxy = (
        proximity_factor * speed_factor * angle_factor * contact_factor
    )

    # 3. 中心邻近奖励：保持上次改动（radius=0.5, max=0.2），不改
    center_bonus = 0.2 * max(0.0, 1.0 - d_next / 0.5)

    total_reward = progress_reward + soft_landing_proxy + center_bonus

    components = {
        "progress_reward": progress_reward,
        "soft_landing_proxy": soft_landing_proxy,
        "center_bonus": center_bonus,
    }
    return float(total_reward), components
```
