## 诊断

### 1. Agent 发生了什么？
**坠毁模式。** Episode 长度仅 71 步、100% 提前终止、原始环境得分 -111——agent 完全没学会控制下降，迅速坠地。

### 2. 哪个组件是主因？

| 证据 | 值 | 判断 |
|---|---|---|
| stability_penalty ratio_to_progress | **-3.56** (abs > 0.5 ✗) | 惩罚是主信号的 3.56 倍 |
| progress mean | 0.016 | 正向信号太弱 |
| soft_landing_proxy nonzero_rate | 0.39% | 几乎不触发，等于摆设 |
| total_reward mean | **-0.040** | 每步净负，agent 被推向"不动" |

**根因：稳定性惩罚过重。** 每步净奖励为负（-0.04），agent 收到的信号是"做任何事都会被罚"，学习崩溃。

### 3. 我之前改了什么？
这是第 1 轮，初始设计。三个系数（w_speed、w_angle、w_angvel）均为 0.05，对当前环境的尺度来说太大了。

### 修改方案

**层次 1 — 削弱惩罚系数（10x）。** 将三个稳定性惩罚系数从 0.05 降到 0.005，预期将 penalty ratio 从 -3.56 拉回到 -0.36，使每步净奖励转正。soft_landing_proxy 的稀疏问题先不动——本轮只改系数，下一轮根据效果再定。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 主学习信号：进度差分奖励（接近目标垫中心）
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = dist_obs - dist_next   # 正值表示接近目标

    # 2. 稳定性约束：惩罚高速、大倾角和高角速度
    #    【本轮修改】系数从 0.05 降到 0.005（10x），解决 penalty ratio -3.56 压倒 progress 的问题
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_abs = abs(next_obs[4])
    angular_speed = abs(next_obs[5])

    w_speed = 0.005
    w_angle = 0.005
    w_angvel = 0.005
    stability_penalty = -(w_speed * speed + w_angle * angle_abs + w_angvel * angular_speed)

    # 3. 任务完成代理：软着陆近似奖励
    near_target = (dist_next < 0.2)
    low_speed = (speed < 0.1)
    stable_angle = (angle_abs < 0.1)
    both_contact = (next_obs[6] == 1.0 and next_obs[7] == 1.0)
    soft_landing = 0.2 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # 总奖励
    total_reward = progress_delta + stability_penalty + soft_landing

    components = {
        "progress": progress_delta,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing,
        "total_reward": total_reward
    }

    return float(total_reward), components
```