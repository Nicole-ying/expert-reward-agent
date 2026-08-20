# Response Record

## 诊断

### 问题分析

1. **Agent 在 exploit soft_landing_proxy**：ratio_to_progress = 6.9x，proxy 均值 0.294 vs progress 均值 0.043。`original_env_reward` = -0.315（ratio -7.4），与我们的 reward 符号相反 → **经典 misalignment**：agent 在目标附近盘旋收集 proxy 奖励，而非真正完成任务。

2. **stability_penalty 已死亡**：nonzero_rate 0.35%，magnitude ≈ 0。既无帮助也无伤害。

3. **上一轮改了什么**：给 soft_landing_proxy 加 `0.25 *` 系数，试图把 ratio 从 6.9 压到 ~1.7。但代码验证失败。

### 本轮策略

**Revert + 层次 2 修改**：回到 best 版本（已验证通过），但用数学形态改动替代裸系数。使用 `x / (1+x)` 饱和函数包裹 proxy：自动将输出压缩到 [0, 0.5]，保留低值区域梯度（导数在 x=0 处为 1.0），同时压制高值区域的 exploitable 信号。这是**改数学形式**而非单纯改系数，属于不同的修复方向。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 位置
    cx, cy = float(obs[0]), float(obs[1])
    nx, ny = float(next_obs[0]), float(next_obs[1])

    d_curr = (cx ** 2 + cy ** 2) ** 0.5
    d_next = (nx ** 2 + ny ** 2) ** 0.5

    # 1. 主学习信号：逐步靠近目标（scale=8，符合 skeleton 推荐 5~20）
    progress_reward = (d_curr - d_next) * 8.0

    # 2. 轻量稳定约束
    vx, vy = float(next_obs[2]), float(next_obs[3])
    speed = (vx ** 2 + vy ** 2) ** 0.5
    angle = abs(float(next_obs[4]))
    angular_v = abs(float(next_obs[5]))

    raw_stability_penalty = (
        -0.005 * speed
        - 0.01 * angle
        - 0.005 * angular_v
    )

    # stage weighting: 早期忽略 stability，后期逐渐加入
    # t=0 → late_weight=0（无惩罚）；t=1 → late_weight=1（全惩罚）
    t = training_progress
    late_weight = max(0.0, min(1.0, 2.0 * t - 0.5))
    stability_penalty = raw_stability_penalty * late_weight

    # 3. 软着陆近似信号：连续乘积因子 + x/(1+x) 饱和
    #    相比裸乘积（ratio 6.9x），饱和形式将输出压缩到 [0, 0.5]
    #    保留低值梯度（导数在 0 处≈1），压制高值 exploit 空间
    proximity_factor = max(0.0, 1.0 - d_next / 0.3)
    speed_factor = max(0.0, 1.0 - speed / 0.5)
    angle_factor = max(0.0, 1.0 - angle / 0.3)
    contact_factor = (
        max(0.0, float(next_obs[6])) * max(0.0, float(next_obs[7]))
    )

    raw_proxy = proximity_factor * speed_factor * angle_factor * contact_factor
    # x/(1+x) 饱和：raw_proxy ∈ [0,1] → soft_landing_proxy ∈ [0, 0.5]
    soft_landing_proxy = raw_proxy / (1.0 + raw_proxy)

    # 总奖励
    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward,
    }
    return float(total_reward), components
```
