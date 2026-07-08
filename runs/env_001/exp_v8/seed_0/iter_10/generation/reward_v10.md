## 诊断

**1. Agent 发生了什么？**
Score 240.91，略低于 best 245.28。Episode length 415（合理），0 crash。Agent 已学会着陆但进入平台期。`original_env_reward` 均值 -0.108，ratio -2.18，与 progress_reward 符号相反——存在 misalignment 信号，但得分本身已经不错，说明 agent 学会了我们定义的奖励，只是环境原生信号不同。

**2. 哪个组件是主要原因？**
- `soft_landing_proxy` ratio 7.5, nonzero 52%：信号活跃，但四个因子乘积形式有一个固有问题——当 agent"接近但某一项还差一点"时，乘积把梯度压得很弱。比如 proximity=0.9, speed=0.1, angle=0.9, contact=0.8 → proxy=0.065，梯度主要通过 speed_factor 传播但被其他三个因子衰减到 0.9×0.9×0.8≈0.65 倍。
- `progress_reward` 从 8.0→15.0 后 mean 从 0.028→0.050，但得分微降（245→241），说明瓶颈不在 progress 信号强度。
- `center_bonus` ratio 2.3，正常工作。

**3. 我之前改了什么？**
Iter 9 把 progress scale 从 8.0→15.0。得分没升反微降 → 这个方向无效。本轮不应再动 progress scale。

**策略：** 回退 progress scale 到 8.0（best 的配置），改为修改 soft_landing_proxy 的数学形式——将三因子的乘积改为 `min()`，让梯度始终流向当前最大的瓶颈维度，而非被其他接近-1 的因子衰减。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    cx, cy = float(obs[0]), float(obs[1])
    nx, ny = float(next_obs[0]), float(next_obs[1])

    d_curr = (cx ** 2 + cy ** 2) ** 0.5
    d_next = (nx ** 2 + ny ** 2) ** 0.5

    # 1. 主学习信号：回退到 best 的 scale=8.0
    #    iter 9 把 scale 提到 15.0 但得分微降，说明瓶颈不在 progress 强度
    progress_reward = (d_curr - d_next) * 8.0

    # 2. 软着陆近似信号：数学形式从乘积改为 min() * contact
    #    乘积形式的问题：当三个因子中有一个很差时，整个 proxy≈0，
    #    且梯度被其他因子衰减。min() 让梯度直接流向瓶颈维度。
    vx, vy = float(next_obs[2]), float(next_obs[3])
    speed = (vx ** 2 + vy ** 2) ** 0.5
    angle = abs(float(next_obs[4]))

    proximity_factor = max(0.0, 1.0 - d_next / 0.3)
    speed_factor = max(0.0, 1.0 - speed / 0.5)
    angle_factor = max(0.0, 1.0 - angle / 0.3)
    contact_factor = (
        max(0.0, float(next_obs[6])) * max(0.0, float(next_obs[7]))
    )

    # 用 min 替代乘积：agent 必须同时满足三个条件才能获得信号，
    # 但梯度始终流向当前最差的那个维度（瓶颈），不被其他因子衰减
    approach_quality = min(proximity_factor, speed_factor, angle_factor)
    soft_landing_proxy = approach_quality * contact_factor

    # 3. 中心邻近奖励（保持 best 配置不变）
    center_bonus = 0.2 * max(0.0, 1.0 - d_next / 0.5)

    total_reward = progress_reward + soft_landing_proxy + center_bonus

    components = {
        "progress_reward": progress_reward,
        "soft_landing_proxy": soft_landing_proxy,
        "center_bonus": center_bonus,
    }
    return float(total_reward), components
```