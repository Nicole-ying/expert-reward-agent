现在让我分析当前情况：

**诊断：**

1. **landing_bonus nonzero_rate = 0.74%** — 仍然几乎不触发。虽然从二值改成了连续乘积，但三个因子相乘导致信号极弱。每个因子都在 [0,1]，乘积后更小，只有三个条件同时满足时才有非零值。

2. **score = -115.02** — 比历史最佳 -108.81 更差，且 100% early terminal。

3. **progress_reward 正常** — mean=0.161，ratio=1.0，是主要学习信号。

4. **stability_penalty 正常** — ratio=-0.09，很弱，没问题。

**问题根因：** 连续乘积 `dist_factor * speed_factor * angle_factor` 仍然太严格。三个因子相乘，只要有一个接近 0，整体就接近 0。这本质上还是"同时满足"的逻辑，只是从二值变成了连续。nonzero_rate 0.74% 说明几乎从不三个条件同时满足。

**修复方案：** 改用**加权和**代替乘积。每个因子独立贡献梯度，agent 可以分别学习靠近、减速、摆正姿态，不需要同时做到。这样 nonzero_rate 会大幅提升。

同时，landing_bonus 的 ratio=0.265 已经不小了（是 progress 的 26.5%），但 nonzero_rate 太低导致它几乎没用。改成加权和后，每个因子独立贡献，信号会更密集。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 诊断：landing_bonus nonzero_rate = 0.74%，仍然几乎不触发
    # 原因：连续乘积 dist_factor * speed_factor * angle_factor 仍然太严格
    #       三个因子相乘，只要一个接近 0，整体就接近 0
    #       本质上还是"同时满足"的逻辑，只是从二值变成了连续
    # 修复：改用加权和代替乘积
    #       每个因子独立贡献梯度，agent 可以分别学习靠近、减速、摆正姿态
    #       不需要同时做到，nonzero_rate 会大幅提升
    # 系数调整：加权和的总系数从 10.0 降到 5.0，因为三个因子独立贡献
    #       每个因子权重：距离 2.0，速度 1.5，姿态角 1.5，总和 5.0
    # ============================================================
    
    # 主学习信号：progress_delta_reward（保持不变）
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta

    # 稳定约束：stability_penalty（保持不变）
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.05 * abs(next_obs[4])
    angular_vel_penalty = 0.02 * abs(next_obs[5])
    speed_penalty = 0.01 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)

    # ============================================================
    # 修复：加权和代替乘积
    # 每个因子用 max(0, 1-x/threshold) 保持 bounded
    # 距离因子：距离 < 0.5 时为正，越近越大
    # 速度因子：速度 < 0.5 时为正，越慢越大
    # 姿态角因子：角度 < 0.3 时为正，越小越大
    # 加权和：每个因子独立贡献，agent 不需要同时满足所有条件
    # ============================================================
    
    dist_factor = max(0.0, 1.0 - next_dist / 0.5)
    speed_factor = max(0.0, 1.0 - speed / 0.5)
    angle_factor = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)
    
    # 加权和：每个因子独立贡献梯度
    # 距离权重 2.0（最重要），速度 1.5，姿态角 1.5
    landing_bonus = 2.0 * dist_factor + 1.5 * speed_factor + 1.5 * angle_factor

    # 动作代价：energy_penalty（保持不变）
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty = -0.05 * engine_use

    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus + energy_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components
```