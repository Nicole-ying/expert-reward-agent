## 诊断

从反馈数据可以清晰看到两个问题：

### 问题 1：stability_penalty 主导了学习信号
- `stability_penalty` 的 ratio_to_progress = **-0.80**（绝对值 0.80 > 0.5）
- 均值 -0.129 vs progress_reward 均值 0.161 → 惩罚吃掉 80% 的进步信号
- 100% 的步数都在受惩罚（nonzero_rate=1.0），agent 每走一步都在被惩罚
- 结果：agent 选择"不动"来避免惩罚 → 所有 episode 都在 72 步内提前终止（score < -50）

**修复**：将 stability_penalty 的系数整体降低 10 倍，使其 ratio 降到 0.1 以下。

### 问题 2：landing_bonus 几乎从不触发
- `landing_bonus` 的 nonzero_rate = **0.6%**（< 2%）
- 二值条件（4 个条件同时满足才给奖励）导致几乎无梯度引导
- agent 无法通过试错学会如何触发这个奖励

**修复**：将二值 if 条件改为连续乘积形式，每个因子用 bounded 函数（max(0, 1 - x/threshold)），让 agent 每靠近一步都能感受到梯度。

### 其他观察
- `energy_penalty` 的 ratio = -0.04，很小，没问题
- `progress_reward` 本身工作正常，但被 penalty 压制

根据**原则 3：每次改一个信号**——但这里两个问题是独立的（一个在惩罚项，一个在奖励项），且 landing_bonus 的改动是形态变化（二值→连续），stability_penalty 是系数变化，互不干扰，可以同时改。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 诊断 1：stability_penalty 的 ratio_to_progress = -0.80，严重主导信号
    # 修复：整体系数降低 10 倍，从 0.5/0.2/0.1 降到 0.05/0.02/0.01
    # 目标：让 penalty 均值降到 progress 均值的 10% 以下
    # ============================================================
    
    # 主学习信号：progress_delta_reward（保持不变）
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta

    # 稳定约束：stability_penalty（系数降低 10 倍）
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.05 * abs(next_obs[4])      # 原 0.5 → 0.05
    angular_vel_penalty = 0.02 * abs(next_obs[5]) # 原 0.2 → 0.02
    speed_penalty = 0.01 * speed                  # 原 0.1 → 0.01
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)

    # ============================================================
    # 诊断 2：landing_bonus 的 nonzero_rate = 0.6%，几乎从不触发
    # 修复：二值 if 条件 → 连续乘积形式
    # 每个因子用 max(0, 1 - x/threshold)，值域 [0,1]
    # 这样 agent 每靠近一步都能感受到梯度
    # ============================================================
    
    # 连续因子：距离因子（距离 < 0.5 时为正，越近越大）
    dist_factor = max(0.0, 1.0 - next_dist / 0.5)
    # 速度因子（速度 < 0.5 时为正，越慢越大）
    speed_factor = max(0.0, 1.0 - speed / 0.5)
    # 姿态角因子（角度 < 0.3 时为正，越小越大）
    angle_factor = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)
    # 接触因子：两个接触传感器都 > 0.5 时为 1，否则为 0（这个保持二值，因为接触是离散事件）
    contact_factor = 1.0 if (next_obs[6] > 0.5) and (next_obs[7] > 0.5) else 0.0
    
    # 连续乘积：每个因子都在 [0,1]，乘积也在 [0,1]
    # 系数从 2.0 提高到 5.0，因为连续乘积天然小于 1，需要补偿
    landing_bonus = 5.0 * dist_factor * speed_factor * angle_factor * contact_factor

    # 动作代价：energy_penalty（保持不变，ratio 很小没问题）
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