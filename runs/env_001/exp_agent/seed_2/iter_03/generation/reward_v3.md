## 诊断

**当前状态分析：**

1. **stability_penalty**：ratio_to_progress = -0.089，已从之前的 -0.80 大幅降低。系数降低 10 倍有效，现在 penalty 均值只有 progress 的 8.9%，不再主导信号。✅ 修好了。

2. **landing_bonus**：nonzero_rate = 0.6%，仍然几乎从不触发。虽然改成了连续乘积形式，但 **contact_factor 是二值的**——它要求两个接触传感器都 > 0.5 才为 1，否则为 0。这意味着只要 agent 还没接触地面，整个乘积就是 0，梯度完全被 contact_factor 的二进制门控杀死。连续乘积的梯度优势被 contact_factor 的 if 条件抵消了。

3. **score = -110.34**，与历史最佳 -108.81 相比没有改善。所有 episode 都在 72 步内提前终止（100% early_terminal），说明 agent 在学会稳定着陆之前就坠毁了。

**修复方案：**

**层次 2：改表达式——去掉 contact_factor 的二进制门控**

contact 传感器是离散事件，在 agent 学会稳定着陆之前几乎不可能触发。正确的做法是：
- **去掉 contact_factor**，让 landing_bonus 只基于距离、速度和姿态角的连续乘积。这样 agent 在接近目标、减速、调整姿态的过程中每一步都能感受到梯度。
- 或者，把 contact 作为**额外的 additive 奖励**而非乘性门控——但考虑到 nonzero_rate 只有 0.6%，先完全去掉 contact 约束，让 agent 先学会"靠近+减速+摆正"。

同时，landing_bonus 的系数从 5.0 提高到 10.0，因为去掉 contact_factor 后乘积值会更大，但我们需要补偿 agent 在接近目标时获得更强的信号。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 诊断 1：stability_penalty 的 ratio_to_progress = -0.089，已正常 ✅
    # 诊断 2：landing_bonus 的 nonzero_rate = 0.6%，仍然几乎不触发
    #   原因：contact_factor 是二进制门控，只要未接触地面乘积就为 0
    #   修复：去掉 contact_factor，只保留距离、速度、姿态角的连续乘积
    #   这样 agent 在靠近目标的过程中每一步都能感受到梯度
    # ============================================================
    
    # 主学习信号：progress_delta_reward（保持不变）
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta

    # 稳定约束：stability_penalty（系数已降低 10 倍，保持）
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.05 * abs(next_obs[4])
    angular_vel_penalty = 0.02 * abs(next_obs[5])
    speed_penalty = 0.01 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)

    # ============================================================
    # 修复：去掉 contact_factor 的二进制门控
    # 连续乘积只基于距离、速度、姿态角，每个因子用 max(0, 1-x/threshold)
    # 这样 agent 在靠近目标、减速、摆正姿态时都能获得梯度
    # 系数从 5.0 提高到 10.0，因为去掉 contact 后信号更纯粹
    # ============================================================
    
    # 距离因子：距离 < 0.5 时为正，越近越大
    dist_factor = max(0.0, 1.0 - next_dist / 0.5)
    # 速度因子：速度 < 0.5 时为正，越慢越大
    speed_factor = max(0.0, 1.0 - speed / 0.5)
    # 姿态角因子：角度 < 0.3 时为正，越小越大
    angle_factor = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)
    
    # 连续乘积：每个因子都在 [0,1]，乘积也在 [0,1]
    # 去掉 contact_factor，让梯度在每一步都能传递
    landing_bonus = 10.0 * dist_factor * speed_factor * angle_factor

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