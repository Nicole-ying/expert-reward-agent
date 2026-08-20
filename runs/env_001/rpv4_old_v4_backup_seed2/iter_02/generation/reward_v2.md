现在我已经掌握了完整的画面。让我整理诊断并制定修改方案。

---

## 1. 诊断报告

### evidence
- **得分**：-203.53，目标 200。差距巨大（~403 分）。
- **终止模式**：20/20 terminated（全部 crash/失败终止），0 truncated。agent 从未成功着陆。
- **组件状态**：
  - `landing`：**active_rate=0.0%，全程死组件**。product-of-5-factors 结构（双脚接触 × 位置 × 速度 × 角度）太苛刻，agent 从未满足全部条件。
  - `progress`：ep_sum=-0.200，100% 活跃但均值为负，agent 平均每 episode 在远离目标。
  - `speed_penalty`：active_rate=15.5%，agent 仅有 15.5% 的时间进入目标 1.5 距离内——且该区域只有惩罚，无正面奖励。
  - `angular_vel_penalty`：100% 活跃，-0.434 ep_sum，正常工作。
- **环境惩罚主导**：我们组件总和 ~-1.3/episode，但总分 -203.53，说明环境 crash 惩罚贡献 ~-202/episode。
- **Checkpoint 趋势**：得分从 -758→-200 单调改善，agent 在学习避免 crash，但速度很慢，且因无 landing 正信号而陷入瓶颈。

### behavior_diagnosis
agent 的策略行为是：**在目标区域外徘徊，避免进入目标 1.5 距离内的"惩罚区"（speed_penalty），因为没有正面着陆奖励来抵消速度惩罚**。agent 学会减少 crash（得分从 -758 提升到 -200），但因为 `landing` 死组件无法提供着陆梯度，agent 无法学会"进入目标区 → 减速 → 双脚接触 → 稳定着陆"这一完整链条。结果就是 agent 一直在目标附近打转但不真正着陆，最终仍 crash。

### signal_completeness
- **approach_target**：有（progress），但信号偏弱且均值为负。
- **stable_landing**：**缺失**——名义上有 `landing` 组件，但实际从未激活，等于没有。agent 完全不知道该"如何着陆"。
- **action_penalty**：缺失，但当前阶段不需要（agent 还没学会基本着陆）。
- **time_efficiency**：缺失，但次要。

核心缺口：**stable_landing 信号完全失效**。

### selected_level
**L2** — 修改 `landing` 组件的数学形态：从"严格乘积门控（5 个因子全部非零才有奖励）"改为"基于 proximity 门控的连续求和公式，提供部分梯度"。这不是简单调权重（L1），也不是换框架（L3）。

### selected_intervention
**目标组件**：`landing`（着陆奖励）

**具体修改**：
1. **激活条件**：从"双脚接触 × 位置 × 速度 × 角度"的严格乘积 → 改为以 **proximity（距离）** 为激活门控，任何进入目标附近的行为都能获得部分信号。
2. **公式形态**：从 product-of-factors → **sum-of-factors（取平均）**，确保即使某一方面不完美，其他方面的改善也能获得奖励。
3. **contact 角色**：从"必须双脚接触"的硬门控 → **soft bonus（0/0.5/1.0）**，单脚接触也能获得部分奖励。
4. **阈值放宽**：位置阈值 0.3→0.5，速度阈值 0.5→1.0，角度阈值 0.2→0.5。
5. **权重调整**：w_land 2.0 → 3.0，补偿更宽松的公式。

### falsifiable_hypothesis
通过 proximity 门控 + sum-of-factors 公式，`landing` 组件将从死组件变为活跃组件（active_rate > 10%），为 agent 提供"靠近目标 + 减速 + 保持姿态 + 双脚接触"的正梯度。agent 将学会主动进入目标区域并尝试着陆，从而减少 crash 次数，下一轮 score 应有显著改善（预期 > -150）。

### expected_next_round
- `landing` active_rate：0% → >10%
- `landing` ep_sum_mean：0 → 正值（至少 0.5+）
- `speed_penalty` active_rate：15.5% → >20%（agent 更愿意进入目标区域）
- `progress` ep_sum_mean：-0.2 → 接近 0 或正（agent 更多时间在接近目标）
- **score**：-203 → 预期 > -150
- **terminated**：仍然会 crash，但 crash 前的行为应更接近目标区域

### main_risk
proximity 门控可能让 agent 在目标附近徘徊以收集 landing 奖励而不真正完成着陆（hover-exploit）。缓解措施：contact_score 作为乘数，确保无接触时的奖励远小于有接触时；progress 信号也鼓励真正到达 x≈0,y≈0。如果下一轮出现此问题，可收紧 proximity 门控或增加 contact 权重。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前步观测
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    # 下一步观测
    next_x, next_y, next_vx, next_vy, next_angle, next_ang_vel, next_left_contact, next_right_contact = next_obs

    # 1. 主进展：鼓励向目标垫 (0,0) 靠近（距离减小）
    old_dist = (x ** 2 + y ** 2) ** 0.5
    new_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = old_dist - new_dist          # >0 接近，<0 远离
    w_dist = 1.0
    progress_reward = w_dist * progress

    # 2. 靠近目标时的速度惩罚：强制减速，防止高速撞击
    closeness = max(0.0, 1.0 - new_dist / 1.5)    # 距离<1.5时激活，距离0时=1
    speed = abs(next_vx) + abs(next_vy)
    w_speed = 0.5
    speed_penalty = -w_speed * closeness * speed

    # 3. 角速度轻量惩罚：保持姿态稳定
    w_angvel = 0.1
    angular_vel_penalty = -w_angvel * abs(next_ang_vel)

    # 4. 着陆奖励：proximity门控 + sum-of-factors，提供连续梯度
    # 激活条件：进入目标附近（曼哈顿距离 < 1.5）
    proximity = max(0.0, 1.0 - (abs(next_x) + abs(next_y)) / 1.5)

    if proximity > 0.0:
        # 各项着陆质量因子（放松阈值，每个在 [0,1]）
        pos_x_factor = max(0.0, 1.0 - abs(next_x) / 0.5)
        pos_y_factor = max(0.0, 1.0 - abs(next_y) / 0.5)
        vel_factor = max(0.0, 1.0 - (abs(next_vx) + abs(next_vy)) / 1.0)
        angle_factor = max(0.0, 1.0 - abs(next_angle) / 0.5)
        # 接触：0/0.5/1.0，作为 soft bonus
        contact_score = (next_left_contact + next_right_contact) / 2.0

        # sum-of-factors 取平均：任何方面的改善都有梯度
        landing_score = proximity * (
            pos_x_factor + pos_y_factor + vel_factor + angle_factor + contact_score
        ) / 5.0
    else:
        landing_score = 0.0

    w_land = 3.0
    landing_reward = w_land * landing_score

    total_reward = (progress_reward + speed_penalty
                    + angular_vel_penalty + landing_reward)

    components = {
        "progress": progress_reward,
        "speed_penalty": speed_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "landing": landing_reward
    }
    return float(total_reward), components
```