现在我已掌握完整画面。以下是诊断和修改方案。

---

## 1. 诊断报告

### evidence

| 维度 | iter_1 | iter_2 |
|---|---|---|
| score | -203.53 | -37.03 |
| len | 185.8 | 962.55 |
| terminated | 20/20 (crash) | 4/20, truncated 16/20 |
| landing ep_sum | 0.0 (dead) | 697.44 |
| landing active_rate | 0.0% | 99.6% |
| landing magnitude_share | 0.0% | **93.8%** |
| progress magnitude_share | 26.2% | 0.2% |
| speed_penalty magnitude_share | 44.8% | 5.7% |

**Checkpoint 得分轨迹**：cp10%=-217 → cp50%=-58.6 → cp80%=-5.4 → cp100%=-29.4。在 cp80% 曾接近正值，随后回落——典型的 exploit 后过拟合模式。

**上一轮我的干预**（iter_1 → iter_2）：将 landing 从 product-of-5-factors（需双脚接触×位置×速度×角度同时满足）改为 proximity-gated sum-of-factors（曼哈顿距离 < 1.5 即激活）。landing 从死组件（0%）变成 99.6% 活跃。

### behavior_diagnosis

Agent 学会了一种**投机策略**：悬停在目标垫附近（曼哈顿距离 < 1.5），持续收割 landing 奖励（每步约 +0.725），而 speed_penalty 仅约 -0.044/步。净收益约 +0.68/步，agent 没有动力真正着陆。16/20 集 truncated（跑满 ~1000 步），仅 4/20 成功着陆。

**landing 组件被 exploit**：magnitude_share=93.8%，proximity gate（dist < 1.5）过于宽松，agent 无需脚接触即可获得 landing 奖励。

### signal_completeness

- `approach_target`：progress 信号存在但被淹没（magnitude_share=0.2%），无法有效引导
- `stable_landing`：landing 信号存在但被 exploit——缺少**接触条件**作为诚实门控
- `action_penalty`：缺失（未引入燃料成本）
- 核心缺陷：**landing 缺乏 contact gate**，导致 reward hacking

### selected_level

**L2** — 需要改变 landing 的数学形态（从 proximity-gated 改为 contact-gated），单纯调权重（L1）无法解决根本逻辑缺陷：无论权重多低，只要 proximity > 0 就给 reward，agent 就会 exploit。

### selected_intervention

**目标组件**：`landing`

**修改内容**：将 landing 的门控条件从「曼哈顿距离 proximity > 0」改为「脚接触 contact > 0」：
- `contact = (left_contact + right_contact) / 2.0`（0/0.5/1.0）作为门控和放大器
- 质量因子保留（位置、速度、角度），但仅在接触时计算
- 权重从 3.0 提升到 5.0，补偿稀疏性

其他组件（progress、speed_penalty、angular_vel_penalty）保持不变——一旦 landing 不再淹没它们，progress 和 speed_penalty 自然会恢复相对权重。

### falsifiable_hypothesis

如果 landing 奖励仅在脚接触时发放，agent 无法通过悬停 exploit。为获得正奖励，agent 必须实际接触目标垫。结合已有的 progress + speed_penalty 引导，agent 应学会：接近目标 → 减速 → 双脚接触着陆。

### expected_next_round

- landing active_rate 应从 99.6% 显著下降（预期 5-30%，取决于接触频率）
- landing magnitude_share 应从 93.8% 大幅下降
- progress 和 speed_penalty 的 magnitude_share 应自然上升
- terminated 比例应上升（更多成功着陆，更少 truncation）
- score 应继续改善，向 200 靠近

### main_risk

**landing 再次变死组件**（类似 iter_1）。缓解因素：agent 在 iter_2 中已学会到达目标垫附近（hovering），因此它应该能发现接触。但若接触太难触发，landing 可能再次归零。权重提升到 5.0 部分缓解此风险。

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

    # 4. 着陆奖励：contact-gated + quality-scaled
    #    仅在脚接触目标垫时激活，杜绝悬停 exploit
    contact = (next_left_contact + next_right_contact) / 2.0  # 0 / 0.5 / 1.0

    if contact > 0.0:
        # 各项着陆质量因子（每个在 [0,1]）
        pos_quality = max(0.0, 1.0 - (abs(next_x) + abs(next_y)) / 0.5)
        vel_quality = max(0.0, 1.0 - (abs(next_vx) + abs(next_vy)) / 1.0)
        angle_quality = max(0.0, 1.0 - abs(next_angle) / 0.5)

        # contact 作为放大器：单脚=0.5x，双脚=1.0x
        # 任何质量因子的改善都有连续梯度
        landing_score = contact * (pos_quality + vel_quality + angle_quality) / 3.0
    else:
        landing_score = 0.0

    w_land = 5.0  # 提升权重补偿稀疏性
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