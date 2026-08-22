现在我已经有了完整的全局视图。让我整理诊断并输出方案。

---

## 1. 诊断报告

### evidence

**六轮全貌（memory + feedback）：**

| iter | 进度信号 | 着陆信号 | score | term | len | 核心问题 |
|------|---------|---------|-------|------|-----|---------|
| 1 | delta (+0.10) | contact(死) | -159 | T20 | 303 | 无着陆引导 |
| 2 | delta (+1.04) | Gaussian σ=2.0 w=3.0 | **-36** | S20 | 1000 | 着陆信号99%主导→盘旋不降 |
| 3 | delta (-1.98) | Gaussian σ=0.7 | -435 | T1/S19 | 967 | 过度收紧→崩溃 |
| 4 | delta (+0.84) | Gaussian w大 | -59 | T1/S19 | 978 | 着陆信号99.5%→再次盘旋 |
| 5 | delta (+1.11) | Gaussian σ=0.7 w=2.0 + fuel | -124 | T20 | 68 | vel_damping 70.5%主导→全撞毁 |
| 6 | **状态值 -distance** | **无** | -119 | T20 | 68 | approach_reward 86.2%主导，无梯度 |

**Iter 6 的致命问题：**
- `approach_reward = -1.0 * distance`：状态值惩罚而非改善量。ep_sum=-66.67，magnitude_share=86.2%，全程覆盖。
- Checkpoint 趋势：cp10%→cp100%，**所有组件完全扁平**——agent 根本没有在学习。
- 原因：`-distance` 只取决于"你在哪"，不取决于"你做了什么"。agent 每一步都收到巨大的位置惩罚，无论动作如何，梯度几乎为零。这就好比每次考试不管答对答错都给个 -70 分——学生无法区分哪些行为更好。

**语义演化追踪（避免打转）：**
- 进度信号：迭代 1-5 用的都是 delta（改善量），迭代 6 突然切换到状态值。**这是第一次出现状态值形态的进度信号，而且是错误的**。改回 delta 不是"打转"，而是纠正一次错误的转向。
- 着陆信号：迭代 2-5 经历了 5 次以上的设计振荡（宽→窄→宽→窄→移除），这条线已经穷尽。不应再碰着陆信号。
- 速度信号：迭代 1-6 经历了 5 次活性切换（死→活→死→活→主导→次要）。当前 iter 6 的 vel_penalty 形态（平方+软门控）是合理的，不需要修改。

### behavior_diagnosis

Agent 在 ~68 步全部撞毁。`approach_reward` 以 86.2% 的 magnitude_share 输出一个巨大的负常数（约 -0.97/step），这个值不随策略改善而变化。其他三个组件（vel_penalty 13.6%、orientation_penalty 0.2%、fuel_penalty 0.1%）完全被淹没。整个奖励函数退化为一个平坦的惩罚场——agent 收到的是噪声而非梯度信号。

### signal_completeness

- **goal_proximity**（强制职责）：当前由 `approach_reward=-distance` 承担，**形态错误**——状态值无法提供行为梯度。
- **soft_landing_success**（强制职责）：缺失。但 iter 2-5 已经证明这个信号极难调校（不是 exploit 就是 crash），本轮不应碰。
- **safe_orientation**：存在但被淹没（0.2% share）。
- **velocity_damping**：存在（13.6% share），形态合理。

核心缺口：**progress 信号需要从状态值改回改善量（delta）**。

### selected_level

**L2** — 改变 `approach_reward` 的数学形态：从状态值 `-distance` 改回改善量 `dist_prev - dist_curr`。

选择理由：当前得分 -119 距离目标 200 差距巨大（>50%），且问题本质是数学形态错误（状态值 vs 改善量），不是权重微调能解决的。这也不是 L3（不需要换框架——delta 进度信号在 iter 1-5 已验证可以提供有效引导）。

### selected_intervention

**目标组件**：`approach_reward` → 重命名为 `progress_reward`

**具体修改**：
1. 使用 `obs` 中的前一位置 `(obs[0], obs[1])` 和 `next_obs` 中的当前位置计算前后距离
2. `progress = dist_prev - dist_curr`（接近目标时为正）
3. `progress_reward = 2.0 * progress`（w=2.0，比 iter 5 的 w=1.0 略高以补偿缺失的着陆信号）
4. 其余三个组件（vel_penalty, orientation_penalty, fuel_penalty）**完全不变**

### falsifiable_hypothesis

将进度信号从状态值改为改善量后：
- Agent 每一步都能感知"靠近"还是"远离"目标——正向引导接近行为
- `progress_reward` 的 magnitude 将大幅下降（从 -66.67 → 估计 +4~+12），不再淹没其他组件
- `vel_penalty` 和 `orientation_penalty` 的相对份额将上升，在近垫区域开始发挥作用
- Agent 应学会靠近目标垫（dist 持续下降），而非在 68 步随机撞毁

如果假设正确：下一轮 score 应有显著改善（从 -119 → 至少进入 -50~-80 区间），且 `progress_reward` 的 ep_sum 为正且随 checkpoint 增长。

### expected_next_round

- `progress_reward` ep_sum：从 -66.67 → 预估 +4~+12（取决于 agent 能接近多少），active_rate 保持 100%，magnitude_share 预估 30-60%
- `vel_penalty` ep_sum：约 -8~-12，magnitude_share 从 13.6% 上升（因为 progress 不再淹没它）
- `orientation_penalty` ep_sum：从 -0.14 → 可能略微上升，active_rate 保持 100%
- `fuel_penalty`：基本不变
- score：预期从 -119 改善至 -50~-80 区间
- Checkpoint 趋势：不应再是扁平线——progress_reward 应随训练增长

### main_risk

进度信号改回 delta 后，agent 可能学会接近目标但无法软着陆（类似 iter 1：接近了但撞毁）。这是因为缺少 soft_landing_success 信号。如果下一轮出现"接近但撞毁"的行为模式（len 略有增长、dist 在终止时较小），则需要引入着陆成功奖励——但那是下一轮的事，本轮先修复最致命的梯度缺失问题。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_prev, y_prev = obs[0], obs[1]
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 1. Progress reward (delta improvement on distance) ──
    #     Positive when approaching the target pad, negative when retreating.
    #     w=2.0 gives stronger gradient than iter-5's w=1.0 to compensate
    #     for the absence of a landing_proximity bonus.
    dist_prev = (x_prev * x_prev + y_prev * y_prev) ** 0.5
    dist_curr = (x * x + y * y) ** 0.5
    progress = dist_prev - dist_curr
    progress_reward = 2.0 * progress

    # ── 2. Soft landing gate (velocity + orientation penalties near pad) ──
    distance = dist_curr
    gate = 1.0 / (1.0 + 3.0 * distance)   # close to 1 when near pad, decays with distance

    vel_penalty = -0.3 * gate * (vx * vx + vy * vy + 0.1 * angular_vel * angular_vel)
    orientation_penalty = -0.2 * gate * (angle * angle)

    # ── 3. Fuel efficiency (light penalty on engine use) ──
    if action == 2:               # main engine
        fuel_penalty = -0.02
    elif action == 1 or action == 3:  # orientation engines
        fuel_penalty = -0.005
    else:                         # no engine
        fuel_penalty = 0.0

    total_reward = progress_reward + vel_penalty + orientation_penalty + fuel_penalty

    components = {
        'progress_reward': progress_reward,
        'vel_penalty': vel_penalty,
        'orientation_penalty': orientation_penalty,
        'fuel_penalty': fuel_penalty,
    }
    return float(total_reward), components
```