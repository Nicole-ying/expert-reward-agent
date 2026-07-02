# Response Record

## 诊断

### 1. 这个 agent 发生了什么？
- episode_length=921.6，没有 crash（early_terminal=0%），得分 137 → agent 在徘徊/探索，但没有高效完成任务。
- 得分从 iter 2 的 187.93 持续下降到 iter 3 的 143.84 再到 iter 4 的 137.07。

### 2. 哪个组件是主要原因？
- `soft_landing_continuous`：iter 3 中 ratio_to_progress ≈ 83x（系数 0.5），iter 4 降系数到 0.08 后 ratio 仍 ≈ 13x，nonzero_rate=63%。连续乘积信号频繁触发，奖励"部分满足"的状态，导致 agent 学会"靠近 + 慢一点"而非真正着陆。
- 对比 iter 2 的 binary `soft_landing_proxy`（0.5 when ALL conditions met）：虽然 ratio 也高，但它只在真正着陆姿态时触发，给 agent 一个清晰的"成功悬崖"信号，agent 学得更好（187.93）。

### 3. 我之前改了什么？
- iter 3：binary → continuous（系数 0.5）→ 得分下降
- iter 4：continuous 系数 0.5 → 0.08，收紧 prox_threshold 0.3 → 0.2 → 得分继续下降
- **结论：连续形式本身就是问题，不是系数大小。** 应该回到 binary 结构。

### 验证失败分析
v5 代码在逻辑上可运行（训练已跑出 137 分），验证失败很可能是代码格式问题——`\` 续行符在不同解析器下可能有问题。本轮完全避免 `\`，使用单行表达式。

### 修改方向
回到 iter 2 的 binary 结构（已证明有效），清理代码格式，并做一个小改进：用 `training_progress` 动态调整 bonus，训练早期给更大 bonus 帮助发现着陆区，后期减弱让 progress 信号主导精细调整。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── 1. 提取观察量 ──
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    vel_x, vel_y = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 2. 主学习信号：进度差奖励 ──
    dist_old = (x ** 2 + y ** 2) ** 0.5
    dist_new = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = dist_old - dist_new

    # ── 3. 轻量稳定约束 ──
    stability_penalty = -0.002 * (abs(vel_x) + abs(vel_y) + abs(angle) + abs(angular_vel))

    # ── 4. 二值着陆引导信号（回到 iter 2 的 binary 结构）──
    # 原因：连续乘积形式（iter 3-4）奖励"部分满足"，得分持续下降
    # binary 信号只在全部条件同时满足时触发，提供清晰的"成功悬崖"
    # 用 training_progress 做小幅退火：早期 bonus=0.5 鼓励发现着陆区，后期降到 0.3
    speed = (vel_x ** 2 + vel_y ** 2) ** 0.5

    near_target = dist_new < 0.15
    low_speed = speed < 0.2
    upright = abs(angle) < 0.1
    both_legs_down = (left_contact > 0.5) and (right_contact > 0.5)

    # training_progress: 0.0 → bonus=0.5, 1.0 → bonus=0.3
    bonus_magnitude = 0.5 - 0.2 * training_progress
    soft_landing_proxy = bonus_magnitude if (near_target and low_speed and upright and both_legs_down) else 0.0

    # ── 5. 组合总奖励 ──
    total_reward = progress + stability_penalty + soft_landing_proxy

    components = {
        "progress": progress,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components
```
