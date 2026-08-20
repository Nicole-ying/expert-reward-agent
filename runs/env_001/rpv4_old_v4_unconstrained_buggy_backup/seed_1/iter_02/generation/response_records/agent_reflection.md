# Response Record

现在我已掌握全部信息。下面是完整的诊断和修改方案。

---

## 诊断报告

### 1. Evidence

| 数据来源 | 关键发现 |
|---|---|
| **checkpoint trend** | `soft_landing_bonus` ep_sum 从 cp10% 的 1.8 暴涨到 cp80% 的 **256.5**，同时 overall score 在 cp80% 见顶 164.9，随后 cp100% 回落到 122.7 |
| **final eval** | score = **-2.98**，远低于 cp100% 训练得分 122.7 → 训练过程中学会的策略在最终评估中完全崩盘 |
| **final eval 组件** | soft_landing_bonus active_rate 仅 **0.3%**，ep_sum 仅 8.0；而 cp80% 时 active_rate 高达 23.4%，ep_sum 256.5 |
| **termination** | terminated=5/20, truncated=15/20, len=813 → 绝大多数 episode 跑满步数而未成功着陆 |
| **proximity_progress** | 全程稳定在 33–37，active_rate ~99%，这是唯一健康运行的组件 |
| **attitude_penalty** | 全程 -1 到 -2，稳定但力度偏弱 |

### 2. Behavior Diagnosis

**当前策略的行为模式：奖励剥削（reward exploitation），不是安全着陆。**

- `soft_landing_bonus = 5.0 × contact_score × velocity_factor × angle_factor` 是一个**状态型奖励**：只要 agent 处于「双腿接触 + 低速 + 水平」状态，每一步都能领取 ~5.0。
- 训练中 agent 学会了反复触发这一状态（可能是接触→弹起→再接触的循环，或在接触状态下持续停留），导致 cp80% 时该组件贡献 256.5 分，主导了整个 reward。
- 但这一策略在 final eval（不同 seed / 略微不同的 dynamics）下无法复现——active_rate 暴跌到 0.3%，暴露了剥削本质。
- 真正的任务指标（到达垫中心、稳定停靠、双腿接触）没有被学到：15/20 episode 被 truncate，说明 agent 从未学会可靠着陆。

**核心问题：`soft_landing_bonus` 是「占据好状态即可持续获奖」模式，需要改为「状态转移事件」模式。**

### 3. 修改方案

- **层级**：L2 — 改变 `soft_landing_bonus` 的数学形态
- **方向**：将状态型奖励（state-value）改为转移型奖励（transition-based），只在「从非接触到接触」的那一步发放一次性奖励
- **联动调整**：将系数从 5.0 提升到 20.0，因为一次性事件需要足够的信号强度来引导学习
- **不变**：`proximity_progress` 和 `attitude_penalty` 保持原样，它们的行为是健康的

---

## 修改后代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v2 — Fixes soft_landing_bonus exploitation by making it transition-based.

    Changes from v1:
    - soft_landing_bonus: state-value → transition-based (only fires on NEW contact)
    - Coefficient raised 5→20 to compensate for one-shot nature
    - proximity_progress and attitude_penalty unchanged (both healthy)
    """

    # ---------- proximity_progress: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 30.0 * (old_dist - new_dist)

    # ---------- attitude_penalty: keep tilt and spin under control ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- soft_landing_bonus: ONE-TIME reward on transition into safe contact ----------
    # Detect the EVENT: previously NOT both legs contacting → NOW both legs contacting
    was_both_contact = obs[6] * obs[7]          # 1.0 if both were on pad
    is_both_contact = next_obs[6] * next_obs[7] # 1.0 if both are on pad now
    new_contact_event = (1.0 - was_both_contact) * is_both_contact  # 1.0 ONLY on first-contact step

    # Quality gates — soft thresholds preserve gradient for learning
    v_mag = abs(next_obs[2]) + abs(next_obs[3])
    velocity_quality = 1.0 / (1.0 + 5.0 * v_mag)       # → 1.0 at v=0, → 0.17 at v=1
    angle_quality    = 1.0 / (1.0 + 5.0 * abs(angle))  # → 1.0 at θ=0, → 0.17 at |θ|=1

    soft_landing_bonus = 20.0 * new_contact_event * velocity_quality * angle_quality

    # ---------- total reward ----------
    total_reward = proximity_progress + attitude_penalty + soft_landing_bonus

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components
```

---

### 预期效果验证

| 验证维度 | 预期 |
|---|---|
| **soft_landing_bonus 不再被剥削** | ep_sum 上限从 ~256 降到 ~20（每次接触事件最多 20），不能再通过停留累积 |
| **active_rate 回升** | 转移型奖励比状态型更容易触发（不需要持续停留），active_rate 应高于 0.3% |
| **策略不再退化** | 训练和评估之间不应再出现巨大分数鸿沟 |
| **proximity_progress 继续主导** | 作为唯一稳定的稠密信号，引导 agent 接近目标，直到学会着陆 |
