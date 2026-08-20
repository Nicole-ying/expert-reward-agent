# Response Record

现在我有了完整的理解。让我整理诊断和代码。

---

## 诊断报告

### Evidence

| 维度 | iter_1 | iter_2 | 解读 |
|---|---|---|---|
| score | -32.35 | 72.90 | 大幅提升，但距目标200仍有差距 |
| len | 1000 (全截断) | 253 (全终止) | 从永不落地→稳定着陆，质的飞跃 |
| contact_bonus ep_sum | 0.00 (active 0%) | 2980 (active 11.8%) | iter_2 成功激活了着陆信号 |
| contact_bonus magnitude_share | 0% | **99.6%** | 压倒性主导，其他组件几乎是噪声 |
| goal_proximity ep_sum | +3.76 | -7.37 | 从势能差（正）变成距离惩罚（负），但量级极小 |
| velocity_penalty ep_sum | -0.49 | -4.60 | 有所增强但仍微弱 |
| best episodes | — | 270 | **单集可达目标**，说明 reward 框架有能力产出高分 |
| early_terminal | — | 3/20 | 仍有崩溃，一致性不足 |

**Checkpoint 趋势关键发现**：
- cp70%→cp80%：contact_bonus 从 405 暴增至 5895，score 从 -80.9 跃升至 50.2——agent 在此阶段「悟到」着陆
- cp80%→cp100%：contact_bonus 从 5895 降至 2820，但 score 继续升至 80.5——agent 在学习更高效的着陆（更少步数累积 bonus）

### Behavior Diagnosis

**当前策略行为模式**：Agent 已经学会着陆——这是一个关键突破。但它的问题在于 **approach 阶段（~88% 的步数）几乎没有有效梯度引导**。goal_proximity 每步仅贡献约 -0.03，velocity_penalty 约 -0.02，相对于 contact_bonus 的 100/步奖励，这两个信号在 approach 阶段形同虚设。

Agent 本质上是**随机探索直到偶然触发 contact_bonus，然后被巨大的 100/步奖励 spike 锚定**。这解释了为什么：
- 分数方差大（范围 -83~270）：探索路径好坏全凭运气
- 3/20 崩溃：随机路径没能抵达着陆区
- 最佳集可达 270：一旦碰上着陆，奖励足够高

**核心问题不是 contact_bonus 太大，而是 approach 阶段信号太弱**。iter_1 证明了：没有 contact anchor 时，纯 potential-based shaping 无法工作。但 iter_2 已经建立了 contact anchor——这意味着现在可以安全地加强 approach 梯度，而不会导致 agent 迷失。

### 干预方案：L2

**语义修改**：将 `goal_proximity` 从「直接距离惩罚」（始终为负）改为「势能差奖励」（接近时为正，远离时为负）。这是 iter_1 中使用的公式形式，但当时因 contact 从未触发而失败；现在 contact 已可触发，势能差将提供从起点到着陆点的连续正向梯度。

**为什么不用 L1**：仅调权重无法改变「approach 阶段缺乏正向反馈」的根本问题。距离惩罚永远是负的，不能区分「正在接近」和「正在远离」。

**联动微调**（同一语义目标——强化 approach 引导）：
- 略微提高 `velocity_penalty` 权重（0.15→0.20），防止势能差引导出过快下降
- `contact_bonus` 保持 100 不变（已验证有效，不可破坏）
- `orientation_penalty` 保持

---

## 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3: potential-based approach shaping + proven sparse contact bonus
    
    Key change from v2: goal_proximity switched from direct distance penalty
    (always negative, weak gradient) to potential-based improvement reward
    (positive when approaching target, negative when retreating).
    This gives the agent meaningful gradient during the ~88% of steps before contact,
    while the proven contact_bonus remains as the terminal anchor.
    
    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity: potential-based improvement reward ---
    # Rewards reduction in squared distance to target (0,0).
    # delta > 0 means approaching; delta < 0 means retreating.
    # Squared distance gives naturally stronger gradient when far away
    # and gentler when close, encouraging rapid initial approach then precision.
    # Weights moderate to avoid dominating the contact_bonus anchor.
    w_x = 0.3
    w_y = 0.6
    delta_x_sq = x_cur ** 2 - x_nxt ** 2   # positive = approaching horizontally
    delta_y_sq = y_cur ** 2 - y_nxt ** 2   # positive = approaching vertically
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity_damping: stronger near ground ---
    # Quadratic penalty on speed, gated by proximity to pad.
    # Slightly increased weight (0.15→0.20) to counterbalance the stronger
    # approach gradient and discourage high-speed dives.
    w_v = 0.20
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: posture stability ---
    # Penalize tilt and angular velocity to keep the lander upright.
    # Weights unchanged from v2 — these worked adequately.
    w_ang = 0.05
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. safe_contact_bonus: large sparse reward for successful landing ---
    # Unchanged from v2 — relaxed thresholds + large per-step bonus.
    # This is the proven anchor that makes landing the global optimum.
    # Per-step nature rewards staying on the pad until body settles.
    w_contact = 100.0
    near_target = (abs(x_nxt) < 0.5) and (abs(y_nxt) < 0.3)
    both_legs = (left_cont_nxt > 0.5) and (right_cont_nxt > 0.5)
    contact_bonus = w_contact if (near_target and both_legs) else 0.0

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'contact_bonus': contact_bonus
    }

    return total_reward, components
```
