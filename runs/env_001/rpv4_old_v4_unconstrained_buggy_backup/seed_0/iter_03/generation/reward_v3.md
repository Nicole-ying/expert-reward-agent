现在我已经掌握了所有关键证据。让我给出诊断和修改方案。

---

## 诊断报告

### Evidence

| 证据来源 | 数据 | 含义 |
|---|---|---|
| iter_1 terminated | 20/20，len=303 | 无 soft_landing_proxy 时，智能体**会终止**（到达某种终态） |
| iter_2 terminated | **0/20**，truncated=20/20，len=1000 | 引入 soft_landing_proxy 后，智能体**永远不终止** |
| iter_2 soft_landing_proxy magnitude_share | **99.3%**，ep_sum=3793 | 该组件完全支配奖励，其他组件沦为噪声 |
| iter_2 soft_landing_proxy checkpoint trend | 69→106→620→2788→…→**3370**，持续增长 | 智能体在持续优化这个 proxy，从 cp30% 起爆发 |
| iter_2 score trend | -2017→-495→**-156(cp40%)**→-228→-249→-82→…→-127 | 得分在 cp40% 见顶后**不再改善**，proxy 却在持续增长 |
| iter_2 progress_reward | ep_sum=0.399, signed_share=0.0% | 智能体几乎没有净位移——不是接近目标，而是在某处悬停 |
| iter_1 progress_reward trend | -3.2→+0.1 | 智能体学会了接近目标（有净正向位移） |

### Behavior Diagnosis

**当前策略学到的行为：在着陆垫上方附近悬停（hovering），永不降落。**

`soft_landing_proxy` 是一个**状态值奖励**（state-value reward）：只要处于"近垫 + 低速 + 直立"状态，每步都能获得约 3-5 的奖励。智能体发现了这个局部最优：与其冒险降落（可能 crash 终止 → 奖励流中断），不如永久悬停收集 proxy 奖励。这解释了：
- 100% truncated（跑满 1000 步），0% terminated
- soft_landing_proxy ep_sum 高达 3793 且持续增长
- 得分在 cp40% 后停滞（proxy 优化与真实任务脱钩）

对比 iter_1：没有状态值奖励，智能体被 `progress_reward`（势能差）驱动接近目标，最终触发环境终止条件。iter_1 的问题是 `contact_reward` 和 `vel_damping` 两个组件死亡（阈值太严），导致接近目标后缺乏软着陆引导。

### 修改方案 — L2：替换主信号形态

**核心思路**：将连续状态值奖励（`soft_landing_proxy`）替换为**接触转移事件奖励**（contact-transition bonus）——只在接触从无到有的瞬间发放奖励，而非对"处于好状态"持续奖励。这消除了悬停的激励，同时保留了软着陆的引导。

同时：
- 提高 `progress_reward` 权重（从 1.0→3.0），让接近目标信号更强
- 新增 `action_penalty`，轻微惩罚发动机使用，促进高效着陆
- 保留 iter_2 的宽门 `vel_damping`（threshold=3.0），正常化其权重

---

## 新奖励函数代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v3: Replace dominant soft_landing_proxy (state-value reward → hovering exploit)
        with contact-transition landing bonus (event-based);
        boost progress weight; add action efficiency penalty.
    """
    # ── Extract signals ──────────────────────────────────────────────
    x_prev, y_prev = obs[0], obs[1]
    x_curr, y_curr = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]

    # ── 1. Progress reward (improvement_delta on Euclidean distance) ─
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    progress = dist_prev - dist_curr                     # >0 when approaching
    w_progress = 3.0                                     # boosted from 1.0
    progress_reward = w_progress * progress

    # ── 2. Orientation penalty (quadratic, unchanged) ────────────────
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # ── 3. Landing contact bonus (transition-based, NOT state-based) ─
    # Count how many legs were touching BEFORE vs AFTER this step
    prev_contacts = (1.0 if obs[6] > 0.5 else 0.0) + (1.0 if obs[7] > 0.5 else 0.0)
    curr_contacts = (1.0 if next_obs[6] > 0.5 else 0.0) + (1.0 if next_obs[7] > 0.5 else 0.0)
    new_contacts = max(0.0, curr_contacts - prev_contacts)   # 0, 1, or 2 newly touching

    # Landing quality: only reward contact when conditions are good
    speed = (vx ** 2 + vy ** 2) ** 0.5
    quality = (
        2.718281828 ** (- (dist_curr ** 2) / (2.0 * 1.5 ** 2)) *
        2.718281828 ** (- (speed ** 2)    / (2.0 * 0.5 ** 2)) *
        2.718281828 ** (- (angle ** 2)    / (2.0 * 0.3 ** 2))
    )

    w_landing = 60.0
    landing_bonus = w_landing * new_contacts * quality

    # ── 4. Near-pad velocity damping (gate from iter_2, kept) ────────
    dist_thresh = 3.0
    gate = max(0.0, 1.0 - dist_curr / dist_thresh)       # 0 far away, 1 at origin
    w_vel_damp = 0.3
    vel_damping = -w_vel_damp * gate * (abs(vx) + abs(vy))

    # ── 5. Action efficiency (mild engine-use penalty) ───────────────
    if action == 0:                                      # no engine
        action_penalty = 0.0
    elif action == 1 or action == 3:                     # orientation engines
        action_penalty = -0.05
    else:                                                # action == 2, main engine
        action_penalty = -0.2

    # ── Assemble ─────────────────────────────────────────────────────
    total_reward = (
        progress_reward +
        orientation_penalty +
        landing_bonus +
        vel_damping +
        action_penalty
    )

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_bonus": landing_bonus,
        "vel_damping": vel_damping,
        "action_penalty": action_penalty
    }
    return float(total_reward), components
```

---

### 设计说明

| 组件 | 与 iter_2 的变化 | 理由 |
|---|---|---|
| `progress_reward` | w: 1.0→**3.0** | 增强接近驱动，补偿移除 proxy 后损失的连续引导 |
| `orientation_penalty` | 不变 | 一直正常工作 |
| `landing_bonus` | **新组件**，替代 `soft_landing_proxy` | 只在接触**从无到有**时发放（transition-based），消除悬停激励；质量因子确保只有好着陆得分 |
| `vel_damping` | 不变 | iter_2 中 gate=3.0 已正常激活(active=100%) |
| `action_penalty` | **新组件** | 轻微抑制无意义发动机使用，促进高效着陆 |

**预期行为转变**：智能体被 `progress_reward` 驱动靠近目标 → 被 `vel_damping` 引导减速 → 接触瞬间获得 `landing_bonus` → 环境终止条件触发 → episode 结束。悬停不再有利可图（没有持续的状态值奖励），而着陆能获得一次性大额奖励（~60-120）。