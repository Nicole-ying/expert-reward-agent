# Response Record

## 诊断

### 1. Agent 状态：存活但徘徊
- Episode length = 1000（满），无 crash，无 early terminal。
- 得分 157.7，低于历史最佳 165.3，分布上限 190 — 还有上升空间。
- Agent 已到达目标附近（`soft_landing_proxy` nonzero_rate = 99%），但 progress 信号极弱。

### 2. 根因：landing proxy 量级碾压 progress
| 组件 | ratio_to_progress |
|---|---|
| progress_delta_reward | 1.00（基准） |
| soft_landing_proxy | **6.95** |
| stability_penalty | -1.10 |

`soft_landing_proxy` 均值是 progress 的 7 倍。progress_delta 本身天然小（~0.003/步，因为 agent 已经靠近目标），而 landing proxy 三个因子的乘积在目标附近持续 ~0.023/步。Agent 被 "保持静止+直立" 的奖励主导，缺乏快速接近目标的动力。

上一轮的 `D_gate=1.5` 门控没有解决问题（agent 始终在门控范围内），得分反而从 165 → 158。说明门控不是关键——**权重本身才是**。

### 3. 本轮行动：降低 w_landing
- 这是层次 1（系数调整），只改一个参数。
- 移除无效的 `D_gate`，回归简洁结构。
- `w_landing`: 0.03 → **0.01**，预期 ratio 从 7x 降到 ~2.3x，让 progress 重新成为主导信号。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- HYPERPARAMETERS ----------
    w_progress = 1.0        # 主学习信号：每步接近目标给予奖励
    w_vel     = 0.01        # 速度惩罚权重（水平+垂直）
    w_angle   = 0.01        # 倾斜角度惩罚权重
    w_angvel  = 0.005       # 角速度惩罚权重
    w_landing = 0.01        # 【降低】0.03→0.01，缓解 landing proxy 对 progress 的碾压（ratio 7x→~2.3x）

    # 连续 landing proxy 的阈值（bounded 因子归零点）
    D_max = 2.0             # 距离阈值：超过此距离 proximity 归零
    V_max = 3.0             # 速度阈值：超过此速度 smooth_speed 归零
    A_max = 0.5             # 角度阈值（弧度，~28°）：超过此角度 upright 归零
    # --------------------------------

    # --- Progress toward target ---
    dist_obs  = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress_delta = dist_obs - dist_next
    progress_reward = w_progress * progress_delta

    # --- Stability penalty ---
    vx, vy = next_obs[2], next_obs[3]
    angle  = next_obs[4]
    angvel = next_obs[5]

    abs_v_sum = abs(vx) + abs(vy)
    abs_angle = abs(angle)
    abs_angvel = abs(angvel)

    stability_penalty = -(w_vel * abs_v_sum + w_angle * abs_angle + w_angvel * abs_angvel)

    # --- Soft landing proxy (CONTINUOUS) ---
    # 三个 bounded 因子，每个 ∈ [0, 1]，乘积提供连续梯度
    proximity    = max(0.0, 1.0 - dist_next / D_max)
    smooth_speed = max(0.0, 1.0 - abs_v_sum / V_max)
    upright      = max(0.0, 1.0 - abs_angle / A_max)

    landing_bonus = w_landing * proximity * smooth_speed * upright

    # --- Total reward ---
    total_reward = progress_reward + stability_penalty + landing_bonus

    components = {
        "total_reward": total_reward,
        "progress_delta_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": landing_bonus
    }

    return float(total_reward), components
```
