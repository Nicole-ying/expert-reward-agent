# Response Record

## 诊断

### 发生了什么？
- **100% crash**（10/10 early termination），episode 只有 71.5 步
- **progress_delta_reward 极弱**（mean=0.016），skeleton 建议 scale 5~20，当前只有 1.0
- **stability_penalty 占比过高**（ratio=0.858），几乎抵消全部 progress 信号 → agent 得不到清晰的"哪个方向好"的梯度
- **soft_landing_proxy 几乎不触发**（nonzero_rate=0.46%），二值条件太苛刻，形同虚设

### 根因判断
核心矛盾：progress_delta 在自由落体/坠毁时也是正的（接近地面=接近目标），agent 无法区分"受控下降"和"坠毁"。而唯一能教它区分"好行为"的信号（soft_landing_proxy）是二值门，触发率 0.46%，等于没有梯度。stability_penalty 虽强但只是惩罚晃动，不教 agent 什么叫"好"。

### 本轮行动
**层次 2：改 soft_landing_proxy 的数学形式**——从二值 AND 门转为连续乘积因子。这是当前最致命的瓶颈：agent 没有任何信号告诉它"慢速、竖直、接近目标"才是正确方向。连续形式会给每一步提供梯度，nonzero_rate 预计从 0.46% → ~100%。

只改这一个组件，不动 stability_penalty 系数（留待下轮验证）。

### 设计细节
- `proximity = max(0, 1 - dist/D_max)` — D_max=2.0，覆盖整个下降过程
- `smooth_speed = max(0, 1 - |v|/V_max)` — V_max=3.0，坠毁速度(~2-3)时因子 → 0
- `upright = max(0, 1 - |angle|/A_max)` — A_max=0.5 (~28°)，倾斜超过此角度因子归零
- 三个因子乘积：同时满足才得高分，天然编码 AND 语义但有连续梯度
- 去掉 contact 条件（二值传感器不提供梯度，且 hovering 也比 crash 好）

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- HYPERPARAMETERS ----------
    w_progress = 1.0        # 主学习信号：每步接近目标给予奖励
    w_vel     = 0.01        # 速度惩罚权重（水平+垂直）
    w_angle   = 0.01        # 倾斜角度惩罚权重
    w_angvel  = 0.005       # 角速度惩罚权重
    w_landing = 0.3         # 软着陆连续代理奖励权重

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
