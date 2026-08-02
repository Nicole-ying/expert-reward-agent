# Response Record

1. `evidence`：所有 episode 均以早期终止（terminated, len≈68）结束且外部 score 极低（-117.88），progress_shaping 均值 +1.065 为正但无法阻止失败，angle_hinge 几乎从未激活，action_cost 微弱；subagent 报告确认缺少 early‑termination penalty 与 success attractor。
2. `behavior_diagnosis`：当前策略为快速坠毁或水平出界——agent 为追逐距离缩小而采用激进轨迹，而奖励函数未对致命状态（出界、撞地、翻覆）施加惩罚，导致不安全行为未受抑制。
3. `signal_completeness`：缺失灾难性失败惩罚与成功着陆吸引子，已声明但未使用的观测维度（support contacts, angular_velocity）留有利用空间；当前主要进步信号完备但危险防护缺位。
4. `selected_level`：Level 2 结构变换——基于信号缺口与几乎死亡组件的证据，新增使用未利用观测的危险惩罚组件。
5. `selected_intervention`：新增 `danger_penalty` 组件，检测 `abs(nx)>1.2`、`ny<-0.2`、`abs(nangle)>0.8`、或速度幅值 >5.0 等致命状态，每命中步给予 −1.0 惩罚。
6. `falsifiable_hypothesis`：引入危险惩罚后，策略将学会避开立即致命的状态，episode length 应明显增长，外部 score 提升，且 `danger_penalty` 早期活跃度较高但随训练下降。
7. `expected_next_round`：episode length 均值超过 150，score 改善（例如升至 >-60），`danger_penalty` 的 active_rate 在训练初期较高随后降低，`progress_shaping` 继续提供前进引导。
8. `main_risk`：惩罚阈值设置可能使 agent 在安全区内极度保守而悬停不前，或误判正常机动（如较大的正常速度）导致无效惩罚；也可能出现利用安全边界来回刷分的 proxy 行为。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current observation
    x = float(obs[0])
    y = float(obs[1])
    vx = float(obs[2])
    vy = float(obs[3])
    # obs[4] body_angle – kept for potential logging, not used separately here

    # Next observation
    nx = float(next_obs[0])
    ny = float(next_obs[1])
    nvx = float(next_obs[2])
    nvy = float(next_obs[3])
    nangle = float(next_obs[4])

    # ---------- 1. Main learning signal: potential-based shaping ----------
    dist_obs = (x * x + y * y) ** 0.5
    dist_next = (nx * nx + ny * ny) ** 0.5
    speed_obs = (vx * vx + vy * vy) ** 0.5
    speed_next = (nvx * nvx + nvy * nvy) ** 0.5

    alpha = 0.5
    potential_obs = -(dist_obs + alpha * speed_obs)
    potential_next = -(dist_next + alpha * speed_next)
    progress_shaping = potential_next - potential_obs

    # ---------- 2. Stability constraint: body angle hinge ----------
    angle_threshold = 0.3
    angle_hinge = -0.5 * max(0.0, abs(nangle) - angle_threshold)

    # ---------- 3. Efficiency bonus: action penalty ----------
    action_cost = -0.01 * (0.0 if action == 0 else 1.0)

    # ---------- 4. NEW: danger penalty for fatal states ----------
    danger = False
    # horizontal out-of-bounds (viewport edge)
    if abs(nx) > 1.2:
        danger = True
    # body below landing pad level (crash into ground)
    elif ny < -0.2:
        danger = True
    # extreme tilt (flipped over)
    elif abs(nangle) > 0.8:
        danger = True
    # excessive speed (crashed at high velocity)
    elif (nvx * nvx + nvy * nvy) ** 0.5 > 5.0:
        danger = True

    danger_penalty = -1.0 if danger else 0.0

    total_reward = progress_shaping + angle_hinge + action_cost + danger_penalty

    components = {
        "progress_shaping": progress_shaping,
        "angle_hinge": angle_hinge,
        "action_cost": action_cost,
        "danger_penalty": danger_penalty
    }

    return float(total_reward), components
```
