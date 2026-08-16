---

## 1. 诊断报告

### evidence
- **Score**: -159.49，目标 200，差距 359 分。20/20 集均以 `terminated`（settled）结束——agent **能着陆**，但得分极低。
- **contact_reward**：全程 0% active，ep_sum=0.000，在所有 checkpoint（10%→100%）均为 0.000。触发条件 `left_contact * right_contact` 从未在评估中为真。这是一个**完全死亡的组件**，`soft_landing_success` 强制职责缺失。
- **vel_damping**：同样全程 0% active，`dist_thresh=1.0` 的门控从未打开——agent 在评估中从未进入距目标 1.0 单位内，或进入时已立即终止。
- **progress_reward**：magnitude_share=93.2%，主导信号，但 ep_sum 仅 0.104（每步 ≈0.00034）。信号方向对（接近时为正），但量级极弱。
- **orientation_penalty**：功能正常，magnitude_share=6.8%，ep_sum=-0.034。
- **Checkpoint 趋势**：Score 从 cp10% 的 -758 稳定上升至 cp90% 的 -145.8（cp100% 微降至 -155.5），说明 agent 在持续学习，但缺乏关键引导信号导致收敛缓慢。

### behavior_diagnosis
Agent 已学会到达目标着陆垫并触发 settled 终止，但学习效率低、得分极负。核心问题：**缺少任何形式的着陆成功信号**——agent 完成着陆时没有得到正向奖励，导致：(1) 学习缺乏明确的成功梯度，(2) 没有动力优化着陆质量（速度、姿态），(3) 接近目标时的行为仅由微弱的 progress delta 驱动，信号几乎消失。

### signal_completeness
| 职责 | 覆盖组件 | 状态 |
|---|---|---|
| goal_proximity（接近目标） | progress_reward | 方向正确但信号极弱 |
| soft_landing_success（软着陆成功） | contact_reward | **完全死亡，0% active** |
| safe_orientation（姿态稳定） | orientation_penalty | 正常 |
| velocity_damping_near_pad（近垫减速） | vel_damping | 完全死亡，0% active |

缺失最关键信号：**soft_landing_success**。

### selected_level
**L2** — 修改组件的数学形态。contact_reward 完全死亡，调权重（L1）无法修复；必须改变触发/检测机制。同时保持同一语义方向（着陆成功信号），不是 L3 换框架。

### selected_intervention
**替换 `contact_reward` → 新建 `landing_proximity_reward`**：
- **旧逻辑**：`left_contact * right_contact * angle_factor * vel_factor` —— 依赖双足接触标志，评估中从未触发。
- **新逻辑**：使用 **距离到目标 + 速度 + 角度** 三因子高斯乘积作为连续软着陆代理奖励。不依赖接触标志，从远处就提供梯度，越接近理想着陆状态（距离小、速度低、角度正）奖励越高。
- 公式：`w * exp(-dist²/σ_d²) * exp(-speed²/σ_s²) * exp(-angle²/σ_a²)`
- 权重 w=3.0，σ_d=2.0（覆盖较广区域），σ_s=1.0，σ_a=0.3。

### falsifiable_hypothesis
contact_reward 死亡是因为接触标志 `left_contact * right_contact` 在评估中从不激活（可能 agent 在双足同时触地之前已触发 settled 终止，或接触标志在终端步不可见）。改用距离+速度+角度作为软着陆代理后，`landing_proximity_reward` 应该 **active_rate > 50%**，在接近目标时提供正向梯度，加速 agent 学习软着陆行为，推动 score 改善 50–150 分。

### expected_next_round
- `landing_proximity_reward` active_rate > 50%，ep_sum_mean 为显著正值（预计 5–30）
- `progress_reward` 的 magnitude_share 从 93% 下降至 40–70%（新组件分担信号）
- Score 改善至 -100 ~ -10 区间
- Checkpoint 趋势中，新组件从 cp30% 起逐步增长
- 终止率保持 > 80%

### main_risk
若权重过高或 σ 过大，agent 可能学会在目标附近**盘旋刷分而不真正着陆**（hovering exploit）。缓解措施：权重适中（3.0），且 settled 终止条件最终会结束回合，限制刷分窗口。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v2: replaced dead contact_reward with landing_proximity_reward —
        continuous proximity+stability reward that does not depend on contact flags.
    """
    # Extract relevant signals
    x_prev, y_prev = obs[0], obs[1]
    x_curr, y_curr = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]

    # 1. Progress reward (improvement_delta on Euclidean distance)
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    progress = dist_prev - dist_curr                     # positive when approaching
    w_progress = 1.0
    progress_reward = w_progress * progress

    # 2. Orientation penalty (quadratic_penalty)
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # 3. Landing proximity reward (continuous soft-landing proxy)
    #    Rewards being close to target, slow, and upright — no contact flags needed.
    #    Gaussian product peaks at the ideal landing state (dist=0, speed=0, angle=0).
    speed = (vx ** 2 + vy ** 2) ** 0.5
    sigma_dist = 2.0
    sigma_speed = 1.0
    sigma_angle = 0.3

    dist_factor = 2.718281828 ** (- (dist_curr ** 2) / (sigma_dist ** 2))
    speed_factor = 2.718281828 ** (- (speed ** 2) / (sigma_speed ** 2))
    angle_factor = 2.718281828 ** (- (angle ** 2) / (sigma_angle ** 2))

    w_landing = 3.0
    landing_proximity_reward = w_landing * dist_factor * speed_factor * angle_factor

    # 4. Near-pad velocity damping (hinge / gated penalty)
    dist_thresh = 1.0
    gate = max(0.0, 1.0 - dist_curr / dist_thresh)       # 0 at far, 1 at origin
    w_vel_damp = 0.5
    vel_damping = -w_vel_damp * gate * (abs(vx) + abs(vy))

    total_reward = progress_reward + orientation_penalty + landing_proximity_reward + vel_damping

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_proximity_reward": landing_proximity_reward,
        "vel_damping": vel_damping
    }
    return float(total_reward), components
```