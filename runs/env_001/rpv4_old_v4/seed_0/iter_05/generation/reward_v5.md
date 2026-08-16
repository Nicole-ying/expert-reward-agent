现在我已掌握全局。让我整理诊断报告并给出修改方案。

---

## 1. 诊断报告

### evidence

**四轮全貌：**

| iter | 核心变化 | score | term | landing_prox ep_sum | landing_prox share | 行为 |
|------|---------|-------|------|---------------------|--------------------|------|
| 1 | contact_reward(死) + 基础信号 | -159.49 | **20/20** | N/A | N/A | 全部着陆但硬着陆，无软着陆引导 |
| 2 | state-Gauss(σ=2.0, w=3.0) | -36.38 | 0/20 | 2704 | **99%** | 中等距离盘旋 exploit |
| 3 | progress-weighted(w=5.0) | -434.87 | 1/19 | 25.5 | 60.5% | 信号崩溃，agent 迷失 |
| 4 | state-Gauss(σ=0.7, w=10.0) | -59.19 | 1/19 | 4731 | **99.5%** | 近距离盘旋，仍不降落 |

**Checkpoint 趋势（iter 4）：**
- landing_proximity_reward 从 cp10%=5.6 猛增至 cp100%=5000——**agent 在训练中学会最大化该奖励**
- 与此同时 terminated 始终仅 1/20——**接近但不降落**
- score cp70%=-68.6 → cp80%=-70.1 → cp90%=-77.5：后期出现退化，说明优化目标与真正着陆目标偏离

**语义打转判断：** "着陆接近度"语义已尝试 3 种形态（宽高斯 → 改善量加权 → 紧高斯），每次都陷入 exploit 或信号崩溃。**不应再在同一语义上修改。**

**缺失语义：** `action_efficiency`（燃料/动作效率惩罚）——专家任务画像中列为条件职责，**从未在任何一轮实现**。同时 iter 1 证明：没有 proximity reward 的情况下 agent 能 20/20 着陆（仅靠 progress + orientation）。问题在于没有信号引导软着陆 + 没有惩罚阻止盘旋。

### behavior_diagnosis

Agent 学会了在着陆垫附近（dist ≈ 0.3-0.5）以低速度、小角度盘旋，持续收集 `landing_proximity_reward`（每步 ~4.84，总共 ~4731）。由于发动机使用完全免费，agent 没有动力结束 episode——盘旋 977 步收集奖励优于冒险着陆（着陆后 episode 终止，奖励流中断）。只有 1/20 集偶然着陆。

### signal_completeness

缺失两个关键职责：
- **`soft_landing_success`**（终端成功奖励）：从未实现。iter 1 的 contact_reward 因触发条件 `left_contact * right_contact` 从未为真而死亡。
- **`action_efficiency`**（燃料/动作效率）：从未实现。当前盘旋行为完全不受惩罚，这是 exploit 的根本原因。

### selected_level

**L2** — 新增 `fuel_penalty` 组件（新语义，新数学形态），配合 `landing_proximity_reward` 权重下调（L1 级别的辅助调整）。两个修改服务于同一语义目标：「打破盘旋 exploit，驱动真正着陆」。

选择理由：`action_efficiency` 是专家画像中明确列出的、从未被尝试的语义方向。它直接攻击 exploit 机制——盘旋需要持续主发动机推力，而燃料惩罚使盘旋变得昂贵。

### selected_intervention

1. **新增 `fuel_penalty`**：按动作类型惩罚发动机使用——无推力(action=0)奖励 0，姿态发动机(action=1/3)轻罚 -0.05，主发动机(action=2)重罚 -0.2。这使得盘旋（需要持续主发动机对抗重力）每步付出代价。
2. **降低 `w_landing`**：从 10.0 → 2.0，将 landing_proximity_reward 从 ~4731 压缩到 ~950，减少其对总奖励的统治力，让燃料惩罚有相对权重。

### falsifiable_hypothesis

盘旋需要持续主发动机推力对抗重力（action=2）。引入燃料惩罚后，每步主发动机使用付出 -0.2 代价，977 步盘旋累计约 -195 的燃料惩罚。同时 proximity reward 被压缩至 ~950（原 ~4731），燃料惩罚的相对权重显著上升。Agent 将学会：**尽快完成着陆以终止 episode，避免累积燃料惩罚** → terminated 率上升、episode 长度下降。

### expected_next_round

- `terminated` 率应从 1/20 上升（预期 ≥ 3/20）
- `len` 应从 977 下降（预期 < 900）
- `fuel_penalty` active_rate 应接近 100%（每步都评估），ep_sum 应为显著负值
- `landing_proximity_reward` magnitude_share 应从 99.5% 下降（预期 < 85%）
- score 可能短期小幅波动（新惩罚引入初期），但应随 terminated 率上升而改善

### main_risk

燃料惩罚可能过度——agent 学会减少发动机使用但不学习如何精确控制，导致无法调整姿态/减速，硬着陆比例上升。这可通过适度权重（-0.2 主发动机、-0.05 姿态发动机）来缓解，确保导航信号仍占主导。

---

## 2. 完整代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v5: Added fuel_penalty to break the hovering exploit — engine usage is no longer free.
        Reduced w_landing from 10.0→2.0 to prevent landing_proximity_reward domination.
        The fuel penalty makes sustained hovering costly, incentivizing the agent to
        complete the landing and terminate the episode.
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

    # 3. Landing proximity reward — tight state-Gaussian
    #    sigma_dist=0.7 ensures reward is near-zero beyond ~1.5 units.
    #    w_landing reduced from 10.0→2.0 to reduce domination and make room
    #    for fuel_penalty to matter.
    speed = (vx ** 2 + vy ** 2) ** 0.5
    sigma_dist = 0.7
    sigma_speed = 0.8
    sigma_angle = 0.3

    dist_factor = 2.718281828 ** (- (dist_curr ** 2) / (sigma_dist ** 2))
    speed_factor = 2.718281828 ** (- (speed ** 2) / (sigma_speed ** 2))
    angle_factor = 2.718281828 ** (- (angle ** 2) / (sigma_angle ** 2))

    w_landing = 2.0
    landing_proximity_reward = w_landing * dist_factor * speed_factor * angle_factor

    # 4. Near-pad velocity damping (hinge / gated penalty)
    dist_thresh = 1.0
    gate = max(0.0, 1.0 - dist_curr / dist_thresh)       # 0 at far, 1 at origin
    w_vel_damp = 0.5
    vel_damping = -w_vel_damp * gate * (abs(vx) + abs(vy))

    # 5. Fuel efficiency penalty — NEW
    #    Penalize engine usage to break the hovering exploit.
    #    action 0 (no engine): zero penalty — gliding is free.
    #    action 2 (main engine): heaviest penalty — maintaining altitude requires this.
    #    action 1/3 (orientation): light penalty — needed for attitude control.
    if action == 0:
        fuel_penalty = 0.0
    elif action == 2:
        fuel_penalty = -0.2
    else:  # action 1 or 3
        fuel_penalty = -0.05

    total_reward = (progress_reward + orientation_penalty +
                    landing_proximity_reward + vel_damping + fuel_penalty)

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_proximity_reward": landing_proximity_reward,
        "vel_damping": vel_damping,
        "fuel_penalty": fuel_penalty
    }
    return float(total_reward), components
```