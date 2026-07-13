# Response Record

## 分析

### 1. Agent 发生了什么？

**核心问题：settle_reward 被 exploit，progress_delta 被淹没，landing_bonus 形成稀疏悬崖。**

证据链：
- `settle_reward` episode_sum_mean=201.65，active_rate=100%，占 signed_share 的 35.7%——agent 在所有步都从中获取正奖励。distance gate `max(0, 1-dist/2.0)` 在 distance<2 的广大区域都开放，agent 学会在目标附近徘徊累积 settle_reward，而非高效着陆。
- `landing_bonus` episode_sum_mean=298，但 active_rate 仅 2.8%——二值条件（dist<0.3 & speed<0.2 & angle<0.1 & both_contacts）形成稀疏悬崖：一步 40 分，其余步 0 分。credit assignment 极难。
- `progress_delta` episode_sum_mean 仅 7.10，signed_share 仅 1.3%——主逼近信号被 settle_reward 和 landing_bonus 淹没。magnitude_share(2.1%) > signed_share(1.3%) 表明 agent 在来回振荡，净进展很小。
- score_range=[-142.53, 285.53] 差异巨大——部分 episode 成功着陆拿到 landing_bonus，其余 crash 大负分。terminated=20/20 说明所有 episode 都触发了终止条件，但质量两极分化。

**结论：settle_reward 过于慷慨（距离阈值太宽、权重太大），agent 在其中找到局部最优（徘徊不前但仍获正奖励）；landing_bonus 太稀疏无法引导学习；progress_delta 太弱无法驱动有效逼近。**

### 2. 哪个组件最值得干预？

**settle_reward + landing_bonus 合并替换为连续 bounded landing_quality。**

理由：
- settle_reward 100% 激活率且 magnitude_share=35.7%，是最主要的问题源
- landing_bonus 2.8% 激活率且单步 40 分，形成稀疏悬崖
- 两者都围绕"着陆质量"这一共同目标，可用一个连续乘积函数替代：`dist_factor * speed_factor * angle_factor * contact_bonus`，每个因子用 `max(0, 1 - value/threshold)` 提供梯度
- 收紧距离阈值到 0.3（原 settle gate 2.0 太宽），确保只在真正着陆阶段激活
- 权重设为 6.0（max 约 12/step），分散到 ~20 步着陆阶段总共约 60-100，避免单步悬崖
- 同时将 progress_delta 权重从 10→12，强化主逼近信号

### 3. 上一轮我改了什么？

Iter 4 在 best（iter 2, score=108）基础上：添加了 landing_bonus(40) + 给 settle_reward 加了 distance gate + 增大了 progress_delta 权重(3→10) + 减小了 distance_penalty/velocity_damping/orientation_penalty。结果是 settle_reward 从 iter2 的 2.29/episode 暴涨到 201.65/episode（因为加了 gate 后反而给了更好 exploit 的结构），landing_bonus 太稀疏，score 从 108 跌到 14。

**本次修改：以当前代码为基础，移除 settle_reward 和 landing_bonus，合并为严格门控的连续 landing_reward，同时微调其他权重。**

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Fixes exploit of settle_reward and sparse landing_bonus cliff:
    - Removes standalone settle_reward (was 100% active, 35.7% magnitude share)
    - Removes binary landing_bonus (2.8% active, 40-point cliff)
    - Replaces both with continuous landing_reward: strict bounded product
      (distance < 0.3, speed < 0.25, angle < 0.15 rad), weight 6
    - Increases progress_delta weight 10→12 as primary approach driver
    """
    px, py, pvx, pvy, pangle, pang_vel, pleft_contact, pright_contact = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft_contact, nright_contact = next_obs

    # --- Distances ---
    prev_distance = (px ** 2 + py ** 2) ** 0.5
    curr_distance = (nx ** 2 + ny ** 2) ** 0.5

    # --- Speed ---
    curr_speed = (nvx ** 2 + nvy ** 2) ** 0.5

    # --- Proximity gate (for damping terms) ---
    k_prox = 7.0
    curr_prox = 1.0 / (1.0 + k_prox * curr_distance)

    # ============================================================
    # Component A: progress_delta (PRIMARY approach driver)
    # ============================================================
    w_progress = 12.0
    progress_reward = w_progress * (prev_distance - curr_distance)

    # ============================================================
    # Component B: distance_penalty (gentle baseline pull)
    # ============================================================
    w_dist = 0.3
    distance_penalty = -w_dist * curr_distance

    # ============================================================
    # Component C: velocity_damping (proximity-gated)
    # ============================================================
    w_vel = 0.5
    velocity_damping = -w_vel * curr_prox * curr_speed

    # ============================================================
    # Component D: orientation_penalty (proximity-gated)
    # ============================================================
    w_orient = 0.3
    w_angvel = 0.1
    orientation_penalty = -w_orient * curr_prox * (nangle ** 2 + w_angvel * nang_vel ** 2)

    # ============================================================
    # Component E: engine_penalty
    # ============================================================
    w_engine = 0.03
    engine_penalty = -w_engine if action != 0 else 0.0

    # ============================================================
    # Component F: landing_reward (continuous bounded product)
    # Replaces exploit-prone settle_reward and sparse landing_bonus.
    # Each factor uses max(0, 1 - value/threshold) for smooth gradient.
    # Strict thresholds ensure activation only in final landing phase.
    # ============================================================
    dist_factor = max(0.0, 1.0 - curr_distance / 0.3)
    speed_factor = max(0.0, 1.0 - curr_speed / 0.25)
    angle_factor = max(0.0, 1.0 - abs(nangle) / 0.15)
    contact_bonus = 0.5 * (nleft_contact + nright_contact)
    landing_quality = dist_factor * speed_factor * angle_factor * (1.0 + contact_bonus)
    w_landing = 6.0
    landing_reward = w_landing * landing_quality

    # --- Total ---
    total_reward = (
        progress_reward
        + distance_penalty
        + velocity_damping
        + orientation_penalty
        + engine_penalty
        + landing_reward
    )

    components = {
        "progress_delta": float(progress_reward),
        "distance_penalty": float(distance_penalty),
        "velocity_damping": float(velocity_damping),
        "orientation_penalty": float(orientation_penalty),
        "engine_penalty": float(engine_penalty),
        "landing_reward": float(landing_reward),
    }

    return float(total_reward), components
```
