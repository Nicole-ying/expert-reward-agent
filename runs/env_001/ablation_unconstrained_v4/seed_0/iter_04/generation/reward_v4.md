## 分析

### 1. 发生了什么？

**Iteration 2 → Iteration 3 灾难性崩溃：**

- Iter 2（absolute settle_reward）：score=+108, len=743, settle_reward=2.29/step 主导。Agent 能存活并接近平台，但存在"hover and farm"问题。
- Iter 3（settle_delta）：score=-121, len=68。所有20个评估回合均以 failure 终止（early terminal 100%）。负向压力（distance_penalty -33.3, velocity_damping -20.2）合计约-54，正向信号（settle_delta +5.7, progress_delta +3.3）合计仅+9。改善信号太弱，无法克服持续负压。

这是典型的 **state_to_improvement 转化风险**：纯改善奖励在 quality function 不合适时无法维持理想终态。settle_delta 在早期有正向贡献（从差状态改善），但一旦 agent 开始恶化（如加速下坠），delta 转负或归零，负向惩罚主导，agent 迅速 crash。

### 2. 哪个组件最值得干预？

**核心矛盾：负向持续性惩罚过强，正向信号过弱且缺乏维持机制。**

`distance_penalty` 和 `velocity_damping` 的 magnitude_share 共占 83.9%，且 active_rate 100%——它们在全轨迹中持续施压。settle_delta 虽 active_rate 100% 但 magnitude 仅 9.5%。需要：①大幅削减负向惩罚（尤其远处时），②恢复有维持力的正向信号但加防 farming 机制，③增加着陆完成激励。

### 3. 我之前改了什么？

上一轮将 absolute settle_reward 改为 settle_delta，意图防止"hover and farm"，但导致正向信号坍缩。应恢复 absolute settle，但加 **距离门控**（远处不奖励 settle，由 progress 驱动；近处 settle reward 鼓励稳定）+ **着陆 bonus**（严格条件的大额奖励，利用 episode 自然终止防 farming）。

---

**修改方案：**
- 恢复 absolute settle_reward，但乘以距离门 `max(0, 1 - dist/2.0)`，远处归零
- progress_delta 权重 3→10，成为远处主导信号
- distance_penalty 0.5→0.2，减轻持续负压
- velocity_damping 保持 proximity-gated，权重 1.5→0.6
- 新增强着陆 bonus：双腿接触+近零速+近零角+近目标时给予大额奖励，因 episode 随后自然终止，无法 farming
- orientation_penalty 保持 proximity-gated，减小系数
- engine_penalty 保持微量

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Hybrid reward: progress-driven approach + distance-gated settle + landing bonus.
    
    Key changes from failed settle_delta:
    - Restore absolute settle_reward (distance-gated, only near target)
    - Increase progress_delta weight (3→10) as primary approach driver
    - Reduce distance_penalty (0.5→0.2) to ease negative pressure
    - Reduce velocity_damping (1.5→0.6) and orientation penalty
    - Add large landing_bonus for stable terminal state (self-limiting via episode termination)
    """
    px, py, pvx, pvy, pangle, pang_vel, pleft_contact, pright_contact = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft_contact, nright_contact = next_obs

    # --- Distances ---
    prev_distance = (px ** 2 + py ** 2) ** 0.5
    curr_distance = (nx ** 2 + ny ** 2) ** 0.5

    # --- Speeds ---
    curr_speed = (nvx ** 2 + nvy ** 2) ** 0.5

    # --- Proximity gate (for velocity/orientation damping) ---
    k_prox = 7.0
    curr_prox = 1.0 / (1.0 + k_prox * curr_distance)

    # --- Distance gate for settle reward: active only near target ---
    settle_dist_gate = max(0.0, 1.0 - curr_distance / 2.0)  # 1 at target, 0 beyond dist 2.0

    # ============================================================
    # Component A: progress_delta (primary approach guidance, STRONG)
    # ============================================================
    w_progress = 10.0
    progress_reward = w_progress * (prev_distance - curr_distance)

    # ============================================================
    # Component B: distance_penalty (gentle baseline pull)
    # ============================================================
    w_dist = 0.2
    distance_penalty = -w_dist * curr_distance

    # ============================================================
    # Component C: velocity_damping (proximity-gated, reduced)
    # ============================================================
    w_vel = 0.6
    velocity_damping = -w_vel * curr_prox * curr_speed

    # ============================================================
    # Component D: settle_reward (absolute, distance-gated)
    # ============================================================
    # Quality factors: speed, angle, contact
    speed_quality = 1.0 / (1.0 + 5.0 * curr_speed)
    angle_quality = 1.0 / (1.0 + 5.0 * abs(nangle))
    contact_score = 0.5 * (nleft_contact + nright_contact)
    settle_quality = settle_dist_gate * speed_quality * angle_quality * (1.0 + contact_score)
    w_settle = 3.0
    settle_reward = w_settle * settle_quality

    # ============================================================
    # Component E: orientation_penalty (proximity-gated, reduced)
    # ============================================================
    w_orient = 0.3
    w_angvel = 0.1
    orientation_penalty = -w_orient * curr_prox * (nangle ** 2 + w_angvel * nang_vel ** 2)

    # ============================================================
    # Component F: engine_penalty (fuel efficiency)
    # ============================================================
    w_engine = 0.03
    engine_penalty = -w_engine if action != 0 else 0.0

    # ============================================================
    # Component G: landing_bonus (strong terminal incentive)
    # ============================================================
    # Only when truly landed: both legs down, stable, near target
    # Episode naturally terminates soon after, preventing farming
    landing_conditions = (
        nleft_contact > 0.5 and nright_contact > 0.5 and
        curr_distance < 0.3 and curr_speed < 0.2 and abs(nangle) < 0.1
    )
    w_landing = 40.0
    landing_bonus = w_landing if landing_conditions else 0.0

    # --- Total ---
    total_reward = (
        progress_reward
        + distance_penalty
        + velocity_damping
        + settle_reward
        + orientation_penalty
        + engine_penalty
        + landing_bonus
    )

    components = {
        "progress_delta": float(progress_reward),
        "distance_penalty": float(distance_penalty),
        "velocity_damping": float(velocity_damping),
        "settle_reward": float(settle_reward),
        "orientation_penalty": float(orientation_penalty),
        "engine_penalty": float(engine_penalty),
        "landing_bonus": float(landing_bonus),
    }

    return float(total_reward), components
```