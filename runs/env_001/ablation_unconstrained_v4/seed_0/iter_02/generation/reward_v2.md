## 分析

### 1. 这个 agent 发生了什么？

**核心现象：全部 20 个 episode 都以 crash 终止（terminated=20/20），平均回合长度仅 68 步，且引擎使用率极低（2.8%）。**

从 reward 组成看：
- `progress_to_goal`（距离惩罚）占 95.2% 的 magnitude share，episode_sum_mean = -66.54。avg steps = 68.35，所以**平均每步距离 ≈ 0.97**。这说明 agent **已经能稳定靠近目标**（距离很小），但随后 crash。
- `successful_settle_proxy` 仅 3.1% active rate，说明**乘积形式（proximity × velocity × angle × contact）过于严苛**——agent 即使距离近、速度低、角度好，只要任一因子不完美，整个 settle 奖励就归零。这本质上是稀疏奖励，无法提供降落过程中的渐进式学习信号。
- `orientation_penalty` 虽然 100% active，但 magnitude 极小（-1.23），被距离惩罚完全淹没。
- `engine_efficiency` 仅 2.8% active rate，agent 几乎不执行动作——它在靠近目标后**无力减速/调整姿态**，最终撞毁。

**结论：agent 学会了靠近目标，但无法完成软着陆。乘积形式的 settle proxy 是稀疏信号，距离惩罚主导但无法教会减速。缺失的核心信号是「接近目标时的速度抑制」。**

### 2. 哪个组件最值得干预？

**successful_settle_proxy** 是问题核心。当前是四因子乘积（proximity × velocity × angle × contact），需要同时满足所有条件才产生非零奖励。应改为**独立因子求和**，每个因子单独提供渐进式奖励，让 agent 在部分达标时也能获得学习信号。

同时需新增**速度阻尼项**（proximity-gated speed penalty），在靠近目标时惩罚高速，直接引导减速行为。这是 crash 问题的直接药方。

### 3. 上一轮做了什么？

这是第一轮（iter 1），new_best，无历史修改。当前得分 -121.66，所有 episode crash。本轮需从零开始改进。

### 修改方案

1. **settle_proxy：乘积 → 独立求和**，每个因子（proximity, velocity, angle, contact）独立计奖，降低稀疏性
2. **新增 velocity_damping**：proximity-gated 速度惩罚，接近目标时强，远离时弱
3. **降低 distance_penalty 权重**：从 1.0 降至 0.3，防止它淹没速度阻尼和 settle 信号
4. **降低 settle 系数**：由于从乘积变为求和，单因子奖励值会变大，适当调整权重
5. **微调 orientation_penalty**：降低 gate 衰减速度，让它在中等距离也有适度作用

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander navigation to a target platform.
    Key changes from previous version:
      - settle_proxy: product -> sum of independent factors (denser signal)
      - new velocity_damping: proximity-gated speed penalty to prevent crash
      - reduced distance penalty weight to avoid dominating other signals
    """
    # Previous state (for delta calculations)
    px, py, pvx, pvy, pangle, pang_vel, pleft_contact, pright_contact = obs
    # Next state (result of action)
    nx, ny, nvx, nvy, nangle, nang_vel, nleft_contact, nright_contact = next_obs

    # --- Shared helpers ---
    prev_distance = (px ** 2 + py ** 2) ** 0.5
    curr_distance = (nx ** 2 + ny ** 2) ** 0.5
    speed = (nvx ** 2 + nvy ** 2) ** 0.5

    # proximity gate: ~1 when close to target, decays with distance
    k_prox = 5.0
    proximity_gate = 1.0 / (1.0 + k_prox * curr_distance)

    # --- Component A: progress_to_goal (delta-distance) ---
    # Reward moving toward target, penalize moving away.
    # Positive when distance decreases.
    w_progress = 1.0
    progress_reward = w_progress * (prev_distance - curr_distance)

    # --- Component B: distance_penalty (small steady pull toward target) ---
    # Prevents loitering far away; kept small to not dominate.
    w_dist = 0.3
    distance_penalty = -w_dist * curr_distance

    # --- Component C: velocity_damping (proximity-gated) ---
    # Penalise high speed, scaled by proximity: strong near target, weak far away.
    # This is the key signal to prevent crash landings.
    w_vel = 1.5
    velocity_damping = -w_vel * proximity_gate * speed

    # --- Component D: settle_reward (sum of independent factors) ---
    # Each factor rewards one aspect of a good landing.
    # Using SUM instead of PRODUCT so partial progress gives partial reward.
    velocity_factor = 1.0 / (1.0 + 3.0 * speed)       # ~1 when slow
    angle_factor = 1.0 / (1.0 + 3.0 * abs(nangle))    # ~1 when upright
    contact_factor = 0.5 * (nleft_contact + nright_contact)  # ~1 when both legs down

    # Gate by proximity: settle bonuses only matter when near the platform
    w_settle = 2.0
    settle_reward = w_settle * proximity_gate * (velocity_factor + angle_factor + contact_factor)

    # --- Component E: orientation_penalty (proximity-gated) ---
    # Penalise tilt and rotation, stronger near target.
    k_gate = 5.0
    orient_gate = 1.0 / (1.0 + k_gate * curr_distance)
    w_orient = 0.5
    w_angvel = 0.15
    orientation_penalty = -w_orient * orient_gate * (nangle ** 2 + w_angvel * nang_vel ** 2)

    # --- Component F: engine_efficiency ---
    # Small penalty for any engine use to encourage fuel efficiency.
    w_engine = 0.05
    engine_penalty = -w_engine if action != 0 else 0.0

    # --- Total ---
    total_reward = (
        progress_reward
        + distance_penalty
        + velocity_damping
        + settle_reward
        + orientation_penalty
        + engine_penalty
    )

    components = {
        "progress_delta": float(progress_reward),
        "distance_penalty": float(distance_penalty),
        "velocity_damping": float(velocity_damping),
        "settle_reward": float(settle_reward),
        "orientation_penalty": float(orientation_penalty),
        "engine_penalty": float(engine_penalty),
    }

    return float(total_reward), components
```