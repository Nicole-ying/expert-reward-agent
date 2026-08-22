**1. 诊断报告**

- **evidence**
  - 最终得分 `score=-159.49`，全部为负，且 terminate 100%（无 truncation），说明每个 episode 都以某种失败（坠毁、出界）结束，从未成功着陆。
  - `orientation_penalty` 从 cp10% 的 -2.57 逐步改善到 cp100% 的 -0.033，agent 学会了保持姿态稳定，是良性的。
  - `progress_reward` 从 -3.2 改善至 +0.107，表明 agent 整体在接近目标（dist 减小），但其 `magnitude_share=93.2%`，意味着每一步的距离变化幅度极大（正负交替），该信号噪声很高，可能主导了梯度。
  - `contact_reward` 与 `vel_damping` 的 `active_rate` 全程为 0，是**死组件**：
    - `contact_reward` 生效条件是 `left_contact * right_contact == 1`，即双脚同时着地。agent 从未完成一次安全着陆，因此永远不触发。
    - `vel_damping` 生效条件是 `dist_curr < 1.0`（到原点距离），agent 从未如此接近过目标垫，因此门控始终为 0。
  - 两个死组件没有被删除的历史记录，因为这是第一次迭代。

- **behavior_diagnosis**
  - 当前策略学到的行为模式是：**一边向目标垫靠近，一边保持姿态稳定，但始终无法减速和准确着陆，最终以高速撞击或越界告终**。这解释了得分仍为强负值、episode 全部 terminate、且长度约为 303 步（可能在某个边界终止）的现象。它表明 agent 正在“接近但从不抵达”，缺少接近地面时的减速引导。

- **signal_completeness**
  - 现有信号：
    - ✅ 姿态稳定（orientation_penalty，有效且持续改善）
    - ✅ 全局接近引导（progress_reward，有效但噪声大）
    - ❌ 接近地面/着陆时的速度约束（vel_damping 因门控过严而死亡）
    - ❌ 成功着陆信号（contact_reward 因触发条件苛刻而死亡）
  - 缺失的**关键信号**是：当飞行器下降到离地面较近的高度时，提供“请减速”的局部反馈。目前的 vel_damping 使用到原点距离作为门控，导致除非非常接近垫子中心否则不会激发，从而 agent 在下降全过程中没有任何减速压力。

- **selected_level**：L2
  - 这里不是单纯的权重问题，vel_damping 的数学条件需要修改才能从“死组件”变为有效信号。因此选择 L2 修改其公式形态。

- **selected_intervention**
  - 目标组件：`vel_damping`（将其重定义为“接近地面时的速度惩罚”）
  - 具体修改：
    1. 将门控条件从 `dist_curr`（到原点距离）改为 `y_curr`（当前高度）。假设环境坐标系中地面为 y=0，目标垫在原点。高度变小比距离变小更早触发，因为 x 偏移不影响高度。
    2. 惩罚项从 `abs(vx)+abs(vy)` 改为侧重垂直速度：`abs(vy) + 0.3 * abs(vx)`。
    3. 为平衡信号强度，将 `w_vel_damp` 从 0.5 提升至 1.0，并将 `w_progress` 从 1.0 降低至 0.5，以抑制 progress_reward 的极高噪声幅值，让速度惩罚在训练中获得足够梯度权重。

- **falsifiable_hypothesis**
  - 将门控改为基于高度后，`vel_damping` 将在 agent 下降到 1.0 高度以下时激活（不论 x 位置），因为在坠落过程中高度必然减小。这将首次为 agent 提供“接近地面时必须减速”的信号，预期能显著降低触地速度，从而减少坠毁率，或使 agent 首次实现安全接触，激活 `contact_reward`。最终应导致整体得分上升，并可能观察到 episode 长度因成功着陆而缩短。

- **expected_next_round**
  - `vel_damping` 组件的 active_rate 将 >0，并随着训练逐渐增大（下降中触发）。agent 将学会在低空减缓垂直速度，可能首次出现成功着陆事件（contact 激活），score 将从 -159 明显改善，最佳得分可能首次转为正值。

---

**2. 修改后的代码**

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2: 修复死组件 vel_damping——改为基于高度的速度惩罚，让 agent 在接近地面时减速，
    同时降低 progress 权重以抑制其噪声幅度。
    contact_reward 暂时保留原样，等待 vel_damping 激活后看能否自然触发。
    """
    # 提取信号
    x_prev, y_prev = obs[0], obs[1]
    x_curr, y_curr = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Progress reward (improvement_delta on Euclidean distance) —— 降权
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    progress = dist_prev - dist_curr                     # 正值 = 接近
    w_progress = 0.5   # 从 1.0 降低
    progress_reward = w_progress * progress

    # 2. Orientation penalty (保持不变)
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # 3. Soft contact reward (保持不变，等待着陆出现)
    contact_flag = left_contact * right_contact
    sigma_angle = 0.1
    sigma_vel = 0.2
    angle_factor = 2.718281828 ** (- (angle ** 2) / (sigma_angle ** 2))
    vel_factor = 2.718281828 ** (- (vx ** 2 + vy ** 2) / (sigma_vel ** 2))
    soft_contact = contact_flag * angle_factor * vel_factor
    w_contact = 2.0
    contact_reward = w_contact * soft_contact

    # 4. 接近地面速度惩罚 (重写 vel_damping) —— 基于高度，不再是全局距离
    altitude = y_curr                   # 假设地面 y=0
    alt_thresh = 1.0                    # 高度 < 1.0 时开始惩罚
    gate_alt = max(0.0, 1.0 - altitude / alt_thresh)   # 0 在高处，1 在地面
    w_vel_damp = 1.0                    # 提升权重以增强信号
    # 主要惩罚垂直速度，也对横向速度适度惩罚
    vel_damping = -w_vel_damp * gate_alt * (abs(vy) + 0.3 * abs(vx))

    total_reward = progress_reward + orientation_penalty + contact_reward + vel_damping

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward,
        "vel_damping": vel_damping
    }
    return float(total_reward), components
```