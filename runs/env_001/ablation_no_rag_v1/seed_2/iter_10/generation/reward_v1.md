# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1: Materially different approach — a multi-objective proximity reward that directly optimizes
    horizontal/vertical approach + velocity damping near the pad, with a contact-conditioned
    landing proxy. Focuses on continuous gradients instead of multiplicative threshold products.
    """
    
    # ── Extract state ────────────────────────────────────────────────────────
    # Current (for delta computations)
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    left_contact, right_contact = obs[6], obs[7]
    
    # Next state
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # Derived signals
    both_feet_contact = 1.0 if (next_left_contact > 0.5 and next_right_contact > 0.5) else 0.0
    
    # ── Component 1: Multi-objective proximity + velocity damping (main learning signal) ──
    # Penetrate toward target (0,0) in both axes, penalize speed when close
    horizontal_proximity = -abs(next_x_pos)          # closer to 0 is better
    vertical_proximity = -abs(next_y_pos)           # closer to 0 is better (pad height)
    
    # Velocity damping: penalize speed proportional to proximity to target
    horizontal_damping = -abs(next_x_vel) * (1.0 - min(abs(next_x_pos), 1.0))
    vertical_damping = -abs(next_y_vel) * (1.0 - min(abs(next_y_pos), 1.0))
    
    # Combine: main signal drives toward (0,0) while encouraging low speed near target
    approach_reward = 2.0 * horizontal_proximity + 1.0 * vertical_proximity + \
                      3.0 * horizontal_damping + 2.0 * vertical_damping
    
    # ── Component 2: Attitude stability (soft constraint) ─────────────────────
    # Penalize large body angles, encouraging near-upright orientation
    angle_penalty = -abs(next_body_angle)
    
    # ── Component 3: Landing proxy (sparse + continuous) ──────────────────────
    # Only active when both feet are in contact, encourages position/speed within tolerance
    distance_from_center = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    speed_magnitude = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    
    # Continuous bounty: larger when feet contact and close/stationary near (0,0)
    landing_bounty = both_feet_contact * (2.0 - distance_from_center - speed_magnitude)
    landing_bounty = max(0.0, landing_bounty) * 5.0  # scale up meaningful values
    
    # ── Total reward ──────────────────────────────────────────────────────────
    total_reward = approach_reward + angle_penalty + landing_bounty
    
    # ── Components dict (only top-level additive terms) ───────────────────────
    components = {
        'approach_reward': approach_reward,
        'angle_penalty': angle_penalty,
        'landing_bounty': landing_bounty
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 设计理念
从之前失败的组件结构（乘积形式、稀疏bonus、加权和proximity）中吸取教训，本设计转向 **多目标连续优化** 思路：直接编码 "水平接近 + 垂直接近 + 近垫速度阻尼" 作为主信号，不使用任何阈值乘积，确保每步都有梯度。

## 使用的奖励组件

### 1. `approach_reward`（主学习信号）
**角色**: 驱动飞行器向目标垫 (0,0) 移动并在接近时减速。

- **水平/垂直接近**: 用 `-abs(pos)` 直接惩罚偏离中心。这是连续的、每步都有梯度。
- **速度阻尼**: `-abs(vel) * (1 - min(abs(pos), 1.0))` 。当远离目标时，速度惩罚被抑制，允许快速移动；当接近目标时（pos < 1.0），速度惩罚逐步激活，引导减速。这避免了 "到达目标附近但高速飞过" 的常见失败模式。
- **为什么这样设计**: 之前的结构尝试用乘积形式组合多个条件，但乘积在未同时满足时梯度塌缩，或导致稀疏信号。这里使用加法组合，每个轴独立提供梯度，更稳定。

### 2. `angle_penalty`（姿态稳定约束）
**角色**: 抑制 body_angle 过大，鼓励飞行器保持接近直立。

- 这是一个轻量级的惩罚项，权重远小于主信号。避免飞行器在空中过度翻滚。

### 3. `landing_bounty`（着陆完成近似信号）
**角色**: 强化双脚接触的正确着陆行为。

- 仅在双脚都接触时激活 (`both_feet_contact=1.0`)。
- 在此条件下，根据距中心距离和总速度给予连续奖励：越近越静止，奖励越高。
- 使用了 `max(0, landing_bounty)` 确保不产生负奖励（双脚接触但未对齐时不惩罚，因为 `approach_reward` 已经在推动靠近）。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- environment_card 明确声明 `explicit_success_flag_available: false` 和 `explicit_failure_flag_available: false`，且 info 为空字典。
- 无法可靠区分成功终止、失败终止和其他终止，因此不能引入依赖终止信号的分量。

## 留到后续迭代的组件
- **效率/动作代价**: v1 阶段不应让 agent 在还没学会着陆时就被推力惩罚吓住。
- **精确的角速度控制**: 当前仅用静态角度惩罚，未处理角速度，留待后续稳定着陆后优化。
- **复杂门控或课程**: 默认 v1 不用。

## 训练后应观察的 failure mode
1. **低速优先导致的悬停失败**: `approach_reward` 中的速度阻尼项如果过强，可能使 agent 在远离目标处就过早减速而无法到达。应观察 agent 是否在 y_pos ≈ -5 左右就悬停。
2. **双脚接触但未稳定着陆**: `landing_bounty` 可能被 agent 通过 "反复轻触垫子" 来利用。若出现，后续需加入接触持续性或速度阈值。
3. **姿态发动机过度使用**: 如果 agent 频繁触发 action 1/3 导致角速度震荡，`angle_penalty` 可能不足以抑制，需考虑加入角速度惩罚。