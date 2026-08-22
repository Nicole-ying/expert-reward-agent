# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Four-legged locomotion reward v1.
    Core design hypothesis: focus on height management and foot contact rhythm.
    Forward velocity is deliberately absent from v1 to prevent fast-then-fail collapse.
    """

    # ---------- Extract next observation signals ----------
    # Body height
    body_z = next_obs[0]

    # Body orientation (quaternion)
    quat_w = next_obs[4]  # scalar component
    quat_x = next_obs[1]
    quat_y = next_obs[2]
    quat_z = next_obs[3]

    # Body velocities
    forward_vel = next_obs[13]   # desired forward movement
    lateral_vel = next_obs[14]   # sideways drift
    vertical_vel = next_obs[15]  # bouncing

    # Angular velocities (body rotation)
    roll_rate = next_obs[16]
    pitch_rate = next_obs[17]

    # Foot contact forces (z-component of each foot's contact)
    foot_contacts = next_obs[45:51]    # front-left foot
    foot_contacts_2 = next_obs[63:69]  # front-right foot
    foot_contacts_3 = next_obs[81:87]  # rear-left foot
    foot_contacts_4 = next_obs[99:105] # rear-right foot

    # Foot z-contact force (index 2 in the 6D vector)
    fl_foot_z = foot_contacts[2]
    fr_foot_z = foot_contacts_2[2]
    rl_foot_z = foot_contacts_3[2]
    rr_foot_z = foot_contacts_4[2]

    # ---------- Component 1: Height Management (主学习信号) ----------
    # Role: core driving force — stay standing
    # Mathematical form: symmetric penalty around target height with deadzone
    target_height = 0.6
    height_error = body_z - target_height

    # Symmetric penalty: penalizes both being too low (collapsing) and too high (jumping)
    # Deadzone of +/-0.05 to allow natural body oscillation
    height_deviation = abs(height_error) - 0.05
    if height_deviation < 0.0:
        height_deviation = 0.0

    # Linear penalty beyond deadzone, capped to avoid explosion
    height_penalty = 4.0 * min(height_deviation, 0.4)
    height_management = -height_penalty

    # ---------- Component 2: Foot Contact Regularization (步态引导) ----------
    # Role: gait regularization — encourage rhythmic foot contact patterns
    # Use tanh on contact forces to convert to ~[0,1] contact indicator
    # Then reward having at least two feet on ground (diagonal pairs preferred)

    contact_threshold = 50.0  # minimum contact force to consider "grounded"
    sensitivity = 0.05

    def soft_contact(force, threshold, sens):
        """Smooth indicator: 0 when force << threshold, 1 when force >> threshold."""
        return 0.5 + 0.5 * (2.718281828 ** ((force - threshold) * sens) - 2.718281828 ** (-(force - threshold) * sens)) / (2.718281828 ** ((force - threshold) * sens) + 2.718281828 ** (-(force - threshold) * sens))

    # If we can't compute tanh easily, use a simpler sigmoid-like function
    fl_contact = 1.0 / (1.0 + (2.718281828 ** (-sensitivity * (fl_foot_z - contact_threshold))))
    fr_contact = 1.0 / (1.0 + (2.718281828 ** (-sensitivity * (fr_foot_z - contact_threshold))))
    rl_contact = 1.0 / (1.0 + (2.718281828 ** (-sensitivity * (rl_foot_z - contact_threshold))))
    rr_contact = 1.0 / (1.0 + (2.718281828 ** (-sensitivity * (rr_foot_z - contact_threshold))))

    # Total ground contact (0 = no feet, 4 = all feet on ground)
    total_contact = fl_contact + fr_contact + rl_contact + rr_contact

    # Reward: prefer 2-3 feet on ground (walking), penalize 0-1 (falling) or 4 (standing still)
    # Use Gaussian-like reward centered at 2.5 (dynamic gait)
    contact_target = 2.5
    contact_deviation = abs(total_contact - contact_target)
    contact_reward = 1.5 * (2.718281828 ** (-2.0 * (contact_deviation ** 2)))

    # ---------- Component 3: Body Stability (姿态约束) ----------
    # Role: stability constraint — keep body upright and suppress roll/pitch
    # Compute body "up" vector from quaternion (projection of body z-axis onto world z-axis)
    # For a quaternion (x, y, z, w), the body z-axis in world frame is:
    # world_z_body = 1 - 2*(x^2 + y^2)  (when quaternion represents body-to-world rotation)
    body_up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)

    # Reward closeness to 1.0 (perfectly upright)
    # Penalize deviation from upright posture
    tilt_penalty = 1.5 * (1.0 - body_up_z)

    # Also penalize angular velocity (roll + pitch rate)
    angular_rate_penalty = 0.3 * (abs(roll_rate) + abs(pitch_rate))

    stability_cost = tilt_penalty + angular_rate_penalty

    # ---------- Component 4: Forward Motivation (弱前进信号) ----------
    # Role: weak forward bias — gentle encouragement to move forward
    # Very small weight to avoid fast-then-fail; height+contact are primary
    forward_encouragement = 0.3 * forward_vel

    # ---------- Component 5: Lateral Drift Penalty (侧向抑制) ----------
    # Role: suppress side-to-side sway
    lateral_penalty = 0.6 * abs(lateral_vel)

    # ---------- Assemble total reward ----------
    total_reward = (
        height_management
        + contact_reward
        - stability_cost
        + forward_encouragement
        - lateral_penalty
    )

    # ---------- Components dictionary ----------
    components = {
        'height_management': height_management,
        'contact_reward': contact_reward,
        'stability_cost': -stability_cost,
        'forward_encouragement': forward_encouragement,
        'lateral_penalty': -lateral_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **height_management（高度管理）- 主学习信号**
   - 角色：核心驱动力，告诉agent“保持站立”
   - 数学形态：对称惩罚 `-w * soft_clip(|body_z - target| - deadzone)`
   - 为什么选它为主信号：历史实验显示所有前向速度主导的设计都因高度违规而崩溃（19/20失败），所以v1必须让高度管理成为最强信号

2. **contact_reward（足部接触正则化）- 步态引导**
   - 角色：鼓励形成有节奏的步态（2-3脚着地），既不死站也不空踩
   - 数学形态：sigmoid软接触检测 → 高斯形奖励（中心2.5，惩罚0/1或4脚着地）
   - 这是全新信号，之前设计从未使用过脚部接触信息

3. **stability_cost（身体稳定性）- 姿态约束**
   - 角色：保持躯体直立，抑制翻滚/俯仰
   - 包含两个子项：`tilt_penalty`（基于四元数计算身体竖直投影）和`angular_rate_penalty`（角速度惩罚）

4. **forward_encouragement（弱前进信号）- 前向偏置**
   - 角色：极轻的前进鼓励，权重仅0.3
   - 避免历史设计中forward_velocity占42.5% signed_share导致“不顾姿态向前冲”的问题

5. **lateral_penalty（侧向抑制）- 抑制侧摆**
   - 角色：惩罚左右晃动

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- explicit_success_flag_available = false，无显式成功终止标志
- explicit_failure_flag_available = false，无显式失败终止标志
- 所有终止信号都在高度范围检查中隐式产生，v1通过height_management连续引导替代离散惩罚

## 留到后续迭代的组件

- **energy_penalty**：v1阶段agent应先学会站立和行走，动作效率优化后续加入
- **action_smoothness_penalty**：需要动作历史，目前信号偏向躯体控制，动作平滑后加
- **alive_bonus**：可能诱导hover_or_stand_still，需等agent能稳定前进后再考虑

## 训练后应观察的failure mode

1. **stand_still_collapse**：agent可能找到静态站立（4脚着地+接触奖励≈2.5附近）的局部最优，不前移
   - 缓解：forward_encouragement权重小但持续存在，应引导缓慢前移
   - 如果出现：v2增加前进信号权重或引入progress-based奖励

2. **oscillatory_gait**：接触正则化可能诱导高频抬腿放腿（快速切换2→3→2脚着地）而非稳定行走
   - 缓解：没有显式奖励高频动作，自然动力学应惩罚无效振动
   - 如果出现：v2加入步频约束或基于速度的gate

3. **height_overcorrection**：高度管理的对称惩罚可能导致agent跳跃（body_z>>target也是小惩罚）
   - 缓解：deadzone 0.05允许小幅波动，vertical_vel未单独惩罚但angular_rate会抑制弹跳
   - 如果出现：v2加入vertical_velocity惩罚或改为非对称惩罚（高于target惩罚更轻）
