1. **evidence**: current score=-111.56, all 20 episodes terminate early at mean 68.4 steps (<150, score<-50), distance_penalty dominates at -66.5 mean (78.2% magnitude), landing_proxy active_rate=0.9%（门控几乎不打开）, success_bonus active_rate=0.1%（基本不触发）。上一轮在best基础上加了门控和success_bonus,导致landing_proxy从稠密变稀疏,得分相比best恶化。

2. **behavior_diagnosis**: agent快速坠毁——所有episode在~68步提前终止（非截断），得分范围[-127,-95]。agent在到达目标前就已crash,因为唯一的正向引导信号landing_proxy被门控锁死（需同时满足prox/vel/angle>0.6阈值），velocity_penalty的距离门控在水平距离仍远时对近地高速无约束，agent无法学到受控下降。

3. **signal_completeness**: position_approaching有distance_penalty覆盖;velocity_reduction有距离门控速度惩罚但**缺少近地高度感知**——当|y|很小但|x|仍大时,dist门控≈0,高速坠落不受罚;landing_proxy因门控几乎不可达(active_rate 0.9%),正向引导实质缺失。需要：恢复landing_proxy为始终活跃+让速度惩罚感知离地高度。

4. **selected_level**: Level 2——以best代码为基础（始终活跃的landing_proxy、无门控、无success_bonus），对velocity_penalty组件做数学结构变换：门控从纯距离`1/(1+dist)`变为`max(距离门控, 高度门控)`。这是单一组件的有方向结构变换，不同时调其他组件。

5. **selected_intervention**: 唯一修改velocity_penalty的门控结构。原gate=`1/(1+dist)`仅当agent同时靠近目标水平与垂直位置时才惩罚速度；新gate=`max(1/(1+dist), 1/(1+|y|*5))`使得无论水平位置如何，只要离地高度|y|较小就触发速度惩罚，阻断高速坠地crash路径。

6. **falsifiable_hypothesis**: 高度门控让agent在近地时（|y|小）立刻收到速度惩罚反馈，即使水平距离仍远；这迫使agent在接触地面前减速，同时始终活跃的landing_proxy提供连续正向梯度引导向目标靠近并稳定姿态。两者结合应减少早坠crash，episode延长，得分向0收敛。

7. **expected_next_round**: score应明显改善（负值减小）；episode_length应增加（存活更久）；velocity_penalty的episode_sum_mean幅度可能增大（门控触发更多）；landing_proxy的active_rate应接近100%（不再被门控锁死）；early_terminal比例应下降。

8. **main_risk**: 高度门控可能在下降初期就过度惩罚速度，导致agent下降过慢、步数耗尽而被截断（若环境有时间上限）；另外`|y|*5`的衰减速率可能偏强，若agent初始y较大且需要快速穿越高空区域时可能造成矛盾信号。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D vehicle landing task.

    Components:
    - distance_penalty: encourages moving towards the target platform (position=0,0)
    - velocity_penalty: penalizes high speed when near target OR near ground,
      using max(distance_gate, height_gate) to prevent high-speed ground impacts
    - orientation_penalty: penalizes tilt and angular velocity
    - landing_proxy: always-active sum of bounded factors rewarding simultaneous
      proximity, low speed, upright posture, and ground contact
    """
    # next_obs unpacking
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Euclidean distance to target
    dist = (x**2 + y**2) ** 0.5

    # Component A: distance penalty (core progress signal)
    w_dist = 1.0
    distance_penalty = -w_dist * dist

    # Component B: velocity penalty (gated by distance AND height)
    # Height gate ensures speed is penalized near the ground even when horizontally far
    w_vel = 0.2
    dist_gate = 1.0 / (1.0 + dist)
    height_gate = 1.0 / (1.0 + abs(y) * 5.0)   # strong when |y| < 0.2, moderate at |y| < 1.0
    gate = max(dist_gate, height_gate)          # either proximity triggers velocity penalty
    speed_sq = vx**2 + vy**2
    velocity_penalty = -w_vel * speed_sq * gate

    # Component C: orientation stabilization penalty
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * abs(angle) - w_angvel * abs(ang_vel)

    # Component D: soft landing proxy (always-active sum of bounded factors)
    w_landing = 3.0
    # proximity factor: 1 at dist=0, 0 when dist >= 2.5
    prox_reward = max(0.0, 1.0 - dist / 2.5)
    # speed factor: 1 at speed=0, 0 when speed >= 2.0 m/s
    speed = (vx**2 + vy**2) ** 0.5
    vel_reward = max(0.0, 1.0 - speed / 2.0)
    # angle factor: 1 at angle=0, 0 when |angle| >= 0.5 rad
    angle_reward = max(0.0, 1.0 - abs(angle) / 0.5)
    # contact factor: 0, 0.5, or 1.0 depending on single/double contact
    contact_reward = (left_contact + right_contact) / 2.0
    # average of four independent bounded factors
    landing_proxy = w_landing * (prox_reward + vel_reward + angle_reward + contact_reward) / 4.0

    # Total reward
    total_reward = distance_penalty + velocity_penalty + orientation_penalty + landing_proxy

    # Component dictionary
    components = {
        "distance_penalty": distance_penalty,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "landing_proxy": landing_proxy
    }

    return float(total_reward), components
```