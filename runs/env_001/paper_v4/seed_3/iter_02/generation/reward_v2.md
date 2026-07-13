`evidence`：当前策略在约78步内全部终止（crash），proximity组件绝对值占magnitude_share的92.7%，且为无界二次型，contact_bonus激活率仅2.7%，episode_sum_mean仅0.35，无法提供有效正向引导。

`behavior_diagnosis`：agent很可能在重力作用下快速坠落并撞击地面，从未学会控制下降——二次距离惩罚在远离目标时产生极大负值，整个奖励空间被负信号淹没，agent学到的最优策略是尽快终止episode以截断累积惩罚。

`signal_completeness`：当前缺少“向目标靠近”的正向过程反馈，proximity只惩罚“远离目标的状态”而不奖励“接近目标的行为”，导致信用分配困难；速度约束和姿态约束形态合理但被主导惩罚压制；接触奖励因门控过严和绝对尺度太小而几乎无法参与学习。

`selected_level`：Level 2 — `state_to_improvement`。证据直接否定了无界二次状态惩罚的数学形态（极端值支配、无正向引导、episode截断策略），单纯降系数（Level 1）不能解决“缺少接近过程正向信号”的结构缺失。

`selected_intervention`：将proximity组件从绝对二次距离惩罚 `-2.0 * dist_sq` 替换为基于势能的进度奖励 `progress_weight * (prev_dist - next_dist)`，其中prev_dist从obs计算、next_dist从next_obs计算，progress_weight=5.0；其余三个组件（velocity_penalty、orientation_penalty、contact_bonus）的系数和形态完全不变。

`falsifiable_hypothesis`：进度奖励为每一步向目标靠近提供即时正向反馈，解决了远距离时信用分配坍塌的问题；agent应先学会朝目标下降（progress为正），然后在接近目标时被速度门控惩罚迫使减速，最终通过接触奖励学会双腿着陆；episode_length应显著增长，terminated率不变但crash类型应从“远距离随机坠落”变为“着陆尝试失败”。

`expected_next_round`：progress组件episode_sum_mean为正且承担主要正向信号角色；velocity_penalty在接近目标时激活并约束着陆速度；episode平均长度超过150步；score_range上限提升；contact_bonus激活率上升（因agent确实到达目标附近并尝试着陆）；signed_share中正向占比不再是0.2%级别。

`main_risk`：进度奖励可能被重力“免费”获取——agent放任自由落体即可持续获得progress_reward，直到临近地面才被velocity_penalty约束，若velocity_penalty门控过渡不够平滑或权重不足，agent可能学会“先赚progress再硬着陆”的局部最优，导致score改善但terminated率不下降。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function with progress-based guidance toward target.
    Components:
    - progress: rewards reducing distance to target center (0,0)
    - velocity_penalty: speed penalty gated by proximity to target
    - orientation_penalty: penalty for tilt and angular velocity
    - contact_bonus: small reward for simultaneous two-leg contact, scaled by proximity
    """
    # Current state
    x_pos, y_pos = obs[0], obs[1]
    # Next state
    nx_pos, ny_pos = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Distances to target
    prev_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (nx_pos ** 2 + ny_pos ** 2) ** 0.5

    # 1. Progress reward: positive for moving toward target, negative for moving away
    progress_weight = 5.0
    progress_reward = progress_weight * (prev_dist - next_dist)

    # 2. Velocity penalty – active only when close to target
    proximity_factor = 1.0 / (1.0 + 10.0 * next_dist)
    vel_weight = 1.0
    velocity_penalty = -vel_weight * proximity_factor * (nx_vel ** 2 + ny_vel ** 2)

    # 3. Orientation stability – keep body upright and avoid spinning
    orient_weight = 0.5
    orientation_penalty = -orient_weight * (n_angle ** 2 + 0.2 * n_angvel ** 2)

    # 4. Dual leg contact – small bonus for proper landing stance, gated by proximity
    contact_weight = 0.5
    contact_product = left_contact * right_contact
    contact_bonus = contact_weight * contact_product * proximity_factor

    total_reward = progress_reward + velocity_penalty + orientation_penalty + contact_bonus

    components = {
        'progress': float(progress_reward),
        'velocity_penalty': float(velocity_penalty),
        'orientation_penalty': float(orientation_penalty),
        'contact_bonus': float(contact_bonus)
    }

    return float(total_reward), components
```