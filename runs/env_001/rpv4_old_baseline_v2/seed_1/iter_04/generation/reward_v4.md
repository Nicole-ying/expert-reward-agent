# 奖励组件分析

## 1. 总体评估
- 最终得分 52.5，距离目标 200 差距明显。  
- 训练过程检查点得分始终为 0（40% 与 80% 时），说明早期学习信号几乎为零，后期才出现微弱的正向行为。  
- 最终策略的 terminated 率较高（16/20），表明大部分 episode 能够正常结束，但 reward 构成揭示了一个严重问题：**接触奖励 magnitude_share 高达 97.8%，完全支配了整个回报信号**，使得所有其他组件对学习几乎没有贡献。  
- 其他惩罚项（距离、角度、速度、角速度）在 magnitude_share 上几乎可以忽略，意味着 agent 在学习过程中只被“是否有接触”这一个二进制信号驱动，忽视了飞行姿态、速度与位置精度。这导致策略不稳定、总得分低。  
- 结论：**核心问题在于接触奖励的绝对值过大，挤占了所有其它塑造性奖励的空间**。需要大幅削减接触相关奖励的量级，同时适度增强基础飞行品质（靠近中心、减速、保持姿态）的引导，并增加“持续稳定降落”的稀疏奖励来帮助 agent 收敛到高质量着陆。

## 2. 逐组件诊断

### (2a) contact_reward
- `episode_sum_mean ≈ 6100`，`magnitude_share ≈ 97.8%`，active_rate 仅 29.5%。  
- 每一步接触可能带来 10 ~ 40 的奖励，在 episode 中累积后成为怪兽级信号，agent 几乎只为获得这一奖励而行动，对其他惩罚项无动于衷。  
- **必须大幅削减**：将单腿奖励从 10 降至 1，双腿同时奖励从 30 降至 2，使该组件不再主宰梯度。

### (2b) distance_penalty
- 当前 `-0.5 * dist`，始终活跃，但量级太小，在接触奖励面前完全被淹没。  
- 虽然它会一直推动 agent 靠近中心，但 agent 根本“看不到”它。需要**减弱系数并改为更偏向接近时的相对奖励**，同时引入“进步奖励”（后述）以提供清晰的方位引导。  
- 改为更温和的 `-0.1 * dist`，并添加基于位置变化的 `dist_reward`。

### (2c) angle_penalty, vel_penalty, angvel_penalty
- 这些项数值极小，agent 对此不敏感。需适当增大速度与角速度惩罚，但不要过度抑制正常机动；角度惩罚保持或微增。  
- 将 `vel_penalty` 从 `-0.02` 调整为 `-0.2`，`angvel_penalty` 从 `-0.1` 调整为 `-0.02`，`angle_penalty` 保持 `-0.5 * angle²`。

### (2d) 新引入组件：dist_reward 与 success_bonus
- **dist_reward**：利用相邻步骤的位置变化 `(prev_dist - curr_dist)` 提供即时正/负反馈，强化“向中心靠近”的行为，这在早期探索阶段很关键。  
- **success_bonus**：当两脚同时接触且距离中心足够近（< 0.5）时给予固定奖励，鼓励 agent 在达成目的后稳定保持，而非“碰一下就弹开”。

## 3. 改动计划
1. **contact_reward** 截断式缩小，从 (10,10,30) 变为 (1,1,2)。  
2. **添加 dist_reward**，系数 0.5×步进距离。  
3. **distance_penalty** 系数降至 0.1。  
4. **vel_penalty** 系数提升至 -0.2，**angvel_penalty** 降至 -0.02（相对于 -0.1 略降，以免过度惩罚必要的旋转矫正）。  
5. 新增 **success_bonus**，当 left_contact & right_contact & curr_dist<0.5 时给予 +2。  
6. 所有组件在返回值中如实记录，便于后续监控 magnitude_share。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next state
    px, py = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Unpack previous state for distance change
    prev_px, prev_py = obs[0], obs[1]

    curr_dist = (px**2 + py**2)**0.5
    prev_dist = (prev_px**2 + prev_py**2)**0.5

    # 1. Progress toward center
    dist_reward = 0.5 * (prev_dist - curr_dist)

    # 2. Small penalty for being far away (shapes final position)
    distance_penalty = -0.1 * curr_dist

    # 3. Keep upright
    angle_penalty = -0.5 * (angle**2)

    # 4. Encourage low speed
    vel_penalty = -0.2 * (vx**2 + vy**2)

    # 5. Penalize excessive rotation
    angvel_penalty = -0.02 * (ang_vel**2)

    # 6. Soft contact incentive (magnitude drastically reduced)
    contact_reward = 1.0 * left_contact + 1.0 * right_contact + 2.0 * (left_contact * right_contact)

    # 7. Success bonus for stable touchdown near center
    success_bonus = 2.0 if (left_contact and right_contact and curr_dist < 0.5) else 0.0

    total = (dist_reward + distance_penalty + angle_penalty +
             vel_penalty + angvel_penalty + contact_reward + success_bonus)

    components = {
        "dist_reward": dist_reward,
        "distance_penalty": distance_penalty,
        "angle_penalty": angle_penalty,
        "vel_penalty": vel_penalty,
        "angvel_penalty": angvel_penalty,
        "contact_reward": contact_reward,
        "success_bonus": success_bonus
    }
    return float(total), components
```