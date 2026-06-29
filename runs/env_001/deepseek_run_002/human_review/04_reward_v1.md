# reward_v1.py

```python
import numpy as np

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for Env_001: 2D lunar lander trajectory optimization.
    
    Args:
        obs: current observation [x_pos, y_pos, x_vel, y_vel, body_angle, angular_vel, left_contact, right_contact]
        action: discrete action (0: no_engine, 1: left_orientation, 2: main_engine, 3: right_orientation)
        next_obs: next observation after taking action
        original_reward: original environment reward (not used)
        info: info dict (empty for this environment)
        training_progress: training progress from 0.0 to 1.0 (not used in v1)
    
    Returns:
        total_reward: scalar reward value
        info: updated info dict with reward_terms
    """
    # Extract relevant signals from observations
    # Position relative to target (target is at origin)
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    # Next state signals
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # ========== 1. Progress-based distance reward (main learning signal) ==========
    # Calculate Euclidean distance to target (origin)
    current_dist = np.sqrt(x_pos**2 + y_pos**2)
    next_dist = np.sqrt(next_x_pos**2 + next_y_pos**2)
    
    # Progress reward: positive when moving closer to target
    progress_reward = current_dist - next_dist
    
    # ========== 2. Lightweight stability penalty ==========
    # Penalize high velocity and large angular deviation
    # These are important for stable landing approach
    speed = np.sqrt(next_x_vel**2 + next_y_vel**2)
    angle_penalty = abs(next_body_angle)  # Penalize deviation from upright (angle=0)
    angular_vel_penalty = abs(next_angular_vel)  # Penalize rotation
    
    # Combine stability penalties with small weights
    # Weight for velocity: 0.01 (gentle penalty for high speed)
    # Weight for angle: 0.02 (moderate penalty for tilting)
    # Weight for angular velocity: 0.005 (light penalty for spinning)
    stability_penalty = -0.01 * speed - 0.02 * angle_penalty - 0.005 * angular_vel_penalty
    
    # ========== 3. Contact bonus (encouraging landing) ==========
    # Reward when both supports are in contact (landed)
    contact_bonus = 0.0
    if next_left_contact > 0.5 and next_right_contact > 0.5:
        contact_bonus = 1.0  # Small bonus for being on the ground
    
    # ========== Combine rewards ==========
    total_reward = progress_reward + stability_penalty + contact_bonus
    
    # ========== Store reward terms for diagnostics ==========
    info["reward_terms"] = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "contact_bonus": contact_bonus,
        "total_reward": total_reward
    }
    
    return total_reward, info
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_reward** (主学习信号)
   - 计算当前距离与下一状态距离的差值（`current_dist - next_dist`）
   - 作用：每一步都奖励智能体向目标（原点）靠近，提供密集的梯度引导
   - 这是导航目标到达任务的核心信号，直接对应任务目标

2. **stability_penalty** (轻量约束项)
   - 包含三个子项：速度惩罚、姿态角惩罚、角速度惩罚
   - 作用：鼓励智能体以低速、直立姿态接近目标，避免高速撞击或姿态失稳
   - 权重较小（0.01~0.02），不会过度抑制探索，但能提供基本的安全信号

3. **contact_bonus** (着陆奖励)
   - 当左右支撑同时接触时给予小奖励
   - 作用：明确鼓励完成着陆动作，提供任务完成的信号
   - 权重较小（1.0），避免过度主导学习

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 根据环境卡片，`explicit_success_flag_available=false` 且 `explicit_failure_flag_available=false`
- info 字典为空，无法获取显式的成功/失败标志
- 使用这些终端奖励会需要猜测终止条件，可能导致错误的奖励信号
- 后续迭代中，如果 wrapper 暴露了成功/失败标志，可以加入

## 后续迭代可以添加

1. **energy_penalty**：当智能体能稳定接近目标后，加入引擎使用惩罚以优化燃料消耗
2. **time_penalty**：如果智能体在目标附近徘徊不完成，加入时间惩罚提高效率
3. **terminal_success_reward**：当能明确检测到成功着陆时，加入大额成功奖励
4. **gated_reward**：在危险状态（如高速、大角度）时，用安全门控保护智能体
5. **stability_penalty 权重调整**：根据训练表现动态调整稳定惩罚的权重

## 训练后应该观察的 failure mode

1. **goal_near_oscillation**：智能体在目标附近来回移动但不完成着陆
   - 观察指标：progress_reward 接近 0 但 contact_bonus 始终为 0
   
2. **high_reward_without_success**：总奖励很高但从未成功着陆
   - 观察指标：progress_reward 持续为正但 contact_bonus 从未触发
   
3. **fast_crash_near_goal**：智能体高速冲向目标但无法减速
   - 观察指标：progress_reward 很高但 stability_penalty 也很高（负值大）
   
4. **agent_afraid_to_move**：稳定惩罚过强导致智能体不敢移动
   - 观察指标：progress_reward 长期接近 0，总奖励主要由 contact_bonus 贡献
   
5. **angle_instability**：智能体在接近目标时姿态失控
   - 观察指标：angle_penalty 和 angular_vel_penalty 持续较大