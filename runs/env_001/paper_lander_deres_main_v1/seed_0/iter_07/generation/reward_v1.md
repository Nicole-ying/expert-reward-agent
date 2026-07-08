# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for lunar-lander-style task (navigation_goal_reaching).
    
    Uses a potential-based shaping as the main progress signal and a soft
    landing proxy as a sparse success approximation.
    """
    
    # ----- Unpack observations -------------------------------------------------
    # Current
    x = obs[0]
    y = obs[1]
    vx = obs[2]
    vy = obs[3]
    angle = obs[4]
    left_contact = obs[6]
    right_contact = obs[7]

    # Next
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_vx = next_obs[2]
    next_vy = next_obs[3]
    next_angle = next_obs[4]
    next_left = next_obs[6]
    next_right = next_obs[7]

    # ----- Derived quantities ---------------------------------------------------
    # Euclidean distance to target (origin)
    dist = (x ** 2.0 + y ** 2.0) ** 0.5
    next_dist = (next_x ** 2.0 + next_y ** 2.0) ** 0.5

    # Speed
    speed = (vx ** 2.0 + vy ** 2.0) ** 0.5
    next_speed = (next_vx ** 2.0 + next_vy ** 2.0) ** 0.5

    # ----- Component 1: Progress reward (potential-based shaping) ----------------
    # Phi(state) = -(dist + β_speed * speed + β_angle * |angle|)
    # The agent receives reward for decreasing distance, slowing down, and
    # aligning upright.
    beta_speed = 0.5
    beta_angle = 0.3
    gamma = 0.99

    phi = -(dist + beta_speed * speed + beta_angle * abs(angle))
    phi_next = -(next_dist + beta_speed * next_speed + beta_angle * abs(next_angle))

    progress_reward = gamma * phi_next - phi

    # ----- Component 2: Landing soft proxy (first touchdown) --------------------
    # Give a one-time bonus when both legs transition from no-contact to contact
    # while velocity and attitude are small (indicating a safe landing).
    landing_reward = 0.0
    landing_bonus = 10.0
    speed_threshold = 0.3
    angle_threshold = 0.1

    both_prev_air = (left_contact == 0.0) and (right_contact == 0.0)
    both_next_ground = (next_left == 1.0) and (next_right == 1.0)
    stable_touchdown = (next_speed < speed_threshold) and (abs(next_angle) < angle_threshold)

    if both_prev_air and both_next_ground and stable_touchdown:
        landing_reward = landing_bonus

    # ----- Total reward ---------------------------------------------------------
    total_reward = progress_reward + landing_reward

    # Components dict (only the terms that are summed)
    components = {
        "progress_reward": progress_reward,
        "landing_reward": landing_reward,
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 组件列表与角色

1. **progress_reward（主学习信号）**  
   - 形态：势能差塑形 `gamma * Φ(next) − Φ(obs)`，`Φ = −(distance + β_speed·speed + β_angle·|angle|)`  
   - 角色：每步提供稠密梯度，引导飞行器靠近目标（减小距离）、减速并保持姿态接近竖直。  
   - 为什么选它：相比单纯的 distance reward 或 progress delta，势能塑形将距离、速度和角度