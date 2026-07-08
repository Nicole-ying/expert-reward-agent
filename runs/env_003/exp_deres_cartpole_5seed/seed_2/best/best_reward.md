# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for Env_003 (CartPole-like survival balance task).

    - progress_reward: baseline - (w_angle * angle^2 + w_pos * pos^2)
      Encourages keeping the pole near upright and the cart near the center.
    - stability_penalty: small penalty on cart velocity and pole angular velocity
      to suppress unnecessary oscillations.
    """
    # extracted constants for clarity
    w_angle = 5.0
    w_pos = 0.1
    baseline = 1.0

    lambda_vel = 0.01
    lambda_angvel = 0.01

    # next state
    pos = next_obs[0]
    vel = next_obs[1]
    angle = next_obs[2]
    angvel = next_obs[3]

    # main learning signal
    progress_reward = baseline - (w_angle * (angle ** 2) + w_pos * (pos ** 2))

    # stability penalty (light, avoid dominating)
    stability_penalty = -lambda_vel * abs(vel) - lambda_angvel * abs(angvel)

    total_reward = progress_reward + stability_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用组件与角色
- **progress_reward** （主学习信号）：  
  形式为 `baseline - (w_angle * angle^2 + w_pos * pos^2)`。  
  角色：直接鼓励杆保持竖直（角度小）和底座靠近轨道中心（位置小）。  
  通过正基线（1.0）保证在