根据格式修复要求，基于历史最佳代码进行最小但可证伪的修改，以解决landing信号激活不足的问题。修改方向：扩大接近程度的敏感距离和静止条件的阈值，使landing_proxy在相对较远的区域也能提供梯度，引导智能体减速靠近目标。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Next state
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # 1. Bounded proximity: always provides gradient toward target
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    r_dist = 1.0 / (1.0 + distance)
    
    # 2. Stability penalty: discourage violent motion
    r_stability = -(
        0.01 * (abs(x_vel) + abs(y_vel)) +
        0.1 * abs(body_angle) +
        0.05 * abs(ang_vel)
    )
    
    # 3. Continuous soft-landing proxy with widened activation ranges
    #    to provide gradient earlier and guide deceleration.
    proximity = max(0.0, 1.0 - distance / 1.0)            # activates when dist < 1.0
    stillness = max(0.0, 1.0 - (abs(x_vel) + abs(y_vel)) / 0.8)  # activates when |v| < 0.8
    upright   = max(0.0, 1.0 - abs(body_angle) / 0.3)    # unchanged
    contact   = (left_contact + right_contact) / 2.0
    r_landing = 8.0 * proximity * stillness * upright * (0.1 + 0.9 * contact)
    
    total_reward = r_dist + r_stability + r_landing
    
    components = {
        "distance_reward": r_dist,
        "stability_penalty": r_stability,
        "landing_proxy": r_landing
    }
    
    return float(total_reward), components
```