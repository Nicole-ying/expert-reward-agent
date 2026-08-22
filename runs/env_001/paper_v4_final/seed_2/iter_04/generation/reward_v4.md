```markdown
# 设计理由
本轮修改 landing_bonus 组件，将其从单纯依赖位置/速度/姿态的三因子乘积，扩展为引入 **双脚接触标志** 的乘数。  
这是因为信号覆盖审计发现 `obs[6]` 和 `obs[7]`（左右支撑脚接触）至今未被使用，而成功着陆恰好以双脚同时站立于平台为标志。  
当前 agent 已经学会接近目标、减速并保持姿态，但 episode 平均长度仍较长（547 步），可能是在低空徘徊以持续收获 `landing_bonus`。  
增加接触乘数后，飞行中奖励不变，但双脚接触时奖励骤然提升为原来的 4 倍，从而更强烈地标记“成功着陆”状态，鼓励 agent 减少空中犹豫、尽快完成平稳接触。  
数学形式：`landing_bonus = 2.0 * proximity * speed_factor * angle_factor * (1.0 + 3.0 * (obs[6] * obs[7]))`。  
系数校准：保持飞行段奖励不塌缩（乘数=1），接触时乘数=4；per‑step 量级仍远小于主信号，不会造成 reward 尺度崩溃。  

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next_obs dimensions
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Primary progress signal
    pos_sq_error = x_pos ** 2 + y_pos ** 2
    vel_sq_error = x_vel ** 2 + y_vel ** 2
    progress = -0.05 * pos_sq_error - 0.1 * vel_sq_error

    # 2. Stability constraint
    pose_penalty = -5.0 * (body_angle ** 2) - 0.5 * (angular_vel ** 2)

    # 3. Approach & soft landing bonus, now amplified when both feet touch ground
    proximity = 1.0 / (1.0 + 10.0 * (x_pos ** 2 + y_pos ** 2))
    speed_magnitude = abs(x_vel) + abs(y_vel)
    speed_factor = 1.0 / (1.0 + 5.0 * speed_magnitude)
    angle_factor = 1.0 / (1.0 + 20.0 * abs(body_angle))
    both_contact = left_contact * right_contact          # 1.0 only when both supports touch
    landing_bonus = 2.0 * proximity * speed_factor * angle_factor * (1.0 + 3.0 * both_contact)

    total_reward = progress + pose_penalty + landing_bonus

    components = {
        'progress': progress,
        'pose_penalty': pose_penalty,
        'landing_bonus': landing_bonus
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 接触标志未使用，成功着陆的信号可以进一步加强以缩短徘徊时间。  
- **behavior**: agent 已学会接近目标、减速并保持姿态，但可能在低空徘徊以持续获取 `landing_bonus`。  
- **signal**: 缺少落地确认信号（双脚接触），导致“到达目标上空”与“完成着陆”的奖励梯度不够陡峭。  
- **level**: Level 2  
- **hypothesis**: 在 `landing_bonus` 中加入双脚接触乘数，会在着陆瞬间将奖励放大 4 倍，从而激励 agent 减少接近目标后的犹豫，更快完成平稳着陆。  
- **risk**: 若 agent 过早以较高速度接触地面，仍会被 `speed_factor` 和 `pose_penalty` 惩罚；最可能的副作用是略微增加落地冲击，但整体着陆质量不会显著下降。  
```