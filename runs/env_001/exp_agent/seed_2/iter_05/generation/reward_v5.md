## 诊断

让我分析当前情况：

**当前分数：** -108.98（迭代4），相比历史最佳 -110.68 略有改善，但幅度很小（+1.70），且 episode length 始终卡在 72 步。

**关键问题分析：**

1. **stability_penalty 主导信号**：ratio_to_progress = -1.70，绝对值远大于 0.5。惩罚项完全主导了梯度信号。agent 在 72 步就终止（100% early_terminal），说明它因为害怕惩罚而不敢动或过早坠毁。

2. **soft_landing_bonus nonzero_rate=1.7%**：虽然从 1.8%（历史最佳）略微下降到 1.7%，但当前代码已经改成了连续乘积形式（这是好的方向）。然而，**contact_factor = next_left_contact * next_right_contact** 是二值乘积——只要有一个接触为 0，整个乘积为 0。这意味着 agent 必须同时接触两个支撑点才能获得任何 bonus，这太苛刻了。

3. **stability_penalty 系数过大**：上一轮我把 angle_penalty 从 -0.1 提到 -1.0（10倍），angular_vel_penalty 从 -0.05 提到 -0.5（10倍），speed_penalty 从 -0.03 提到 -0.2（7倍）。结果 ratio_to_progress 从 -0.039（迭代3）飙升到 -1.70（迭代4），惩罚完全压倒了进度信号。

**修改计划（只改一个方面：stability_penalty 的系数）：**

把 stability_penalty 的系数大幅降低，回到合理水平。同时保持 soft_landing_bonus 的连续乘积形式（但去掉 contact_factor 的苛刻二值条件）。

等等，让我重新思考——我应该只改一个方面。当前最严重的问题是 stability_penalty 主导（ratio=-1.70）。我应该先解决这个。

**方案：大幅降低 stability_penalty 系数**，回到接近迭代3的水平（当时 ratio=-0.039，表现最好）。

迭代3的系数：angle=-0.5, angular_vel=-0.3, speed=-0.2 → ratio=-0.039
迭代4的系数：angle=-1.0, angular_vel=-0.5, speed=-0.2 → ratio=-1.70

问题出在 angle_penalty 从 -0.5 提到 -1.0，angular_vel_penalty 从 -0.3 提到 -0.5。让我把它们降回迭代3的水平。

等等，让我再看看迭代2和3的数据。

迭代2：stability_penalty mean=-0.006, ratio≈-0.04 — 但 score=-118.44（最差）
迭代3：stability_penalty mean=-0.039, ratio≈-0.24 — score=-111.05
迭代4：stability_penalty mean=-0.271, ratio=-1.70 — score=-108.98

看起来 stability_penalty 越强（负值越大），分数反而越好？但 ratio=-1.70 说明惩罚主导了信号，这不是健康状态。

实际上，score 从 -118 → -111 → -109 在缓慢改善，但 episode length 始终是 72。这说明 agent 在 72 步时因为 crash_or_body_contact 终止了。

让我换个思路：**stability_penalty 需要，但不能主导。** 同时 soft_landing_bonus 的 nonzero_rate 太低（1.7%），需要提高触发率。

我决定这次**只改 soft_landing_bonus**——把 contact_factor 从二值乘积改为更宽松的形式，让 agent 在只有单侧接触时也能获得部分奖励，从而提供更密集的梯度信号。

不对，让我重新审视。根据"改动的可检验性"原则，我应该一次只改一个方面。

当前最严重的问题是 **stability_penalty 主导（ratio=-1.70）**。先解决这个。

把 stability_penalty 系数降回迭代3的水平（angle=-0.5, angular_vel=-0.3, speed=-0.2），这样 ratio 应该回到 -0.2~-0.3 的健康范围。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. Main learning signal: progress_delta_reward
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_delta_reward = 10.0 * progress_delta
    
    # 2. Stability penalty — REDUCED back to iteration 3 levels
    # Iteration 4 had angle=-1.0, angular_vel=-0.5, speed=-0.2 → ratio=-1.70 (dominating!)
    # Rolling back to iteration 3 levels where ratio was healthy (~-0.24)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.5 * abs(next_body_angle)        # was -1.0, back to iter3 level
    angular_vel_penalty = -0.3 * abs(next_angular_vel)  # was -0.5, back to iter3 level
    speed_penalty = -0.2 * speed                         # unchanged
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. Soft landing proxy — keep the continuous product form from iter4
    # but relax the contact_factor to avoid zeroing out the whole product
    proximity_factor = 1.0 / (1.0 + 3.0 * next_dist)
    speed_factor = 1.0 / (1.0 + 3.0 * speed)
    angle_factor = 1.0 / (1.0 + 5.0 * abs(next_body_angle))
    angular_vel_factor = 1.0 / (1.0 + 3.0 * abs(next_angular_vel))
    # Use max of left/right contact instead of product — agent gets partial credit
    # for touching even one support, providing gradient toward both-contact
    contact_factor = max(next_left_contact, next_right_contact)  # was product, now max
    
    soft_landing_bonus = 20.0 * proximity_factor * speed_factor * angle_factor * angular_vel_factor * contact_factor
    
    # 4. Small energy penalty for using engines
    energy_penalty = 0.0
    if action != 0:
        energy_penalty = -0.05
    
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + energy_penalty
    
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

等等，我同时改了 stability_penalty 的系数（3个系数）和 soft_landing_bonus 的 contact_factor。这违反了"一次只改一个方面"的原则。

让我重新思考。最核心的问题是 stability_penalty 主导（ratio=-1.70）。我应该只改这个。

但 soft_landing_bonus 的 nonzero_rate=1.7% 也很低，而且 contact_factor 用乘积确实有问题。

好吧，这两个改动其实属于不同组件。让我这次**只改 stability_penalty 的系数**，把 contact_factor 保持原样（乘积形式），看看效果。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. Main learning signal: progress_delta_reward (unchanged)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_delta_reward = 10.0 * progress_delta
    
    # 2. Stability penalty — REDUCED coefficients
    # Iteration 4: angle=-1.0, angular_vel=-0.5, speed=-0.2 → ratio=-1.70 (dominating!)
    # Rolling back to iteration 3 levels where ratio was ~-0.24 (healthy)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.5 * abs(next_body_angle)        
    angular_vel_penalty = -0.3 * abs(next_angular_vel)  
    speed_penalty = -0.2 * speed                         
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. Soft landing proxy (unchanged from iter4)
    proximity_factor = 1.0 / (1.0 + 3.0 * next_dist)
    speed_factor = 1.0 / (1.0 + 3.0 * speed)
    angle_factor = 1.0 / (1.0 + 5.0 * abs(next_body_angle))
    angular_vel_factor = 1.0 / (1.0 + 3.0 * abs(next_angular_vel))
    contact_factor = next_left_contact * next_right_contact
    
    soft_landing_bonus = 20.0 * proximity_factor * speed_factor * angle_factor * angular_vel_factor * contact_factor
    
    # 4. Small energy penalty (unchanged)
    energy_penalty = 0.0
    if action != 0:
        energy_penalty = -0.05
    
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + energy_penalty
    
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

嗯，但仔细想想——迭代3的 stability 系数就是 angle=-0.5, angular_vel=-0.3, speed=-0.2，当时 score=-111.05。迭代4把 angle 提到 -1.0, angular_vel 提到 -0.5 后 score=-108.98。所以更强的惩罚反而分数略好？

但 ratio=-1.70 确实不健康。让我换个角度——也许问题不是 stability 太强，而是 progress_delta 太弱（mean=0.16），导致 ratio 看起来很大。

实际上 progress_delta_reward 的 mean=0.159，而 stability_penalty 的 mean=-0.271。绝对值上 stability 是 progress 的 1.7 倍。

让我试试把 stability 系数降到比迭代3还低，看看效果。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. Main learning signal: progress_delta_reward (unchanged)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_delta_reward = 10.0 * progress_delta
    
    # 2. Stability penalty — REDUCED to prevent dominating the signal
    # Iteration 4 ratio=-1.70 → penalty completely dominates progress signal
    # Reducing all coefficients by ~3x from iter4 levels to get ratio < 0.5
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.3 * abs(next_body_angle)         # was -1.0 in iter4, -0.5 in iter3
    angular_vel_penalty = -0.15 * abs(next_angular_vel)  # was -0.5 in iter4, -0.3 in iter3
    speed_penalty = -0.1 * speed                          # was -0.2 in both
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. Soft landing proxy (unchanged from iter4)
    proximity_factor = 1.0 / (1.0 + 3.0 * next_dist)
    speed_factor = 1.0 / (1.0 + 3.0 * speed)
    angle_factor = 1.0 / (1.0 + 5.0 * abs(next_body_angle))
    angular_vel_factor = 1.0 / (1.0 + 3.0 * abs(next_angular_vel))
    contact_factor = next_left_contact * next_right_contact
    
    soft_landing_bonus = 20.0 * proximity_factor * speed_factor * angle_factor * angular_vel_factor * contact_factor
    
    # 4. Small energy penalty (unchanged)
    energy_penalty = 0.0
    if action != 0:
        energy_penalty = -0.05
    
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + energy_penalty
    
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

等等，让我再想想。迭代3的 stability 系数（angle=-0.5, angular_vel=-0.3, speed=-0.2）得到的 ratio 是 -0.24（从 mean 值 -0.039 / 0.160 ≈ -0.24）。这其实挺健康的。

迭代4把 angle 翻倍到 -1.0，angular_vel 从 -0.3 提到 -0.5，ratio 跳到 -1.70。

所以回到迭代3的系数应该就能解决主导问题。但迭代3的 score=-111.05，迭代4的 score=-108.98... 迭代4略好。

也许问题不是 stability 本身，而是 agent 在 72 步 crash 是因为它飞得太快/姿态不对。stability 惩罚虽然 ratio 大，但确实在帮助 agent 活得更久（从 crash 角度看）。

让我换个思路——也许我应该保持 stability 的强度，但**把 progress_delta_reward 的权重也提高**，让两个信号平衡。

progress_delta_reward 当前 weight=10.0，mean=0.159。如果提高到 20.0，mean 会到 ~0.32，ratio 就会降到 ~-0.85。

但这样还是 > 0.5。而且提高 progress 权重可能让 agent 更激进地冲向目标然后 crash。

好吧，最稳妥的方案是：**把 stability 系数降回迭代3水平**，这是唯一一次只改一个组件的改动。迭代3的 ratio=-0.24 是健康的。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y