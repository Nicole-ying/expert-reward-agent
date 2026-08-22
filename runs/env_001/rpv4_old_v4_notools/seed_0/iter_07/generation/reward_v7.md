**1. 诊断报告**

- **evidence**：  
  `iter_6` 最终得分 **-103.79**，所有集均 **truncated**(0/20 terminated)，没有任何一次真正着陆。  
  `landing_reward` magnitude_share 99.2%，episode_sum_mean 高达 **1348**，而 `progress_reward` 几乎为 0。  
  checkpoint 趋势显示 `landing_reward` 从 36 暴涨到 1500，但得分仅从 -2556 微升至 -108，并未转为正值。  
  以上说明 `landing_reward` 提供了巨大的持续正奖励，但外部任务（着陆并稳定）并未被完成，奖励被 exploit。

- **behavior_diagnosis**：  
  策略学会了将飞行器保持在 **低空、低速、小角度** 的状态来持续获取 `landing_reward`（基于高度、速度、角度的静态质量分数），而不是下降并接触地面。由于所有集都是 1000 步截断，且没有 early terminal，说明这种“低空悬浮”行为可以维持整个 episode 而不触发终止。这是 **不健康的奖励利用**，代理找到了一个局部最优但未能完成着陆。

- **signal_completeness**：  
  缺失 **下降过程奖励** 与 **着陆事件信号**。现有 `landing_reward` 是纯状态奖励，不区分是否在下降、是否首次接触。`progress_reward` 被健康门控严重削弱，形同虚设。

- **selected_level**：**L2**  
  将 `landing_reward` 从“连续状态质量奖励”改为“下降过程奖励 + 接触事件奖励”，从根本上消除静态悬停可利用点。同步删除 `progress_reward` 的健康门控使其重新生效。

- **selected_intervention**：  
  - **删除**：原有的 `height_factor`、`speed_scale`、`angle_scale`、`y_threshold` 及整个基于 `quality` 的 `landing_reward` 公式。  
  - **新增**：  
    - `descent_reward`：基于高度下降量 `max(0, obs[1] - next_obs[1])`，奖励向地面移动。  
    - `contact_reward`：检测双腿从「未接触」→「接触」的转移事件，给予一次性大额奖励，避免持续积累。  
  - **修改**：`progress_reward` 去掉健康门控，仅使用 `dist_cur - dist_next`，使水平接近目标也能提供有效梯度。  
  - **保留**：`action_penalty` 不变。

- **falsifiable_hypothesis**：  
  若用“下降量”和“首次接触”代替“静态悬浮质量”，代理将被迫下降高度并触发接触才能获得主要奖励，从而消除悬浮 exploit，下一轮应出现 **terminated 集、score 明显上升**。

- **expected_next_round**：  
  训练初期代理会尝试下降并触碰地面，可能存在一些过早接触或碰撞，但随着训练推进，预计整体得分提升，部分 episode 将达成着陆并终止，`terminated/20` 不再为 0。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------------------
    # Hyperparameters
    # -------------------------------
    w_progress = 1.0       # weight for horizontal distance reduction
    w_descent = 2.0        # weight for height descent reward
    contact_bonus = 50.0   # one-time reward for first dual-leg contact event
    w_action = 0.01        # action sparsity penalty weight

    # -------------------------------
    # 1. Horizontal progress (distance to target, no health gate)
    # -------------------------------
    dist_cur = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = w_progress * (dist_cur - dist_next)

    # -------------------------------
    # 2. Descent reward (altitude reduction only)
    # -------------------------------
    descent_reward = w_descent * max(0.0, obs[1] - next_obs[1])

    # -------------------------------
    # 3. First dual-leg contact event reward
    # -------------------------------
    prev_contact = (obs[6] > 0.5 and obs[7] > 0.5)
    curr_contact = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    contact_reward = 0.0
    if (not prev_contact) and curr_contact:
        contact_reward = contact_bonus

    # -------------------------------
    # 4. Action sparsity penalty
    # -------------------------------
    action_penalty = -w_action * (1.0 if action != 0 else 0.0)

    # -------------------------------
    # Total reward
    # -------------------------------
    total_reward = progress_reward + descent_reward + contact_reward + action_penalty
    components = {
        "progress_reward": progress_reward,
        "descent_reward": descent_reward,
        "contact_reward": contact_reward,
        "action_penalty": action_penalty
    }
    return float(total_reward), components
```