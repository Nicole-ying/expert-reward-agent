# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 计算当前位置与目标(0,0)的欧氏距离
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    # 主学习信号：每一步靠近目标的进展
    progress_reward = dist_obs - dist_next

    # 稳定/安全约束：抑制过度速度、倾角和角速度
    w_v = 0.01   # 线速度惩罚权重
    w_a = 0.01   # 倾角惩罚权重
    w_w = 0.01   # 角速度惩罚权重
    stability_penalty = (
        -w_v * (abs(next_obs[2]) + abs(next_obs[3]))
        - w_a * abs(next_obs[4])
        - w_w * abs(next_obs[5])
    )

    # 任务完成近似信号：多条件组合的着陆奖励代理
    x_thresh = 0.1
    y_thresh = 0.1
    v_thresh = 0.2
    angle_thresh = 0.1
    left_contact = next_obs[6] > 0.5
    right_contact = next_obs[7] > 0.5
    if (abs(next_obs[0]) < x_thresh and abs(next_obs[1]) < y_thresh and
        abs(next_obs[2]) < v_thresh and abs(next_obs[3]) < v_thresh and
        abs(next_obs[4]) < angle_thresh and left_contact and right_contact):
        landing_bonus = 0.1
    else:
        landing_bonus = 0.0

    total_reward = progress_reward + stability_penalty + landing_bonus

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'landing_bonus': landing_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **使用的奖励组件（3 个）：**
  - `progress_reward`：**主学习信号**。每一步奖励向目标平台（`(0,0)`）靠近的欧氏距离减少量，稠密梯度引导 agent 持续接近着陆点。
  - `stability_penalty`：**轻量稳定约束**。对水平/垂直线速度、机体倾角、角速度施加小权重负奖励，促使 agent 在接近目标过程中逐步减速、调平姿态，防止剧烈翻滚或高速撞击。
  - `landing_bonus`：**任务完成近似信号**。当 agent 同时满足中心位置（`|x|,|y| < 0.1`）、低速（`|vx|,|vy| < 0.2`）、小倾角（`|angle| < 0.1`）且双支撑接触时，给予持续的正奖励。它提供额外的着陆-保持激励，降低在目标附近震荡而不完成的可能。

- **每个组件的角色：**
  - `progress_reward` 承担密集学习驱动，告诉 agent “朝目标移动”是得分的主要途径。
  - `stability_penalty` 作为方向盘式软约束，避免完全以高速、大角度代价换取距离接近，同时为安全接触铺垫。
  - `landing_bonus` 作为目标完成代理，将离散的着陆成功事件分解为连续检查条件，给真正稳定着陆提供正向信号，又不伪造本不存在的 success flag。

- **为何未使用 `terminal_success_reward` / `terminal_failure_penalty`：**
  环境卡明确 `explicit_success_flag_available=false`、`explicit_failure_flag_available=false`，且 info 为空字典，无任何成功/失败标记可用。因此设计完全避开依赖此类信号的骨架。

- **后续迭代中可能加入的组件：**
  - 效率代价（`energy_penalty`）：在当前设计学会着陆后，再通过小权重惩罚主引擎使用，优化能耗。
  - 时间惩罚：若着陆后仍滞留过长，可加入平坦的时间成本。
  - 更强姿态约束或门控（`gated_reward`）：若出现利用 stability_penalty 的漏洞（例如倒立接触仍获得 landing_bonus），可收紧倾角阈值或引入基于姿态的门控。
  - 真正的 terminal 成功/失败奖励：如果将来 wrapper 暴露 termination 原因（如‘settled’ vs ‘crash’），再按需添加稀疏事件奖励。

- **训练后应重点观察的 failure modes：**
  - **high_reward_without_success**：agent 可能在目标上方反复振荡却不真正着陆，因为 progress_reward 激励靠近后继续保持小幅度上下，需检查 landing_bonus 是否足以压制这种振荡。
  - **crash near goal**：agent 以高速冲向平台获得大 progress_reward 但撞击触发 failure，应看 stability_penalty 权重是否不足以抑制危险速度。
  - **contact hacking**：如果 landing_bonus 的条件过于宽松，agent 可能通过反复轻触平台边缘触发奖励而不稳定停靠；需观察接触频率与位置分布。
  - **agent_afraid_to_move**：若 stability_penalty 过强，agent 可能不敢点火，导致距离长时间不减小；应监控初期 progress_reward 是否持续为负。