### 1. 诊断报告

- **evidence**：
  - 最终 score=171.4，接近目标 200 但未达标；len=781.4 很长，terminated=7/20（仅 35% 成功着陆），truncated=13/20（多数因超时截断）。
  - contact_guidance 以 99.2% magnitude_share 主导奖励，ep_sum_mean=252.9，active_rate=68%。
  - checkpoint 趋势：cp70% 时 contact_guidance 从≈0 骤升至 248，此后维持高位，同时 score 从负转正并攀升至 223.9（训练末），但截断的 episode 说明 agent 学会了在平台上持续领取接触奖励，而不是快速完成着陆。
  - orientation_penalty、velocity_penalty、progress 都很小，未起有效引导作用。
  - **核心矛盾**：contact_guidance 作为状态值（接触程度 × 门控）允许 agent 维持接触时无限累积奖励，造成 proxy 与“尽快稳定着陆”任务目标对齐不良——agent 学会在平台上徘徊以收割奖励，而非满足终止条件。

- **behavior_diagnosis**：
  策略已学会飞向目标并建立双腿接触，但随后在平台上长时间缓慢调整或悬浮，不断触发 contact_guidance 以获取大量奖励，避免 episode 终止（因为一旦成功着陆环境会提前结束，奖励终止）。这表现为高 contact_guidance、长 episode、低 terminated 率。行为不健康，属于典型的 “proxy exploit”。

- **signal_completeness**：
  主信号框架（接近-接触奖励 + 速度/姿态惩罚）方向基本正确，但缺少“一次性接触成功”或“接触改善”信号，导致重复奖励。目前的 contact_guidance 没有对“首次稳定接触”与“持续占有”做出区分。

- **selected_level**：**L2**  
  选取原因是 exploit 源于数学形态（状态值 vs. 改善量），仅调整系数（L1）无法根除重复累积问题。改动一个组件的表达式即可修复，无需更换主框架（L3）。

- **selected_intervention**：
  目标组件：`contact_guidance`。 
  将 `contact_guidance` 从基于**当前接触状态**（绝对值）改为基于**接触改善量**：
  ```
  contact_improvement = max(0, next_contact - prev_contact)
  contact_guidance = 0.3 * contact_improvement * proximity_factor * velocity_ok
  ```
  保持其余组件不变。

- **falsifiable_hypothesis**：
  改为改善量后，agent 在维持接触时不再获得持续奖励，也就丧失了无限累积的动机；同时，接近目标并尝试增加接触腿数会得到正向信号，仍可引导着陆动作。预计 agent 会更快满足终止条件（提高 terminated 比例，缩短 episode length），proxy 与任务得分重新对齐，整体 score 有望跨过 200 门槛。

- **expected_next_round**：
  terminated 率上升至 ≥50%，episode length 显著缩短（可能 <400），contact_guidance 的 ep_sum_mean 大幅降低（约一个 episode 内最多增加 2 次接触），但 `progress` 和姿态/速度惩罚的相对贡献会提升，最终 score 达到或超过 200。

### 2. 修改后的奖励函数代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # distance to target (target at origin (0,0))
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = dist_prev - dist_next                # >0 when moving closer to target

    # velocity components
    vx = next_obs[2]
    vy = next_obs[3]

    # vertical velocity penalty, amplified when close to the ground (y near 0)
    altitude_factor = 1.0 / (1.0 + 10.0 * abs(next_obs[1]))
    vel_vertical_penalty = -0.02 * (vy ** 2) * altitude_factor

    # horizontal velocity penalty (small constant weight)
    vel_horizontal_penalty = -0.01 * (vx ** 2)

    velocity_penalty = vel_vertical_penalty + vel_horizontal_penalty

    # orientation penalty: body angle and angular velocity
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.02 * (next_obs[5] ** 2)

    # contact guidance: reward *increase* in leg contact, not the absolute contact state
    prev_contact = float(obs[6] + obs[7])
    next_contact = float(next_obs[6] + next_obs[7])
    contact_improvement = max(0.0, next_contact - prev_contact)   # only positive changes

    proximity_factor = max(0.0, 1.0 - dist_next / 5.0)
    velocity_ok = max(0.0, 1.0 - abs(vy) / 0.5)
    contact_guidance = 0.3 * contact_improvement * proximity_factor * velocity_ok

    # combine all components
    total_reward = (0.5 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 0.5 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }

    reward = float(total_reward)
    return reward, components
```