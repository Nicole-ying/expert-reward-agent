# Response Record

**1. 诊断报告**

- **evidence**  
  - iter_5 分数 -121.6，episode 长度 68.4，**全部 20 集 early terminal**（步数<150 且分数<-50），说明策略从未学会安全着陆，每次都因坠毁或飞出而提前终止。  
  - **landing_reward** 的 `active_rate` 仅 **0.8%**，`ep_sum_mean` 1.78，意味着 99.2% 的步数该分量为 0；只有极少数时刻两条腿同时触地才收到奖励。这种极度稀疏的奖励无法为着陆过程提供任何连续的姿态/速度反馈，agent 基本不知道“怎样算好的着陆”。  
  - `progress_reward` 全程 100% 激活，均值 0.41，但分数仍大幅为负，说明接近目标的进度奖励不足以弥补坠毁造成的低分（agent 可能在接近时因高速/大角度撞地）。  
  - `action_penalty` 仅 -0.0625，active_rate 9.1%，大部分步没有动作，可能是 hover 掉地，无动作导致坠毁。  
  - checkpoint 趋势：landing_reward 全程极低激活；progress_reward 稳定但无提升；整体分数始终 -105 到 -122，无任何改善趋势。  
  - 历史记录中（通过 `read_past_reflection` 和 `read_reward_code` 检索）**未出现**“将 landing 奖励改为连续的高度‑速度‑姿态质量函数”的尝试，因此本轮修改不是已有语义的重复。

- **behavior_diagnosis**  
  当前学到的策略是：随机移动（或不移动）并快速坠毁。progress_reward 的存在让 agent 略微倾向于向中心移动，但 landing 缺失中间反馈，无法引导它在接近地面时减速、调姿。极短的 episode 和持续负分表明 agent 完全没有感知到“安全着陆”的路径，整个学习陷入失败状态。

- **signal_completeness**  
  必要信号当前缺失一个**连续的着陆过程反馈**。现有组件：progress（接近目标）✅、着陆质量（只给终点接触）❌ 太稀疏、动作稀疏 ✅。需要将 landing 奖励从二元接触开关变为**随高度降低而增强的姿态‑速度软奖励**，同时保留接触时的额外奖金，从而形成“接近地面 ‑ 减速 ‑ 摆正 ‑ 双腿触地”的完整奖励链条。

- **selected_level**  
  **L2** — 改变 landing_reward 的数学形态，从离散双腿接触奖励变为连续的高度/速度/姿态质量函数。历史中未做过此变换，且当前分数离目标（200）差距远超 10%，L1 微调不足以解决稀疏性问题。

- **selected_intervention**  
  将 `landing_reward` 改写为：
  - **高度因子** `height_factor = max(0, 1 – y / y_threshold)`，当飞行器高度 `y`（`next_obs[1]`）进入 5 单位以内时激活。
  - **质量因子** 用指数衰减结合总速率和身体角度：`quality = height_factor × exp(-speed_total / speed_scale) × exp(-abs(angle) / angle_scale)`。  
  - **接触奖金** 当两条腿均触地（`next_obs[6]>0.5 and next_obs[7]>0.5`）时额外加上固定奖金（如 10.0），保证最终成功着陆地获得强正信号。  
  其余组件（progress_reward + health_gate、action_penalty）保持不变，仅调整 `w_land` 为 2.0。

- **falsifiable_hypothesis**  
  连续着陆奖励使 agent 在**整个下降过程**都能获得关于速度和姿态的反馈，预期：
  - `landing_reward` 的 `active_rate` 显著上升（不再仅 0.8%），`ep_sum_mean` 增加，
  - episode 长度延长（因为 agent 学会减速而非直接撞击），
  - 至少部分 episode 能完成双腿触地，从而获得接触奖金，整体分数明显高于 -121。

- **expected_next_round**  
  策略将开始学习在低空区域减速并保持直立，episode 长度可超过 150 步，出现部分成功着陆的 episode；分数向正值方向移动，landing_reward 的活跃度和贡献大幅提升。

**2. 完整 Python 代码**

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------------------
    # Hyperparameters
    # -------------------------------
    w_progress = 1.0          # weight for distance-reduction reward
    c_angle = 2.0             # health gate angle sensitivity
    c_vel = 1.0               # health gate speed sensitivity
    w_land = 2.0              # weight for continuous landing quality reward
    y_threshold = 5.0         # height below which landing quality activates
    speed_scale = 1.0         # speed penalty temperature
    angle_scale = 0.5         # angle penalty temperature
    contact_bonus = 10.0      # bonus for stable two-leg contact
    w_action = 0.01           # weight for action sparsity penalty

    # -------------------------------
    # 1. Goal-reaching progress (distance reduction with health gate)
    # -------------------------------
    dist_cur = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_raw = dist_cur - dist_next  # positive when getting closer

    health_penalty = c_angle * abs(obs[4]) + c_vel * (abs(obs[2]) + abs(obs[3]))
    health_gate = 1.0 / (1.0 + health_penalty)

    progress_reward = w_progress * progress_raw * health_gate

    # -------------------------------
    # 2. Continuous landing quality (height, speed, angle)
    # -------------------------------
    y = next_obs[1]                          # altitude (assuming ground at 0)
    height_factor = max(0.0, 1.0 - y / y_threshold)

    speed_total = abs(next_obs[2]) + abs(next_obs[3])
    angle_abs = abs(next_obs[4])

    # quality: high when low, slow, and upright
    quality = (
        height_factor *
        (2.718281828 ** (-speed_total / speed_scale)) *
        (2.718281828 ** (-angle_abs / angle_scale))
    )
    landing_reward = w_land * quality

    # bonus for actual stable two-leg contact
    both_legs_contact = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    if both_legs_contact:
        landing_reward += contact_bonus

    # -------------------------------
    # 3. Action sparsity penalty
    # -------------------------------
    action_penalty = -w_action * (1.0 if action != 0 else 0.0)

    # -------------------------------
    # Total reward
    # -------------------------------
    total_reward = progress_reward + landing_reward + action_penalty
    components = {
        "progress_reward": progress_reward,
        "landing_reward": landing_reward,
        "action_penalty": action_penalty
    }
    return float(total_reward), components
```
