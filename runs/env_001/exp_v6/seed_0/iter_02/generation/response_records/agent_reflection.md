# Response Record

## 诊断

**症状分析：**
- Episode 长度 74（<150），100% 提前终止，得分 -112 → **agent 在坠毁**。
- `stability_penalty` ratio_to_progress = **-0.88** → 惩罚几乎是进度奖励的 90%。agent 每走一步，辛苦赚来的 progress_reward (0.016) 被 stability_penalty (-0.014) 几乎完全抵消，净奖励仅 0.003。等同于"动也被罚、不动也被罚"——agent 学不到有效策略。
- `soft_landing_proxy` nonzero_rate = **0.2%** → 几乎不触发。二值条件太苛刻，agent 活不到靠近目标就已经坠毁，这个信号形同虚设。
- `original_env_reward` = -1.59/步 → 环境在持续给强负信号（坠毁惩罚），但我们的 reward 太弱无法对抗。

**根因：** stability_penalty 是主犯（ratio 0.88，把 progress 信号淹没），soft_landing_proxy 是从犯（0.2% 触发率，无引导作用）。

**本轮行动：** 按原则 3，先修最致命的问题——大幅削弱 stability_penalty（系数降低 10 倍），让 progress_reward 成为真正的主导信号。soft_landing_proxy 留到下一轮，因为 agent 得先学会不坠毁，才谈得上软着陆。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置 (0,0)，因为 obs 中 x,y 是相对于目标平台的坐标
    target_pos = (0.0, 0.0)

    # 1. 主学习信号：基于距离变化的进度奖励
    dist = ((obs[0] - target_pos[0]) ** 2 + (obs[1] - target_pos[1]) ** 2) ** 0.5
    next_dist = ((next_obs[0] - target_pos[0]) ** 2 + (next_obs[1] - target_pos[1]) ** 2) ** 0.5
    progress_reward = dist - next_dist  # 每一步靠近目标获得正奖励

    # 2. 稳定性惩罚（系数降低 10 倍：0.01 → 0.001）
    #    原因：上轮 ratio_to_progress = -0.88，惩罚几乎抵消全部进度信号，
    #    导致 agent 无有效梯度可学。削弱后预计 ratio ≈ -0.09，progress 将主导。
    vel_x = abs(next_obs[2])
    vel_y = abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    stability_penalty = -0.001 * (vel_x + vel_y) - 0.001 * angle - 0.001 * ang_vel

    # 3. 软着陆 proxy（本轮未改，留待下轮观察）
    near_target = (next_dist < 0.1)
    low_speed = (vel_x + vel_y < 0.2)
    stable_angle = (angle < 0.1)
    both_contact = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    soft_landing_proxy = 0.5 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }
    return float(total_reward), components
```

**预期效果：** stability_penalty ratio 从 -0.88 降至约 -0.09，progress_reward 成为净正信号主导，agent 获得清晰梯度去靠近目标而非"不敢动"。episode 长度应显著提升。soft_landing_proxy 仍然稀疏，下轮再处理。
